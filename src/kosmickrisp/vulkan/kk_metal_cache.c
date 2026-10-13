#include "kk_metal_cache.h"
#include "kosmickrisp/bridge/mtl_bridge.h"
#include "util/u_debug.h"

#include "util/list.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_thread.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static _Thread_local struct kk_metal_trace kk_metal_trace;

struct kk_metal_trace
kk_metal_cache_set_trace(struct kk_metal_trace trace)
{
   struct kk_metal_trace previous = kk_metal_trace;
   kk_metal_trace = trace;
   return previous;
}

uint32_t
kk_metal_cache_parse_size(const char *value)
{
   char *end = NULL;
   uint64_t size = value ? strtoull(value, &end, 10) : 0;
   if (!size || end == value)
      return 512u << 20;
   unsigned shift = 30;
   if (*end == 'K' || *end == 'k')
      shift = 10;
   else if (*end == 'M' || *end == 'm')
      shift = 20;
   return size > (UINT32_MAX >> shift) ? UINT32_MAX : size << shift;
}

static void
kk_metal_format_key(const uint8_t *key, char value[CACHE_KEY_SIZE * 2 + 1])
{
   static const char digits[] = "0123456789abcdef";
   for (unsigned i = 0; i < CACHE_KEY_SIZE; ++i) {
      value[2 * i] = digits[key[i] >> 4];
      value[2 * i + 1] = digits[key[i] & 15];
   }
   value[CACHE_KEY_SIZE * 2] = 0;
}

static struct mtl_compile_phase
kk_metal_phase_begin(struct kk_metal_cache *cache, const char *operation,
                      const uint8_t *key)
{
   if (!mtl_compile_phase_enabled())
      return (struct mtl_compile_phase){0};
   char value[CACHE_KEY_SIZE * 2 + 1] = {0};
   if (key)
      kk_metal_format_key(key, value);
   return mtl_compile_phase_begin(operation, cache, value, "");
}

static int64_t
kk_metal_trace_time(void)
{
   if (!kk_metal_trace.callback)
      return 0;
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return (int64_t)now.tv_sec * 1000000000 + now.tv_nsec;
}

static void
kk_metal_trace_event(const char *phase, const uint8_t key[CACHE_KEY_SIZE],
                      int64_t start)
{
   if (kk_metal_trace.callback)
      kk_metal_trace.callback(kk_metal_trace.data, phase, key, CACHE_KEY_SIZE,
                             start, kk_metal_trace_time());
}

#define KK_METAL_ARCHIVE_CAPACITY 256
#define KK_METAL_POLL_NS 250000000ll
#define KK_METAL_QUIET_MS_DEFAULT 1500
#define KK_METAL_MAX_AGE_MS_DEFAULT 60000
#define KK_METAL_MAX_PENDING_DEFAULT 32
#define KK_METAL_RECAPTURE_LIMIT 256

struct kk_metal_cache_key {
   cache_key key;
   struct kk_metal_cache_key *next;
};

struct kk_metal_recapture {
   cache_key key;
   void *descriptor;
   uint64_t max_threads;
   bool compute;
   struct kk_metal_recapture *next;
};

struct kk_metal_loaded_archive {
   cache_key key;
   mtl_archive *archive;
   struct kk_metal_loaded_archive *hash_next;
   struct kk_metal_loaded_archive *prev;
   struct kk_metal_loaded_archive *next;
};

struct kk_metal_cache {
   struct disk_cache *disk;
   mtl_device *device;
   mtl_compiler *compiler;
   mtl_compiler *uncaptured;
   mtx_t mutex;
   unsigned captured_compiles;
   unsigned uncaptured_compiles;
   unsigned bypassed_compiles;
   unsigned recaptured_compiles;
   unsigned dropped_recaptures;
   unsigned recapture_count;
   struct kk_metal_recapture *recapture;
   unsigned active_operations;
   unsigned exit_pins;
   bool closing;
   bool worker_finished;
   int64_t retry_after_ns;
   unsigned retry_seconds;
   bool serializing;
   int64_t last_compile_ns;
   int64_t oldest_pending_ns;
   int64_t quiet_ns;
   int64_t max_age_ns;
   unsigned pending_count;
   unsigned max_pending;
   uint64_t blob_budget;
   struct list_head link;
   cnd_t cond;
   thrd_t worker;
   uint64_t flush_requested;
   uint64_t flush_completed;
   bool stopping;
   char identity[256];
   struct kk_metal_cache_key *pending;
   struct kk_metal_loaded_archive *archives[KK_METAL_ARCHIVE_CAPACITY];
   struct kk_metal_loaded_archive *archive_head;
   struct kk_metal_loaded_archive *archive_tail;
   unsigned archive_count;
};

static void
kk_metal_reference_key(struct kk_metal_cache *cache,
                       const uint8_t key[CACHE_KEY_SIZE], cache_key reference)
{
   uint8_t data[sizeof(cache->identity) + CACHE_KEY_SIZE] = {0};
   memcpy(data, cache->identity, sizeof(cache->identity));
   memcpy(data + sizeof(cache->identity), key, CACHE_KEY_SIZE);
   disk_cache_compute_key(cache->disk, data, sizeof(data), reference);
}

static void
kk_metal_log_event(const char *outcome, const char *fields)
{
   const char *path = debug_get_option("MESA_KK_CACHE_LOG", NULL);
   if (path) {
      FILE *file = fopen(path, "a");
      if (file) {
         fprintf(file, "{\"event\":\"persistence\",\"outcome\":\"%s\"%s}\n",
                 outcome, fields);
         fclose(file);
      }
   }
   if (debug_get_bool_option("MESA_KK_PSO_CACHE_DEBUG", false))
      fprintf(stderr, "KosmicKrisp: %s%s\n", outcome, fields);
}

static void
kk_metal_log_failure(const char *outcome)
{
   kk_metal_log_event(outcome, "");
}

static void
kk_metal_log_miss(const char *outcome, const uint8_t key[CACHE_KEY_SIZE],
                  int64_t start)
{
   kk_metal_trace_event(outcome, key, start);
   char value[CACHE_KEY_SIZE * 2 + 1];
   kk_metal_format_key(key, value);
   char fields[sizeof(value) + 16];
   snprintf(fields, sizeof(fields), ",\"key\":\"%s\"", value);
   kk_metal_log_event(outcome, fields);
}

static void
kk_metal_log_flush(const char *outcome, size_t size, unsigned keys,
                   unsigned limit, unsigned bypassed, unsigned recaptured,
                   unsigned dropped)
{
   char fields[160];
   snprintf(fields, sizeof(fields),
            ",\"size\":%zu,\"keys\":%u,\"limit\":%u,\"bypassed\":%u,"
            "\"recaptured\":%u,\"dropped\":%u",
            size, keys, limit, bypassed, recaptured, dropped);
   kk_metal_log_event(outcome, fields);
}

static void
kk_metal_flush_locked(struct kk_metal_cache *cache)
{
   if (!cache->pending)
      return;
   cache->serializing = true;
   cache->bypassed_compiles = 0;
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "serialization_drain", NULL);
   while (cache->captured_compiles)
      cnd_wait(&cache->cond, &cache->mutex);
   mtl_compile_phase_end(phase, true);
   struct kk_metal_cache_key *pending = cache->pending;
   unsigned count = cache->pending_count;
   unsigned limit = cache->max_pending;
   unsigned recaptured = cache->recaptured_compiles;
   unsigned dropped = cache->dropped_recaptures;
   cache->pending = NULL;
   cache->pending_count = 0;
   cache->recaptured_compiles = 0;
   cache->dropped_recaptures = 0;
   int64_t oldest = cache->oldest_pending_ns;
   mtx_unlock(&cache->mutex);

   size_t size = 0;
   void *data = pending ? mtl_compiler_serialize(cache->compiler, &size) : NULL;
   mtx_lock(&cache->mutex);
   cache->serializing = false;
   unsigned bypassed = cache->bypassed_compiles;
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
   if (!data || !size) {
      free(data);
      kk_metal_log_failure("serialization_failed");
      mtx_lock(&cache->mutex);
      struct kk_metal_cache_key **tail = &pending;
      while (*tail)
         tail = &(*tail)->next;
      *tail = cache->pending;
      cache->pending = pending;
      cache->pending_count += count;
      cache->recaptured_compiles += recaptured;
      cache->dropped_recaptures += dropped;
      cache->oldest_pending_ns = oldest;
      cache->retry_seconds = cache->retry_seconds ?
         MIN2(cache->retry_seconds * 2, 60) : 1;
      cache->retry_after_ns = os_time_get_nano() +
                             cache->retry_seconds * 1000000000ll;
      return;
   }
   bool oversized = cache->blob_budget && size > cache->blob_budget;
   kk_metal_log_flush(oversized ? "archive_oversized" : "archive_flushed", size,
                      count, limit, bypassed, recaptured, dropped);
   cache_key blob;
   if (!oversized) {
      disk_cache_compute_key(cache->disk, data, size, blob);
      disk_cache_put(cache->disk, blob, data, size, NULL);
   }
   while (pending) {
      struct kk_metal_cache_key *entry = pending;
      if (!oversized) {
         cache_key reference;
         kk_metal_reference_key(cache, entry->key, reference);
         disk_cache_put(cache->disk, reference, blob, sizeof(blob), NULL);
      }
      pending = entry->next;
      free(entry);
   }
   free(data);
   mtx_lock(&cache->mutex);
   if (oversized)
      cache->max_pending = MAX2(cache->max_pending / 2, 1);
   cache->retry_seconds = 0;
   cache->retry_after_ns = 0;
}

static bool
kk_metal_flush_due_locked(struct kk_metal_cache *cache, int64_t now)
{
   if (cache->flush_requested != cache->flush_completed)
      return true;
   if (!cache->pending || !cache->uncaptured || now < cache->retry_after_ns)
      return false;
   if (cache->pending_count >= cache->max_pending)
      return true;
   if (!cache->captured_compiles && !cache->uncaptured_compiles &&
       now - cache->last_compile_ns >= cache->quiet_ns)
      return true;
   return now - cache->oldest_pending_ns >= cache->max_age_ns;
}

static void kk_metal_recapture_locked(struct kk_metal_cache *cache, bool drain);

static int
kk_metal_worker(void *arg)
{
   struct kk_metal_cache *cache = arg;
   mtx_lock(&cache->mutex);
   while (!cache->stopping) {
      if (cache->flush_requested == cache->flush_completed &&
          !cache->recapture) {
         struct timespec deadline;
         clock_gettime(CLOCK_REALTIME, &deadline);
         deadline.tv_nsec += KK_METAL_POLL_NS;
         if (deadline.tv_nsec >= 1000000000) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000;
         }
         cnd_timedwait(&cache->cond, &cache->mutex, &deadline);
      }
      if (cache->stopping)
         break;
      if (kk_metal_flush_due_locked(cache, os_time_get_nano())) {
         uint64_t requested = cache->flush_requested;
         kk_metal_flush_locked(cache);
         cache->flush_completed = requested;
         cnd_broadcast(&cache->cond);
      }
      kk_metal_recapture_locked(cache, false);
   }
   while (cache->active_operations)
      cnd_wait(&cache->cond, &cache->mutex);
   kk_metal_recapture_locked(cache, true);
   kk_metal_flush_locked(cache);
   cache->worker_finished = true;
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
   return 0;
}

static simple_mtx_t kk_metal_caches_lock = SIMPLE_MTX_INITIALIZER;
static struct list_head kk_metal_caches = {
   .prev = &kk_metal_caches,
   .next = &kk_metal_caches,
};
static once_flag kk_metal_exit_once = ONCE_FLAG_INIT;

static bool kk_metal_exit_registered;

static void
kk_metal_cache_close(struct kk_metal_cache *cache)
{
   mtx_lock(&cache->mutex);
   cache->closing = true;
   while (cache->active_operations)
      cnd_wait(&cache->cond, &cache->mutex);
   cache->stopping = true;
   cnd_broadcast(&cache->cond);
   while (!cache->worker_finished)
      cnd_wait(&cache->cond, &cache->mutex);
   mtx_unlock(&cache->mutex);
   disk_cache_wait_for_idle(cache->disk);
}

static void
kk_metal_cache_exit(void)
{
   simple_mtx_lock(&kk_metal_caches_lock);
   size_t count = 0;
   list_for_each_entry(struct kk_metal_cache, cache, &kk_metal_caches, link)
      count++;
   struct kk_metal_cache **caches =
      count ? calloc(count, sizeof(*caches)) : NULL;
   if (count && !caches) {
      simple_mtx_unlock(&kk_metal_caches_lock);
      kk_metal_log_failure("exit_snapshot_failed");
      return;
   }
   size_t i = 0;
   list_for_each_entry(struct kk_metal_cache, cache, &kk_metal_caches, link) {
      mtx_lock(&cache->mutex);
      cache->exit_pins++;
      mtx_unlock(&cache->mutex);
      caches[i++] = cache;
   }
   simple_mtx_unlock(&kk_metal_caches_lock);
   for (i = 0; i < count; i++) {
      struct kk_metal_cache *cache = caches[i];
      kk_metal_cache_close(cache);
      mtx_lock(&cache->mutex);
      cache->exit_pins--;
      cnd_broadcast(&cache->cond);
      mtx_unlock(&cache->mutex);
   }
   free(caches);
}

static void
kk_metal_register_exit(void)
{
   kk_metal_exit_registered = atexit(kk_metal_cache_exit) == 0;
   if (!kk_metal_exit_registered)
      kk_metal_log_failure("exit_registration_failed");
}

static unsigned
kk_metal_count_option(const char *name, unsigned fallback)
{
   const char *value = debug_get_option(name, NULL);
   if (!value)
      return fallback;
   errno = 0;
   char *end;
   long long count = strtoll(value, &end, 10);
   if (errno || end == value || *end || count <= 0 || count > INT_MAX)
      return fallback;
   return count;
}

static int64_t
kk_metal_timing_option(const char *name, unsigned fallback)
{
   return kk_metal_count_option(name, fallback) * 1000000ll;
}

struct kk_metal_cache *
kk_metal_cache_create(struct disk_cache *disk_cache, mtl_device *device,
                      uint32_t gpu_family, mtl_compiler *compiler,
                      uint64_t blob_budget)
{
   if (!disk_cache ||
       debug_get_bool_option("MESA_KK_PSO_CACHE_DISABLE", false) ||
       !mtl_compiler_has_serializer(compiler))
      return NULL;
   struct kk_metal_cache *cache = calloc(1, sizeof(*cache));
   if (!cache)
      return NULL;
   cache->disk = disk_cache;
   cache->device = device;
   cache->compiler = compiler;
   if (!debug_get_bool_option("MESA_KK_PSO_CACHE_BYPASS_DISABLE", false))
      cache->uncaptured = mtl_new_uncaptured_compiler(device);
   if (!cache->uncaptured)
      kk_metal_log_failure("automatic_flush_deferred");
   cache->quiet_ns = kk_metal_timing_option("MESA_KK_PSO_CACHE_QUIET_MS",
                                            KK_METAL_QUIET_MS_DEFAULT);
   cache->max_age_ns = kk_metal_timing_option("MESA_KK_PSO_CACHE_MAX_AGE_MS",
                                              KK_METAL_MAX_AGE_MS_DEFAULT);
   cache->max_pending = kk_metal_count_option("MESA_KK_PSO_CACHE_MAX_PENDING",
                                              KK_METAL_MAX_PENDING_DEFAULT);
   cache->blob_budget = blob_budget;
   cache->last_compile_ns = os_time_get_nano();
   mtl_pipeline_cache_identity(device, gpu_family, cache->identity,
                               sizeof(cache->identity));
   if (mtx_init(&cache->mutex, mtx_plain) != thrd_success)
      goto fail;
   if (cnd_init(&cache->cond) != thrd_success)
      goto fail_mutex;
   if (thrd_create(&cache->worker, kk_metal_worker, cache) != thrd_success)
      goto fail_cond;
   if (debug_get_bool_option("MESA_KK_PSO_CACHE_EXIT_FLUSH", true)) {
      call_once(&kk_metal_exit_once, kk_metal_register_exit);
   }
   list_inithead(&cache->link);
   if (debug_get_bool_option("MESA_KK_PSO_CACHE_EXIT_FLUSH", true) &&
       kk_metal_exit_registered) {
      simple_mtx_lock(&kk_metal_caches_lock);
      list_addtail(&cache->link, &kk_metal_caches);
      simple_mtx_unlock(&kk_metal_caches_lock);
   }
   return cache;

fail_cond:
   cnd_destroy(&cache->cond);
fail_mutex:
   mtx_destroy(&cache->mutex);
fail:
   mtl_release(cache->uncaptured);
   free(cache);
   return NULL;
}

void
kk_metal_cache_flush(struct kk_metal_cache *cache)
{
   if (!cache)
      return;
   mtx_lock(&cache->mutex);
   if (cache->closing) {
      mtx_unlock(&cache->mutex);
      return;
   }
   cache->active_operations++;
   uint64_t requested = ++cache->flush_requested;
   cnd_broadcast(&cache->cond);
   while (cache->flush_completed < requested && !cache->stopping)
      cnd_wait(&cache->cond, &cache->mutex);
   cache->active_operations--;
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
}

void
kk_metal_cache_destroy(struct kk_metal_cache *cache)
{
   if (!cache)
      return;
   simple_mtx_lock(&kk_metal_caches_lock);
   list_delinit(&cache->link);
   simple_mtx_unlock(&kk_metal_caches_lock);
   kk_metal_cache_close(cache);
   mtx_lock(&cache->mutex);
   while (cache->exit_pins)
      cnd_wait(&cache->cond, &cache->mutex);
   mtx_unlock(&cache->mutex);
   thrd_join(cache->worker, NULL);
   cnd_destroy(&cache->cond);
   mtx_destroy(&cache->mutex);
   while (cache->pending) {
      struct kk_metal_cache_key *next = cache->pending->next;
      free(cache->pending);
      cache->pending = next;
   }
   while (cache->recapture) {
      struct kk_metal_recapture *next = cache->recapture->next;
      mtl_release(cache->recapture->descriptor);
      free(cache->recapture);
      cache->recapture = next;
   }
   while (cache->archive_head) {
      struct kk_metal_loaded_archive *next = cache->archive_head->next;
      mtl_release(cache->archive_head->archive);
      free(cache->archive_head);
      cache->archive_head = next;
   }
   mtl_release(cache->uncaptured);
   free(cache);
}

static void
kk_metal_archive_unlink(struct kk_metal_cache *cache,
                        struct kk_metal_loaded_archive *entry)
{
   if (entry->prev)
      entry->prev->next = entry->next;
   else
      cache->archive_head = entry->next;
   if (entry->next)
      entry->next->prev = entry->prev;
   else
      cache->archive_tail = entry->prev;
}

static void
kk_metal_archive_promote(struct kk_metal_cache *cache,
                         struct kk_metal_loaded_archive *entry)
{
   entry->prev = NULL;
   entry->next = cache->archive_head;
   if (cache->archive_head)
      cache->archive_head->prev = entry;
   else
      cache->archive_tail = entry;
   cache->archive_head = entry;
}

static mtl_archive *
kk_metal_find_archive_locked(struct kk_metal_cache *cache, const void *ref)
{
   unsigned bucket = ((const uint8_t *)ref)[0];
   for (struct kk_metal_loaded_archive *entry = cache->archives[bucket]; entry;
        entry = entry->hash_next) {
      if (!memcmp(entry->key, ref, CACHE_KEY_SIZE)) {
         kk_metal_archive_unlink(cache, entry);
         kk_metal_archive_promote(cache, entry);
         return mtl_retain(entry->archive);
      }
   }
   return NULL;
}

static mtl_archive *
kk_metal_load(struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE])
{
   int64_t lookup_start = kk_metal_trace_time();
   cache_key reference;
   kk_metal_reference_key(cache, key, reference);
   size_t ref_size = 0;
   void *ref = disk_cache_get(cache->disk, reference, &ref_size);
   if (!ref || ref_size != CACHE_KEY_SIZE) {
      kk_metal_trace_event("archive_lookup", key, lookup_start);
      kk_metal_log_miss("ref_missing", key, lookup_start);
      free(ref);
      return NULL;
   }
   mtx_lock(&cache->mutex);
   mtl_archive *archive = kk_metal_find_archive_locked(cache, ref);
   mtx_unlock(&cache->mutex);
   kk_metal_trace_event("archive_lookup", key, lookup_start);
   if (archive) {
      free(ref);
      return archive;
   }
   int64_t load_start = kk_metal_trace_time();
   size_t size = 0;
   void *data = disk_cache_get(cache->disk, ref, &size);
   cache_key actual;
   const char *miss = "blob_missing";
   if (data && size) {
      disk_cache_compute_key(cache->disk, data, size, actual);
      if (!memcmp(actual, ref, CACHE_KEY_SIZE)) {
         archive = mtl_new_archive(cache->device, data, size);
         miss = "archive_rejected";
      }
   }
   kk_metal_trace_event("archive_load", key, load_start);
   if (!archive)
      kk_metal_log_miss(miss, key, load_start);
   if (archive) {
      mtx_lock(&cache->mutex);
      mtl_archive *existing = kk_metal_find_archive_locked(cache, ref);
      struct kk_metal_loaded_archive *entry =
         existing ? NULL : malloc(sizeof(*entry));
      if (existing) {
         mtl_release(archive);
         archive = existing;
      } else if (entry) {
         memcpy(entry->key, ref, CACHE_KEY_SIZE);
         entry->archive = mtl_retain(archive);
         unsigned bucket = entry->key[0];
         entry->hash_next = cache->archives[bucket];
         cache->archives[bucket] = entry;
         kk_metal_archive_promote(cache, entry);
         if (++cache->archive_count > KK_METAL_ARCHIVE_CAPACITY) {
            struct kk_metal_loaded_archive *last = cache->archive_tail;
            struct kk_metal_loaded_archive **link =
               &cache->archives[last->key[0]];
            while (*link != last)
               link = &(*link)->hash_next;
            *link = last->hash_next;
            kk_metal_archive_unlink(cache, last);
            mtl_release(last->archive);
            free(last);
            cache->archive_count--;
         }
      }
      mtx_unlock(&cache->mutex);
   }
   free(ref);
   free(data);
   return archive;
}

static void
kk_metal_capture_locked(struct kk_metal_cache *cache,
                         const uint8_t key[CACHE_KEY_SIZE])
{
   for (struct kk_metal_cache_key *entry = cache->pending;
        entry; entry = entry->next)
      if (!memcmp(entry->key, key, CACHE_KEY_SIZE))
         return;
   struct kk_metal_cache_key *entry = malloc(sizeof(*entry));
   if (!entry)
      return;
   memcpy(entry->key, key, CACHE_KEY_SIZE);
   if (!cache->pending)
      cache->oldest_pending_ns = os_time_get_nano();
   entry->next = cache->pending;
   cache->pending = entry;
   cache->pending_count++;
}

static void
kk_metal_recapture_locked(struct kk_metal_cache *cache, bool drain)
{
   while (cache->recapture &&
          (drain || (!cache->stopping &&
                     cache->flush_requested == cache->flush_completed))) {
      struct kk_metal_recapture *entry = cache->recapture;
      cache->recapture = entry->next;
      cache->recapture_count--;
      cache->captured_compiles++;
      cache->last_compile_ns = os_time_get_nano();
      mtx_unlock(&cache->mutex);
      struct mtl_compile_phase phase =
         kk_metal_phase_begin(cache, "recapture", entry->key);
      void *pipeline = entry->compute ?
         (void *)mtl_new_compute_pipeline_state(cache->compiler,
                                                entry->descriptor,
                                                entry->max_threads) :
         (void *)mtl_new_render_pipeline(cache->compiler, entry->descriptor);
      mtl_compile_phase_end(phase, pipeline != NULL);
      mtl_release(pipeline);
      mtl_release(entry->descriptor);
      mtx_lock(&cache->mutex);
      if (pipeline) {
         kk_metal_capture_locked(cache, entry->key);
         cache->recaptured_compiles++;
      }
      cache->captured_compiles--;
      cnd_broadcast(&cache->cond);
      free(entry);
   }
}

static void
kk_metal_defer_recapture_locked(struct kk_metal_cache *cache,
                                const uint8_t key[CACHE_KEY_SIZE],
                                void *descriptor, bool compute,
                                uint64_t max_threads)
{
   struct kk_metal_recapture *entry =
      cache->recapture_count >= KK_METAL_RECAPTURE_LIMIT ?
         NULL : malloc(sizeof(*entry));
   if (!entry) {
      cache->dropped_recaptures++;
      return;
   }
   memcpy(entry->key, key, CACHE_KEY_SIZE);
   entry->descriptor = mtl_retain(descriptor);
   entry->max_threads = max_threads;
   entry->compute = compute;
   entry->next = cache->recapture;
   cache->recapture = entry;
   cache->recapture_count++;
}

static mtl_compiler *
kk_metal_begin_compile(struct kk_metal_cache *cache,
                       const uint8_t key[CACHE_KEY_SIZE], bool *captured)
{
   mtx_lock(&cache->mutex);
   cache->last_compile_ns = os_time_get_nano();
   if (cache->serializing && cache->uncaptured) {
      struct mtl_compile_phase phase =
         kk_metal_phase_begin(cache, "serialization_bypass", key);
      mtl_compile_phase_end(phase, true);
      cache->uncaptured_compiles++;
      cache->bypassed_compiles++;
      mtl_compiler *compiler = cache->uncaptured;
      mtx_unlock(&cache->mutex);
      *captured = false;
      return compiler;
   }
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "serialization_wait", key);
   while (cache->serializing)
      cnd_wait(&cache->cond, &cache->mutex);
   mtl_compile_phase_end(phase, true);
   cache->captured_compiles++;
   mtx_unlock(&cache->mutex);
   *captured = true;
   return cache->compiler;
}

static void
kk_metal_end_compile(struct kk_metal_cache *cache,
                     const uint8_t key[CACHE_KEY_SIZE], bool captured,
                     bool success, void *descriptor, bool compute,
                     uint64_t max_threads)
{
   mtx_lock(&cache->mutex);
   if (captured) {
      if (success)
         kk_metal_capture_locked(cache, key);
      cache->captured_compiles--;
   } else {
      if (success)
         kk_metal_defer_recapture_locked(cache, key, descriptor, compute,
                                         max_threads);
      cache->uncaptured_compiles--;
   }
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
}

static bool
kk_metal_operation_begin(struct kk_metal_cache *cache)
{
   mtx_lock(&cache->mutex);
   bool admitted = !cache->closing;
   if (admitted)
      cache->active_operations++;
   mtx_unlock(&cache->mutex);
   return admitted;
}

static void
kk_metal_operation_end(struct kk_metal_cache *cache)
{
   mtx_lock(&cache->mutex);
   cache->active_operations--;
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
}

static mtl_render_pipeline_state *
kk_metal_lookup_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor)
{
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "archive_load", key);
   mtl_archive *archive = kk_metal_load(cache, key);
   mtl_compile_phase_end(phase, archive != NULL);
   mtl_render_pipeline_state *pipeline = NULL;
   if (archive) {
      int64_t start = kk_metal_trace_time();
      pipeline = mtl_archive_new_render_pipeline(archive, descriptor);
      kk_metal_trace_event("native_create", key, start);
      if (!pipeline)
         kk_metal_log_miss("pipeline_absent", key, start);
   }
   mtl_release(archive);
   return pipeline;
}

mtl_render_pipeline_state *
kk_metal_cache_lookup_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor)
{
   if (!kk_metal_operation_begin(cache))
      return NULL;
   mtl_render_pipeline_state *pipeline =
      kk_metal_lookup_render_pipeline(cache, key, descriptor);
   kk_metal_operation_end(cache);
   return pipeline;
}

mtl_render_pipeline_state *
kk_metal_cache_new_render_pipeline(struct kk_metal_cache *cache,
                                   const uint8_t key[CACHE_KEY_SIZE],
                                   mtl_render_pipeline_descriptor *descriptor,
                                   bool *archive_hit)
{
   *archive_hit = false;
   if (!kk_metal_operation_begin(cache))
      return NULL;
   mtl_render_pipeline_state *pipeline =
      kk_metal_lookup_render_pipeline(cache, key, descriptor);
   *archive_hit = pipeline != NULL;
   if (!pipeline) {
      bool captured;
      mtl_compiler *compiler = kk_metal_begin_compile(cache, key, &captured);
      int64_t start = kk_metal_trace_time();
      pipeline = mtl_new_render_pipeline(compiler, descriptor);
      kk_metal_trace_event("native_create", key, start);
      kk_metal_end_compile(cache, key, captured, pipeline != NULL, descriptor,
                           false, 0);
   }
   kk_metal_operation_end(cache);
   return pipeline;
}

mtl_compute_pipeline_state *
kk_metal_cache_new_compute_pipeline(struct kk_metal_cache *cache,
                                    const uint8_t key[CACHE_KEY_SIZE],
                                    mtl_function_descriptor *function,
                                    uint64_t max_threads, bool *archive_hit)
{
   *archive_hit = false;
   if (!kk_metal_operation_begin(cache))
      return NULL;
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "archive_load", key);
   mtl_archive *archive = kk_metal_load(cache, key);
   mtl_compile_phase_end(phase, archive != NULL);
   mtl_compute_pipeline_state *pipeline = NULL;
   if (archive) {
      int64_t start = kk_metal_trace_time();
      pipeline =
         mtl_archive_new_compute_pipeline(archive, function, max_threads);
      kk_metal_trace_event("native_create", key, start);
      if (!pipeline)
         kk_metal_log_miss("pipeline_absent", key, start);
   }
   mtl_release(archive);
   *archive_hit = pipeline != NULL;
   if (!pipeline) {
      bool captured;
      mtl_compiler *compiler = kk_metal_begin_compile(cache, key, &captured);
      int64_t start = kk_metal_trace_time();
      pipeline =
         mtl_new_compute_pipeline_state(compiler, function, max_threads);
      kk_metal_trace_event("native_create", key, start);
      kk_metal_end_compile(cache, key, captured, pipeline != NULL, function,
                           true, max_threads);
   }
   kk_metal_operation_end(cache);
   return pipeline;
}
