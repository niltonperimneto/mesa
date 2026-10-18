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
#define KK_METAL_RECAPTURE_BATCH 8
#define KK_METAL_RECAPTURE_ATTEMPTS 2
#define KK_METAL_FINAL_FLUSH_ATTEMPTS 3
#define KK_METAL_REF_CAPACITY 4096

struct kk_metal_ref {
   cache_key key;
   cache_key ref;
   bool valid;
   uint8_t stale;
};

struct kk_metal_loading {
   cache_key ref;
   uint8_t miss;
   bool done;
   unsigned users;
   struct kk_metal_loading *next;
};

struct kk_metal_cache_key {
   cache_key key;
   struct kk_metal_cache_key *next;
};

struct kk_metal_recapture {
   cache_key key;
   void *descriptor;
   uint64_t max_threads;
   bool compute;
   unsigned attempts;
   struct kk_metal_recapture *next;
};

struct kk_metal_loaded_archive {
   cache_key key;
   mtl_archive *archive;
   size_t size;
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
   unsigned dropped_capacity;
   unsigned dropped_alloc;
   unsigned dropped_failed;
   unsigned recapture_count;
   struct kk_metal_recapture *recapture;
   struct kk_metal_recapture *recapture_tail;
   struct kk_metal_recapture *recapturing;
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
   unsigned archive_peak;
   uint64_t archive_bytes;
   uint64_t archive_peak_bytes;
   uint64_t archive_loads;
   uint64_t archive_evictions;
   uint64_t ref_hits;
   uint64_t ref_misses;
   uint64_t ref_stale_hits;
   uint64_t ref_collisions;
   struct kk_metal_loading *loading;
   struct kk_metal_ref refs[KK_METAL_REF_CAPACITY];
};

static struct kk_metal_ref *
kk_metal_ref_slot(struct kk_metal_cache *cache,
                  const uint8_t key[CACHE_KEY_SIZE])
{
   uint32_t index;
   memcpy(&index, key, sizeof(index));
   return &cache->refs[index % KK_METAL_REF_CAPACITY];
}

static bool
kk_metal_ref_get_locked(struct kk_metal_cache *cache,
                        const uint8_t key[CACHE_KEY_SIZE], cache_key ref,
                        uint8_t *stale)
{
   struct kk_metal_ref *slot = kk_metal_ref_slot(cache, key);
   if (!slot->valid || memcmp(slot->key, key, CACHE_KEY_SIZE))
      return false;
   memcpy(ref, slot->ref, CACHE_KEY_SIZE);
   *stale = slot->stale;
   return true;
}

static void
kk_metal_ref_set_locked(struct kk_metal_cache *cache,
                        const uint8_t key[CACHE_KEY_SIZE],
                        const uint8_t ref[CACHE_KEY_SIZE])
{
   struct kk_metal_ref *slot = kk_metal_ref_slot(cache, key);
   cache->ref_collisions +=
      slot->valid && memcmp(slot->key, key, CACHE_KEY_SIZE) != 0;
   memcpy(slot->key, key, CACHE_KEY_SIZE);
   memcpy(slot->ref, ref, CACHE_KEY_SIZE);
   slot->valid = true;
   slot->stale = 0;
}

static void
kk_metal_ref_mark_stale_locked(struct kk_metal_cache *cache,
                               const uint8_t key[CACHE_KEY_SIZE],
                               const uint8_t ref[CACHE_KEY_SIZE], uint8_t miss)
{
   struct kk_metal_ref *slot = kk_metal_ref_slot(cache, key);
   if (slot->valid && !memcmp(slot->key, key, CACHE_KEY_SIZE) &&
       !memcmp(slot->ref, ref, CACHE_KEY_SIZE))
      slot->stale = miss;
}

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
kk_metal_uncacheable_ref(cache_key ref)
{
   memset(ref, 0xff, CACHE_KEY_SIZE);
}

static bool
kk_metal_ref_is_uncacheable(const uint8_t ref[CACHE_KEY_SIZE])
{
   for (unsigned i = 0; i < CACHE_KEY_SIZE; ++i)
      if (ref[i] != 0xff)
         return false;
   return true;
}

static void
kk_metal_limit_key(struct kk_metal_cache *cache, cache_key key)
{
   struct {
      char identity[sizeof(cache->identity)];
      char tag[16];
      uint64_t budget;
   } data;
   memset(&data, 0, sizeof(data));
   memcpy(data.identity, cache->identity, sizeof(data.identity));
   memcpy(data.tag, "batch_limit", sizeof("batch_limit"));
   data.budget = cache->blob_budget;
   disk_cache_compute_key(cache->disk, &data, sizeof(data), key);
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

enum kk_metal_miss {
   KK_METAL_MISS_NONE,
   KK_METAL_MISS_REF,
   KK_METAL_MISS_UNCACHEABLE,
   KK_METAL_MISS_BLOB,
   KK_METAL_MISS_REJECTED,
   KK_METAL_MISS_PIPELINE,
   KK_METAL_MISS_CORRUPT,
   KK_METAL_MISS_COUNT,
};

static const char *const kk_metal_miss_names[KK_METAL_MISS_COUNT] = {
   [KK_METAL_MISS_NONE] = "none",
   [KK_METAL_MISS_REF] = "ref_missing",
   [KK_METAL_MISS_UNCACHEABLE] = "ref_uncacheable",
   [KK_METAL_MISS_BLOB] = "blob_missing",
   [KK_METAL_MISS_REJECTED] = "archive_rejected",
   [KK_METAL_MISS_PIPELINE] = "pipeline_absent",
   [KK_METAL_MISS_CORRUPT] = "blob_corrupt",
};

enum kk_metal_route {
   KK_METAL_ROUTE_CAPTURED,
   KK_METAL_ROUTE_SERIALIZING,
   KK_METAL_ROUTE_BATCH_FULL,
   KK_METAL_ROUTE_UNCACHEABLE,
   KK_METAL_ROUTE_CLOSING,
   KK_METAL_ROUTE_COUNT,
};

static const char *const kk_metal_route_names[KK_METAL_ROUTE_COUNT] = {
   [KK_METAL_ROUTE_CAPTURED] = "captured",
   [KK_METAL_ROUTE_SERIALIZING] = "uncaptured_serializing",
   [KK_METAL_ROUTE_BATCH_FULL] = "uncaptured_batch_full",
   [KK_METAL_ROUTE_UNCACHEABLE] = "uncaptured_uncacheable",
   [KK_METAL_ROUTE_CLOSING] = "closing",
};

static bool
kk_metal_log_enabled(void)
{
   return debug_get_option("MESA_KK_CACHE_LOG", NULL) ||
          debug_get_bool_option("MESA_KK_PSO_CACHE_DEBUG", false);
}

static void
kk_metal_log_compile(const char *kind, const uint8_t key[CACHE_KEY_SIZE],
                     enum kk_metal_miss miss, enum kk_metal_route route,
                     int64_t start, bool success)
{
   if (!kk_metal_log_enabled())
      return;
   char value[CACHE_KEY_SIZE * 2 + 1];
   kk_metal_format_key(key, value);
   char fields[sizeof(value) + 160];
   snprintf(fields, sizeof(fields),
            ",\"key\":\"%s\",\"kind\":\"%s\",\"reason\":\"%s\","
            "\"route\":\"%s\",\"duration_ns\":%lld,\"success\":%s",
            value, kind, kk_metal_miss_names[miss], kk_metal_route_names[route],
            (long long)(os_time_get_nano() - start),
            success ? "true" : "false");
   kk_metal_log_event("native_compile", fields);
}

static void
kk_metal_log_drop(const char *reason, const uint8_t key[CACHE_KEY_SIZE])
{
   if (!kk_metal_log_enabled())
      return;
   char value[CACHE_KEY_SIZE * 2 + 1];
   kk_metal_format_key(key, value);
   char fields[sizeof(value) + 64];
   snprintf(fields, sizeof(fields), ",\"key\":\"%s\",\"reason\":\"%s\"", value,
            reason);
   kk_metal_log_event("persist_dropped", fields);
}

struct kk_metal_drops {
   unsigned capacity;
   unsigned alloc;
   unsigned failed;
};

static struct kk_metal_drops
kk_metal_take_drops_locked(struct kk_metal_cache *cache)
{
   struct kk_metal_drops drops = {
      .capacity = cache->dropped_capacity,
      .alloc = cache->dropped_alloc,
      .failed = cache->dropped_failed,
   };
   cache->dropped_capacity = 0;
   cache->dropped_alloc = 0;
   cache->dropped_failed = 0;
   return drops;
}

static void
kk_metal_restore_drops_locked(struct kk_metal_cache *cache,
                              struct kk_metal_drops drops)
{
   cache->dropped_capacity += drops.capacity;
   cache->dropped_alloc += drops.alloc;
   cache->dropped_failed += drops.failed;
}

static void
kk_metal_log_flush(const char *outcome, size_t size, unsigned keys,
                   unsigned limit, unsigned bypassed, unsigned recaptured,
                   struct kk_metal_drops drops)
{
   char fields[256];
   snprintf(fields, sizeof(fields),
            ",\"size\":%zu,\"keys\":%u,\"limit\":%u,\"bypassed\":%u,"
            "\"recaptured\":%u,\"dropped\":%u,\"dropped_capacity\":%u,"
            "\"dropped_alloc\":%u,\"dropped_failed\":%u",
            size, keys, limit, bypassed, recaptured,
            drops.capacity + drops.alloc + drops.failed, drops.capacity,
            drops.alloc, drops.failed);
   kk_metal_log_event(outcome, fields);
}

static void
kk_metal_log_limit(const char *outcome, unsigned limit)
{
   char fields[32];
   snprintf(fields, sizeof(fields), ",\"limit\":%u", limit);
   kk_metal_log_event(outcome, fields);
}

static void
kk_metal_load_limit(struct kk_metal_cache *cache)
{
   if (!cache->blob_budget)
      return;
   cache_key key;
   kk_metal_limit_key(cache, key);
   size_t size = 0;
   uint32_t *stored = disk_cache_get(cache->disk, key, &size);
   if (stored && size == sizeof(*stored) && *stored &&
       *stored < cache->max_pending) {
      cache->max_pending = *stored;
      kk_metal_log_limit("archive_limit_loaded", *stored);
   }
   free(stored);
}

static void
kk_metal_store_limit(struct kk_metal_cache *cache, uint32_t limit)
{
   cache_key key;
   kk_metal_limit_key(cache, key);
   disk_cache_put(cache->disk, key, &limit, sizeof(limit), NULL);
   kk_metal_log_limit("archive_limit_lowered", limit);
}

static void
kk_metal_mark_uncacheable(struct kk_metal_cache *cache,
                          const uint8_t key[CACHE_KEY_SIZE])
{
   cache_key reference, sentinel;
   kk_metal_reference_key(cache, key, reference);
   kk_metal_uncacheable_ref(sentinel);
   disk_cache_put(cache->disk, reference, sentinel, sizeof(sentinel), NULL);
   mtx_lock(&cache->mutex);
   kk_metal_ref_set_locked(cache, key, sentinel);
   mtx_unlock(&cache->mutex);
   kk_metal_log_miss("archive_uncacheable", key, 0);
}

static bool
kk_metal_flush_locked(struct kk_metal_cache *cache)
{
   if (!cache->pending)
      return true;
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
   struct kk_metal_drops dropped = kk_metal_take_drops_locked(cache);
   cache->pending = NULL;
   cache->pending_count = 0;
   cache->recaptured_compiles = 0;
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
      kk_metal_restore_drops_locked(cache, dropped);
      cache->oldest_pending_ns = oldest;
      cache->retry_seconds = cache->retry_seconds ?
         MIN2(cache->retry_seconds * 2, 60) : 1;
      cache->retry_after_ns = os_time_get_nano() +
                             cache->retry_seconds * 1000000000ll;
      return false;
   }
   bool oversized = cache->blob_budget && size > cache->blob_budget;
   bool uncacheable = oversized && count == 1;
   unsigned lowered = MAX2(MIN2(limit, count) / 2, 1);
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
         mtx_lock(&cache->mutex);
         kk_metal_ref_set_locked(cache, entry->key, blob);
         mtx_unlock(&cache->mutex);
      } else if (uncacheable) {
         kk_metal_mark_uncacheable(cache, entry->key);
      } else {
         kk_metal_log_drop("oversized_batch", entry->key);
      }
      pending = entry->next;
      free(entry);
   }
   free(data);
   if (oversized && !uncacheable)
      kk_metal_store_limit(cache, lowered);
   mtx_lock(&cache->mutex);
   if (oversized && !uncacheable)
      cache->max_pending = MIN2(cache->max_pending, lowered);
   cache->retry_seconds = 0;
   cache->retry_after_ns = 0;
   return true;
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

static bool
kk_metal_batch_full_locked(struct kk_metal_cache *cache)
{
   return cache->pending_count + cache->captured_compiles >= cache->max_pending;
}

static void kk_metal_recapture_locked(struct kk_metal_cache *cache, bool drain);

static void
kk_metal_shutdown_drain_locked(struct kk_metal_cache *cache)
{
   unsigned failures = 0;
   while (cache->pending || cache->recapture) {
      kk_metal_recapture_locked(cache, true);
      if (kk_metal_flush_locked(cache))
         failures = 0;
      else if (++failures >= KK_METAL_FINAL_FLUSH_ATTEMPTS)
         break;
   }
   if (!cache->pending && !cache->recapture)
      return;
   unsigned lost = 0;
   for (struct kk_metal_cache_key *entry = cache->pending; entry;
        entry = entry->next, ++lost)
      kk_metal_log_drop("final_flush_failed", entry->key);
   for (struct kk_metal_recapture *entry = cache->recapture; entry;
        entry = entry->next, ++lost)
      kk_metal_log_drop("final_flush_failed", entry->key);
   char fields[32];
   snprintf(fields, sizeof(fields), ",\"keys\":%u", lost);
   kk_metal_log_event("final_flush_failed", fields);
}

static int
kk_metal_worker(void *arg)
{
   struct kk_metal_cache *cache = arg;
   mtx_lock(&cache->mutex);
   while (!cache->stopping) {
      bool due = kk_metal_flush_due_locked(cache, os_time_get_nano());
      if (!due && (!cache->recapture || kk_metal_batch_full_locked(cache))) {
         struct timespec deadline;
         clock_gettime(CLOCK_REALTIME, &deadline);
         deadline.tv_nsec += KK_METAL_POLL_NS;
         if (deadline.tv_nsec >= 1000000000) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000;
         }
         cnd_timedwait(&cache->cond, &cache->mutex, &deadline);
         if (cache->stopping)
            break;
         due = kk_metal_flush_due_locked(cache, os_time_get_nano());
      }
      if (due) {
         uint64_t requested = cache->flush_requested;
         kk_metal_flush_locked(cache);
         cache->flush_completed = requested;
         cnd_broadcast(&cache->cond);
      }
      kk_metal_recapture_locked(cache, false);
   }
   while (cache->active_operations)
      cnd_wait(&cache->cond, &cache->mutex);
   kk_metal_shutdown_drain_locked(cache);
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
   kk_metal_load_limit(cache);
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

static void
kk_metal_log_summary(struct kk_metal_cache *cache)
{
   if (!kk_metal_log_enabled())
      return;
   unsigned occupied = 0;
   for (unsigned i = 0; i < KK_METAL_REF_CAPACITY; ++i)
      occupied += cache->refs[i].valid;
   char fields[512];
   snprintf(fields, sizeof(fields),
            ",\"archives\":%u,\"archive_peak\":%u,"
            "\"archive_bytes\":%llu,\"archive_peak_bytes\":%llu,"
            "\"archive_loads\":%llu,\"archive_evictions\":%llu,"
            "\"ref_slots\":%u,\"ref_occupied\":%u,\"ref_hits\":%llu,"
            "\"ref_misses\":%llu,\"ref_stale_hits\":%llu,"
            "\"ref_collisions\":%llu,\"ref_index_bytes\":%zu",
            cache->archive_count, cache->archive_peak,
            (unsigned long long)cache->archive_bytes,
            (unsigned long long)cache->archive_peak_bytes,
            (unsigned long long)cache->archive_loads,
            (unsigned long long)cache->archive_evictions, KK_METAL_REF_CAPACITY,
            occupied, (unsigned long long)cache->ref_hits,
            (unsigned long long)cache->ref_misses,
            (unsigned long long)cache->ref_stale_hits,
            (unsigned long long)cache->ref_collisions, sizeof(cache->refs));
   kk_metal_log_event("cache_summary", fields);
}

void
kk_metal_cache_request_flush(struct kk_metal_cache *cache)
{
   if (!cache)
      return;
   mtx_lock(&cache->mutex);
   if (!cache->closing) {
      cache->flush_requested++;
      cnd_broadcast(&cache->cond);
   }
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
   kk_metal_log_summary(cache);
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

static struct kk_metal_loading *
kk_metal_loading_find_locked(struct kk_metal_cache *cache,
                             const uint8_t ref[CACHE_KEY_SIZE])
{
   for (struct kk_metal_loading *entry = cache->loading; entry;
        entry = entry->next)
      if (!memcmp(entry->ref, ref, CACHE_KEY_SIZE))
         return entry;
   return NULL;
}

static void
kk_metal_loading_finish_locked(struct kk_metal_cache *cache,
                               struct kk_metal_loading *loading,
                               enum kk_metal_miss miss)
{
   loading->miss = miss;
   loading->done = true;
   for (struct kk_metal_loading **link = &cache->loading; *link;
        link = &(*link)->next) {
      if (*link == loading) {
         *link = loading->next;
         break;
      }
   }
   cnd_broadcast(&cache->cond);
   if (!--loading->users)
      free(loading);
}

static void
kk_metal_insert_archive_locked(struct kk_metal_cache *cache,
                               const uint8_t ref[CACHE_KEY_SIZE],
                               mtl_archive *archive, size_t size)
{
   cache->archive_loads++;
   struct kk_metal_loaded_archive *entry = malloc(sizeof(*entry));
   if (!entry)
      return;
   memcpy(entry->key, ref, CACHE_KEY_SIZE);
   entry->archive = mtl_retain(archive);
   entry->size = size;
   cache->archive_bytes += size;
   unsigned bucket = entry->key[0];
   entry->hash_next = cache->archives[bucket];
   cache->archives[bucket] = entry;
   kk_metal_archive_promote(cache, entry);
   if (++cache->archive_count > KK_METAL_ARCHIVE_CAPACITY) {
      struct kk_metal_loaded_archive *last = cache->archive_tail;
      struct kk_metal_loaded_archive **link = &cache->archives[last->key[0]];
      while (*link != last)
         link = &(*link)->hash_next;
      *link = last->hash_next;
      kk_metal_archive_unlink(cache, last);
      cache->archive_bytes -= last->size;
      cache->archive_evictions++;
      mtl_release(last->archive);
      free(last);
      cache->archive_count--;
   }
   cache->archive_peak = MAX2(cache->archive_peak, cache->archive_count);
   cache->archive_peak_bytes =
      MAX2(cache->archive_peak_bytes, cache->archive_bytes);
}

static mtl_archive *
kk_metal_acquire_archive(struct kk_metal_cache *cache,
                         const uint8_t key[CACHE_KEY_SIZE],
                         const uint8_t ref[CACHE_KEY_SIZE],
                         enum kk_metal_miss *miss_reason)
{
   mtx_lock(&cache->mutex);
   for (;;) {
      mtl_archive *archive = kk_metal_find_archive_locked(cache, ref);
      if (archive) {
         mtx_unlock(&cache->mutex);
         return archive;
      }
      struct kk_metal_loading *loading =
         kk_metal_loading_find_locked(cache, ref);
      if (!loading)
         break;
      loading->users++;
      while (!loading->done)
         cnd_wait(&cache->cond, &cache->mutex);
      enum kk_metal_miss miss = loading->miss;
      if (!--loading->users)
         free(loading);
      if (miss != KK_METAL_MISS_NONE) {
         kk_metal_ref_mark_stale_locked(cache, key, ref, miss);
         mtx_unlock(&cache->mutex);
         *miss_reason = miss;
         return NULL;
      }
   }
   struct kk_metal_loading *loading = calloc(1, sizeof(*loading));
   if (loading) {
      memcpy(loading->ref, ref, CACHE_KEY_SIZE);
      loading->users = 1;
      loading->next = cache->loading;
      cache->loading = loading;
   }
   mtx_unlock(&cache->mutex);

   int64_t load_start = kk_metal_trace_time();
   size_t size = 0;
   void *data = disk_cache_get(cache->disk, ref, &size);
   mtl_archive *archive = NULL;
   enum kk_metal_miss miss = KK_METAL_MISS_BLOB;
   if (data && size) {
      cache_key actual;
      disk_cache_compute_key(cache->disk, data, size, actual);
      miss = KK_METAL_MISS_CORRUPT;
      if (!memcmp(actual, ref, CACHE_KEY_SIZE)) {
         archive = mtl_new_archive(cache->device, data, size);
         miss = archive ? KK_METAL_MISS_NONE : KK_METAL_MISS_REJECTED;
      }
   }
   free(data);
   kk_metal_trace_event("archive_load", key, load_start);
   if (!archive)
      kk_metal_log_miss(kk_metal_miss_names[miss], key, load_start);

   mtx_lock(&cache->mutex);
   if (archive) {
      mtl_archive *existing = kk_metal_find_archive_locked(cache, ref);
      if (existing) {
         mtl_release(archive);
         archive = existing;
      } else {
         kk_metal_insert_archive_locked(cache, ref, archive, size);
      }
   } else {
      kk_metal_ref_mark_stale_locked(cache, key, ref, miss);
   }
   if (loading)
      kk_metal_loading_finish_locked(cache, loading, miss);
   mtx_unlock(&cache->mutex);
   *miss_reason = miss;
   return archive;
}

static mtl_archive *
kk_metal_load(struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
              enum kk_metal_miss *miss_reason)
{
   *miss_reason = KK_METAL_MISS_NONE;
   int64_t lookup_start = kk_metal_trace_time();
   cache_key ref;
   uint8_t stale = KK_METAL_MISS_NONE;
   mtx_lock(&cache->mutex);
   bool indexed = kk_metal_ref_get_locked(cache, key, ref, &stale);
   cache->ref_hits += indexed;
   cache->ref_misses += !indexed;
   cache->ref_stale_hits += indexed && stale != KK_METAL_MISS_NONE;
   mtx_unlock(&cache->mutex);
   if (indexed && stale != KK_METAL_MISS_NONE) {
      kk_metal_trace_event("archive_lookup", key, lookup_start);
      *miss_reason = stale;
      return NULL;
   }
   if (!indexed) {
      cache_key reference;
      kk_metal_reference_key(cache, key, reference);
      size_t ref_size = 0;
      void *stored = disk_cache_get(cache->disk, reference, &ref_size);
      if (!stored || ref_size != CACHE_KEY_SIZE) {
         kk_metal_trace_event("archive_lookup", key, lookup_start);
         kk_metal_log_miss("ref_missing", key, lookup_start);
         free(stored);
         *miss_reason = KK_METAL_MISS_REF;
         return NULL;
      }
      memcpy(ref, stored, CACHE_KEY_SIZE);
      free(stored);
      mtx_lock(&cache->mutex);
      kk_metal_ref_set_locked(cache, key, ref);
      mtx_unlock(&cache->mutex);
   }
   kk_metal_trace_event("archive_lookup", key, lookup_start);
   if (kk_metal_ref_is_uncacheable(ref)) {
      *miss_reason = KK_METAL_MISS_UNCACHEABLE;
      if (!indexed)
         kk_metal_log_miss("ref_uncacheable", key, lookup_start);
      return NULL;
   }
   return kk_metal_acquire_archive(cache, key, ref, miss_reason);
}

static bool
kk_metal_pending_locked(struct kk_metal_cache *cache,
                        const uint8_t key[CACHE_KEY_SIZE])
{
   for (struct kk_metal_cache_key *entry = cache->pending;
        entry; entry = entry->next)
      if (!memcmp(entry->key, key, CACHE_KEY_SIZE))
         return true;
   return false;
}

static void
kk_metal_capture_locked(struct kk_metal_cache *cache,
                         const uint8_t key[CACHE_KEY_SIZE])
{
   if (kk_metal_pending_locked(cache, key))
      return;
   struct kk_metal_cache_key *entry = malloc(sizeof(*entry));
   if (!entry) {
      cache->dropped_alloc++;
      kk_metal_log_drop("capture_alloc", key);
      return;
   }
   memcpy(entry->key, key, CACHE_KEY_SIZE);
   if (!cache->pending)
      cache->oldest_pending_ns = os_time_get_nano();
   entry->next = cache->pending;
   cache->pending = entry;
   cache->pending_count++;
}

static bool
kk_metal_recapture_known_locked(struct kk_metal_cache *cache,
                                const uint8_t key[CACHE_KEY_SIZE])
{
   if (cache->recapturing &&
       !memcmp(cache->recapturing->key, key, CACHE_KEY_SIZE))
      return true;
   for (struct kk_metal_recapture *entry = cache->recapture;
        entry; entry = entry->next)
      if (!memcmp(entry->key, key, CACHE_KEY_SIZE))
         return true;
   return kk_metal_pending_locked(cache, key);
}

static void
kk_metal_recapture_push_locked(struct kk_metal_cache *cache,
                               struct kk_metal_recapture *entry)
{
   entry->next = NULL;
   if (cache->recapture_tail)
      cache->recapture_tail->next = entry;
   else
      cache->recapture = entry;
   cache->recapture_tail = entry;
   cache->recapture_count++;
}

static struct kk_metal_recapture *
kk_metal_recapture_pop_locked(struct kk_metal_cache *cache)
{
   struct kk_metal_recapture *entry = cache->recapture;
   cache->recapture = entry->next;
   if (!cache->recapture)
      cache->recapture_tail = NULL;
   cache->recapture_count--;
   return entry;
}

static void
kk_metal_recapture_locked(struct kk_metal_cache *cache, bool drain)
{
   unsigned budget = KK_METAL_RECAPTURE_BATCH;
   while (cache->recapture && !kk_metal_batch_full_locked(cache) &&
          (drain || (budget && !cache->stopping &&
                     cache->flush_requested == cache->flush_completed))) {
      struct kk_metal_recapture *entry = kk_metal_recapture_pop_locked(cache);
      if (kk_metal_pending_locked(cache, entry->key)) {
         mtl_release(entry->descriptor);
         free(entry);
         continue;
      }
      budget -= !!budget;
      entry->attempts++;
      cache->recapturing = entry;
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
      mtx_lock(&cache->mutex);
      cache->recapturing = NULL;
      cache->captured_compiles--;
      cnd_broadcast(&cache->cond);
      if (pipeline) {
         kk_metal_capture_locked(cache, entry->key);
         cache->recaptured_compiles++;
      } else if (entry->attempts < KK_METAL_RECAPTURE_ATTEMPTS) {
         kk_metal_recapture_push_locked(cache, entry);
         continue;
      } else {
         cache->dropped_failed++;
         kk_metal_log_miss("recapture_failed", entry->key, 0);
         kk_metal_log_drop("recapture_failed", entry->key);
      }
      mtl_release(entry->descriptor);
      free(entry);
   }
}

static void
kk_metal_defer_recapture_locked(struct kk_metal_cache *cache,
                                const uint8_t key[CACHE_KEY_SIZE],
                                void *descriptor, bool compute,
                                uint64_t max_threads)
{
   if (kk_metal_recapture_known_locked(cache, key))
      return;
   if (cache->recapture_count >= KK_METAL_RECAPTURE_LIMIT) {
      cache->dropped_capacity++;
      kk_metal_log_drop("recapture_capacity", key);
      return;
   }
   struct kk_metal_recapture *entry = calloc(1, sizeof(*entry));
   if (!entry) {
      cache->dropped_alloc++;
      kk_metal_log_drop("recapture_alloc", key);
      return;
   }
   memcpy(entry->key, key, CACHE_KEY_SIZE);
   entry->descriptor = mtl_retain(descriptor);
   entry->max_threads = max_threads;
   entry->compute = compute;
   kk_metal_recapture_push_locked(cache, entry);
}

static mtl_compiler *
kk_metal_begin_compile(struct kk_metal_cache *cache,
                       const uint8_t key[CACHE_KEY_SIZE],
                       enum kk_metal_route *route, bool capture)
{
   mtx_lock(&cache->mutex);
   cache->last_compile_ns = os_time_get_nano();
   if (cache->uncaptured &&
       (cache->serializing || !capture || kk_metal_batch_full_locked(cache))) {
      if (cache->serializing) {
         struct mtl_compile_phase phase =
            kk_metal_phase_begin(cache, "serialization_bypass", key);
         mtl_compile_phase_end(phase, true);
         cache->bypassed_compiles++;
      }
      *route = !capture             ? KK_METAL_ROUTE_UNCACHEABLE
               : cache->serializing ? KK_METAL_ROUTE_SERIALIZING
                                    : KK_METAL_ROUTE_BATCH_FULL;
      cache->uncaptured_compiles++;
      mtl_compiler *compiler = cache->uncaptured;
      mtx_unlock(&cache->mutex);
      return compiler;
   }
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "serialization_wait", key);
   while (cache->serializing)
      cnd_wait(&cache->cond, &cache->mutex);
   mtl_compile_phase_end(phase, true);
   cache->captured_compiles++;
   mtx_unlock(&cache->mutex);
   *route = KK_METAL_ROUTE_CAPTURED;
   return cache->compiler;
}

static void
kk_metal_end_compile(struct kk_metal_cache *cache,
                     const uint8_t key[CACHE_KEY_SIZE],
                     enum kk_metal_route route, bool persist, void *descriptor,
                     bool compute, uint64_t max_threads)
{
   mtx_lock(&cache->mutex);
   if (route == KK_METAL_ROUTE_CAPTURED) {
      if (persist)
         kk_metal_capture_locked(cache, key);
      cache->captured_compiles--;
   } else {
      if (persist)
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

static mtl_compiler *
kk_metal_closed_compiler(struct kk_metal_cache *cache)
{
   return cache->uncaptured ? cache->uncaptured : cache->compiler;
}

static mtl_render_pipeline_state *
kk_metal_lookup_render_pipeline(struct kk_metal_cache *cache,
                                const uint8_t key[CACHE_KEY_SIZE],
                                mtl_render_pipeline_descriptor *descriptor,
                                enum kk_metal_miss *miss)
{
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "archive_load", key);
   mtl_archive *archive = kk_metal_load(cache, key, miss);
   mtl_compile_phase_end(phase, archive != NULL);
   mtl_render_pipeline_state *pipeline = NULL;
   if (archive) {
      int64_t start = kk_metal_trace_time();
      pipeline = mtl_archive_new_render_pipeline(archive, descriptor);
      kk_metal_trace_event("native_create", key, start);
      if (!pipeline) {
         *miss = KK_METAL_MISS_PIPELINE;
         kk_metal_log_miss("pipeline_absent", key, start);
      }
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
   enum kk_metal_miss miss;
   mtl_render_pipeline_state *pipeline =
      kk_metal_lookup_render_pipeline(cache, key, descriptor, &miss);
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
   if (!kk_metal_operation_begin(cache)) {
      int64_t start = os_time_get_nano();
      mtl_render_pipeline_state *pipeline =
         mtl_new_render_pipeline(kk_metal_closed_compiler(cache), descriptor);
      kk_metal_log_compile("render", key, KK_METAL_MISS_NONE,
                           KK_METAL_ROUTE_CLOSING, start, pipeline != NULL);
      return pipeline;
   }
   enum kk_metal_miss miss;
   mtl_render_pipeline_state *pipeline =
      kk_metal_lookup_render_pipeline(cache, key, descriptor, &miss);
   *archive_hit = pipeline != NULL;
   if (!pipeline) {
      bool uncacheable = miss == KK_METAL_MISS_UNCACHEABLE;
      enum kk_metal_route route;
      mtl_compiler *compiler =
         kk_metal_begin_compile(cache, key, &route, !uncacheable);
      int64_t start = kk_metal_trace_time();
      int64_t log_start = os_time_get_nano();
      pipeline = mtl_new_render_pipeline(compiler, descriptor);
      kk_metal_trace_event("native_compile", key, start);
      kk_metal_log_compile("render", key, miss, route, log_start,
                           pipeline != NULL);
      kk_metal_end_compile(cache, key, route, pipeline != NULL && !uncacheable,
                           descriptor, false, 0);
   }
   kk_metal_operation_end(cache);
   return pipeline;
}

mtl_render_pipeline_state *
kk_metal_cache_specialize_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor, mtl_render_pipeline_state *base)
{
   if (!kk_metal_operation_begin(cache))
      return mtl_specialize_render_pipeline(kk_metal_closed_compiler(cache),
                                            descriptor, base);
   enum kk_metal_route route;
   mtl_compiler *compiler = kk_metal_begin_compile(cache, key, &route, false);
   int64_t start = kk_metal_trace_time();
   mtl_render_pipeline_state *pipeline =
      mtl_specialize_render_pipeline(compiler, descriptor, base);
   kk_metal_trace_event("native_specialize", key, start);
   kk_metal_end_compile(cache, key, route, false, descriptor, false, 0);
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
   if (!kk_metal_operation_begin(cache)) {
      int64_t start = os_time_get_nano();
      mtl_compute_pipeline_state *pipeline = mtl_new_compute_pipeline_state(
         kk_metal_closed_compiler(cache), function, max_threads);
      kk_metal_log_compile("compute", key, KK_METAL_MISS_NONE,
                           KK_METAL_ROUTE_CLOSING, start, pipeline != NULL);
      return pipeline;
   }
   struct mtl_compile_phase phase =
      kk_metal_phase_begin(cache, "archive_load", key);
   enum kk_metal_miss miss;
   mtl_archive *archive = kk_metal_load(cache, key, &miss);
   mtl_compile_phase_end(phase, archive != NULL);
   mtl_compute_pipeline_state *pipeline = NULL;
   if (archive) {
      int64_t start = kk_metal_trace_time();
      pipeline =
         mtl_archive_new_compute_pipeline(archive, function, max_threads);
      kk_metal_trace_event("native_create", key, start);
      if (!pipeline) {
         miss = KK_METAL_MISS_PIPELINE;
         kk_metal_log_miss("pipeline_absent", key, start);
      }
   }
   mtl_release(archive);
   *archive_hit = pipeline != NULL;
   if (!pipeline) {
      bool uncacheable = miss == KK_METAL_MISS_UNCACHEABLE;
      enum kk_metal_route route;
      mtl_compiler *compiler =
         kk_metal_begin_compile(cache, key, &route, !uncacheable);
      int64_t start = kk_metal_trace_time();
      int64_t log_start = os_time_get_nano();
      pipeline =
         mtl_new_compute_pipeline_state(compiler, function, max_threads);
      kk_metal_trace_event("native_compile", key, start);
      kk_metal_log_compile("compute", key, miss, route, log_start,
                           pipeline != NULL);
      kk_metal_end_compile(cache, key, route, pipeline != NULL && !uncacheable,
                           function, true, max_threads);
   }
   kk_metal_operation_end(cache);
   return pipeline;
}
