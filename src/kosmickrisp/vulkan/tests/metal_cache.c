#define mtl_new_render_pipeline mock_new_render_pipeline
#define mtl_specialize_render_pipeline mock_specialize_render_pipeline
#define mtl_new_compute_pipeline_state mock_new_compute_pipeline_state
#define mtl_new_uncaptured_compiler mock_new_uncaptured_compiler
#define mtl_new_compiler mock_new_compiler
#define mtl_compiler_has_serializer mock_compiler_has_serializer
#define mtl_compiler_serialize mock_compiler_serialize
#define mtl_pipeline_cache_identity mock_pipeline_cache_identity
#define mtl_new_archive mock_new_archive
#define mtl_archive_new_render_pipeline mock_archive_new_render_pipeline
#define mtl_archive_new_compute_pipeline mock_archive_new_compute_pipeline
#define mtl_retain mock_retain
#define mtl_release mock_release
#define mtl_compile_phase_enabled mock_compile_phase_enabled
#define mtl_compile_phase_begin mock_compile_phase_begin
#define mtl_compile_phase_end mock_compile_phase_end
#define disk_cache_compute_key mock_disk_cache_compute_key
#define disk_cache_get mock_disk_cache_get
#define disk_cache_put mock_disk_cache_put
#define disk_cache_wait_for_idle mock_disk_cache_wait_for_idle
#include "../kk_metal_cache.c"

#include "util/mesa-blake3.h"

#include <stdio.h>
#include <unistd.h>

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort(); \
   } \
} while (0)

#define MAX_CALLS 1024
#define MAX_KEYS 512
#define MAX_DISK 4096

struct mock_desc {
   uint32_t id;
   unsigned failures;
   uint32_t vertex;
   uint32_t fragment;
};

#define ARCHIVE_ALIVE 0x41524348u
#define ARCHIVE_DEAD 0xdeadu

struct mock_archive {
   uint32_t magic;
   int refs;
   unsigned count;
   uint32_t ids[MAX_KEYS];
};

struct mock_entry {
   cache_key key;
   void *data;
   size_t size;
};

static int captured_compiler, captured_compiler2, uncaptured_compiler, device,
   pipeline;

static struct {
   mtx_t lock;
   cnd_t cond;
   struct kk_metal_cache *cache;
   uint32_t calls[MAX_CALLS];
   unsigned call_count;
   unsigned uncaptured_calls;
   uint32_t captured[MAX_KEYS];
   uint32_t captured_vertex[MAX_KEYS];
   uint32_t captured_fragment[MAX_KEYS];
   unsigned captured_count;
   uint32_t captured2[MAX_KEYS];
   uint32_t captured2_vertex[MAX_KEYS];
   uint32_t captured2_fragment[MAX_KEYS];
   unsigned captured2_count;
   bool serialize_gate;
   bool serialize_blocked;
   size_t key_bytes;
   unsigned serializes;
   int first_serialize_queue;
   uint32_t block_id;
   bool blocked;
   unsigned archive_hits;
   struct mock_archive **archives;
   unsigned archive_count;
   unsigned archives_dead;
   unsigned archive_creates;
   bool archive_gate;
   bool archive_gated;
   bool archive_reject;
   unsigned disk_gets;
   unsigned serialize_failures;
   unsigned max_batch;
   unsigned delay_ms;
   struct mock_entry disk[MAX_DISK];
   unsigned disk_count;
} mock;

bool
mock_compile_phase_enabled(void)
{
   return false;
}

struct mtl_compile_phase
mock_compile_phase_begin(const char *operation, const void *object,
                         const char *key, const char *origin)
{
   return (struct mtl_compile_phase){0};
}

void
mock_compile_phase_end(struct mtl_compile_phase phase, bool success)
{
}

static struct mock_archive *
mock_find_archive_locked(void *object)
{
   for (unsigned i = 0; i < mock.archive_count; ++i)
      if (mock.archives[i] == object)
         return object;
   return NULL;
}

void *
mock_retain(void *object)
{
   mtx_lock(&mock.lock);
   struct mock_archive *archive = mock_find_archive_locked(object);
   if (archive) {
      CHECK(archive->magic == ARCHIVE_ALIVE && archive->refs > 0);
      archive->refs++;
   }
   mtx_unlock(&mock.lock);
   return object;
}

void
mock_release(void *object)
{
   mtx_lock(&mock.lock);
   struct mock_archive *archive = mock_find_archive_locked(object);
   if (archive) {
      CHECK(archive->magic == ARCHIVE_ALIVE && archive->refs > 0);
      if (!--archive->refs) {
         archive->magic = ARCHIVE_DEAD;
         mock.archives_dead++;
      }
   }
   mtx_unlock(&mock.lock);
}

mtl_compiler *
mock_new_compiler(mtl_device *dev)
{
   return (mtl_compiler *)&captured_compiler2;
}

mtl_compiler *
mock_new_uncaptured_compiler(mtl_device *dev)
{
   return (mtl_compiler *)&uncaptured_compiler;
}

bool
mock_compiler_has_serializer(mtl_compiler *compiler)
{
   return true;
}

void
mock_pipeline_cache_identity(mtl_device *dev, uint32_t gpu_family,
                             char *identity, size_t size)
{
   snprintf(identity, size, "kk-metal-cache-test");
}

static void *
mock_compile(mtl_compiler *compiler, void *descriptor)
{
   struct mock_desc *desc = descriptor;
   mtx_lock(&mock.lock);
   if (compiler == (mtl_compiler *)&uncaptured_compiler) {
      mock.uncaptured_calls++;
   } else {
      CHECK(mock.call_count < MAX_CALLS);
      mock.calls[mock.call_count++] = desc->id;
   }
   if (desc->id == mock.block_id) {
      mock.blocked = true;
      cnd_broadcast(&mock.cond);
      while (mock.block_id == desc->id)
         cnd_wait(&mock.cond, &mock.lock);
      mock.blocked = false;
   }
   bool fail = desc->failures > 0;
   if (fail)
      desc->failures--;
   bool second = compiler == (mtl_compiler *)&captured_compiler2;
   if (!fail && (second || compiler == (mtl_compiler *)&captured_compiler)) {
      uint32_t *ids = second ? mock.captured2 : mock.captured;
      uint32_t *vertex = second ? mock.captured2_vertex : mock.captured_vertex;
      uint32_t *fragment =
         second ? mock.captured2_fragment : mock.captured_fragment;
      unsigned *count = second ? &mock.captured2_count : &mock.captured_count;
      CHECK(*count < MAX_KEYS);
      for (unsigned i = 0; i < *count; ++i)
         CHECK(!desc->vertex || desc->vertex != vertex[i] ||
               desc->fragment == fragment[i]);
      vertex[*count] = desc->vertex;
      fragment[*count] = desc->fragment;
      ids[(*count)++] = desc->id;
   }
   unsigned delay = mock.delay_ms;
   mtx_unlock(&mock.lock);
   if (delay)
      thrd_sleep(&(struct timespec){ .tv_nsec = delay * 1000000l }, NULL);
   return fail ? NULL : &pipeline;
}

mtl_render_pipeline_state *
mock_new_render_pipeline(mtl_compiler *compiler,
                         mtl_render_pass_descriptor *descriptor)
{
   return mock_compile(compiler, descriptor);
}

mtl_render_pipeline_state *
mock_specialize_render_pipeline(mtl_compiler *compiler,
                                mtl_render_pipeline_descriptor *descriptor,
                                mtl_render_pipeline_state *base)
{
   return mock_compile(compiler, descriptor);
}

mtl_compute_pipeline_state *
mock_new_compute_pipeline_state(mtl_compiler *compiler,
                                mtl_function_descriptor *function,
                                uint64_t max_threads)
{
   return mock_compile(compiler, function);
}

void *
mock_compiler_serialize(mtl_compiler *compiler, size_t *size)
{
   mtx_lock(&mock.cache->mutex);
   int queued = mock.cache->recapture_count;
   mtx_unlock(&mock.cache->mutex);
   mtx_lock(&mock.lock);
   mock.serialize_blocked = mock.serialize_gate;
   cnd_broadcast(&mock.cond);
   while (mock.serialize_gate)
      cnd_wait(&mock.cond, &mock.lock);
   mock.serialize_blocked = false;
   if (mock.serialize_failures) {
      mock.serialize_failures--;
      mtx_unlock(&mock.lock);
      *size = 0;
      return NULL;
   }
   bool second = compiler == (mtl_compiler *)&captured_compiler2;
   uint32_t *ids = second ? mock.captured2 : mock.captured;
   unsigned *count = second ? &mock.captured2_count : &mock.captured_count;
   mock.max_batch = MAX2(mock.max_batch, *count);
   if (!mock.serializes++)
      mock.first_serialize_queue = queued;
   *size = sizeof(uint32_t) * (2 + *count) + mock.key_bytes * *count;
   uint32_t *data = calloc(1, *size);
   data[0] = mock.serializes;
   data[1] = *count;
   memcpy(data + 2, ids, *count * sizeof(uint32_t));
   *count = 0;
   mtx_unlock(&mock.lock);
   return data;
}

mtl_archive *
mock_new_archive(mtl_device *dev, const void *data, size_t size)
{
   const uint32_t *words = data;
   mtx_lock(&mock.lock);
   mock.archive_creates++;
   mock.archive_gated = mock.archive_gate;
   cnd_broadcast(&mock.cond);
   while (mock.archive_gate)
      cnd_wait(&mock.cond, &mock.lock);
   mock.archive_gated = false;
   bool reject = mock.archive_reject;
   mtx_unlock(&mock.lock);
   if (reject)
      return NULL;
   struct mock_archive *archive = calloc(1, sizeof(*archive));
   archive->magic = ARCHIVE_ALIVE;
   archive->refs = 1;
   archive->count = words[1];
   memcpy(archive->ids, words + 2, archive->count * sizeof(uint32_t));
   mtx_lock(&mock.lock);
   mock.archives = realloc(mock.archives,
                           (mock.archive_count + 1) * sizeof(*mock.archives));
   CHECK(mock.archives);
   mock.archives[mock.archive_count++] = archive;
   mtx_unlock(&mock.lock);
   return (mtl_archive *)archive;
}

static void *
mock_archive_lookup(mtl_archive *object, void *descriptor)
{
   struct mock_archive *archive = (struct mock_archive *)object;
   struct mock_desc *desc = descriptor;
   mtx_lock(&mock.lock);
   CHECK(archive->magic == ARCHIVE_ALIVE && archive->refs > 0);
   mtx_unlock(&mock.lock);
   for (unsigned i = 0; i < archive->count; ++i) {
      if (archive->ids[i] == desc->id) {
         mtx_lock(&mock.lock);
         mock.archive_hits++;
         mtx_unlock(&mock.lock);
         return &pipeline;
      }
   }
   return NULL;
}

mtl_render_pipeline_state *
mock_archive_new_render_pipeline(mtl_archive *archive,
                                 mtl_render_pipeline_descriptor *descriptor)
{
   return mock_archive_lookup(archive, descriptor);
}

mtl_compute_pipeline_state *
mock_archive_new_compute_pipeline(mtl_archive *archive,
                                  mtl_function_descriptor *function,
                                  uint64_t max_threads)
{
   return mock_archive_lookup(archive, function);
}

void
mock_disk_cache_compute_key(struct disk_cache *cache, const void *data,
                            size_t size, cache_key key)
{
   _mesa_blake3_compute(data, size, key);
}

static struct mock_entry *
mock_disk_find(const cache_key key)
{
   for (unsigned i = 0; i < mock.disk_count; ++i)
      if (!memcmp(mock.disk[i].key, key, CACHE_KEY_SIZE))
         return &mock.disk[i];
   return NULL;
}

void
mock_disk_cache_put(struct disk_cache *cache, const cache_key key,
                    const void *data, size_t size,
                    struct cache_item_metadata *metadata)
{
   mtx_lock(&mock.lock);
   struct mock_entry *entry = mock_disk_find(key);
   if (!entry) {
      CHECK(mock.disk_count < MAX_DISK);
      entry = &mock.disk[mock.disk_count++];
      memcpy(entry->key, key, CACHE_KEY_SIZE);
   } else {
      free(entry->data);
   }
   entry->data = malloc(size);
   memcpy(entry->data, data, size);
   entry->size = size;
   mtx_unlock(&mock.lock);
}

void *
mock_disk_cache_get(struct disk_cache *cache, const cache_key key,
                    size_t *size)
{
   mtx_lock(&mock.lock);
   mock.disk_gets++;
   struct mock_entry *entry = mock_disk_find(key);
   void *data = NULL;
   if (entry) {
      data = malloc(entry->size);
      memcpy(data, entry->data, entry->size);
      *size = entry->size;
   }
   mtx_unlock(&mock.lock);
   return data;
}

void
mock_disk_cache_wait_for_idle(struct disk_cache *cache)
{
}

static void
mock_reset(bool clear_disk)
{
   mtx_lock(&mock.lock);
   mock.call_count = mock.uncaptured_calls = mock.captured_count = 0;
   mock.captured2_count = 0;
   mock.serialize_gate = mock.serialize_blocked = false;
   mock.serializes = 0;
   mock.first_serialize_queue = -1;
   mock.block_id = 0;
   mock.archive_hits = 0;
   mock.key_bytes = 64;
   for (unsigned i = 0; i < mock.archive_count; ++i) {
      CHECK(mock.archives[i]->magic == ARCHIVE_DEAD && !mock.archives[i]->refs);
      free(mock.archives[i]);
   }
   free(mock.archives);
   mock.archives = NULL;
   mock.archive_count = mock.archives_dead = 0;
   mock.archive_creates = mock.disk_gets = 0;
   mock.serialize_failures = mock.max_batch = mock.delay_ms = 0;
   mock.archive_gate = mock.archive_gated = mock.archive_reject = false;
   if (clear_disk) {
      for (unsigned i = 0; i < mock.disk_count; ++i)
         free(mock.disk[i].data);
      mock.disk_count = 0;
   }
   mtx_unlock(&mock.lock);
}

static struct kk_metal_cache *
make_cache(uint64_t budget, unsigned max_pending)
{
   struct kk_metal_cache *cache =
      kk_metal_cache_create((struct disk_cache *)&device, (mtl_device *)&device,
                            0, (mtl_compiler *)&captured_compiler, budget);
   CHECK(cache);
   mtx_lock(&cache->mutex);
   if (max_pending)
      cache->max_pending = max_pending;
   mtx_unlock(&cache->mutex);
   mock.cache = cache;
   return cache;
}

static void
desc_key(const struct mock_desc *desc, cache_key key)
{
   _mesa_blake3_compute(&desc->id, sizeof(desc->id), key);
}

static struct kk_metal_capture_key
capture_key(const struct mock_desc *desc)
{
   struct kk_metal_capture_key key = {0};
   memcpy(key.vertex, &desc->vertex, sizeof(desc->vertex));
   memcpy(key.fragment, &desc->fragment, sizeof(desc->fragment));
   return key;
}

static void
defer_locked(struct kk_metal_cache *cache, struct mock_desc *desc)
{
   cache_key key;
   desc_key(desc, key);
   struct kk_metal_capture_key group = capture_key(desc);
   kk_metal_defer_recapture_locked(cache, key, desc, false, 0,
                                   desc->vertex ? &group : NULL);
}

static bool
pending_has(struct kk_metal_cache *cache, const struct mock_desc *desc)
{
   cache_key key;
   desc_key(desc, key);
   mtx_lock(&cache->mutex);
   bool found = kk_metal_pending_locked(cache, key);
   mtx_unlock(&cache->mutex);
   return found;
}

static void
wait_recapture_idle(struct kk_metal_cache *cache)
{
   int64_t give_up = os_time_get_nano() + 5000000000ll;
   for (;;) {
      mtx_lock(&cache->mutex);
      cnd_broadcast(&cache->cond);
      bool idle = !cache->recapture && !cache->recapturing &&
                  !kk_metal_captured_locked(cache);
      mtx_unlock(&cache->mutex);
      if (idle)
         return;
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   }
}

static unsigned
calls_for(uint32_t id)
{
   unsigned count = 0;
   mtx_lock(&mock.lock);
   for (unsigned i = 0; i < mock.call_count; ++i)
      count += mock.calls[i] == id;
   mtx_unlock(&mock.lock);
   return count;
}

static void
test_recapture_order_and_dedup(void)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 1000);
   struct mock_desc a = { 1 }, b = { 2 }, c = { 3 }, d = { 4 };
   cache_key key;
   desc_key(&c, key);
   mtx_lock(&cache->mutex);
   kk_metal_capture_locked(cache, kk_metal_active(cache), key);
   defer_locked(cache, &a);
   defer_locked(cache, &a);
   defer_locked(cache, &b);
   defer_locked(cache, &a);
   defer_locked(cache, &c);
   defer_locked(cache, &d);
   CHECK(cache->recapture_count == 3);
   mtx_unlock(&cache->mutex);
   wait_recapture_idle(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 3);
   CHECK(mock.calls[0] == 1 && mock.calls[1] == 2 && mock.calls[2] == 4);
   mtx_unlock(&mock.lock);
   CHECK(pending_has(cache, &a) && pending_has(cache, &b) &&
         pending_has(cache, &c) && pending_has(cache, &d));

   struct mock_desc e = { 5 };
   mtx_lock(&mock.lock);
   mock.block_id = e.id;
   mtx_unlock(&mock.lock);
   mtx_lock(&cache->mutex);
   defer_locked(cache, &e);
   mtx_unlock(&cache->mutex);
   mtx_lock(&cache->mutex);
   cnd_broadcast(&cache->cond);
   mtx_unlock(&cache->mutex);
   mtx_lock(&mock.lock);
   while (!mock.blocked)
      cnd_wait(&mock.cond, &mock.lock);
   mtx_unlock(&mock.lock);
   mtx_lock(&cache->mutex);
   defer_locked(cache, &e);
   CHECK(!cache->recapture && cache->recapturing);
   mtx_unlock(&cache->mutex);
   mtx_lock(&mock.lock);
   mock.block_id = 0;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   wait_recapture_idle(cache);
   CHECK(calls_for(e.id) == 1 && pending_has(cache, &e));
   kk_metal_cache_destroy(cache);
}

static bool
log_contains(const char *path, const char *needle)
{
   FILE *file = fopen(path, "r");
   if (!file)
      return false;
   char line[1024];
   bool found = false;
   while (!found && fgets(line, sizeof(line), file))
      found = strstr(line, needle) != NULL;
   fclose(file);
   return found;
}

static bool
log_has_drop(const char *path, struct mock_desc *desc, const char *reason)
{
   cache_key key;
   desc_key(desc, key);
   char hex[CACHE_KEY_SIZE * 2 + 1], needle[256];
   kk_metal_format_key(key, hex);
   snprintf(needle, sizeof(needle),
            "\"outcome\":\"persist_dropped\",\"key\":\"%s\",\"reason\":\"%s\"",
            hex, reason);
   return log_contains(path, needle);
}

static bool
log_has_compile(const char *path, struct mock_desc *desc, const char *reason,
                const char *route)
{
   cache_key key;
   desc_key(desc, key);
   char hex[CACHE_KEY_SIZE * 2 + 1], needle[256];
   kk_metal_format_key(key, hex);
   snprintf(needle, sizeof(needle),
            "\"outcome\":\"native_compile\",\"key\":\"%s\",\"kind\":\"render\","
            "\"reason\":\"%s\",\"route\":\"%s\"",
            hex, reason, route);
   return log_contains(path, needle);
}

static void
test_recapture_failures(void)
{
   mock_reset(true);
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   struct kk_metal_cache *cache = make_cache(0, 1000);
   struct mock_desc f = { 10, 1 }, g = { 11, 5 };
   mtx_lock(&cache->mutex);
   defer_locked(cache, &f);
   defer_locked(cache, &g);
   mtx_unlock(&cache->mutex);
   wait_recapture_idle(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 4);
   CHECK(mock.calls[0] == 10 && mock.calls[1] == 11 &&
         mock.calls[2] == 10 && mock.calls[3] == 11);
   mtx_unlock(&mock.lock);
   CHECK(pending_has(cache, &f) && !pending_has(cache, &g));

   static struct mock_desc many[KK_METAL_RECAPTURE_LIMIT + 1];
   mtx_lock(&cache->mutex);
   CHECK(cache->dropped_failed == 1);
   for (unsigned i = 0; i < ARRAY_SIZE(many); ++i) {
      many[i] = (struct mock_desc){ 1000 + i };
      defer_locked(cache, &many[i]);
   }
   CHECK(cache->recapture_count == KK_METAL_RECAPTURE_LIMIT);
   CHECK(cache->dropped_capacity == 1 && cache->dropped_alloc == 0);
   mtx_unlock(&cache->mutex);
   wait_recapture_idle(cache);
   kk_metal_cache_flush(cache);
   CHECK(log_contains(path, "\"dropped\":2,\"dropped_capacity\":1,"
                            "\"dropped_alloc\":0,\"dropped_failed\":1"));
   CHECK(log_has_drop(path, &g, "recapture_failed"));
   CHECK(!log_has_drop(path, &f, "recapture_failed"));
   CHECK(log_has_drop(path, &many[KK_METAL_RECAPTURE_LIMIT],
                      "recapture_capacity"));
   CHECK(!log_has_drop(path, &many[0], "recapture_capacity"));
   mtx_lock(&cache->mutex);
   CHECK(!cache->dropped_capacity && !cache->dropped_failed && !kk_metal_dirty_locked(cache));
   mtx_unlock(&cache->mutex);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_recapture_yields_to_flush(void)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 4);
   static struct mock_desc burst[64];
   mtx_lock(&cache->mutex);
   for (unsigned i = 0; i < ARRAY_SIZE(burst); ++i) {
      burst[i] = (struct mock_desc){ 2000 + i };
      defer_locked(cache, &burst[i]);
   }
   mtx_unlock(&cache->mutex);
   wait_recapture_idle(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == ARRAY_SIZE(burst));
   CHECK(mock.serializes >= 1);
   CHECK(mock.first_serialize_queue > 0);
   CHECK(mock.calls[0] == 2000 && mock.calls[63] == 2063);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
}

static mtl_render_pipeline_state *
compile_desc(struct kk_metal_cache *cache, struct mock_desc *desc, bool *hit)
{
   cache_key key;
   desc_key(desc, key);
   struct kk_metal_capture_key group = capture_key(desc);
   return kk_metal_cache_new_render_pipeline(
      cache, key, (mtl_render_pipeline_descriptor *)desc,
      desc->vertex ? &group : NULL, hit);
}

static void
wait_serializes(unsigned count)
{
   int64_t give_up = os_time_get_nano() + 5000000000ll;
   for (;;) {
      mtx_lock(&mock.lock);
      bool done = mock.serializes >= count;
      mtx_unlock(&mock.lock);
      if (done)
         return;
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   }
}

static unsigned
cache_limit(struct kk_metal_cache *cache)
{
   mtx_lock(&cache->mutex);
   unsigned limit = cache->max_pending;
   mtx_unlock(&cache->mutex);
   return limit;
}

static void
test_oversized_batch_limit_persists(void)
{
   mock_reset(true);
   mock.key_bytes = 1000;
   const uint64_t budget = 6500;
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   setenv("MESA_KK_PSO_CACHE_MAX_PENDING", "8", 1);
   struct mock_desc keys[8];
   for (unsigned i = 0; i < ARRAY_SIZE(keys); ++i)
      keys[i] = (struct mock_desc){ 3000 + i };

   struct kk_metal_cache *cache = make_cache(budget, 0);
   CHECK(cache_limit(cache) == 8);
   bool hit;
   for (unsigned i = 0; i < ARRAY_SIZE(keys); ++i)
      CHECK(compile_desc(cache, &keys[i], &hit) && !hit);
   wait_serializes(1);
   wait_recapture_idle(cache);
   kk_metal_cache_flush(cache);
   CHECK(cache_limit(cache) == 4);
   for (unsigned i = 0; i < ARRAY_SIZE(keys); ++i) {
      CHECK(log_has_drop(path, &keys[i], "oversized_batch"));
      CHECK(log_has_compile(path, &keys[i], "ref_missing", "captured"));
   }
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);

   mock_reset(false);
   mock.key_bytes = 1000;
   cache = make_cache(budget, 0);
   CHECK(cache_limit(cache) == 4);
   for (unsigned i = 0; i < ARRAY_SIZE(keys); ++i)
      CHECK(compile_desc(cache, &keys[i], &hit) && !hit);
   wait_recapture_idle(cache);
   kk_metal_cache_flush(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes >= 2 && mock.archive_hits == 0);
   CHECK(mock.call_count == ARRAY_SIZE(keys));
   mtx_unlock(&mock.lock);
   CHECK(cache_limit(cache) == 4);
   kk_metal_cache_destroy(cache);

   mock_reset(false);
   cache = make_cache(budget, 0);
   for (unsigned i = 0; i < ARRAY_SIZE(keys); ++i)
      CHECK(compile_desc(cache, &keys[i], &hit) && hit);
   mtx_lock(&mock.lock);
   CHECK(mock.archive_hits == ARRAY_SIZE(keys));
   CHECK(mock.call_count == 0 && mock.uncaptured_calls == 0);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_PSO_CACHE_MAX_PENDING");
}

static void
test_single_oversized_pso(void)
{
   mock_reset(true);
   mock.key_bytes = 10000;
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   setenv("MESA_KK_PSO_CACHE_MAX_PENDING", "8", 1);
   struct mock_desc big = { 4000 }, small = { 4001 };
   struct kk_metal_cache *cache = make_cache(4500, 0);
   bool hit;
   CHECK(compile_desc(cache, &big, &hit) && !hit);
   CHECK(log_has_compile(path, &big, "ref_missing", "captured"));
   kk_metal_cache_flush(cache);
   CHECK(cache_limit(cache) == 8);
   CHECK(log_contains(path, "\"outcome\":\"archive_uncacheable\""));
   CHECK(!log_contains(path, "archive_limit_lowered"));
   CHECK(!log_has_drop(path, &big, "oversized_batch"));
   CHECK(compile_desc(cache, &big, &hit) && !hit);
   CHECK(
      log_has_compile(path, &big, "ref_uncacheable", "uncaptured_uncacheable"));
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 1 && mock.uncaptured_calls == 1);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);

   mock_reset(false);
   mock.key_bytes = 10000;
   cache = make_cache(4500, 0);
   CHECK(compile_desc(cache, &big, &hit) && !hit);
   mtx_lock(&cache->mutex);
   CHECK(!cache->recapture && !kk_metal_dirty_locked(cache));
   mtx_unlock(&cache->mutex);
   mock.key_bytes = 100;
   CHECK(compile_desc(cache, &small, &hit) && !hit);
   kk_metal_cache_flush(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.uncaptured_calls == 1 && mock.call_count == 1);
   CHECK(mock.calls[0] == small.id && mock.serializes == 1);
   mtx_unlock(&mock.lock);
   CHECK(log_contains(path, "\"outcome\":\"ref_uncacheable\""));
   kk_metal_cache_destroy(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes == 1);
   mtx_unlock(&mock.lock);
   unsetenv("MESA_KK_PSO_CACHE_MAX_PENDING");
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

struct cold_user {
   struct kk_metal_cache *cache;
   struct mock_desc *desc;
   bool hit;
   bool compiled;
};

static int
run_cold_user(void *data)
{
   struct cold_user *user = data;
   user->compiled = compile_desc(user->cache, user->desc, &user->hit) != NULL;
   return 0;
}

static void
persist_descs(struct mock_desc *descs, unsigned count)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 0);
   bool hit;
   for (unsigned i = 0; i < count; ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   kk_metal_cache_flush(cache);
   kk_metal_cache_destroy(cache);
}

static unsigned
loading_users(struct kk_metal_cache *cache)
{
   mtx_lock(&cache->mutex);
   unsigned users = cache->loading ? cache->loading->users : 0;
   mtx_unlock(&cache->mutex);
   return users;
}

static void
run_cold_users(struct kk_metal_cache *cache, struct mock_desc *descs,
               struct cold_user *users, unsigned count)
{
   mtx_lock(&mock.lock);
   mock.archive_gate = true;
   mtx_unlock(&mock.lock);
   thrd_t threads[8];
   for (unsigned i = 0; i < count; ++i) {
      users[i] = (struct cold_user){.cache = cache, .desc = &descs[i]};
      CHECK(thrd_create(&threads[i], run_cold_user, &users[i]) == thrd_success);
   }
   int64_t give_up = os_time_get_nano() + 5000000000ll;
   while (loading_users(cache) < count) {
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){.tv_nsec = 1000000}, NULL);
   }
   mtx_lock(&mock.lock);
   CHECK(mock.archive_gated && mock.archive_creates == 1);
   mock.archive_gate = false;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   for (unsigned i = 0; i < count; ++i)
      CHECK(thrd_join(threads[i], NULL) == thrd_success);
}

static void
test_concurrent_cold_archive(void)
{
   struct mock_desc descs[4];
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      descs[i] = (struct mock_desc){6000 + i};
   persist_descs(descs, ARRAY_SIZE(descs));

   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(false);
   struct kk_metal_cache *cache = make_cache(0, 0);
   struct cold_user users[ARRAY_SIZE(descs)];
   run_cold_users(cache, descs, users, ARRAY_SIZE(descs));
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      CHECK(users[i].compiled && users[i].hit);
   bool hit;
   CHECK(compile_desc(cache, &descs[0], &hit) && hit);
   mtx_lock(&mock.lock);
   CHECK(mock.archive_creates == 1 &&
         mock.archive_hits == ARRAY_SIZE(descs) + 1);
   CHECK(mock.call_count == 0 && mock.uncaptured_calls == 0);
   mtx_unlock(&mock.lock);
   CHECK(!cache->loading);
   kk_metal_cache_destroy(cache);
   CHECK(log_contains(path, "\"archives\":1,\"archive_peak\":1,"));
   CHECK(log_contains(path, "\"archive_loads\":1,\"archive_evictions\":0,"));
   CHECK(log_contains(path, "\"ref_occupied\":4,\"ref_hits\":1,"
                            "\"ref_misses\":4,\"ref_stale_hits\":0,"
                            "\"ref_collisions\":0,"));
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_ref_collision(void)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 0);
   struct mock_desc a = {7000}, b = {7001};
   cache_key ka, kb;
   desc_key(&a, ka);
   for (;; b.id++) {
      desc_key(&b, kb);
      if (kk_metal_ref_slot(cache, ka) == kk_metal_ref_slot(cache, kb))
         break;
   }
   cache_key ref_a, ref_b, out;
   memset(ref_a, 1, sizeof(ref_a));
   memset(ref_b, 2, sizeof(ref_b));
   uint8_t stale;
   mtx_lock(&cache->mutex);
   kk_metal_ref_set_locked(cache, ka, ref_a);
   kk_metal_ref_set_locked(cache, ka, ref_a);
   CHECK(cache->ref_collisions == 0);
   kk_metal_ref_set_locked(cache, kb, ref_b);
   CHECK(cache->ref_collisions == 1);
   CHECK(!kk_metal_ref_get_locked(cache, ka, out, &stale));
   CHECK(kk_metal_ref_get_locked(cache, kb, out, &stale) &&
         !memcmp(out, ref_b, sizeof(out)) && !stale);
   mtx_unlock(&cache->mutex);
   kk_metal_cache_destroy(cache);
}

static void
test_rejected_archive_waiters(void)
{
   struct mock_desc descs[4];
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      descs[i] = (struct mock_desc){6100 + i};
   persist_descs(descs, ARRAY_SIZE(descs));

   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(false);
   mock.archive_reject = true;
   struct kk_metal_cache *cache = make_cache(0, 0);
   struct cold_user users[ARRAY_SIZE(descs)];
   run_cold_users(cache, descs, users, ARRAY_SIZE(descs));
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      CHECK(users[i].compiled && !users[i].hit);
      CHECK(log_has_compile(path, &descs[i], "archive_rejected", "captured"));
   }
   mtx_lock(&mock.lock);
   CHECK(mock.archive_creates == 1 && mock.call_count == ARRAY_SIZE(descs));
   unsigned gets = mock.disk_gets;
   mtx_unlock(&mock.lock);
   cache_key key;
   desc_key(&descs[0], key);
   CHECK(!kk_metal_cache_lookup_render_pipeline(
      cache, key, (mtl_render_pipeline_descriptor *)&descs[0]));
   mtx_lock(&mock.lock);
   CHECK(mock.disk_gets == gets && mock.archive_creates == 1);
   mock.archive_reject = false;
   mtx_unlock(&mock.lock);

   kk_metal_cache_flush(cache);
   bool hit;
   CHECK(compile_desc(cache, &descs[0], &hit) && hit);
   mtx_lock(&mock.lock);
   CHECK(mock.archive_creates == 2 && mock.call_count == ARRAY_SIZE(descs));
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static struct mock_entry *
mock_blob_entry(void)
{
   struct mock_entry *blob = NULL;
   for (unsigned i = 0; i < mock.disk_count; ++i)
      if (mock.disk[i].size > CACHE_KEY_SIZE)
         blob = &mock.disk[i];
   return blob;
}

static void
test_damaged_blob(bool corrupt)
{
   struct mock_desc desc = {corrupt ? 6200 : 6201};
   persist_descs(&desc, 1);
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(false);
   mtx_lock(&mock.lock);
   struct mock_entry *blob = mock_blob_entry();
   CHECK(blob);
   if (corrupt)
      ((uint8_t *)blob->data)[blob->size - 1] ^= 0xff;
   else
      blob->size = 0;
   mtx_unlock(&mock.lock);

   struct kk_metal_cache *cache = make_cache(0, 0);
   bool hit;
   CHECK(compile_desc(cache, &desc, &hit) && !hit);
   const char *reason = corrupt ? "blob_corrupt" : "blob_missing";
   CHECK(log_has_compile(path, &desc, reason, "captured"));
   CHECK(log_contains(path, corrupt ? "\"outcome\":\"blob_corrupt\""
                                    : "\"outcome\":\"blob_missing\""));
   mtx_lock(&mock.lock);
   unsigned gets = mock.disk_gets;
   mtx_unlock(&mock.lock);
   cache_key key;
   desc_key(&desc, key);
   CHECK(!kk_metal_cache_lookup_render_pipeline(
      cache, key, (mtl_render_pipeline_descriptor *)&desc));
   mtx_lock(&mock.lock);
   CHECK(mock.disk_gets == gets && mock.archive_creates == 0);
   mtx_unlock(&mock.lock);
   kk_metal_cache_flush(cache);
   CHECK(compile_desc(cache, &desc, &hit) && hit);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_shutdown_drain_respects_limit(void)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 2);
   struct mock_desc descs[5];
   mtx_lock(&cache->mutex);
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      descs[i] = (struct mock_desc){8000 + i};
      defer_locked(cache, &descs[i]);
   }
   mtx_unlock(&cache->mutex);
   kk_metal_cache_destroy(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.max_batch && mock.max_batch <= 2);
   CHECK(mock.serializes >= 3);
   mtx_unlock(&mock.lock);

   mock_reset(false);
   cache = make_cache(0, 0);
   bool hit;
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && hit);
   kk_metal_cache_destroy(cache);
}

static void
test_final_flush_failure(void)
{
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 0);
   struct mock_desc descs[3];
   bool hit;
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      descs[i] = (struct mock_desc){8100 + i};
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   }
   mtx_lock(&mock.lock);
   mock.serialize_failures = 100;
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serialize_failures == 100 - KK_METAL_FINAL_FLUSH_ATTEMPTS);
   CHECK(!mock.serializes);
   mtx_unlock(&mock.lock);
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      CHECK(log_has_drop(path, &descs[i], "final_flush_failed"));
   CHECK(log_contains(path, "\"outcome\":\"final_flush_failed\",\"keys\":3"));
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_closing_compiles_uncached(void)
{
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 0);
   kk_metal_cache_close(cache);
   struct mock_desc desc = {8200};
   bool hit = true;
   CHECK(compile_desc(cache, &desc, &hit) && !hit);
   mtx_lock(&mock.lock);
   CHECK(mock.uncaptured_calls == 1 && mock.call_count == 0);
   mtx_unlock(&mock.lock);
   CHECK(log_has_compile(path, &desc, "none", "closing"));
   mtx_lock(&cache->mutex);
   CHECK(!kk_metal_dirty_locked(cache) && !cache->recapture && !cache->active_operations);
   mtx_unlock(&cache->mutex);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_capture_group_age(void)
{
   struct kk_metal_cache cache = {
      .buffer_count = 1,
      .uncaptured = (void *)&uncaptured_compiler,
      .last_compile_ns = os_time_get_nano(),
      .quiet_ns = 1500000000ll,
      .max_age_ns = 60000000000ll,
      .max_pending = 32,
   };
   struct kk_metal_buffer *buffer = kk_metal_active(&cache);
   struct kk_metal_capture_key group = {.vertex = {1}, .fragment = {1}};
   for (unsigned i = 0; i < 2; ++i) {
      CHECK(kk_metal_capture_admit_locked(buffer, &group));
      buffer->captured_compiles = 1;
      CHECK(!kk_metal_flush_due_locked(&cache, os_time_get_nano()));
      buffer->captured_compiles = 0;
      kk_metal_capture_groups_clear(buffer);
      buffer->oldest_pending_ns = os_time_get_nano() - 120000000000ll;
   }
}

static void
test_groups_only_flush_discarded(void)
{
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(4, 0);
   unsigned limit = cache_limit(cache);
   struct mock_desc failing = {
      .id = 14000, .failures = 1, .vertex = 1, .fragment = 1,
   };
   bool hit;
   CHECK(!compile_desc(cache, &failing, &hit) && !hit);
   mtx_lock(&cache->mutex);
   CHECK(kk_metal_active(cache)->capture_groups &&
         !kk_metal_active(cache)->pending &&
         kk_metal_active(cache)->oldest_pending_ns);
   mtx_unlock(&cache->mutex);
   mtx_lock(&mock.lock);
   unsigned disk = mock.disk_count;
   mtx_unlock(&mock.lock);
   kk_metal_cache_flush(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes == 1 && mock.disk_count == disk);
   mtx_unlock(&mock.lock);
   mtx_lock(&cache->mutex);
   CHECK(!kk_metal_dirty_locked(cache) &&
         !kk_metal_active(cache)->capture_conflict);
   mtx_unlock(&cache->mutex);
   CHECK(cache_limit(cache) == limit);
   CHECK(log_contains(path, "\"outcome\":\"archive_discarded\""));
   CHECK(!log_contains(path, "\"outcome\":\"archive_flushed\""));
   CHECK(!log_contains(path, "\"outcome\":\"archive_oversized\""));
   CHECK(!log_contains(path, "archive_limit_lowered"));
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_shutdown_budget(void)
{
   char path[] = "/tmp/kk-metal-cache-log-XXXXXX";
   int fd = mkstemp(path);
   CHECK(fd >= 0);
   close(fd);
   setenv("MESA_KK_CACHE_LOG", path, 1);
   setenv("MESA_KK_PSO_CACHE_SHUTDOWN_MS", "60", 1);
   mock_reset(true);
   mock.delay_ms = 20;
   struct kk_metal_cache *cache = make_cache(0, 0);
   static struct mock_desc descs[24];
   mtx_lock(&cache->mutex);
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      descs[i] = (struct mock_desc){ .id = 15000 + i };
      defer_locked(cache, &descs[i]);
   }
   mtx_unlock(&cache->mutex);
   int64_t start = os_time_get_nano();
   kk_metal_cache_destroy(cache);
   int64_t elapsed = os_time_get_nano() - start;
   CHECK(elapsed < 350000000ll);
   mtx_lock(&mock.lock);
   unsigned compiled = mock.call_count;
   CHECK(compiled > 0 && compiled < ARRAY_SIZE(descs) && mock.serializes >= 1);
   for (unsigned i = 0; i < compiled; ++i)
      CHECK(mock.calls[i] == descs[i].id);
   mock.delay_ms = 0;
   mtx_unlock(&mock.lock);
   unsigned skipped = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      bool dropped = log_has_drop(path, &descs[i], "shutdown_budget");
      CHECK(dropped == (i >= compiled));
      skipped += dropped;
   }
   CHECK(skipped == ARRAY_SIZE(descs) - compiled);
   CHECK(log_contains(path, "\"outcome\":\"shutdown_budget_expired\""));
   CHECK(!log_contains(path, "\"outcome\":\"final_flush_failed\""));

   mock_reset(false);
   cache = make_cache(0, 0);
   bool hit;
   CHECK(compile_desc(cache, &descs[0], &hit) && hit);
   CHECK(compile_desc(cache, &descs[compiled - 1], &hit) && hit);
   CHECK(compile_desc(cache, &descs[ARRAY_SIZE(descs) - 1], &hit) && !hit);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_PSO_CACHE_SHUTDOWN_MS");
   unsetenv("MESA_KK_CACHE_LOG");
   unlink(path);
}

static void
test_flush_both_buffers(void)
{
   setenv("MESA_KK_PSO_CACHE_DOUBLE_BUFFER", "true", 1);
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 4);
   struct mock_desc descs[7];
   mtx_lock(&mock.lock);
   mock.serialize_gate = true;
   mock.serialize_failures = 1;
   mtx_unlock(&mock.lock);
   bool hit;
   for (unsigned i = 0; i < 4; ++i) {
      descs[i] = (struct mock_desc){.id = 17000 + i};
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   }
   int64_t give_up = os_time_get_nano() + 5000000000ll;
   mtx_lock(&mock.lock);
   while (!mock.serialize_blocked) {
      mtx_unlock(&mock.lock);
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){.tv_nsec = 1000000}, NULL);
      mtx_lock(&mock.lock);
   }
   mtx_unlock(&mock.lock);
   for (unsigned i = 4; i < ARRAY_SIZE(descs); ++i) {
      descs[i] = (struct mock_desc){.id = 17000 + i};
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   }
   mtx_lock(&mock.lock);
   mock.serialize_gate = false;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   for (;;) {
      mtx_lock(&cache->mutex);
      bool failed = cache->retry_after_ns != 0;
      mtx_unlock(&cache->mutex);
      if (failed)
         break;
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){.tv_nsec = 1000000}, NULL);
   }
   kk_metal_cache_flush(cache);
   mtx_lock(&cache->mutex);
   CHECK(!kk_metal_dirty_locked(cache));
   mtx_unlock(&cache->mutex);
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && hit);
   kk_metal_cache_destroy(cache);
   setenv("MESA_KK_PSO_CACHE_DOUBLE_BUFFER", "false", 1);
}

static void
test_double_buffer(bool enabled, bool fail)
{
   setenv("MESA_KK_PSO_CACHE_DOUBLE_BUFFER", enabled ? "true" : "false", 1);
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 4);
   CHECK(cache->buffer_count == (enabled ? 2u : 1u));
   static struct mock_desc first[4], second[3];
   mtx_lock(&mock.lock);
   mock.serialize_gate = true;
   mock.serialize_failures = fail ? 1 : 0;
   mtx_unlock(&mock.lock);
   bool hit;
   for (unsigned i = 0; i < ARRAY_SIZE(first); ++i) {
      first[i] = (struct mock_desc){ .id = 16000 + i };
      CHECK(compile_desc(cache, &first[i], &hit) && !hit);
   }
   int64_t give_up = os_time_get_nano() + 5000000000ll;
   mtx_lock(&mock.lock);
   while (!mock.serialize_blocked) {
      mtx_unlock(&mock.lock);
      CHECK(os_time_get_nano() < give_up);
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
      mtx_lock(&mock.lock);
   }
   mtx_unlock(&mock.lock);
   for (unsigned i = 0; i < ARRAY_SIZE(second); ++i) {
      second[i] = (struct mock_desc){ .id = 16100 + i };
      CHECK(compile_desc(cache, &second[i], &hit) && !hit);
   }
   mtx_lock(&mock.lock);
   if (enabled)
      CHECK(mock.uncaptured_calls == 0 &&
            mock.captured2_count == ARRAY_SIZE(second));
   else
      CHECK(mock.uncaptured_calls == ARRAY_SIZE(second));
   mock.serialize_gate = false;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   wait_recapture_idle(cache);
   kk_metal_cache_destroy(cache);

   mock_reset(false);
   cache = make_cache(0, 0);
   for (unsigned i = 0; i < ARRAY_SIZE(first); ++i)
      CHECK(compile_desc(cache, &first[i], &hit) && hit);
   for (unsigned i = 0; i < ARRAY_SIZE(second); ++i)
      CHECK(compile_desc(cache, &second[i], &hit) && hit);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 0 && mock.uncaptured_calls == 0);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
   setenv("MESA_KK_PSO_CACHE_DOUBLE_BUFFER", "false", 1);
}

struct hit_loop {
   struct kk_metal_cache *cache;
   struct mock_desc *descs;
   unsigned first, count, rounds;
};

static int
run_hit_loop(void *data)
{
   struct hit_loop *loop = data;
   for (unsigned round = 0; round < loop->rounds; ++round) {
      for (unsigned i = loop->first; i < loop->first + loop->count; ++i) {
         bool hit;
         CHECK(compile_desc(loop->cache, &loop->descs[i], &hit) && hit);
      }
   }
   return 0;
}

static void
test_archive_eviction_in_use(void)
{
   mock_reset(true);
   static struct mock_desc descs[KK_METAL_ARCHIVE_CAPACITY + 44];
   struct kk_metal_cache *cache = make_cache(0, 1);
   bool hit;
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      descs[i] = (struct mock_desc){ 5000 + i };
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
      kk_metal_cache_flush(cache);
   }
   wait_recapture_idle(cache);
   kk_metal_cache_flush(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes >= ARRAY_SIZE(descs));
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);

   mock_reset(false);
   cache = make_cache(0, 1000);
   struct hit_loop hot = { cache, descs, 0, 1, 3000 };
   struct hit_loop sweep = { cache, descs, 1, ARRAY_SIZE(descs) - 1, 2 };
   thrd_t hot_thread, sweep_thread;
   CHECK(thrd_create(&hot_thread, run_hit_loop, &hot) == thrd_success);
   CHECK(thrd_create(&sweep_thread, run_hit_loop, &sweep) == thrd_success);
   CHECK(thrd_join(hot_thread, NULL) == thrd_success);
   CHECK(thrd_join(sweep_thread, NULL) == thrd_success);
   mtx_lock(&cache->mutex);
   CHECK(cache->archive_count == KK_METAL_ARCHIVE_CAPACITY);
   mtx_unlock(&cache->mutex);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 0 && mock.uncaptured_calls == 0);
   CHECK(mock.archives_dead >= ARRAY_SIZE(descs) - KK_METAL_ARCHIVE_CAPACITY);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.archives_dead == mock.archive_count);
   mtx_unlock(&mock.lock);
}

struct destroy_op {
   struct kk_metal_cache *cache;
   struct mock_desc *desc;
   void *result;
   bool done;
};

static int
run_compile_op(void *data)
{
   struct destroy_op *op = data;
   bool hit;
   op->result = compile_desc(op->cache, op->desc, &hit);
   return 0;
}

static int
run_destroy(void *data)
{
   struct destroy_op *op = data;
   kk_metal_cache_destroy(op->cache);
   p_atomic_set(&op->done, true);
   return 0;
}

static void
test_destroy_drains_operation(void)
{
   mock_reset(true);
   struct mock_desc desc = { 6000 };
   struct kk_metal_cache *cache = make_cache(0, 1000);
   struct destroy_op op = { cache, &desc };
   mtx_lock(&mock.lock);
   mock.block_id = desc.id;
   mtx_unlock(&mock.lock);
   thrd_t compile, destroy;
   CHECK(thrd_create(&compile, run_compile_op, &op) == thrd_success);
   mtx_lock(&mock.lock);
   while (!mock.blocked)
      cnd_wait(&mock.cond, &mock.lock);
   mtx_unlock(&mock.lock);
   CHECK(thrd_create(&destroy, run_destroy, &op) == thrd_success);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   CHECK(!p_atomic_read(&op.done));
   mtx_lock(&cache->mutex);
   CHECK(cache->closing && cache->active_operations == 1);
   mtx_unlock(&cache->mutex);
   mtx_lock(&mock.lock);
   mock.block_id = 0;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   CHECK(thrd_join(compile, NULL) == thrd_success);
   CHECK(thrd_join(destroy, NULL) == thrd_success);
   CHECK(op.result && op.done);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes == 1);
   mtx_unlock(&mock.lock);

   mock_reset(false);
   cache = make_cache(0, 1000);
   bool hit;
   CHECK(compile_desc(cache, &desc, &hit) && hit);
   kk_metal_cache_destroy(cache);
}

static void
test_linked_vertex_batches(bool bypass_disabled, bool shutdown, bool fail)
{
   mock_reset(true);
   setenv("MESA_KK_PSO_CACHE_BYPASS_DISABLE",
          bypass_disabled ? "true" : "false", 1);
   struct kk_metal_cache *cache = make_cache(0, 1000);
   struct mock_desc descs[] = {
      {.id = 12000, .vertex = 1, .fragment = 1},
      {.id = 12001, .vertex = 1, .fragment = 1},
      {.id = 12002, .vertex = 2, .fragment = 2},
      {.id = 12003},
      {.id = 12004, .vertex = 1, .fragment = 2},
      {.id = 12005, .vertex = 1, .fragment = 3},
   };
   bool hit;
   for (unsigned i = 0; i < 4; ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes == 0 && mock.captured_count == 4);
   mock.serialize_failures = fail ? 1 : 0;
   mtx_unlock(&mock.lock);
   for (unsigned i = 4; i < ARRAY_SIZE(descs); ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && !hit);
   if (!shutdown) {
      wait_recapture_idle(cache);
      kk_metal_cache_flush(cache);
   }
   kk_metal_cache_destroy(cache);
   mtx_lock(&mock.lock);
   CHECK(mock.serializes == 3 && mock.max_batch == 4);
   mtx_unlock(&mock.lock);
   mock_reset(false);
   cache = make_cache(0, 1000);
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      CHECK(compile_desc(cache, &descs[i], &hit) && hit);
   CHECK(mock.call_count == 0 && mock.uncaptured_calls == 0);
   kk_metal_cache_destroy(cache);
   unsetenv("MESA_KK_PSO_CACHE_BYPASS_DISABLE");
}

static void
test_linked_vertex_inflight(void)
{
   mock_reset(true);
   struct kk_metal_cache *cache = make_cache(0, 1000);
   struct mock_desc first = {.id = 13000, .vertex = 1, .fragment = 1};
   struct mock_desc second = {.id = 13001, .vertex = 1, .fragment = 2};
   struct cold_user user = {.cache = cache, .desc = &first};
   mock.block_id = first.id;
   thrd_t thread;
   CHECK(thrd_create(&thread, run_cold_user, &user) == thrd_success);
   mtx_lock(&mock.lock);
   while (!mock.blocked)
      cnd_wait(&mock.cond, &mock.lock);
   mtx_unlock(&mock.lock);
   bool hit;
   CHECK(compile_desc(cache, &second, &hit) && !hit);
   mtx_lock(&mock.lock);
   CHECK(mock.uncaptured_calls == 1 && mock.serializes == 0);
   mock.block_id = 0;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(user.compiled && !user.hit);
   kk_metal_cache_destroy(cache);
   CHECK(mock.serializes == 2);
   mock_reset(false);
   cache = make_cache(0, 1000);
   CHECK(compile_desc(cache, &first, &hit) && hit);
   CHECK(compile_desc(cache, &second, &hit) && hit);
   kk_metal_cache_destroy(cache);
}

int
main(void)
{
   setenv("MESA_KK_PSO_CACHE_EXIT_FLUSH", "false", 1);
   setenv("MESA_KK_PSO_CACHE_DOUBLE_BUFFER", "false", 1);
   setenv("MESA_KK_PSO_CACHE_QUIET_MS", "1000000000", 1);
   setenv("MESA_KK_PSO_CACHE_MAX_AGE_MS", "1000000000", 1);
   CHECK(mtx_init(&mock.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&mock.cond) == thrd_success);
   for (unsigned i = 0; i < 8; ++i)
      test_linked_vertex_batches(i & 1, i & 2, i & 4);
   test_linked_vertex_inflight();
   test_capture_group_age();
   test_groups_only_flush_discarded();
   test_shutdown_budget();
   test_flush_both_buffers();
   for (unsigned i = 0; i < 4; ++i)
      test_double_buffer(i & 1, i & 2);
   test_recapture_order_and_dedup();
   test_recapture_failures();
   test_recapture_yields_to_flush();
   test_oversized_batch_limit_persists();
   test_single_oversized_pso();
   test_archive_eviction_in_use();
   test_destroy_drains_operation();
   test_concurrent_cold_archive();
   test_ref_collision();
   test_rejected_archive_waiters();
   test_damaged_blob(true);
   test_damaged_blob(false);
   test_shutdown_drain_respects_limit();
   test_final_flush_failure();
   test_closing_compiles_uncached();
   mock_reset(true);
   cnd_destroy(&mock.cond);
   mtx_destroy(&mock.lock);
   return 0;
}
