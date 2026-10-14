#define mtl_new_render_pipeline mock_new_render_pipeline
#define mtl_specialize_render_pipeline mock_specialize_render_pipeline
#define mtl_new_compute_pipeline_state mock_new_compute_pipeline_state
#define mtl_new_uncaptured_compiler mock_new_uncaptured_compiler
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

static int captured_compiler, uncaptured_compiler, device, pipeline;

static struct {
   mtx_t lock;
   cnd_t cond;
   struct kk_metal_cache *cache;
   uint32_t calls[MAX_CALLS];
   unsigned call_count;
   unsigned uncaptured_calls;
   uint32_t captured[MAX_KEYS];
   unsigned captured_count;
   size_t key_bytes;
   unsigned serializes;
   int first_serialize_queue;
   uint32_t block_id;
   bool blocked;
   unsigned archive_hits;
   struct mock_archive **archives;
   unsigned archive_count;
   unsigned archives_dead;
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
   if (!fail && compiler == (mtl_compiler *)&captured_compiler) {
      CHECK(mock.captured_count < MAX_KEYS);
      mock.captured[mock.captured_count++] = desc->id;
   }
   mtx_unlock(&mock.lock);
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
   if (!mock.serializes++)
      mock.first_serialize_queue = queued;
   *size = sizeof(uint32_t) * (2 + mock.captured_count) +
           mock.key_bytes * mock.captured_count;
   uint32_t *data = calloc(1, *size);
   data[0] = mock.serializes;
   data[1] = mock.captured_count;
   memcpy(data + 2, mock.captured, mock.captured_count * sizeof(uint32_t));
   mock.captured_count = 0;
   mtx_unlock(&mock.lock);
   return data;
}

mtl_archive *
mock_new_archive(mtl_device *dev, const void *data, size_t size)
{
   const uint32_t *words = data;
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

static void
defer_locked(struct kk_metal_cache *cache, struct mock_desc *desc)
{
   cache_key key;
   desc_key(desc, key);
   kk_metal_defer_recapture_locked(cache, key, desc, false, 0);
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
                  !cache->captured_compiles;
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
   kk_metal_capture_locked(cache, key);
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
   mtx_lock(&cache->mutex);
   CHECK(!cache->dropped_capacity && !cache->dropped_failed && !cache->pending);
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
   return kk_metal_cache_new_render_pipeline(
      cache, key, (mtl_render_pipeline_descriptor *)desc, hit);
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
   CHECK(cache_limit(cache) == 4);
   kk_metal_cache_destroy(cache);

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
   kk_metal_cache_flush(cache);
   CHECK(cache_limit(cache) == 8);
   CHECK(log_contains(path, "\"outcome\":\"archive_uncacheable\""));
   CHECK(!log_contains(path, "archive_limit_lowered"));
   CHECK(compile_desc(cache, &big, &hit) && !hit);
   mtx_lock(&mock.lock);
   CHECK(mock.call_count == 1 && mock.uncaptured_calls == 1);
   mtx_unlock(&mock.lock);
   kk_metal_cache_destroy(cache);

   mock_reset(false);
   mock.key_bytes = 10000;
   cache = make_cache(4500, 0);
   CHECK(compile_desc(cache, &big, &hit) && !hit);
   mtx_lock(&cache->mutex);
   CHECK(!cache->recapture && !cache->pending);
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

int
main(void)
{
   setenv("MESA_KK_PSO_CACHE_EXIT_FLUSH", "false", 1);
   setenv("MESA_KK_PSO_CACHE_QUIET_MS", "1000000000", 1);
   setenv("MESA_KK_PSO_CACHE_MAX_AGE_MS", "1000000000", 1);
   CHECK(mtx_init(&mock.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&mock.cond) == thrd_success);
   test_recapture_order_and_dedup();
   test_recapture_failures();
   test_recapture_yields_to_flush();
   test_oversized_batch_limit_persists();
   test_single_oversized_pso();
   test_archive_eviction_in_use();
   test_destroy_drains_operation();
   mock_reset(true);
   cnd_destroy(&mock.cond);
   mtx_destroy(&mock.lock);
   return 0;
}
