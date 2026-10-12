#include <stdlib.h>
#include "util/u_queue.h"

static void *test_calloc(size_t count, size_t size);
static bool test_queue_init(struct util_queue *queue, const char *name,
                            unsigned max_jobs, unsigned num_threads,
                            unsigned flags, void *global_data);

#define calloc test_calloc
#define util_queue_init test_queue_init
#define mtl_new_render_pipeline test_render_pipeline
#define mtl_release test_release
#define __vk_errorf test_vk_errorf
#include "../kk_pso_cache.c"
#undef mtl_new_render_pipeline
#undef mtl_release
#undef __vk_errorf
#undef calloc
#undef util_queue_init

void mtl_release(void *object);

VkResult
test_vk_errorf(const void *object, VkResult error, const char *file, int line,
               const char *format, ...)
{
   return error;
}

#include <stdio.h>

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort(); \
   } \
} while (0)

__THREAD_INITIAL_EXEC enum kk_compile_origin kk_compile_origin;
__THREAD_INITIAL_EXEC blake3_hash kk_variant_parent_id;
__THREAD_INITIAL_EXEC bool kk_variant_parent_valid;
__THREAD_INITIAL_EXEC blake3_hash kk_variant_family_id;
__THREAD_INITIAL_EXEC bool kk_variant_family_valid;
enum kk_debug kk_mesa_debug_flags;

static unsigned fail_next_allocation;
static bool fail_stub_queue;
static enum kk_wait_timeout_policy failed_queue_policy;

static void *
test_calloc(size_t count, size_t size)
{
   if (p_atomic_xchg(&fail_next_allocation, 0))
      return NULL;
   return calloc(count, size);
}

static bool
test_queue_init(struct util_queue *queue, const char *name,
                unsigned max_jobs, unsigned num_threads, unsigned flags,
                void *global_data)
{
   if (fail_stub_queue && !strcmp(name, "kk-stub")) {
      struct kk_pso_cache *cache =
         container_of(queue, struct kk_pso_cache, stub_queue);
      failed_queue_policy = cache->timeout_policy;
      return false;
   }
   return util_queue_init(
      queue, name, max_jobs, num_threads, flags, global_data);
}

static struct {
   mtx_t lock;
   cnd_t ready;
   bool entered;
   bool release;
   bool fail;
   unsigned calls;
   unsigned blocking_calls;
} producer;
static int pipelines[16];

mtl_render_pipeline_state *
test_render_pipeline(mtl_compiler *compiler, mtl_render_pass_descriptor *desc)
{
   mtx_lock(&producer.lock);
   unsigned call = producer.calls++;
   producer.entered = true;
   cnd_broadcast(&producer.ready);
   while (!producer.release &&
          (!producer.blocking_calls || call < producer.blocking_calls))
      cnd_wait(&producer.ready, &producer.lock);
   bool fail = producer.fail;
   mtx_unlock(&producer.lock);
   return fail ? NULL :
      (mtl_render_pipeline_state *)&pipelines[MIN2(call, 15)];
}

void
test_release(void *object)
{
   if ((int *)object < pipelines || (int *)object >= pipelines + 16)
      mtl_release(object);
}

void kk_log_compile(struct kk_device *dev, const char *kind, int64_t start) {}
void kk_log_cache_pso(struct kk_device *dev, const char *event,
                       const void *key, size_t size) {}
void kk_variant_trace(struct kk_device *dev, enum kk_variant_phase phase,
                       const void *key, size_t size, int64_t start,
                       int64_t end) {}
void kk_variant_metal_event(void *data, const char *phase, const uint8_t *key,
                             size_t size, int64_t start, int64_t end) {}

const struct kk_fs_variant *
kk_shader_get_rast(struct kk_device *dev, struct kk_shader *vs, bool points)
{
   return &vs->fast;
}

VkResult
kk_shader_get_fs_variant(struct kk_device *dev, struct kk_shader *fs,
                         const struct kk_fs_link_key *key,
                         const struct kk_fs_variant **out,
                         const struct kk_pso_request *request)
{
   *out = &fs->fast;
   return VK_SUCCESS;
}

void
kk_shader_variant_identity(struct kk_shader *shader, blake3_hash output)
{
   memcpy(output, shader->fast.hash, sizeof(blake3_hash));
}

struct kk_shader *
kk_device_get_empty_fs(struct kk_device *dev)
{
   return NULL;
}

bool
kk_shader_rast_ready(struct kk_shader *vs, bool points)
{
   return true;
}

static struct kk_fs_variant discard_fs = { .hash = { 0xd1 } };
static bool discard_fs_created;

const struct kk_fs_variant *
kk_device_get_discard_fs(struct kk_device *dev, bool create)
{
   if (create)
      p_atomic_set(&discard_fs_created, true);
   return p_atomic_read(&discard_fs_created) ? &discard_fs : NULL;
}

static struct {
   mtx_t lock;
   cnd_t ready;
   bool entered;
   bool release;
} queue_gate;

static void
block_queue(void *data, void *global, int thread)
{
   mtx_lock(&queue_gate.lock);
   queue_gate.entered = true;
   cnd_broadcast(&queue_gate.ready);
   while (!queue_gate.release)
      cnd_wait(&queue_gate.ready, &queue_gate.lock);
   mtx_unlock(&queue_gate.lock);
}

struct demand {
   struct kk_device *dev;
   struct kk_shader *vs;
   struct kk_shader *fs;
   struct kk_variant_recipe recipe;
   struct kk_pso_result result;
   enum kk_compile_origin origin;
};

static int
run_demand(void *data)
{
   struct demand *d = data;
   kk_compile_origin = d->origin;
   d->result = kk_recipe_get(d->dev, d->vs, d->fs, &d->recipe, NULL);
   return 0;
}

static void
reset_producer(unsigned blocking_calls)
{
   mtx_lock(&producer.lock);
   producer.entered = producer.release = producer.fail = false;
   producer.calls = 0;
   producer.blocking_calls = blocking_calls;
   mtx_unlock(&producer.lock);
}

static void
wait_for_producer(void)
{
   mtx_lock(&producer.lock);
   while (!producer.entered)
      cnd_wait(&producer.ready, &producer.lock);
   mtx_unlock(&producer.lock);
}

static void
release_producer(void)
{
   mtx_lock(&producer.lock);
   producer.release = true;
   cnd_broadcast(&producer.ready);
   mtx_unlock(&producer.lock);
}

static unsigned
producer_calls(void)
{
   mtx_lock(&producer.lock);
   unsigned calls = producer.calls;
   mtx_unlock(&producer.lock);
   return calls;
}

static void
wait_for_calls(unsigned count)
{
   int64_t give_up = os_time_get_nano() + 2000000000ll;
   while (producer_calls() < count && os_time_get_nano() < give_up)
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   CHECK(producer_calls() == count);
}

static void
test_recipe(bool fail)
{
   struct kk_device dev = {0};
   struct kk_physical_device pdev = {0};
   dev.vk.physical = &pdev.vk;
   dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&dev.pso_cache));
   struct kk_shader vs = {0}, fs = {0};
   struct demand d = { .dev = &dev, .vs = &vs, .fs = &fs,
                       .origin = KK_COMPILE_ORIGIN_DRAW };
   kk_pso_key_init(&d.recipe.pso);
   mtx_lock(&dev.pso_cache.lock);
   struct kk_recipe_entry *entry =
      kk_recipe_insert_locked(&dev.pso_cache, &d.recipe);
   CHECK(entry);
   entry->queued = true;
   mtx_unlock(&dev.pso_cache.lock);
   producer.entered = producer.release = false;
   producer.fail = fail;
   producer.calls = 0;

   thrd_t owner, late_worker;
   CHECK(thrd_create(&owner, run_demand, &d) == thrd_success);
   wait_for_producer();
   mtx_lock(&dev.pso_cache.lock);
   CHECK(!entry->queued && entry->compiling);
   mtx_unlock(&dev.pso_cache.lock);

   struct demand late = d;
   late.origin = KK_COMPILE_ORIGIN_PREWARM;
   CHECK(thrd_create(&late_worker, run_demand, &late) == thrd_success);
   struct kk_pso_request request = kk_pso_request_create(20);
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   struct kk_pso_result timed =
      kk_recipe_get(&dev, &vs, &fs, &d.recipe, &request);
   CHECK(timed.status == VK_TIMEOUT && !timed.pso);
   mtx_lock(&dev.pso_cache.lock);
   CHECK(entry->compiling && !entry->complete && !entry->drawn &&
         entry->demanded);
   mtx_unlock(&dev.pso_cache.lock);

   struct kk_variant_recipe other = d.recipe;
   other.family[0] = 1;
   request = kk_pso_request_create(20);
   timed = kk_recipe_get(&dev, &vs, &fs, &other, &request);
   CHECK(timed.status == VK_TIMEOUT && !timed.pso);
   mtx_lock(&dev.pso_cache.lock);
   struct kk_recipe_entry *retry =
      _mesa_hash_table_search(dev.pso_cache.recipes, &other)->data;
   CHECK(!retry->compiling && !retry->queued && !retry->complete &&
         !retry->drawn);
   mtx_unlock(&dev.pso_cache.lock);

   release_producer();
   CHECK(thrd_join(owner, NULL) == thrd_success);
   CHECK(thrd_join(late_worker, NULL) == thrd_success);
   CHECK(producer_calls() == 1);
   CHECK(d.result.status == (fail ? VK_ERROR_UNKNOWN : VK_SUCCESS));
   CHECK(d.result.pso == late.result.pso);
   CHECK(late.result.status == d.result.status);
   struct kk_pso_result reused = kk_recipe_get(&dev, &vs, &fs, &other, NULL);
   CHECK(reused.status == d.result.status && reused.pso == d.result.pso);
   CHECK(producer_calls() == 1 && retry->complete);
   CHECK(retry->drawn == !fail);
   CHECK(!dev.pso_cache.pending_jobs && !dev.pso_cache.exact_jobs);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_wait_budget(void)
{
   bool compiling = true;
   struct kk_pso_request request = kk_pso_request_create(20);
   mtx_lock(&producer.lock);
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &compiling,
                     &request) == VK_TIMEOUT);
   int64_t expired = os_time_get_nano();
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &compiling,
                     &request) == VK_TIMEOUT);
   CHECK(os_time_get_nano() - expired < 100000000);
   CHECK(compiling);
   compiling = false;
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &compiling,
                     &request) == VK_SUCCESS);
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &compiling,
                     NULL) == VK_SUCCESS);
   mtx_unlock(&producer.lock);
}

struct wait_test {
   bool compiling;
   bool stop;
   unsigned wakes;
};

static int
spurious_wakes(void *data)
{
   struct wait_test *test = data;
   for (;;) {
      mtx_lock(&producer.lock);
      if (test->stop) {
         mtx_unlock(&producer.lock);
         return 0;
      }
      test->wakes++;
      cnd_broadcast(&producer.ready);
      mtx_unlock(&producer.lock);
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   }
}

static int
complete_after_deadline(void *data)
{
   struct wait_test *test = data;
   mtx_lock(&producer.lock);
   thrd_sleep(&(struct timespec){ .tv_nsec = 30000000 }, NULL);
   test->compiling = false;
   cnd_broadcast(&producer.ready);
   mtx_unlock(&producer.lock);
   return 0;
}

static void
test_wait_wakeups(void)
{
   struct wait_test test = { .compiling = true };
   thrd_t thread;
   mtx_lock(&producer.lock);
   CHECK(thrd_create(&thread, spurious_wakes, &test) == thrd_success);
   struct kk_pso_request request = kk_pso_request_create(20);
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &test.compiling,
                     &request) == VK_TIMEOUT);
   CHECK(test.compiling && test.wakes);
   test.stop = true;
   mtx_unlock(&producer.lock);
   CHECK(thrd_join(thread, NULL) == thrd_success);

   mtx_lock(&producer.lock);
   CHECK(thrd_create(&thread, complete_after_deadline, &test) == thrd_success);
   request = kk_pso_request_create(20);
   CHECK(kk_pso_wait(&producer.ready, &producer.lock, &test.compiling,
                     &request) == VK_SUCCESS);
   CHECK(!test.compiling);
   mtx_unlock(&producer.lock);
   CHECK(thrd_join(thread, NULL) == thrd_success);
}

static void
test_queued_job(void)
{
   struct kk_device dev = {0};
   struct kk_physical_device pdev = {0};
   dev.vk.physical = &pdev.vk;
   dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&dev.pso_cache));
   CHECK(mtx_init(&queue_gate.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&queue_gate.ready) == thrd_success);
   queue_gate.entered = queue_gate.release = false;
   struct util_queue_fence fence;
   util_queue_fence_init(&fence);
   util_queue_add_job(&dev.pso_cache.exact_queue, &queue_gate, &fence,
                      block_queue, NULL, 0);
   mtx_lock(&queue_gate.lock);
   while (!queue_gate.entered)
      cnd_wait(&queue_gate.ready, &queue_gate.lock);
   mtx_unlock(&queue_gate.lock);

   struct kk_shader vs = {0}, fs = {0};
   vs.info.partial = fs.info.partial = true;
   p_atomic_set(&vs.vk.pipeline.cache_obj.ref_cnt, 1);
   p_atomic_set(&fs.vk.pipeline.cache_obj.ref_cnt, 1);
   struct vk_dynamic_graphics_state dyn = {0};
   dyn.ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   dyn.ms.rasterization_samples = 1;
   dyn.ms.sample_mask = UINT16_MAX;
   struct kk_link_state ls = { .dyn = &dyn };
   ls.shaders[MESA_SHADER_VERTEX] = &vs;
   ls.shaders[MESA_SHADER_FRAGMENT] = &fs;
   kk_schedule_recipe(&dev, &ls, true, NULL);
   CHECK(dev.pso_cache.pending_jobs == 1 && dev.pso_cache.exact_jobs == 1);
   CHECK(p_atomic_read(&vs.vk.pipeline.cache_obj.ref_cnt) == 2);
   CHECK(p_atomic_read(&fs.vk.pipeline.cache_obj.ref_cnt) == 2);
   struct demand d = { .dev = &dev, .vs = &vs, .fs = &fs,
                       .origin = KK_COMPILE_ORIGIN_DRAW };
   kk_make_recipe(&ls, &vs, &fs, false, &d.recipe);
   producer.entered = producer.release = false;
   producer.fail = false;
   producer.calls = 0;
   thrd_t draw;
   CHECK(thrd_create(&draw, run_demand, &d) == thrd_success);
   wait_for_producer();

   struct kk_cmd_buffer cmd = {0};
   cmd.vk.base.device = &dev.vk;
   cmd.vk.dynamic_graphics_state = dyn;
   cmd.state.shaders[MESA_SHADER_VERTEX] = &vs;
   cmd.state.shaders[MESA_SHADER_FRAGMENT] = &fs;
   dev.pso_cache.wait_timeout_ms = 20;
   dev.pso_cache.timeout_policy = KK_WAIT_TIMEOUT_STRICT;
   bool provisional = true;
   CHECK(!kk_cmd_link_pipeline(&cmd, &provisional));
   CHECK(!provisional);
   CHECK(cmd.vk.record_result == VK_ERROR_UNKNOWN);
   CHECK(producer_calls() == 1);
   CHECK(dev.pso_cache.recipe_count == 1);

   release_producer();
   CHECK(thrd_join(draw, NULL) == thrd_success);
   mtx_lock(&queue_gate.lock);
   queue_gate.release = true;
   cnd_broadcast(&queue_gate.ready);
   mtx_unlock(&queue_gate.lock);
   CHECK(kk_pso_prewarm_wait(&dev) == VK_SUCCESS);
   CHECK(producer_calls() == 1);
   CHECK(!dev.pso_cache.pending_jobs && !dev.pso_cache.exact_jobs);
   CHECK(p_atomic_read(&vs.vk.pipeline.cache_obj.ref_cnt) == 1);
   CHECK(p_atomic_read(&fs.vk.pipeline.cache_obj.ref_cnt) == 1);
   kk_pso_cache_finish(&dev.pso_cache);
   util_queue_fence_destroy(&fence);
   cnd_destroy(&queue_gate.ready);
   mtx_destroy(&queue_gate.lock);
}

static void
test_lookup_only_miss(void)
{
   struct kk_device dev = {0};
   struct kk_physical_device pdev = {0};
   dev.vk.physical = &pdev.vk;
   dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&dev.pso_cache));
   reset_producer(0);
   struct kk_pso_key key;
   kk_pso_key_init(&key);
   struct kk_pso_request request = { .lookup_only = true };
   int64_t start = os_time_get_nano();
   struct kk_pso_result result =
      kk_pso_cache_get(&dev, &key, NULL, NULL, &request);
   CHECK(os_time_get_nano() - start < 100000000);
   CHECK(result.status == VK_NOT_READY && !result.pso);
   CHECK(producer_calls() == 0);
   CHECK(_mesa_hash_table_num_entries(dev.pso_cache.table) == 0);
   kk_pso_cache_finish(&dev.pso_cache);
}

struct link_fixture {
   struct kk_device dev;
   struct kk_physical_device pdev;
   struct kk_shader vs, fs;
   struct kk_cmd_buffer cmd;
};

static void
link_fixture_init(struct link_fixture *f, enum kk_wait_timeout_policy policy)
{
   memset(f, 0, sizeof(*f));
   f->dev.vk.physical = &f->pdev.vk;
   f->dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&f->dev.pso_cache));
   if (fail_stub_queue) {
      enum kk_wait_timeout_policy expected =
         failed_queue_policy == KK_WAIT_TIMEOUT_STUB ?
            KK_WAIT_TIMEOUT_STRICT : failed_queue_policy;
      CHECK(f->dev.pso_cache.timeout_policy == expected);
   }
   f->dev.pso_cache.wait_timeout_ms = 20;
   f->dev.pso_cache.timeout_policy = policy;
   f->vs.info.partial = f->fs.info.partial = true;
   f->vs.info.stage = MESA_SHADER_VERTEX;
   f->fs.info.stage = MESA_SHADER_FRAGMENT;
   f->vs.fast.hash[0] = 1;
   f->fs.fast.hash[0] = 2;
   p_atomic_set(&f->vs.vk.pipeline.cache_obj.ref_cnt, 1);
   p_atomic_set(&f->fs.vk.pipeline.cache_obj.ref_cnt, 1);
   f->cmd.vk.base.device = &f->dev.vk;
   struct vk_dynamic_graphics_state *dyn = &f->cmd.vk.dynamic_graphics_state;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   dyn->ms.rasterization_samples = 1;
   dyn->ms.sample_mask = UINT16_MAX;
   f->cmd.state.shaders[MESA_SHADER_VERTEX] = &f->vs;
   f->cmd.state.shaders[MESA_SHADER_FRAGMENT] = &f->fs;
}

static void
link_fixture_finish(struct link_fixture *f)
{
   VkResult expected = f->dev.pso_cache.limited ? VK_NOT_READY : VK_SUCCESS;
   CHECK(kk_pso_prewarm_wait(&f->dev) == expected);
   mtx_lock(&f->dev.pso_cache.lock);
   CHECK(!f->dev.pso_cache.pending_jobs);
   mtx_unlock(&f->dev.pso_cache.lock);
   CHECK(p_atomic_read(&f->vs.vk.pipeline.cache_obj.ref_cnt) == 1);
   CHECK(p_atomic_read(&f->fs.vk.pipeline.cache_obj.ref_cnt) == 1);
   kk_pso_cache_finish(&f->dev.pso_cache);
}

static void
fixture_recipe(struct link_fixture *f, struct kk_variant_recipe *recipe)
{
   struct kk_link_state ls = { .dyn = &f->cmd.vk.dynamic_graphics_state };
   ls.shaders[MESA_SHADER_VERTEX] = &f->vs;
   ls.shaders[MESA_SHADER_FRAGMENT] = &f->fs;
   kk_make_recipe(&ls, &f->vs, &f->fs, false, recipe);
}

static bool
stub_ready(struct link_fixture *f)
{
   struct kk_variant_recipe recipe;
   fixture_recipe(f, &recipe);
   return kk_stub_get(&f->dev, &recipe.pso, f->vs.fast.hash, NULL, false);
}

static void
test_stub_policy(void)
{
   struct link_fixture f;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STUB);
   p_atomic_set(&discard_fs_created, false);
   reset_producer(1);

   bool provisional = false;
   mtl_render_pipeline_state *pso = kk_cmd_link_pipeline(&f.cmd, &provisional);
   CHECK(provisional && !pso);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);

   int64_t give_up = os_time_get_nano() + 2000000000ll;
   while (!stub_ready(&f) && os_time_get_nano() < give_up)
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   CHECK(stub_ready(&f));

   f.cmd.state.gfx.pso_is_stub = true;
   int64_t start = os_time_get_nano();
   mtl_render_pipeline_state *stub = kk_cmd_link_pipeline(&f.cmd, &provisional);
   CHECK(os_time_get_nano() - start < 15000000);
   CHECK(provisional && stub);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);

   release_producer();
   give_up = os_time_get_nano() + 2000000000ll;
   do {
      f.cmd.state.gfx.pso_is_stub = provisional;
      pso = kk_cmd_link_pipeline(&f.cmd, &provisional);
      if (provisional)
         thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   } while (provisional && os_time_get_nano() < give_up);
   CHECK(!provisional && pso && pso != stub);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);
   CHECK(producer_calls() == 2);
   link_fixture_finish(&f);
}

static void
test_stub_saturation(void)
{
   struct link_fixture f;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STUB);
   util_queue_adjust_num_threads(&f.dev.pso_cache.draw_queue, 2, false);
   p_atomic_set(&discard_fs_created, false);
   reset_producer(2);

   struct kk_variant_recipe recipe;
   fixture_recipe(&f, &recipe);
   struct kk_pso_request request = kk_pso_request_create(20);
   request.handoff = true;
   struct kk_pso_result result =
      kk_recipe_get(&f.dev, &f.vs, &f.fs, &recipe, &request);
   CHECK(result.status == VK_TIMEOUT && !result.pso);

   struct kk_variant_recipe second = recipe;
   second.pso.sample_count = 2;
   request = kk_pso_request_create(20);
   request.handoff = true;
   result = kk_recipe_get(&f.dev, &f.vs, &f.fs, &second, &request);
   CHECK(result.status == VK_TIMEOUT && !result.pso);
   wait_for_calls(2);

   kk_schedule_stub(&f.dev, &f.vs, &recipe);
   int64_t give_up = os_time_get_nano() + 2000000000ll;
   while (!stub_ready(&f) && os_time_get_nano() < give_up)
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   CHECK(stub_ready(&f));
   CHECK(producer_calls() == 3);
   mtx_lock(&producer.lock);
   CHECK(!producer.release);
   mtx_unlock(&producer.lock);
   kk_schedule_stub(&f.dev, &f.vs, &recipe);

   release_producer();
   link_fixture_finish(&f);
   CHECK(producer_calls() == 3);
}

static void
test_stub_allocation_retry(void)
{
   struct link_fixture f;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STUB);
   p_atomic_set(&discard_fs_created, false);
   reset_producer(0);
   release_producer();
   struct kk_variant_recipe recipe;
   fixture_recipe(&f, &recipe);
   mtx_lock(&f.dev.pso_cache.lock);
   struct kk_recipe_entry *entry =
      kk_recipe_insert_locked(&f.dev.pso_cache, &recipe);
   CHECK(entry);
   mtx_unlock(&f.dev.pso_cache.lock);

   p_atomic_set(&fail_next_allocation, 1);
   kk_schedule_stub(&f.dev, &f.vs, &recipe);
   CHECK(!p_atomic_read(&fail_next_allocation));
   mtx_lock(&f.dev.pso_cache.lock);
   CHECK(!entry->stub_queued && !f.dev.pso_cache.pending_jobs);
   mtx_unlock(&f.dev.pso_cache.lock);
   CHECK(producer_calls() == 0);

   kk_schedule_stub(&f.dev, &f.vs, &recipe);
   CHECK(kk_pso_prewarm_wait(&f.dev) == VK_SUCCESS);
   CHECK(stub_ready(&f));
   kk_schedule_stub(&f.dev, &f.vs, &recipe);
   link_fixture_finish(&f);
   CHECK(producer_calls() == 1);
}

static void
test_strict_policy(bool queue_failure)
{
   struct link_fixture f;
   fail_stub_queue = queue_failure;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STRICT);
   fail_stub_queue = false;
   if (queue_failure) {
      CHECK(!util_queue_is_initialized(&f.dev.pso_cache.stub_queue));
      CHECK(f.dev.pso_cache.limited && !kk_stub_enabled(&f.dev));
   }
   reset_producer(1);
   bool provisional = true;
   CHECK(!kk_cmd_link_pipeline(&f.cmd, &provisional));
   CHECK(!provisional);
   CHECK(f.cmd.vk.record_result == VK_ERROR_UNKNOWN);
   release_producer();
   link_fixture_finish(&f);
   CHECK(producer_calls() == 1);
}

static int
release_late(void *data)
{
   wait_for_producer();
   thrd_sleep(&(struct timespec){ .tv_nsec = 50000000 }, NULL);
   release_producer();
   return 0;
}

static void
test_stub_teardown(void)
{
   struct link_fixture f;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STUB);
   p_atomic_set(&discard_fs_created, false);
   reset_producer(0);
   struct kk_variant_recipe recipe;
   fixture_recipe(&f, &recipe);
   kk_schedule_stub(&f.dev, &f.vs, &recipe);
   wait_for_producer();
   mtx_lock(&f.dev.pso_cache.lock);
   CHECK(f.dev.pso_cache.pending_jobs == 1);
   mtx_unlock(&f.dev.pso_cache.lock);
   CHECK(p_atomic_read(&f.vs.vk.pipeline.cache_obj.ref_cnt) == 2);

   thrd_t releaser;
   CHECK(thrd_create(&releaser, release_late, NULL) == thrd_success);
   kk_pso_cache_finish(&f.dev.pso_cache);
   CHECK(thrd_join(releaser, NULL) == thrd_success);
   CHECK(!f.dev.pso_cache.pending_jobs);
   CHECK(p_atomic_read(&f.vs.vk.pipeline.cache_obj.ref_cnt) == 1);
   CHECK(p_atomic_read(&f.fs.vk.pipeline.cache_obj.ref_cnt) == 1);
   CHECK(producer_calls() == 1);
}

static void
test_block_policy(void)
{
   struct link_fixture f;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_BLOCK);
   reset_producer(1);
   thrd_t releaser;
   CHECK(thrd_create(&releaser, release_late, NULL) == thrd_success);
   bool provisional = true;
   mtl_render_pipeline_state *pso = kk_cmd_link_pipeline(&f.cmd, &provisional);
   CHECK(thrd_join(releaser, NULL) == thrd_success);
   CHECK(pso && !provisional);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);
   CHECK(producer_calls() == 1);
   link_fixture_finish(&f);
}

static void
pending_render_init(struct link_fixture *f, struct kk_render_pending *pending)
{
   memset(pending, 0, sizeof(*pending));
   CHECK(cnd_init(&pending->ready) == thrd_success);
   pending->compiling = true;
   kk_pso_key_init(&pending->key);
   pending->key.rast[0] = 3;
   pending->key.fs[0] = 4;
   pending->key.topology_class = MTL_PRIMITIVE_TOPOLOGY_CLASS_TRIANGLE;
   f->vs.info.partial = false;
   f->vs.pending = pending;
}

static bool
pending_stub_ready(struct link_fixture *f)
{
   struct kk_render_pending *pending = f->vs.pending;
   return kk_stub_get(&f->dev, &pending->key, pending->key.rast, NULL, false);
}

static void
test_pending_render_block(void)
{
   struct link_fixture f;
   struct kk_render_pending pending;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_BLOCK);
   pending_render_init(&f, &pending);
   reset_producer(0);
   release_producer();
   bool provisional = true;
   mtl_render_pipeline_state *pso =
      kk_cmd_acquire_render(&f.cmd, &f.vs, &provisional);
   CHECK(pso && !provisional);
   CHECK(pending.queued && !pending.compiling);
   CHECK(f.vs.pipeline.gfx.render == pso);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);
   CHECK(producer_calls() == 1);
   link_fixture_finish(&f);
   cnd_destroy(&pending.ready);
}

static void
test_pending_render_stub(void)
{
   struct link_fixture f;
   struct kk_render_pending pending;
   link_fixture_init(&f, KK_WAIT_TIMEOUT_STUB);
   pending_render_init(&f, &pending);
   p_atomic_set(&discard_fs_created, false);
   reset_producer(0);
   release_producer();

   struct util_queue_fence fence;
   CHECK(mtx_init(&queue_gate.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&queue_gate.ready) == thrd_success);
   queue_gate.entered = queue_gate.release = false;
   util_queue_fence_init(&fence);
   util_queue_add_job(&f.dev.pso_cache.prewarm_queue, &queue_gate, &fence,
                      block_queue, NULL, 0);
   mtx_lock(&queue_gate.lock);
   while (!queue_gate.entered)
      cnd_wait(&queue_gate.ready, &queue_gate.lock);
   mtx_unlock(&queue_gate.lock);

   bool provisional = false;
   kk_cmd_acquire_render(&f.cmd, &f.vs, &provisional);
   CHECK(provisional && pending.queued);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);

   int64_t give_up = os_time_get_nano() + 2000000000ll;
   while (!pending_stub_ready(&f) && os_time_get_nano() < give_up)
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   CHECK(pending_stub_ready(&f));
   CHECK(producer_calls() == 1);

   f.cmd.state.gfx.pso_is_stub = true;
   mtl_render_pipeline_state *stub =
      kk_cmd_acquire_render(&f.cmd, &f.vs, &provisional);
   CHECK(provisional && stub);

   mtx_lock(&queue_gate.lock);
   queue_gate.release = true;
   cnd_broadcast(&queue_gate.ready);
   mtx_unlock(&queue_gate.lock);
   util_queue_fence_wait(&fence);
   util_queue_fence_destroy(&fence);

   mtl_render_pipeline_state *pso;
   give_up = os_time_get_nano() + 2000000000ll;
   do {
      f.cmd.state.gfx.pso_is_stub = provisional;
      pso = kk_cmd_acquire_render(&f.cmd, &f.vs, &provisional);
      if (provisional)
         thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   } while (provisional && os_time_get_nano() < give_up);
   CHECK(!provisional && pso && pso != stub);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);
   CHECK(producer_calls() == 2);
   link_fixture_finish(&f);
   cnd_destroy(&pending.ready);
   cnd_destroy(&queue_gate.ready);
   mtx_destroy(&queue_gate.lock);
}

static void
test_metal_cache_size(void)
{
   CHECK(kk_metal_cache_parse_size(NULL) == 512u << 20);
   CHECK(kk_metal_cache_parse_size("") == 512u << 20);
   CHECK(kk_metal_cache_parse_size("junk") == 512u << 20);
   CHECK(kk_metal_cache_parse_size("0") == 512u << 20);
   CHECK(kk_metal_cache_parse_size("8M") == 8u << 20);
   CHECK(kk_metal_cache_parse_size("8m") == 8u << 20);
   CHECK(kk_metal_cache_parse_size("64K") == 64u << 10);
   CHECK(kk_metal_cache_parse_size("2G") == 2u << 30);
   CHECK(kk_metal_cache_parse_size("1") == 1u << 30);
   CHECK(kk_metal_cache_parse_size("4G") == UINT32_MAX);
   CHECK(kk_metal_cache_parse_size("99999999999999M") == UINT32_MAX);
}

static void
make_blend_recipe(struct vk_dynamic_graphics_state *dyn, struct kk_shader *vs,
                  struct kk_shader *fs, struct kk_variant_recipe *recipe)
{
   struct kk_link_state ls = { .dyn = dyn, .color_att_count = 1 };
   ls.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
   ls.shaders[MESA_SHADER_VERTEX] = vs;
   ls.shaders[MESA_SHADER_FRAGMENT] = fs;
   kk_make_recipe(&ls, vs, fs, false, recipe);
}

static void
init_blend_state(struct vk_dynamic_graphics_state *dyn)
{
   memset(dyn, 0, sizeof(*dyn));
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   dyn->ms.rasterization_samples = 1;
   dyn->ms.sample_mask = UINT16_MAX;
   dyn->cb.color_write_enables = 1;
   dyn->cal.color_map[0] = 0;
   for (unsigned i = 1; i < MESA_VK_MAX_COLOR_ATTACHMENTS; ++i)
      dyn->cal.color_map[i] = MESA_VK_ATTACHMENT_UNUSED;
   struct vk_color_blend_attachment_state *att = &dyn->cb.attachments[0];
   att->write_mask = 0xf;
   att->src_color_blend_factor = VK_BLEND_FACTOR_ONE;
   att->dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
   att->src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
   att->dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
}

static void
test_normalized_recipes(void)
{
   struct kk_shader vs = {0}, fs = {0};
   vs.fast.hash[0] = 1;
   fs.fast.hash[0] = 2;
   fs.info.fs.color_outputs = 1;
   struct vk_dynamic_graphics_state a, b;
   struct kk_variant_recipe ra, rb;

   init_blend_state(&a);
   init_blend_state(&b);
   b.cb.attachments[0].src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA;
   b.cb.attachments[0].color_blend_op = VK_BLEND_OP_SUBTRACT;
   make_blend_recipe(&a, &vs, &fs, &ra);
   make_blend_recipe(&b, &vs, &fs, &rb);
   CHECK(!memcmp(&ra, &rb, sizeof(ra)));

   a.cb.attachments[0].blend_enable = b.cb.attachments[0].blend_enable = true;
   make_blend_recipe(&a, &vs, &fs, &ra);
   make_blend_recipe(&b, &vs, &fs, &rb);
   CHECK(memcmp(&ra, &rb, sizeof(ra)));

   a.cb.attachments[0].write_mask = 0;
   b.cb.color_write_enables = 0;
   make_blend_recipe(&a, &vs, &fs, &ra);
   make_blend_recipe(&b, &vs, &fs, &rb);
   CHECK(!memcmp(&ra, &rb, sizeof(ra)));
   CHECK(!ra.pso.blend[0].enable && !ra.pso.blend[0].write_mask);

   init_blend_state(&a);
   init_blend_state(&b);
   b.cb.logic_op = VK_LOGIC_OP_XOR;
   b.ms.sample_mask = 1;
   make_blend_recipe(&a, &vs, &fs, &ra);
   make_blend_recipe(&b, &vs, &fs, &rb);
   CHECK(!memcmp(&ra, &rb, sizeof(ra)));

   init_blend_state(&a);
   init_blend_state(&b);
   a.cb.logic_op_enable = b.cb.logic_op_enable = true;
   a.cb.logic_op = VK_LOGIC_OP_COPY;
   b.cb.logic_op = VK_LOGIC_OP_XOR;
   make_blend_recipe(&a, &vs, &fs, &ra);
   make_blend_recipe(&b, &vs, &fs, &rb);
   CHECK(memcmp(&ra, &rb, sizeof(ra)));
}

int
main(void)
{
   CHECK(mtx_init(&producer.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&producer.ready) == thrd_success);
   test_wait_budget();
   test_wait_wakeups();
   test_recipe(false);
   test_recipe(true);
   test_queued_job();
   test_lookup_only_miss();
   test_normalized_recipes();
   test_strict_policy(false);
   test_strict_policy(true);
   test_block_policy();
   test_stub_policy();
   test_stub_saturation();
   test_stub_allocation_retry();
   test_stub_teardown();
   test_pending_render_block();
   test_pending_render_stub();
   test_metal_cache_size();
   cnd_destroy(&producer.ready);
   mtx_destroy(&producer.lock);
   return 0;
}
