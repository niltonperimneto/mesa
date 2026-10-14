#define mtl_new_render_pipeline test_render_pipeline
#define mtl_specialize_render_pipeline test_specialize_pipeline
#define mtl_release test_release
#define mtl_retain test_retain
#define __vk_errorf test_vk_errorf
#include "../kk_pso_cache.c"
#undef mtl_retain
#undef mtl_new_render_pipeline
#undef mtl_specialize_render_pipeline
#undef mtl_release
#undef __vk_errorf

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

static struct {
   mtx_t lock;
   cnd_t ready;
   bool entered;
   bool block;
   bool fail;
   unsigned calls;
} specializer;
static int specialized[16];

mtl_render_pipeline_state *
test_specialize_pipeline(mtl_compiler *compiler,
                         mtl_render_pipeline_descriptor *desc,
                         mtl_render_pipeline_state *base)
{
   mtx_lock(&specializer.lock);
   unsigned call = specializer.calls++;
   specializer.entered = true;
   cnd_broadcast(&specializer.ready);
   while (specializer.block)
      cnd_wait(&specializer.ready, &specializer.lock);
   bool fail = specializer.fail;
   mtx_unlock(&specializer.lock);
   if (fail || (int *)base < pipelines || (int *)base >= pipelines + 16)
      return NULL;
   return (mtl_render_pipeline_state *)&specialized[MIN2(call, 15)];
}

void *mtl_retain(void *object);

void *
test_retain(void *object)
{
   if (((int *)object < pipelines || (int *)object >= pipelines + 16) &&
       ((int *)object < specialized || (int *)object >= specialized + 16))
      return mtl_retain(object);
   return object;
}

void
test_release(void *object)
{
   if (((int *)object < pipelines || (int *)object >= pipelines + 16) &&
       ((int *)object < specialized || (int *)object >= specialized + 16))
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
   u_rwlock_wrlock(&dev.pso_cache.index_lock);
   struct kk_recipe_entry *entry =
      kk_recipe_insert_locked(&dev.pso_cache, kk_recipe_hash(&d.recipe),
                              &d.recipe);
   u_rwlock_wrunlock(&dev.pso_cache.index_lock);
   CHECK(entry);
   mtx_lock(&dev.pso_cache.lock);
   entry->state = KK_ENTRY_QUEUED;
   mtx_unlock(&dev.pso_cache.lock);
   producer.entered = producer.release = false;
   producer.fail = fail;
   producer.calls = 0;

   thrd_t owner, late_worker, second_draw;
   CHECK(thrd_create(&owner, run_demand, &d) == thrd_success);
   wait_for_producer();
   mtx_lock(&dev.pso_cache.lock);
   CHECK(entry->state == KK_ENTRY_COMPILING);
   mtx_unlock(&dev.pso_cache.lock);

   struct demand late = d;
   late.origin = KK_COMPILE_ORIGIN_PREWARM;
   struct demand second = d;
   CHECK(thrd_create(&late_worker, run_demand, &late) == thrd_success);
   CHECK(thrd_create(&second_draw, run_demand, &second) == thrd_success);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   mtx_lock(&dev.pso_cache.lock);
   CHECK(entry->state == KK_ENTRY_COMPILING && !entry->drawn &&
         entry->demanded && dev.pso_cache.draw_demand == 2);
   mtx_unlock(&dev.pso_cache.lock);

   release_producer();
   CHECK(thrd_join(owner, NULL) == thrd_success);
   CHECK(thrd_join(late_worker, NULL) == thrd_success);
   CHECK(thrd_join(second_draw, NULL) == thrd_success);
   CHECK(producer_calls() == 1);
   CHECK(!dev.pso_cache.draw_demand);
   CHECK(d.result.status == (fail ? VK_ERROR_UNKNOWN : VK_SUCCESS));
   CHECK(d.result.pso == late.result.pso && d.result.pso == second.result.pso);
   CHECK(late.result.status == d.result.status &&
         second.result.status == d.result.status);

   unsigned calls = 1;
   if (fail) {
      CHECK(entry->state == KK_ENTRY_FAILED && entry->failures == 1 &&
            entry->result == VK_ERROR_UNKNOWN && !entry->pso);
      kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
      struct kk_pso_result skipped =
         kk_recipe_get(&dev, &vs, &fs, &d.recipe, NULL);
      CHECK(skipped.status == VK_ERROR_UNKNOWN && !skipped.pso);
      CHECK(producer_calls() == 1);
      mtx_lock(&producer.lock);
      producer.fail = false;
      mtx_unlock(&producer.lock);
      kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
      d.result = kk_recipe_get(&dev, &vs, &fs, &d.recipe, NULL);
      CHECK(d.result.status == VK_SUCCESS && d.result.pso);
      CHECK(entry->state == KK_ENTRY_READY && entry->failures == 1);
      calls = 2;
   }
   CHECK(producer_calls() == calls);

   struct kk_variant_recipe other = d.recipe;
   other.family[0] = 1;
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   struct kk_pso_result reused =
      kk_recipe_get(&dev, &vs, &fs, &other, NULL);
   CHECK(reused.status == VK_SUCCESS && reused.pso == d.result.pso);
   struct kk_recipe_entry *retry =
      kk_recipe_lookup(&dev.pso_cache, &other, false);
   CHECK(retry && retry->state == KK_ENTRY_READY);
   CHECK(producer_calls() == calls);
   CHECK(retry->drawn);
   CHECK(!dev.pso_cache.pending_jobs && !dev.pso_cache.exact_jobs);
   kk_pso_cache_finish(&dev.pso_cache);
}

struct wait_test {
   struct kk_pso_cache *cache;
   struct kk_pso_entry *entry;
   unsigned wakes;
};

static int
finish_after_wakes(void *data)
{
   struct wait_test *test = data;
   for (unsigned i = 0; i < 10; ++i) {
      mtx_lock(&test->cache->lock);
      test->wakes++;
      cnd_broadcast(&test->entry->ready);
      mtx_unlock(&test->cache->lock);
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   }
   mtx_lock(&test->cache->lock);
   p_atomic_set(&test->entry->pso, (void *)&pipelines[0]);
   test->entry->state = KK_ENTRY_READY;
   cnd_broadcast(&test->entry->ready);
   mtx_unlock(&test->cache->lock);
   return 0;
}

static void
test_wait_wakeups(void)
{
   struct kk_device dev = {0};
   struct kk_pso_entry entry = { .state = KK_ENTRY_COMPILING };
   struct wait_test test = { .cache = &dev.pso_cache, .entry = &entry };
   CHECK(mtx_init(&dev.pso_cache.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&entry.ready) == thrd_success);
   thrd_t thread;
   CHECK(thrd_create(&thread, finish_after_wakes, &test) == thrd_success);
   bool compile = false;
   struct kk_pso_result result =
      kk_pso_entry_get(&dev, &entry, &entry.key, true, &compile);
   CHECK(result.status == VK_SUCCESS && result.pso == (void *)&pipelines[0]);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(!compile && entry.state == KK_ENTRY_READY && test.wakes == 10);
   cnd_destroy(&entry.ready);
   mtx_destroy(&dev.pso_cache.lock);
}

struct link_demand {
   struct kk_cmd_buffer *cmd;
   mtl_render_pipeline_state *pso;
};

static int
run_link(void *data)
{
   struct link_demand *d = data;
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   d->pso = kk_cmd_link_pipeline(d->cmd);
   return 0;
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
   struct link_demand linker = { .cmd = &cmd };
   thrd_t link;
   CHECK(thrd_create(&link, run_link, &linker) == thrd_success);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   CHECK(producer_calls() == 1);
   CHECK(dev.pso_cache.recipe_count == 1);

   release_producer();
   CHECK(thrd_join(draw, NULL) == thrd_success);
   CHECK(thrd_join(link, NULL) == thrd_success);
   CHECK(linker.pso && linker.pso == d.result.pso);
   CHECK(cmd.vk.record_result == VK_SUCCESS);
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

struct child_demand {
   struct kk_device *dev;
   struct kk_pso_key key;
   struct kk_pso_result result;
};

static int
run_child(void *data)
{
   struct child_demand *d = data;
   d->result = kk_pso_cache_get(d->dev, &d->key, NULL, NULL, NULL);
   return 0;
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

   reset_producer(0);
   struct child_demand owner = { .dev = &dev, .key = key };
   thrd_t thread;
   CHECK(thrd_create(&thread, run_child, &owner) == thrd_success);
   wait_for_producer();
   start = os_time_get_nano();
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, &request);
   CHECK(os_time_get_nano() - start < 10000000);
   CHECK(result.status == VK_NOT_READY && !result.pso);
   CHECK(!kk_pso_cache_acquire(&dev, &key, NULL, NULL, true));
   CHECK(producer_calls() == 1);
   release_producer();
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(owner.result.status == VK_SUCCESS && owner.result.pso);
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, &request);
   CHECK(result.status == VK_SUCCESS && result.pso == owner.result.pso);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_lookup_only_recipe(void)
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
   d.recipe.family[0] = 7;
   const struct kk_pso_request lookup = { .lookup_only = true };
   kk_compile_origin = KK_COMPILE_ORIGIN_CREATE;
   struct kk_pso_result result =
      kk_recipe_get(&dev, &vs, &fs, &d.recipe, &lookup);
   CHECK(result.status == VK_NOT_READY && !result.pso);
   CHECK(!kk_recipe_lookup(&dev.pso_cache, &d.recipe, false));

   reset_producer(0);
   thrd_t owner;
   CHECK(thrd_create(&owner, run_demand, &d) == thrd_success);
   wait_for_producer();
   struct kk_recipe_entry *entry =
      kk_recipe_lookup(&dev.pso_cache, &d.recipe, false);
   CHECK(entry);
   int64_t start = os_time_get_nano();
   result = kk_recipe_get(&dev, &vs, &fs, &d.recipe, &lookup);
   CHECK(os_time_get_nano() - start < 10000000);
   CHECK(result.status == VK_NOT_READY && !result.pso);
   mtx_lock(&dev.pso_cache.lock);
   CHECK(entry->state == KK_ENTRY_COMPILING &&
         dev.pso_cache.draw_demand == 1);
   mtx_unlock(&dev.pso_cache.lock);
   release_producer();
   CHECK(thrd_join(owner, NULL) == thrd_success);
   CHECK(d.result.status == VK_SUCCESS && d.result.pso);
   result = kk_recipe_get(&dev, &vs, &fs, &d.recipe, &lookup);
   CHECK(result.status == VK_SUCCESS && result.pso == d.result.pso);
   CHECK(producer_calls() == 1);

   struct kk_variant_recipe queued = d.recipe;
   queued.family[0] = 8;
   u_rwlock_wrlock(&dev.pso_cache.index_lock);
   entry = kk_recipe_insert_locked(&dev.pso_cache, kk_recipe_hash(&queued),
                                   &queued);
   u_rwlock_wrunlock(&dev.pso_cache.index_lock);
   entry->state = KK_ENTRY_QUEUED;
   result = kk_recipe_get(&dev, &vs, &fs, &queued, &lookup);
   CHECK(result.status == VK_NOT_READY && entry->state == KK_ENTRY_QUEUED);
   CHECK(producer_calls() == 1);
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   kk_pso_cache_finish(&dev.pso_cache);
}

struct link_fixture {
   struct kk_device dev;
   struct kk_physical_device pdev;
   struct kk_shader vs, fs;
   struct kk_cmd_buffer cmd;
};

static void
link_fixture_init(struct link_fixture *f)
{
   memset(f, 0, sizeof(*f));
   f->dev.vk.physical = &f->pdev.vk;
   f->dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&f->dev.pso_cache));
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

static int
release_late(void *data)
{
   wait_for_producer();
   thrd_sleep(&(struct timespec){ .tv_nsec = 50000000 }, NULL);
   release_producer();
   return 0;
}

static void
test_link_waits_for_completion(void)
{
   struct link_fixture f;
   link_fixture_init(&f);
   reset_producer(1);
   thrd_t releaser;
   CHECK(thrd_create(&releaser, release_late, NULL) == thrd_success);
   mtl_render_pipeline_state *pso = kk_cmd_link_pipeline(&f.cmd);
   CHECK(thrd_join(releaser, NULL) == thrd_success);
   CHECK(pso);
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

static void
test_pending_render_waits_for_completion(void)
{
   struct link_fixture f;
   struct kk_render_pending pending;
   link_fixture_init(&f);
   pending_render_init(&f, &pending);
   reset_producer(0);
   release_producer();
   mtl_render_pipeline_state *pso = kk_cmd_acquire_render(&f.cmd, &f.vs);
   CHECK(pso);
   CHECK(pending.queued && !pending.compiling);
   CHECK(f.vs.pipeline.gfx.render == pso);
   CHECK(f.cmd.vk.record_result == VK_SUCCESS);
   CHECK(producer_calls() == 1);
   link_fixture_finish(&f);
   cnd_destroy(&pending.ready);
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

struct concurrent_hits {
   struct kk_device *dev;
   struct kk_pso_key key;
   mtl_render_pipeline_state *pso;
   unsigned inserts;
};

static int
run_hits(void *data)
{
   struct concurrent_hits *c = data;
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   for (unsigned i = 0; i < 20000; ++i) {
      struct kk_pso_result result =
         kk_pso_cache_get(c->dev, &c->key, NULL, NULL, NULL);
      CHECK(result.status == VK_SUCCESS && result.pso == c->pso);
   }
   return 0;
}

static int
run_inserts(void *data)
{
   struct concurrent_hits *c = data;
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   for (unsigned i = 0; i < c->inserts; ++i) {
      struct kk_pso_key key;
      kk_pso_key_init(&key);
      key.rast[0] = 1;
      key.rast[1] = i & 0xff;
      key.rast[2] = i >> 8;
      CHECK(kk_pso_cache_get(c->dev, &key, NULL, NULL, NULL).pso);
   }
   return 0;
}

static void
test_concurrent_hits(void)
{
   struct kk_device dev = {0};
   struct kk_physical_device pdev = {0};
   dev.vk.physical = &pdev.vk;
   dev.vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&dev.pso_cache));
   reset_producer(0);
   release_producer();
   struct concurrent_hits c = { .dev = &dev, .inserts = 300 };
   kk_pso_key_init(&c.key);
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   c.pso = kk_pso_cache_get(&dev, &c.key, NULL, NULL, NULL).pso;
   CHECK(c.pso);

   thrd_t readers[4], writer;
   for (unsigned i = 0; i < ARRAY_SIZE(readers); ++i)
      CHECK(thrd_create(&readers[i], run_hits, &c) == thrd_success);
   CHECK(thrd_create(&writer, run_inserts, &c) == thrd_success);
   for (unsigned i = 0; i < ARRAY_SIZE(readers); ++i)
      CHECK(thrd_join(readers[i], NULL) == thrd_success);
   CHECK(thrd_join(writer, NULL) == thrd_success);

   CHECK(producer_calls() == c.inserts + 1);
   CHECK(_mesa_hash_table_num_entries(dev.pso_cache.table) == c.inserts + 1);
   struct hash_entry *he = _mesa_hash_table_search(dev.pso_cache.table, &c.key);
   struct kk_pso_entry *entry = he->data;
   CHECK(entry->drawn && entry->prewarmed);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
reset_specializer(bool block, bool fail)
{
   mtx_lock(&specializer.lock);
   specializer.entered = false;
   specializer.block = block;
   specializer.fail = fail;
   specializer.calls = 0;
   mtx_unlock(&specializer.lock);
}

static unsigned
specializer_calls(void)
{
   mtx_lock(&specializer.lock);
   unsigned calls = specializer.calls;
   mtx_unlock(&specializer.lock);
   return calls;
}

static void
color_key(struct kk_pso_key *key, uint32_t format, bool blend, uint8_t mask)
{
   kk_pso_key_init(key);
   key->rast[0] = 5;
   key->fs[0] = 6;
   key->sample_count = 1;
   key->topology_class = MTL_PRIMITIVE_TOPOLOGY_CLASS_TRIANGLE;
   key->rt_formats[0] = format;
   key->blend[0].write_mask = mask;
   if (blend) {
      key->blend[0].enable = true;
      key->blend[0].src_rgb = MTL_BLEND_FACTOR_SOURCE_ALPHA;
      key->blend[0].dst_rgb = MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA;
   }
}

static struct kk_pso_entry *
table_entry(struct kk_device *dev, const struct kk_pso_key *key)
{
   struct hash_entry *he = _mesa_hash_table_search(dev->pso_cache.table, key);
   return he ? he->data : NULL;
}

static void
specialize_fixture(struct kk_device *dev, struct kk_physical_device *pdev)
{
   memset(dev, 0, sizeof(*dev));
   memset(pdev, 0, sizeof(*pdev));
   dev->vk.physical = &pdev->vk;
   dev->vk.disable_internal_cache = true;
   CHECK(kk_pso_cache_init(&dev->pso_cache));
}

static void
test_shared_base(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   reset_producer(0);
   release_producer();
   reset_specializer(false, false);
   struct kk_pso_key keys[4];
   color_key(&keys[0], MTL_PIXEL_FORMAT_R8G8B8A8_UNORM, false, 0xf);
   color_key(&keys[1], MTL_PIXEL_FORMAT_B8G8R8A8_UNORM, false, 0xf);
   color_key(&keys[2], MTL_PIXEL_FORMAT_R16G16B16A16_FLOAT, true, 0xf);
   color_key(&keys[3], MTL_PIXEL_FORMAT_R8G8B8A8_UNORM, true, 0x7);
   struct kk_pso_result results[4];
   for (unsigned i = 0; i < 4; ++i) {
      results[i] = kk_pso_cache_get(&dev, &keys[i], NULL, NULL, NULL);
      CHECK(results[i].status == VK_SUCCESS && results[i].pso);
      for (unsigned j = 0; j < i; ++j)
         CHECK(results[i].pso != results[j].pso);
   }
   CHECK(producer_calls() == 1);
   CHECK(specializer_calls() == 4);
   struct kk_pso_key base;
   kk_pso_base_key(&keys[2], &base);
   struct kk_pso_entry *entry = table_entry(&dev, &base);
   CHECK(entry && entry->pso == (void *)&pipelines[0]);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      CHECK(base.rt_formats[i] == MTL_PIXEL_FORMAT_UNSPECIALIZED &&
            !base.blend[i].enable && !base.blend[i].write_mask);
   CHECK(_mesa_hash_table_num_entries(dev.pso_cache.table) == 5);
   struct kk_pso_result again =
      kk_pso_cache_get(&dev, &keys[1], NULL, NULL, NULL);
   CHECK(again.pso == results[1].pso);
   CHECK(producer_calls() == 1 && specializer_calls() == 4);

   struct kk_pso_key msaa = keys[0];
   msaa.sample_count = 4;
   CHECK(kk_pso_cache_get(&dev, &msaa, NULL, NULL, NULL).pso);
   struct kk_pso_key a2c = keys[0];
   a2c.alpha_to_coverage = 1;
   CHECK(kk_pso_cache_get(&dev, &a2c, NULL, NULL, NULL).pso);
   CHECK(producer_calls() == 3 && specializer_calls() == 6);

   struct kk_pso_key depth_only;
   kk_pso_key_init(&depth_only);
   depth_only.rast[0] = 5;
   CHECK(!kk_pso_key_specializable(&depth_only));
   CHECK(kk_pso_cache_get(&dev, &depth_only, NULL, NULL, NULL).pso);
   CHECK(producer_calls() == 4 && specializer_calls() == 6);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_concurrent_specialize(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   reset_producer(0);
   release_producer();
   reset_specializer(true, false);
   struct child_demand d[4];
   thrd_t threads[4];
   for (unsigned i = 0; i < 4; ++i) {
      d[i] = (struct child_demand){ .dev = &dev };
      color_key(&d[i].key, i < 3 ? MTL_PIXEL_FORMAT_R8G8B8A8_UNORM
                                 : MTL_PIXEL_FORMAT_B8G8R8A8_UNORM, false, 0xf);
      CHECK(thrd_create(&threads[i], run_child, &d[i]) == thrd_success);
   }
   int64_t give_up = os_time_get_nano() + 2000000000ll;
   while (specializer_calls() < 2 && os_time_get_nano() < give_up)
      thrd_sleep(&(struct timespec){ .tv_nsec = 1000000 }, NULL);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   CHECK(specializer_calls() == 2);
   mtx_lock(&specializer.lock);
   specializer.block = false;
   cnd_broadcast(&specializer.ready);
   mtx_unlock(&specializer.lock);
   for (unsigned i = 0; i < 4; ++i) {
      CHECK(thrd_join(threads[i], NULL) == thrd_success);
      CHECK(d[i].result.status == VK_SUCCESS && d[i].result.pso);
   }
   CHECK(d[0].result.pso == d[1].result.pso &&
         d[1].result.pso == d[2].result.pso);
   CHECK(d[3].result.pso != d[0].result.pso);
   CHECK(producer_calls() == 1 && specializer_calls() == 2);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_specialize_fallback(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   reset_producer(0);
   release_producer();
   reset_specializer(false, true);
   struct kk_pso_key key;
   color_key(&key, MTL_PIXEL_FORMAT_R8G8B8A8_UNORM, false, 0xf);
   struct kk_pso_result result = kk_pso_cache_get(&dev, &key, NULL, NULL, NULL);
   CHECK(result.status == VK_SUCCESS && result.pso == (void *)&pipelines[1]);
   CHECK(producer_calls() == 2 && specializer_calls() == 1);

   struct kk_pso_key base;
   kk_pso_base_key(&key, &base);
   mtx_lock(&producer.lock);
   producer.fail = true;
   mtx_unlock(&producer.lock);
   reset_specializer(false, false);
   struct kk_pso_key other;
   color_key(&other, MTL_PIXEL_FORMAT_B8G8R8A8_UNORM, false, 0xf);
   other.fs[0] = 7;
   result = kk_pso_cache_get(&dev, &other, NULL, NULL, NULL);
   CHECK(result.status == VK_ERROR_UNKNOWN && !result.pso);
   CHECK(producer_calls() == 4 && specializer_calls() == 0);
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_specialize_lookup_only(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   reset_producer(0);
   release_producer();
   reset_specializer(false, false);
   struct kk_pso_key key;
   color_key(&key, MTL_PIXEL_FORMAT_R8G8B8A8_UNORM, false, 0xf);
   struct kk_pso_request lookup = { .lookup_only = true };
   struct kk_pso_result result =
      kk_pso_cache_get(&dev, &key, NULL, NULL, &lookup);
   CHECK(result.status == VK_NOT_READY && !result.pso);
   CHECK(producer_calls() == 0 && specializer_calls() == 0);
   CHECK(_mesa_hash_table_num_entries(dev.pso_cache.table) == 0);

   struct kk_pso_key warm;
   color_key(&warm, MTL_PIXEL_FORMAT_B8G8R8A8_UNORM, true, 0xf);
   CHECK(kk_pso_cache_get(&dev, &warm, NULL, NULL, NULL).pso);
   CHECK(producer_calls() == 1 && specializer_calls() == 1);
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, &lookup);
   CHECK(result.status == VK_SUCCESS && result.pso);
   CHECK(producer_calls() == 1 && specializer_calls() == 2);
   CHECK(table_entry(&dev, &key));
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
set_producer_fail(bool fail)
{
   mtx_lock(&producer.lock);
   producer.fail = fail;
   mtx_unlock(&producer.lock);
}

static void
test_native_retry(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   reset_producer(0);
   release_producer();
   set_producer_fail(true);
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   const struct kk_pso_request lookup = { .lookup_only = true };

   struct kk_pso_key key;
   kk_pso_key_init(&key);
   key.rast[0] = 9;
   struct kk_pso_result result = kk_pso_cache_get(&dev, &key, NULL, NULL, NULL);
   CHECK(result.status == VK_ERROR_UNKNOWN && !result.pso);
   struct kk_pso_entry *entry = table_entry(&dev, &key);
   CHECK(entry && entry->state == KK_ENTRY_FAILED && entry->failures == 1);
   CHECK(producer_calls() == 1);

   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, NULL);
   CHECK(result.status == VK_ERROR_UNKNOWN && producer_calls() == 1);
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, &lookup);
   CHECK(result.status == VK_NOT_READY && producer_calls() == 1);

   set_producer_fail(false);
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, NULL);
   CHECK(result.status == VK_SUCCESS && result.pso == (void *)&pipelines[1]);
   CHECK(entry->state == KK_ENTRY_READY && producer_calls() == 2);
   result = kk_pso_cache_get(&dev, &key, NULL, NULL, &lookup);
   CHECK(result.status == VK_SUCCESS && result.pso == (void *)&pipelines[1]);

   struct kk_pso_key permanent = key;
   permanent.rast[0] = 10;
   set_producer_fail(true);
   kk_compile_origin = KK_COMPILE_ORIGIN_CREATE;
   for (unsigned i = 0; i < 5; ++i) {
      result = kk_pso_cache_get(&dev, &permanent, NULL, NULL, NULL);
      CHECK(result.status == VK_ERROR_UNKNOWN && !result.pso);
   }
   CHECK(producer_calls() == 2 + KK_PSO_MAX_ATTEMPTS);
   entry = table_entry(&dev, &permanent);
   CHECK(entry->state == KK_ENTRY_FAILED &&
         entry->failures == KK_PSO_MAX_ATTEMPTS);
   result = kk_pso_cache_get(&dev, &permanent, NULL, NULL, &lookup);
   CHECK(result.status == VK_ERROR_UNKNOWN);
   CHECK(producer_calls() == 2 + KK_PSO_MAX_ATTEMPTS);
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
run_failing_round(struct kk_device *dev, const struct kk_pso_key *key)
{
   reset_producer(0);
   set_producer_fail(true);
   struct child_demand d[4];
   thrd_t threads[4];
   d[0] = (struct child_demand){ .dev = dev, .key = *key };
   CHECK(thrd_create(&threads[0], run_child, &d[0]) == thrd_success);
   wait_for_producer();
   for (unsigned i = 1; i < 4; ++i) {
      d[i] = (struct child_demand){ .dev = dev, .key = *key };
      CHECK(thrd_create(&threads[i], run_child, &d[i]) == thrd_success);
   }
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   struct kk_pso_entry *entry = table_entry(dev, key);
   mtx_lock(&dev->pso_cache.lock);
   CHECK(entry->state == KK_ENTRY_COMPILING);
   mtx_unlock(&dev->pso_cache.lock);
   release_producer();
   for (unsigned i = 0; i < 4; ++i) {
      CHECK(thrd_join(threads[i], NULL) == thrd_success);
      CHECK(d[i].result.status == VK_ERROR_UNKNOWN && !d[i].result.pso);
   }
   CHECK(producer_calls() == 1);
}

static void
test_native_waiters(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   struct kk_pso_key key;
   kk_pso_key_init(&key);
   key.rast[0] = 11;
   run_failing_round(&dev, &key);
   CHECK(table_entry(&dev, &key)->failures == 1);
   run_failing_round(&dev, &key);
   CHECK(table_entry(&dev, &key)->failures == 2);
   reset_producer(0);
   release_producer();
   kk_compile_origin = KK_COMPILE_ORIGIN_DRAW;
   CHECK(kk_pso_cache_get(&dev, &key, NULL, NULL, NULL).status ==
         VK_ERROR_UNKNOWN);
   CHECK(producer_calls() == 0);
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   kk_pso_cache_finish(&dev.pso_cache);
}

static void
test_normalized_fs_keys(void)
{
   struct kk_shader fs = {0};
   fs.info.fs.color_outputs = BITFIELD_BIT(0) | BITFIELD_BIT(1);
   const VkFormat floats[] = {
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8_UNORM,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_B8G8R8A8_SRGB,
   };
   struct kk_fs_link_key reference;
   for (unsigned i = 0; i < ARRAY_SIZE(floats); ++i) {
      struct kk_fs_link_key key = {0};
      key.missing_inputs = 1;
      key.rt_formats[0] = floats[i];
      key.rt_formats[1] = VK_FORMAT_R32_UINT;
      key.rt_formats[2] = floats[ARRAY_SIZE(floats) - 1 - i];
      kk_normalize_fs_link_key(&fs, &key);
      CHECK(key.rt_formats[0] == VK_FORMAT_R32G32B32A32_SFLOAT);
      CHECK(key.rt_formats[1] == VK_FORMAT_R32G32B32A32_UINT);
      CHECK(key.rt_formats[2] == VK_FORMAT_UNDEFINED);
      if (i)
         CHECK(!memcmp(&key, &reference, sizeof(key)));
      reference = key;
   }
   struct kk_fs_link_key sint = {0};
   sint.rt_formats[0] = VK_FORMAT_R8G8B8A8_SINT;
   kk_normalize_fs_link_key(&fs, &sint);
   CHECK(sint.rt_formats[0] == VK_FORMAT_R32G32B32A32_SINT);
   struct kk_fs_link_key snorm = {0};
   snorm.rt_formats[0] = VK_FORMAT_R8G8B8A8_SNORM;
   kk_normalize_fs_link_key(&fs, &snorm);
   CHECK(snorm.rt_formats[0] == VK_FORMAT_R8G8B8A8_SNORM);
   struct kk_fs_link_key blended = {0};
   blended.shader_blend = true;
   blended.rt_formats[0] = VK_FORMAT_B8G8R8A8_UNORM;
   blended.rt_formats[2] = VK_FORMAT_R8_UNORM;
   kk_normalize_fs_link_key(&fs, &blended);
   CHECK(blended.rt_formats[0] == VK_FORMAT_B8G8R8A8_UNORM &&
         blended.rt_formats[2] == VK_FORMAT_R8_UNORM);
}

struct yield_test {
   struct kk_pso_cache *cache;
   bool done;
};

static int
run_yield(void *data)
{
   struct yield_test *t = data;
   kk_yield_to_draws(t->cache);
   p_atomic_set(&t->done, true);
   return 0;
}

static void
test_history_yields_to_draws(void)
{
   struct kk_device dev;
   struct kk_physical_device pdev;
   specialize_fixture(&dev, &pdev);
   kk_draw_demand_begin(&dev.pso_cache);
   kk_draw_demand_begin(&dev.pso_cache);
   struct yield_test t = { .cache = &dev.pso_cache };
   thrd_t thread;
   CHECK(thrd_create(&thread, run_yield, &t) == thrd_success);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   CHECK(!p_atomic_read(&t.done));
   kk_draw_demand_end(&dev.pso_cache);
   thrd_sleep(&(struct timespec){ .tv_nsec = 20000000 }, NULL);
   CHECK(!p_atomic_read(&t.done));
   kk_draw_demand_end(&dev.pso_cache);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(t.done && !dev.pso_cache.draw_demand);
   kk_yield_to_draws(&dev.pso_cache);
   kk_pso_cache_finish(&dev.pso_cache);
}

int
main(void)
{
   CHECK(mtx_init(&producer.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&producer.ready) == thrd_success);
   test_wait_wakeups();
   test_recipe(false);
   test_recipe(true);
   test_queued_job();
   test_lookup_only_miss();
   test_lookup_only_recipe();
   test_concurrent_hits();
   test_normalized_recipes();
   test_link_waits_for_completion();
   test_pending_render_waits_for_completion();
   test_metal_cache_size();
   CHECK(mtx_init(&specializer.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&specializer.ready) == thrd_success);
   test_shared_base();
   test_concurrent_specialize();
   test_specialize_fallback();
   test_specialize_lookup_only();
   test_native_retry();
   test_native_waiters();
   test_normalized_fs_keys();
   test_history_yields_to_draws();
   cnd_destroy(&specializer.ready);
   mtx_destroy(&specializer.lock);
   cnd_destroy(&producer.ready);
   mtx_destroy(&producer.lock);
   return 0;
}
