#define mtl_specialize_render_pipeline test_specialize_pipeline
#include "../kk_metal_cache.c"
#undef mtl_specialize_render_pipeline

#include <stdio.h>

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort(); \
   } \
} while (0)

static struct {
   mtx_t lock;
   cnd_t cond;
   mtl_compiler *compiler;
   bool entered;
   bool release;
   bool fail;
} mock;
static int captured_compiler, uncaptured_compiler;
static int base_pipeline, child_pipeline;

mtl_render_pipeline_state *
test_specialize_pipeline(mtl_compiler *compiler,
                         mtl_render_pipeline_descriptor *descriptor,
                         mtl_render_pipeline_state *base)
{
   CHECK(base == (void *)&base_pipeline);
   mtx_lock(&mock.lock);
   mock.compiler = compiler;
   mock.entered = true;
   cnd_broadcast(&mock.cond);
   while (!mock.release)
      cnd_wait(&mock.cond, &mock.lock);
   bool fail = mock.fail;
   mtx_unlock(&mock.lock);
   return fail ? NULL : (void *)&child_pipeline;
}

struct demand {
   struct kk_metal_cache *cache;
   mtl_render_pipeline_state *result;
};

static int
specialize(void *data)
{
   struct demand *demand = data;
   cache_key key = {0};
   demand->result = kk_metal_cache_specialize_render_pipeline(
      demand->cache, key, NULL, (void *)&base_pipeline);
   return 0;
}

static void
test_specialization(bool uncaptured, bool serializing, bool fail)
{
   struct kk_metal_cache cache = {
      .compiler = (void *)&captured_compiler,
      .uncaptured = uncaptured ? (void *)&uncaptured_compiler : NULL,
      .serializing = serializing,
   };
   CHECK(mtx_init(&cache.mutex, mtx_plain) == thrd_success);
   CHECK(cnd_init(&cache.cond) == thrd_success);
   mock.entered = mock.release = false;
   mock.fail = fail;
   mock.compiler = NULL;
   struct demand demand = { .cache = &cache };
   thrd_t thread;
   CHECK(thrd_create(&thread, specialize, &demand) == thrd_success);
   if (!uncaptured && serializing) {
      int64_t deadline = os_time_get_nano() + 1000000000ll;
      bool admitted = false;
      while (!admitted && os_time_get_nano() < deadline) {
         mtx_lock(&cache.mutex);
         admitted = cache.active_operations == 1;
         mtx_unlock(&cache.mutex);
         thrd_yield();
      }
      CHECK(admitted);
      mtx_lock(&mock.lock);
      CHECK(!mock.entered);
      mtx_unlock(&mock.lock);
      mtx_lock(&cache.mutex);
      cache.serializing = false;
      cnd_broadcast(&cache.cond);
      mtx_unlock(&cache.mutex);
   }
   mtx_lock(&mock.lock);
   while (!mock.entered)
      cnd_wait(&mock.cond, &mock.lock);
   CHECK(mock.compiler == (uncaptured ? (void *)&uncaptured_compiler :
                                         (void *)&captured_compiler));
   mtx_unlock(&mock.lock);
   mtx_lock(&cache.mutex);
   CHECK(cache.active_operations == 1);
   CHECK(cache.uncaptured_compiles == (unsigned)uncaptured);
   CHECK(cache.captured_compiles == (unsigned)!uncaptured);
   CHECK(cache.bypassed_compiles == (unsigned)(uncaptured && serializing));
   cache.closing = true;
   mtx_unlock(&cache.mutex);
   mtx_lock(&mock.lock);
   mock.release = true;
   cnd_broadcast(&mock.cond);
   mtx_unlock(&mock.lock);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(demand.result == (fail ? NULL : (void *)&child_pipeline));
   CHECK(!cache.active_operations && !cache.captured_compiles &&
         !cache.uncaptured_compiles && !cache.pending && !cache.recapture);
   mock.compiler = NULL;
   cache_key key = {0};
   mtl_render_pipeline_state *closed = kk_metal_cache_specialize_render_pipeline(
      &cache, key, NULL, (void *)&base_pipeline);
   CHECK(closed == (fail ? NULL : (void *)&child_pipeline));
   CHECK(mock.compiler == (uncaptured ? (void *)&uncaptured_compiler :
                                         (void *)&captured_compiler));
   CHECK(!cache.active_operations && !cache.captured_compiles &&
         !cache.uncaptured_compiles && !cache.pending && !cache.recapture);
   cnd_destroy(&cache.cond);
   mtx_destroy(&cache.mutex);
}

int
main(void)
{
   CHECK(mtx_init(&mock.lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&mock.cond) == thrd_success);
   for (unsigned uncaptured = 0; uncaptured < 2; ++uncaptured)
      for (unsigned serializing = 0; serializing < 2; ++serializing)
         for (unsigned fail = 0; fail < 2; ++fail)
            test_specialization(uncaptured, serializing, fail);
   cnd_destroy(&mock.cond);
   mtx_destroy(&mock.lock);
   return 0;
}
