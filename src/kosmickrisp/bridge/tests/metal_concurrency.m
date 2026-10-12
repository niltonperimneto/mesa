#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "kosmickrisp/bridge/mtl_compiler.h"
#include "kosmickrisp/bridge/mtl_format.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "check failed: %s at %d\n", #x, __LINE__); \
      abort(); \
   } \
} while (0)
#define WORKERS 8
#define CYCLES 10

const char *debug_get_option(const char *name, const char *fallback)
{
   const char *value = getenv(name);
   return value ? value : fallback;
}

bool debug_get_bool_option(const char *name, bool fallback)
{
   const char *value = getenv(name);
   return value ? !strcmp(value, "1") : fallback;
}

typedef pthread_mutex_t gate_mutex_t;
typedef pthread_cond_t gate_cond_t;

struct gate {
   gate_mutex_t mutex;
   gate_cond_t cond;
   unsigned arrived, generation, total;
};

static struct gate gate = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static mtl_compiler *compiler;
static mtl_compiler *uncaptured;
static bool bypass_phase;
static mtl_archive *archive;
static mtl_library *libraries[WORKERS];
static mtl_function_descriptor *compute[WORKERS];
static mtl_function_descriptor *vertex[WORKERS];
static mtl_function_descriptor *fragment[WORKERS];

static void wait_gate(void)
{
   (void)pthread_mutex_lock(&gate.mutex);
   unsigned generation = gate.generation;
   if (++gate.arrived == gate.total) {
      gate.arrived = 0;
      gate.generation++;
      (void)pthread_cond_broadcast(&gate.cond);
   } else {
      while (generation == gate.generation)
         (void)pthread_cond_wait(&gate.cond, &gate.mutex);
   }
   (void)pthread_mutex_unlock(&gate.mutex);
}

static mtl_library *library(unsigned index)
{
   char source[2048];
   snprintf(source, sizeof(source),
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void cs(device uint *out [[buffer(0)]], "
      "uint i [[thread_position_in_grid]]) { out[i] = i + %u; }\n"
      "vertex float4 vs(uint i [[vertex_id]]) { "
      "float2 p[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)}; "
      "return float4(p[i],0,1); }\n"
      "fragment float4 fs() { return float4(1,0,0,1); }\n", 1234 + index);
   mtl_library *result = mtl_new_library(
      compiler, source, MTL_LANGUAGE_VERSION_4_0,
      MTL_MATH_MODE_SAFE, MTL_MATH_FLOATING_POINT_FUNCTIONS_PRECISE);
   CHECK(result);
   return result;
}

static mtl_render_pipeline_descriptor *descriptor(mtl_function_descriptor *vs,
                                                   mtl_function_descriptor *fs)
{
   mtl_render_pipeline_descriptor *desc = mtl_new_render_pipeline_descriptor();
   mtl_render_pipeline_descriptor_set_vertex_shader(desc, vs);
   mtl_render_pipeline_descriptor_set_fragment_shader(desc, fs);
   mtl_render_pipeline_descriptor_set_color_attachment_format(
      desc, 0, MTL_PIXEL_FORMAT_R8G8B8A8_UNORM);
   return desc;
}

static void verify_compute(id<MTLComputePipelineState> pipeline, unsigned index)
{
   CHECK(pipeline);
   id<MTLBuffer> buffer =
      [device newBufferWithLength:64 * sizeof(uint32_t)
                          options:MTLResourceStorageModeShared];
   id<MTLCommandBuffer> command = [queue commandBuffer];
   id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
   [encoder setComputePipelineState:pipeline];
   [encoder setBuffer:buffer offset:0 atIndex:0];
   [encoder dispatchThreads:MTLSizeMake(64, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
   [encoder endEncoding];
   [command commit];
   [command waitUntilCompleted];
   CHECK(command.status == MTLCommandBufferStatusCompleted);
   for (unsigned i = 0; i < 64; i++)
      CHECK(((uint32_t *)buffer.contents)[i] == i + 1234 + index);
   [buffer release];
}

static void verify_render(id<MTLRenderPipelineState> pipeline)
{
   CHECK(pipeline);
   MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
         MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
   td.storageMode = MTLStorageModeShared;
   td.usage = MTLTextureUsageRenderTarget;
   id<MTLTexture> texture = [device newTextureWithDescriptor:td];
   MTLRenderPassDescriptor *pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
   pass.colorAttachments[0].texture = texture;
   pass.colorAttachments[0].loadAction = MTLLoadActionClear;
   pass.colorAttachments[0].storeAction = MTLStoreActionStore;
   id<MTLCommandBuffer> command = [queue commandBuffer];
   id<MTLRenderCommandEncoder> encoder =
      [command renderCommandEncoderWithDescriptor:pass];
   [encoder setRenderPipelineState:pipeline];
   [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0 vertexCount:3];
   [encoder endEncoding];
   [command commit];
   [command waitUntilCompleted];
   CHECK(command.status == MTLCommandBufferStatusCompleted);
   unsigned char pixels[8 * 8 * 4];
   [texture getBytes:pixels bytesPerRow:32
          fromRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0];
   for (unsigned i = 0; i < 64; i++)
      CHECK(pixels[4 * i] == 255 && pixels[4 * i + 1] == 0 &&
            pixels[4 * i + 2] == 0 && pixels[4 * i + 3] == 255);
   [texture release];
}

static void *compile_worker(void *arg)
{
   unsigned index = (unsigned)(uintptr_t)arg;
   for (unsigned cycle = 0; cycle < CYCLES; cycle++) {
      @autoreleasepool {
         if (bypass_phase && index == WORKERS) {
            id pipeline = (id)mtl_new_compute_pipeline_state(
               compiler, compute[cycle % WORKERS], 64);
            CHECK(pipeline);
            [pipeline release];
         }
         wait_gate();
         if (index == WORKERS) {
            size_t size;
            void *data = mtl_compiler_serialize(compiler, &size);
            if (bypass_phase)
               CHECK(data && size);
            free(data);
         } else {
            if (bypass_phase) {
               mtl_render_pipeline_descriptor *desc =
                  descriptor(vertex[index], fragment[index]);
               id<MTLComputePipelineState> cp =
                  (id)mtl_new_compute_pipeline_state(
                     uncaptured, compute[index], 64);
               id<MTLRenderPipelineState> rp =
                  (id)mtl_new_render_pipeline(uncaptured, desc);
               verify_compute(cp, index + 1000);
               verify_render(rp);
               [cp release]; [rp release]; [(id)desc release];
               wait_gate();
               continue;
            }
            unsigned value = index + cycle * WORKERS;
            mtl_library *lib = library(value);
            mtl_function_descriptor *cs =
               mtl_new_library_function_descriptor(lib, "cs");
            mtl_function_descriptor *vs =
               mtl_new_library_function_descriptor(lib, "vs");
            mtl_function_descriptor *fs =
               mtl_new_library_function_descriptor(lib, "fs");
            mtl_render_pipeline_descriptor *desc = descriptor(vs, fs);
            id<MTLComputePipelineState> cp =
               (id)mtl_new_compute_pipeline_state(compiler, cs, 64);
            id<MTLRenderPipelineState> rp =
               (id)mtl_new_render_pipeline(compiler, desc);
            verify_compute(cp, value);
            verify_render(rp);
            [cp release]; [rp release]; [(id)desc release];
            [(id)cs release]; [(id)vs release];
            [(id)fs release]; [(id)lib release];
         }
         wait_gate();
      }
   }
   return NULL;
}

static void *archive_worker(void *arg)
{
   unsigned index = (unsigned)(uintptr_t)arg;
   for (unsigned iteration = 0; iteration < 32; iteration++) {
      @autoreleasepool {
         wait_gate();
         if (iteration % 2) {
            mtl_render_pipeline_descriptor *desc =
               descriptor(vertex[index], fragment[index]);
            id<MTLRenderPipelineState> p =
               (id)mtl_archive_new_render_pipeline(archive, desc);
            verify_render(p); [p release]; [(id)desc release];
         } else {
            id<MTLComputePipelineState> p =
               (id)mtl_archive_new_compute_pipeline(
                  archive, compute[index], 64);
            verify_compute(p, index + 1000); [p release];
         }
         wait_gate();
      }
   }
   return NULL;
}

static void run_workers(unsigned count, void *(*worker)(void *))
{
   pthread_t threads[WORKERS + 1];
   gate.total = count;
   for (unsigned i = 0; i < count; i++)
      CHECK(!pthread_create(&threads[i], NULL, worker, (void *)(uintptr_t)i));
   for (unsigned i = 0; i < count; i++)
      CHECK(!pthread_join(threads[i], NULL));
}

int main(int argc, char **argv)
{
   @autoreleasepool {
      device = MTLCreateSystemDefaultDevice(); CHECK(device);
      queue = [device newCommandQueue]; CHECK(queue);
      compiler = mtl_new_compiler((void *)device); CHECK(compiler);
      CHECK(mtl_compiler_has_serializer(compiler));
      uncaptured = mtl_new_uncaptured_compiler((void *)device);
      CHECK(uncaptured);
      CHECK(!mtl_compiler_has_serializer(uncaptured));
      mtl_library *seed = library(2000);
      mtl_function_descriptor *seed_function =
         mtl_new_library_function_descriptor(seed, "cs");
      id seed_pipeline =
         (id)mtl_new_compute_pipeline_state(compiler, seed_function, 64);
      CHECK(seed_pipeline);
      [seed_pipeline release]; [(id)seed_function release]; [(id)seed release];
      if (argc == 1)
         run_workers(WORKERS + 1, compile_worker);
      size_t size;
      void *discard = mtl_compiler_serialize(compiler, &size); free(discard);
      for (unsigned i = 0; i < WORKERS; i++) {
         libraries[i] = library(i + 1000);
         compute[i] = mtl_new_library_function_descriptor(libraries[i], "cs");
         vertex[i] = mtl_new_library_function_descriptor(libraries[i], "vs");
         fragment[i] = mtl_new_library_function_descriptor(libraries[i], "fs");
         id cp = (id)mtl_new_compute_pipeline_state(compiler, compute[i], 64);
         CHECK(cp); [cp release];
         mtl_render_pipeline_descriptor *desc =
            descriptor(vertex[i], fragment[i]);
         id rp = (id)mtl_new_render_pipeline(compiler, desc);
         CHECK(rp); [rp release]; [(id)desc release];
      }
      const char *archive_path = getenv("KK_TEST_ARCHIVE_FILE");
      void *bytes;
      if (argc == 2 && !strcmp(argv[1], "--read-archive")) {
         CHECK(archive_path);
         FILE *file = fopen(archive_path, "rb"); CHECK(file);
         CHECK(!fseek(file, 0, SEEK_END));
         long length = ftell(file); CHECK(length > 0);
         CHECK(!fseek(file, 0, SEEK_SET));
         size = length;
         bytes = malloc(size); CHECK(bytes);
         CHECK(fread(bytes, 1, size, file) == size);
         CHECK(!fclose(file));
      } else {
         void *seed_bytes = mtl_compiler_serialize(compiler, &size);
         CHECK(seed_bytes && size);
         bypass_phase = true;
         struct mtl_compile_phase phase =
            mtl_compile_phase_begin(
               "uncaptured_session", uncaptured, NULL, NULL);
         run_workers(WORKERS + 1, compile_worker);
         mtl_compile_phase_end(phase, true);
         bypass_phase = false;
         bytes = seed_bytes;
         if (archive_path) {
            FILE *file = fopen(archive_path, "wb"); CHECK(file);
            CHECK(fwrite(bytes, 1, size, file) == size);
            CHECK(!fclose(file));
         }
      }
      archive = mtl_new_archive((void *)device, bytes, size);
      CHECK(archive); free(bytes);
      run_workers(WORKERS, archive_worker);
      [(id)archive release];
      for (unsigned i = 0; i < WORKERS; i++) {
         [(id)compute[i] release]; [(id)vertex[i] release];
         [(id)fragment[i] release]; [(id)libraries[i] release];
      }
      [(id)uncaptured release]; [(id)compiler release];
      [queue release]; [device release];
      (void)pthread_cond_destroy(&gate.cond);
      (void)pthread_mutex_destroy(&gate.mutex);
      puts("real Metal compilation, serialization, "
           "shared archive and GPU readback: PASS");
   }
   return 0;
}
