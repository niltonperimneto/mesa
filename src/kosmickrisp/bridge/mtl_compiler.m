/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_compiler.h"
#include "mtl_format.h"

#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include "util/log.h"
#include "util/u_debug.h"
#include <Metal/MTL4Archive.h>
#include <Metal/MTL4ComputePipeline.h>
#include <Metal/MTL4LibraryDescriptor.h>
#include <Metal/MTL4LibraryFunctionDescriptor.h>
#include <Metal/MTL4LinkingDescriptor.h>
#include <Metal/MTL4PipelineDataSetSerializer.h>
#include <Metal/MTL4RenderPipeline.h>
#include <Metal/MTLDevice.h>
#include <objc/runtime.h>
#include <sys/sysctl.h>

static pthread_once_t kk_metal_log_once = PTHREAD_ONCE_INIT;
static FILE *kk_metal_log;
static uint64_t kk_metal_operation_id;
static _Thread_local uint64_t kk_metal_current_phase;

static void
kk_metal_log_init(void)
{
   const char *path = debug_get_option("MESA_KK_METAL_CONCURRENCY_LOG", NULL);
   if (path)
      kk_metal_log = fopen(path, "a");
}

static uint64_t
kk_metal_log_time(void)
{
   pthread_once(&kk_metal_log_once, kk_metal_log_init);
   if (!kk_metal_log)
      return 0;
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return (uint64_t)now.tv_sec * 1000000000 + now.tv_nsec;
}

static void
kk_metal_log_end(const char *operation, const void *object, uint64_t start,
                 bool success)
{
   if (!start)
      return;
   uint64_t end = kk_metal_log_time();
   uint64_t thread = 0;
   pthread_threadid_np(NULL, &thread);
   flockfile(kk_metal_log);
   fprintf(kk_metal_log,
           "{\"operation\":\"%s\",\"object\":\"%" PRIxPTR
           "\",\"pid\":%d,\"thread_id\":%" PRIu64
           ",\"start_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64
           ",\"success\":%s}\n",
           operation, (uintptr_t)object, getpid(), thread, start, end,
           success ? "true" : "false");
   fflush(kk_metal_log);
   funlockfile(kk_metal_log);
}

bool
mtl_compile_phase_enabled(void)
{
   pthread_once(&kk_metal_log_once, kk_metal_log_init);
   return kk_metal_log != NULL;
}

struct mtl_compile_phase
mtl_compile_phase_begin(const char *operation, const void *object,
                         const char *key, const char *origin)
{
   struct mtl_compile_phase phase = {
      .start = kk_metal_log_time(), .parent = kk_metal_current_phase,
   };
   if (!phase.start)
      return phase;
   uint64_t thread = 0;
   pthread_threadid_np(NULL, &thread);
   flockfile(kk_metal_log);
   phase.id = ++kk_metal_operation_id;
   kk_metal_current_phase = phase.id;
   fprintf(kk_metal_log,
           "{\"phase\":\"begin\",\"operation\":\"%s\",\"operation_id\":%" PRIu64
           ",\"parent_operation_id\":%" PRIu64
           ",\"object\":\"%" PRIxPTR "\",\"pid\":%d,\"thread_id\":%" PRIu64
           ",\"key\":\"%s\",\"origin\":\"%s\",\"time_ns\":%" PRIu64 "}\n",
           operation, phase.id, phase.parent, (uintptr_t)object, getpid(),
           thread, key ? key : "", origin ? origin : "", phase.start);
   fflush(kk_metal_log);
   funlockfile(kk_metal_log);
   return phase;
}

void
mtl_compile_phase_end(struct mtl_compile_phase phase, bool success)
{
   if (!phase.id)
      return;
   uint64_t end = kk_metal_log_time();
   kk_metal_current_phase = phase.parent;
   flockfile(kk_metal_log);
   fprintf(kk_metal_log,
           "{\"phase\":\"end\",\"operation_id\":%" PRIu64
           ",\"pid\":%d,\"time_ns\":%" PRIu64 ",\"success\":%s}\n",
           phase.id, getpid(), end, success ? "true" : "false");
   fflush(kk_metal_log);
   funlockfile(kk_metal_log);
}

@interface KKCompilerLock : NSObject {
@public
   pthread_rwlock_t rwlock;
   bool initialized;
}
@end

@implementation KKCompilerLock
- (instancetype)init
{
   self = [super init];
   if (self) {
      if (pthread_rwlock_init(&rwlock, NULL)) {
         [self release];
         return nil;
      }
      initialized = true;
   }
   return self;
}

- (void)dealloc
{
   if (initialized)
      pthread_rwlock_destroy(&rwlock);
   [super dealloc];
}
@end

static const char mtl_compiler_lock_key;

static KKCompilerLock *
mtl_compiler_lock(mtl_compiler *compiler)
{
   return objc_getAssociatedObject((id)compiler, &mtl_compiler_lock_key);
}

@interface KKMetalArchive : NSObject {
@public
   id<MTL4Archive> handle;
   NSURL *url;
}
@end

@implementation KKMetalArchive
- (void)dealloc
{
   [handle release];
   [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
   [url release];
   [super dealloc];
}
@end

static const char mtl_linked_functions_key;

static MTL4StaticLinkingDescriptor *
mtl_static_linking(mtl_function_descriptor *function)
{
   NSArray *functions = function ? objc_getAssociatedObject(
                                      (id)function, &mtl_linked_functions_key)
                                 : nil;
   if (!functions)
      return nil;
   MTL4StaticLinkingDescriptor *linking =
      [[MTL4StaticLinkingDescriptor new] autorelease];
   linking.functionDescriptors = functions;
   return linking;
}

/* Compiler */
static mtl_compiler *
mtl_new_compiler_with_capture(mtl_device *device, bool capture)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;

      MTL4CompilerDescriptor *desc = [[MTL4CompilerDescriptor new] autorelease];

      if (capture &&
          !debug_get_bool_option("MESA_KK_PSO_CACHE_DISABLE", false) &&
          !debug_get_bool_option("MESA_SHADER_CACHE_DISABLE", false)) {
         MTL4PipelineDataSetSerializerDescriptor *serializer_desc =
            [[MTL4PipelineDataSetSerializerDescriptor new] autorelease];
         serializer_desc.configuration =
            MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
         desc.pipelineDataSetSerializer =
            [[dev newPipelineDataSetSerializerWithDescriptor:
               serializer_desc] autorelease];
      }

      NSError *error = nil;
      id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:desc error:&error];

      if (compiler == nil) {
         fprintf(stderr, "Failed to create MTL4Compiler: %s\n", [error.localizedDescription UTF8String]);
      } else if (compiler.pipelineDataSetSerializer) {
         KKCompilerLock *lock = [[KKCompilerLock new] autorelease];
         if (!lock) {
            [compiler release];
            return NULL;
         }
         objc_setAssociatedObject(compiler, &mtl_compiler_lock_key, lock,
                                  OBJC_ASSOCIATION_RETAIN);
      }

      return compiler;
   }
}

mtl_compiler *
mtl_new_compiler(mtl_device *device)
{
   return mtl_new_compiler_with_capture(device, true);
}

mtl_compiler *
mtl_new_uncaptured_compiler(mtl_device *device)
{
   return mtl_new_compiler_with_capture(device, false);
}

/* Library */
mtl_library *
mtl_new_library(mtl_compiler *compiler, const char *src,
                enum mtl_language_version msl_version,
                enum mtl_math_mode math_mode,
                enum mtl_math_floating_point_functions math_fp_fns)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      NSString *ns_src = [NSString stringWithCString:src encoding:NSASCIIStringEncoding];

      MTLCompileOptions *opts = [[MTLCompileOptions new] autorelease];
      opts.languageVersion = (MTLLanguageVersion)msl_version;
      opts.mathMode = (MTLMathMode)math_mode;
      opts.mathFloatingPointFunctions = (MTLMathFloatingPointFunctions)math_fp_fns;

      MTL4LibraryDescriptor *desc = [[MTL4LibraryDescriptor new] autorelease];
      desc.source = ns_src;
      desc.options = opts;

      NSError *error = nil;
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("library", compiler, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLLibrary> lib = [comp newLibraryWithDescriptor:desc error:&error];
      kk_metal_log_end("library", compiler, start, lib != nil);
      mtl_compile_phase_end(phase, lib != nil);

      if (lib == nil) {
         fprintf(stderr, "Failed to create MTLLibrary: %s\n", [error.localizedDescription UTF8String]);
      }

      return lib;
   }
}

mtl_function_descriptor *
mtl_new_library_function_descriptor(mtl_library *library, const char *entry_point)
{
   @autoreleasepool {
      id<MTLLibrary> lib = (id<MTLLibrary>)library;
      NSString *ns_entry_point = [NSString stringWithCString:entry_point encoding:NSASCIIStringEncoding];

      MTL4LibraryFunctionDescriptor *desc = [MTL4LibraryFunctionDescriptor new];
      desc.name = ns_entry_point;
      desc.library = lib;
      return desc;
   }
}

void
mtl_function_descriptor_set_linked_functions(mtl_function_descriptor *function,
                                             mtl_function_descriptor **functions,
                                             uint32_t count)
{
   @autoreleasepool {
      NSArray *linked =
         count ? [NSArray arrayWithObjects:(id *)functions count:count] : nil;
      objc_setAssociatedObject((id)function, &mtl_linked_functions_key, linked,
                               OBJC_ASSOCIATION_RETAIN);
   }
}

/* Compute pipeline */
mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state(mtl_compiler *compiler,
                               mtl_function_descriptor *function,
                               uint64_t max_total_threads_per_threadgroup)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      KKCompilerLock *lock = mtl_compiler_lock(compiler);

      MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.staticLinkingDescriptor = mtl_static_linking(function);
      desc.maxTotalThreadsPerThreadgroup = max_total_threads_per_threadgroup;

      NSError *error = nil;
      if (lock) {
         struct mtl_compile_phase lock_phase =
            mtl_compile_phase_begin("compiler_read_lock", compiler, NULL, NULL);
         pthread_rwlock_rdlock(&lock->rwlock);
         mtl_compile_phase_end(lock_phase, true);
      }
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("compute", compiler, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLComputePipelineState> pipeline = [comp newComputePipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                   error:&error];
      kk_metal_log_end("compute", compiler, start, pipeline != nil);
      mtl_compile_phase_end(phase, pipeline != nil);
      if (lock)
         pthread_rwlock_unlock(&lock->rwlock);

      if (pipeline == nil) {
         fprintf(stderr, "Failed to create MTLComputePipelineState: %s\n", [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}

/* Render pipeline descriptor */
mtl_render_pipeline_descriptor *
mtl_new_render_pipeline_descriptor()
{
   @autoreleasepool {
      return [MTL4RenderPipelineDescriptor new];
   }
}

void
mtl_render_pipeline_descriptor_set_vertex_shader(mtl_render_pipeline_descriptor *descriptor,
                                                 mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.vertexFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
      desc.vertexStaticLinkingDescriptor = mtl_static_linking(shader);
   }
}

void
mtl_render_pipeline_descriptor_set_fragment_shader(mtl_render_pipeline_descriptor *descriptor,
                                                   mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.fragmentFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
      desc.fragmentStaticLinkingDescriptor = mtl_static_linking(shader);
   }
}

void
mtl_render_pipeline_descriptor_set_input_primitive_topology(mtl_render_pipeline_descriptor *descriptor,
                                                            enum mtl_primitive_topology_class class)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)class;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_format(mtl_render_pipeline_descriptor *descriptor,
                                                           uint8_t index,
                                                           enum mtl_pixel_format format)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.colorAttachments[index].pixelFormat = (MTLPixelFormat)format;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_blend(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index, bool enable,
   enum mtl_blend_factor src_rgb, enum mtl_blend_factor dst_rgb,
   enum mtl_blend_operation op_rgb, enum mtl_blend_factor src_alpha,
   enum mtl_blend_factor dst_alpha, enum mtl_blend_operation op_alpha,
   uint8_t write_mask)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc =
         (MTL4RenderPipelineDescriptor *)descriptor;
      MTL4RenderPipelineColorAttachmentDescriptor *att =
         desc.colorAttachments[index];
      att.blendingState =
         enable ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
      att.sourceRGBBlendFactor = (MTLBlendFactor)src_rgb;
      att.destinationRGBBlendFactor = (MTLBlendFactor)dst_rgb;
      att.rgbBlendOperation = (MTLBlendOperation)op_rgb;
      att.sourceAlphaBlendFactor = (MTLBlendFactor)src_alpha;
      att.destinationAlphaBlendFactor = (MTLBlendFactor)dst_alpha;
      att.alphaBlendOperation = (MTLBlendOperation)op_alpha;
      att.writeMask = (MTLColorWriteMask)write_mask;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_unspecialized(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc =
         (MTL4RenderPipelineDescriptor *)descriptor;
      MTL4RenderPipelineColorAttachmentDescriptor *att =
         desc.colorAttachments[index];
      att.pixelFormat = MTLPixelFormatUnspecialized;
      att.blendingState = MTL4BlendStateUnspecialized;
      att.sourceRGBBlendFactor = MTLBlendFactorUnspecialized;
      att.destinationRGBBlendFactor = MTLBlendFactorUnspecialized;
      att.rgbBlendOperation = MTLBlendOperationUnspecialized;
      att.sourceAlphaBlendFactor = MTLBlendFactorUnspecialized;
      att.destinationAlphaBlendFactor = MTLBlendFactorUnspecialized;
      att.alphaBlendOperation = MTLBlendOperationUnspecialized;
      att.writeMask = MTLColorWriteMaskUnspecialized;
   }
}

void
mtl_render_pipeline_descriptor_set_raster_sample_count(mtl_render_pipeline_descriptor *descriptor,
                                                       uint32_t sample_count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterSampleCount = sample_count;
   }
}

void
mtl_render_pipeline_descriptor_set_multisample_alpha(
   mtl_render_pipeline_descriptor *descriptor, bool coverage, bool one)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc =
         (MTL4RenderPipelineDescriptor *)descriptor;
      desc.alphaToCoverageState =
         coverage ? MTL4AlphaToCoverageStateEnabled
                  : MTL4AlphaToCoverageStateDisabled;
      desc.alphaToOneState =
         one ? MTL4AlphaToOneStateEnabled
             : MTL4AlphaToOneStateDisabled;
   }
}

void
mtl_render_pipeline_descriptor_set_rasterization_enabled(mtl_render_pipeline_descriptor *descriptor,
                                                         bool enabled)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterizationEnabled = enabled;
   }
}

void
mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(mtl_render_pipeline_descriptor *descriptor,
                                                                  uint32_t count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.maxVertexAmplificationCount = count;
   }
}

/* Render pipeline */
mtl_render_pipeline_state *
mtl_new_render_pipeline(mtl_compiler *compiler, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      KKCompilerLock *lock = mtl_compiler_lock(compiler);

      NSError *error = nil;
      if (lock) {
         struct mtl_compile_phase lock_phase =
            mtl_compile_phase_begin("compiler_read_lock", compiler, NULL, NULL);
         pthread_rwlock_rdlock(&lock->rwlock);
         mtl_compile_phase_end(lock_phase, true);
      }
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("render", compiler, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLRenderPipelineState> pipeline = [comp newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                 error:&error];
      kk_metal_log_end("render", compiler, start, pipeline != nil);
      mtl_compile_phase_end(phase, pipeline != nil);
      if (lock)
         pthread_rwlock_unlock(&lock->rwlock);

      if (pipeline == nil) {
         mesa_logd("Failed to create MTLRenderPipelineState: %s",
                   [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}

mtl_render_pipeline_state *
mtl_specialize_render_pipeline(mtl_compiler *compiler,
                               mtl_render_pipeline_descriptor *descriptor,
                               mtl_render_pipeline_state *base)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      MTL4RenderPipelineDescriptor *desc =
         (MTL4RenderPipelineDescriptor *)descriptor;
      KKCompilerLock *lock = mtl_compiler_lock(compiler);

      NSError *error = nil;
      if (lock) {
         struct mtl_compile_phase lock_phase =
            mtl_compile_phase_begin("compiler_read_lock", compiler, NULL, NULL);
         pthread_rwlock_rdlock(&lock->rwlock);
         mtl_compile_phase_end(lock_phase, true);
      }
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("specialize", compiler, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLRenderPipelineState> base_pso =
         (id<MTLRenderPipelineState>)base;
      id<MTLRenderPipelineState> pipeline =
         [comp newRenderPipelineStateBySpecializationWithDescriptor:desc
                                                           pipeline:base_pso
                                                              error:&error];
      kk_metal_log_end("specialize", compiler, start, pipeline != nil);
      mtl_compile_phase_end(phase, pipeline != nil);
      if (lock)
         pthread_rwlock_unlock(&lock->rwlock);

      if (pipeline == nil) {
         mesa_logd("Failed to specialize MTLRenderPipelineState: %s",
                   [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}

bool
mtl_compiler_has_serializer(mtl_compiler *compiler)
{
   return [(id<MTL4Compiler>)compiler pipelineDataSetSerializer] != nil;
}

static NSURL *
mtl_archive_temporary_url(void)
{
   NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
      [NSString stringWithFormat:@"mesa-kk-%@", [[NSUUID UUID] UUIDString]]];
   return [NSURL fileURLWithPath:path];
}

void *
mtl_compiler_serialize(mtl_compiler *compiler, size_t *size)
{
   *size = 0;
   @autoreleasepool {
      id<MTL4PipelineDataSetSerializer> serializer =
         [(id<MTL4Compiler>)compiler pipelineDataSetSerializer];
      KKCompilerLock *lock = mtl_compiler_lock(compiler);
      if (!serializer || !lock)
         return NULL;
      NSURL *url = mtl_archive_temporary_url();
      NSError *error = nil;
      struct mtl_compile_phase lock_phase =
         mtl_compile_phase_begin("compiler_write_lock", compiler, NULL, NULL);
      pthread_rwlock_wrlock(&lock->rwlock);
      mtl_compile_phase_end(lock_phase, true);
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("serialize", compiler, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      BOOL ok = [serializer serializeAsArchiveAndFlushToURL:url error:&error];
      kk_metal_log_end("serialize", compiler, start, ok);
      mtl_compile_phase_end(phase, ok);
      pthread_rwlock_unlock(&lock->rwlock);
      NSData *data = ok ? [NSData dataWithContentsOfURL:url] : nil;
      [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
      if (!data || !data.length)
         return NULL;
      void *bytes = malloc(data.length);
      if (bytes) {
         memcpy(bytes, data.bytes, data.length);
         *size = data.length;
      }
      return bytes;
   }
}

void
mtl_pipeline_cache_identity(mtl_device *device, uint32_t gpu_family,
                            char *identity, size_t size)
{
   char build[128] = {0};
   size_t build_size = sizeof(build);
   sysctlbyname("kern.osversion", build, &build_size, NULL, 0);
   @autoreleasepool {
      const char *name = [[(id<MTLDevice>)device name] UTF8String];
      snprintf(identity, size, "metal4-pso-3:%s:%u:%s", build, gpu_family,
               name ? name : "");
   }
}

mtl_archive *
mtl_new_archive(mtl_device *device, const void *bytes, size_t size)
{
   @autoreleasepool {
      NSURL *url = mtl_archive_temporary_url();
      NSData *data = [NSData dataWithBytesNoCopy:(void *)bytes length:size
                                    freeWhenDone:NO];
      if (![data writeToURL:url options:NSDataWritingAtomic error:NULL])
         return NULL;
      NSError *error = nil;
      id<MTL4Archive> archive =
         [(id<MTLDevice>)device newArchiveWithURL:url error:&error];
      if (!archive) {
         [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
         return NULL;
      }
      KKMetalArchive *result = [KKMetalArchive new];
      result->handle = archive;
      result->url = [url retain];
      return result;
   }
}

mtl_render_pipeline_state *
mtl_archive_new_render_pipeline(mtl_archive *archive,
                                mtl_render_pipeline_descriptor *descriptor)
{
   @autoreleasepool {
      NSError *error = nil;
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("archive_render", archive, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLRenderPipelineState> pipeline =
         [((KKMetalArchive *)archive)->handle
            newRenderPipelineStateWithDescriptor:
               (MTL4RenderPipelineDescriptor *)descriptor error:&error];
      kk_metal_log_end("archive_render", archive, start, pipeline != nil);
      mtl_compile_phase_end(phase, pipeline != nil);
      return pipeline;
   }
}

mtl_compute_pipeline_state *
mtl_archive_new_compute_pipeline(mtl_archive *archive,
                                 mtl_function_descriptor *function,
                                 uint64_t max_total_threads_per_threadgroup)
{
   @autoreleasepool {
      MTL4ComputePipelineDescriptor *desc =
         [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.staticLinkingDescriptor = mtl_static_linking(function);
      desc.maxTotalThreadsPerThreadgroup = max_total_threads_per_threadgroup;
      NSError *error = nil;
      struct mtl_compile_phase phase =
         mtl_compile_phase_begin("archive_compute", archive, NULL, NULL);
      uint64_t start = kk_metal_log_time();
      id<MTLComputePipelineState> pipeline =
         [((KKMetalArchive *)archive)->handle
            newComputePipelineStateWithDescriptor:desc error:&error];
      kk_metal_log_end("archive_compute", archive, start, pipeline != nil);
      mtl_compile_phase_end(phase, pipeline != nil);
      return pipeline;
   }
}
