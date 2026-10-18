/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_DEVICE_H
#define KK_DEVICE_H 1

#include "kk_msl_helpers.h"
#include "kk_private.h"
#include "kk_pso_cache.h"

#include "kk_query_table.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/clc/kk_precompiled_shader.h"
#include "libkk_shaders.h"

#include "util/os_time.h"
#include "util/u_dynarray.h"
#include "util/u_thread.h"

#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct vk_shader_cache_trace;
struct kk_metal_cache;
struct kk_bo;
struct kk_physical_device;
struct vk_pipeline_cache;

struct kk_residency_set {
   simple_mtx_t mutex;
   mtl_residency_set *handle;
};

struct mtl_sampler_packed {
   enum mtl_sampler_address_mode mode_u;
   enum mtl_sampler_address_mode mode_v;
   enum mtl_sampler_address_mode mode_w;
   enum mtl_sampler_border_color border_color;

   enum mtl_sampler_min_mag_filter min_filter;
   enum mtl_sampler_min_mag_filter mag_filter;
   enum mtl_sampler_mip_filter mip_filter;
   enum mtl_sampler_reduction_mode reduction_mode;

   enum mtl_compare_function compare_func;
   float min_lod;
   float max_lod;
   uint32_t max_anisotropy;
   bool normalized_coordinates;
};

struct kk_rc_sampler {
   struct mtl_sampler_packed key;

   mtl_sampler *handle;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct kk_sampler_heap {
   simple_mtx_t lock;

   struct kk_query_table table;

   /* Map of mtl_sampler_packed to kk_rc_sampler */
   struct hash_table *ht;
};

struct kk_precompiled_cache {
   struct kk_precompiled_shader shaders[LIBKK_NUM_PROGRAMS];
};

enum kk_variant_phase {
   KK_VARIANT_ENQUEUE,
   KK_VARIANT_WORKER_START,
   KK_VARIANT_FS_PREPARE,
   KK_VARIANT_FS_WAIT,
   KK_VARIANT_RASTER_PREPARE,
   KK_VARIANT_RASTER_WAIT,
   KK_VARIANT_PSO_CREATE,
   KK_VARIANT_PSO_WAIT,
   KK_VARIANT_ARCHIVE_LOOKUP,
   KK_VARIANT_ARCHIVE_LOAD,
   KK_VARIANT_NATIVE_CREATE,
   KK_VARIANT_COMPLETE,
   KK_VARIANT_FIRST_DEMAND,
   KK_VARIANT_READY_HIT,
   KK_VARIANT_FAILURE,
   KK_VARIANT_HISTORY_LOAD,
   KK_VARIANT_REF_MISSING,
   KK_VARIANT_BLOB_MISSING,
   KK_VARIANT_ARCHIVE_REJECTED,
   KK_VARIANT_PIPELINE_ABSENT,
   KK_VARIANT_QUEUE_WAIT,
   KK_VARIANT_NATIVE_COMPILE,
   KK_VARIANT_NATIVE_SPECIALIZE,
   KK_VARIANT_REF_UNCACHEABLE,
   KK_VARIANT_BLOB_CORRUPT,
   KK_VARIANT_COUNT,
};

struct kk_variant_counter {
   uint64_t count;
   uint64_t duration_ns;
   uint64_t draw_count;
   uint64_t draw_duration_ns;
};

struct kk_device {
   struct vk_device vk;
   const char *capture_trigger;
   const char *capture_directory;
   unsigned capture_frames;
   const char *present_log_path;
   FILE *present_log;
   FILE *compile_log;
   FILE *cache_log;
   bool cache_log_hits;
   FILE *variant_log;
   FILE *variant_stats;
   bool variant_trace_enabled;
   struct kk_variant_counter variant_counters[KK_VARIANT_COUNT];
   uint64_t fs_variants_created;
   struct kk_metal_cache *metal_cache;
   struct disk_cache *metal_disk;

   mtl_device *mtl_handle;
   mtl_compiler *mtl_compiler_handle;

   /* Dispatch table exposed to the user. Required since we need to record all
    * commands due to Metal limitations */
   struct vk_device_dispatch_table exposed_dispatch_table;

   struct kk_sampler_heap samplers;
   struct kk_query_table occlusion_queries;

   /* Track all heaps the user allocated so we can set them all as resident when
    * recording as required by Metal. */
   struct kk_residency_set residency_set;

   struct kk_precompiled_cache precompiled_cache;

   bool has_queue;
   struct kk_queue queue;

   struct vk_meta_device meta;

   /* Geomtry heap */
   struct kk_bo *heap;
   util_once_flag heap_init_once;

   struct kk_pso_cache pso_cache;
   struct kk_msl_helpers msl_helpers;
   simple_mtx_t empty_fs_lock;
   struct kk_shader *empty_fs;
};

VK_DEFINE_HANDLE_CASTS(kk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct kk_physical_device *
kk_device_physical(const struct kk_device *dev)
{
   return (struct kk_physical_device *)dev->vk.physical;
}

VkResult kk_device_init_meta(struct kk_device *dev);
void kk_device_finish_meta(struct kk_device *dev);
VkResult kk_device_init_lib(struct kk_device *dev);
void kk_device_finish_lib(struct kk_device *dev);
void kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap);
void kk_device_remove_heap_from_residency_set(struct kk_device *dev,
                                              mtl_heap *heap);
void kk_device_add_buffer_to_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer);
void kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                                mtl_buffer *buffer);
void kk_device_make_resources_resident(struct kk_device *dev);

/* Required to create a sampler */
mtl_sampler *kk_sampler_create(struct kk_device *dev,
                               const struct mtl_sampler_packed *packed);
VkResult kk_sampler_heap_add(struct kk_device *dev,
                             struct mtl_sampler_packed desc,
                             struct kk_rc_sampler **out);
void kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc);

enum kk_compile_origin {
   KK_COMPILE_ORIGIN_DRAW,
   KK_COMPILE_ORIGIN_PREWARM,
   KK_COMPILE_ORIGIN_CREATE,
};

extern __THREAD_INITIAL_EXEC enum kk_compile_origin kk_compile_origin;
extern __THREAD_INITIAL_EXEC bool kk_fail_on_compile;
extern __THREAD_INITIAL_EXEC bool kk_compile_required;
extern __THREAD_INITIAL_EXEC blake3_hash kk_variant_parent_id;
extern __THREAD_INITIAL_EXEC bool kk_variant_parent_valid;
extern __THREAD_INITIAL_EXEC blake3_hash kk_variant_family_id;
extern __THREAD_INITIAL_EXEC bool kk_variant_family_valid;

void kk_variant_metal_event(void *data, const char *phase, const uint8_t *key,
                            size_t key_size, int64_t start, int64_t end);
void kk_variant_trace(struct kk_device *dev, enum kk_variant_phase phase,
                      const void *key, size_t key_size,
                      int64_t start, int64_t end);

void kk_log_compile(struct kk_device *dev, const char *kind, int64_t start);
void kk_log_cache_shader(struct kk_device *dev,
                         const struct vk_shader_cache_trace *trace,
                         const char *outcome, int64_t start, int64_t end);
void kk_log_cache_component(struct kk_physical_device *physical,
                            VkShaderStageFlags stages, const char *name,
                            const blake3_hash component,
                            const blake3_hash state);
void kk_log_cache_pso(struct kk_device *dev, const char *event,
                      const void *key, size_t key_size);

#endif // KK_DEVICE_H
