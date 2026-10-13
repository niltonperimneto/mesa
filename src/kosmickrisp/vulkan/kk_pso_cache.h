/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_PSO_CACHE_H
#define KK_PSO_CACHE_H 1

#include "kk_private.h"
#include "kk_pso_wait.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/mesa-blake3.h"
#include "util/rwlock.h"
#include "util/simple_mtx.h"
#include "util/u_queue.h"

#include "vk_graphics_state.h"

struct hash_table;
struct kk_cmd_buffer;
struct kk_device;
struct kk_shader;
struct kk_variant_history;

struct kk_pso_blend {
   uint8_t enable;
   uint8_t src_rgb;
   uint8_t dst_rgb;
   uint8_t op_rgb;
   uint8_t src_alpha;
   uint8_t dst_alpha;
   uint8_t op_alpha;
   uint8_t write_mask;
};

struct kk_pso_key {
   blake3_hash rast;
   blake3_hash fs;
   uint32_t rt_formats[MAX_DRAW_BUFFERS];
   uint32_t sample_count;
   uint32_t topology_class;
   uint32_t amplification;
   uint32_t alpha_to_coverage;
   uint32_t alpha_to_one;
   struct kk_pso_blend blend[MAX_DRAW_BUFFERS];
};

static_assert(sizeof(struct kk_pso_key) == 180,
              "kk_pso_key is hashed and serialized");

enum kk_wait_timeout_policy {
   KK_WAIT_TIMEOUT_STUB,
   KK_WAIT_TIMEOUT_BLOCK,
   KK_WAIT_TIMEOUT_STRICT,
};

struct kk_link_state {
   const struct vk_dynamic_graphics_state *dyn;
   struct kk_shader *shaders[MESA_SHADER_COMPUTE];
   enum mesa_prim tess_prim;
   uint32_t view_mask;
   uint32_t color_att_count;
   VkFormat color_formats[MESA_VK_MAX_COLOR_ATTACHMENTS];
   VkFormat depth_format;
};

struct kk_pso_cache {
   mtx_t lock;
   struct u_rwlock index_lock;
   struct hash_table *table;
   struct util_queue prewarm_queue;
   struct util_queue exact_queue;
   struct util_queue draw_queue;
   struct util_queue stub_queue;
   struct hash_table *recipes;
   struct hash_table *families;
   struct kk_variant_history *history;
   cnd_t jobs_done;
   unsigned pending_jobs;
   unsigned exact_jobs;
   unsigned failed_jobs;
   unsigned recipe_count;
   unsigned recipe_limit;
   unsigned family_count;
   unsigned wait_timeout_ms;
   enum kk_wait_timeout_policy timeout_policy;
   bool limited;
};

struct vk_device;
struct vk_shader;
struct vk_graphics_pipeline_state;

void kk_prewarm_pipeline(struct vk_device *device,
                         const struct vk_graphics_pipeline_state *state,
                         uint32_t shader_count,
                         struct vk_shader *const *shaders);

void kk_pso_key_init(struct kk_pso_key *key);

mtl_render_pipeline_state *
kk_new_render_pipeline(struct kk_device *dev, mtl_function_descriptor *vs,
                       mtl_function_descriptor *fs,
                       const struct kk_pso_key *key, bool lookup_only);

bool kk_pso_cache_init(struct kk_pso_cache *cache);
void kk_pso_cache_finish(struct kk_pso_cache *cache);
VkResult kk_pso_prewarm_wait(struct kk_device *dev);
VkBool32 kk_pso_prewarm_ready(struct kk_device *dev);

struct kk_pso_result {
   mtl_render_pipeline_state *pso;
   VkResult status;
};

struct mtl_compile_phase
kk_pso_phase_begin(struct kk_device *dev, const char *operation,
                    const void *key, size_t size);

struct kk_pso_result
kk_pso_cache_get(struct kk_device *dev, const struct kk_pso_key *key,
                 mtl_function_descriptor *vs, mtl_function_descriptor *fs,
                 const struct kk_pso_request *request);

struct kk_pso_result
kk_link_pipeline(struct kk_device *dev, const struct kk_link_state *ls,
                  const struct kk_pso_request *request);

mtl_render_pipeline_state *kk_cmd_link_pipeline(struct kk_cmd_buffer *cmd,
                                                bool *provisional);

void kk_pso_schedule_render(struct kk_device *dev, struct kk_shader *vs);
bool kk_pso_render_pending(struct kk_device *dev, struct kk_shader *vs);
mtl_render_pipeline_state *kk_cmd_acquire_render(struct kk_cmd_buffer *cmd,
                                                 struct kk_shader *vs,
                                                 bool *provisional);

#endif /* KK_PSO_CACHE_H */
