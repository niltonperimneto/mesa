#ifndef KK_SHADER_NIR_H
#define KK_SHADER_NIR_H

#include "util/blob.h"
#include "nir.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"

struct kk_shader_nir {
   void *data;
   size_t size;
};

bool kk_shader_nir_store(struct kk_shader_nir *recipe, const nir_shader *nir,
                         const struct vk_vertex_input_state *vi,
                         const struct vk_pipeline_robustness_state *robustness,
                         bool lower_depth_clip);
nir_shader *kk_shader_nir_load(const struct kk_shader_nir *recipe,
                               const nir_shader_compiler_options *options,
                               struct vk_graphics_pipeline_state *state,
                               struct vk_vertex_input_state *vi,
                               struct vk_pipeline_robustness_state *robustness);
void kk_shader_nir_write(struct blob *blob, const struct kk_shader_nir *recipe);
VkResult kk_shader_nir_read(struct blob_reader *blob,
                            struct kk_shader_nir *recipe);

#endif
