#include "kk_shader_nir.h"
#include "nir_serialize.h"

#include <stdlib.h>

bool
kk_shader_nir_store(struct kk_shader_nir *recipe, const nir_shader *nir,
                    const struct vk_vertex_input_state *vi,
                    const struct vk_pipeline_robustness_state *robustness,
                    bool lower_depth_clip)
{
   struct blob blob;
   blob_init(&blob);
   blob_write_uint32(&blob, (vi != NULL) | (lower_depth_clip << 1));
   const struct vk_pipeline_robustness_state zero = {0};
   if (!robustness)
      robustness = &zero;
   blob_write_uint32(&blob, robustness->storage_buffers);
   blob_write_uint32(&blob, robustness->uniform_buffers);
   blob_write_uint32(&blob, robustness->vertex_inputs);
   blob_write_uint32(&blob, robustness->images);
   blob_write_uint32(&blob, robustness->null_uniform_buffer_descriptor);
   blob_write_uint32(&blob, robustness->null_storage_buffer_descriptor);
   if (vi) {
      blob_write_uint32(&blob, vi->bindings_valid);
      blob_write_uint32(&blob, vi->attributes_valid);
      u_foreach_bit(i, vi->bindings_valid) {
         blob_write_uint32(&blob, vi->bindings[i].stride);
         blob_write_uint32(&blob, vi->bindings[i].input_rate);
         blob_write_uint32(&blob, vi->bindings[i].divisor);
      }
      u_foreach_bit(i, vi->attributes_valid) {
         blob_write_uint32(&blob, vi->attributes[i].binding);
         blob_write_uint32(&blob, vi->attributes[i].format);
         blob_write_uint32(&blob, vi->attributes[i].offset);
      }
   }
   nir_serialize(&blob, nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return false;
   }
   blob_finish_get_buffer(&blob, &recipe->data, &recipe->size);
   return true;
}

nir_shader *
kk_shader_nir_load(const struct kk_shader_nir *recipe,
                   const nir_shader_compiler_options *options,
                   struct vk_graphics_pipeline_state *state,
                   struct vk_vertex_input_state *vi,
                   struct vk_pipeline_robustness_state *robustness)
{
   if (!recipe->data || recipe->size <= 7 * sizeof(uint32_t))
      return NULL;
   struct blob_reader blob;
   blob_reader_init(&blob, recipe->data, recipe->size);
   uint32_t flags = blob_read_uint32(&blob);
   if (flags & ~3u)
      return NULL;
   robustness->storage_buffers = blob_read_uint32(&blob);
   robustness->uniform_buffers = blob_read_uint32(&blob);
   robustness->vertex_inputs = blob_read_uint32(&blob);
   robustness->images = blob_read_uint32(&blob);
   uint32_t null_uniform = blob_read_uint32(&blob);
   uint32_t null_storage = blob_read_uint32(&blob);
   if (null_uniform > 1 || null_storage > 1)
      return NULL;
   robustness->null_uniform_buffer_descriptor = null_uniform;
   robustness->null_storage_buffer_descriptor = null_storage;
   if (flags & 1) {
      vi->bindings_valid = blob_read_uint32(&blob);
      vi->attributes_valid = blob_read_uint32(&blob);
      u_foreach_bit(i, vi->bindings_valid) {
         uint32_t stride = blob_read_uint32(&blob);
         uint32_t rate = blob_read_uint32(&blob);
         if (stride > UINT16_MAX || rate > VK_VERTEX_INPUT_RATE_INSTANCE)
            return NULL;
         vi->bindings[i].stride = stride;
         vi->bindings[i].input_rate = rate;
         vi->bindings[i].divisor = blob_read_uint32(&blob);
      }
      u_foreach_bit(i, vi->attributes_valid) {
         uint32_t binding = blob_read_uint32(&blob);
         if (binding >= ARRAY_SIZE(vi->bindings) ||
             !(vi->bindings_valid & BITFIELD_BIT(binding)))
            return NULL;
         vi->attributes[i].binding = binding;
         vi->attributes[i].format = blob_read_uint32(&blob);
         vi->attributes[i].offset = blob_read_uint32(&blob);
      }
      state->vi = vi;
   }
   if (flags & 2) {
      BITSET_SET(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
      BITSET_SET(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLIP_ENABLE);
   }
   if (blob.overrun || blob.current == blob.end)
      return NULL;
   nir_shader *nir = nir_deserialize(NULL, options, &blob);
   if (blob.overrun || blob.current != blob.end) {
      ralloc_free(nir);
      return NULL;
   }
   return nir;
}

void
kk_shader_nir_write(struct blob *blob, const struct kk_shader_nir *recipe)
{
   blob_write_uint64(blob, recipe->size);
   blob_write_bytes(blob, recipe->data, recipe->size);
}

VkResult
kk_shader_nir_read(struct blob_reader *blob, struct kk_shader_nir *recipe)
{
   uint64_t size = blob_read_uint64(blob);
   if (blob->overrun || size <= 7 * sizeof(uint32_t) ||
       size > (size_t)(blob->end - blob->current))
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
   recipe->data = malloc(size);
   if (!recipe->data)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   recipe->size = size;
   blob_copy_bytes(blob, recipe->data, size);
   return VK_SUCCESS;
}
