#include "../kk_shader_nir.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static const nir_shader_compiler_options options = {0};

static void
round_trip(bool static_vi, bool depth_clip)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &options,
                                                  "lazy_point");
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .base = VARYING_SLOT_POS, .write_mask = 15,
                    .io_semantics = {.location = VARYING_SLOT_POS});
   nir_shader_gather_info(b.shader, b.impl);
   struct vk_vertex_input_state vi = {
      .bindings_valid = BITFIELD_BIT(3),
      .bindings[3] = {.stride = 24,
                      .input_rate = VK_VERTEX_INPUT_RATE_INSTANCE,
                      .divisor = 7},
      .attributes_valid = BITFIELD_BIT(5),
      .attributes[5] = {.binding = 3,
                        .format = VK_FORMAT_R32G32B32_SFLOAT,
                        .offset = 12},
   };
   struct vk_pipeline_robustness_state robustness = {
      .vertex_inputs =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
      .uniform_buffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT,
      .null_storage_buffer_descriptor = true,
   };
   struct kk_shader_nir recipe = {0}, same = {0};
   CHECK(kk_shader_nir_store(&recipe, b.shader, static_vi ? &vi : NULL,
                             &robustness, depth_clip));
   CHECK(kk_shader_nir_store(&same, b.shader, static_vi ? &vi : NULL,
                             &robustness, depth_clip));
   CHECK(recipe.size == same.size &&
         !memcmp(recipe.data, same.data, recipe.size));
   vi.attributes[5].offset = 99;
   ralloc_free(b.shader);

   struct blob binary;
   blob_init(&binary);
   kk_shader_nir_write(&binary, &recipe);
   struct blob_reader reader;
   blob_reader_init(&reader, binary.data, binary.size);
   struct kk_shader_nir cached = {0};
   CHECK(kk_shader_nir_read(&reader, &cached) == VK_SUCCESS);
   CHECK(reader.current == reader.end);
   struct vk_graphics_pipeline_state state = {0};
   struct vk_vertex_input_state loaded_vi = {0};
   struct vk_pipeline_robustness_state loaded_robustness = {0};
   nir_shader *nir = kk_shader_nir_load(&cached, &options, &state, &loaded_vi,
                                        &loaded_robustness);
   CHECK(nir && nir->info.stage == MESA_SHADER_VERTEX);
   CHECK(nir->info.outputs_written == VARYING_BIT_POS);
   CHECK(!memcmp(&robustness, &loaded_robustness, sizeof(robustness)));
   CHECK((state.vi != NULL) == static_vi);
   if (static_vi) {
      CHECK(loaded_vi.bindings_valid == BITFIELD_BIT(3));
      CHECK(loaded_vi.bindings[3].divisor == 7);
      CHECK(loaded_vi.bindings[3].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE);
      CHECK(loaded_vi.bindings[3].stride == 24);
      CHECK(loaded_vi.attributes_valid == BITFIELD_BIT(5));
      CHECK(loaded_vi.attributes[5].binding == 3);
      CHECK(loaded_vi.attributes[5].format == VK_FORMAT_R32G32B32_SFLOAT);
      CHECK(loaded_vi.attributes[5].offset == 12);
   }
   CHECK(BITSET_TEST(state.dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE) ==
         depth_clip);
   CHECK(BITSET_TEST(state.dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLIP_ENABLE) ==
         depth_clip);
   ralloc_free(nir);
   CHECK(!memcmp(recipe.data, cached.data, recipe.size));

   for (unsigned i = 0; i < binary.size; ++i) {
      struct kk_shader_nir truncated = {0};
      blob_reader_init(&reader, binary.data, i);
      CHECK(kk_shader_nir_read(&reader, &truncated) ==
            VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      CHECK(!truncated.data);
   }
   uint32_t bad_flags = 4;
   memcpy(cached.data, &bad_flags, sizeof(bad_flags));
   CHECK(!kk_shader_nir_load(&cached, &options, &state, &loaded_vi,
                             &loaded_robustness));
   free(cached.data);
   free(same.data);
   free(recipe.data);
   blob_finish(&binary);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned i = 0; i < 4; ++i)
      round_trip(i & 1, i & 2);
   glsl_type_singleton_decref();
   puts("lazy shader NIR recipes passed");
   return 0;
}
