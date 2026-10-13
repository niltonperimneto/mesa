/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "../kk_private.h"
#include "../kk_shader.h"
#include "nir_builder.h"

#include <stdio.h>

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort(); \
   } \
} while (0)

static const nir_shader_compiler_options options = {0};

static nir_shader *
create_gs(unsigned control, bool writes_memory)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
                                                  &options, "point_test");
   struct poly_passthrough_gs_key key = {
      .prim = MESA_PRIM_TRIANGLES,
      .outputs = VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0),
   };
   key.output_types[VARYING_SLOT_POS] = nir_type_float32;
   key.output_types[VARYING_SLOT_VAR0] = nir_type_float32;
   key.output_components[VARYING_SLOT_POS] = 4;
   key.output_components[VARYING_SLOT_VAR0] = 4;

   if (writes_memory)
      nir_store_global(&b, nir_imm_int(&b, 1), nir_imm_int64(&b, 4096),
                        .align_mul = 4, .write_mask = 1);
   nir_variable *iteration = NULL;
   if (control == 2) {
      iteration = nir_local_variable_create(b.impl, glsl_uint_type(), NULL);
      nir_store_var(&b, iteration, nir_imm_int(&b, 0), 1);
      nir_def *limit = nir_iand_imm(&b, nir_load_primitive_id(&b), 1);
      nir_push_loop(&b);
      nir_push_if(&b, nir_uge(&b, nir_load_var(&b, iteration), limit));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
   }
   if (control == 1)
      nir_push_if(&b, nir_ine_imm(&b, nir_load_primitive_id(&b), 0));
   poly_nir_passthrough_gs(&b, &key);
   if (control == 1)
      nir_pop_if(&b, NULL);
   if (control == 2) {
      nir_store_var(&b, iteration,
                    nir_iadd_imm(&b, nir_load_var(&b, iteration), 1), 1);
      nir_pop_loop(&b, NULL);
   }

   b.shader->info.io_lowered = true;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
test_topology(unsigned control, bool writes_memory)
{
   bool dynamic = control != 0;
   nir_shader *gs = create_gs(control, writes_memory);
   CHECK(kk_nir_lower_gs_polygon_point(gs));
   CHECK(gs->info.writes_memory == writes_memory);

   if (!dynamic) {
      unsigned emits = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(gs);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_emit_vertex) {
               CHECK(block == nir_start_block(impl));
               emits++;
            }
         }
      }
      CHECK(emits == 12);
   }

   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   struct poly_gs_info info;
   CHECK(poly_nir_lower_gs(gs, &count, &rast, &pre_gs, &info, true));
   if (dynamic) {
      CHECK(info.shape == POLY_GS_SHAPE_DYNAMIC_INDEXED);
   } else {
      const uint8_t expected[] = {
         0, 1, 2, 3, UINT8_MAX, 4, 5, 6, 7, UINT8_MAX, 8, 9, 10, 11,
      };
      CHECK(info.shape == POLY_GS_SHAPE_STATIC_INDEXED);
      CHECK(info.max_indices == ARRAY_SIZE(expected));
      CHECK(memcmp(info.topology, expected, sizeof(expected)) == 0);

      bool main_writes = false, rast_writes = false;
      nir_foreach_function_impl(impl, gs) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic)
                  main_writes |= nir_intrinsic_writes_external_memory(
                     nir_instr_as_intrinsic(instr));
            }
         }
      }
      nir_foreach_function_impl(impl, rast) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic)
                  rast_writes |= nir_intrinsic_writes_external_memory(
                     nir_instr_as_intrinsic(instr));
            }
         }
      }
      CHECK(main_writes == writes_memory);
      CHECK(!rast_writes);
   }

   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre_gs);
   ralloc_free(gs);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_topology(0, false);
   test_topology(1, false);
   test_topology(2, false);
   test_topology(0, true);
   glsl_type_singleton_decref();
   puts("polygon point topology tests passed");
   return 0;
}
