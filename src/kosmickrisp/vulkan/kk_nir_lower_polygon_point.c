/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "kk_private.h"
#include "kk_shader.h"

#include "nir_builder.h"
#include "util/bitset.h"
#include "util/u_dynarray.h"

#define KK_POINT_RECORDS 3u

struct kk_point_slot {
   nir_io_semantics sem;
   unsigned base;
   unsigned offset;
   unsigned bit_size;
   unsigned num_components;
   uint8_t mask;
   nir_alu_type types[4];
   nir_variable *records[KK_POINT_RECORDS];
};

struct kk_point_state {
   BITSET_DECLARE(written, NUM_TOTAL_VARYING_SLOTS);
   struct kk_point_slot slots[NUM_TOTAL_VARYING_SLOTS];
   nir_variable *count;
   bool unsupported;
};

static bool
gather_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct kk_point_state *state = data;

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *value = intr->src[0].ssa;
   if (!nir_src_is_const(intr->src[1]) || sem.gs_streams ||
       value->num_components != 1 ||
       (value->bit_size != 16 && value->bit_size != 32)) {
      state->unsupported = true;
      return false;
   }

   unsigned offset = nir_src_as_uint(intr->src[1]);
   unsigned location = sem.location + offset;
   unsigned component = nir_intrinsic_component(intr);
   if (location >= NUM_TOTAL_VARYING_SLOTS || component >= 4) {
      state->unsupported = true;
      return false;
   }

   struct kk_point_slot *slot = &state->slots[location];
   if (!BITSET_TEST(state->written, location)) {
      BITSET_SET(state->written, location);
      slot->sem = sem;
      slot->base = nir_intrinsic_base(intr);
      slot->offset = offset;
      slot->bit_size = value->bit_size;
   } else if (slot->bit_size != value->bit_size) {
      state->unsupported = true;
      return false;
   }

   slot->num_components = MAX2(slot->num_components, component + 1);
   slot->mask |= BITFIELD_BIT(component);
   slot->types[component] = nir_intrinsic_src_type(intr);
   return false;
}

static void
create_records(nir_function_impl *impl, struct kk_point_state *state)
{
   unsigned location;
   BITSET_FOREACH_SET(location, state->written, NUM_TOTAL_VARYING_SLOTS) {
      struct kk_point_slot *slot = &state->slots[location];
      const struct glsl_type *type = glsl_vector_type(
         slot->bit_size == 16 ? GLSL_TYPE_UINT16 : GLSL_TYPE_UINT,
         slot->num_components);
      for (unsigned i = 0; i < KK_POINT_RECORDS; ++i)
         slot->records[i] = nir_local_variable_create(impl, type, NULL);
   }

   state->count = nir_local_variable_create(impl, glsl_uint_type(), NULL);
}

static nir_def *
load_record(nir_builder *b, const struct kk_point_slot *slot, unsigned record)
{
   return nir_load_var(b, slot->records[record]);
}

static void
store_slot_output(nir_builder *b, const struct kk_point_slot *slot,
                  unsigned component, nir_def *value)
{
   nir_store_output(b, value, nir_imm_int(b, slot->offset),
                    .base = slot->base, .component = component,
                    .src_type = slot->types[component],
                    .io_semantics = slot->sem);
}

static nir_def *
corner_sign(nir_builder *b, nir_def *ccw, float when_ccw, float otherwise)
{
   if (when_ccw == otherwise)
      return nir_imm_float(b, when_ccw);

   return nir_bcsel(b, ccw, nir_imm_float(b, when_ccw),
                    nir_imm_float(b, otherwise));
}

static nir_def *
point_value(nir_builder *b, const struct kk_point_slot *slot,
            unsigned location, unsigned record, nir_def *flat_mask,
            nir_def *provoking_last)
{
   nir_def *value = load_record(b, slot, record);
   if (location >= 64)
      return value;

   nir_def *provoking =
      nir_bcsel(b, provoking_last, load_record(b, slot, KK_POINT_RECORDS - 1u),
                load_record(b, slot, 0));
   nir_def *flat =
      nir_ine_imm(b, nir_iand_imm(b, flat_mask, BITFIELD64_BIT(location)), 0);
   return nir_bcsel(b, flat, provoking, value);
}

static void
emit_point(nir_builder *b, struct kk_point_state *state, unsigned record,
           nir_def *ccw, nir_def *flat_mask, nir_def *provoking_last)
{
   const struct kk_point_slot *pos_slot = &state->slots[VARYING_SLOT_POS];
   nir_def *pos = load_record(b, pos_slot, record);
   nir_def *x = nir_channel(b, pos, 0);
   nir_def *y = nir_channel(b, pos, 1);
   nir_def *w = nir_channel(b, pos, 3);

   nir_def *inside = nir_iand(
      b, nir_flt(b, nir_imm_float(b, 0.0f), w),
      nir_iand(b, nir_fge(b, w, nir_fabs(b, x)),
               nir_fge(b, w, nir_fabs(b, y))));

   nir_def *viewport = nir_imm_int(b, 0);
   if (BITSET_TEST(state->written, VARYING_SLOT_VIEWPORT)) {
      const struct kk_point_slot *slot =
         &state->slots[VARYING_SLOT_VIEWPORT];
      viewport = nir_umin(b, nir_channel(b, load_record(b, slot, record), 0),
                          nir_imm_int(b, KK_MAX_VIEWPORTS - 1u));
   }

   nir_def *inv_size = nir_load_viewport_inv_size_kk(b, 2, 32, viewport);
   nir_def *dx = nir_fmul(b, nir_channel(b, inv_size, 0), w);
   nir_def *dy = nir_fmul(b, nir_channel(b, inv_size, 1), w);
   dx = nir_bcsel(b, inside, dx, nir_imm_float(b, 0.0f));
   dy = nir_bcsel(b, inside, dy, nir_imm_float(b, 0.0f));

   nir_def *signs[4][2] = {
      {corner_sign(b, ccw, -1.0f, -1.0f), corner_sign(b, ccw, -1.0f, -1.0f)},
      {corner_sign(b, ccw, 1.0f, -1.0f), corner_sign(b, ccw, -1.0f, 1.0f)},
      {corner_sign(b, ccw, -1.0f, 1.0f), corner_sign(b, ccw, 1.0f, -1.0f)},
      {corner_sign(b, ccw, 1.0f, 1.0f), corner_sign(b, ccw, 1.0f, 1.0f)},
   };

   nir_def *values[NUM_TOTAL_VARYING_SLOTS] = {NULL};
   unsigned location;
   BITSET_FOREACH_SET(location, state->written, NUM_TOTAL_VARYING_SLOTS) {
      if (location == VARYING_SLOT_POS || location == VARYING_SLOT_PSIZ)
         continue;

      values[location] = point_value(b, &state->slots[location], location,
                                     record, flat_mask, provoking_last);
   }

   for (unsigned corner = 0; corner < 4; ++corner) {
      BITSET_FOREACH_SET(location, state->written, NUM_TOTAL_VARYING_SLOTS) {
         if (!values[location])
            continue;

         const struct kk_point_slot *slot = &state->slots[location];
         u_foreach_bit(c, slot->mask)
            store_slot_output(b, slot, c, nir_channel(b, values[location], c));
      }

      nir_def *corner_pos[4] = {
         nir_ffma(b, signs[corner][0], dx, x),
         nir_ffma(b, signs[corner][1], dy, y),
         nir_channel(b, pos, 2),
         w,
      };
      for (unsigned c = 0; c < 4; ++c)
         store_slot_output(b, pos_slot, c, corner_pos[c]);

      nir_emit_vertex(b, 0);
   }

   nir_end_primitive(b, 0);
}

static nir_def *
triangle_ccw(nir_builder *b, struct kk_point_state *state, nir_def *odd)
{
   const struct kk_point_slot *pos_slot = &state->slots[VARYING_SLOT_POS];
   nir_def *x[KK_POINT_RECORDS], *y[KK_POINT_RECORDS], *w[KK_POINT_RECORDS];
   for (unsigned i = 0; i < KK_POINT_RECORDS; ++i) {
      nir_def *pos = load_record(b, pos_slot, i);
      x[i] = nir_channel(b, pos, 0);
      y[i] = nir_channel(b, pos, 1);
      w[i] = nir_channel(b, pos, 3);
   }

   nir_def *c0 = nir_fsub(b, nir_fmul(b, y[1], w[2]), nir_fmul(b, y[2], w[1]));
   nir_def *c1 = nir_fsub(b, nir_fmul(b, y[0], w[2]), nir_fmul(b, y[2], w[0]));
   nir_def *c2 = nir_fsub(b, nir_fmul(b, y[0], w[1]), nir_fmul(b, y[1], w[0]));
   nir_def *det = nir_fadd(
      b, nir_fsub(b, nir_fmul(b, x[0], c0), nir_fmul(b, x[1], c1)),
      nir_fmul(b, x[2], c2));
   det = nir_bcsel(b, odd, nir_fneg(b, det), det);

   return nir_fge_imm(b, det, 0.0f);
}

static void
lower_emit_vertex(nir_builder *b, struct kk_point_state *state)
{
   nir_def *count = nir_load_var(b, state->count);

   nir_push_if(b, nir_uge_imm(b, count, 2));
   {
      nir_def *odd = nir_i2b(b, nir_iand_imm(b, count, 1));
      nir_def *ccw = triangle_ccw(b, state, odd);
      nir_def *flat_mask = nir_load_flat_mask(b);
      nir_def *provoking_last = nir_i2b(b, nir_load_provoking_last(b));
      for (unsigned i = 0; i < KK_POINT_RECORDS; ++i)
         emit_point(b, state, i, ccw, flat_mask, provoking_last);
   }
   nir_pop_if(b, NULL);

   unsigned location;
   BITSET_FOREACH_SET(location, state->written, NUM_TOTAL_VARYING_SLOTS) {
      const struct kk_point_slot *slot = &state->slots[location];
      for (unsigned i = 0; i + 1 < KK_POINT_RECORDS; ++i)
         nir_store_var(b, slot->records[i], load_record(b, slot, i + 1),
                       nir_component_mask(slot->num_components));
   }

   nir_store_var(b, state->count, nir_iadd_imm(b, count, 1), 0x1);
}

static void
lower_store_output(nir_builder *b, struct kk_point_state *state,
                   nir_intrinsic_instr *intr)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned location = sem.location + nir_src_as_uint(intr->src[1]);
   unsigned component = nir_intrinsic_component(intr);
   const struct kk_point_slot *slot = &state->slots[location];
   nir_def *value = nir_vector_insert_imm(
      b, nir_undef(b, slot->num_components, slot->bit_size), intr->src[0].ssa,
      component);

   nir_store_var(b, slot->records[KK_POINT_RECORDS - 1u], value,
                 BITFIELD_BIT(component));
}

bool
kk_nir_lower_gs_polygon_point(nir_shader *gs)
{
   assert(gs->info.stage == MESA_SHADER_GEOMETRY);

   if (gs->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP ||
       gs->info.gs.active_stream_mask != 0x1 ||
       gs->info.gs.vertices_out < 3)
      return false;

   NIR_PASS(_, gs, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   NIR_PASS(_, gs, nir_opt_constant_folding);

   struct kk_point_state *state = calloc(1, sizeof(*state));
   if (!state)
      return false;

   nir_shader_intrinsics_pass(gs, gather_output, nir_metadata_all, state);
   if (state->unsupported || !BITSET_TEST(state->written, VARYING_SLOT_POS) ||
       state->slots[VARYING_SLOT_POS].bit_size != 32 ||
       state->slots[VARYING_SLOT_POS].num_components != 4) {
      free(state);
      return false;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   create_records(impl, state);

   struct util_dynarray intrinsics;
   util_dynarray_init(&intrinsics, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_output ||
             intr->intrinsic == nir_intrinsic_emit_vertex ||
             intr->intrinsic == nir_intrinsic_end_primitive)
            util_dynarray_append(&intrinsics, intr);
      }
   }

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, state->count, nir_imm_int(&b, 0), 0x1);

   util_dynarray_foreach(&intrinsics, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      b.cursor = nir_before_instr(&intr->instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_store_output:
         lower_store_output(&b, state, intr);
         break;
      case nir_intrinsic_emit_vertex:
         lower_emit_vertex(&b, state);
         break;
      case nir_intrinsic_end_primitive:
         nir_store_var(&b, state->count, nir_imm_int(&b, 0), 0x1);
         break;
      default:
         UNREACHABLE("collected above");
      }

      nir_instr_remove(&intr->instr);
   }

   util_dynarray_fini(&intrinsics);
   free(state);

   gs->info.gs.vertices_out = 12u * (gs->info.gs.vertices_out - 2u);
   nir_progress(true, impl, nir_metadata_none);

   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, gs, nir_opt_constant_folding);
      NIR_PASS(progress, gs, nir_opt_dead_cf);
      NIR_PASS(progress, gs, nir_opt_cse);
      NIR_PASS(progress, gs, nir_opt_dce);
   } while (progress);
   nir_shader_gather_info(gs, impl);
   return true;
}
