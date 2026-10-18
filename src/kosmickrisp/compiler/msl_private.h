/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/string_buffer.h"
#include "nir.h"

struct io_slot_info {
   nir_alu_type type;
   uint32_t interpolation;
   unsigned num_components;
   bool centroid;
   bool sample;
   bool uses_interpolant;
};

struct msl_tex_group {
   enum glsl_sampler_dim dim;
   nir_alu_type type;
   uint8_t access;
   bool array;
   bool depth;
   bool coherent;
};

enum msl_tex_helper_kind {
   MSL_TEX_HELPER_SAMPLE,
   MSL_TEX_HELPER_GATHER,
   MSL_TEX_HELPER_READ,
   MSL_TEX_HELPER_WRITE,
   MSL_TEX_HELPER_ATOMIC,
   MSL_TEX_HELPER_CMPXCHG,
   MSL_TEX_HELPER_FENCE,
   MSL_TEX_HELPER_SIZE,
   MSL_TEX_HELPER_LEVELS,
   MSL_TEX_HELPER_SAMPLES,
   MSL_TEX_HELPER_LOD_CLAMPED,
   MSL_TEX_HELPER_LOD_UNCLAMPED,
};

enum msl_tex_helper_lod {
   MSL_TEX_HELPER_LOD_NONE,
   MSL_TEX_HELPER_LOD_BIAS,
   MSL_TEX_HELPER_LOD_LEVEL,
   MSL_TEX_HELPER_LOD_GRADIENT,
};

struct msl_tex_helper {
   enum msl_tex_helper_kind kind;
   enum msl_tex_helper_lod lod;
   bool compare;
   bool min_lod;
   uint8_t component;
   nir_atomic_op atomic;
};

#define MSL_TEX_HELPER_MAX_PARAMS 8

struct msl_tex_helper_sig {
   const char *ret;
   const char *params[MSL_TEX_HELPER_MAX_PARAMS];
   unsigned num_params;
};

struct msl_helper_ref {
   struct msl_tex_group group;
   struct msl_tex_helper op;
};

bool msl_tex_group_init(struct msl_tex_group *group,
                        const nir_intrinsic_instr *handle);
uint32_t msl_tex_group_id(const struct msl_tex_group *group);
void msl_print_texture_type(struct _mesa_string_buffer *buf,
                            const struct msl_tex_group *group);
bool msl_tex_helper_valid(const struct msl_tex_group *group,
                          const struct msl_tex_helper *op);
void msl_tex_helper_signature(const struct msl_tex_group *group,
                              const struct msl_tex_helper *op,
                              struct msl_tex_helper_sig *sig);
void msl_print_tex_helper_name(struct _mesa_string_buffer *buf,
                               uint32_t group_id,
                               const struct msl_tex_helper *op);
void msl_print_tex_helper_decl(struct _mesa_string_buffer *buf,
                               const struct msl_tex_group *group,
                               const struct msl_tex_helper *op);
unsigned msl_tex_coord_components(enum glsl_sampler_dim dim, bool read);
const char *msl_texture_dim(enum glsl_sampler_dim dim);
const char *msl_tex_type_name(nir_alu_type type);

struct nir_to_msl_ctx {
   FILE *output;
   struct hash_table *types;
   nir_shader *shader;
   struct _mesa_string_buffer *text;
   uint64_t disabled_workarounds;
   bool uses_per_draw_data;
   bool separate_io;
   bool tex_helpers;
   unsigned short indentlevel;
   struct io_slot_info inputs_info[NUM_TOTAL_VARYING_SLOTS];
   struct io_slot_info outputs_info[NUM_TOTAL_VARYING_SLOTS];
};

#define P_IND(ctx, ...)                                                        \
   do {                                                                        \
      for (unsigned i = 0; i < (ctx)->indentlevel; i++)                        \
         _mesa_string_buffer_append((ctx)->text, "    ");                      \
      _mesa_string_buffer_printf((ctx)->text, __VA_ARGS__);                    \
   } while (0);

#define P(ctx, ...) _mesa_string_buffer_printf((ctx)->text, __VA_ARGS__);

#define P_INDENT(ctx)                                                          \
   do {                                                                        \
      for (unsigned i = 0; i < (ctx)->indentlevel; i++)                        \
         _mesa_string_buffer_append((ctx)->text, "    ");                      \
   } while (0)

/* Perform type inference. The returned value is a
 * map from nir_def* to base type.*/

struct hash_table *msl_infer_types(nir_shader *shader);

const char *msl_type_for_def(struct hash_table *types, nir_def *def);

const char *msl_uint_type(uint8_t bit_size, uint8_t num_components);

const char *msl_type_for_src(struct hash_table *types, nir_src *src);

const char *msl_bitcast_for_src(struct hash_table *types, nir_src *src);

void msl_src_as_const(struct nir_to_msl_ctx *ctx, nir_src *src);

void msl_emit_io_blocks(struct nir_to_msl_ctx *ctx, nir_shader *shader);

void msl_emit_output_var(struct nir_to_msl_ctx *ctx, nir_shader *shader);

void msl_gather_io_info(struct nir_to_msl_ctx *ctx,
                        struct io_slot_info *info_array_input,
                        struct io_slot_info *info_array_output);

void msl_input_name(struct nir_to_msl_ctx *ctx, unsigned location,
                    unsigned component);

void msl_output_name(struct nir_to_msl_ctx *ctx, unsigned location,
                     unsigned component, bool from_input);

uint32_t msl_input_num_components(struct nir_to_msl_ctx *ctx,
                                  uint32_t location);

uint32_t msl_output_num_components(struct nir_to_msl_ctx *ctx,
                                   uint32_t location);

bool msl_is_separate_varying(struct nir_to_msl_ctx *ctx, uint32_t location);

bool msl_src_is_float(struct nir_to_msl_ctx *ctx, nir_src *src);
bool msl_def_is_sampler(struct nir_to_msl_ctx *ctx, nir_def *def);
bool msl_def_is_texture(struct nir_to_msl_ctx *ctx, nir_def *def);

void msl_nir_lower_subgroups(nir_shader *nir);

bool msl_nir_lower_algebraic_late(nir_shader *shader);

bool msl_reg_needs_initializer(nir_intrinsic_instr *reg);
