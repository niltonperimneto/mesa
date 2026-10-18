/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_nir_lower_vbo.h"

#include "kk_cmd_buffer.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/bitset.h"
#include "util/u_math.h"
#include "shader_enums.h"

struct dynamic_ctx {
   bool robustness2;
   bool use_helpers;
};

struct ctx {
   struct kk_attribute *attribs;
   bool requires_robustness2;
};

static bool
is_rgb10_a2(const struct util_format_description *desc)
{
   return desc->channel[0].shift == 0 && desc->channel[0].size == 10 &&
          desc->channel[1].shift == 10 && desc->channel[1].size == 10 &&
          desc->channel[2].shift == 20 && desc->channel[2].size == 10 &&
          desc->channel[3].shift == 30 && desc->channel[3].size == 2;
}

static bool
is_rg11_b10(const struct util_format_description *desc)
{
   return desc->channel[0].shift == 0 && desc->channel[0].size == 11 &&
          desc->channel[1].shift == 11 && desc->channel[1].size == 11 &&
          desc->channel[2].shift == 22 && desc->channel[2].size == 10;
}

static enum pipe_format
kk_vbo_internal_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);

   /* RGB10A2 and RG11B10 require loading as uint and then unpack */
   if (is_rgb10_a2(desc) || is_rg11_b10(desc))
      return PIPE_FORMAT_R32_UINT;

   /* R11G11B10F is native and special */
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return format;

   /* No other non-array formats handled */
   if (!desc->is_array)
      return PIPE_FORMAT_NONE;

   /* Otherwise look at one (any) channel */
   int idx = util_format_get_first_non_void_channel(format);
   if (idx < 0)
      return PIPE_FORMAT_NONE;

   /* We only handle RGB formats (we could do SRGB if we wanted though?) */
   if ((desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB) ||
       (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN))
      return PIPE_FORMAT_NONE;

   /* We have native 8-bit and 16-bit normalized formats */
   struct util_format_channel_description chan = desc->channel[idx];

   /* Otherwise map to the corresponding integer format */
   switch (chan.size) {
   case 32:
      return PIPE_FORMAT_R32_UINT;
   case 16:
      return PIPE_FORMAT_R16_UINT;
   case 8:
      return PIPE_FORMAT_R8_UINT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

bool
kk_vbo_supports_format(enum pipe_format format)
{
   return kk_vbo_internal_format(format) != PIPE_FORMAT_NONE;
}

static nir_def *
apply_swizzle_channel(nir_builder *b, nir_def *vec, unsigned swizzle,
                      bool is_int)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
      return nir_channel(b, vec, 0);
   case PIPE_SWIZZLE_Y:
      return nir_channel(b, vec, 1);
   case PIPE_SWIZZLE_Z:
      return nir_channel(b, vec, 2);
   case PIPE_SWIZZLE_W:
      return nir_channel(b, vec, 3);
   case PIPE_SWIZZLE_0:
      return nir_imm_intN_t(b, 0, vec->bit_size);
   case PIPE_SWIZZLE_1:
      return is_int ? nir_imm_intN_t(b, 1, vec->bit_size)
                    : nir_imm_floatN_t(b, 1.0, vec->bit_size);
   default:
      UNREACHABLE("Invalid swizzle channel");
   }
}

static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct ctx *ctx = data;
   struct kk_attribute *attribs = ctx->attribs;
   b->cursor = nir_instr_remove(&intr->instr);

   nir_src *offset_src = nir_get_io_offset_src(intr);
   assert(nir_src_is_const(*offset_src) && "no attribute indirects");
   unsigned index = nir_intrinsic_base(intr) + nir_src_as_uint(*offset_src);

   struct kk_attribute attrib = attribs[index];

   const struct util_format_description *desc =
      util_format_description(attrib.format);
   int chan = util_format_get_first_non_void_channel(attrib.format);
   assert(chan >= 0);

   bool is_float = desc->channel[chan].type == UTIL_FORMAT_TYPE_FLOAT;
   bool is_unsigned = desc->channel[chan].type == UTIL_FORMAT_TYPE_UNSIGNED;
   bool is_signed = desc->channel[chan].type == UTIL_FORMAT_TYPE_SIGNED;
   bool is_fixed = desc->channel[chan].type == UTIL_FORMAT_TYPE_FIXED;
   bool is_int = util_format_is_pure_integer(attrib.format);

   assert((is_float ^ is_unsigned ^ is_signed ^ is_fixed) && "Invalid format");

   enum pipe_format interchange_format = kk_vbo_internal_format(attrib.format);
   assert(interchange_format != PIPE_FORMAT_NONE);

   unsigned interchange_align = util_format_get_blocksize(interchange_format);
   unsigned interchange_comps = util_format_get_nr_components(attrib.format);

   /* In the hardware, uint formats zero-extend and float formats convert.
    * However, non-uint formats using a uint interchange format shouldn't be
    * zero extended.
    */
   unsigned interchange_register_size =
      util_format_is_pure_uint(interchange_format) ? (interchange_align * 8)
                                                   : intr->def.bit_size;

   /* Non-UNORM R10G10B10A2 loaded as a scalar and unpacked */
   if (interchange_format == PIPE_FORMAT_R32_UINT && !desc->is_array)
      interchange_comps = 1;

   /* Calculate the element to fetch the vertex for. Divide the instance ID by
    * the divisor for per-instance data. Divisor=0 specifies per-vertex data.
    */
   nir_def *el;
   if (attrib.instanced) {
      if (attrib.divisor > 0) {
         el = nir_udiv_imm(b, nir_load_instance_id(b), attrib.divisor);
         BITSET_SET(b->shader->info.system_values_read,
                    SYSTEM_VALUE_INSTANCE_ID);
      } else
         el = nir_imm_int(b, 0);

      el = nir_iadd(b, el, nir_load_base_instance(b));

      BITSET_SET(b->shader->info.system_values_read,
                 SYSTEM_VALUE_BASE_INSTANCE);
   } else {
      el = nir_load_vertex_id(b);
      BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_VERTEX_ID);
   }

   /* Load the pointer of the buffer from the argument buffer */
   nir_def *argbuf = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   uint64_t attrib_base_offset =
      offsetof(struct kk_root_descriptor_table, draw.attrib_base[index]);
   nir_def *base = nir_load_global_constant(
      b, 1, 64, nir_iadd_imm(b, argbuf, attrib_base_offset));

   uint64_t buffer_stride_offset = offsetof(
      struct kk_root_descriptor_table, draw.buffer_strides[attrib.binding]);
   nir_def *stride = nir_load_global_constant(
      b, 1, 32, nir_iadd_imm(b, argbuf, buffer_stride_offset));
   nir_def *stride_offset_el =
      nir_imul(b, el, nir_udiv_imm(b, stride, interchange_align));

   /* Load the raw vector */
   nir_def *memory = nir_load_constant_agx(
      b, interchange_comps, interchange_register_size, base, stride_offset_el,
      .format = interchange_format, .base = 0u);

   if (ctx->requires_robustness2) {
      uint64_t attrib_clamp_offset =
         offsetof(struct kk_root_descriptor_table, draw.attrib_clamps[index]);
      nir_def *bounds = nir_load_global_constant(
         b, 1, 32, nir_iadd_imm(b, argbuf, attrib_clamp_offset));
      nir_def *oob = nir_ult(b, bounds, el);

      /* Produce zero for out-of-bounds access */
      nir_def *zero = nir_imm_zero(b, memory->num_components, memory->bit_size);
      memory = nir_bcsel(b, oob, zero, memory);
   }

   unsigned dest_size = intr->def.bit_size;
   unsigned bits[] = {desc->channel[chan].size, desc->channel[chan].size,
                      desc->channel[chan].size, desc->channel[chan].size};

   /* Unpack non-native formats */
   if (is_rg11_b10(desc)) {
      memory = nir_format_unpack_11f11f10f(b, memory);
   } else if (is_rgb10_a2(desc)) {
      bits[0] = 10;
      bits[1] = 10;
      bits[2] = 10;
      bits[3] = 2;
      if (is_signed)
         memory = nir_format_unpack_sint(b, memory, bits, 4);
      else
         memory = nir_format_unpack_uint(b, memory, bits, 4);
   }

   if (desc->channel[chan].normalized) {
      if (is_signed)
         memory = nir_format_snorm_to_float(b, memory, bits);
      else
         memory = nir_format_unorm_to_float(b, memory, bits);
   } else if (desc->channel[chan].pure_integer) {
      if (is_signed)
         memory = nir_i2iN(b, memory, dest_size);
      else
         memory = nir_u2uN(b, memory, dest_size);
   } else {
      if (is_unsigned)
         memory = nir_u2fN(b, memory, dest_size);
      else if (is_signed || is_fixed)
         memory = nir_i2fN(b, memory, dest_size);
      else
         memory = nir_f2fN(b, memory, dest_size);

      /* 16.16 fixed-point weirdo GL formats need to be scaled */
      if (is_fixed) {
         assert(desc->is_array && desc->channel[chan].size == 32);
         assert(dest_size == 32 && "overflow if smaller");
         memory = nir_fmul_imm(b, memory, 1.0 / 65536.0);
      }
   }

   /* We now have a properly formatted vector of the components in memory. Apply
    * the format swizzle forwards to trim/pad/reorder as needed.
    */
   nir_def *channels[4] = {NULL};

   for (unsigned i = 0; i < intr->num_components; ++i) {
      unsigned c = nir_intrinsic_component(intr) + i;
      channels[i] = apply_swizzle_channel(b, memory, desc->swizzle[c], is_int);
   }

   nir_def *logical = nir_vec(b, channels, intr->num_components);
   nir_def_rewrite_uses(&intr->def, logical);
   return true;
}

bool
kk_nir_lower_vbo(nir_shader *nir, struct kk_attribute *attribs,
                 bool robustness2)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX);

   struct ctx ctx = {
      .attribs = attribs,
      .requires_robustness2 = robustness2,
   };
   return nir_shader_intrinsics_pass(nir, pass, nir_metadata_control_flow,
                                     &ctx);
}

uint32_t
kk_vbo_encode_format(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE || !kk_vbo_supports_format(format))
      return 0;

   const struct util_format_description *desc = util_format_description(format);
   int idx = util_format_get_first_non_void_channel(format);
   if (idx < 0)
      return 0;

   struct util_format_channel_description chan = desc->channel[idx];
   enum kk_vbo_type type;
   switch (chan.type) {
   case UTIL_FORMAT_TYPE_UNSIGNED:
      type = chan.normalized     ? KK_VBO_TYPE_UNORM
             : chan.pure_integer ? KK_VBO_TYPE_UINT
                                 : KK_VBO_TYPE_USCALED;
      break;
   case UTIL_FORMAT_TYPE_SIGNED:
      type = chan.normalized     ? KK_VBO_TYPE_SNORM
             : chan.pure_integer ? KK_VBO_TYPE_SINT
                                 : KK_VBO_TYPE_SSCALED;
      break;
   case UTIL_FORMAT_TYPE_FLOAT:
      type = KK_VBO_TYPE_SFLOAT;
      break;
   default:
      return 0;
   }

   uint32_t fmt = KK_VBO_FMT_VALID | (type << KK_VBO_FMT_TYPE_SHIFT);

   if (is_rg11_b10(desc)) {
      fmt |= KK_VBO_FMT_RG11B10F | (2u << KK_VBO_FMT_SIZE_SHIFT);
   } else if (is_rgb10_a2(desc)) {
      fmt |= KK_VBO_FMT_RGB10A2 | (2u << KK_VBO_FMT_SIZE_SHIFT);
   } else {
      fmt |= util_logbase2(chan.size / 8) << KK_VBO_FMT_SIZE_SHIFT;
      fmt |= (desc->nr_channels - 1) << KK_VBO_FMT_COUNT_SHIFT;
   }

   for (unsigned c = 0; c < 4; ++c) {
      assert(desc->swizzle[c] <= PIPE_SWIZZLE_1);
      fmt |= (uint32_t)desc->swizzle[c] << (KK_VBO_FMT_SWIZZLE_SHIFT + c * 3);
   }

   return fmt;
}

static nir_def *
fmt_field(nir_builder *b, nir_def *fmt, unsigned shift, unsigned mask)
{
   return nir_iand_imm(b, nir_ushr_imm(b, fmt, shift), mask);
}

static nir_def *
load_root(nir_builder *b, nir_def *argbuf, unsigned bit_size, uint64_t offset)
{
   return nir_load_global_constant(b, 1, bit_size,
                                   nir_iadd_imm(b, argbuf, offset));
}

static nir_def *
fetch_component(nir_builder *b, nir_def *addr, nir_def *size_log2, unsigned i)
{
   nir_def *v32, *v16, *v8;

   nir_push_if(b, nir_ieq_imm(b, size_log2, 2));
   {
      v32 = nir_load_global_constant(b, 1, 32, nir_iadd_imm(b, addr, i * 4));
   }
   nir_push_else(b, NULL);
   {
      nir_push_if(b, nir_ieq_imm(b, size_log2, 1));
      {
         v16 = nir_u2u32(b, nir_load_global_constant(
                               b, 1, 16, nir_iadd_imm(b, addr, i * 2)));
      }
      nir_push_else(b, NULL);
      {
         v8 = nir_u2u32(
            b, nir_load_global_constant(b, 1, 8, nir_iadd_imm(b, addr, i)));
      }
      nir_pop_if(b, NULL);
      v16 = nir_if_phi(b, v16, v8);
   }
   nir_pop_if(b, NULL);

   return nir_if_phi(b, v32, v16);
}

static nir_def *
convert_float(nir_builder *b, nir_def *fmt, nir_def *size_log2, nir_def *raw,
              nir_def *u, nir_def *s)
{
   static const unsigned bits8[4] = {8, 8, 8, 8};
   static const unsigned bits16[4] = {16, 16, 16, 16};
   static const unsigned bits_rgb10a2[4] = {10, 10, 10, 2};

   nir_def *is8 = nir_ieq_imm(b, size_log2, 0);
   nir_def *is16 = nir_ieq_imm(b, size_log2, 1);
   nir_def *rgb10a2 = nir_test_mask(b, fmt, KK_VBO_FMT_RGB10A2);
   nir_def *rg11b10 = nir_test_mask(b, fmt, KK_VBO_FMT_RG11B10F);

   nir_def *pu = nir_format_unpack_uint(b, raw, bits_rgb10a2, 4);
   nir_def *ps = nir_format_unpack_sint(b, raw, bits_rgb10a2, 4);

   nir_def *unorm =
      nir_bcsel(b, rgb10a2, nir_format_unorm_to_float(b, pu, bits_rgb10a2),
                nir_bcsel(b, is8, nir_format_unorm_to_float(b, u, bits8),
                          nir_format_unorm_to_float(b, u, bits16)));
   nir_def *snorm =
      nir_bcsel(b, rgb10a2, nir_format_snorm_to_float(b, ps, bits_rgb10a2),
                nir_bcsel(b, is8, nir_format_snorm_to_float(b, s, bits8),
                          nir_format_snorm_to_float(b, s, bits16)));
   nir_def *uscaled = nir_u2f32(b, nir_bcsel(b, rgb10a2, pu, u));
   nir_def *sscaled = nir_i2f32(b, nir_bcsel(b, rgb10a2, ps, s));

   nir_def *rg11b10_val =
      nir_pad_vector_imm_int(b, nir_format_unpack_11f11f10f(b, raw), 0, 4);
   nir_def *sfloat =
      nir_bcsel(b, rg11b10, rg11b10_val,
                nir_bcsel(b, is16, nir_f2f32(b, nir_u2u16(b, u)), u));

   nir_def *type =
      fmt_field(b, fmt, KK_VBO_FMT_TYPE_SHIFT, KK_VBO_FMT_TYPE_MASK);
   nir_def *v = sfloat;
   v = nir_bcsel(b, nir_ieq_imm(b, type, KK_VBO_TYPE_UNORM), unorm, v);
   v = nir_bcsel(b, nir_ieq_imm(b, type, KK_VBO_TYPE_SNORM), snorm, v);
   v = nir_bcsel(b, nir_ieq_imm(b, type, KK_VBO_TYPE_USCALED), uscaled, v);
   v = nir_bcsel(b, nir_ieq_imm(b, type, KK_VBO_TYPE_SSCALED), sscaled, v);
   return v;
}

static nir_def *
convert_int(nir_builder *b, nir_def *fmt, nir_def *raw, nir_def *u, nir_def *s)
{
   static const unsigned bits_rgb10a2[4] = {10, 10, 10, 2};

   nir_def *rgb10a2 = nir_test_mask(b, fmt, KK_VBO_FMT_RGB10A2);
   u =
      nir_bcsel(b, rgb10a2, nir_format_unpack_uint(b, raw, bits_rgb10a2, 4), u);
   s =
      nir_bcsel(b, rgb10a2, nir_format_unpack_sint(b, raw, bits_rgb10a2, 4), s);

   nir_def *type =
      fmt_field(b, fmt, KK_VBO_FMT_TYPE_SHIFT, KK_VBO_FMT_TYPE_MASK);
   return nir_bcsel(b, nir_ieq_imm(b, type, KK_VBO_TYPE_SINT), s, u);
}

static bool
pass_dynamic(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct dynamic_ctx *ctx = data;
   bool robustness2 = ctx->robustness2;
   b->cursor = nir_instr_remove(&intr->instr);

   nir_src *offset_src = nir_get_io_offset_src(intr);
   assert(nir_src_is_const(*offset_src) && "no attribute indirects");
   unsigned index = nir_intrinsic_base(intr) + nir_src_as_uint(*offset_src);

   nir_alu_type base_type =
      nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));
   bool is_int = base_type == nir_type_int || base_type == nir_type_uint;

   nir_def *argbuf = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   nir_def *fmt = load_root(
      b, argbuf, 32,
      offsetof(struct kk_root_descriptor_table, draw.attrib_format[index]));
   nir_def *divisor = load_root(
      b, argbuf, 32,
      offsetof(struct kk_root_descriptor_table, draw.attrib_divisor[index]));
   nir_def *stride = load_root(
      b, argbuf, 32,
      offsetof(struct kk_root_descriptor_table, draw.attrib_stride[index]));
   nir_def *base = load_root(
      b, argbuf, 64,
      offsetof(struct kk_root_descriptor_table, draw.attrib_base[index]));

   nir_def *instance_el = nir_bcsel(
      b, nir_ieq_imm(b, divisor, 0), nir_imm_int(b, 0),
      nir_udiv(b, nir_load_instance_id(b), nir_umax_imm(b, divisor, 1)));
   instance_el = nir_iadd(b, instance_el, nir_load_base_instance(b));
   nir_def *el = nir_bcsel(b, nir_test_mask(b, fmt, KK_VBO_FMT_INSTANCED),
                           instance_el, nir_load_vertex_id(b));

   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_INSTANCE_ID);
   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_BASE_INSTANCE);
   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_VERTEX_ID);

   nir_def *valid = nir_test_mask(b, fmt, KK_VBO_FMT_VALID);
   nir_def *fetch = valid;
   if (robustness2) {
      nir_def *bounds = load_root(
         b, argbuf, 32,
         offsetof(struct kk_root_descriptor_table, draw.attrib_clamps[index]));
      fetch = nir_iand(b, fetch, nir_uge(b, bounds, el));
   }

   nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, el, stride)));
   if (ctx->use_helpers) {
      nir_alu_type type = is_int ? base_type | 32 : nir_type_float32;
      nir_def *memory = nir_load_vbo_kk(b, addr, fmt, fetch, .dest_type = type);
      nir_def *channels[4];
      for (unsigned i = 0; i < intr->num_components; i++) {
         nir_def *ch =
            nir_channel(b, memory, nir_intrinsic_component(intr) + i);
         if (intr->def.bit_size != 32)
            ch = is_int ? nir_u2uN(b, ch, intr->def.bit_size)
                        : nir_f2fN(b, ch, intr->def.bit_size);
         channels[i] = ch;
      }
      nir_def_rewrite_uses(&intr->def,
                           nir_vec(b, channels, intr->num_components));
      return true;
   }

   nir_def *size_log2 =
      fmt_field(b, fmt, KK_VBO_FMT_SIZE_SHIFT, KK_VBO_FMT_SIZE_MASK);
   nir_def *count_m1 =
      fmt_field(b, fmt, KK_VBO_FMT_COUNT_SHIFT, KK_VBO_FMT_COUNT_MASK);

   nir_def *zero = nir_imm_int(b, 0);
   nir_def *loaded[4];
   nir_push_if(b, fetch);
   {
      loaded[0] = fetch_component(b, addr, size_log2, 0);
      for (unsigned i = 1; i < 4; ++i) {
         nir_push_if(b, nir_uge_imm(b, count_m1, i));
         nir_def *c = fetch_component(b, addr, size_log2, i);
         nir_pop_if(b, NULL);
         loaded[i] = nir_if_phi(b, c, zero);
      }
   }
   nir_pop_if(b, NULL);

   nir_def *raw[4];
   for (unsigned i = 0; i < 4; ++i)
      raw[i] = nir_if_phi(b, loaded[i], zero);

   nir_def *u = nir_vec(b, raw, 4);
   nir_def *s =
      nir_bcsel(b, nir_ieq_imm(b, size_log2, 0), nir_i2i32(b, nir_u2u8(b, u)),
                nir_bcsel(b, nir_ieq_imm(b, size_log2, 1),
                          nir_i2i32(b, nir_u2u16(b, u)), u));

   nir_def *memory = is_int ? convert_int(b, fmt, raw[0], u, s)
                            : convert_float(b, fmt, size_log2, raw[0], u, s);
   nir_def *zero_out = is_int ? zero : nir_imm_float(b, 0.0);
   nir_def *one_out = is_int ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0);

   nir_def *channels[4] = {NULL};
   for (unsigned i = 0; i < intr->num_components; ++i) {
      unsigned c = nir_intrinsic_component(intr) + i;
      nir_def *swz = fmt_field(b, fmt, KK_VBO_FMT_SWIZZLE_SHIFT + c * 3,
                               KK_VBO_FMT_SWIZZLE_MASK);

      nir_def *ch = nir_channel(b, memory, 0);
      for (unsigned k = 1; k < 4; ++k)
         ch = nir_bcsel(b, nir_ieq_imm(b, swz, PIPE_SWIZZLE_X + k),
                        nir_channel(b, memory, k), ch);
      ch = nir_bcsel(b, nir_ieq_imm(b, swz, PIPE_SWIZZLE_0), zero_out, ch);
      ch = nir_bcsel(b, nir_ieq_imm(b, swz, PIPE_SWIZZLE_1), one_out, ch);
      ch = nir_bcsel(b, valid, ch, c == 3 ? one_out : zero_out);

      if (intr->def.bit_size != 32)
         ch = is_int ? nir_u2uN(b, ch, intr->def.bit_size)
                     : nir_f2fN(b, ch, intr->def.bit_size);
      channels[i] = ch;
   }

   nir_def *logical = nir_vec(b, channels, intr->num_components);
   nir_def_rewrite_uses(&intr->def, logical);
   return true;
}

bool
kk_nir_lower_vbo_dynamic(nir_shader *nir, bool robustness2, bool use_helpers)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX);

   struct dynamic_ctx ctx = {robustness2, use_helpers};
   return nir_shader_intrinsics_pass(nir, pass_dynamic, nir_metadata_none,
                                     &ctx);
}
