/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "msl_private.h"
#include "msl_vbo.h"
#include "nir_to_msl.h"

#include "util/ralloc.h"

#define GROUP_DIM_SHIFT    0
#define GROUP_DIM_MASK     0xf
#define GROUP_ARRAY        (1u << 4)
#define GROUP_DEPTH        (1u << 5)
#define GROUP_TYPE_SHIFT   6
#define GROUP_TYPE_MASK    0x7
#define GROUP_ACCESS_SHIFT 9
#define GROUP_ACCESS_MASK  0x3
#define GROUP_COHERENT     (1u << 11)
#define GROUP_TEXTURE_MASK 0xfffu

static const nir_alu_type group_types[] = {
   nir_type_float32, nir_type_float16, nir_type_int32,
   nir_type_uint32,  nir_type_int16,   nir_type_uint16,
};

const char *
msl_texture_dim(enum glsl_sampler_dim dim)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      return "1d";
   case GLSL_SAMPLER_DIM_2D:
      return "2d";
   case GLSL_SAMPLER_DIM_3D:
      return "3d";
   case GLSL_SAMPLER_DIM_CUBE:
      return "cube";
   case GLSL_SAMPLER_DIM_BUF:
      return "_buffer";
   case GLSL_SAMPLER_DIM_MS:
      return "2d_ms";
   default:
      fprintf(stderr, "Bad texture dim %d\n", dim);
      assert(!"Bad texture dimension");
      return "BAD";
   }
}

const char *
msl_tex_type_name(nir_alu_type ty)
{
   switch (ty) {
   case nir_type_int16:
      return "short";
   case nir_type_int32:
      return "int";
   case nir_type_uint16:
      return "ushort";
   case nir_type_uint32:
      return "uint";
   case nir_type_float16:
      return "half";
   case nir_type_float32:
      return "float";
   default:
      return "BAD";
   }
}

static int
group_type_index(nir_alu_type type)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_types); i++) {
      if (group_types[i] == type)
         return i;
   }
   return -1;
}

static bool
group_supported(const struct msl_tex_group *group)
{
   if (group_type_index(group->type) < 0 ||
       (group->access == MSL_ACCESS_SAMPLE && group->coherent))
      return false;

   switch (group->dim) {
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_CUBE:
      return true;
   case GLSL_SAMPLER_DIM_MS:
      return group->access == MSL_ACCESS_SAMPLE ||
             group->access == MSL_ACCESS_READ;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_BUF:
      return !group->array && !group->depth;
   default:
      return false;
   }
}

bool
msl_tex_group_init(struct msl_tex_group *group,
                   const nir_intrinsic_instr *handle)
{
   memset(group, 0, sizeof(*group));
   group->dim = nir_intrinsic_image_dim(handle);
   group->array = nir_intrinsic_image_array(handle);
   if (handle->intrinsic == nir_intrinsic_load_depth_texture_kk) {
      group->depth = true;
      group->type = nir_type_float32;
   } else {
      group->type = nir_intrinsic_dest_type(handle);
      group->access = nir_intrinsic_flags(handle);
      /* TODO_KOSMICKRISP We shouldn't need this line below but it doesn't
       * seem we get the correct access values for what should be device
       * coherent textures from NIR. Example test:
       * dEQP-VK.memory_model.message_passing.ext.u32.coherent.fence_fence
       * .atomicwrite.device.payload_local.image.guard_local.buffer.comp
       * This test declares the texture as devicecoherent, but in NIR it
       * appears as resctrict only with no coherent.
       */
      group->coherent = group->access == MSL_ACCESS_READ_WRITE ||
                        (nir_intrinsic_access(handle) & ACCESS_COHERENT);
   }
   return group_supported(group);
}

uint32_t
msl_tex_group_id(const struct msl_tex_group *group)
{
   return (uint32_t)group->dim << GROUP_DIM_SHIFT |
          (group->array ? GROUP_ARRAY : 0) | (group->depth ? GROUP_DEPTH : 0) |
          (uint32_t)group_type_index(group->type) << GROUP_TYPE_SHIFT |
          (uint32_t)group->access << GROUP_ACCESS_SHIFT |
          (group->coherent ? GROUP_COHERENT : 0);
}

static bool
group_from_id(uint32_t id, struct msl_tex_group *group)
{
   unsigned type = (id >> GROUP_TYPE_SHIFT) & GROUP_TYPE_MASK;
   if ((id & ~GROUP_TEXTURE_MASK) || type >= ARRAY_SIZE(group_types))
      return false;
   *group = (struct msl_tex_group){
      .dim = (id >> GROUP_DIM_SHIFT) & GROUP_DIM_MASK,
      .array = id & GROUP_ARRAY,
      .depth = id & GROUP_DEPTH,
      .type = group_types[type],
      .access = (id >> GROUP_ACCESS_SHIFT) & GROUP_ACCESS_MASK,
      .coherent = id & GROUP_COHERENT,
   };
   if (group->depth &&
       (group->type != nir_type_float32 || group->access || group->coherent))
      return false;
   return group_supported(group) && msl_tex_group_id(group) == id;
}

void
msl_print_texture_type(struct _mesa_string_buffer *buf,
                       const struct msl_tex_group *group)
{
   const char *dim = msl_texture_dim(group->dim);
   const char *arrayed = group->array ? "_array" : "";
   if (group->depth) {
      _mesa_string_buffer_printf(buf, "depth%s%s<float>", dim, arrayed);
      return;
   }

   const char *access = "";
   switch (group->access) {
   case MSL_ACCESS_READ:
      access = ", access::read";
      break;
   case MSL_ACCESS_WRITE:
      access = ", access::write";
      break;
   case MSL_ACCESS_READ_WRITE:
      access = ", access::read_write";
      break;
   }
   _mesa_string_buffer_printf(
      buf, "texture%s%s<%s%s%s>", dim, arrayed, msl_tex_type_name(group->type),
      access, group->coherent ? ", memory_coherence_device" : "");
}

unsigned
msl_tex_coord_components(enum glsl_sampler_dim dim, bool read)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
      return 1;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS:
      return 2;
   case GLSL_SAMPLER_DIM_3D:
      return 3;
   case GLSL_SAMPLER_DIM_CUBE:
      return read ? 0 : 3;
   default:
      return 0;
   }
}

static const char *const float_vec[] = {"float", "float2", "float3", "float4"};
static const char *const uint_vec[] = {"uint", "uint2", "uint3", "uint4"};
static const char *const int_vec[] = {"int", "int2", "int3", "int4"};

static const char *
atomic_method(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd:
      return "atomic_fetch_add";
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
      return "atomic_fetch_min";
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
      return "atomic_fetch_max";
   case nir_atomic_op_iand:
      return "atomic_fetch_and";
   case nir_atomic_op_ior:
      return "atomic_fetch_or";
   case nir_atomic_op_ixor:
      return "atomic_fetch_xor";
   case nir_atomic_op_xchg:
      return "atomic_exchange";
   default:
      return NULL;
   }
}

static const char *
atomic_suffix(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd:
      return "add";
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
      return "min";
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
      return "max";
   case nir_atomic_op_iand:
      return "and";
   case nir_atomic_op_ior:
      return "or";
   case nir_atomic_op_ixor:
      return "xor";
   case nir_atomic_op_xchg:
      return "xchg";
   default:
      return NULL;
   }
}

static bool
group_samples(const struct msl_tex_group *group)
{
   return group->access == MSL_ACCESS_SAMPLE &&
          (group->dim == GLSL_SAMPLER_DIM_2D ||
           group->dim == GLSL_SAMPLER_DIM_3D ||
           group->dim == GLSL_SAMPLER_DIM_CUBE);
}

static bool
group_has_mips(const struct msl_tex_group *group)
{
   return group->dim != GLSL_SAMPLER_DIM_MS &&
          group->dim != GLSL_SAMPLER_DIM_BUF;
}

bool
msl_tex_helper_valid(const struct msl_tex_group *group,
                     const struct msl_tex_helper *op)
{
   bool is_32bit_int =
      group->type == nir_type_int32 || group->type == nir_type_uint32;
   bool cube = group->dim == GLSL_SAMPLER_DIM_CUBE;
   bool ms = group->dim == GLSL_SAMPLER_DIM_MS;

   switch (op->kind) {
   case MSL_TEX_HELPER_SAMPLE:
      if (!group_samples(group) || (op->compare && !group->depth))
         return false;
      if (op->min_lod && op->lod == MSL_TEX_HELPER_LOD_LEVEL)
         return false;
      if (op->offset && group->dim == GLSL_SAMPLER_DIM_CUBE)
         return false;
      return true;
   case MSL_TEX_HELPER_GATHER:
      if (!group_samples(group) || group->dim == GLSL_SAMPLER_DIM_3D ||
          (op->offset && group->dim == GLSL_SAMPLER_DIM_CUBE))
         return false;
      return group->depth ? op->compare : !op->compare && op->component < 4;
   case MSL_TEX_HELPER_READ:
      return !cube && group->access != MSL_ACCESS_WRITE;
   case MSL_TEX_HELPER_WRITE:
      return !cube && !ms && !group->depth &&
             (group->access == MSL_ACCESS_WRITE ||
              group->access == MSL_ACCESS_READ_WRITE);
   case MSL_TEX_HELPER_ATOMIC:
      if (!atomic_method(op->atomic))
         return false;
      FALLTHROUGH;
   case MSL_TEX_HELPER_CMPXCHG:
      return !cube && !ms && is_32bit_int &&
             group->access == MSL_ACCESS_READ_WRITE;
   case MSL_TEX_HELPER_FENCE:
      return group->access == MSL_ACCESS_READ_WRITE;
   case MSL_TEX_HELPER_SIZE:
      return true;
   case MSL_TEX_HELPER_LEVELS:
      return group_has_mips(group);
   case MSL_TEX_HELPER_SAMPLES:
      return ms;
   case MSL_TEX_HELPER_LOD_CLAMPED:
   case MSL_TEX_HELPER_LOD_UNCLAMPED:
      return group_samples(group);
   }
   return false;
}

static const char *
group_vec4(const struct msl_tex_group *group)
{
   switch (group->type) {
   case nir_type_float16:
      return "half4";
   case nir_type_int32:
      return "int4";
   case nir_type_uint32:
      return "uint4";
   case nir_type_int16:
      return "short4";
   case nir_type_uint16:
      return "ushort4";
   default:
      return "float4";
   }
}

static void
add_param(struct msl_tex_helper_sig *sig, const char *type)
{
   assert(sig->num_params < MSL_TEX_HELPER_MAX_PARAMS);
   sig->params[sig->num_params++] = type;
}

static void
add_coord_params(struct msl_tex_helper_sig *sig,
                 const struct msl_tex_group *group, bool read)
{
   unsigned comps = msl_tex_coord_components(group->dim, read);
   add_param(sig, read ? uint_vec[comps - 1] : float_vec[comps - 1]);
   if (group->array)
      add_param(sig, "uint");
}

void
msl_tex_helper_signature(const struct msl_tex_group *group,
                         const struct msl_tex_helper *op,
                         struct msl_tex_helper_sig *sig)
{
   const char *vec4 = group->depth ? "float4" : group_vec4(group);
   const char *texel = group->depth ? "float" : vec4;
   const char *scalar = msl_tex_type_name(group->type);
   unsigned coords = msl_tex_coord_components(group->dim, false);
   bool has_lod = group->dim != GLSL_SAMPLER_DIM_BUF;

   memset(sig, 0, sizeof(*sig));
   switch (op->kind) {
   case MSL_TEX_HELPER_SAMPLE:
      sig->ret = texel;
      add_param(sig, "sampler");
      add_coord_params(sig, group, false);
      if (op->compare)
         add_param(sig, "float");
      if (op->lod == MSL_TEX_HELPER_LOD_GRADIENT) {
         add_param(sig, float_vec[coords - 1]);
         add_param(sig, float_vec[coords - 1]);
      } else if (op->lod != MSL_TEX_HELPER_LOD_NONE) {
         add_param(sig, "float");
      }
      if (op->min_lod)
         add_param(sig, "float");
      if (op->offset)
         add_param(sig, int_vec[coords - 1]);
      break;
   case MSL_TEX_HELPER_GATHER:
      sig->ret = vec4;
      add_param(sig, "sampler");
      add_coord_params(sig, group, false);
      if (op->compare)
         add_param(sig, "float");
      if (op->offset)
         add_param(sig, int_vec[coords - 1]);
      break;
   case MSL_TEX_HELPER_READ:
      sig->ret = texel;
      add_coord_params(sig, group, true);
      if (has_lod)
         add_param(sig, "uint");
      break;
   case MSL_TEX_HELPER_WRITE:
      sig->ret = "void";
      add_param(sig, vec4);
      add_coord_params(sig, group, true);
      if (has_lod)
         add_param(sig, "uint");
      break;
   case MSL_TEX_HELPER_ATOMIC:
      sig->ret = vec4;
      add_coord_params(sig, group, true);
      add_param(sig, scalar);
      break;
   case MSL_TEX_HELPER_CMPXCHG:
      sig->ret = scalar;
      add_coord_params(sig, group, true);
      add_param(sig, scalar);
      add_param(sig, scalar);
      break;
   case MSL_TEX_HELPER_FENCE:
      sig->ret = "void";
      break;
   case MSL_TEX_HELPER_SIZE:
      sig->ret = "uint4";
      add_param(sig, "uint");
      break;
   case MSL_TEX_HELPER_LEVELS:
   case MSL_TEX_HELPER_SAMPLES:
      sig->ret = "uint";
      break;
   case MSL_TEX_HELPER_LOD_CLAMPED:
   case MSL_TEX_HELPER_LOD_UNCLAMPED:
      sig->ret = "float";
      add_param(sig, "sampler");
      add_param(sig, float_vec[coords - 1]);
      break;
   }
}

void
msl_print_tex_helper_name(struct _mesa_string_buffer *buf, uint32_t group_id,
                          const struct msl_tex_helper *op)
{
   _mesa_string_buffer_printf(buf, "kk_t_%03x_", group_id);
   switch (op->kind) {
   case MSL_TEX_HELPER_SAMPLE:
      _mesa_string_buffer_printf(buf, "s%s%s%s%s", op->compare ? "c" : "",
                                 (const char *[]){"", "b", "l", "g"}[op->lod],
                                 op->min_lod ? "m" : "", op->offset ? "o" : "");
      break;
   case MSL_TEX_HELPER_GATHER:
      if (op->compare)
         _mesa_string_buffer_append(buf, "gc");
      else
         _mesa_string_buffer_printf(buf, "g%c", "xyzw"[op->component]);
      if (op->offset)
         _mesa_string_buffer_append(buf, "o");
      break;
   case MSL_TEX_HELPER_READ:
      _mesa_string_buffer_append(buf, "read");
      break;
   case MSL_TEX_HELPER_WRITE:
      _mesa_string_buffer_append(buf, "write");
      break;
   case MSL_TEX_HELPER_ATOMIC:
      _mesa_string_buffer_printf(buf, "a%s", atomic_suffix(op->atomic));
      break;
   case MSL_TEX_HELPER_CMPXCHG:
      _mesa_string_buffer_append(buf, "acmpxchg");
      break;
   case MSL_TEX_HELPER_FENCE:
      _mesa_string_buffer_append(buf, "fence");
      break;
   case MSL_TEX_HELPER_SIZE:
      _mesa_string_buffer_append(buf, "size");
      break;
   case MSL_TEX_HELPER_LEVELS:
      _mesa_string_buffer_append(buf, "levels");
      break;
   case MSL_TEX_HELPER_SAMPLES:
      _mesa_string_buffer_append(buf, "samples");
      break;
   case MSL_TEX_HELPER_LOD_CLAMPED:
      _mesa_string_buffer_append(buf, "lodc");
      break;
   case MSL_TEX_HELPER_LOD_UNCLAMPED:
      _mesa_string_buffer_append(buf, "lodu");
      break;
   }
}

static void
print_prototype(struct _mesa_string_buffer *buf,
                const struct msl_tex_group *group,
                const struct msl_tex_helper *op, bool named)
{
   struct msl_tex_helper_sig sig;
   msl_tex_helper_signature(group, op, &sig);
   _mesa_string_buffer_printf(buf, "%s ", sig.ret);
   msl_print_tex_helper_name(buf, msl_tex_group_id(group), op);
   _mesa_string_buffer_append(buf, named ? "(ulong tex" : "(ulong");
   for (unsigned i = 0; i < sig.num_params; i++) {
      if (named)
         _mesa_string_buffer_printf(buf, ", %s p%u", sig.params[i], i);
      else
         _mesa_string_buffer_printf(buf, ", %s", sig.params[i]);
   }
   _mesa_string_buffer_append(buf, ")");
}

void
msl_print_tex_helper_decl(struct _mesa_string_buffer *buf,
                          const struct msl_tex_group *group,
                          const struct msl_tex_helper *op)
{
   _mesa_string_buffer_append(buf, "[[visible]] extern ");
   print_prototype(buf, group, op, false);
   _mesa_string_buffer_append(buf, ";\n");
}

static void
print_args(struct _mesa_string_buffer *buf, unsigned first, unsigned count)
{
   for (unsigned i = 0; i < count; i++)
      _mesa_string_buffer_printf(buf, "%sp%u", i ? ", " : "", first + i);
}

static void
print_definition(struct _mesa_string_buffer *buf,
                 const struct msl_tex_group *group,
                 const struct msl_tex_helper *op)
{
   bool array = group->array;
   bool cube = group->dim == GLSL_SAMPLER_DIM_CUBE;
   bool has_lod = group->dim != GLSL_SAMPLER_DIM_BUF;
   unsigned coord_params = 1 + array;
   const char *vec4 = group_vec4(group);
   const char *scalar = msl_tex_type_name(group->type);

   _mesa_string_buffer_append(buf, "[[visible]] ");
   print_prototype(buf, group, op, true);
   _mesa_string_buffer_append(buf, " {\n    ");
   msl_print_texture_type(buf, group);
   _mesa_string_buffer_append(buf, " t = *(constant ");
   msl_print_texture_type(buf, group);
   _mesa_string_buffer_append(buf, "*)tex;\n    ");

   switch (op->kind) {
   case MSL_TEX_HELPER_SAMPLE: {
      unsigned p = 1 + coord_params;
      _mesa_string_buffer_printf(buf, "return t.%s(p0, ",
                                 op->compare ? "sample_compare" : "sample");
      print_args(buf, 1, coord_params + op->compare);
      p += op->compare;
      switch (op->lod) {
      case MSL_TEX_HELPER_LOD_BIAS:
         _mesa_string_buffer_printf(buf, ", bias(p%u)", p++);
         break;
      case MSL_TEX_HELPER_LOD_LEVEL:
         _mesa_string_buffer_printf(buf, ", level(p%u)", p++);
         break;
      case MSL_TEX_HELPER_LOD_GRADIENT:
         _mesa_string_buffer_printf(buf, ", gradient%s(p%u, p%u)",
                                    msl_texture_dim(group->dim), p, p + 1);
         p += 2;
         break;
      case MSL_TEX_HELPER_LOD_NONE:
         break;
      }
      if (op->min_lod)
         _mesa_string_buffer_printf(buf, ", min_lod_clamp(p%u)", p++);
      if (op->offset)
         _mesa_string_buffer_printf(buf, ", p%u", p++);
      _mesa_string_buffer_append(buf, ");\n");
      break;
   }
   case MSL_TEX_HELPER_GATHER:
      _mesa_string_buffer_printf(buf, "return t.%s(p0, ",
                                 op->compare ? "gather_compare" : "gather");
      print_args(buf, 1, coord_params + op->compare);
      if (op->offset)
         _mesa_string_buffer_printf(buf, ", p%u",
                                    1 + coord_params + op->compare);
      else if (!cube)
         _mesa_string_buffer_append(buf, ", int2(0)");
      if (!op->compare)
         _mesa_string_buffer_printf(buf, ", component::%c",
                                    "xyzw"[op->component]);
      _mesa_string_buffer_append(buf, ");\n");
      break;
   case MSL_TEX_HELPER_READ:
      _mesa_string_buffer_append(buf, "return t.read(");
      print_args(buf, 0, coord_params + has_lod);
      _mesa_string_buffer_append(buf, ");\n");
      break;
   case MSL_TEX_HELPER_WRITE:
      _mesa_string_buffer_append(buf, "t.write(");
      print_args(buf, 0, 1 + coord_params + has_lod);
      _mesa_string_buffer_append(buf, ");\n");
      break;
   case MSL_TEX_HELPER_ATOMIC:
      _mesa_string_buffer_printf(buf, "return t.%s(",
                                 atomic_method(op->atomic));
      print_args(buf, 0, coord_params + 1);
      _mesa_string_buffer_append(buf, ");\n");
      break;
   case MSL_TEX_HELPER_CMPXCHG:
      _mesa_string_buffer_printf(buf, "%s e = %s(p%u);\n    ", vec4, vec4,
                                 coord_params);
      _mesa_string_buffer_append(buf, "t.atomic_compare_exchange_weak(");
      print_args(buf, 0, coord_params);
      _mesa_string_buffer_printf(buf, ", &e, %s(p%u));\n    ", vec4,
                                 coord_params + 1);
      _mesa_string_buffer_printf(buf, "return %s(e.x);\n", scalar);
      break;
   case MSL_TEX_HELPER_FENCE:
      _mesa_string_buffer_append(buf, "t.fence();\n");
      break;
   case MSL_TEX_HELPER_SIZE: {
      const char *lod = group_has_mips(group) ? "p0" : "";
      bool height = group->dim != GLSL_SAMPLER_DIM_BUF;
      bool depth = group->dim == GLSL_SAMPLER_DIM_3D;
      _mesa_string_buffer_printf(buf, "return uint4(t.get_width(%s), ", lod);
      if (height)
         _mesa_string_buffer_printf(buf, "t.get_height(%s), ", lod);
      else
         _mesa_string_buffer_append(buf, "0, ");
      if (depth)
         _mesa_string_buffer_printf(buf, "t.get_depth(%s), ", lod);
      else
         _mesa_string_buffer_append(buf, "0, ");
      _mesa_string_buffer_append(buf,
                                 array ? "t.get_array_size());\n" : "0);\n");
      break;
   }
   case MSL_TEX_HELPER_LEVELS:
      _mesa_string_buffer_append(buf, "return t.get_num_mip_levels();\n");
      break;
   case MSL_TEX_HELPER_SAMPLES:
      _mesa_string_buffer_append(buf, "return t.get_num_samples();\n");
      break;
   case MSL_TEX_HELPER_LOD_CLAMPED:
      _mesa_string_buffer_append(buf,
                                 "return t.calculate_clamped_lod(p0, p1);\n");
      break;
   case MSL_TEX_HELPER_LOD_UNCLAMPED:
      _mesa_string_buffer_append(buf,
                                 "return t.calculate_unclamped_lod(p0, p1);\n");
      break;
   }
   _mesa_string_buffer_append(buf, "}\n");
}

static void
print_group_catalogue(struct _mesa_string_buffer *buf,
                      const struct msl_tex_group *group)
{
   static const nir_atomic_op atomics[] = {
      nir_atomic_op_iadd, nir_atomic_op_imin, nir_atomic_op_imax,
      nir_atomic_op_iand, nir_atomic_op_ior,  nir_atomic_op_ixor,
      nir_atomic_op_xchg,
   };
   struct msl_tex_helper ops[64];
   unsigned n = 0;

   for (unsigned compare = 0; compare < 2; compare++) {
      for (unsigned lod = 0; lod < 4; lod++) {
         for (unsigned min_lod = 0; min_lod < 2; min_lod++) {
            for (unsigned offset = 0; offset < 2; offset++) {
               ops[n++] = (struct msl_tex_helper){
                  .kind = MSL_TEX_HELPER_SAMPLE,
                  .compare = compare,
                  .lod = lod,
                  .min_lod = min_lod,
                  .offset = offset,
               };
            }
         }
      }
   }
   for (unsigned offset = 0; offset < 2; offset++) {
      for (unsigned c = 0; c < 4; c++)
         ops[n++] = (struct msl_tex_helper){.kind = MSL_TEX_HELPER_GATHER,
                                            .component = c,
                                            .offset = offset};
      ops[n++] = (struct msl_tex_helper){.kind = MSL_TEX_HELPER_GATHER,
                                         .compare = true,
                                         .offset = offset};
   }
   for (unsigned i = 0; i < ARRAY_SIZE(atomics); i++)
      ops[n++] = (struct msl_tex_helper){.kind = MSL_TEX_HELPER_ATOMIC,
                                         .atomic = atomics[i]};
   static const enum msl_tex_helper_kind simple[] = {
      MSL_TEX_HELPER_READ,          MSL_TEX_HELPER_WRITE,
      MSL_TEX_HELPER_CMPXCHG,       MSL_TEX_HELPER_FENCE,
      MSL_TEX_HELPER_SIZE,          MSL_TEX_HELPER_LEVELS,
      MSL_TEX_HELPER_SAMPLES,       MSL_TEX_HELPER_LOD_CLAMPED,
      MSL_TEX_HELPER_LOD_UNCLAMPED,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(simple); i++)
      ops[n++] = (struct msl_tex_helper){.kind = simple[i]};
   assert(n <= ARRAY_SIZE(ops));

   for (unsigned i = 0; i < n; i++) {
      if (msl_tex_helper_valid(group, &ops[i]))
         print_definition(buf, group, &ops[i]);
   }
}

bool
msl_helper_name_group(const char *name, uint32_t *group)
{
   if (!strcmp(name, "kk_v_f") || !strcmp(name, "kk_v_i") ||
       !strcmp(name, "kk_v_u")) {
      *group = MSL_VBO_HELPER_GROUP;
      return true;
   }
   unsigned id;
   int end = 0;
   if (sscanf(name, "kk_t_%3x_%n", &id, &end) == 1 && end == 9) {
      struct msl_tex_group tex_group;
      if (!group_from_id(id, &tex_group))
         return false;
      *group = id;
      return true;
   }
   return false;
}

char *
msl_helper_group_source(void *mem_ctx, uint32_t group_id)
{
   if (group_id == MSL_VBO_HELPER_GROUP)
      return msl_vbo_helper_source(mem_ctx);
   struct msl_tex_group group;
   if (!group_from_id(group_id, &group))
      return NULL;

   struct _mesa_string_buffer *buf = _mesa_string_buffer_create(NULL, 4096);
   _mesa_string_buffer_append(buf, "#include <metal_stdlib>\n"
                                   "using namespace metal;\n");
   print_group_catalogue(buf, &group);
   char *source = ralloc_strdup(mem_ctx, buf->buf);
   ralloc_free(buf);
   return source;
}
