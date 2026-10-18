#include "msl_vbo.h"
#include "util/ralloc.h"
#include "msl_private.h"

const char *
msl_vbo_helper_name(nir_alu_type type)
{
   switch (type) {
   case nir_type_float32:
      return "kk_v_f";
   case nir_type_int32:
      return "kk_v_i";
   case nir_type_uint32:
      return "kk_v_u";
   default:
      UNREACHABLE("Invalid vertex fetch type");
   }
}

void
msl_print_vbo_helper_decl(struct _mesa_string_buffer *buf, nir_alu_type type)
{
   _mesa_string_buffer_printf(
      buf, "[[visible]] extern %s4 %s(ulong, uint, bool);\n",
      msl_tex_type_name(type), msl_vbo_helper_name(type));
}

char *
msl_vbo_helper_source(void *mem_ctx)
{
   struct _mesa_string_buffer *buf = _mesa_string_buffer_create(NULL, 4096);
   _mesa_string_buffer_append(
      buf, "#include <metal_stdlib>\nusing namespace metal;\n");
#define DEFINE(name)                                                           \
   _mesa_string_buffer_printf(buf, "#define " #name " %uu\n", name)
   DEFINE(KK_VBO_FMT_VALID);
   DEFINE(KK_VBO_FMT_RGB10A2);
   DEFINE(KK_VBO_FMT_RG11B10F);
   DEFINE(KK_VBO_FMT_SIZE_SHIFT);
   DEFINE(KK_VBO_FMT_SIZE_MASK);
   DEFINE(KK_VBO_FMT_COUNT_SHIFT);
   DEFINE(KK_VBO_FMT_COUNT_MASK);
   DEFINE(KK_VBO_FMT_TYPE_SHIFT);
   DEFINE(KK_VBO_FMT_TYPE_MASK);
   DEFINE(KK_VBO_FMT_SWIZZLE_SHIFT);
   DEFINE(KK_VBO_FMT_SWIZZLE_MASK);
   DEFINE(KK_VBO_TYPE_UNORM);
   DEFINE(KK_VBO_TYPE_SNORM);
   DEFINE(KK_VBO_TYPE_USCALED);
   DEFINE(KK_VBO_TYPE_SSCALED);
   DEFINE(KK_VBO_TYPE_SINT);
#undef DEFINE
   _mesa_string_buffer_append(
      buf,
      "inline uint4 kk_v_raw(ulong addr, uint fmt, bool fetch) {\n"
      "   uint4 v = uint4(0);\n"
      "   if (!fetch || !(fmt & KK_VBO_FMT_VALID)) return v;\n"
      "   uint n = ((fmt >> KK_VBO_FMT_COUNT_SHIFT) & KK_VBO_FMT_COUNT_MASK) + 1;\n"
      "   uint size = (fmt >> KK_VBO_FMT_SIZE_SHIFT) & KK_VBO_FMT_SIZE_MASK;\n"
      "   if (size == 2) {\n"
      "      constant uint *p = (constant uint *)addr;\n"
      "      v.x = p[0]; if (n > 1) v.y = p[1]; if (n > 2) v.z = p[2]; if (n > 3) v.w = p[3];\n"
      "   } else if (size == 1) {\n"
      "      constant ushort *p = (constant ushort *)addr;\n"
      "      v.x = p[0]; if (n > 1) v.y = p[1]; if (n > 2) v.z = p[2]; if (n > 3) v.w = p[3];\n"
      "   } else {\n"
      "      constant uchar *p = (constant uchar *)addr;\n"
      "      v.x = p[0]; if (n > 1) v.y = p[1]; if (n > 2) v.z = p[2]; if (n > 3) v.w = p[3];\n"
      "   }\n"
      "   return v;\n"
      "}\n"
      "inline uint4 kk_v_unpack(uint4 v, uint fmt) {\n"
      "   if (fmt & KK_VBO_FMT_RGB10A2)\n"
      "      return (uint4(v.x) >> uint4(0, 10, 20, 30)) & uint4(1023, 1023, 1023, 3);\n"
      "   return v;\n"
      "}\n"
      "inline int4 kk_v_signed(uint4 v, uint fmt) {\n"
      "   if (fmt & KK_VBO_FMT_RGB10A2)\n"
      "      return as_type<int4>(v << uint4(22, 22, 22, 30)) >> uint4(22, 22, 22, 30);\n"
      "   uint size = (fmt >> KK_VBO_FMT_SIZE_SHIFT) & KK_VBO_FMT_SIZE_MASK;\n"
      "   if (size == 0) return int4(as_type<char4>(uchar4(v)));\n"
      "   if (size == 1) return int4(as_type<short4>(ushort4(v)));\n"
      "   return as_type<int4>(v);\n"
      "}\n"
      "inline float4 kk_v_float(uint4 raw, uint fmt) {\n"
      "   uint4 u = kk_v_unpack(raw, fmt);\n"
      "   uint type = (fmt >> KK_VBO_FMT_TYPE_SHIFT) & KK_VBO_FMT_TYPE_MASK;\n"
      "   bool packed = (fmt & KK_VBO_FMT_RGB10A2) != 0;\n"
      "   bool is8 = ((fmt >> KK_VBO_FMT_SIZE_SHIFT) & KK_VBO_FMT_SIZE_MASK) == 0;\n"
      "   switch (type) {\n"
      "   case KK_VBO_TYPE_UNORM:\n"
      "      return float4(u) * (packed ? float4(1.0f/1023.0f, 1.0f/1023.0f, 1.0f/1023.0f, 1.0f/3.0f) : is8 ? float4(1.0f/255.0f) : float4(1.0f/65535.0f));\n"
      "   case KK_VBO_TYPE_SNORM:\n"
      "      return max(float4(kk_v_signed(u, fmt)) * (packed ? float4(1.0f/511.0f, 1.0f/511.0f, 1.0f/511.0f, 1.0f) : is8 ? float4(1.0f/127.0f) : float4(1.0f/32767.0f)), -1.0f);\n"
      "   case KK_VBO_TYPE_USCALED: return float4(u);\n"
      "   case KK_VBO_TYPE_SSCALED: return float4(kk_v_signed(u, fmt));\n"
      "   default:\n"
      "      if (fmt & KK_VBO_FMT_RG11B10F)\n"
      "         return float4(float3(as_type<half3>(ushort3((raw.x & 0x7ffu) << 4, (raw.x & 0x3ff800u) >> 7, (raw.x & 0xffc00000u) >> 17))), 0.0f);\n"
      "      if (((fmt >> KK_VBO_FMT_SIZE_SHIFT) & KK_VBO_FMT_SIZE_MASK) == 1)\n"
      "         return float4(as_type<half4>(ushort4(raw)));\n"
      "      return as_type<float4>(raw);\n"
      "   }\n"
      "}\n");

   const nir_alu_type types[] = {nir_type_float32, nir_type_int32,
                                 nir_type_uint32};
   for (unsigned i = 0; i < ARRAY_SIZE(types); i++) {
      const char *type = msl_tex_type_name(types[i]);
      _mesa_string_buffer_printf(
         buf,
         "[[visible]] %s4 %s(ulong addr, uint fmt, bool fetch) {\n"
         "   if (!(fmt & KK_VBO_FMT_VALID)) return %s4(0, 0, 0, 1);\n"
         "   uint4 raw = kk_v_raw(addr, fmt, fetch);\n",
         type, msl_vbo_helper_name(types[i]), type);
      if (types[i] == nir_type_float32) {
         _mesa_string_buffer_append(buf,
                                    "   float4 v = kk_v_float(raw, fmt);\n");
      } else {
         _mesa_string_buffer_printf(
            buf,
            "   uint4 u = kk_v_unpack(raw, fmt);\n"
            "   %s4 v = %s4(((fmt >> KK_VBO_FMT_TYPE_SHIFT) & KK_VBO_FMT_TYPE_MASK) == KK_VBO_TYPE_SINT ? as_type<uint4>(kk_v_signed(u, fmt)) : u);\n",
            type, type);
      }
      _mesa_string_buffer_printf(
         buf,
         "   uint4 swz = (uint4(fmt) >> uint4(KK_VBO_FMT_SWIZZLE_SHIFT, KK_VBO_FMT_SWIZZLE_SHIFT + 3, KK_VBO_FMT_SWIZZLE_SHIFT + 6, KK_VBO_FMT_SWIZZLE_SHIFT + 9)) & uint4(KK_VBO_FMT_SWIZZLE_MASK);\n"
         "   %s4 out = %s4(v.x);\n"
         "   out = select(out, %s4(v.y), swz == uint4(1));\n"
         "   out = select(out, %s4(v.z), swz == uint4(2));\n"
         "   out = select(out, %s4(v.w), swz == uint4(3));\n"
         "   out = select(out, %s4(0), swz == uint4(4));\n"
         "   return select(out, %s4(1), swz == uint4(5));\n"
         "}\n",
         type, type, type, type, type, type, type);
   }
   char *source = ralloc_strdup(mem_ctx, buf->buf);
   ralloc_free(buf);
   return source;
}
