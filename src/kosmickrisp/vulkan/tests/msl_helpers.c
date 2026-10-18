/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_compiler.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/mtl_format.h"
#include "kosmickrisp/compiler/msl_private.h"
#include "kosmickrisp/compiler/msl_vbo.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "../kk_metal_cache.h"
#include "util/os_time.h"
#include "util/ralloc.h"

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

static mtl_library *
compile(mtl_compiler *compiler, const char *src)
{
   return mtl_new_library(compiler, src, MTL_LANGUAGE_VERSION_4_0,
                          MTL_MATH_MODE_FAST,
                          MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
}

static unsigned
check_names(const char *src, uint32_t group)
{
   unsigned count = 0;
   const char *prefix = "[[visible]] ";
   for (const char *line = strstr(src, prefix); line;
        line = strstr(line + 1, prefix)) {
      const char *paren = strchr(line, '(');
      const char *name = paren;
      while (name[-1] != ' ')
         name--;
      char buf[128];
      CHECK(paren - name < (ptrdiff_t)sizeof(buf));
      memcpy(buf, name, paren - name);
      buf[paren - name] = '\0';
      uint32_t parsed;
      CHECK(msl_helper_name_group(buf, &parsed) && parsed == group);
      count++;
   }
   return count;
}

static void
test_group_libraries(mtl_compiler *compiler)
{
   unsigned groups = 0, functions = 0, failures = 0;
   for (uint32_t id = 0; id <= MSL_VBO_HELPER_GROUP; id++) {
      char *src = msl_helper_group_source(NULL, id);
      if (!src)
         continue;
      groups++;
      functions += check_names(src, id);
      mtl_library *lib = compile(compiler, src);
      if (lib) {
         mtl_release(lib);
      } else {
         fprintf(stderr, "group %03x failed:\n%s\n", id, src);
         failures++;
      }
      ralloc_free(src);
   }
   printf("helper groups: %u, functions: %u, failures: %u\n", groups, functions,
          failures);
   CHECK(groups > 0 && failures == 0);
   uint32_t group;
   CHECK(!msl_helper_name_group("kk_t_fff_s", &group));
   CHECK(!msl_helper_name_group("main_entrypoint", &group));
}

static const char *vs_src =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "struct VOut { float4 pos [[position]]; float2 uv [[user(locn0)]]; };\n"
   "vertex VOut main_vs(uint vid [[vertex_id]]) {\n"
   "   float2 p = float2((vid << 1) & 2, vid & 2);\n"
   "   VOut o; o.pos = float4(p * 2 - 1, 0, 1); o.uv = p; return o;\n"
   "}\n";

static char *
helper_name(const struct msl_tex_group *group, const struct msl_tex_helper *op)
{
   struct _mesa_string_buffer *buf = _mesa_string_buffer_create(NULL, 64);
   msl_print_tex_helper_name(buf, msl_tex_group_id(group), op);
   char *name = ralloc_strdup(NULL, buf->buf);
   ralloc_free(buf);
   return name;
}

static mtl_function_descriptor *
linked_function(mtl_library *main_lib, const char *entry,
                mtl_library *helper_lib, const char *helper)
{
   mtl_function_descriptor *fn =
      mtl_new_library_function_descriptor(main_lib, entry);
   mtl_function_descriptor *helper_fn =
      mtl_new_library_function_descriptor(helper_lib, helper);
   mtl_function_descriptor_set_linked_functions(fn, &helper_fn, 1);
   mtl_release(helper_fn);
   return fn;
}

static mtl_render_pipeline_descriptor *
render_descriptor(mtl_library *vs, mtl_library *fs, mtl_library *helper,
                  const char *name, bool link)
{
   mtl_render_pipeline_descriptor *desc = mtl_new_render_pipeline_descriptor();
   mtl_function_descriptor *vfn =
      mtl_new_library_function_descriptor(vs, "main_vs");
   mtl_function_descriptor *ffn =
      link ? linked_function(fs, "main_fs", helper, name)
           : mtl_new_library_function_descriptor(fs, "main_fs");
   mtl_render_pipeline_descriptor_set_vertex_shader(desc, vfn);
   mtl_render_pipeline_descriptor_set_fragment_shader(desc, ffn);
   mtl_render_pipeline_descriptor_set_color_attachment_format(
      desc, 0, MTL_PIXEL_FORMAT_R8G8B8A8_UNORM);
   mtl_release(vfn);
   mtl_release(ffn);
   return desc;
}

static void
test_static_link_archive(mtl_device *dev)
{
   mtl_compiler *compiler = mtl_new_compiler(dev);
   CHECK(compiler && mtl_compiler_has_serializer(compiler));

   struct msl_tex_group group = {
      .dim = GLSL_SAMPLER_DIM_2D,
      .type = nir_type_float32,
   };
   struct msl_tex_helper sample = {
      .kind = MSL_TEX_HELPER_SAMPLE,
      .lod = MSL_TEX_HELPER_LOD_BIAS,
   };
   struct msl_tex_helper read = {.kind = MSL_TEX_HELPER_READ};
   char *sample_name = helper_name(&group, &sample);
   char *read_name = helper_name(&group, &read);

   char *helper_src = msl_helper_group_source(NULL, msl_tex_group_id(&group));
   mtl_library *helper = compile(compiler, helper_src);
   CHECK(helper);

   struct _mesa_string_buffer *buf = _mesa_string_buffer_create(NULL, 1024);
   _mesa_string_buffer_append(
      buf, "#include <metal_stdlib>\nusing namespace metal;\n");
   msl_print_tex_helper_decl(buf, &group, &sample);
   _mesa_string_buffer_printf(
      buf,
      "struct VOut { float4 pos [[position]]; float2 uv [[user(locn0)]]; };\n"
      "struct Buffer { ulong contents[2]; };\n"
      "fragment float4 main_fs(VOut in [[stage_in]],\n"
      "                        constant Buffer &buf0 [[buffer(0)]]) {\n"
      "   sampler s = *(constant sampler *)buf0.contents[1];\n"
      "   return %s(buf0.contents[0], s, float2(in.uv), float(0));\n"
      "}\n",
      sample_name);
   mtl_library *fs = compile(compiler, buf->buf);
   CHECK(fs);
   ralloc_free(buf);

   buf = _mesa_string_buffer_create(NULL, 1024);
   _mesa_string_buffer_append(
      buf, "#include <metal_stdlib>\nusing namespace metal;\n");
   msl_print_tex_helper_decl(buf, &group, &read);
   _mesa_string_buffer_printf(
      buf,
      "struct Buffer { ulong contents[2]; };\n"
      "kernel void main_cs(uint2 id [[thread_position_in_grid]],\n"
      "                    constant Buffer &buf0 [[buffer(0)]]) {\n"
      "   ((device float4 *)buf0.contents[1])[id.x] =\n"
      "      %s(buf0.contents[0], uint2(id), uint(0));\n"
      "}\n",
      read_name);
   mtl_library *cs = compile(compiler, buf->buf);
   CHECK(cs);
   ralloc_free(buf);

   mtl_library *vs = compile(compiler, vs_src);
   CHECK(vs);

   mtl_render_pipeline_descriptor *desc =
      render_descriptor(vs, fs, helper, sample_name, true);
   mtl_render_pipeline_state *pso = mtl_new_render_pipeline(compiler, desc);
   CHECK(pso);
   mtl_release(desc);
   desc = render_descriptor(vs, fs, helper, sample_name, false);
   CHECK(!mtl_new_render_pipeline(compiler, desc));
   mtl_release(desc);

   mtl_function_descriptor *cfn =
      linked_function(cs, "main_cs", helper, read_name);
   mtl_compute_pipeline_state *cpso =
      mtl_new_compute_pipeline_state(compiler, cfn, 64);
   CHECK(cpso);

   size_t size = 0;
   void *data = mtl_compiler_serialize(compiler, &size);
   CHECK(data && size);
   mtl_archive *archive = mtl_new_archive(dev, data, size);
   CHECK(archive);

   desc = render_descriptor(vs, fs, helper, sample_name, true);
   mtl_render_pipeline_state *archived =
      mtl_archive_new_render_pipeline(archive, desc);
   CHECK(archived);
   mtl_release(desc);
   mtl_compute_pipeline_state *archived_cs =
      mtl_archive_new_compute_pipeline(archive, cfn, 64);
   CHECK(archived_cs);
   printf("static link archive round trip: %zu bytes\n", size);

   mtl_release(archived_cs);
   mtl_release(archived);
   mtl_release(archive);
   free(data);
   mtl_release(cfn);
   mtl_release(cpso);
   mtl_release(pso);
   mtl_release(vs);
   mtl_release(cs);
   mtl_release(fs);
   mtl_release(helper);
   ralloc_free(helper_src);
   ralloc_free(sample_name);
   ralloc_free(read_name);
   mtl_release(compiler);
}

static void
test_linked_vertex_cache(mtl_device *dev)
{
   static const char helper_src[] =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "[[visible]] float4 kk_h(float4 x) { return x * 2; }\n";
   static const char vertex_src[] =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "[[visible]] extern float4 kk_h(float4);\n"
      "struct O { float4 p [[position]]; float4 u [[user(u)]]; "
      "float4 v [[user(v)]]; };\n"
      "vertex O vs(uint id [[vertex_id]]) { "
      "float4 x = kk_h(float4(id, 0, 0, 1)); "
      "O o; o.p = x; o.u = x.wzyx; o.v = x.yzwx; return o; }\n";
   static const char *fragment_src[] = {
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "struct I { float4 x [[user(u)]]; };\n"
      "fragment float4 fs(I in [[stage_in]]) { return in.x; }\n",
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "struct I { float4 x [[user(v)]]; };\n"
      "fragment float4 fs(I in [[stage_in]]) { return in.x; }\n",
   };
   char identity[64];
   snprintf(identity, sizeof(identity), "%lld", (long long)os_time_get_nano());
   struct disk_cache *disk =
      disk_cache_create("kk-linked-vertex-test", identity, 0);
   CHECK(disk);
   mtl_compiler *compiler = mtl_new_compiler(dev);
   CHECK(compiler && mtl_compiler_has_serializer(compiler));
   mtl_library *helper = compile(compiler, helper_src);
   mtl_library *vertex = compile(compiler, vertex_src);
   CHECK(helper && vertex);
   mtl_function_descriptor *vs = linked_function(vertex, "vs", helper, "kk_h");
   CHECK(mtl_function_descriptor_has_linked_functions(vs));
   mtl_render_pipeline_descriptor *descs[2];
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
      mtl_library *fragment = compile(compiler, fragment_src[i]);
      CHECK(fragment);
      mtl_function_descriptor *fs =
         mtl_new_library_function_descriptor(fragment, "fs");
      CHECK(!mtl_function_descriptor_has_linked_functions(fs));
      descs[i] = mtl_new_render_pipeline_descriptor();
      mtl_render_pipeline_descriptor_set_vertex_shader(descs[i], vs);
      mtl_render_pipeline_descriptor_set_fragment_shader(descs[i], fs);
      mtl_render_pipeline_descriptor_set_input_primitive_topology(
         descs[i], MTL_PRIMITIVE_TOPOLOGY_CLASS_TRIANGLE);
      mtl_render_pipeline_descriptor_set_raster_sample_count(descs[i], 1);
      for (unsigned j = 0; j < 8; ++j)
         mtl_render_pipeline_descriptor_set_color_attachment_unspecialized(
            descs[i], j);
      mtl_release(fs);
      mtl_release(fragment);
   }
   for (unsigned warm = 0; warm < 2; ++warm) {
      struct kk_metal_cache *cache =
         kk_metal_cache_create(disk, dev, 0, compiler, 0);
      CHECK(cache);
      for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i) {
         cache_key key = {i + 1};
         struct kk_metal_capture_key group = {
            .vertex = {1},
            .fragment = {i + 1},
         };
         bool hit;
         mtl_render_pipeline_state *pipeline =
            kk_metal_cache_new_render_pipeline(cache, key, descs[i], &group,
                                               &hit);
         CHECK(pipeline && hit == (bool)warm);
         mtl_release(pipeline);
      }
      kk_metal_cache_destroy(cache);
      disk_cache_wait_for_idle(disk);
      mtl_release(compiler);
      compiler = warm ? NULL : mtl_new_compiler(dev);
      CHECK(warm || compiler);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(descs); ++i)
      mtl_release(descs[i]);
   mtl_release(vs);
   mtl_release(vertex);
   mtl_release(helper);
   disk_cache_destroy(disk);
   printf("linked vertex variants: 2/2 archive hits after cache reload\n");
}

int
main(void)
{
   unsetenv("MESA_KK_PSO_CACHE_DISABLE");
   unsetenv("MESA_SHADER_CACHE_DISABLE");
   mtl_device *dev = mtl_device_create();
   if (!dev) {
      /* Virtualised CI runners expose no Metal device; 77 is meson's skip. */
      fprintf(stderr, "no Metal device; skipping\n");
      return 77;
   }
   mtl_compiler *compiler = mtl_new_uncaptured_compiler(dev);
   CHECK(compiler);
   test_group_libraries(compiler);
   mtl_release(compiler);
   test_static_link_archive(dev);
   test_linked_vertex_cache(dev);
   mtl_release(dev);
   return 0;
}
