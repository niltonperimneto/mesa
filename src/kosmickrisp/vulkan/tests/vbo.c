#include "../kk_cmd_buffer.h"
#include "kosmickrisp/compiler/msl_private.h"
#include "kosmickrisp/compiler/nir_to_msl.h"
#include "util/format/u_format.h"
#include "util/os_time.h"
#include "nir_builder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include "kosmickrisp/bridge/mtl_argument_table.h"
#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_buffer.h"
#include "kosmickrisp/bridge/mtl_command_buffer.h"
#include "kosmickrisp/bridge/mtl_command_queue.h"
#include "kosmickrisp/bridge/mtl_compiler.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/mtl_residency_set.h"
#include "kosmickrisp/bridge/mtl_sync.h"
#endif

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static bool
replace_ids(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def *value;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_instance_id:
      value = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
      break;
   case nir_intrinsic_load_base_instance:
      value = nir_imm_int(b, 3);
      break;
   default:
      return false;
   }
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

static char *
fetch_shader(bool helpers, bool robustness, unsigned attributes,
             unsigned component, unsigned components, unsigned width)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  &kk_nir_options, "vbo_test");
   nir_def *root = nir_load_buffer_ptr_kk(&b, 1, 64, .binding = 0);
   nir_def *out = nir_load_global_constant(&b, 1, 64, root);
   nir_def *id = nir_load_vertex_id(&b);
   const nir_alu_type types[] = {nir_type_float, nir_type_int, nir_type_uint};
   for (unsigned a = 0; a < attributes; a++) {
      nir_alu_type type = types[a % 3] | width;
      nir_def *v =
         nir_load_input(&b, components, width, nir_imm_int(&b, 0), .base = a,
                        .component = component, .dest_type = type);
      if (width != 32)
         v = types[a % 3] == nir_type_float ? nir_f2f32(&b, v)
                                            : nir_u2u32(&b, v);
      nir_def *offset =
         nir_iadd_imm(&b, nir_imul_imm(&b, id, attributes * 16), a * 16);
      nir_store_global(&b, v, nir_iadd(&b, out, nir_u2u64(&b, offset)),
                       .align_mul = 4, .write_mask = BITFIELD_MASK(components));
   }
   CHECK(kk_nir_lower_vbo_dynamic(b.shader, robustness, helpers));
   b.shader->info.stage = MESA_SHADER_COMPUTE;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   memset(b.shader->info.system_values_read, 0,
          sizeof(b.shader->info.system_values_read));
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, replace_ids,
            nir_metadata_none, NULL);
   nir_validate_shader(b.shader, "dynamic fetch");
   msl_optimize_nir(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   struct nir_to_msl_options options = {.use_helpers = helpers};
   char *source = nir_to_msl(b.shader, &options);
   ralloc_free(b.shader);
   return source;
}

static void
check_register(nir_builder *b, nir_def *reg, bool needs)
{
   nir_metadata_require(b->impl,
                        nir_metadata_dominance | nir_metadata_instr_index);
   CHECK(msl_reg_needs_initializer(nir_def_as_intrinsic(reg)) == needs);
}

static void
test_registers(void)
{
   for (unsigned mode = 0; mode < 10; mode++) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &kk_nir_options, "reg_test");
      nir_def *reg = nir_decl_reg(&b, 4, 32, 0);
      nir_def *value = nir_imm_ivec4(&b, 1, 2, 3, 4);
      if (mode == 0 || mode == 6 || mode == 9)
         nir_store_reg(&b, value, reg);
      if (mode == 1)
         nir_build_store_reg(&b, value, reg, .write_mask = 3);
      if (mode == 2 || mode == 3) {
         nir_push_if(&b,
                     nir_ine_imm(&b, nir_load_local_invocation_index(&b), 0));
         nir_store_reg(&b, value, reg);
         if (mode == 3) {
            nir_push_else(&b, NULL);
            nir_store_reg(&b, value, reg);
         }
         nir_pop_if(&b, NULL);
      }
      if (mode == 4) {
         nir_push_loop(&b);
         nir_store_reg(&b, value, reg);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, NULL);
      }
      if (mode == 6) {
         nir_push_loop(&b);
         nir_load_reg(&b, reg);
         nir_store_reg(&b, value, reg);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, NULL);
      }
      nir_load_reg(&b, reg);
      if (mode == 7)
         nir_store_reg(&b, value, reg);
      if (mode == 8)
         nir_iadd_imm(&b, reg, 1);
      if (mode == 9) {
         nir_build_store_reg(&b, value, reg, .write_mask = 3);
         nir_load_reg(&b, reg);
      }
      check_register(&b, reg, mode != 0 && mode != 4 && mode != 6 && mode != 9);
      ralloc_free(b.shader);
   }
}

static void
test_sources(void)
{
   char *old = fetch_shader(false, true, 16, 0, 4, 32);
   char *compact = fetch_shader(true, true, 16, 0, 4, 32);
   printf("dynamic fetch MSL: %zu -> %zu bytes\n", strlen(old),
          strlen(compact));
   CHECK(strlen(compact) < strlen(old) / 2);
   const char *names[] = {"kk_v_f(", "kk_v_i(", "kk_v_u("};
   for (unsigned i = 0; i < 3; i++) {
      char decl[64];
      snprintf(decl, sizeof(decl), "%s(ulong, uint, bool);",
               msl_vbo_helper_name(i == 0   ? nir_type_float32
                                   : i == 1 ? nir_type_int32
                                            : nir_type_uint32));
      const char *first = strstr(compact, decl);
      CHECK(first && !strstr(first + 1, decl));
      CHECK(strstr(compact, names[i]));
      CHECK(!strstr(old, names[i]));
   }
   ralloc_free(old);
   ralloc_free(compact);
   for (unsigned width = 16; width <= 64; width *= 2) {
      char *src = fetch_shader(true, false, 3, 1, 2, width);
      CHECK(strstr(src, "kk_v_f("));
      ralloc_free(src);
   }
}

#ifdef __APPLE__
struct gpu {
   mtl_device *dev;
   mtl_compiler *compiler;
   mtl_command_queue *queue;
   mtl_command_allocator *allocator;
   mtl_command_buffer *commands;
   mtl_residency_set *residency;
   mtl_argument_table *table;
   mtl_shared_event *done;
   mtl_buffer *memory;
   uint8_t *cpu;
   uint64_t address;
   uint64_t serial;
};

static void
submit(struct gpu *g, mtl_compute_pipeline_state *pso, unsigned count)
{
   mtl_command_allocator_reset(g->allocator);
   mtl_begin_command_buffer(g->commands, g->allocator);
   mtl_compute_encoder *encoder = mtl_new_compute_command_encoder(g->commands);
   mtl_compute_set_pipeline_state(encoder, pso);
   mtl_compute_set_argument_table(encoder, g->table);
   mtl_dispatch_threads(encoder, (struct mtl_size){count, 1, 1},
                        (struct mtl_size){32, 1, 1});
   mtl_end_encoding(encoder);
   mtl_release(encoder);
   mtl_end_command_buffer(g->commands);
   mtl_command_queue_commit(g->queue, &g->commands, 1, NULL);
   mtl_signal_event(g->queue, (mtl_event *)g->done, ++g->serial);
   CHECK(mtl_shared_event_wait_until_signaled_value(g->done, g->serial, 10000));
}

static mtl_compute_pipeline_state *
pipeline(struct gpu *g, const char *src, mtl_library *helpers, bool link,
         mtl_function_descriptor **descriptor)
{
   mtl_library *lib = mtl_new_library(
      g->compiler, src, MTL_LANGUAGE_VERSION_4_0, MTL_MATH_MODE_FAST,
      MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   CHECK(lib);
   mtl_function_descriptor *fn =
      mtl_new_library_function_descriptor(lib, "main_entrypoint");
   CHECK(fn);
   if (link) {
      mtl_function_descriptor *functions[3];
      const nir_alu_type types[] = {nir_type_float32, nir_type_int32,
                                    nir_type_uint32};
      for (unsigned i = 0; i < 3; i++) {
         functions[i] = mtl_new_library_function_descriptor(
            helpers, msl_vbo_helper_name(types[i]));
         CHECK(functions[i]);
      }
      mtl_function_descriptor_set_linked_functions(fn, functions, 3);
      for (unsigned i = 0; i < 3; i++)
         mtl_release(functions[i]);
   }
   mtl_compute_pipeline_state *pso =
      mtl_new_compute_pipeline_state(g->compiler, fn, 32);
   CHECK(pso);
   mtl_release(lib);
   if (descriptor)
      *descriptor = fn;
   else
      mtl_release(fn);
   return pso;
}

static void
test_gpu(void)
{
   struct gpu g = {0};
   g.dev = mtl_device_create();
   if (!g.dev) {
      printf("no Metal device; GPU fetch tests skipped\n");
      return;
   }
   g.compiler = mtl_new_compiler(g.dev);
   g.queue = mtl_new_command_queue(g.dev);
   g.allocator = mtl_new_command_allocator(g.dev);
   g.commands = mtl_new_command_buffer(g.dev);
   g.residency = mtl_new_residency_set(g.dev);
   g.done = mtl_new_shared_event(g.dev);
   CHECK(!posix_memalign((void **)&g.cpu, 16384, 131072));
   memset(g.cpu, 0, 131072);
   g.memory = mtl_new_buffer_with_bytes_no_copy(g.dev, g.cpu, 131072);
   CHECK(g.memory);
   g.address = mtl_buffer_get_gpu_address(g.memory);
   mtl_residency_set_add_allocation(g.residency, (mtl_allocation *)g.memory);
   mtl_residency_set_commit(g.residency);
   mtl_command_queue_add_residency_set(g.queue, g.residency);
   mtl_argument_table_descriptor *td = mtl_new_argument_table_descriptor();
   mtl_set_max_buffer_binding_count(td, 2);
   g.table = mtl_new_argument_table(g.dev, td);
   mtl_release(td);
   mtl_set_address(g.table, g.address, 0);
   mtl_set_address(g.table, g.address, 1);
   char *helper_src = msl_vbo_helper_source(NULL);
   mtl_library *helpers = mtl_new_library(
      g.compiler, helper_src, MTL_LANGUAGE_VERSION_4_0, MTL_MATH_MODE_FAST,
      MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   CHECK(helpers);
   const struct {
      bool robust;
      unsigned component, components, width;
   } configs[] = {{true, 0, 4, 32},
                  {false, 0, 4, 32},
                  {true, 1, 2, 16},
                  {true, 1, 2, 64}};
   unsigned formats = 0, cases = 0;
   for (unsigned config = 0; config < ARRAY_SIZE(configs); config++) {
      char *sources[2] = {
         fetch_shader(false, configs[config].robust, 3,
                      configs[config].component, configs[config].components,
                      configs[config].width),
         fetch_shader(true, configs[config].robust, 3,
                      configs[config].component, configs[config].components,
                      configs[config].width)};
      mtl_function_descriptor *fn;
      mtl_compute_pipeline_state *psos[2] = {
         pipeline(&g, sources[0], helpers, false, NULL),
         pipeline(&g, sources[1], helpers, true, &fn)};
      size_t archive_size;
      void *archive_data = mtl_compiler_serialize(g.compiler, &archive_size);
      CHECK(archive_data && archive_size);
      mtl_archive *archive = mtl_new_archive(g.dev, archive_data, archive_size);
      CHECK(archive);
      mtl_compute_pipeline_state *restored =
         mtl_archive_new_compute_pipeline(archive, fn, 32);
      CHECK(restored);
      mtl_release(psos[1]);
      psos[1] = restored;
      struct kk_root_descriptor_table *root = (void *)g.cpu;
      const unsigned threads = 128;
      uint32_t random = 0x837219ab;
      uint32_t *input = (void *)(g.cpu + 32768);
      for (unsigned i = 0; i < 4096; i++) {
         random ^= random << 13;
         random ^= random >> 17;
         random ^= random << 5;
         input[i] = i < 16 ? (uint32_t[]){0,           ~0u,         0x80000000u,
                                          0x7fffffffu, 0x3f800000u, 0xbf800000u,
                                          0x7f800000u, 0xff800000u, 0x7fc00000u,
                                          0x00000001u, 0x80008000u, 0x7fff7fffu,
                                          0x3c003c00u, 0x7c007c00u, 0x7e007e00u,
                                          0x80808080u}[i]
                           : random;
      }
      uint32_t expected[128 * 12];
      for (enum pipe_format format = 1; format < PIPE_FORMAT_COUNT; format++) {
         uint32_t encoded = kk_vbo_encode_format(format);
         if (!encoded)
            continue;
         if (config == 0)
            formats++;
         for (unsigned mode = 0; mode < 8; mode++) {
            if (mode == 7 && !configs[config].robust)
               continue;
            for (unsigned a = 0; a < 3; a++) {
               root->draw.attrib_format[a] = mode == 1 ? 0 : encoded;
               root->draw.attrib_base[a] = g.address + 32768;
               root->draw.attrib_stride[a] =
                  mode == 2 ? 0 : util_format_get_blocksize(format);
               root->draw.attrib_clamps[a] = (mode == 3 || mode == 7) ? 0
                                             : mode == 4              ? 63
                                                                      : 127;
               root->draw.attrib_divisor[a] = mode == 5 ? 0 : 3;
               if (mode >= 5)
                  root->draw.attrib_format[a] |= KK_VBO_FMT_INSTANCED;
               if (mode == 1 || mode == 7)
                  root->draw.attrib_base[a] = 0;
            }
            root->addr = g.address + 65536;
            memset(g.cpu + 65536, 0xcd, sizeof(expected));
            submit(&g, psos[0], threads);
            memcpy(expected, g.cpu + 65536, sizeof(expected));
            memset(g.cpu + 65536, 0xcd, sizeof(expected));
            submit(&g, psos[1], threads);
            uint32_t *actual = (void *)(g.cpu + 65536);
            for (unsigned i = 0; i < ARRAY_SIZE(expected); i++) {
               if (expected[i] == actual[i])
                  continue;
               float x, y;
               memcpy(&x, &expected[i], 4);
               memcpy(&y, &actual[i], 4);
               if (i % 12 < 4 &&
                   ((isnan(x) && isnan(y)) ||
                    (isfinite(x) && isfinite(y) &&
                     fabsf(x - y) <= fmaxf(fabsf(x), fabsf(y)) * 1e-6f)))
                  continue;
               fprintf(stderr, "%s config %u mode %u lane %u: %08x != %08x\n",
                       util_format_name(format), config, mode, i, expected[i],
                       actual[i]);
               abort();
            }
            cases++;
         }
      }
      mtl_release(fn);
      for (unsigned i = 0; i < 2; i++) {
         mtl_release(psos[i]);
         ralloc_free(sources[i]);
      }
      mtl_release(archive);
      free(archive_data);
   }
   printf(
      "GPU dynamic fetch: %u formats, %u cases; linked archive reload passed\n",
      formats, cases);
   mtl_release(helpers);
   ralloc_free(helper_src);
   mtl_command_queue_remove_residency_set(g.queue, g.residency);
   mtl_release(g.table);
   mtl_release(g.residency);
   mtl_release(g.done);
   mtl_release(g.commands);
   mtl_release(g.allocator);
   mtl_release(g.queue);
   mtl_release(g.memory);
   free(g.cpu);
   mtl_release(g.compiler);
   mtl_release(g.dev);
}
#endif

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_registers();
   test_sources();
#ifdef __APPLE__
   test_gpu();
#endif
   glsl_type_singleton_decref();
   return 0;
}
