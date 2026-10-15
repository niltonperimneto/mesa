/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_compiler.h"

#include "mtl_format.h"

/* Compiler */
mtl_compiler *
mtl_new_compiler(mtl_device *device)
{
   return NULL;
}

mtl_compiler *
mtl_new_uncaptured_compiler(mtl_device *device)
{
   return NULL;
}

/* Library */
mtl_library *
mtl_new_library(mtl_compiler *compiler, const char *src,
                enum mtl_language_version msl_version,
                enum mtl_math_mode math_mode,
                enum mtl_math_floating_point_functions math_fp_fns)
{
   return NULL;
}

mtl_function_descriptor *
mtl_new_library_function_descriptor(mtl_library *library,
                                    const char *entry_point)
{
   return NULL;
}

void
mtl_function_descriptor_set_linked_functions(mtl_function_descriptor *function,
                                             mtl_function_descriptor **functions,
                                             uint32_t count)
{
}

/* Compute pipeline */
mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state(mtl_compiler *compiler,
                               mtl_function_descriptor *function,
                               uint64_t max_total_threads_per_threadgroup)
{
   return NULL;
}

/* Render pipeline descriptor */
mtl_render_pipeline_descriptor *
mtl_new_render_pipeline_descriptor(void)
{
   return NULL;
}

void
mtl_render_pipeline_descriptor_set_vertex_shader(
   mtl_render_pipeline_descriptor *descriptor, mtl_function_descriptor *shader)
{
}

void
mtl_render_pipeline_descriptor_set_fragment_shader(
   mtl_render_pipeline_descriptor *descriptor, mtl_function_descriptor *shader)
{
}

void
mtl_render_pipeline_descriptor_set_input_primitive_topology(
   mtl_render_pipeline_descriptor *descriptor,
   enum mtl_primitive_topology_class topology_class)
{
}

void
mtl_render_pipeline_descriptor_set_color_attachment_format(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index,
   enum mtl_pixel_format format)
{
}

void
mtl_render_pipeline_descriptor_set_color_attachment_blend(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index, bool enable,
   enum mtl_blend_factor src_rgb, enum mtl_blend_factor dst_rgb,
   enum mtl_blend_operation op_rgb, enum mtl_blend_factor src_alpha,
   enum mtl_blend_factor dst_alpha, enum mtl_blend_operation op_alpha,
   uint8_t write_mask)
{
}

void
mtl_render_pipeline_descriptor_set_color_attachment_unspecialized(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index)
{
}

void
mtl_render_pipeline_descriptor_set_raster_sample_count(
   mtl_render_pipeline_descriptor *descriptor, uint32_t sample_count)
{
}

void
mtl_render_pipeline_descriptor_set_multisample_alpha(
   mtl_render_pipeline_descriptor *descriptor, bool coverage, bool one)
{
}

void
mtl_render_pipeline_descriptor_set_rasterization_enabled(
   mtl_render_pipeline_descriptor *descriptor, bool enabled)
{
}

void
mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(
   mtl_render_pipeline_descriptor *descriptor, uint32_t count)
{
}

/* Render pipeline */
mtl_render_pipeline_state *
mtl_new_render_pipeline(mtl_compiler *compiler,
                        mtl_render_pass_descriptor *descriptor)
{
   return NULL;
}

mtl_render_pipeline_state *
mtl_specialize_render_pipeline(mtl_compiler *compiler,
                               mtl_render_pipeline_descriptor *descriptor,
                               mtl_render_pipeline_state *base)
{
   return NULL;
}

bool mtl_compiler_has_serializer(mtl_compiler *compiler)
{
   return false;
}

void *mtl_compiler_serialize(mtl_compiler *compiler, size_t *size)
{
   *size = 0;
   return NULL;
}

void mtl_pipeline_cache_identity(mtl_device *device, uint32_t gpu_family,
                                 char *identity, size_t size)
{
   if (size)
      identity[0] = 0;
}

mtl_archive *mtl_new_archive(mtl_device *device, const void *data, size_t size)
{
   return NULL;
}

mtl_render_pipeline_state *mtl_archive_new_render_pipeline(
   mtl_archive *archive, mtl_render_pipeline_descriptor *descriptor)
{
   return NULL;
}

mtl_compute_pipeline_state *mtl_archive_new_compute_pipeline(
   mtl_archive *archive, mtl_function_descriptor *function,
   uint64_t max_threads)
{
   return NULL;
}

bool
mtl_compile_phase_enabled(void)
{
   return false;
}

struct mtl_compile_phase
mtl_compile_phase_begin(const char *operation, const void *object,
                         const char *key, const char *origin)
{
   return (struct mtl_compile_phase){0};
}

void
mtl_compile_phase_end(struct mtl_compile_phase phase, bool success)
{
}
