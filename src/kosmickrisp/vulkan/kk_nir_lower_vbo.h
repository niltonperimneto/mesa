/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/format/u_formats.h"
#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KK_MAX_ATTRIBS (32)
#define KK_MAX_VBUFS   (32)

/* See pipe_vertex_element for justification on the sizes. This structure should
 * be small so it can be embedded into a shader key.
 */
struct kk_attribute {
   /* If instanced, Zero means all get the same value (Vulkan semantics). */
   uint32_t divisor;
   /* Buffer binding to load stride from root_table */
   uint32_t binding;

   /* pipe_format, all vertex formats should be <= 255 */
   uint8_t format;

   unsigned buf   : 7;
   bool instanced : 1;
};

bool kk_nir_lower_vbo(nir_shader *shader, struct kk_attribute *attribs,
                      bool robustness2);

bool kk_vbo_supports_format(enum pipe_format format);

enum kk_vbo_type {
   KK_VBO_TYPE_UNORM = 0,
   KK_VBO_TYPE_SNORM = 1,
   KK_VBO_TYPE_USCALED = 2,
   KK_VBO_TYPE_SSCALED = 3,
   KK_VBO_TYPE_UINT = 4,
   KK_VBO_TYPE_SINT = 5,
   KK_VBO_TYPE_SFLOAT = 6,
};

#define KK_VBO_FMT_SIZE_SHIFT    0
#define KK_VBO_FMT_SIZE_MASK     0x3
#define KK_VBO_FMT_COUNT_SHIFT   2
#define KK_VBO_FMT_COUNT_MASK    0x3
#define KK_VBO_FMT_TYPE_SHIFT    4
#define KK_VBO_FMT_TYPE_MASK     0x7
#define KK_VBO_FMT_RGB10A2       (1u << 7)
#define KK_VBO_FMT_RG11B10F      (1u << 8)
#define KK_VBO_FMT_INSTANCED     (1u << 9)
#define KK_VBO_FMT_VALID         (1u << 10)
#define KK_VBO_FMT_SWIZZLE_SHIFT 11
#define KK_VBO_FMT_SWIZZLE_MASK  0x7

uint32_t kk_vbo_encode_format(enum pipe_format format);

bool kk_nir_lower_vbo_dynamic(nir_shader *shader, bool robustness2);

#ifdef __cplusplus
} /* extern C */
#endif
