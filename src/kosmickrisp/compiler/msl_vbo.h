#pragma once

#include "util/string_buffer.h"
#include "nir.h"

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

#define MSL_VBO_HELPER_GROUP 0x1000u

const char *msl_vbo_helper_name(nir_alu_type type);
void msl_print_vbo_helper_decl(struct _mesa_string_buffer *buf,
                               nir_alu_type type);
char *msl_vbo_helper_source(void *mem_ctx);
