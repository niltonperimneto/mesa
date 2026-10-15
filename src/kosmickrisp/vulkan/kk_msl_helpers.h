/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_MSL_HELPERS_H
#define KK_MSL_HELPERS_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/simple_mtx.h"

struct hash_table_u64;
struct kk_device;
struct msl_compile_data;

struct kk_msl_helpers {
   simple_mtx_t lock;
   struct hash_table_u64 *groups;
};

bool kk_msl_helpers_init(struct kk_msl_helpers *helpers);
void kk_msl_helpers_finish(struct kk_msl_helpers *helpers);

VkResult kk_msl_helpers_link(struct kk_device *dev,
                             mtl_function_descriptor *function,
                             const struct msl_compile_data *data);

#endif /* KK_MSL_HELPERS_H */
