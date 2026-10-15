/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "kk_msl_helpers.h"

#include "kk_debug.h"
#include "kk_device.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_compiler.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "util/hash_table.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"

struct kk_msl_helper_group {
   simple_mtx_t lock;
   bool compiled;
   mtl_library *library;
   struct hash_table *functions;
};

bool
kk_msl_helpers_init(struct kk_msl_helpers *helpers)
{
   simple_mtx_init(&helpers->lock, mtx_plain);
   helpers->groups = _mesa_hash_table_u64_create(NULL);
   return helpers->groups != NULL;
}

void
kk_msl_helpers_finish(struct kk_msl_helpers *helpers)
{
   if (!helpers->groups)
      return;
   hash_table_u64_foreach(helpers->groups, entry) {
      struct kk_msl_helper_group *group = entry.data;
      hash_table_foreach(group->functions, function)
         mtl_release(function->data);
      _mesa_hash_table_destroy(group->functions, NULL);
      if (group->library)
         mtl_release(group->library);
      simple_mtx_destroy(&group->lock);
      ralloc_free(group);
   }
   _mesa_hash_table_u64_destroy(helpers->groups);
   helpers->groups = NULL;
   simple_mtx_destroy(&helpers->lock);
}

static struct kk_msl_helper_group *
kk_msl_helper_group(struct kk_msl_helpers *helpers, uint32_t id)
{
   simple_mtx_lock(&helpers->lock);
   struct kk_msl_helper_group *group =
      _mesa_hash_table_u64_search(helpers->groups, id);
   if (!group) {
      group = rzalloc(NULL, struct kk_msl_helper_group);
      if (group) {
         group->functions = _mesa_string_hash_table_create(group);
         if (group->functions) {
            simple_mtx_init(&group->lock, mtx_plain);
            _mesa_hash_table_u64_insert(helpers->groups, id, group);
         } else {
            ralloc_free(group);
            group = NULL;
         }
      }
   }
   simple_mtx_unlock(&helpers->lock);
   return group;
}

static mtl_function_descriptor *
kk_msl_helper_function(struct kk_device *dev, uint32_t id, const char *name)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);
   struct kk_msl_helper_group *group =
      kk_msl_helper_group(&dev->msl_helpers, id);
   if (!group)
      return NULL;

   mtl_function_descriptor *function = NULL;
   simple_mtx_lock(&group->lock);
   if (!group->compiled) {
      group->compiled = true;
      char *source = msl_helper_group_source(NULL, id);
      if (source) {
         int64_t start = os_time_get_nano();
         group->library = mtl_new_library(
            dev->mtl_compiler_handle, source, pdev->info.msl_version,
            MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
         if (KK_DEBUG(PSO))
            mesa_logi("MSL helper group %03x: %.1f ms", id,
                      (os_time_get_nano() - start) / 1e6);
         ralloc_free(source);
      }
   }
   if (group->library) {
      struct hash_entry *entry =
         _mesa_hash_table_search(group->functions, name);
      if (entry) {
         function = entry->data;
      } else {
         function = mtl_new_library_function_descriptor(group->library, name);
         if (function)
            _mesa_hash_table_insert(group->functions,
                                    ralloc_strdup(group, name), function);
      }
   }
   simple_mtx_unlock(&group->lock);
   return function;
}

VkResult
kk_msl_helpers_link(struct kk_device *dev, mtl_function_descriptor *function,
                    const struct msl_compile_data *data)
{
   static const char header[] = "using namespace metal;\n";
   static const char decl[] = "[[visible]] extern ";
   const char *line = strstr(data->code, header);
   if (!line)
      return VK_SUCCESS;
   line += strlen(header);

   struct util_dynarray functions;
   util_dynarray_init(&functions, NULL);
   VkResult result = VK_SUCCESS;
   while (!strncmp(line, decl, strlen(decl))) {
      const char *paren = strchr(line, '(');
      const char *end = strchr(line, '\n');
      if (!paren || !end || paren > end) {
         result = VK_ERROR_INVALID_SHADER_NV;
         break;
      }
      const char *name = paren;
      while (name > line && name[-1] != ' ')
         name--;
      char buf[64];
      uint32_t id;
      if (paren - name >= (ptrdiff_t)sizeof(buf)) {
         result = VK_ERROR_INVALID_SHADER_NV;
         break;
      }
      memcpy(buf, name, paren - name);
      buf[paren - name] = '\0';
      mtl_function_descriptor *helper =
         msl_helper_name_group(buf, &id) ? kk_msl_helper_function(dev, id, buf)
                                         : NULL;
      if (!helper) {
         result = VK_ERROR_INVALID_SHADER_NV;
         break;
      }
      util_dynarray_append(&functions, helper);
      line = end + 1;
   }

   if (result == VK_SUCCESS && functions.size)
      mtl_function_descriptor_set_linked_functions(
         function, functions.data,
         util_dynarray_num_elements(&functions, mtl_function_descriptor *));
   util_dynarray_fini(&functions);
   return result;
}
