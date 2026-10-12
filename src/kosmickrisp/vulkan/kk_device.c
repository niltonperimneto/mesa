/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_device.h"
#include "kk_metal_cache.h"
#include "util/disk_cache.h"
#include "vk_shader.h"

#include "kk_buffer.h"
#include "kk_buffer_view.h"
#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_entrypoints.h"
#include "kk_image.h"
#include "kk_image_view.h"
#include "kk_instance.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/ns_process_info.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "kk_dispatch_cmd.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include "vulkan/wsi/wsi_common.h"
#include "vk_pipeline_cache.h"

#include <time.h>
#include <pthread.h>
#include "util/u_atomic.h"

struct kk_mtl_compiler {
   mtl_compiler *handle;
   uint32_t refcount;
   struct disk_cache *disk;
   struct kk_metal_cache *cache;
};

static struct hash_table compilers_ht;
static simple_mtx_t compilers_ht_lock;
static once_flag compilers_ht_once = ONCE_FLAG_INIT;

static void
kk_init_compiler_table()
{
   _mesa_pointer_hash_table_init(&compilers_ht, NULL);
   simple_mtx_init(&compilers_ht_lock, mtx_plain);
}

static uint64_t
kk_metal_blob_budget(void)
{
   uint64_t total = kk_metal_cache_parse_size(
      debug_get_option("MESA_KK_CACHE_MAX_SIZE", NULL));
   uint64_t parts = MAX2(
      debug_get_num_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS", 50), 1);
   return total / parts;
}

static struct disk_cache *
kk_create_metal_disk_cache(struct kk_device *dev)
{
   if (dev->vk.disable_internal_cache ||
       debug_get_bool_option("MESA_KK_PSO_CACHE_DISABLE", false))
      return NULL;
   char identity[VK_UUID_SIZE * 2 + 1];
   static const char hex[] = "0123456789abcdef";
   for (unsigned i = 0; i < VK_UUID_SIZE; ++i) {
      uint8_t value = dev->vk.physical->properties.pipelineCacheUUID[i];
      identity[i * 2] = hex[value >> 4];
      identity[i * 2 + 1] = hex[value & 15];
   }
   identity[VK_UUID_SIZE * 2] = 0;
   return disk_cache_create_custom(
      "kosmickrisp-metal", identity, 0, "kosmickrisp_metal_cache",
      kk_metal_cache_parse_size(
         debug_get_option("MESA_KK_CACHE_MAX_SIZE", NULL)));
}

static mtl_compiler *
kk_acquire_compiler(struct kk_device *dev)
{
   /* KK_WORKAROUND_11 */
   struct kk_physical_device *pdev = kk_device_physical(dev);
   if (pdev->settings.disabled_workarounds & BITFIELD64_BIT(11)) {
      mtl_compiler *handle = mtl_new_compiler(dev->mtl_handle);
      if (!handle)
         return NULL;
      dev->metal_disk = kk_create_metal_disk_cache(dev);
      dev->metal_cache = kk_metal_cache_create(dev->metal_disk, dev->mtl_handle,
                                               pdev->info.gpu_apple_family,
                                               handle, kk_metal_blob_budget());
      return handle;
   }

   call_once(&compilers_ht_once, kk_init_compiler_table);
   simple_mtx_lock(&compilers_ht_lock);

   struct hash_entry *ent =
      _mesa_hash_table_search(&compilers_ht, dev->mtl_handle);

   struct kk_mtl_compiler *compiler;
   if (ent == NULL) {
      compiler = rzalloc(NULL, struct kk_mtl_compiler);
      if (compiler == NULL) {
         simple_mtx_unlock(&compilers_ht_lock);
         return NULL;
      }

      compiler->handle = mtl_new_compiler(dev->mtl_handle);
      if (compiler->handle == NULL) {
         ralloc_free(compiler);
         simple_mtx_unlock(&compilers_ht_lock);
         return NULL;
      }

      compiler->disk = kk_create_metal_disk_cache(dev);
      compiler->cache = kk_metal_cache_create(compiler->disk, dev->mtl_handle,
                                             pdev->info.gpu_apple_family,
                                             compiler->handle,
                                             kk_metal_blob_budget());
      compiler->refcount = 1;
      _mesa_hash_table_insert(&compilers_ht, dev->mtl_handle, compiler);
   } else {
      compiler = ent->data;
      compiler->refcount++;
   }

   if (!dev->vk.disable_internal_cache && !compiler->cache) {
      if (!compiler->disk)
         compiler->disk = kk_create_metal_disk_cache(dev);
      compiler->cache = kk_metal_cache_create(compiler->disk, dev->mtl_handle,
                                             pdev->info.gpu_apple_family,
                                             compiler->handle,
                                             kk_metal_blob_budget());
   }
   if (!dev->vk.disable_internal_cache)
      dev->metal_cache = compiler->cache;
   simple_mtx_unlock(&compilers_ht_lock);
   return compiler->handle;
}

static void
kk_release_compiler(struct kk_device *dev)
{
   /* KK_WORKAROUND_11 */
   struct kk_physical_device *pdev = kk_device_physical(dev);
   if (pdev->settings.disabled_workarounds & BITFIELD64_BIT(11)) {
      kk_metal_cache_destroy(dev->metal_cache);
      disk_cache_destroy(dev->metal_disk);
      mtl_release(dev->mtl_compiler_handle);
      return;
   }

   simple_mtx_lock(&compilers_ht_lock);

   struct hash_entry *ent =
      _mesa_hash_table_search(&compilers_ht, dev->mtl_handle);
   if (ent != NULL) {
      struct kk_mtl_compiler *compiler = ent->data;
      --compiler->refcount;

      if (compiler->refcount == 0) {
         kk_metal_cache_destroy(compiler->cache);
         disk_cache_destroy(compiler->disk);
         _mesa_hash_table_remove(&compilers_ht, ent);
         mtl_release(compiler->handle);
         ralloc_free(compiler);
      }
   }

   simple_mtx_unlock(&compilers_ht_lock);
}

DERIVE_HASH_TABLE(mtl_sampler_packed);

static VkResult
kk_init_sampler_heap(struct kk_device *dev, struct kk_sampler_heap *h)
{
   h->ht = mtl_sampler_packed_table_create(NULL);
   if (!h->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* We optimistically size the table to fit the maximum number of samplers we
    * advertise. If this exceeds the hardware sampler limit, it is handled by
    * additional checks in `kk_sampler_heap_add_locked` */
   VkResult result = kk_query_table_init(dev, &h->table, MSL_MAX_SAMPLERS);

   if (result != VK_SUCCESS) {
      ralloc_free(h->ht);
      return result;
   }

   simple_mtx_init(&h->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
kk_destroy_sampler_heap(struct kk_device *dev, struct kk_sampler_heap *h)
{
   struct hash_entry *entry = _mesa_hash_table_next_entry(h->ht, NULL);
   while (entry) {
      struct kk_rc_sampler *sampler = (struct kk_rc_sampler *)entry->data;
      mtl_release(sampler->handle);
      entry = _mesa_hash_table_next_entry(h->ht, entry);
   }
   kk_query_table_finish(dev, &h->table);
   ralloc_free(h->ht);
   simple_mtx_destroy(&h->lock);
}

static VkResult
kk_sampler_heap_add_locked(struct kk_device *dev, struct kk_sampler_heap *h,
                           struct mtl_sampler_packed desc,
                           struct kk_rc_sampler **out)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   struct hash_entry *ent = _mesa_hash_table_search(h->ht, &desc);
   if (ent != NULL) {
      *out = ent->data;

      assert((*out)->refcount != 0);
      (*out)->refcount++;

      return VK_SUCCESS;
   }

   /* Constrain to device max sampler count */
   if (h->ht->entries >= pdev->info.max_sampler_count)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct kk_rc_sampler *rc = ralloc(h->ht, struct kk_rc_sampler);
   if (!rc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   mtl_sampler *handle = kk_sampler_create(dev, &desc);
   uint64_t gpu_id = mtl_sampler_get_gpu_resource_id(handle);

   uint32_t index;
   VkResult result = kk_query_table_add(dev, &h->table, gpu_id, &index);
   if (result != VK_SUCCESS) {
      mtl_release(handle);
      ralloc_free(rc);
      return result;
   }

   memset(rc, 0, sizeof(*rc));
   memcpy(&rc->key, &desc, sizeof(desc));
   rc->handle = handle;
   rc->refcount = 1;
   rc->index = index;

   _mesa_hash_table_insert(h->ht, &rc->key, rc);
   *out = rc;

   return VK_SUCCESS;
}

VkResult
kk_sampler_heap_add(struct kk_device *dev, struct mtl_sampler_packed desc,
                    struct kk_rc_sampler **out)
{
   struct kk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   VkResult result = kk_sampler_heap_add_locked(dev, h, desc, out);
   simple_mtx_unlock(&h->lock);

   return result;
}

static void
kk_sampler_heap_remove_locked(struct kk_device *dev, struct kk_sampler_heap *h,
                              struct kk_rc_sampler *rc)
{
   assert(rc->refcount != 0);
   rc->refcount--;

   if (rc->refcount == 0) {
      mtl_release(rc->handle);
      kk_query_table_remove(dev, &h->table, rc->index);
      _mesa_hash_table_remove_key(h->ht, &rc->key);
      ralloc_free(rc);
   }
}

void
kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc)
{
   struct kk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   kk_sampler_heap_remove_locked(dev, h, rc);
   simple_mtx_unlock(&h->lock);
}

static VkResult
kk_get_timestamp(struct vk_device *device, uint64_t *timestamp)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);

   uint64_t gpu_ns = mtl_device_get_gpu_timestamp(dev->mtl_handle);
   uint64_t frequency = mtl_device_timestamp_frequency(dev->mtl_handle);

   *timestamp =
      (uint64_t)(((unsigned __int128)gpu_ns * frequency) / 1000000000ull);
   return VK_SUCCESS;
}

__THREAD_INITIAL_EXEC enum kk_compile_origin kk_compile_origin;
__THREAD_INITIAL_EXEC bool kk_fail_on_compile;
__THREAD_INITIAL_EXEC bool kk_compile_required;

void
kk_log_compile(struct kk_device *dev, const char *kind, int64_t start)
{
   if (!dev->compile_log)
      return;

   static const char *const origins[] = {
      [KK_COMPILE_ORIGIN_DRAW] = "draw",
      [KK_COMPILE_ORIGIN_PREWARM] = "prewarm",
      [KK_COMPILE_ORIGIN_CREATE] = "create",
   };
   fprintf(dev->compile_log, "%s,%s,%lld,%lld\n", kind,
           origins[kk_compile_origin], (long long)start,
           (long long)os_time_get_nano());
}

__THREAD_INITIAL_EXEC blake3_hash kk_variant_parent_id;
__THREAD_INITIAL_EXEC bool kk_variant_parent_valid;
__THREAD_INITIAL_EXEC blake3_hash kk_variant_family_id;
__THREAD_INITIAL_EXEC bool kk_variant_family_valid;

static const char *const kk_variant_names[KK_VARIANT_COUNT] = {
   "enqueue", "worker_start", "fs_prepare", "fs_wait", "raster_prepare",
   "raster_wait", "pso_create", "pso_wait", "archive_lookup", "archive_load",
   "native_create", "complete", "first_demand", "ready_hit", "failure",
   "history_load", "ref_missing", "blob_missing", "archive_rejected",
   "pipeline_absent",
};

void
kk_variant_trace(struct kk_device *dev, enum kk_variant_phase phase,
                 const void *key, size_t key_size, int64_t start, int64_t end)
{
   if (!dev->variant_trace_enabled || phase >= KK_VARIANT_COUNT)
      return;
   uint64_t duration = end >= start ? end - start : 0;
   struct kk_variant_counter *counter = &dev->variant_counters[phase];
   p_atomic_inc(&counter->count);
   p_atomic_add(&counter->duration_ns, duration);
   if (kk_compile_origin == KK_COMPILE_ORIGIN_DRAW) {
      p_atomic_inc(&counter->draw_count);
      p_atomic_add(&counter->draw_duration_ns, duration);
   }
   if (!dev->variant_log || phase == KK_VARIANT_READY_HIT)
      return;
   if (!start && !end)
      start = end = os_time_get_nano();
   blake3_hash digest;
   char id[BLAKE3_HEX_LEN], parent[BLAKE3_HEX_LEN] = {0};
   char family[BLAKE3_HEX_LEN] = {0};
   _mesa_blake3_compute(key, key_size, digest);
   _mesa_blake3_format(id, digest);
   if (kk_variant_parent_valid)
      _mesa_blake3_format(parent, kk_variant_parent_id);
   if (kk_variant_family_valid)
      _mesa_blake3_format(family, kk_variant_family_id);
   uint64_t thread_id = (uintptr_t)pthread_self();
#ifdef __APPLE__
   pthread_threadid_np(NULL, &thread_id);
#endif
   static const char *const origins[] = {"draw", "prewarm", "create"};
   fprintf(dev->variant_log,
           "{\"event\":\"variant\",\"phase\":\"%s\","
           "\"request_id\":\"%s\",\"parent_id\":\"%s\","
           "\"family_id\":\"%s\",\"origin\":\"%s\","
           "\"thread_id\":%llu,\"start_ns\":%lld,\"end_ns\":%lld}\n",
           kk_variant_names[phase], id, parent,
           family, origins[kk_compile_origin],
           (unsigned long long)thread_id, (long long)start, (long long)end);
}

void
kk_variant_metal_event(void *data, const char *phase, const uint8_t *key,
                       size_t key_size, int64_t start, int64_t end)
{
   enum kk_variant_phase variant_phase;
   if (strcmp(phase, "archive_lookup") == 0)
      variant_phase = KK_VARIANT_ARCHIVE_LOOKUP;
   else if (strcmp(phase, "archive_load") == 0)
      variant_phase = KK_VARIANT_ARCHIVE_LOAD;
   else if (strcmp(phase, "native_create") == 0)
      variant_phase = KK_VARIANT_NATIVE_CREATE;
   else if (strcmp(phase, "ref_missing") == 0)
      variant_phase = KK_VARIANT_REF_MISSING;
   else if (strcmp(phase, "blob_missing") == 0)
      variant_phase = KK_VARIANT_BLOB_MISSING;
   else if (strcmp(phase, "archive_rejected") == 0)
      variant_phase = KK_VARIANT_ARCHIVE_REJECTED;
   else if (strcmp(phase, "pipeline_absent") == 0)
      variant_phase = KK_VARIANT_PIPELINE_ABSENT;
   else
      return;
   kk_variant_trace(data, variant_phase, key, key_size, start, end);
}

static void
kk_variant_trace_finish(struct kk_device *dev)
{
   if (dev->variant_stats) {
      for (unsigned i = 0; i < KK_VARIANT_COUNT; ++i) {
         const struct kk_variant_counter *counter = &dev->variant_counters[i];
         fprintf(dev->variant_stats,
                 "{\"event\":\"variant_summary\",\"phase\":\"%s\","
                 "\"count\":%llu,\"duration_ns\":%llu,"
                 "\"draw_count\":%llu,\"draw_duration_ns\":%llu}\n",
                 kk_variant_names[i], (unsigned long long)counter->count,
                 (unsigned long long)counter->duration_ns,
                 (unsigned long long)counter->draw_count,
                 (unsigned long long)counter->draw_duration_ns);
      }
      fclose(dev->variant_stats);
   }
   if (dev->variant_log)
      fclose(dev->variant_log);
}

static void
kk_cache_namespace(struct vk_physical_device *physical, char out[33])
{
   static const char hex[] = "0123456789abcdef";
   for (unsigned i = 0; i < VK_UUID_SIZE; ++i) {
      uint8_t value = physical->properties.pipelineCacheUUID[i];
      out[i * 2] = hex[value >> 4];
      out[i * 2 + 1] = hex[value & 15];
   }
   out[32] = 0;
}

void
kk_log_cache_shader(struct kk_device *dev,
                     const struct vk_shader_cache_trace *trace,
                     const char *outcome, int64_t start, int64_t end)
{
   if (!dev->cache_log || !trace)
      return;
   char ns[33], key[BLAKE3_HEX_LEN], precomp[BLAKE3_HEX_LEN];
   char flags[BLAKE3_HEX_LEN], partition[BLAKE3_HEX_LEN];
   char layout[BLAKE3_HEX_LEN], state[BLAKE3_HEX_LEN];
   kk_cache_namespace(dev->vk.physical, ns);
   _mesa_blake3_format(key, trace->key);
   _mesa_blake3_format(precomp, trace->precomp_key);
   _mesa_blake3_format(flags, trace->flags_hash);
   _mesa_blake3_format(partition, trace->partition_hash);
   _mesa_blake3_format(layout, trace->layout_hash);
   _mesa_blake3_format(state, trace->state_hash);
   fprintf(dev->cache_log,
           "{\"event\":\"shader\",\"outcome\":\"%s\",\"stage\":%u,"
           "\"namespace\":\"%s\",\"key\":\"%s\",\"precomp\":\"%s\","
           "\"flags\":\"%s\",\"partition\":\"%s\",\"layout\":\"%s\","
           "\"state\":\"%s\",\"start_ns\":%lld,\"end_ns\":%lld}\n",
           outcome, trace->stage, ns, key, precomp, flags, partition,
           layout, state, (long long)start, (long long)end);
}

void
kk_log_cache_component(struct kk_physical_device *physical,
                        VkShaderStageFlags stages, const char *name,
                        const blake3_hash component, const blake3_hash state)
{
   const char *path = debug_get_option("MESA_KK_CACHE_LOG", NULL);
   if (!path)
      return;
   FILE *file = fopen(path, "a");
   if (!file)
      return;
   char ns[33], value[BLAKE3_HEX_LEN], full[BLAKE3_HEX_LEN];
   kk_cache_namespace(&physical->vk, ns);
   _mesa_blake3_format(value, component);
   _mesa_blake3_format(full, state);
   fprintf(file, "{\"event\":\"state_component\",\"namespace\":\"%s\","
                 "\"stages\":%u,\"component\":\"%s\",\"hash\":\"%s\","
                 "\"state\":\"%s\"}\n", ns, stages, name, value, full);
   fclose(file);
}

void
kk_log_cache_pso(struct kk_device *dev, const char *event,
                  const void *key, size_t key_size)
{
   if (!dev->cache_log ||
       (!dev->cache_log_hits && strcmp(event, "pso_cache_hit") == 0))
      return;
   blake3_hash digest;
   char value[BLAKE3_HEX_LEN], ns[33];
   _mesa_blake3_compute(key, key_size, digest);
   _mesa_blake3_format(value, digest);
   kk_cache_namespace(dev->vk.physical, ns);
   static const char *const origins[] = {"draw", "prewarm", "create"};
   fprintf(dev->cache_log, "{\"event\":\"pso\",\"outcome\":\"%s\","
                           "\"origin\":\"%s\",\"namespace\":\"%s\","
                           "\"key\":\"%s\",\"time_ns\":%lld}\n",
           event, origins[kk_compile_origin], ns, value,
           (long long)os_time_get_nano());
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;
   struct kk_device *dev;

   dev = vk_zalloc2(&pdev->vk.instance->alloc, pAllocator, sizeof(*dev), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Fill the dispatch table we will expose to the users */
   dev->exposed_dispatch_table = kk_device_cmd_trampolines;
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &kk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dev->exposed_dispatch_table, &vk_common_device_entrypoints, false);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &kk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);

   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table, pCreateInfo,
                           pAllocator);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   dev->vk.shader_ops = &kk_device_shader_ops;
   dev->mtl_handle = pdev->mtl_dev_handle;
   dev->vk.command_buffer_ops = &kk_cmd_buffer_ops;
   dev->vk.command_dispatch_table = &dev->vk.dispatch_table;
   dev->vk.get_timestamp = kk_get_timestamp;

   /* Create a new Metal pipeline compiler for the device */
   dev->mtl_compiler_handle = kk_acquire_compiler(dev);
   if (dev->mtl_compiler_handle == NULL)
      goto fail_init;

   /* Initialize the device residency set before any bo is created. */
   simple_mtx_init(&dev->residency_set.mutex, mtx_plain);
   dev->residency_set.handle = mtl_new_residency_set(dev->mtl_handle);
   if (dev->residency_set.handle == NULL)
      goto fail_compiler;

   if (pCreateInfo->queueCreateInfoCount > 0) {
      result =
         kk_queue_init(dev, &dev->queue, &pCreateInfo->pQueueCreateInfos[0], 0);
      if (result != VK_SUCCESS)
         goto fail_vab_memory;
      dev->has_queue = true;
   }

   /* Also cache pipelines when applications pass VK_NULL_HANDLE. Weak
    * references avoid retaining every Metal pipeline for device lifetime. */
   struct vk_pipeline_cache_create_info cache_info = {
      .weak_ref = true,
      .skip_disk_cache = dev->vk.disable_internal_cache,
   };
   dev->vk.mem_cache = vk_pipeline_cache_create(&dev->vk, &cache_info, NULL);
   if (dev->vk.mem_cache == NULL) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_mem_cache;
   }

   if (!kk_pso_cache_init(&dev->pso_cache)) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_mem_cache;
   }
   simple_mtx_init(&dev->empty_fs_lock, mtx_plain);

   result = kk_device_init_meta(dev);
   if (result != VK_SUCCESS)
      goto fail_mem_cache;

   result = kk_query_table_init(dev, &dev->occlusion_queries,
                                KK_MAX_OCCLUSION_QUERIES);
   if (result != VK_SUCCESS)
      goto fail_meta;

   result = kk_init_sampler_heap(dev, &dev->samplers);
   if (result != VK_SUCCESS)
      goto fail_query_table;

   result = kk_device_init_lib(dev);
   if (result != VK_SUCCESS)
      goto fail_sampler_heap;

   if (pdev->settings.gpu_capture_enabled) {
      const char *capture_directory =
         debug_get_option("MESA_KK_GPU_CAPTURE_DIRECTORY", NULL);
      mtl_start_gpu_capture(dev->mtl_handle, capture_directory);
   }

   dev->capture_trigger = debug_get_option("MESA_KK_GPU_CAPTURE_TRIGGER", NULL);
   dev->capture_directory =
      debug_get_option("MESA_KK_GPU_CAPTURE_DIRECTORY", NULL);
   dev->capture_frames =
      CLAMP(debug_get_num_option("MESA_KK_GPU_CAPTURE_FRAMES", 3), 1, 120);
   dev->present_log_path = debug_get_option("MESA_KK_PRESENT_LOG", NULL);
   const char *compile_log_path = debug_get_option("MESA_KK_COMPILE_LOG", NULL);
   if (compile_log_path) {
      dev->compile_log = fopen(compile_log_path, "a");
      if (dev->compile_log)
         setvbuf(dev->compile_log, NULL, _IOLBF, 0);
      else
         fprintf(stderr, "KK cannot open compile log\n");
   }
   const char *cache_log_path = debug_get_option("MESA_KK_CACHE_LOG", NULL);
   if (cache_log_path) {
      dev->cache_log = fopen(cache_log_path, "a");
      if (dev->cache_log)
         setvbuf(dev->cache_log, NULL, _IOLBF, 0);
   }
   dev->cache_log_hits = debug_get_bool_option("MESA_KK_CACHE_LOG_HITS", false);
   const char *variant_log_path = debug_get_option("MESA_KK_VARIANT_LOG", NULL);
   const char *variant_stats_path =
      debug_get_option("MESA_KK_VARIANT_STATS", NULL);
   if (variant_log_path) {
      dev->variant_log = fopen(variant_log_path, "a");
      if (dev->variant_log)
         setvbuf(dev->variant_log, NULL, _IOLBF, 0);
   }
   if (variant_stats_path)
      dev->variant_stats = fopen(variant_stats_path, "a");
   dev->variant_trace_enabled = dev->variant_log || dev->variant_stats;
   *pDevice = kk_device_to_handle(dev);

   return VK_SUCCESS;

fail_sampler_heap:
   kk_destroy_sampler_heap(dev, &dev->samplers);
fail_query_table:
   kk_query_table_finish(dev, &dev->occlusion_queries);
fail_meta:
   kk_device_finish_meta(dev);
fail_mem_cache:
   if (dev->pso_cache.table) {
      kk_pso_cache_finish(&dev->pso_cache);
      simple_mtx_destroy(&dev->empty_fs_lock);
   }
   if (dev->vk.mem_cache)
      vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   if (dev->has_queue) {
      kk_queue_finish(dev, &dev->queue);
      dev->has_queue = false;
   }
fail_vab_memory:
   mtl_release(dev->residency_set.handle);
   simple_mtx_destroy(&dev->residency_set.mutex);
fail_compiler:
   kk_release_compiler(dev);
fail_init:
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(&dev->vk.alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   if (!dev)
      return;

   /* End capture before we start releasing resources. Otherwise, Metal capture
    * may run into issues. */
   if (pdev->settings.gpu_capture_enabled) {
      mtl_stop_gpu_capture();
   }

   mtl_bounded_capture_finish(dev);
   if (dev->present_log)
      fclose(dev->present_log);

   /* Meta first since it may destroy Vulkan objects */
   kk_device_finish_meta(dev);
   kk_device_finish_lib(dev);
   kk_pso_cache_finish(&dev->pso_cache);
   if (dev->compile_log)
      fclose(dev->compile_log);
   if (dev->cache_log)
      fclose(dev->cache_log);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   kk_device_finish_empty_fs(dev);
   simple_mtx_destroy(&dev->empty_fs_lock);
   kk_query_table_finish(dev, &dev->occlusion_queries);
   kk_destroy_sampler_heap(dev, &dev->samplers);

   /* Geometry heap */
   if (dev->heap)
      kk_destroy_bo(dev, dev->heap);

   if (dev->has_queue) {
      kk_queue_finish(dev, &dev->queue);
      dev->has_queue = false;
   }

   /* Release the residency set last once all BOs are released. */
   mtl_release(dev->residency_set.handle);
   simple_mtx_destroy(&dev->residency_set.mutex);

   kk_release_compiler(dev);
   kk_variant_trace_finish(dev);

   vk_device_finish(&dev->vk);

   vk_free(&dev->vk.alloc, dev);
}

/* We need to implement this ourselves so we give the fake ones for vk_common_*
 * to work when executing actual commands */
static VkResult VKAPI_CALL
kk_wait_variant_prewarm(VkDevice device)
{
   return kk_pso_prewarm_wait(kk_device_from_handle(device));
}

static VkBool32 VKAPI_CALL
kk_variant_prewarm_ready(VkDevice device)
{
   return kk_pso_prewarm_ready(kk_device_from_handle(device));
}

static PFN_vkVoidFunction
kk_device_get_proc_addr(const struct kk_device *device, const char *name)
{
   if (device == NULL || name == NULL)
      return NULL;
   if (debug_get_bool_option("MESA_KK_TEST_PREWARM", false)) {
      if (strcmp(name, "vkWaitForVariantPrewarmMESA") == 0)
         return (PFN_vkVoidFunction)kk_wait_variant_prewarm;
      if (strcmp(name, "vkIsVariantPrewarmReadyMESA") == 0)
         return (PFN_vkVoidFunction)kk_variant_prewarm_ready;
   }

   struct vk_instance *instance = device->vk.physical->instance;
   return vk_device_dispatch_table_get_if_supported(
      &device->exposed_dispatch_table, name, instance->app_info.api_version,
      &instance->enabled_extensions, &device->vk.enabled_extensions);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
kk_GetDeviceProcAddr(VkDevice _device, const char *pName)
{
   VK_FROM_HANDLE(kk_device, device, _device);
   return kk_device_get_proc_addr(device, pName);
}

/* Plain vkGetDeviceProcAddr export for loaders that dlopen the driver directly
 * (Wine/CrossOver win32u requires both vkGetInstanceProcAddr and this). */
PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *pName)
{
   return kk_GetDeviceProcAddr(device, pName);
}

void
kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_heap_from_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_add_buffer_to_residency_set(struct kk_device *dev, mtl_buffer *buffer)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, buffer);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, buffer);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_make_resources_resident(struct kk_device *dev)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_commit(dev->residency_set.handle);
   mtl_residency_set_request_residency(dev->residency_set.handle);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

/* VK_EXT_debug_utils */
VKAPI_ATTR VkResult VKAPI_CALL
kk_SetDebugUtilsObjectNameEXT(VkDevice device,
                              const VkDebugUtilsObjectNameInfoEXT *pNameInfo)
{
   VkResult result = vk_common_SetDebugUtilsObjectNameEXT(device, pNameInfo);
   if (result != VK_SUCCESS)
      return result;

   if (pNameInfo->pObjectName == NULL)
      return result;

#define CASE(obj_type, kk_type, vk_type)                                       \
   case VK_OBJECT_TYPE_##obj_type: {                                           \
      VK_FROM_HANDLE(kk_type, obj, (vk_type)pNameInfo->objectHandle);          \
      kk_type##_set_label(obj, pNameInfo->pObjectName);                        \
      break;                                                                   \
   }

   switch (pNameInfo->objectType) {
      CASE(COMMAND_BUFFER, kk_cmd_buffer, VkCommandBuffer);
      CASE(COMMAND_POOL, kk_cmd_pool, VkCommandPool);
      CASE(DEVICE_MEMORY, kk_device_memory, VkDeviceMemory);
      CASE(BUFFER, kk_buffer, VkBuffer);
      CASE(BUFFER_VIEW, kk_buffer_view, VkBufferView);
      CASE(IMAGE, kk_image, VkImage);
      CASE(IMAGE_VIEW, kk_image_view, VkImageView);
      CASE(DESCRIPTOR_POOL, kk_descriptor_pool, VkDescriptorPool);
      /* TODO: it would be nice to label pipelines and shaders, but we don't
       * have easy access to the relevant vulkan runtime objects inside
       * kk_compile_shaders(), where the Metal objects are set up.
       */
   default:
      break;
   }
#undef CASE

   return VK_SUCCESS;
}
