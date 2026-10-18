#ifndef KK_METAL_CACHE_H
#define KK_METAL_CACHE_H

#include "kosmickrisp/bridge/mtl_compiler.h"
#include "util/disk_cache.h"

struct kk_metal_cache;

struct kk_metal_capture_key {
   uint8_t vertex[CACHE_KEY_SIZE];
   uint8_t fragment[CACHE_KEY_SIZE];
};

struct kk_metal_trace {
   void (*callback)(void *data, const char *phase, const uint8_t *key,
                    size_t key_size, int64_t start, int64_t end);
   void *data;
};

struct kk_metal_trace kk_metal_cache_set_trace(struct kk_metal_trace trace);
uint32_t kk_metal_cache_parse_size(const char *value);

struct kk_metal_cache *kk_metal_cache_create(struct disk_cache *disk_cache,
                                            mtl_device *device,
                                            uint32_t gpu_family,
                                            mtl_compiler *compiler,
                                            uint64_t blob_budget);
void kk_metal_cache_flush(struct kk_metal_cache *cache);
void kk_metal_cache_request_flush(struct kk_metal_cache *cache);
void kk_metal_cache_destroy(struct kk_metal_cache *cache);

mtl_render_pipeline_state *kk_metal_cache_lookup_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor);
mtl_render_pipeline_state *kk_metal_cache_new_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor,
   const struct kk_metal_capture_key *capture_key, bool *archive_hit);
mtl_render_pipeline_state *kk_metal_cache_specialize_render_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_render_pipeline_descriptor *descriptor, mtl_render_pipeline_state *base);
mtl_compute_pipeline_state *kk_metal_cache_new_compute_pipeline(
   struct kk_metal_cache *cache, const uint8_t key[CACHE_KEY_SIZE],
   mtl_function_descriptor *function, uint64_t max_threads, bool *archive_hit);

#endif
