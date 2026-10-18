/*
 * Copyright 2026 Nilton Perim Neto
 * SPDX-License-Identifier: MIT
 */

#include "kk_pso_cache.h"
#include "kk_metal_cache.h"
#include "kk_variant_history.h"
#include "kk_physical_device.h"

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_debug.h"
#include "util/log.h"
#include "kk_format.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "util/format/u_format.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/u_string.h"

#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline_cache.h"
#include "vk_shader.h"

#define KK_PSO_MAX_ATTEMPTS 2

enum kk_entry_state {
   KK_ENTRY_IDLE,
   KK_ENTRY_QUEUED,
   KK_ENTRY_COMPILING,
   KK_ENTRY_READY,
   KK_ENTRY_FAILED,
};

struct kk_pso_entry {
   struct kk_pso_key key;
   mtl_render_pipeline_state *pso;
   cnd_t ready;
   enum kk_entry_state state;
   unsigned failures;
   VkResult result;
   bool prewarmed;
   bool drawn;
};

struct kk_variant_recipe {
   blake3_hash family;
   struct kk_fs_link_key fs;
   struct kk_pso_key pso;
   uint32_t sample_mask;
   uint32_t write_enables;
   uint32_t logic_enable;
   uint32_t logic_op;
   struct vk_color_blend_attachment_state input_blend[MAX_DRAW_BUFFERS];
};

static_assert(sizeof(struct kk_variant_recipe) == 736,
              "kk_variant_recipe is hashed and serialized");

#define KK_RECIPE_SCHEMA      1
#define KK_RECIPE_BLEND_BYTES 21
#define KK_RECIPE_FS_BYTES                                                     \
   (32 + 8 + 4 + 3 + 6 + MAX_DRAW_BUFFERS * KK_RECIPE_BLEND_BYTES)
#define KK_RECIPE_PSO_BYTES                                                    \
   (32 + 32 + MAX_DRAW_BUFFERS * 4 + 5 * 4 + MAX_DRAW_BUFFERS * 8)
#define KK_RECIPE_ENCODED_SIZE                                                 \
   (4 + 32 + KK_RECIPE_FS_BYTES + KK_RECIPE_PSO_BYTES + 16 +                   \
    MAX_DRAW_BUFFERS * KK_RECIPE_BLEND_BYTES)

struct kk_recipe_stream {
   uint8_t *data;
   size_t offset;
   bool valid;
};

static void
kk_put_u8(struct kk_recipe_stream *s, uint8_t value)
{
   s->data[s->offset++] = value;
}

static void
kk_put_u32(struct kk_recipe_stream *s, uint32_t value)
{
   for (unsigned i = 0; i < 4; ++i)
      kk_put_u8(s, value >> (8 * i));
}

static void
kk_put_u64(struct kk_recipe_stream *s, uint64_t value)
{
   for (unsigned i = 0; i < 8; ++i)
      kk_put_u8(s, value >> (8 * i));
}

static void
kk_put_bytes(struct kk_recipe_stream *s, const void *data, size_t size)
{
   memcpy(s->data + s->offset, data, size);
   s->offset += size;
}

static uint8_t
kk_get_u8(struct kk_recipe_stream *s)
{
   if (s->offset >= KK_RECIPE_ENCODED_SIZE) {
      s->valid = false;
      return 0;
   }
   return s->data[s->offset++];
}

static uint32_t
kk_get_u32(struct kk_recipe_stream *s)
{
   uint32_t value = 0;
   for (unsigned i = 0; i < 4; ++i)
      value |= (uint32_t)kk_get_u8(s) << (8 * i);
   return value;
}

static uint64_t
kk_get_u64(struct kk_recipe_stream *s)
{
   uint64_t value = 0;
   for (unsigned i = 0; i < 8; ++i)
      value |= (uint64_t)kk_get_u8(s) << (8 * i);
   return value;
}

static bool
kk_get_bool(struct kk_recipe_stream *s)
{
   uint8_t value = kk_get_u8(s);
   s->valid &= value <= 1;
   return value;
}

static void
kk_get_bytes(struct kk_recipe_stream *s, void *data, size_t size)
{
   if (s->offset + size > KK_RECIPE_ENCODED_SIZE) {
      s->valid = false;
      return;
   }
   memcpy(data, s->data + s->offset, size);
   s->offset += size;
}

static void
kk_put_blend(struct kk_recipe_stream *s,
             const struct vk_color_blend_attachment_state *b)
{
   kk_put_u8(s, b->blend_enable);
   kk_put_u8(s, b->src_color_blend_factor);
   kk_put_u8(s, b->dst_color_blend_factor);
   kk_put_u8(s, b->src_alpha_blend_factor);
   kk_put_u8(s, b->dst_alpha_blend_factor);
   kk_put_u8(s, b->write_mask);
   kk_put_u32(s, b->color_blend_op);
   kk_put_u32(s, b->alpha_blend_op);
   kk_put_u8(s, b->src_premultiplied);
   kk_put_u8(s, b->dst_premultiplied);
   kk_put_u32(s, b->blend_overlap);
   kk_put_u8(s, b->clamp_results);
}

static void
kk_get_blend(struct kk_recipe_stream *s,
             struct vk_color_blend_attachment_state *b)
{
   b->blend_enable = kk_get_bool(s);
   b->src_color_blend_factor = kk_get_u8(s);
   b->dst_color_blend_factor = kk_get_u8(s);
   b->src_alpha_blend_factor = kk_get_u8(s);
   b->dst_alpha_blend_factor = kk_get_u8(s);
   b->write_mask = kk_get_u8(s);
   b->color_blend_op = kk_get_u32(s);
   b->alpha_blend_op = kk_get_u32(s);
   b->src_premultiplied = kk_get_bool(s);
   b->dst_premultiplied = kk_get_bool(s);
   b->blend_overlap = kk_get_u32(s);
   b->clamp_results = kk_get_bool(s);
}

static void
kk_put_pso_blend(struct kk_recipe_stream *s, const struct kk_pso_blend *b)
{
   const uint8_t fields[] = {
      b->enable,    b->src_rgb,   b->dst_rgb,  b->op_rgb,
      b->src_alpha, b->dst_alpha, b->op_alpha, b->write_mask,
   };
   kk_put_bytes(s, fields, sizeof(fields));
}

static void
kk_get_pso_blend(struct kk_recipe_stream *s, struct kk_pso_blend *b)
{
   b->enable = kk_get_u8(s);
   b->src_rgb = kk_get_u8(s);
   b->dst_rgb = kk_get_u8(s);
   b->op_rgb = kk_get_u8(s);
   b->src_alpha = kk_get_u8(s);
   b->dst_alpha = kk_get_u8(s);
   b->op_alpha = kk_get_u8(s);
   b->write_mask = kk_get_u8(s);
}

static void
kk_recipe_encode(const struct kk_variant_recipe *recipe,
                 uint8_t data[KK_RECIPE_ENCODED_SIZE])
{
   struct kk_recipe_stream s = {.data = data};
   kk_put_u32(&s, KK_RECIPE_SCHEMA);
   kk_put_bytes(&s, recipe->family, sizeof(recipe->family));
   const struct kk_fs_link_key *fs = &recipe->fs;
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_put_u32(&s, fs->rt_formats[i]);
   kk_put_u64(&s, fs->missing_inputs);
   kk_put_u32(&s, fs->advanced_blend_mask);
   kk_put_u8(&s, fs->logic_op);
   kk_put_u8(&s, fs->num_cull_distances);
   kk_put_u8(&s, fs->primitive_id_slot);
   kk_put_u8(&s, fs->logic_op_enable);
   kk_put_u8(&s, fs->shader_blend);
   kk_put_u8(&s, fs->remove_depth);
   kk_put_u8(&s, fs->alpha_to_coverage);
   kk_put_u8(&s, fs->alpha_to_one);
   kk_put_u8(&s, fs->sample_mask);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_put_blend(&s, &fs->blend[i]);
   const struct kk_pso_key *pso = &recipe->pso;
   kk_put_bytes(&s, pso->rast, sizeof(pso->rast));
   kk_put_bytes(&s, pso->fs, sizeof(pso->fs));
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_put_u32(&s, pso->rt_formats[i]);
   kk_put_u32(&s, pso->sample_count);
   kk_put_u32(&s, pso->topology_class);
   kk_put_u32(&s, pso->amplification);
   kk_put_u32(&s, pso->alpha_to_coverage);
   kk_put_u32(&s, pso->alpha_to_one);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_put_pso_blend(&s, &pso->blend[i]);
   kk_put_u32(&s, recipe->sample_mask);
   kk_put_u32(&s, recipe->write_enables);
   kk_put_u32(&s, recipe->logic_enable);
   kk_put_u32(&s, recipe->logic_op);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_put_blend(&s, &recipe->input_blend[i]);
   assert(s.offset == KK_RECIPE_ENCODED_SIZE);
}

static bool
kk_recipe_decode(const uint8_t data[KK_RECIPE_ENCODED_SIZE],
                 struct kk_variant_recipe *recipe)
{
   struct kk_recipe_stream s = {.data = (uint8_t *)data, .valid = true};
   memset(recipe, 0, sizeof(*recipe));
   if (kk_get_u32(&s) != KK_RECIPE_SCHEMA)
      return false;
   kk_get_bytes(&s, recipe->family, sizeof(recipe->family));
   struct kk_fs_link_key *fs = &recipe->fs;
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      fs->rt_formats[i] = kk_get_u32(&s);
   fs->missing_inputs = kk_get_u64(&s);
   fs->advanced_blend_mask = kk_get_u32(&s);
   fs->logic_op = kk_get_u8(&s);
   fs->num_cull_distances = kk_get_u8(&s);
   fs->primitive_id_slot = kk_get_u8(&s);
   fs->logic_op_enable = kk_get_bool(&s);
   fs->shader_blend = kk_get_bool(&s);
   fs->remove_depth = kk_get_bool(&s);
   fs->alpha_to_coverage = kk_get_bool(&s);
   fs->alpha_to_one = kk_get_bool(&s);
   fs->sample_mask = kk_get_bool(&s);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_get_blend(&s, &fs->blend[i]);
   struct kk_pso_key *pso = &recipe->pso;
   kk_get_bytes(&s, pso->rast, sizeof(pso->rast));
   kk_get_bytes(&s, pso->fs, sizeof(pso->fs));
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      pso->rt_formats[i] = kk_get_u32(&s);
   pso->sample_count = kk_get_u32(&s);
   pso->topology_class = kk_get_u32(&s);
   pso->amplification = kk_get_u32(&s);
   pso->alpha_to_coverage = kk_get_u32(&s);
   pso->alpha_to_one = kk_get_u32(&s);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_get_pso_blend(&s, &pso->blend[i]);
   recipe->sample_mask = kk_get_u32(&s);
   recipe->write_enables = kk_get_u32(&s);
   recipe->logic_enable = kk_get_u32(&s);
   recipe->logic_op = kk_get_u32(&s);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      kk_get_blend(&s, &recipe->input_blend[i]);
   return s.valid && s.offset == KK_RECIPE_ENCODED_SIZE;
}

struct kk_recipe_entry {
   struct kk_variant_recipe recipe;
   struct kk_pso_key native_key;
   mtl_render_pipeline_state *pso;
   cnd_t ready;
   enum kk_entry_state state;
   unsigned failures;
   VkResult result;
   bool drawn;
   bool demanded;
   bool prewarmed;
};

struct mtl_compile_phase
kk_pso_phase_begin(struct kk_device *dev, const char *operation,
                    const void *key, size_t size)
{
   if (!mtl_compile_phase_enabled())
      return (struct mtl_compile_phase){0};
   blake3_hash digest;
   char value[BLAKE3_HEX_LEN];
   _mesa_blake3_compute(key, size, digest);
   _mesa_blake3_format(value, digest);
   static const char *const origins[] = {"draw", "prewarm", "create"};
   return mtl_compile_phase_begin(operation, dev, value,
                                  origins[kk_compile_origin]);
}

static uint32_t
kk_recipe_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct kk_variant_recipe));
}

static bool
kk_recipe_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct kk_variant_recipe)) == 0;
}

static uint32_t
kk_family_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(blake3_hash));
}

static bool
kk_family_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(blake3_hash)) == 0;
}

static struct kk_recipe_entry *
kk_recipe_insert_locked(struct kk_pso_cache *cache, uint32_t hash,
                        const struct kk_variant_recipe *recipe)
{
   if (!cache->recipes || cache->recipe_count >= cache->recipe_limit)
      return NULL;
   struct kk_recipe_entry *entry =
      rzalloc(cache->recipes, struct kk_recipe_entry);
   if (!entry)
      return NULL;
   entry->recipe = *recipe;
   if (cnd_init(&entry->ready) != thrd_success) {
      ralloc_free(entry);
      return NULL;
   }
   if (!_mesa_hash_table_insert_pre_hashed(cache->recipes, hash,
                                           &entry->recipe, entry)) {
      cnd_destroy(&entry->ready);
      ralloc_free(entry);
      return NULL;
   }
   cache->recipe_count++;
   return entry;
}

static struct kk_recipe_entry *
kk_recipe_lookup(struct kk_pso_cache *cache,
                 const struct kk_variant_recipe *recipe, bool insert)
{
   if (!cache->recipes)
      return NULL;
   uint32_t hash = kk_recipe_hash(recipe);
   u_rwlock_rdlock(&cache->index_lock);
   struct hash_entry *he =
      _mesa_hash_table_search_pre_hashed(cache->recipes, hash, recipe);
   struct kk_recipe_entry *entry = he ? he->data : NULL;
   u_rwlock_rdunlock(&cache->index_lock);
   if (entry || !insert)
      return entry;
   u_rwlock_wrlock(&cache->index_lock);
   he = _mesa_hash_table_search_pre_hashed(cache->recipes, hash, recipe);
   entry = he ? he->data : kk_recipe_insert_locked(cache, hash, recipe);
   u_rwlock_wrunlock(&cache->index_lock);
   return entry;
}

static uint32_t
kk_pso_key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct kk_pso_key));
}

static bool
kk_pso_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct kk_pso_key)) == 0;
}

void
kk_pso_key_init(struct kk_pso_key *key)
{
   memset(key, 0, sizeof(*key));
   for (uint32_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
      key->rt_formats[i] = MTL_PIXEL_FORMAT_INVALID;
      key->blend[i] = (struct kk_pso_blend){
         .src_rgb = MTL_BLEND_FACTOR_ONE,
         .dst_rgb = MTL_BLEND_FACTOR_ZERO,
         .op_rgb = MTL_BLEND_OPERATION_ADD,
         .src_alpha = MTL_BLEND_FACTOR_ONE,
         .dst_alpha = MTL_BLEND_FACTOR_ZERO,
         .op_alpha = MTL_BLEND_OPERATION_ADD,
         .write_mask = MTL_COLOR_WRITE_MASK_RED | MTL_COLOR_WRITE_MASK_GREEN |
                       MTL_COLOR_WRITE_MASK_BLUE | MTL_COLOR_WRITE_MASK_ALPHA,
      };
   }
}

static mtl_render_pipeline_descriptor *
kk_pso_descriptor(mtl_function_descriptor *vs, mtl_function_descriptor *fs,
                  const struct kk_pso_key *key)
{
   mtl_render_pipeline_descriptor *desc = mtl_new_render_pipeline_descriptor();
   mtl_render_pipeline_descriptor_set_vertex_shader(desc, vs);
   if (fs)
      mtl_render_pipeline_descriptor_set_fragment_shader(desc, fs);

   mtl_render_pipeline_descriptor_set_input_primitive_topology(
      desc, key->topology_class);

   for (uint8_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
      if (key->rt_formats[i] == MTL_PIXEL_FORMAT_INVALID)
         continue;

      if (key->rt_formats[i] == MTL_PIXEL_FORMAT_UNSPECIALIZED) {
         mtl_render_pipeline_descriptor_set_color_attachment_unspecialized(
            desc, i);
         continue;
      }

      const struct kk_pso_blend *blend = &key->blend[i];
      mtl_render_pipeline_descriptor_set_color_attachment_format(
         desc, i, key->rt_formats[i]);
      mtl_render_pipeline_descriptor_set_color_attachment_blend(
         desc, i, blend->enable, blend->src_rgb, blend->dst_rgb,
         blend->op_rgb, blend->src_alpha, blend->dst_alpha, blend->op_alpha,
         blend->write_mask);
   }

   if (key->amplification)
      mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(
         desc, key->amplification);

   if (key->sample_count)
      mtl_render_pipeline_descriptor_set_raster_sample_count(desc,
                                                             key->sample_count);

   mtl_render_pipeline_descriptor_set_multisample_alpha(
      desc, key->alpha_to_coverage, key->alpha_to_one);

   return desc;
}

static bool
kk_pso_key_specializable(const struct kk_pso_key *key)
{
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i)
      if (key->rt_formats[i] != MTL_PIXEL_FORMAT_INVALID &&
          key->rt_formats[i] != MTL_PIXEL_FORMAT_UNSPECIALIZED)
         return true;
   return false;
}

static void
kk_pso_base_key(const struct kk_pso_key *key, struct kk_pso_key *base)
{
   *base = *key;
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i) {
      base->rt_formats[i] = MTL_PIXEL_FORMAT_UNSPECIALIZED;
      memset(&base->blend[i], 0, sizeof(base->blend[i]));
   }
}

static mtl_render_pipeline_state *
kk_specialize_render_pipeline(struct kk_device *dev,
                              mtl_function_descriptor *vs,
                              mtl_function_descriptor *fs,
                              const struct kk_pso_key *key,
                              mtl_render_pipeline_state *base)
{
   mtl_render_pipeline_descriptor *desc = kk_pso_descriptor(vs, fs, key);
   int64_t start = os_time_get_nano();
   struct mtl_compile_phase phase =
      kk_pso_phase_begin(dev, "native_specialize", key, sizeof(*key));
   blake3_hash digest;
   _mesa_blake3_compute(key, sizeof(*key), digest);
   struct kk_metal_trace previous_trace =
      kk_metal_cache_set_trace((struct kk_metal_trace){
         .callback = dev->variant_trace_enabled
                        ? kk_variant_metal_event : NULL,
         .data = dev,
   });
   mtl_render_pipeline_state *pso = dev->metal_cache ?
      kk_metal_cache_specialize_render_pipeline(dev->metal_cache, digest,
                                                 desc, base) :
      mtl_specialize_render_pipeline(dev->mtl_compiler_handle, desc, base);
   kk_metal_cache_set_trace(previous_trace);
   mtl_compile_phase_end(phase, pso != NULL);
   mtl_release(desc);
   if (pso) {
      kk_variant_trace(dev, KK_VARIANT_PSO_CREATE, key, sizeof(*key),
                       start, os_time_get_nano());
      kk_log_compile(dev, "pso_specialize", start);
   }
   kk_log_cache_pso(dev, pso ? "specialized" : "specialize_failed",
                    key, sizeof(*key));
   return pso;
}

mtl_render_pipeline_state *
kk_new_render_pipeline(struct kk_device *dev, mtl_function_descriptor *vs,
                       mtl_function_descriptor *fs,
                       const struct kk_pso_key *key, bool lookup_only)
{
   mtl_render_pipeline_descriptor *desc = kk_pso_descriptor(vs, fs, key);

   blake3_hash digest;
   _mesa_blake3_compute(key, sizeof(*key), digest);
   bool archive_hit = false;
   int64_t start = os_time_get_nano();
   struct mtl_compile_phase phase =
      kk_pso_phase_begin(dev, lookup_only ? "native_lookup" : "native_pipeline",
                         key, sizeof(*key));
   struct kk_metal_trace previous_trace =
      kk_metal_cache_set_trace((struct kk_metal_trace){
         .callback = dev->variant_trace_enabled
                        ? kk_variant_metal_event : NULL,
         .data = dev,
   });
   mtl_render_pipeline_state *pso = NULL;
   if (lookup_only) {
      if (dev->metal_cache)
         pso = kk_metal_cache_lookup_render_pipeline(
            dev->metal_cache, digest, desc);
      archive_hit = pso != NULL;
   } else {
      pso = dev->metal_cache ?
         kk_metal_cache_new_render_pipeline(
            dev->metal_cache, digest, desc, &archive_hit) :
         mtl_new_render_pipeline(dev->mtl_compiler_handle, desc);
   }
   kk_metal_cache_set_trace(previous_trace);
   mtl_compile_phase_end(phase, pso != NULL);
   if (lookup_only && !pso) {
      kk_log_cache_pso(dev, "archive_miss", key, sizeof(*key));
      mtl_release(desc);
      return NULL;
   }
   kk_variant_trace(dev, KK_VARIANT_PSO_CREATE, key, sizeof(*key),
                    start, os_time_get_nano());
   kk_log_compile(dev, archive_hit ? "pso_archive" : "pso", start);
   kk_log_cache_pso(dev, archive_hit ? "archive_hit" : "compiler_fallback",
                     key, sizeof(*key));
   mtl_release(desc);
   return pso;
}

bool
kk_pso_cache_init(struct kk_pso_cache *cache)
{
   if (mtx_init(&cache->lock, mtx_plain) != thrd_success)
      return false;
   if (u_rwlock_init(&cache->index_lock)) {
      mtx_destroy(&cache->lock);
      return false;
   }
   if (cnd_init(&cache->jobs_done) != thrd_success) {
      u_rwlock_destroy(&cache->index_lock);
      mtx_destroy(&cache->lock);
      return false;
   }
   if (cnd_init(&cache->draw_idle) != thrd_success) {
      cnd_destroy(&cache->jobs_done);
      u_rwlock_destroy(&cache->index_lock);
      mtx_destroy(&cache->lock);
      return false;
   }
   cache->recipe_limit =
      CLAMP(debug_get_num_option("MESA_KK_VARIANT_RECIPE_LIMIT", 4096),
            1, 8192);
   cache->table =
      _mesa_hash_table_create(NULL, kk_pso_key_hash, kk_pso_key_equal);
   cache->recipes =
      _mesa_hash_table_create(NULL, kk_recipe_hash, kk_recipe_equal);
   cache->families =
      _mesa_hash_table_create(NULL, kk_family_hash, kk_family_equal);
   if (!cache->table || !cache->recipes || !cache->families) {
      _mesa_hash_table_destroy(cache->table, NULL);
      _mesa_hash_table_destroy(cache->recipes, NULL);
      _mesa_hash_table_destroy(cache->families, NULL);
      cache->table = NULL;
      cache->recipes = NULL;
      cache->families = NULL;
      cnd_destroy(&cache->draw_idle);
      cnd_destroy(&cache->jobs_done);
      u_rwlock_destroy(&cache->index_lock);
      mtx_destroy(&cache->lock);
      return false;
   }
   struct kk_device *dev = container_of(cache, struct kk_device, pso_cache);
   struct disk_cache *disk = kk_device_physical(dev)->vk.disk_cache;
   if (!dev->vk.disable_internal_cache &&
       !debug_get_bool_option("MESA_KK_PSO_CACHE_DISABLE", false)) {
      cache->history = kk_variant_history_create(disk, KK_RECIPE_ENCODED_SIZE);
      cache->limited = !cache->history;
   }
   bool background = util_queue_init(&cache->prewarm_queue, "kk-pso", 64, 2,
                                    UTIL_QUEUE_INIT_RESIZE_IF_FULL, NULL);
   bool exact = util_queue_init(&cache->exact_queue, "kk-exact", 32, 1,
                               UTIL_QUEUE_INIT_RESIZE_IF_FULL, NULL);
   cache->limited |= !background || !exact;
   return true;
}

void
kk_pso_cache_finish(struct kk_pso_cache *cache)
{
   if (util_queue_is_initialized(&cache->exact_queue)) {
      util_queue_finish(&cache->exact_queue);
      util_queue_destroy(&cache->exact_queue);
   }
   if (util_queue_is_initialized(&cache->prewarm_queue)) {
      util_queue_finish(&cache->prewarm_queue);
      util_queue_destroy(&cache->prewarm_queue);
   }
   cache->summary.recipes =
      cache->recipes ? _mesa_hash_table_num_entries(cache->recipes) : 0;
   cache->summary.families =
      cache->families ? _mesa_hash_table_num_entries(cache->families) : 0;
   cache->summary.native =
      cache->table ? _mesa_hash_table_num_entries(cache->table) : 0;
   if (cache->recipes) {
      hash_table_foreach(cache->recipes, he) {
         const struct kk_recipe_entry *entry = he->data;
         cache->summary.recipes_ready += entry->state == KK_ENTRY_READY;
      }
   }
   if (cache->table) {
      hash_table_foreach(cache->table, he) {
         const struct kk_pso_entry *entry = he->data;
         cache->summary.native_ready += entry->state == KK_ENTRY_READY;
         cache->summary.native_failed += entry->state == KK_ENTRY_FAILED;
      }
   }
   cache->summary.bytes =
      (uint64_t)cache->summary.native * sizeof(struct kk_pso_entry) +
      (uint64_t)cache->summary.recipes * sizeof(struct kk_recipe_entry) +
      (uint64_t)cache->summary.families * sizeof(blake3_hash);
   kk_variant_history_destroy(cache->history);
   cache->history = NULL;
   if (cache->recipes) {
      hash_table_foreach(cache->recipes, he) {
         struct kk_recipe_entry *entry = he->data;
         cnd_destroy(&entry->ready);
      }
      _mesa_hash_table_destroy(cache->recipes, NULL);
   }
   if (cache->families)
      _mesa_hash_table_destroy(cache->families, NULL);
   if (cache->table) {
      hash_table_foreach(cache->table, he) {
         struct kk_pso_entry *entry = he->data;
         if (entry->prewarmed && !entry->drawn) {
            struct kk_device *dev =
               container_of(cache, struct kk_device, pso_cache);
            kk_log_cache_pso(dev, "prewarm_unused", &entry->key,
                              sizeof(entry->key));
         }
         mtl_release(entry->pso);
         cnd_destroy(&entry->ready);
      }
      _mesa_hash_table_destroy(cache->table, NULL);
      cache->table = NULL;
   }
   u_rwlock_destroy(&cache->index_lock);
   mtx_destroy(&cache->lock);
   cnd_destroy(&cache->jobs_done);
   cnd_destroy(&cache->draw_idle);
}

VkResult
kk_pso_prewarm_wait(struct kk_device *dev)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   mtx_lock(&cache->lock);
   while (cache->pending_jobs)
      cnd_wait(&cache->jobs_done, &cache->lock);
   VkResult result = cache->failed_jobs ? VK_ERROR_UNKNOWN :
      cache->limited ? VK_NOT_READY : VK_SUCCESS;
   mtx_unlock(&cache->lock);
   return result;
}

VkBool32
kk_pso_prewarm_ready(struct kk_device *dev)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   mtx_lock(&cache->lock);
   bool ready = !cache->pending_jobs && !cache->failed_jobs && !cache->limited;
   mtx_unlock(&cache->lock);
   return ready;
}

static bool
kk_entry_retryable(enum kk_entry_state state, unsigned failures)
{
   return state == KK_ENTRY_FAILED && failures < KK_PSO_MAX_ATTEMPTS &&
          kk_compile_origin != KK_COMPILE_ORIGIN_PREWARM;
}

static VkResult
kk_entry_status(enum kk_entry_state state, unsigned failures, VkResult result,
                bool lookup_only)
{
   if (state == KK_ENTRY_READY)
      return VK_SUCCESS;
   if (state == KK_ENTRY_FAILED &&
       (!lookup_only || !kk_entry_retryable(state, failures)))
      return result;
   return VK_NOT_READY;
}

static struct kk_pso_result
kk_pso_entry_get(struct kk_device *dev, struct kk_pso_entry *entry,
                 const struct kk_pso_key *key, bool wait, bool *compile)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   if (kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM &&
       !p_atomic_read(&entry->prewarmed))
      p_atomic_set(&entry->prewarmed, true);
   mtl_render_pipeline_state *pso = p_atomic_read(&entry->pso);
   VkResult status = VK_SUCCESS;
   if (!pso) {
      int64_t start = 0;
      mtx_lock(&cache->lock);
      bool waited = entry->state == KK_ENTRY_COMPILING;
      if (waited && !wait) {
         mtx_unlock(&cache->lock);
         kk_log_cache_pso(dev, "pso_in_flight", key, sizeof(*key));
         return (struct kk_pso_result){ .status = VK_NOT_READY };
      }
      if (waited) {
         kk_log_cache_pso(dev, "pso_wait", key, sizeof(*key));
         start = os_time_get_nano();
         struct mtl_compile_phase phase =
            kk_pso_phase_begin(dev, "pso_wait", key, sizeof(*key));
         while (entry->state == KK_ENTRY_COMPILING)
            cnd_wait(&entry->ready, &cache->lock);
         mtl_compile_phase_end(phase, true);
      }
      if (compile && !waited &&
          kk_entry_retryable(entry->state, entry->failures)) {
         entry->state = KK_ENTRY_COMPILING;
         mtx_unlock(&cache->lock);
         *compile = true;
         kk_log_cache_pso(dev, "pso_retry", key, sizeof(*key));
         return (struct kk_pso_result){ .status = VK_SUCCESS };
      }
      pso = entry->pso;
      status = kk_entry_status(entry->state, entry->failures, entry->result,
                               !wait);
      mtx_unlock(&cache->lock);
      if (waited) {
         kk_log_compile(dev, "pso_wait", start);
         kk_variant_trace(dev, KK_VARIANT_PSO_WAIT, key, sizeof(*key),
                          start, os_time_get_nano());
      }
   }
   if (KK_DEBUG(PSO))
      mesa_logi("PSO cache hit");
   if (pso && kk_compile_origin == KK_COMPILE_ORIGIN_DRAW &&
       !p_atomic_read(&entry->drawn) && !p_atomic_xchg(&entry->drawn, true) &&
       p_atomic_read(&entry->prewarmed))
      kk_log_cache_pso(dev, "prewarm_match", key, sizeof(*key));
   kk_log_cache_pso(dev, pso ? "pso_cache_hit" : "pso_cache_failed",
                     key, sizeof(*key));
   if (pso)
      kk_variant_trace(dev, KK_VARIANT_READY_HIT, key, sizeof(*key), 0, 0);
   return (struct kk_pso_result){ pso, status };
}

static struct kk_pso_entry *
kk_pso_entry_insert_locked(struct kk_pso_cache *cache, uint32_t hash,
                           const struct kk_pso_key *key,
                           mtl_render_pipeline_state *pso)
{
   struct kk_pso_entry *entry = rzalloc(cache->table, struct kk_pso_entry);
   if (entry == NULL)
      return NULL;
   entry->key = *key;
   entry->pso = pso;
   entry->state = pso ? KK_ENTRY_READY : KK_ENTRY_COMPILING;
   entry->prewarmed = kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM;
   entry->drawn = kk_compile_origin == KK_COMPILE_ORIGIN_DRAW;
   if (cnd_init(&entry->ready) != thrd_success) {
      ralloc_free(entry);
      return NULL;
   }
   if (!_mesa_hash_table_insert_pre_hashed(cache->table, hash, &entry->key,
                                          entry)) {
      cnd_destroy(&entry->ready);
      ralloc_free(entry);
      return NULL;
   }
   return entry;
}

static mtl_render_pipeline_state *
kk_pso_build(struct kk_device *dev, const struct kk_pso_key *key,
             mtl_function_descriptor *vs, mtl_function_descriptor *fs,
             bool lookup_only, bool specialize)
{
   if (!specialize || !kk_pso_key_specializable(key))
      return kk_new_render_pipeline(dev, vs, fs, key, lookup_only);

   struct kk_pso_key base_key;
   kk_pso_base_key(key, &base_key);
   const struct kk_pso_request lookup = { .lookup_only = true };
   struct kk_pso_result base =
      kk_pso_cache_get(dev, &base_key, vs, fs, lookup_only ? &lookup : NULL);
   if (base.pso) {
      mtl_render_pipeline_state *pso =
         kk_specialize_render_pipeline(dev, vs, fs, key, base.pso);
      if (pso)
         return pso;
   }
   if (lookup_only)
      return NULL;
   return kk_new_render_pipeline(dev, vs, fs, key, false);
}

static struct kk_pso_result
kk_pso_entry_compile(struct kk_device *dev, struct kk_pso_entry *entry,
                     const struct kk_pso_key *key, mtl_function_descriptor *vs,
                     mtl_function_descriptor *fs, bool specialize)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   mtl_render_pipeline_state *pso =
      kk_pso_build(dev, key, vs, fs, false, specialize);
   VkResult status = pso ? VK_SUCCESS : VK_ERROR_UNKNOWN;
   mtx_lock(&cache->lock);
   if (pso) {
      p_atomic_set(&entry->pso, pso);
      entry->state = KK_ENTRY_READY;
   } else {
      entry->failures++;
      entry->result = status;
      entry->state = KK_ENTRY_FAILED;
   }
   cnd_broadcast(&entry->ready);
   mtx_unlock(&cache->lock);
   kk_variant_trace(dev, pso ? KK_VARIANT_COMPLETE : KK_VARIANT_FAILURE,
                    key, sizeof(*key), 0, 0);
   return (struct kk_pso_result){ pso, status };
}

static struct kk_pso_result
kk_pso_found(struct kk_device *dev, struct kk_pso_entry *entry,
             const struct kk_pso_key *key, mtl_function_descriptor *vs,
             mtl_function_descriptor *fs, bool lookup_only, bool specialize)
{
   bool compile = false;
   struct kk_pso_result result = kk_pso_entry_get(
      dev, entry, key, !lookup_only, lookup_only ? NULL : &compile);
   if (compile)
      return kk_pso_entry_compile(dev, entry, key, vs, fs, specialize);
   return result;
}

struct kk_pso_result
kk_pso_cache_get(struct kk_device *dev, const struct kk_pso_key *key,
                 mtl_function_descriptor *vs, mtl_function_descriptor *fs,
                 const struct kk_pso_request *request)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   uint32_t hash = kk_pso_key_hash(key);
   bool lookup_only = request && request->lookup_only;
   bool specialize = !(request && request->native_primitive_id);

   u_rwlock_rdlock(&cache->index_lock);
   struct hash_entry *he =
      _mesa_hash_table_search_pre_hashed(cache->table, hash, key);
   struct kk_pso_entry *found = he ? he->data : NULL;
   u_rwlock_rdunlock(&cache->index_lock);
   if (found)
      return kk_pso_found(dev, found, key, vs, fs, lookup_only, specialize);

   if (lookup_only) {
      mtl_render_pipeline_state *pso =
         kk_pso_build(dev, key, vs, fs, true, specialize);
      if (!pso)
         return (struct kk_pso_result){ .status = VK_NOT_READY };
      u_rwlock_wrlock(&cache->index_lock);
      he = _mesa_hash_table_search_pre_hashed(cache->table, hash, key);
      found = he ? he->data : NULL;
      struct kk_pso_entry *entry =
         found ? NULL : kk_pso_entry_insert_locked(cache, hash, key, pso);
      u_rwlock_wrunlock(&cache->index_lock);
      if (found) {
         mtl_release(pso);
         return kk_pso_found(dev, found, key, vs, fs, true, specialize);
      }
      if (!entry) {
         mtl_release(pso);
         return (struct kk_pso_result){ .status = VK_ERROR_OUT_OF_HOST_MEMORY };
      }
      kk_variant_trace(dev, KK_VARIANT_COMPLETE, key, sizeof(*key), 0, 0);
      return (struct kk_pso_result){ pso, VK_SUCCESS };
   }

   u_rwlock_wrlock(&cache->index_lock);
   he = _mesa_hash_table_search_pre_hashed(cache->table, hash, key);
   found = he ? he->data : NULL;
   struct kk_pso_entry *entry =
      found ? NULL : kk_pso_entry_insert_locked(cache, hash, key, NULL);
   u_rwlock_wrunlock(&cache->index_lock);
   if (found)
      return kk_pso_found(dev, found, key, vs, fs, false, specialize);
   if (entry == NULL)
      return (struct kk_pso_result){ .status = VK_ERROR_OUT_OF_HOST_MEMORY };

   kk_log_cache_pso(dev, kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM ?
                          "prewarm_miss" : "pso_cache_miss",
                     key, sizeof(*key));
   if (KK_DEBUG(PSO))
      mesa_logi("PSO compile");
   return kk_pso_entry_compile(dev, entry, key, vs, fs, specialize);
}

mtl_render_pipeline_state *
kk_pso_cache_acquire(struct kk_device *dev, const struct kk_pso_key *key,
                     mtl_function_descriptor *vs, mtl_function_descriptor *fs,
                     const struct kk_pso_request *request)
{
   struct kk_pso_result result = kk_pso_cache_get(dev, key, vs, fs, request);
   return result.pso ? mtl_retain(result.pso) : NULL;
}

static nir_alu_type
kk_format_base_type(enum pipe_format format)
{
   if (util_format_is_pure_sint(format))
      return nir_type_int;
   if (util_format_is_pure_uint(format))
      return nir_type_uint;
   return nir_type_float;
}

static enum mtl_primitive_topology_class
kk_link_topology_class(const struct kk_link_state *ls)
{
   struct kk_shader *gs = ls->shaders[MESA_SHADER_GEOMETRY];
   struct kk_shader *tes = ls->shaders[MESA_SHADER_TESS_EVAL];
   VkPrimitiveTopology topology = ls->dyn->ia.primitive_topology;

   enum mesa_prim prim = MESA_PRIM_COUNT;
   if (gs)
      prim = u_reduced_prim(gs->info.gs.mode);
   else if (tes)
      prim = ls->tess_prim;

   if (prim == MESA_PRIM_POINTS)
      topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   else if (prim == MESA_PRIM_LINES)
      topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
   else if (prim != MESA_PRIM_COUNT)
      topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   return vk_primitive_topology_to_mtl_primitive_topology_class(topology);
}

static bool
kk_fs_link_key_needed(const struct kk_shader *fs,
                      const struct kk_fs_link_key *key)
{
   if (key->shader_blend || key->remove_depth || key->missing_inputs ||
       key->num_cull_distances || key->primitive_id_slot)
      return true;

   if (key->sample_mask)
      return true;

   u_foreach_bit(rt, fs->info.fs.color_outputs) {
      VkFormat format = key->rt_formats[rt];
      if (format != VK_FORMAT_UNDEFINED &&
          kk_format_base_type(vk_format_to_pipe_format(format)) !=
             fs->info.fs.output_base_type[rt])
         return true;
   }

   return false;
}

static void
kk_copy_blend(struct vk_color_blend_attachment_state *dst,
               const struct vk_color_blend_attachment_state *src)
{
   memset(dst, 0, sizeof(*dst));
   dst->blend_enable = src->blend_enable;
   dst->src_color_blend_factor = src->src_color_blend_factor;
   dst->dst_color_blend_factor = src->dst_color_blend_factor;
   dst->src_alpha_blend_factor = src->src_alpha_blend_factor;
   dst->dst_alpha_blend_factor = src->dst_alpha_blend_factor;
   dst->write_mask = src->write_mask;
   dst->color_blend_op = src->color_blend_op;
   dst->alpha_blend_op = src->alpha_blend_op;
   dst->src_premultiplied = src->src_premultiplied;
   dst->dst_premultiplied = src->dst_premultiplied;
   dst->blend_overlap = src->blend_overlap;
   dst->clamp_results = src->clamp_results;
}

static void
kk_fs_key_blend(const struct kk_shader *fs, uint32_t rt,
                const struct vk_color_blend_attachment_state *src,
                bool write_enabled,
                struct vk_color_blend_attachment_state *dst)
{
   memset(dst, 0, sizeof(*dst));
   if (!(fs->info.fs.color_outputs & BITFIELD_BIT(rt)))
      return;
   uint8_t write_mask = write_enabled ? src->write_mask : 0;
   if (write_mask && src->blend_enable)
      kk_copy_blend(dst, src);
   dst->write_mask = write_mask;
}

static void
kk_build_fs_link_key(const struct kk_link_state *ls, const struct kk_shader *vs,
                     const struct kk_shader *fs, bool force_shader_blend,
                     struct kk_fs_link_key *key)
{
   const struct vk_dynamic_graphics_state *dyn = ls->dyn;

   memset(key, 0, sizeof(*key));

   bool shader_blend = force_shader_blend || fs->info.fs.dual_source_blend ||
                       (dyn->cb.logic_op_enable && fs->info.fs.color_outputs);
   for (uint32_t i = 0u; i < ls->color_att_count; ++i) {
      uint8_t rt = dyn->cal.color_map[i];
      if (rt == MESA_VK_ATTACHMENT_UNUSED || rt >= MAX_DRAW_BUFFERS)
         continue;

      VkFormat vk_format = ls->color_formats[i];
      if (vk_format == VK_FORMAT_UNDEFINED)
         continue;

      enum pipe_format format = vk_format_to_pipe_format(vk_format);
      key->rt_formats[rt] = vk_format;
      kk_fs_key_blend(fs, rt, &dyn->cb.attachments[i],
                      dyn->cb.color_write_enables & BITFIELD_BIT(i),
                      &key->blend[rt]);

      if (!(fs->info.fs.color_outputs & BITFIELD_BIT(rt)))
         continue;

      if (!kk_get_va_format(format)->is_native)
         shader_blend = true;

      if (key->blend[rt].blend_enable &&
          key->blend[rt].color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
         shader_blend = true;
         key->advanced_blend_mask |= BITFIELD_BIT(rt);
      }
   }

   if (shader_blend) {
      key->shader_blend = true;
      key->logic_op_enable = dyn->cb.logic_op_enable;
      key->logic_op = dyn->cb.logic_op_enable ? dyn->cb.logic_op : 0;
   } else {
      for (uint32_t rt = 0u; rt < MAX_DRAW_BUFFERS; ++rt)
         memset(&key->blend[rt], 0, sizeof(key->blend[rt]));
   }

   key->remove_depth = fs->info.fs.writes_depth &&
                       ls->depth_format == VK_FORMAT_UNDEFINED;
   key->missing_inputs =
      fs->info.fs.inputs_read & ~vs->info.rast_outputs_written;
   key->num_cull_distances = vs->info.rast_num_cull_distances;
   if (fs->info.fs.reads_primitive_id)
      key->primitive_id_slot = vs->info.rast_primitive_id_slot;

   if (shader_blend) {
      key->alpha_to_coverage = dyn->ms.alpha_to_coverage_enable;
      key->alpha_to_one = dyn->ms.alpha_to_one_enable;
   }

   if (!fs->info.fs.ms_static) {
      uint32_t all_samples = BITFIELD_MASK(dyn->ms.rasterization_samples);
      key->sample_mask = dyn->ms.rasterization_samples &&
                         (dyn->ms.sample_mask & all_samples) != all_samples;
   }
}

static void
kk_build_pso_blend(const struct kk_link_state *ls,
                   const struct kk_fs_link_key *fs_key,
                   const struct kk_fs_variant *variant,
                   struct kk_pso_key *key)
{
   const struct vk_dynamic_graphics_state *dyn = ls->dyn;

   for (uint32_t i = 0u; i < ls->color_att_count; ++i) {
      uint8_t rt = dyn->cal.color_map[i];
      if (rt == MESA_VK_ATTACHMENT_UNUSED || rt >= MAX_DRAW_BUFFERS)
         continue;

      VkFormat vk_format = ls->color_formats[i];
      if (vk_format == VK_FORMAT_UNDEFINED)
         continue;

      key->rt_formats[rt] = vk_format_to_mtl_pixel_format(vk_format);
      if (variant->shader_blend)
         continue;

      const struct vk_color_blend_attachment_state *att =
         &dyn->cb.attachments[i];
      uint8_t write_mask = att->write_mask;
      if (!(dyn->cb.color_write_enables & BITFIELD_BIT(i)))
         write_mask = 0;

      struct kk_pso_blend *blend = &key->blend[rt];
      blend->write_mask =
         vk_color_write_mask_to_mtl_color_write_mask(write_mask);

      enum pipe_format format = vk_format_to_pipe_format(vk_format);
      if (!write_mask || !att->blend_enable ||
          util_format_is_pure_integer(format))
         continue;

      blend->enable = true;
      blend->src_rgb =
         vk_blend_factor_to_mtl_blend_factor(att->src_color_blend_factor);
      blend->dst_rgb =
         vk_blend_factor_to_mtl_blend_factor(att->dst_color_blend_factor);
      blend->op_rgb = vk_blend_op_to_mtl_blend_operation(att->color_blend_op);
      blend->src_alpha =
         vk_blend_factor_to_mtl_blend_factor(att->src_alpha_blend_factor);
      blend->dst_alpha =
         vk_blend_factor_to_mtl_blend_factor(att->dst_alpha_blend_factor);
      blend->op_alpha = vk_blend_op_to_mtl_blend_operation(att->alpha_blend_op);
   }
}

static struct kk_shader *
kk_link_rast_shader(const struct kk_link_state *ls)
{
   struct kk_shader *vs = ls->shaders[MESA_SHADER_VERTEX];
   if (!vs)
      return NULL;

   struct kk_shader *gs = ls->shaders[MESA_SHADER_GEOMETRY];
   if (gs && gs->info.shader_object)
      return gs;
   if (!vs->info.shader_object)
      return vs;

   struct kk_shader *tes = ls->shaders[MESA_SHADER_TESS_EVAL];
   if (gs)
      return gs;
   if (tes)
      return tes->info.so_hw ? tes : NULL;
   return vs->info.so_hw ? vs : NULL;
}

static void
kk_recipe_family(struct kk_shader *vs, struct kk_shader *fs, blake3_hash family)
{
   struct {
      uint32_t version;
      blake3_hash rast;
      blake3_hash fragment;
   } identity = { .version = 1 };
   kk_shader_variant_identity(vs, identity.rast);
   kk_shader_variant_identity(fs, identity.fragment);
   _mesa_blake3_compute(&identity, sizeof(identity), family);
}

static uint32_t
kk_normalize_sample_mask(uint32_t sample_mask, uint32_t samples)
{
   uint32_t all = BITFIELD_MASK(MAX2(samples, 1));
   sample_mask &= all;
   return sample_mask == all ? UINT16_MAX : sample_mask;
}

static void
kk_normalize_recipe(struct kk_variant_recipe *recipe)
{
   for (unsigned rt = 0; rt < MAX_DRAW_BUFFERS; ++rt) {
      struct vk_color_blend_attachment_state *blend = &recipe->input_blend[rt];
      uint8_t write_mask = blend->write_mask;
      if (!(recipe->write_enables & BITFIELD_BIT(rt)) || !write_mask) {
         memset(blend, 0, sizeof(*blend));
         recipe->write_enables &= ~BITFIELD_BIT(rt);
      } else if (!blend->blend_enable) {
         memset(blend, 0, sizeof(*blend));
         blend->write_mask = write_mask;
      }
   }
   if (!recipe->logic_enable)
      recipe->logic_op = 0;
   recipe->sample_mask =
      kk_normalize_sample_mask(recipe->sample_mask, recipe->pso.sample_count);
}

static void
kk_fill_recipe(const struct kk_link_state *ls, struct kk_shader *vs,
               struct kk_shader *fs, bool force_shader_blend,
               struct kk_variant_recipe *recipe)
{
   memset(recipe, 0, sizeof(*recipe));
   kk_recipe_family(vs, fs, recipe->family);
   kk_build_fs_link_key(ls, vs, fs, force_shader_blend, &recipe->fs);
   recipe->sample_mask = ls->dyn->ms.sample_mask;
   recipe->logic_enable = ls->dyn->cb.logic_op_enable;
   recipe->logic_op = ls->dyn->cb.logic_op;
   for (unsigned i = 0; i < ls->color_att_count; ++i) {
      unsigned rt = ls->dyn->cal.color_map[i];
      if (rt >= MAX_DRAW_BUFFERS)
         continue;
      kk_copy_blend(&recipe->input_blend[rt], &ls->dyn->cb.attachments[i]);
      if (ls->dyn->cb.color_write_enables & BITFIELD_BIT(i))
         recipe->write_enables |= BITFIELD_BIT(rt);
   }
   kk_pso_key_init(&recipe->pso);
   recipe->pso.topology_class = kk_link_topology_class(ls);
   struct kk_fs_variant blend = { .shader_blend = recipe->fs.shader_blend };
   kk_build_pso_blend(ls, &recipe->fs, &blend, &recipe->pso);
   recipe->pso.amplification = util_bitcount(ls->view_mask);
   recipe->pso.sample_count = ls->dyn->ms.rasterization_samples;
   recipe->pso.alpha_to_coverage =
      !recipe->fs.shader_blend && ls->dyn->ms.alpha_to_coverage_enable;
   recipe->pso.alpha_to_one =
      !recipe->fs.shader_blend && ls->dyn->ms.alpha_to_one_enable;
}

static void
kk_make_recipe(const struct kk_link_state *ls, struct kk_shader *vs,
               struct kk_shader *fs, bool force_shader_blend,
               struct kk_variant_recipe *recipe)
{
   kk_fill_recipe(ls, vs, fs, force_shader_blend, recipe);
   kk_normalize_recipe(recipe);
}

static VkFormat
kk_fs_key_format_class(VkFormat format)
{
   if (format == VK_FORMAT_UNDEFINED)
      return format;
   enum pipe_format pipe = vk_format_to_pipe_format(format);
   if (util_format_is_pure_sint(pipe))
      return VK_FORMAT_R32G32B32A32_SINT;
   if (util_format_is_pure_uint(pipe))
      return VK_FORMAT_R32G32B32A32_UINT;
   if (util_format_is_float(pipe) || util_format_is_unorm(pipe))
      return VK_FORMAT_R32G32B32A32_SFLOAT;
   return format;
}

static void
kk_normalize_fs_link_key(const struct kk_shader *fs,
                         struct kk_fs_link_key *key)
{
   if (key->shader_blend)
      return;
   for (unsigned rt = 0; rt < MAX_DRAW_BUFFERS; ++rt) {
      key->rt_formats[rt] = fs->info.fs.color_outputs & BITFIELD_BIT(rt) ?
         kk_fs_key_format_class(key->rt_formats[rt]) : VK_FORMAT_UNDEFINED;
   }
}

static struct kk_pso_result
kk_prepare_recipe(struct kk_device *dev, struct kk_shader *vs,
                   struct kk_shader *fs, const struct kk_variant_recipe *recipe,
                   struct kk_pso_key *native_key,
                   const struct kk_pso_request *request)
{
   const struct kk_fs_variant *variant = &fs->fast;
   if (kk_fs_link_key_needed(fs, &recipe->fs)) {
      struct kk_fs_link_key fs_key = recipe->fs;
      kk_normalize_fs_link_key(fs, &fs_key);
      VkResult status = kk_shader_get_fs_variant(dev, fs, &fs_key,
                                                 &variant, request);
      if (status != VK_SUCCESS)
         return (struct kk_pso_result){ .status = status };
   }
   bool points =
      recipe->pso.topology_class == MTL_PRIMITIVE_TOPOLOGY_CLASS_POINT;
   if (request && request->lookup_only && !kk_shader_rast_ready(vs, points))
      return (struct kk_pso_result){ .status = VK_NOT_READY };
   const struct kk_fs_variant *rast = kk_shader_get_rast(dev, vs, points);
   if (!rast)
      return (struct kk_pso_result){ .status = VK_ERROR_UNKNOWN };
   struct kk_pso_key key = recipe->pso;
   memcpy(key.rast, rast->hash, sizeof(key.rast));
   memcpy(key.fs, variant->hash, sizeof(key.fs));
   *native_key = key;
   struct kk_pso_request pso_request = {
      .lookup_only = request && request->lookup_only,
      .native_primitive_id = variant->native_primitive_id,
   };
   return kk_pso_cache_get(dev, &key, rast->fn, variant->fn, &pso_request);
}

enum kk_job_type {
   KK_JOB_PREWARM,
   KK_JOB_RENDER,
};

struct kk_prewarm_job {
   struct kk_device *dev;
   enum kk_job_type type;
   struct kk_shader *shaders[MESA_SHADER_COMPUTE];
   struct kk_shader *vs;
   struct kk_variant_recipe recipe;
   struct kk_variant_recipe raw;
   struct util_queue_fence fence;
   int64_t enqueue_ns;
   bool exact;
   bool load_history;
   BITSET_DECLARE(dynamic, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX);
};

static void kk_prewarm_execute(void *data, void *global_data,
                               int thread_index);
static void kk_prewarm_cleanup(void *data, void *global_data,
                               int thread_index);

static void
kk_job_set_shader(struct kk_prewarm_job *job, struct kk_shader *shader)
{
   if (!shader || shader == job->dev->empty_fs)
      return;
   job->shaders[shader->info.stage] = shader;
   vk_pipeline_cache_object_ref(&shader->vk.pipeline.cache_obj);
}

static void
kk_enqueue_job(struct kk_device *dev, struct util_queue *queue,
               struct kk_prewarm_job *job)
{
   mtx_lock(&dev->pso_cache.lock);
   dev->pso_cache.pending_jobs++;
   dev->pso_cache.peak_pending_jobs =
      MAX2(dev->pso_cache.peak_pending_jobs, dev->pso_cache.pending_jobs);
   mtx_unlock(&dev->pso_cache.lock);
   util_queue_fence_init(&job->fence);
   job->enqueue_ns = os_time_get_nano();
   util_queue_add_job(queue, job, &job->fence, kk_prewarm_execute,
                      kk_prewarm_cleanup, sizeof(*job));
}

static void
kk_render_compile(struct kk_device *dev, struct kk_shader *vs)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   struct kk_render_pending *pending = vs->pending;
   const struct kk_pso_request request = {
      .native_primitive_id = pending->native_primitive_id,
   };
   mtl_render_pipeline_state *pso =
      kk_pso_cache_acquire(dev, &pending->key, pending->vertex_function,
                           pending->fragment_function, &request);
   mtx_lock(&cache->lock);
   vs->pipeline.gfx.render = pso;
   pending->compiling = false;
   if (!pso && !cache->limited)
      cache->failed_jobs++;
   cnd_broadcast(&pending->ready);
   mtx_unlock(&cache->lock);
}

static bool
kk_render_claim(struct kk_pso_cache *cache, struct kk_render_pending *pending)
{
   mtx_lock(&cache->lock);
   bool claim = pending->compiling && !pending->started;
   pending->started |= claim;
   mtx_unlock(&cache->lock);
   return claim;
}

static void
kk_job_execute(struct kk_prewarm_job *job)
{
   enum kk_compile_origin origin = kk_compile_origin;
   switch (job->type) {
   case KK_JOB_RENDER:
      kk_compile_origin = KK_COMPILE_ORIGIN_CREATE;
      if (kk_render_claim(&job->dev->pso_cache, job->vs->pending))
         kk_render_compile(job->dev, job->vs);
      break;
   default:
      UNREACHABLE("prewarm jobs are executed by kk_prewarm_execute");
   }
   kk_compile_origin = origin;
}

void
kk_pso_schedule_render(struct kk_device *dev, struct kk_shader *vs)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   struct kk_render_pending *pending = vs->pending;
   if (!pending)
      return;
   mtx_lock(&cache->lock);
   bool schedule = !pending->queued;
   pending->queued = true;
   mtx_unlock(&cache->lock);
   if (!schedule)
      return;
   struct kk_prewarm_job *job =
      util_queue_is_initialized(&cache->prewarm_queue) ?
         calloc(1, sizeof(*job)) : NULL;
   if (!job) {
      if (kk_render_claim(cache, pending))
         kk_render_compile(dev, vs);
      return;
   }
   job->dev = dev;
   job->type = KK_JOB_RENDER;
   job->vs = vs;
   kk_job_set_shader(job, vs);
   kk_enqueue_job(dev, &cache->prewarm_queue, job);
}

bool
kk_pso_render_pending(struct kk_device *dev, struct kk_shader *vs)
{
   if (!vs->pending)
      return false;
   mtx_lock(&dev->pso_cache.lock);
   bool pending = vs->pending->compiling;
   mtx_unlock(&dev->pso_cache.lock);
   return pending;
}

mtl_render_pipeline_state *
kk_cmd_acquire_render(struct kk_cmd_buffer *cmd, struct kk_shader *vs)
{
   struct kk_render_pending *pending = vs->pending;
   if (!pending)
      return vs->pipeline.gfx.render;

   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_pso_cache *cache = &dev->pso_cache;
   kk_pso_schedule_render(dev, vs);
   if (kk_render_claim(cache, pending)) {
      kk_log_cache_pso(dev, "render_takeover", &pending->key,
                       sizeof(pending->key));
      kk_render_compile(dev, vs);
   }
   mtx_lock(&cache->lock);
   while (pending->compiling)
      cnd_wait(&pending->ready, &cache->lock);
   mtl_render_pipeline_state *pso = vs->pipeline.gfx.render;
   mtx_unlock(&cache->lock);
   return pso;
}

static void
kk_draw_demand_begin(struct kk_pso_cache *cache)
{
   mtx_lock(&cache->lock);
   cache->draw_demand++;
   mtx_unlock(&cache->lock);
}

static void
kk_draw_demand_end(struct kk_pso_cache *cache)
{
   mtx_lock(&cache->lock);
   if (!--cache->draw_demand)
      cnd_broadcast(&cache->draw_idle);
   mtx_unlock(&cache->lock);
}

static void
kk_yield_to_draws(struct kk_pso_cache *cache)
{
   mtx_lock(&cache->lock);
   while (cache->draw_demand)
      cnd_wait(&cache->draw_idle, &cache->lock);
   mtx_unlock(&cache->lock);
}

static void
kk_recipe_restore_trace(bool trace_enabled, const blake3_hash parent,
                        bool parent_valid, const blake3_hash family,
                        bool family_valid)
{
   if (!trace_enabled)
      return;
   memcpy(kk_variant_parent_id, parent, sizeof(blake3_hash));
   kk_variant_parent_valid = parent_valid;
   memcpy(kk_variant_family_id, family, sizeof(blake3_hash));
   kk_variant_family_valid = family_valid;
}

static struct kk_pso_result
kk_recipe_get(struct kk_device *dev, struct kk_shader *vs,
               struct kk_shader *fs, const struct kk_variant_recipe *recipe,
               const struct kk_pso_request *request)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   bool lookup_only = request && request->lookup_only;
   bool trace_enabled = dev->variant_log != NULL;
   blake3_hash previous, previous_family;
   bool previous_valid = false, previous_family_valid = false;
   if (trace_enabled) {
      memcpy(previous, kk_variant_parent_id, sizeof(previous));
      memcpy(previous_family, kk_variant_family_id, sizeof(previous_family));
      previous_valid = kk_variant_parent_valid;
      previous_family_valid = kk_variant_family_valid;
      _mesa_blake3_compute(recipe, sizeof(*recipe), kk_variant_parent_id);
      kk_variant_parent_valid = true;
      memcpy(kk_variant_family_id, recipe->family,
             sizeof(kk_variant_family_id));
      kk_variant_family_valid = true;
   }
   struct kk_recipe_entry *entry =
      kk_recipe_lookup(cache, recipe, !lookup_only);
   if (!entry && lookup_only) {
      kk_recipe_restore_trace(trace_enabled, previous, previous_valid,
                              previous_family, previous_family_valid);
      return (struct kk_pso_result){ .status = VK_NOT_READY };
   }
   if (!entry && kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM) {
      mtx_lock(&cache->lock);
      cache->limited = true;
      mtx_unlock(&cache->lock);
      kk_variant_trace(dev, KK_VARIANT_FAILURE, recipe, sizeof(*recipe), 0, 0);
      kk_recipe_restore_trace(trace_enabled, previous, previous_valid,
                              previous_family, previous_family_valid);
      return (struct kk_pso_result){ .status = VK_ERROR_UNKNOWN };
   }
   if (entry && !lookup_only && kk_compile_origin == KK_COMPILE_ORIGIN_DRAW &&
       !p_atomic_read(&entry->demanded) &&
       !p_atomic_xchg(&entry->demanded, true))
      kk_variant_trace(dev, KK_VARIANT_FIRST_DEMAND, recipe,
                       sizeof(*recipe), 0, 0);
   struct kk_pso_result result = { .status = VK_ERROR_UNKNOWN };
   mtl_render_pipeline_state *pso = entry ? p_atomic_read(&entry->pso) : NULL;
   bool demand = !pso && !lookup_only &&
                 kk_compile_origin == KK_COMPILE_ORIGIN_DRAW;
   if (demand)
      kk_draw_demand_begin(cache);
   bool compile = false;
   if (entry && !pso) {
      mtx_lock(&cache->lock);
      bool in_flight = entry->state == KK_ENTRY_COMPILING &&
                       kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM;
      bool waited =
         !lookup_only && !in_flight && entry->state == KK_ENTRY_COMPILING;
      if (in_flight)
         kk_log_cache_pso(dev, "prewarm_in_flight", recipe, sizeof(*recipe));
      if (waited) {
         int64_t start = os_time_get_nano();
         struct mtl_compile_phase phase =
            kk_pso_phase_begin(dev, "recipe_wait", recipe, sizeof(*recipe));
         while (entry->state == KK_ENTRY_COMPILING)
            cnd_wait(&entry->ready, &cache->lock);
         mtl_compile_phase_end(phase, true);
         kk_log_compile(dev, "variant_wait", start);
         kk_variant_trace(dev, KK_VARIANT_PSO_WAIT, recipe, sizeof(*recipe),
                          start, os_time_get_nano());
      }
      compile = !lookup_only &&
                (entry->state == KK_ENTRY_IDLE ||
                 entry->state == KK_ENTRY_QUEUED ||
                 (!waited &&
                  kk_entry_retryable(entry->state, entry->failures)));
      if (compile) {
         if (entry->state == KK_ENTRY_QUEUED &&
             kk_compile_origin == KK_COMPILE_ORIGIN_DRAW)
            kk_log_cache_pso(dev, "recipe_takeover", recipe, sizeof(*recipe));
         else if (entry->state == KK_ENTRY_FAILED)
            kk_log_cache_pso(dev, "recipe_retry", recipe, sizeof(*recipe));
         entry->state = KK_ENTRY_COMPILING;
         if (kk_compile_origin == KK_COMPILE_ORIGIN_PREWARM)
            p_atomic_set(&entry->prewarmed, true);
      } else {
         pso = entry->pso;
         result.status = kk_entry_status(entry->state, entry->failures,
                                         entry->result, lookup_only);
      }
      mtx_unlock(&cache->lock);
   }
   if (!entry || compile) {
      struct kk_pso_key native_key;
      struct mtl_compile_phase phase =
         kk_pso_phase_begin(dev, "recipe_prepare", recipe, sizeof(*recipe));
      result = kk_prepare_recipe(dev, vs, fs, recipe, &native_key, NULL);
      mtl_compile_phase_end(phase, result.status == VK_SUCCESS);
      pso = result.pso;
      if (!pso && result.status == VK_SUCCESS)
         result.status = VK_ERROR_UNKNOWN;
      if (entry) {
         mtx_lock(&cache->lock);
         if (pso) {
            entry->native_key = native_key;
            p_atomic_set(&entry->pso, pso);
            entry->state = KK_ENTRY_READY;
         } else {
            entry->failures++;
            entry->result = result.status;
            entry->state = KK_ENTRY_FAILED;
         }
         cnd_broadcast(&entry->ready);
         mtx_unlock(&cache->lock);
      }
      kk_variant_trace(dev, pso ? KK_VARIANT_COMPLETE : KK_VARIANT_FAILURE,
                       recipe, sizeof(*recipe), 0, 0);
   } else if (pso) {
      kk_variant_trace(dev, KK_VARIANT_READY_HIT, recipe,
                       sizeof(*recipe), 0, 0);
   }
   if (demand)
      kk_draw_demand_end(cache);
   result.pso = pso;
   if (pso)
      result.status = VK_SUCCESS;
   if (pso && entry && !lookup_only &&
       kk_compile_origin == KK_COMPILE_ORIGIN_DRAW &&
       !p_atomic_read(&entry->drawn) && !p_atomic_xchg(&entry->drawn, true)) {
      uint32_t hash = kk_pso_key_hash(&entry->native_key);
      u_rwlock_rdlock(&cache->index_lock);
      struct hash_entry *native = _mesa_hash_table_search_pre_hashed(
         cache->table, hash, &entry->native_key);
      struct kk_pso_entry *native_entry = native ? native->data : NULL;
      u_rwlock_rdunlock(&cache->index_lock);
      if (native_entry)
         p_atomic_set(&native_entry->drawn, true);
      if (p_atomic_read(&entry->prewarmed))
         kk_log_cache_pso(dev, "prewarm_match", &entry->native_key,
                          sizeof(entry->native_key));
      uint8_t encoded[KK_RECIPE_ENCODED_SIZE];
      kk_recipe_encode(recipe, encoded);
      kk_variant_history_record(cache->history, recipe->family, encoded);
   }
   kk_recipe_restore_trace(trace_enabled, previous, previous_valid,
                           previous_family, previous_family_valid);
   return result;
}

static struct kk_pso_result
kk_link(struct kk_device *dev, const struct kk_link_state *ls,
        struct kk_shader *vs, struct kk_shader *fs, bool force_shader_blend)
{
   struct kk_variant_recipe recipe;
   kk_make_recipe(ls, vs, fs, force_shader_blend, &recipe);
   return kk_recipe_get(dev, vs, fs, &recipe, NULL);
}

static struct kk_shader *
kk_link_fs(struct kk_device *dev, const struct kk_link_state *ls)
{
   struct kk_shader *fs = ls->shaders[MESA_SHADER_FRAGMENT];
   if (fs == NULL || !fs->info.partial)
      fs = kk_device_get_empty_fs(dev);
   return fs;
}

struct kk_pso_result
kk_link_pipeline(struct kk_device *dev, const struct kk_link_state *ls)
{
   struct kk_shader *vs = kk_link_rast_shader(ls);
   if (vs == NULL)
      return (struct kk_pso_result){ .status = VK_ERROR_UNKNOWN };

   struct kk_shader *fs = kk_link_fs(dev, ls);
   if (fs == NULL)
      return (struct kk_pso_result){ .status = VK_ERROR_UNKNOWN };

   struct kk_pso_result result = kk_link(dev, ls, vs, fs, false);
   if (!result.pso)
      result = kk_link(dev, ls, vs, fs, true);
   return result;
}

mtl_render_pipeline_state *
kk_cmd_link_pipeline(struct kk_cmd_buffer *cmd)
{
   const struct kk_rendering_state *render = &cmd->state.gfx.render;
   struct kk_link_state ls = {
      .dyn = &cmd->vk.dynamic_graphics_state,
      .tess_prim = cmd->state.gfx.tess.prim,
      .view_mask = render->view_mask,
      .color_att_count = render->color_att_count,
      .depth_format = render->depth_att.vk_format,
   };
   memcpy(ls.shaders, cmd->state.shaders, sizeof(ls.shaders));
   for (uint32_t i = 0; i < render->color_att_count; ++i)
      ls.color_formats[i] = render->color_att[i].vk_format;

   return kk_link_pipeline(kk_cmd_buffer_device(cmd), &ls).pso;
}

static bool
kk_recipe_bool_valid(const bool *value)
{
   return *(const uint8_t *)value <= 1;
}

static bool
kk_recipe_valid(struct kk_device *dev,
                struct kk_shader *vs, struct kk_shader *fs,
                 const struct kk_variant_recipe *recipe)
{
   blake3_hash family;
   kk_recipe_family(vs, fs, family);
   if (memcmp(family, recipe->family, sizeof(family)) ||
       recipe->pso.topology_class < MTL_PRIMITIVE_TOPOLOGY_CLASS_POINT ||
       recipe->pso.topology_class > MTL_PRIMITIVE_TOPOLOGY_CLASS_TRIANGLE ||
       !util_is_power_of_two_or_zero(recipe->pso.sample_count) ||
       !recipe->pso.sample_count || recipe->pso.sample_count > 8 ||
       !(dev->vk.physical->properties.framebufferColorSampleCounts &
         recipe->pso.sample_count) ||
       recipe->pso.amplification > 32 || recipe->pso.alpha_to_coverage > 1 ||
       recipe->pso.alpha_to_one > 1 || recipe->fs.logic_op > VK_LOGIC_OP_SET ||
       recipe->fs.advanced_blend_mask & ~BITFIELD_MASK(MAX_DRAW_BUFFERS) ||
       recipe->logic_enable > 1 || recipe->logic_op > VK_LOGIC_OP_SET ||
       recipe->write_enables & ~BITFIELD_MASK(MAX_DRAW_BUFFERS) ||
       recipe->fs.missing_inputs !=
          (fs->info.fs.inputs_read & ~vs->info.rast_outputs_written) ||
       recipe->fs.num_cull_distances != vs->info.rast_num_cull_distances ||
       recipe->fs.primitive_id_slot !=
          (fs->info.fs.reads_primitive_id
              ? vs->info.rast_primitive_id_slot : 0))
      return false;
   const bool *flags[] = {
      &recipe->fs.logic_op_enable, &recipe->fs.shader_blend,
      &recipe->fs.remove_depth, &recipe->fs.alpha_to_coverage,
      &recipe->fs.alpha_to_one, &recipe->fs.sample_mask,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(flags); ++i)
      if (!kk_recipe_bool_valid(flags[i]))
         return false;
   if ((!recipe->fs.shader_blend &&
        (recipe->fs.logic_op_enable || recipe->fs.alpha_to_coverage ||
         recipe->fs.alpha_to_one)) ||
       (!recipe->fs.logic_op_enable && recipe->fs.logic_op) ||
       (recipe->fs.remove_depth && !fs->info.fs.writes_depth))
      return false;
   uint32_t advanced = 0;
   for (unsigned rt = 0; rt < MAX_DRAW_BUFFERS; ++rt) {
      const struct vk_color_blend_attachment_state *b = &recipe->fs.blend[rt];
      const struct vk_color_blend_attachment_state none = {0};
      if (!recipe->fs.shader_blend && memcmp(b, &none, sizeof(none)))
         return false;
      if ((fs->info.fs.color_outputs & BITFIELD_BIT(rt)) && b->blend_enable &&
          b->color_blend_op >= VK_BLEND_OP_ZERO_EXT)
         advanced |= BITFIELD_BIT(rt);
   }
   if (recipe->fs.advanced_blend_mask != advanced)
      return false;
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i) {
      VkFormat format = recipe->fs.rt_formats[i];
      enum pipe_format pipe = vk_format_to_pipe_format(format);
      if (format != VK_FORMAT_UNDEFINED &&
          (pipe == PIPE_FORMAT_NONE || util_format_is_depth_or_stencil(pipe)))
         return false;
      const struct kk_va_format *va =
         pipe == PIPE_FORMAT_NONE ? NULL : kk_get_va_format(pipe);
      if (format != VK_FORMAT_UNDEFINED &&
          (!va || !va->color || (recipe->pso.sample_count > 1 && !va->msaa)))
         return false;
      if (recipe->pso.rt_formats[i] !=
          (va ? va->mtl_pixel_format : MTL_PIXEL_FORMAT_INVALID))
         return false;
      const struct kk_pso_blend *b = &recipe->pso.blend[i];
      if (b->enable > 1 ||
          b->src_rgb > MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA ||
          b->dst_rgb > MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA ||
          b->src_alpha > MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA ||
          b->dst_alpha > MTL_BLEND_FACTOR_ONE_MINUS_SOURCE1_ALPHA ||
          b->op_rgb > MTL_BLEND_OPERATION_MAX ||
          b->op_alpha > MTL_BLEND_OPERATION_MAX ||
          b->write_mask > 15)
         return false;
      for (unsigned group = 0; group < 2; ++group) {
         const struct vk_color_blend_attachment_state *a = group ?
            &recipe->input_blend[i] : &recipe->fs.blend[i];
         if (!kk_recipe_bool_valid(&a->blend_enable) ||
             !kk_recipe_bool_valid(&a->src_premultiplied) ||
             !kk_recipe_bool_valid(&a->dst_premultiplied) ||
             !kk_recipe_bool_valid(&a->clamp_results) || a->write_mask > 15 ||
             a->src_color_blend_factor > VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA ||
             a->dst_color_blend_factor > VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA ||
             a->src_alpha_blend_factor > VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA ||
             a->dst_alpha_blend_factor > VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA ||
             a->blend_overlap > VK_BLEND_OVERLAP_CONJOINT_EXT)
            return false;
         VkBlendOp ops[] = {a->color_blend_op, a->alpha_blend_op};
         for (unsigned j = 0; j < ARRAY_SIZE(ops); ++j)
            if ((unsigned)ops[j] > VK_BLEND_OP_MAX &&
                ((unsigned)ops[j] < VK_BLEND_OP_ZERO_EXT ||
                 (unsigned)ops[j] > VK_BLEND_OP_BLUE_EXT))
               return false;
         struct vk_color_blend_attachment_state normalized;
         kk_copy_blend(&normalized, a);
         if (memcmp(a, &normalized, sizeof(normalized)))
            return false;
      }
   }
   struct kk_variant_recipe normalized;
   memcpy(&normalized, recipe, sizeof(normalized));
   kk_normalize_recipe(&normalized);
   if (memcmp(&normalized, recipe, sizeof(normalized)))
      return false;
   uint8_t zero[sizeof(blake3_hash)] = {0};
   if (memcmp(recipe->pso.rast, zero, sizeof(zero)) ||
       memcmp(recipe->pso.fs, zero, sizeof(zero)) ||
       memcmp(recipe->fs.pad, zero, sizeof(recipe->fs.pad)))
      return false;
   return true;
}

static bool
kk_prewarm_recipe(struct kk_device *dev, struct kk_shader *vs,
                  struct kk_shader *fs, const struct kk_variant_recipe *recipe)
{
   struct kk_pso_result result = kk_recipe_get(dev, vs, fs, recipe, NULL);
   if (result.pso || result.status == VK_NOT_READY)
      return true;
   if (recipe->fs.shader_blend)
      return false;
   struct kk_variant_recipe fallback = *recipe;
   fallback.fs.shader_blend = true;
   fallback.fs.logic_op_enable = recipe->logic_enable;
   fallback.fs.logic_op = recipe->logic_enable ? recipe->logic_op : 0;
   fallback.fs.alpha_to_coverage = recipe->pso.alpha_to_coverage;
   fallback.fs.alpha_to_one = recipe->pso.alpha_to_one;
   fallback.pso.alpha_to_coverage = 0;
   fallback.pso.alpha_to_one = 0;
   struct kk_pso_key defaults;
   kk_pso_key_init(&defaults);
   memcpy(fallback.pso.blend, defaults.blend, sizeof(fallback.pso.blend));
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i) {
      if (recipe->fs.rt_formats[i] == VK_FORMAT_UNDEFINED)
         continue;
      kk_fs_key_blend(fs, i, &recipe->input_blend[i],
                      recipe->write_enables & BITFIELD_BIT(i),
                      &fallback.fs.blend[i]);
   }
   result = kk_recipe_get(dev, vs, fs, &fallback, NULL);
   return result.pso || result.status == VK_NOT_READY;
}

static bool
kk_history_matches(const struct kk_prewarm_job *job,
                   const struct kk_variant_recipe *r)
{
   const struct kk_variant_recipe *d = &job->raw;
   if (memcmp(r->fs.rt_formats, d->fs.rt_formats, sizeof(r->fs.rt_formats)) ||
       r->fs.remove_depth != d->fs.remove_depth ||
       r->pso.amplification != d->pso.amplification ||
       r->pso.topology_class != d->pso.topology_class)
      return false;
   if ((!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
        r->pso.sample_count != d->pso.sample_count) ||
       (!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_MS_SAMPLE_MASK) &&
        r->sample_mask !=
           kk_normalize_sample_mask(d->sample_mask, r->pso.sample_count)) ||
       (!BITSET_TEST(job->dynamic,
                     MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
        (r->pso.alpha_to_coverage || r->fs.alpha_to_coverage) !=
        (d->pso.alpha_to_coverage || d->fs.alpha_to_coverage)) ||
       (!BITSET_TEST(job->dynamic,
                     MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE) &&
        (r->pso.alpha_to_one || r->fs.alpha_to_one) !=
           (d->pso.alpha_to_one || d->fs.alpha_to_one)) ||
       (!BITSET_TEST(job->dynamic,
                     MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE) &&
        r->logic_enable != d->logic_enable) ||
       (!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_LOGIC_OP) &&
        r->logic_enable && r->logic_op != d->logic_op))
      return false;
   bool static_enables =
      !BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   bool static_masks =
      !BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   for (unsigned i = 0; i < MAX_DRAW_BUFFERS; ++i) {
      const struct vk_color_blend_attachment_state *a = &r->input_blend[i];
      const struct vk_color_blend_attachment_state *b = &d->input_blend[i];
      bool d_enabled = d->write_enables & BITFIELD_BIT(i);
      if (!(r->write_enables & BITFIELD_BIT(i))) {
         if (static_enables && static_masks && d_enabled && b->write_mask)
            return false;
         continue;
      }
      if ((static_enables && !d_enabled) ||
          (static_masks && a->write_mask != b->write_mask) ||
          (!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) &&
           a->blend_enable != b->blend_enable))
         return false;
      if (!a->blend_enable)
         continue;
      if (!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS) &&
          (a->src_color_blend_factor != b->src_color_blend_factor ||
           a->dst_color_blend_factor != b->dst_color_blend_factor ||
           a->src_alpha_blend_factor != b->src_alpha_blend_factor ||
           a->dst_alpha_blend_factor != b->dst_alpha_blend_factor ||
           a->color_blend_op != b->color_blend_op ||
           a->alpha_blend_op != b->alpha_blend_op))
         return false;
      if (!BITSET_TEST(job->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED) &&
          (a->src_premultiplied != b->src_premultiplied ||
           a->dst_premultiplied != b->dst_premultiplied ||
           a->blend_overlap != b->blend_overlap ||
           a->clamp_results != b->clamp_results))
         return false;
   }
   return true;
}

static void
kk_prewarm_execute(void *data, void *global_data, int thread_index)
{
   struct kk_prewarm_job *job = data;
   if (job->type != KK_JOB_PREWARM) {
      kk_variant_trace(job->dev, KK_VARIANT_QUEUE_WAIT, &job->vs->pending->key,
                       sizeof(job->vs->pending->key), job->enqueue_ns,
                       os_time_get_nano());
      kk_job_execute(job);
      return;
   }
   enum kk_compile_origin origin = kk_compile_origin;
   kk_compile_origin = KK_COMPILE_ORIGIN_PREWARM;
   bool trace_enabled = job->dev->variant_log != NULL;
   blake3_hash previous_parent, previous_family;
   bool previous_parent_valid = false, previous_family_valid = false;
   if (trace_enabled) {
      memcpy(previous_parent, kk_variant_parent_id, sizeof(previous_parent));
      memcpy(previous_family, kk_variant_family_id, sizeof(previous_family));
      previous_parent_valid = kk_variant_parent_valid;
      previous_family_valid = kk_variant_family_valid;
      _mesa_blake3_compute(&job->recipe, sizeof(job->recipe),
                           kk_variant_parent_id);
      kk_variant_parent_valid = true;
      memcpy(kk_variant_family_id, job->recipe.family,
             sizeof(kk_variant_family_id));
      kk_variant_family_valid = true;
   }
   int64_t start = os_time_get_nano();
   kk_variant_trace(job->dev, KK_VARIANT_QUEUE_WAIT, &job->recipe,
                    sizeof(job->recipe), job->enqueue_ns, start);
   kk_variant_trace(job->dev, KK_VARIANT_WORKER_START, &job->recipe,
                    sizeof(job->recipe), start, start);
   struct kk_shader *vs = job->shaders[MESA_SHADER_VERTEX];
   struct kk_shader *fs = job->shaders[MESA_SHADER_FRAGMENT];
   bool failed = false;
   if (job->load_history) {
      struct kk_variant_recipe *recipes = calloc(32, sizeof(*recipes));
      uint8_t *encoded = calloc(32, KK_RECIPE_ENCODED_SIZE);
      if (recipes && encoded) {
         int64_t history_start = os_time_get_nano();
         unsigned count = kk_variant_history_load(
            job->dev->pso_cache.history, job->recipe.family, encoded, 32);
         kk_variant_trace(job->dev, KK_VARIANT_HISTORY_LOAD, job->recipe.family,
                          sizeof(job->recipe.family), history_start,
                          os_time_get_nano());
         for (unsigned i = 0; i < count; ++i) {
            if (!kk_recipe_decode(encoded + i * KK_RECIPE_ENCODED_SIZE,
                                  &recipes[i]) ||
                !kk_recipe_valid(job->dev, vs, fs, &recipes[i])) {
               kk_log_cache_pso(job->dev, "history_rejected",
                                &recipes[i], sizeof(recipes[i]));
               continue;
            }
            if (!kk_history_matches(job, &recipes[i]))
               continue;
            kk_yield_to_draws(&job->dev->pso_cache);
            kk_log_cache_pso(job->dev, "history_replay",
                             &recipes[i], sizeof(recipes[i]));
            failed |= !kk_prewarm_recipe(job->dev, vs, fs, &recipes[i]);
         }
      } else {
         failed = true;
      }
      free(recipes);
      free(encoded);
   }
   failed |= !kk_prewarm_recipe(job->dev, vs, fs, &job->recipe);
   if (failed) {
      mtx_lock(&job->dev->pso_cache.lock);
      if (!job->dev->pso_cache.limited)
         job->dev->pso_cache.failed_jobs++;
      mtx_unlock(&job->dev->pso_cache.lock);
   }
   if (trace_enabled) {
      memcpy(kk_variant_parent_id, previous_parent, sizeof(previous_parent));
      kk_variant_parent_valid = previous_parent_valid;
      memcpy(kk_variant_family_id, previous_family, sizeof(previous_family));
      kk_variant_family_valid = previous_family_valid;
   }
   kk_compile_origin = origin;
}

static void
kk_prewarm_cleanup(void *data, void *global_data, int thread_index)
{
   struct kk_prewarm_job *job = data;
   for (unsigned i = 0; i < MESA_SHADER_COMPUTE; ++i)
      if (job->shaders[i])
         vk_pipeline_cache_object_unref(&job->dev->vk,
                                        &job->shaders[i]->vk.pipeline.
                                            cache_obj);
   mtx_lock(&job->dev->pso_cache.lock);
   job->dev->pso_cache.pending_jobs--;
   job->dev->pso_cache.exact_jobs -= job->exact;
   cnd_broadcast(&job->dev->pso_cache.jobs_done);
   mtx_unlock(&job->dev->pso_cache.lock);
   util_queue_fence_destroy(&job->fence);
   free(job);
}

static void
kk_schedule_recipe(struct kk_device *dev,
                   const struct kk_link_state *ls, bool exact,
                    const BITSET_WORD *dynamic)
{
   struct kk_pso_cache *cache = &dev->pso_cache;
   struct util_queue *queue =
      exact ? &cache->exact_queue : &cache->prewarm_queue;
   struct kk_shader *vs = kk_link_rast_shader(ls);
   struct kk_shader *fs = ls->shaders[MESA_SHADER_FRAGMENT];
   if (!util_queue_is_initialized(queue) || !vs || !vs->info.partial ||
       vs->info.shader_object || !fs || !fs->info.partial ||
       fs->info.shader_object)
      return;
   struct kk_variant_recipe raw, recipe;
   kk_fill_recipe(ls, vs, fs, false, &raw);
   memcpy(&recipe, &raw, sizeof(recipe));
   kk_normalize_recipe(&recipe);
   BITSET_DECLARE(dynamic_set, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX) = {0};
   if (dynamic)
      memcpy(dynamic_set, dynamic, sizeof(dynamic_set));
   blake3_hash load_key;
   if (!exact) {
      struct mesa_blake3 hash;
      _mesa_blake3_init(&hash);
      _mesa_blake3_update(&hash, &recipe, sizeof(recipe));
      _mesa_blake3_update(&hash, dynamic_set, sizeof(dynamic_set));
      _mesa_blake3_final(&hash, load_key);
   }
   uint32_t recipe_hash = kk_recipe_hash(&recipe);
   uint32_t family_hash = exact ? 0 : kk_family_hash(load_key);
   u_rwlock_rdlock(&cache->index_lock);
   struct hash_entry *he = cache->recipes ?
      _mesa_hash_table_search_pre_hashed(cache->recipes, recipe_hash,
                                         &recipe) : NULL;
   struct kk_recipe_entry *known = he ? he->data : NULL;
   bool needs_history = !exact && cache->history && cache->families &&
      !_mesa_hash_table_search_pre_hashed(cache->families, family_hash,
                                          load_key);
   u_rwlock_rdunlock(&cache->index_lock);
   if (known && !needs_history) {
      if (p_atomic_read(&known->pso))
         return;
      mtx_lock(&cache->lock);
      bool idle = known->state == KK_ENTRY_IDLE;
      mtx_unlock(&cache->lock);
      if (!idle)
         return;
   }

   u_rwlock_wrlock(&cache->index_lock);
   mtx_lock(&cache->lock);
   he = cache->recipes ?
      _mesa_hash_table_search_pre_hashed(cache->recipes, recipe_hash,
                                         &recipe) : NULL;
   needs_history = !exact && cache->history && cache->families &&
      !_mesa_hash_table_search_pre_hashed(cache->families, family_hash,
                                          load_key);
   known = he ? he->data : NULL;
   bool idle = known && known->state == KK_ENTRY_IDLE;
   if (he && !needs_history && !idle) {
      mtx_unlock(&cache->lock);
      u_rwlock_wrunlock(&cache->index_lock);
      return;
   }
   if (cache->pending_jobs >= 64 || (exact && cache->exact_jobs >= 16) ||
       (needs_history && cache->family_count >= cache->recipe_limit)) {
      cache->limited = true;
      mtx_unlock(&cache->lock);
      u_rwlock_wrunlock(&cache->index_lock);
      return;
   }
   struct kk_prewarm_job *job = calloc(1, sizeof(*job));
   struct kk_recipe_entry *entry = !job ? NULL :
      he ? he->data : kk_recipe_insert_locked(cache, recipe_hash, &recipe);
   if (!entry) {
      cache->limited = true;
      mtx_unlock(&cache->lock);
      u_rwlock_wrunlock(&cache->index_lock);
      free(job);
      return;
   }
   if (!he || idle) {
      entry->state = KK_ENTRY_QUEUED;
      p_atomic_set(&entry->prewarmed, true);
   }
   if (needs_history) {
      uint8_t *family = ralloc_size(cache->families, sizeof(blake3_hash));
      if (family) {
         memcpy(family, load_key, sizeof(blake3_hash));
         if (_mesa_hash_table_insert_pre_hashed(cache->families, family_hash,
                                                family, family)) {
            job->load_history = true;
            cache->family_count++;
         } else {
            ralloc_free(family);
            cache->limited = true;
         }
      } else {
         cache->limited = true;
      }
   }
   cache->pending_jobs++;
   cache->exact_jobs += exact;
   cache->peak_pending_jobs =
      MAX2(cache->peak_pending_jobs, cache->pending_jobs);
   cache->peak_exact_jobs = MAX2(cache->peak_exact_jobs, cache->exact_jobs);
   mtx_unlock(&cache->lock);
   u_rwlock_wrunlock(&cache->index_lock);
   job->dev = dev;
   job->exact = exact;
   memcpy(job->dynamic, dynamic_set, sizeof(job->dynamic));
   job->raw = raw;
   job->recipe = recipe;
   job->shaders[MESA_SHADER_VERTEX] = vs;
   job->shaders[MESA_SHADER_FRAGMENT] = fs;
   for (unsigned i = 0; i < MESA_SHADER_COMPUTE; ++i)
      if (job->shaders[i])
         vk_pipeline_cache_object_ref(&job->shaders[i]->vk.pipeline.cache_obj);
   util_queue_fence_init(&job->fence);
   int64_t time = os_time_get_nano();
   job->enqueue_ns = time;
   kk_variant_trace(dev, KK_VARIANT_ENQUEUE, &job->recipe,
                    sizeof(job->recipe), time, time);
   util_queue_add_job(queue, job, &job->fence,
                      kk_prewarm_execute, kk_prewarm_cleanup, sizeof(*job));
}

static void
kk_prewarm_dynamic_defaults(struct vk_dynamic_graphics_state *dyn,
                            const struct vk_graphics_pipeline_state *state)
{
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES))
      dyn->ms.rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_SAMPLE_MASK))
      dyn->ms.sample_mask = UINT16_MAX;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE))
      dyn->ms.alpha_to_coverage_enable = false;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE))
      dyn->ms.alpha_to_one_enable = false;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE))
      dyn->cb.logic_op_enable = false;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_LOGIC_OP))
      dyn->cb.logic_op = VK_LOGIC_OP_COPY;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT))
      dyn->cb.attachment_count = state->rp->color_attachment_count;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES))
      dyn->cb.color_write_enables =
         (uint8_t)BITFIELD_MASK(MESA_VK_MAX_COLOR_ATTACHMENTS);

   for (uint32_t i = 0; i < MESA_VK_MAX_COLOR_ATTACHMENTS; ++i) {
      struct vk_color_blend_attachment_state *att = &dyn->cb.attachments[i];
      if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES))
         att->blend_enable = false;
      if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS)) {
         att->src_color_blend_factor = VK_BLEND_FACTOR_ONE;
         att->dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
         att->src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
         att->dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
         att->color_blend_op = VK_BLEND_OP_ADD;
         att->alpha_blend_op = VK_BLEND_OP_ADD;
      }
      if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS)) {
         att->write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      }
      if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED)) {
         att->src_premultiplied = true;
         att->dst_premultiplied = true;
         att->blend_overlap = VK_BLEND_OVERLAP_UNCORRELATED_EXT;
         att->clamp_results = false;
      }
   }
}

void
kk_prewarm_pipeline(struct vk_device *vk_dev,
                    const struct vk_graphics_pipeline_state *state,
                    uint32_t shader_count, struct vk_shader *const *shaders)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);
   if (!util_queue_is_initialized(&dev->pso_cache.prewarm_queue) ||
       !state->ia || !state->rs || !state->ms || !state->rp ||
       !vk_render_pass_state_has_attachment_info(state->rp))
      return;

   const enum mesa_vk_dynamic_graphics_state required[] = {
      MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
   };
   for (uint32_t i = 0; i < ARRAY_SIZE(required); ++i) {
      if (BITSET_TEST(state->dynamic, required[i]))
         return;
   }

   struct kk_shader *vs = NULL;
   for (uint32_t i = 0; i < shader_count; ++i) {
      if (shaders[i] && shaders[i]->stage == MESA_SHADER_VERTEX)
         vs = container_of(shaders[i], struct kk_shader, vk);
   }
   if (!vs || !vs->info.partial)
      return;

   struct vk_dynamic_graphics_state dyn;
   vk_dynamic_graphics_state_init(&dyn);
   vk_dynamic_graphics_state_fill(&dyn, state);
   kk_prewarm_dynamic_defaults(&dyn, state);
   struct kk_link_state ls = {
      .dyn = &dyn,
      .view_mask = state->mv ? state->mv->view_mask : 0,
      .color_att_count = state->rp->color_attachment_count,
      .depth_format = state->rp->depth_attachment_format,
   };
   for (unsigned i = 0; i < ls.color_att_count; ++i)
      ls.color_formats[i] = state->rp->color_attachment_formats[i];
   for (unsigned i = 0; i < shader_count; ++i) {
      if (!shaders[i])
         continue;
      struct kk_shader *shader = container_of(shaders[i], struct kk_shader, vk);
      ls.shaders[shader->info.stage] = shader;
      if (shader->info.stage == MESA_SHADER_TESS_EVAL)
         ls.tess_prim = shader->info.tess.info.points ? MESA_PRIM_POINTS :
            shader->info.tess.info.mode == TESS_PRIMITIVE_ISOLINES
               ? MESA_PRIM_LINES : MESA_PRIM_TRIANGLES;
   }
   kk_schedule_recipe(dev, &ls, false, state->dynamic);
}

static bool
kk_link_output_types_match(const struct kk_link_state *ls)
{
   const struct kk_shader *fs = ls->shaders[MESA_SHADER_FRAGMENT];
   if (!fs)
      return true;
   for (uint32_t i = 0; i < ls->color_att_count; ++i) {
      uint8_t rt = ls->dyn->cal.color_map[i];
      VkFormat format = ls->color_formats[i];
      if (rt >= MAX_DRAW_BUFFERS || format == VK_FORMAT_UNDEFINED ||
          !(fs->info.fs.color_outputs & BITFIELD_BIT(rt)))
         continue;
      if (kk_format_base_type(vk_format_to_pipe_format(format)) !=
          fs->info.fs.output_base_type[rt])
         return false;
   }
   return true;
}

void
kk_cmd_prewarm_pipeline(struct kk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct kk_rendering_state *render = &cmd->state.gfx.render;
   if (!cmd->state.shaders[MESA_SHADER_VERTEX] ||
       !cmd->state.gfx.render_pass_descriptor ||
       render->color_att_count > MAX_DRAW_BUFFERS)
      return;
   const enum mesa_vk_dynamic_graphics_state required[] = {
      MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY,
      MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES,
      MESA_VK_DYNAMIC_MS_SAMPLE_MASK,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(required); ++i)
      if (!BITSET_TEST(dyn->set, required[i]))
         return;
   if (!dyn->ms.rasterization_samples || dyn->ms.rasterization_samples > 8)
      return;
   if (render->color_att_count) {
      const enum mesa_vk_dynamic_graphics_state output[] = {
         MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
         MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
         MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
         MESA_VK_DYNAMIC_CB_WRITE_MASKS,
         MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(output); ++i)
         if (!BITSET_TEST(dyn->set, output[i]))
            return;
      if (kk_cmd_buffer_device(cmd)->vk.enabled_features.colorWriteEnable &&
          !BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES))
         return;
      if (dyn->cb.logic_op_enable &&
          !BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_CB_LOGIC_OP))
         return;
      for (unsigned i = 0; i < render->color_att_count; ++i) {
         if (dyn->cb.attachments[i].blend_enable &&
             dyn->cb.attachments[i].color_blend_op >= VK_BLEND_OP_ZERO_EXT &&
             !BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED))
            return;
      }
   }
   struct kk_link_state ls = {
      .dyn = dyn,
      .view_mask = render->view_mask,
      .color_att_count = render->color_att_count,
      .depth_format = render->depth_att.vk_format,
   };
   memcpy(ls.shaders, cmd->state.shaders, sizeof(ls.shaders));
   struct kk_shader *tes = ls.shaders[MESA_SHADER_TESS_EVAL];
   if (tes)
      ls.tess_prim = tes->info.tess.info.points ? MESA_PRIM_POINTS :
         tes->info.tess.info.mode == TESS_PRIMITIVE_ISOLINES
            ? MESA_PRIM_LINES : MESA_PRIM_TRIANGLES;
   for (unsigned i = 0; i < ls.color_att_count; ++i)
      ls.color_formats[i] = render->color_att[i].vk_format;
   if (!kk_link_output_types_match(&ls))
      return;
   kk_schedule_recipe(kk_cmd_buffer_device(cmd), &ls, true, NULL);
}
