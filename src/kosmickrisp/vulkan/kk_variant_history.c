#include "kk_variant_history.h"
#include "util/disk_cache.h"
#ifdef ENABLE_SHADER_CACHE
#include "util/disk_cache_os.h"
#endif

#include "util/u_thread.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define KK_HISTORY_FAMILIES 256
#define KK_HISTORY_VARIANTS 32
#define KK_HISTORY_HEADER 24
#define KK_HISTORY_METADATA 16

struct kk_history_family {
   uint8_t key[32];
   uint8_t *recipes;
   uint64_t frequency[KK_HISTORY_VARIANTS];
   uint64_t order[KK_HISTORY_VARIANTS];
   unsigned count;
   unsigned users;
   uint64_t generation;
   uint64_t use;
   bool loaded;
   bool busy;
   bool dirty;
};

struct kk_variant_history {
   struct disk_cache *disk;
   size_t recipe_size;
   mtx_t mutex;
   cnd_t cond;
   thrd_t worker;
   bool stopping;
   uint64_t clock;
   struct kk_history_family families[KK_HISTORY_FAMILIES];
};

static uint64_t
kk_history_read(const uint8_t *data, unsigned bytes)
{
   uint64_t value = 0;
   for (unsigned i = 0; i < bytes; i++)
      value |= (uint64_t)data[i] << (8 * i);
   return value;
}

static void
kk_history_write(uint8_t *data, uint64_t value, unsigned bytes)
{
   for (unsigned i = 0; i < bytes; i++)
      data[i] = value >> (8 * i);
}

static void
kk_history_disk_key(struct kk_variant_history *history,
                    struct kk_history_family *family, cache_key key)
{
   static const char label[] = "kosmickrisp-exact-variant-history-v3";
   uint8_t data[sizeof(label) + 32 + 4];
   memcpy(data, label, sizeof(label));
   memcpy(data + sizeof(label), family->key, 32);
   kk_history_write(data + sizeof(label) + 32, history->recipe_size, 4);
   disk_cache_compute_key(history->disk, data, sizeof(data), key);
}

static struct kk_history_family *
kk_history_family_get(struct kk_variant_history *history,
                      const uint8_t key[32])
{
   struct kk_history_family *available = NULL;
   for (unsigned i = 0; i < KK_HISTORY_FAMILIES; i++) {
      struct kk_history_family *family = &history->families[i];
      if (family->recipes && !memcmp(family->key, key, 32)) {
         family->use = ++history->clock;
         return family;
      }
      if (!family->busy && !family->dirty && !family->users &&
          (!available || !family->recipes || family->use < available->use))
         available = family;
   }
   if (!available)
      return NULL;
   if (!available->recipes) {
      available->recipes = malloc(history->recipe_size * KK_HISTORY_VARIANTS);
      if (!available->recipes)
         return NULL;
   }
   uint8_t *recipes = available->recipes;
   memset(available, 0, sizeof(*available));
   available->recipes = recipes;
   memcpy(available->key, key, 32);
   available->use = ++history->clock;
   return available;
}

static void
kk_history_add(struct kk_variant_history *history,
               struct kk_history_family *family, const void *recipe,
               uint64_t frequency, uint64_t order)
{
   uint64_t last_order = 0;
   unsigned slot = family->count;
   for (unsigned i = 0; i < family->count; i++) {
      if (family->order[i] > last_order)
         last_order = family->order[i];
      if (!memcmp(family->recipes + i * history->recipe_size, recipe,
                  history->recipe_size)) {
         family->frequency[i] = UINT64_MAX - family->frequency[i] < frequency ?
                                  UINT64_MAX : family->frequency[i] + frequency;
         if (order && order < family->order[i])
            family->order[i] = order;
         return;
      }
   }
   if (slot == KK_HISTORY_VARIANTS) {
      slot = 0;
      for (unsigned i = 1; i < family->count; i++) {
         if (family->frequency[i] < family->frequency[slot] ||
             (family->frequency[i] == family->frequency[slot] &&
              family->order[i] > family->order[slot]))
            slot = i;
      }
      if (frequency < family->frequency[slot])
         return;
   } else {
      family->count++;
   }
   memcpy(family->recipes + slot * history->recipe_size, recipe,
          history->recipe_size);
   family->frequency[slot] = frequency;
   family->order[slot] =
      order ? order : last_order + (last_order != UINT64_MAX);
}

static void
kk_history_load_locked(struct kk_variant_history *history,
                       struct kk_history_family *family)
{
   if (family->loaded)
      return;
   mtx_unlock(&history->mutex);
   cache_key key;
   kk_history_disk_key(history, family, key);
   size_t size = 0;
   uint8_t *data = disk_cache_get(history->disk, key, &size);
   unsigned count = 0;
   bool valid = data && size >= KK_HISTORY_HEADER &&
                !memcmp(data, "KKVHIST3", 8) &&
                kk_history_read(data + 8, 4) == 3 &&
                kk_history_read(data + 12, 4) == history->recipe_size &&
                kk_history_read(data + 20, 4) == 0;
   if (valid) {
      count = kk_history_read(data + 16, 4);
      valid = count <= KK_HISTORY_VARIANTS &&
              size == KK_HISTORY_HEADER + count *
                         (KK_HISTORY_METADATA + history->recipe_size);
   }
   uint64_t max_order = 0;
   for (unsigned i = 0; valid && i < count; i++) {
      uint8_t *entry = data + KK_HISTORY_HEADER +
                       i * (KK_HISTORY_METADATA + history->recipe_size);
      uint64_t frequency = kk_history_read(entry, 8);
      uint64_t order = kk_history_read(entry + 8, 8);
      valid = frequency && order && order <= UINT64_MAX - KK_HISTORY_VARIANTS;
      if (order > max_order)
         max_order = order;
   }
   mtx_lock(&history->mutex);
   if (valid) {
      for (unsigned i = 0; i < family->count; i++)
         family->order[i] = max_order + i + 1;
      for (unsigned i = 0; i < count; i++) {
         uint8_t *entry = data + KK_HISTORY_HEADER +
                          i * (KK_HISTORY_METADATA + history->recipe_size);
         kk_history_add(history, family, entry + KK_HISTORY_METADATA,
                        kk_history_read(entry, 8),
                        kk_history_read(entry + 8, 8));
      }
   }
   family->loaded = true;
   free(data);
}

static void
kk_history_flush_locked(struct kk_variant_history *history,
                        struct kk_history_family *family)
{
   family->busy = true;
   kk_history_load_locked(history, family);
   size_t size = KK_HISTORY_HEADER + family->count *
                   (KK_HISTORY_METADATA + history->recipe_size);
   uint8_t *data = calloc(1, size);
   if (!data) {
      family->dirty = false;
      family->busy = false;
      cnd_broadcast(&history->cond);
      return;
   }
   memcpy(data, "KKVHIST3", 8);
   kk_history_write(data + 8, 3, 4);
   kk_history_write(data + 12, history->recipe_size, 4);
   kk_history_write(data + 16, family->count, 4);
   for (unsigned i = 0; i < family->count; i++) {
      uint8_t *entry = data + KK_HISTORY_HEADER +
                       i * (KK_HISTORY_METADATA + history->recipe_size);
      kk_history_write(entry, family->frequency[i], 8);
      kk_history_write(entry + 8, family->order[i], 8);
      memcpy(entry + KK_HISTORY_METADATA,
             family->recipes + i * history->recipe_size, history->recipe_size);
   }
   uint64_t generation = family->generation;
   mtx_unlock(&history->mutex);
   cache_key key;
   kk_history_disk_key(history, family, key);
   disk_cache_wait_for_idle(history->disk);
   disk_cache_remove(history->disk, key);
   disk_cache_put(history->disk, key, data, size, NULL);
   disk_cache_wait_for_idle(history->disk);
   free(data);
   mtx_lock(&history->mutex);
   if (family->generation == generation)
      family->dirty = false;
   family->busy = false;
   cnd_broadcast(&history->cond);
}

static int
kk_history_worker(void *arg)
{
   struct kk_variant_history *history = arg;
   mtx_lock(&history->mutex);
   for (;;) {
      if (!history->stopping) {
         struct timespec deadline;
         clock_gettime(CLOCK_REALTIME, &deadline);
         deadline.tv_sec++;
         while (!history->stopping &&
                cnd_timedwait(&history->cond, &history->mutex,
                                       &deadline) == thrd_success)
            ;
      }
      bool pending = false;
      for (unsigned i = 0; i < KK_HISTORY_FAMILIES; i++) {
         struct kk_history_family *family = &history->families[i];
         if (family->dirty && !family->busy)
            kk_history_flush_locked(history, family);
         pending |= family->dirty;
      }
      if (history->stopping && !pending)
         break;
      if (history->stopping && pending)
         cnd_wait(&history->cond, &history->mutex);
   }
   mtx_unlock(&history->mutex);
   disk_cache_wait_for_idle(history->disk);
   return 0;
}

struct kk_variant_history *
kk_variant_history_create(struct disk_cache *disk, size_t recipe_size)
{
   if (!disk || !recipe_size || recipe_size > 4096)
      return NULL;
#ifdef ENABLE_SHADER_CACHE
   if ((disk->type != DISK_CACHE_MULTI_FILE &&
        disk->type != DISK_CACHE_DATABASE) ||
       disk->blob_put_cb || disk->blob_get_cb || disk->foz_ro_cache)
      return NULL;
#else
   return NULL;
#endif
   struct kk_variant_history *history = calloc(1, sizeof(*history));
   if (!history)
      return NULL;
   history->disk = disk;
   history->recipe_size = recipe_size;
   if (mtx_init(&history->mutex, mtx_plain) != thrd_success)
      goto fail;
   if (cnd_init(&history->cond) != thrd_success)
      goto fail_mutex;
   if (thrd_create(&history->worker, kk_history_worker, history) !=
       thrd_success)
      goto fail_cond;
   return history;

fail_cond:
   cnd_destroy(&history->cond);
fail_mutex:
   mtx_destroy(&history->mutex);
fail:
   free(history);
   return NULL;
}

void
kk_variant_history_destroy(struct kk_variant_history *history)
{
   if (!history)
      return;
   mtx_lock(&history->mutex);
   history->stopping = true;
   cnd_broadcast(&history->cond);
   mtx_unlock(&history->mutex);
   thrd_join(history->worker, NULL);
   for (unsigned i = 0; i < KK_HISTORY_FAMILIES; i++)
      free(history->families[i].recipes);
   cnd_destroy(&history->cond);
   mtx_destroy(&history->mutex);
   free(history);
}

void
kk_variant_history_record(struct kk_variant_history *history,
                          const uint8_t family_key[32], const void *recipe)
{
   if (!history || !family_key || !recipe)
      return;
   mtx_lock(&history->mutex);
   struct kk_history_family *family =
      kk_history_family_get(history, family_key);
   if (family && !history->stopping) {
      kk_history_add(history, family, recipe, 1, 0);
      family->generation++;
      family->dirty = true;
      cnd_signal(&history->cond);
   }
   mtx_unlock(&history->mutex);
}

unsigned
kk_variant_history_load(struct kk_variant_history *history,
                        const uint8_t family_key[32], void *recipes,
                        unsigned capacity)
{
   if (!history || !family_key || !recipes || !capacity)
      return 0;
   mtx_lock(&history->mutex);
   struct kk_history_family *family =
      kk_history_family_get(history, family_key);
   if (!family || history->stopping) {
      mtx_unlock(&history->mutex);
      return 0;
   }
   family->users++;
   while (family->busy)
      cnd_wait(&history->cond, &history->mutex);
   family->busy = true;
   kk_history_load_locked(history, family);
   unsigned count = family->count < capacity ? family->count : capacity;
   unsigned indices[KK_HISTORY_VARIANTS];
   for (unsigned i = 0; i < family->count; i++) {
      indices[i] = i;
      for (unsigned j = i; j > 0; j--) {
         unsigned a = indices[j - 1], b = indices[j];
         if (family->order[a] < family->order[b] ||
             (family->order[a] == family->order[b] &&
              family->frequency[a] >= family->frequency[b]))
            break;
         indices[j - 1] = b;
         indices[j] = a;
      }
   }
   for (unsigned i = 0; i < count; i++)
      memcpy((uint8_t *)recipes + i * history->recipe_size,
             family->recipes + indices[i] * history->recipe_size,
             history->recipe_size);
   family->busy = false;
   family->users--;
   cnd_broadcast(&history->cond);
   mtx_unlock(&history->mutex);
   return count;
}
