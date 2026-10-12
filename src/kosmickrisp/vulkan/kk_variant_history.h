#ifndef KK_VARIANT_HISTORY_H
#define KK_VARIANT_HISTORY_H

#include <stddef.h>
#include <stdint.h>

struct disk_cache;
struct kk_variant_history;

struct kk_variant_history *
kk_variant_history_create(struct disk_cache *disk, size_t recipe_size);
void kk_variant_history_destroy(struct kk_variant_history *history);
void kk_variant_history_record(struct kk_variant_history *history,
                               const uint8_t family[32], const void *recipe);
unsigned kk_variant_history_load(struct kk_variant_history *history,
                                 const uint8_t family[32], void *recipes,
                                 unsigned capacity);

#endif
