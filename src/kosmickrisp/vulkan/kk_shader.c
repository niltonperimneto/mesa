/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_shader.h"
#include "kk_bo.h"
#include "kk_metal_cache.h"

#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_descriptor_set_layout.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_nir_lower_vbo.h"
#include "kk_physical_device.h"
#include "kk_pso_cache.h"
#include "kk_sampler.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/ns_process_info.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "nir_builder.h"
#include "nir_lower_blend.h"
#include "nir_serialize.h"
#include "util/disk_cache.h"

#include "util/hash_table.h"
#include "util/u_atomic.h"

#include "poly/nir/poly_nir.h"

#include "vk_blend.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_common_entrypoints.h"
#include "vk_pipeline.h"

#define KK_SHADER_BINARY_MAGIC 0x4b4b5348u

static const nir_shader_compiler_options *
kk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &kk_nir_options;
}

static const struct vk_pipeline_robustness_state rs_all_supported = {
   .uniform_buffers =
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2,
   .storage_buffers =
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2,
   .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,
};

static struct spirv_to_nir_options
kk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   if (KK_DEBUG(FORCE_ROBUSTNESS))
      rs = &rs_all_supported;
   return (struct spirv_to_nir_options){
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = kk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = kk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = KK_MIN_SSBO_ALIGNMENT,
      .min_ubo_alignment = KK_MIN_UBO_ALIGNMENT,
   };
}

static void
kk_preprocess_nir(UNUSED struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   msl_preprocess_nir(nir);

   /* Cannot be part of msl_preprocess_nir since clc does not expose
    * has_base_workgroup_id */
   nir_lower_compute_system_values_options csv_options = {
      .has_base_global_invocation_id = 0,
      .has_base_workgroup_id = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}

static bool
has_static_depth_stencil_state(const struct vk_graphics_pipeline_state *state)
{
   if (!state->ds)
      return false;

   return !(
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_COMPARE_OP) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_OP) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_COMPARE_MASK) |
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_WRITE_MASK));
}

static const struct vk_multiview_state kk_unknown_mv;
static const struct vk_render_pass_state kk_unknown_rp;
static const struct vk_graphics_pipeline_state kk_unknown_state = {
   .dynamic = {[0 ... BITSET_WORDS(MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX) -
               1] = ~0u},
   .mv = &kk_unknown_mv,
   .rp = &kk_unknown_rp,
};

static const struct vk_multisample_state *
kk_static_ms(const struct vk_graphics_pipeline_state *state)
{
   if (!state || !state->ms ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_SAMPLE_MASK) ||
       BITSET_TEST(state->dynamic,
                   MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE))
      return NULL;

   return state->ms;
}

static bool
kk_state_needs_link(const struct vk_graphics_pipeline_state *state)
{
   return !state->ia || !state->rp ||
          !vk_render_pass_state_has_attachment_info(state->rp) ||
          !state->cb ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_LOGIC_OP) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES) ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED) ||
          state->ms != kk_static_ms(state);
}

static bool
kk_vi_is_static(const struct vk_graphics_pipeline_state *state)
{
   return !KK_DEBUG(DYNAMIC_VI) && state && state->vi &&
          !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI);
}

struct kk_vs_key {
   bool is_points;
   bool dynamic_vi;
   struct vk_vertex_input_state vi;
};

static void
kk_populate_vs_key(struct kk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));
   key->is_points = state->ia && state->ia->primitive_topology ==
                                    VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   key->dynamic_vi = !kk_vi_is_static(state);
   if (!key->dynamic_vi)
      key->vi = *state->vi;
}

struct kk_fs_key {
   VkFormat color_formats[MESA_VK_MAX_COLOR_ATTACHMENTS];
   struct vk_color_blend_state color_blend;
   struct vk_depth_stencil_state ds;
   uint32_t rasterization_samples;
   uint16_t static_sample_mask;
   bool sample_shading_enable;
   bool alpha_to_coverage;
   bool alpha_to_one;
   bool has_depth;
   bool has_stencil;
   bool dynamic_input_attachment_map;
   uint8_t depth_input_att;
   uint8_t stencil_input_att;
};

static void
kk_populate_fs_key(struct kk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state,
                   bool partition)
{
   memset(key, 0, sizeof(*key));

   /* Required since we [de]serialize blend, and render target swizzle for
    * non-native formats */
   if (!partition)
      memcpy(key->color_formats, state->rp->color_attachment_formats,
             sizeof(key->color_formats));

   if (has_static_depth_stencil_state(state))
      key->ds = *(state->ds);

   /* Blend state gets [de]serialized, so we need to hash it */
   if (state->cb && !partition)
      key->color_blend = *(state->cb);

   if (state->ms) {
      if (!partition) {
         key->rasterization_samples = state->ms->rasterization_samples;
         key->static_sample_mask = state->ms->sample_mask;
         key->alpha_to_coverage = state->ms->alpha_to_coverage_enable;
         key->alpha_to_one = state->ms->alpha_to_one_enable;
      }
      key->sample_shading_enable = state->ms->sample_shading_enable;
   }

   /* Depth writes are removed unless there's an actual attachment */
   key->has_depth = state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   key->has_stencil =
      state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;

   /* Values that modify if a forced depth is present or not */
   key->depth_input_att =
      state->ial ? state->ial->depth_att : MESA_VK_ATTACHMENT_NO_INDEX;
   key->stencil_input_att =
      state->ial ? state->ial->stencil_att : MESA_VK_ATTACHMENT_NO_INDEX;
   key->dynamic_input_attachment_map =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP);
}

enum kk_feature_key {
   KK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   KK_FEAT_NULL_DESCRIPTOR = BITFIELD_BIT(1),
   KK_FEAT_IMAGE_VIEW_MIN_LOD = BITFIELD_BIT(2),
};

static enum kk_feature_key
kk_make_feature_key(const struct vk_features *feats)
{
   enum kk_feature_key key = 0;
   if (feats->customBorderColors)
      key |= KK_FEAT_CUSTOM_BORDER;
   if (feats->nullDescriptor)
      key |= KK_FEAT_NULL_DESCRIPTOR;
   if (feats->minLod)
      key |= KK_FEAT_IMAGE_VIEW_MIN_LOD;
   return key;
}

static bool
kk_is_possible_both_depth_clip_clamp(
   const struct vk_graphics_pipeline_state *state, bool enabled);

static bool
kk_state_may_draw_points(const struct vk_graphics_pipeline_state *state);

#define KK_HASH_FIELD(ctx, value) \
   _mesa_blake3_update((ctx), &(value), sizeof(value))

static void
kk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *enabled_features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   enum { VI, PRE_RASTER, DS, BLEND, MS, ATTACHMENTS, MV, FEATURES, COUNT };
   const char *names[COUNT] = {
      "vertex_input", "pre_raster", "depth_stencil", "blend",
      "multisampling", "attachments", "multiview", "features",
   };
   struct mesa_blake3 components[COUNT];
   blake3_hash digests[COUNT];
   for (unsigned i = 0; i < COUNT; i++)
      _mesa_blake3_init(&components[i]);

   if (stages & VK_SHADER_STAGE_VERTEX_BIT) {
      struct kk_vs_key key;
      kk_populate_vs_key(&key, state);
      KK_HASH_FIELD(&components[PRE_RASTER], key.is_points);
      KK_HASH_FIELD(&components[VI], key.dynamic_vi);
      KK_HASH_FIELD(&components[VI], key.vi.bindings_valid);
      KK_HASH_FIELD(&components[VI], key.vi.attributes_valid);
      for (unsigned i = 0; i < MESA_VK_MAX_VERTEX_BINDINGS; i++) {
         if (!(key.vi.bindings_valid & BITFIELD_BIT(i)))
            continue;
         KK_HASH_FIELD(&components[VI], key.vi.bindings[i].stride);
         KK_HASH_FIELD(&components[VI], key.vi.bindings[i].input_rate);
         KK_HASH_FIELD(&components[VI], key.vi.bindings[i].divisor);
      }
      for (unsigned i = 0; i < MESA_VK_MAX_VERTEX_ATTRIBUTES; i++) {
         if (!(key.vi.attributes_valid & BITFIELD_BIT(i)))
            continue;
         KK_HASH_FIELD(&components[VI], key.vi.attributes[i].binding);
         KK_HASH_FIELD(&components[VI], key.vi.attributes[i].format);
         KK_HASH_FIELD(&components[VI], key.vi.attributes[i].offset);
      }
      const uint32_t view_mask = state->mv ? state->mv->view_mask : 0u;
      const bool emulate_viewport_z =
         kk_is_possible_both_depth_clip_clamp(state, false);
      const bool may_draw_points = kk_state_may_draw_points(state);
      KK_HASH_FIELD(&components[MV], view_mask);
      KK_HASH_FIELD(&components[PRE_RASTER], emulate_viewport_z);
      KK_HASH_FIELD(&components[PRE_RASTER], may_draw_points);
   }

   if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      struct kk_fs_key key;
      kk_populate_fs_key(&key, state,
                         !(stages & VK_SHADER_STAGE_VERTEX_BIT) ||
                            KK_DEBUG(PARTITION) || kk_state_needs_link(state) ||
                            kk_state_may_draw_points(state));
      KK_HASH_FIELD(&components[DS], key.ds.depth.test_enable);
      KK_HASH_FIELD(&components[DS], key.ds.depth.write_enable);
      KK_HASH_FIELD(&components[DS], key.ds.depth.compare_op);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.test_enable);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.write_enable);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.op.fail);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.op.pass);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.op.depth_fail);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.op.compare);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.compare_mask);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.front.write_mask);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.op.fail);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.op.pass);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.op.depth_fail);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.op.compare);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.compare_mask);
      KK_HASH_FIELD(&components[DS], key.ds.stencil.back.write_mask);
      KK_HASH_FIELD(&components[BLEND], key.color_blend.logic_op_enable);
      KK_HASH_FIELD(&components[BLEND], key.color_blend.logic_op);
      KK_HASH_FIELD(&components[BLEND], key.color_blend.attachment_count);
      KK_HASH_FIELD(&components[BLEND], key.color_blend.color_write_enables);
      for (unsigned i = 0; i < key.color_blend.attachment_count; i++) {
         const struct vk_color_blend_attachment_state *att =
            &key.color_blend.attachments[i];
         KK_HASH_FIELD(&components[BLEND], att->blend_enable);
         KK_HASH_FIELD(&components[BLEND], att->src_color_blend_factor);
         KK_HASH_FIELD(&components[BLEND], att->dst_color_blend_factor);
         KK_HASH_FIELD(&components[BLEND], att->src_alpha_blend_factor);
         KK_HASH_FIELD(&components[BLEND], att->dst_alpha_blend_factor);
         KK_HASH_FIELD(&components[BLEND], att->write_mask);
         KK_HASH_FIELD(&components[BLEND], att->color_blend_op);
         KK_HASH_FIELD(&components[BLEND], att->alpha_blend_op);
         KK_HASH_FIELD(&components[BLEND], att->src_premultiplied);
         KK_HASH_FIELD(&components[BLEND], att->dst_premultiplied);
         KK_HASH_FIELD(&components[BLEND], att->blend_overlap);
         KK_HASH_FIELD(&components[BLEND], att->clamp_results);
      }
      KK_HASH_FIELD(&components[MS], key.rasterization_samples);
      KK_HASH_FIELD(&components[MS], key.static_sample_mask);
      KK_HASH_FIELD(&components[MS], key.sample_shading_enable);
      KK_HASH_FIELD(&components[MS], key.alpha_to_coverage);
      KK_HASH_FIELD(&components[MS], key.alpha_to_one);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.color_formats);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.has_depth);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.has_stencil);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.dynamic_input_attachment_map);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.depth_input_att);
      KK_HASH_FIELD(&components[ATTACHMENTS], key.stencil_input_att);
      const uint32_t view_mask = state->mv ? state->mv->view_mask : 0u;
      KK_HASH_FIELD(&components[MV], view_mask);
   }

   enum kk_feature_key feature_key = kk_make_feature_key(enabled_features);
   KK_HASH_FIELD(&components[FEATURES], feature_key);

   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   for (unsigned i = 0; i < COUNT; i++) {
      _mesa_blake3_final(&components[i], digests[i]);
      _mesa_blake3_update(&blake3_ctx, digests[i], sizeof(digests[i]));
   }
   _mesa_blake3_final(&blake3_ctx, blake3_out);
   struct kk_physical_device *physical =
      container_of(device, struct kk_physical_device, vk);
   for (unsigned i = 0; i < COUNT; i++)
      kk_log_cache_component(physical, stages, names[i], digests[i],
                             blake3_out);
}

#undef KK_HASH_FIELD

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct kk_descriptor_set_layout *set_layout =
      vk_to_kk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct kk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct kk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion
             ? &sampler->vk.ycbcr_conversion->state
             : NULL;
}

static unsigned
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static bool
kk_nir_swizzle_fragment_output(nir_builder *b, nir_intrinsic_instr *intrin,
                               void *data)
{
   if (intrin->intrinsic != nir_intrinsic_store_output &&
       intrin->intrinsic != nir_intrinsic_load_output)
      return false;

   unsigned slot = nir_intrinsic_io_semantics(intrin).location;
   if (slot < FRAG_RESULT_DATA0)
      return false;

   const struct vk_graphics_pipeline_state *state =
      (const struct vk_graphics_pipeline_state *)data;
   VkFormat vk_format =
      state->rp->color_attachment_formats[slot - FRAG_RESULT_DATA0];
   if (vk_format == VK_FORMAT_UNDEFINED)
      return false;

   enum pipe_format format = vk_format_to_pipe_format(vk_format);
   const struct kk_va_format *supported_format = kk_get_va_format(format);

   /* Check if we have to apply any swizzle */
   if (!supported_format->is_native) {
      unsigned channel_unswizzle[] = {
         supported_format->unswizzle.red, supported_format->unswizzle.green,
         supported_format->unswizzle.blue, supported_format->unswizzle.alpha};

      if (intrin->intrinsic == nir_intrinsic_store_output) {
         unsigned channel_swizzle[4] = {0u};
         for (uint32_t i = 0u; i < 4; ++i)
            channel_swizzle[channel_unswizzle[i]] = i;

         b->cursor = nir_before_instr(&intrin->instr);
         nir_def *to_replace = intrin->src[0].ssa;
         nir_def *swizzled = nir_swizzle(b, to_replace, channel_swizzle,
                                         to_replace->num_components);
         nir_src_rewrite(&intrin->src[0], swizzled);
      } else {
         b->cursor = nir_after_instr(&intrin->instr);
         nir_def *to_replace = &intrin->def;
         nir_def *swizzled = nir_swizzle(b, to_replace, channel_unswizzle,
                                         to_replace->num_components);
         nir_def_rewrite_uses_after(to_replace, swizzled);
      }
      return true;
   }

   return false;
}

static bool
kk_is_possible_both_depth_clip_clamp(
   const struct vk_graphics_pipeline_state *state, bool enabled)
{
   bool dyn_clamp =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
   bool dyn_clip =
      BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLIP_ENABLE);

   /* If rasterization state is null, clamp must be dynamic, and clip must be
    * inverted from clamp if not also dynamic. Thus, they cannot match unless
    * both are dynamic. */
   if (!state->rs)
      return dyn_clamp && dyn_clip;

   /* If clip is static and inverted from clamp, they can never be the same */
   if (!dyn_clip &&
       state->rs->depth_clip_enable == VK_MESA_DEPTH_CLIP_ENABLE_NOT_CLAMP)
      return false;

   /* Need to account for:
    * - Both are dynamic
    * - Both are static and match the desired value
    * - One is dynamic and the other is static and matches the desired value
    */
   bool static_clamp_match = state->rs->depth_clamp_enable == enabled;
   bool static_clip_match =
      vk_rasterization_state_depth_clip_enable(state->rs) == enabled;
   return (dyn_clamp || static_clamp_match) && (dyn_clip || static_clip_match);
}

static void
kk_lower_vs_vbo(nir_shader *nir, const struct vk_graphics_pipeline_state *state,
                const struct vk_pipeline_robustness_state *rs)
{
   assert(!(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
          "Fixed-function attributes not used in Vulkan");
   NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
   /* the shader_out portion of this is load-bearing even for tess eval */
   /* Fold constant offset srcs for IO. */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   bool robustness2 =
      rs->vertex_inputs ==
      VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2;
   if (!kk_vi_is_static(state)) {
      NIR_PASS(_, nir, kk_nir_lower_vbo_dynamic, robustness2);
      return;
   }

   struct kk_attribute attributes[KK_MAX_ATTRIBS] = {};
   uint64_t attribs_read = nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;
   u_foreach_bit(i, state->vi->attributes_valid) {
      const struct vk_vertex_attribute_state *attr = &state->vi->attributes[i];
      assert(state->vi->bindings_valid & BITFIELD_BIT(attr->binding));
      const struct vk_vertex_binding_state *binding =
         &state->vi->bindings[attr->binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(i));
      attributes[slot].divisor = binding->divisor;
      attributes[slot].binding = attr->binding;
      attributes[slot].format = vk_format_to_pipe_format(attr->format);
      attributes[slot].buf = attr->binding;
      attributes[slot].instanced =
         binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   }
   NIR_PASS(_, nir, kk_nir_lower_vbo, attributes, robustness2);
}

/* Lowering for the stage which ends up as the vertex stage on hardware */
static void
kk_lower_hw_vs(nir_shader *nir, const struct vk_graphics_pipeline_state *state,
               bool is_point, bool lower_z)
{
   if (is_point)
      NIR_PASS(_, nir, msl_ensure_vertex_point_size_output);
   else
      nir_shader_intrinsics_pass(nir, msl_nir_vs_remove_point_size_write,
                                 nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, msl_ensure_vertex_position_output);
   if (lower_z) {
      NIR_PASS(_, nir, nir_lower_clip_halfz_dynamic);

      /* In any situation where both clip and clamp can be disabled, we need to
       * add viewport Z transform emulation to the shader */
      if (kk_is_possible_both_depth_clip_clamp(state, false))
         NIR_PASS(_, nir, msl_nir_lower_vs_disabled_depth_clamp_clip);
   }

   NIR_PASS(_, nir, msl_nir_vs_io_types);
}

static inline uint8_t
kk_get_logical_color_att_index(const struct vk_graphics_pipeline_state *state,
                               uint8_t i)
{
   return (state->cal) ? state->cal->color_map[i] : i;
}

static void
kk_lower_fs_blend(nir_shader *nir,
                  const struct vk_graphics_pipeline_state *state)
{
   nir_lower_blend_options opts = {
      .scalar_blend_const = false,
      .logicop_enable = state->cb->logic_op_enable,
      .logicop_func = vk_logic_op_to_pipe(state->cb->logic_op),
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");
   for (unsigned i = 0; i < ARRAY_SIZE(opts.rt); ++i) {
      opts.rt[i].format = PIPE_FORMAT_NONE;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(opts.rt); ++i) {
      uint8_t logical_index = kk_get_logical_color_att_index(state, i);
      if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
         continue;
      }

      nir_lower_blend_rt *rt = &opts.rt[logical_index];

      const struct vk_color_blend_attachment_state *att =
         &state->cb->attachments[i];
      rt->format =
         vk_format_to_pipe_format(state->rp->color_attachment_formats[i]);
      rt->colormask = att->write_mask;

      if (!att->blend_enable) {
         rt->rgb.src_factor = rt->alpha.src_factor = PIPE_BLENDFACTOR_ONE;
         rt->rgb.dst_factor = rt->alpha.dst_factor = PIPE_BLENDFACTOR_ZERO;
         rt->rgb.func = rt->alpha.func = PIPE_BLEND_ADD;
      } else if (att->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
         rt->advanced_blend = true;
         rt->blend_mode = vk_advanced_blend_op_to_pipe(att->color_blend_op);
         rt->src_premultiplied = att->src_premultiplied;
         rt->dst_premultiplied = att->dst_premultiplied;
         rt->overlap = vk_blend_overlap_to_pipe(att->blend_overlap);

         /* Unused by runtime and not supported */
         assert(!att->clamp_results);
      } else {
         rt->rgb.src_factor =
            vk_blend_factor_to_pipe(att->src_color_blend_factor);
         rt->rgb.dst_factor =
            vk_blend_factor_to_pipe(att->dst_color_blend_factor);
         rt->rgb.func = vk_blend_op_to_pipe(att->color_blend_op);

         rt->alpha.src_factor =
            vk_blend_factor_to_pipe(att->src_alpha_blend_factor);
         rt->alpha.dst_factor =
            vk_blend_factor_to_pipe(att->dst_alpha_blend_factor);
         rt->alpha.func = vk_blend_op_to_pipe(att->alpha_blend_op);
      }
   }
   /* Fold constant offset srcs for IO. */
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_lower_blend, &opts);
   NIR_PASS(_, nir, nir_lower_undef_to_zero, NULL);
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static void
kk_lower_fs(struct kk_device *dev, nir_shader *nir,
            const struct vk_graphics_pipeline_state *state, bool partition)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   const struct vk_multisample_state *ms =
      partition ? NULL : kk_static_ms(state);

   nir->info.fs.uses_sample_shading |=
      state->ms && state->ms->sample_shading_enable;

   /* Must be lowered before blending */
   if (ms) {
      NIR_PASS(_, nir, msl_nir_lower_multisample_alpha,
               ms->alpha_to_coverage_enable, ms->alpha_to_one_enable);
   }

   /* msl_nir_lower_sample_shading needs to go before blending since
    * nir_lower_blend will always set uses_sample_shading to true if there's any
    * output read. I believe we do not need to lower it always, that is why it
    * goes here but need to double check. */
   if (nir->info.fs.uses_sample_shading)
      NIR_PASS(_, nir, msl_nir_lower_sample_shading);

   if (state->cb && !partition)
      kk_lower_fs_blend(nir, state);

   if (!partition) {
      enum pipe_format rts[MAX_DRAW_BUFFERS] = {PIPE_FORMAT_NONE};
      const struct vk_render_pass_state *rp = state->rp;
      for (uint32_t i = 0u; i < rp->color_attachment_count; ++i) {
         uint8_t logical_index = kk_get_logical_color_att_index(state, i);
         if (logical_index != MESA_VK_ATTACHMENT_UNUSED) {
            rts[logical_index] =
               vk_format_to_pipe_format(rp->color_attachment_formats[i]);
         }
      }

      NIR_PASS(_, nir, msl_nir_fs_force_output_signedness, rts);
   }

   if ((!partition &&
        state->rp->depth_attachment_format == VK_FORMAT_UNDEFINED) ||
       nir->info.fs.early_fragment_tests) {
      NIR_PASS(_, nir, msl_nir_fs_remove_depth_write);
   }

   /* In any situation where both clip and clamp can be enabled, we need to
    * add clamp emulation to the shader */
   if (partition || kk_is_possible_both_depth_clip_clamp(state, true))
      NIR_PASS(_, nir, msl_nir_lower_fs_combined_depth_clamp_clip);

   /* Input attachments are treated as 2D textures. Fixes sampler dimension */
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
            NULL);

   /* Swizzle non-native formats' outputs */
   if (!partition)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               kk_nir_swizzle_fragment_output, nir_metadata_control_flow,
               (void *)state);

   NIR_PASS(_, nir, msl_nir_fs_io_types);

   if (ms && ms->rasterization_samples && ms->sample_mask != UINT16_MAX) {

      /* KK_WORKAROUND_7 */
      if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(7))) {
         if (!nir->info.fs.early_fragment_tests) {
            nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
            nir_builder b = nir_builder_at(nir_after_impl(entrypoint));

            nir_def *sample_id = nir_load_sample_id(&b);
            nir_def *sample_bit = nir_ishl(&b, nir_imm_int(&b, 1), sample_id);
            nir_def *sample_mask_bit =
               nir_iand(&b, nir_load_sample_mask_in(&b), sample_bit);
            nir_discard_if(&b, nir_ieq_imm(&b, sample_mask_bit, 0u));
         }
      }
      NIR_PASS(_, nir, msl_lower_static_sample_mask, ms->sample_mask);
   }
   /* Check https://github.com/KhronosGroup/Vulkan-Portability/issues/54 for
    * explanation on why we need this. */
   else if (nir->info.fs.needs_full_quad_helper_invocations ||
            nir->info.fs.needs_coarse_quad_helper_invocations) {
      struct kk_physical_device *pdev = kk_device_physical(dev);

      /* Metal by default merges triangles which results in incorrect edges, see
       * mentioned issue above. The issue is that if we disable them through
       * writing to the sample mask, then derivatives are broken for multisample
       * rendering in M1 and M2 with macOS 26. This bug is fixed in macOS 27.
       *
       * This is why we are choosing the lesser evil which is incorrect edges
       * for multisampled rendering. Since CTS does not have tests for the edge
       * case with multisample, we get a clean run. This does not mean we are
       * Vulkan conformant with macOS26 for M1 and M2.
       */
      bool ms_bug_present = !ns_is_os_version_at_least(27, 0, 0) &&
                            (pdev->info.gpu_apple_family == 7 ||
                             pdev->info.gpu_apple_family == 8) &&
                            ms && ms->rasterization_samples > 1;
      if (!ms_bug_present)
         NIR_PASS(_, nir, msl_disable_triangle_merge);
   }

   /* KK_WORKAROUND_5 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(5)))
      NIR_PASS(_, nir, msl_nir_fake_guard_for_discards);
   /* KK_WORKAROUND_4 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(4))) {
      NIR_PASS(_, nir, nir_lower_helper_writes, true);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }
}

static bool
kk_fs_reads_input_attachment(const nir_shader *nir, uint32_t depth_att,
                             uint32_t stencil_att, bool indices_known)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_deref_load &&
                intr->intrinsic != nir_intrinsic_image_deref_sparse_load)
               continue;

            nir_deref_instr *deref = nir_src_as_deref(intr->src[0u]);
            enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
            if (dim != GLSL_SAMPLER_DIM_SUBPASS &&
                dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
               continue;

            nir_variable *var = nir_deref_instr_get_variable(deref);
            if (!indices_known || var == NULL || var->data.index == depth_att ||
                var->data.index == stencil_att)
               return true;
         }
      }
   }

   return false;
}

static bool
kk_fs_needs_forced_depth_write(const nir_shader *nir,
                               const struct vk_graphics_pipeline_state *state)
{
   const struct vk_input_attachment_location_state *ial = state->ial;
   if (ial == NULL)
      return false;

   const bool has_depth_ia =
      state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED &&
      ial->depth_att != MESA_VK_ATTACHMENT_NO_INDEX;
   const bool has_stencil_ia =
      state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED &&
      ial->stencil_att != MESA_VK_ATTACHMENT_NO_INDEX;

   if (!has_depth_ia && !has_stencil_ia)
      return false;

   if (has_static_depth_stencil_state(state)) {
      const struct vk_depth_stencil_state *ds = state->ds;

      const bool writes_depth =
         has_depth_ia && ds->depth.test_enable && ds->depth.write_enable;
      const bool writes_stencil =
         has_stencil_ia && ds->stencil.test_enable &&
         ds->stencil.write_enable &&
         (ds->stencil.front.write_mask | ds->stencil.back.write_mask);

      if (!writes_depth && !writes_stencil)
         return false;
   }

   const uint32_t depth_att =
      has_depth_ia ? ial->depth_att : NIR_VARIABLE_NO_INDEX;
   const uint32_t stencil_att =
      has_stencil_ia ? ial->stencil_att : NIR_VARIABLE_NO_INDEX;

   const bool indices_known =
      !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP);

   return kk_fs_reads_input_attachment(nir, depth_att, stencil_att,
                                       indices_known);
}

static bool
kk_topology_is_unknown(const struct vk_graphics_pipeline_state *state,
                       bool partition)
{
   return partition &&
          (!state->ia ||
           BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY));
}

static bool
kk_topology_is_point(const struct vk_graphics_pipeline_state *state,
                     bool partition)
{
   if (kk_topology_is_unknown(state, partition))
      return true;

   return state->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
}

static void
kk_lower_nir(struct kk_device *dev, nir_shader *nir, bool emulated_stage,
             bool partition, const struct vk_pipeline_robustness_state *rs,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             const struct vk_graphics_pipeline_state *state,
             enum kk_feature_key features)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   if (nir->info.io_lowered)
      return;

   if (nir->info.stage == MESA_SHADER_GEOMETRY ||
       (!emulated_stage && (nir->info.stage == MESA_SHADER_VERTEX ||
                            nir->info.stage == MESA_SHADER_TESS_EVAL))) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
         var->data.interpolation = INTERP_MODE_NONE;
         var->data.centroid = false;
         var->data.sample = false;
      }
      nir_validate_shader(nir, "KK independent producer interpolation");
   }

   if (KK_DEBUG(FORCE_ROBUSTNESS))
      rs = &rs_all_supported;

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   /* Massage IO related variables to please Metal */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, nir, kk_nir_lower_vs_multiview, state->mv->view_mask);

      /* kk_nir_lower_vs_multiview may create a temporary array to assign the
       * correct view index. Since we don't handle derefs, we need to get rid of
       * them. */
      NIR_PASS(_, nir, nir_lower_vars_to_scratch, 0,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, kk_nir_lower_fs_multiview, state->mv->view_mask);

      /* Run before nir_lower_input_attachments() below, which rewrites the
       * subpass loads it looks for. */
      if (kk_fs_needs_forced_depth_write(nir, state))
         NIR_PASS(_, nir, msl_ensure_depth_write);

      msl_nir_lower_input_attachments(nir, state->ial, state->cal);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Common msl texture lowering needs to happen after ycbcr lowering and
    * before descriptor lowering. */
   NIR_PASS(_, nir, msl_lower_textures);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            kk_buffer_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            kk_buffer_addr_format(rs->uniform_buffers));

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            type_size_vec4,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* QMD::SHARED_MEMORY_SIZE requires an alignment of 256B so it's safe to
       * align everything up to 16B so we can write whole vec4s.
       */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value,
            NULL);
   nir->info.io_lowered = true;

   /* Required before kk_nir_lower_vbo so load_input intrinsics' parents are
    * load_const, otherwise the pass will complain */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   /* These passes operate on lowered IO. */
   if ((nir->info.stage == MESA_SHADER_VERTEX ||
        nir->info.stage == MESA_SHADER_TESS_EVAL) &&
       !emulated_stage) {
      bool is_point = nir->info.stage == MESA_SHADER_TESS_EVAL
                         ? nir->info.tess.point_mode
                         : kk_topology_is_point(state, partition);
      kk_lower_hw_vs(nir, state, is_point, true);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      kk_lower_fs(dev, nir, state, partition);
   }

   if (features & KK_FEAT_CUSTOM_BORDER)
      NIR_PASS(_, nir, kk_nir_lower_custom_border);

   if (features & KK_FEAT_IMAGE_VIEW_MIN_LOD)
      NIR_PASS(_, nir, kk_nir_lower_image_view_min_lod);

   /* Descriptor lowering needs to happen after lowering blend since we will
    * generate a nir_intrinsic_load_blend_const_color_rgba which gets lowered by
    * the lower descriptor pass
    */
   NIR_PASS(_, nir, kk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts);

   /* KK_WORKAROUND_16 - must be called after kk_nir_lower_descriptors(), which
    * lowers all image intrinsics to be bindless */
   if (rs->images ==
          VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2 &&
       !(pdev->settings.disabled_workarounds & BITFIELD64_BIT(16)))
      NIR_PASS(_, nir, msl_lower_robustness2_images);

   /* KK_WORKAROUND_18 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(18)))
      NIR_PASS(_, nir, kk_nir_add_device_barrier_workaround);

   NIR_PASS(_, nir, kk_nir_lower_textures);

   if (features & KK_FEAT_NULL_DESCRIPTOR)
      NIR_PASS(_, nir, kk_nir_lower_null_images);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
}

static const struct vk_shader_ops kk_shader_ops;

struct kk_fs_variant_entry {
   struct kk_fs_link_key key;
   struct kk_fs_variant variant;
   cnd_t ready;
   bool compiling;
   VkResult result;
};

static void
kk_shader_destroy(struct vk_device *vk_dev, struct kk_shader *shader,
                  const VkAllocationCallbacks *pAllocator)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);

   struct kk_pipeline_handles *pipe = &shader->pipeline;

   if (shader->info.stage == MESA_SHADER_COMPUTE) {
      mtl_release(pipe->cs);
      pipe->cs = NULL;
   } else if (shader->info.stage == MESA_SHADER_VERTEX) {
      /* Pre-render compute pipelines. */
      for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i) {
         mtl_release(pipe->gfx.pre_render[i]);
         pipe->gfx.pre_render[i] = NULL;
      }
      pipe->gfx.pre_render_count = 0u;

      /* Graphics pipeline. */
      mtl_release(pipe->gfx.render);
      if (pipe->gfx.ds_handle)
         mtl_release(pipe->gfx.ds_handle);
      pipe->gfx.render = NULL;
      pipe->gfx.ds_handle = NULL;

      for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i)
         mtl_release(pipe->gfx.pre_render[i]);

      /* Serialization data. */
      u_foreach_bit(i, shader->info.vs.additional_stages_bits) {
         struct msl_compile_data *data = &shader->msl_data[i];
         ralloc_free((void *)data->entrypoint_name);
         ralloc_free((void *)data->code);
      }
   }

   struct msl_compile_data *data = &shader->msl_data[shader->info.stage];
   ralloc_free((void *)data->entrypoint_name);
   ralloc_free((void *)data->code);

   ralloc_free((void *)shader->gs_main.entrypoint_name);
   ralloc_free((void *)shader->gs_main.code);
   if (shader->gs_indices)
      kk_destroy_bo(dev, shader->gs_indices);

   if (shader->fast.fn)
      mtl_release(shader->fast.fn);
   if (shader->fast.lib)
      mtl_release(shader->fast.lib);

   if (shader->fast_points.fn)
      mtl_release(shader->fast_points.fn);
   if (shader->fast_points.lib)
      mtl_release(shader->fast_points.lib);
   ralloc_free((void *)shader->msl_points.entrypoint_name);
   ralloc_free((void *)shader->msl_points.code);

   if (shader->so_cs)
      mtl_release(shader->so_cs);
   ralloc_free((void *)shader->msl_emu.entrypoint_name);
   ralloc_free((void *)shader->msl_emu.code);

   if (shader->point_cs)
      mtl_release(shader->point_cs);
   if (shader->point_gs)
      kk_shader_destroy(vk_dev, shader->point_gs, NULL);
   ralloc_free((void *)shader->point_emu.entrypoint_name);
   ralloc_free((void *)shader->point_emu.code);
   ralloc_free((void *)shader->point_gs_main.entrypoint_name);
   ralloc_free((void *)shader->point_gs_main.code);
   ralloc_free((void *)shader->point_rast.entrypoint_name);
   ralloc_free((void *)shader->point_rast.code);

   if (shader->pending) {
      mtl_release(shader->pending->fragment_function);
      mtl_release(shader->pending->fragment_library);
      mtl_release(shader->pending->vertex_function);
      mtl_release(shader->pending->vertex_library);
      cnd_destroy(&shader->pending->ready);
      free(shader->pending);
   }

   if (shader->variants) {
      hash_table_foreach(shader->variants, he) {
         struct kk_fs_variant_entry *entry = he->data;
         mtl_release(entry->variant.fn);
         mtl_release(entry->variant.lib);
         cnd_destroy(&entry->ready);
      }
      _mesa_hash_table_destroy(shader->variants, NULL);
   }
   mtx_destroy(&shader->variants_lock);
   free(shader->nir_blob);

   vk_shader_free(&dev->vk, pAllocator, &shader->vk);
}

static bool
gather_vs_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   BITSET_WORD *attribs_read = data;
   BITSET_SET(attribs_read, (io.location - VERT_ATTRIB_GENERIC0));
   return false;
}

static bool
fs_uses_flat_varying(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block_safe(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_input)
                  return true;
            }
         }
      }
   }
   return false;
}

static bool
gather_fs_io(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct kk_shader_info *info = data;

   if (intr->intrinsic == nir_intrinsic_load_primitive_id) {
      info->fs.reads_primitive_id = true;
      return false;
   }

   if (intr->intrinsic == nir_intrinsic_load_input ||
       intr->intrinsic == nir_intrinsic_load_interpolated_input) {
      nir_io_semantics io = nir_intrinsic_io_semantics(intr);
      if (io.location >= VARYING_SLOT_VAR0 && io.location <= VARYING_SLOT_VAR31)
         info->fs.inputs_read |= BITFIELD64_RANGE(io.location, io.num_slots);
      if (intr->intrinsic == nir_intrinsic_load_input &&
          io.location + io.num_slots <= 64)
         info->fs.flat_inputs |= BITFIELD64_RANGE(io.location, io.num_slots);
      return false;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   if (io.dual_source_blend_index ||
       io.location == FRAG_RESULT_DUAL_SRC_BLEND)
      info->fs.dual_source_blend = true;

   if (io.location == FRAG_RESULT_DEPTH) {
      info->fs.writes_depth = true;
   } else if (io.location == FRAG_RESULT_STENCIL) {
      info->fs.writes_stencil = true;
   } else if (io.location >= FRAG_RESULT_DATA0 &&
              io.location < FRAG_RESULT_DATA0 + MAX_DRAW_BUFFERS) {
      unsigned rt = io.location - FRAG_RESULT_DATA0;
      info->fs.color_outputs |= BITFIELD_BIT(rt);
      info->fs.output_base_type[rt] =
         nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
   }
   return false;
}

static void
gather_shader_info(struct kk_shader *shader, nir_shader *nir,
                   const struct vk_graphics_pipeline_state *state)
{
   shader->info.stage = nir->info.stage;
   shader->info.uses_per_draw_data = msl_gather_uses_per_draw_data(nir);
   shader->info.num_cull_distances = nir->info.cull_distance_array_size;
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_shader_intrinsics_pass(nir, gather_vs_inputs, nir_metadata_all,
                                 &shader->info.vs.attribs_read);
      shader->info.vs.dynamic_vi = !kk_vi_is_static(state);
      shader->info.vs.outputs_written = nir->info.outputs_written;
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* Some meta shaders like vk-meta-resolve will have depth_layout as NONE
       * which is not a valid Metal layout */
      if (nir->info.fs.depth_layout == FRAG_DEPTH_LAYOUT_NONE)
         nir->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      shader->info.fs.uses_flat_varyings = fs_uses_flat_varying(nir);
      shader->info.fs.ms_static =
         !shader->info.partial && kk_static_ms(state) != NULL;
      nir_shader_intrinsics_pass(nir, gather_fs_io, nir_metadata_all,
                                 &shader->info);
   } else if (nir->info.stage == MESA_SHADER_COMPUTE) {
      shader->info.cs.local_size.x = nir->info.workgroup_size[0];
      shader->info.cs.local_size.y = nir->info.workgroup_size[1];
      shader->info.cs.local_size.z = nir->info.workgroup_size[2];
   }
}

static void
modify_nir_info(nir_shader *nir)
{
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* Vertex attribute fetch is done in shader through argument buffers. */
      nir->info.inputs_read = 0u;
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* Some meta shaders like vk-meta-resolve will have depth_layout as NONE
       * which is not a valid Metal layout */
      if (nir->info.fs.depth_layout == FRAG_DEPTH_LAYOUT_NONE)
         nir->info.fs.depth_layout = FRAG_DEPTH_LAYOUT_ANY;

      /* These values are part of the declaration and go with IO. We only
       * require the instrunctions to understand interpolation mode. */
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE);
   }
}

static bool
lower_vertex_id_zero_base(struct nir_builder *b, nir_intrinsic_instr *intrin,
                          void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *id = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   nir_def_rewrite_uses(&intrin->def, id);

   return true;
}

static bool
kk_nir_lower_vertex_id_zero_base(struct nir_shader *nir)
{
   bool progress = nir_shader_intrinsics_pass(nir, lower_vertex_id_zero_base,
                                              nir_metadata_control_flow, NULL);

   if (progress) {
      BITSET_SET(nir->info.system_values_read,
                 SYSTEM_VALUE_GLOBAL_INVOCATION_ID);
      BITSET_CLEAR(nir->info.system_values_read,
                   SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   }
   return progress;
}

/* Consumes nir */
static void
kk_compile_msl(struct kk_physical_device *pdev, nir_shader *nir,
               const struct vk_graphics_pipeline_state *state, bool partition,
               struct msl_compile_data *data, uint64_t *outputs_written)
{
   NIR_PASS(_, nir, kk_nir_lower_poly);

   if (partition && (nir->info.stage == MESA_SHADER_VERTEX ||
                     nir->info.stage == MESA_SHADER_FRAGMENT))
      NIR_PASS(_, nir, msl_nir_widen_generic_varyings);

   msl_optimize_nir(nir);
   modify_nir_info(nir);

   if (outputs_written)
      *outputs_written = nir->info.outputs_written;

   struct nir_to_msl_options translate_options = {
      .mem_ctx = NULL,
      .disabled_workarounds = pdev->settings.disabled_workarounds,
      .separate_io = partition,
      .use_helpers = !KK_DEBUG(NO_MSL_HELPERS),
   };
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      for (uint32_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
         /* If the attachment is unused, default to 4 to avoid issues with alpha
          * to coverage with unused attachments. */
         translate_options.rts_component_count[i] = 4;
      }
      for (uint32_t i = 0u; state && i < state->rp->color_attachment_count;
           ++i) {
         uint8_t logical_index = kk_get_logical_color_att_index(state, i);
         if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
            continue;
         }
         enum pipe_format format =
            vk_format_to_pipe_format(state->rp->color_attachment_formats[i]);
         if (format != PIPE_FORMAT_NONE) {
            translate_options.rts_component_count[logical_index] =
               util_format_get_nr_components(format);
         }
      }
   }

   data->code = nir_to_msl(nir, &translate_options);
   const char *entrypoint_name = nir_shader_get_entrypoint(nir)->function->name;

   /* We need to steal so it doesn't get destroyed with the nir. Needs to happen
    * after nir_to_msl since that's where we rename the entrypoint.
    */
   ralloc_steal(NULL, (void *)entrypoint_name);
   data->entrypoint_name = (char *)entrypoint_name;

   if (KK_DEBUG(MSL))
      mesa_logi("%s\n", data->code);

   ralloc_free(nir);
}

static bool
kk_state_may_draw_points(const struct vk_graphics_pipeline_state *state)
{
   if (!state)
      return false;

   if (state->rs &&
       !BITSET_TEST(state->dynamic,
                    MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE) &&
       state->rs->rasterizer_discard_enable)
      return false;

   return BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE) ||
          (state->rs && state->rs->polygon_mode == VK_POLYGON_MODE_POINT);
}

static bool
kk_wants_point_variant(const nir_shader *nir,
                       const struct vk_graphics_pipeline_state *state,
                       bool partition)
{
   if (!partition || !kk_state_may_draw_points(state))
      return false;

   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
      if (nir->info.next_stage == MESA_SHADER_TESS_CTRL ||
          nir->info.next_stage == MESA_SHADER_GEOMETRY)
         return false;
      if (!state->ia ||
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY))
         return true;
      return vk_topology_to_mesa(state->ia->primitive_topology) ==
                MESA_PRIM_TRIANGLES ||
             vk_topology_to_mesa(state->ia->primitive_topology) ==
                MESA_PRIM_TRIANGLE_STRIP ||
             vk_topology_to_mesa(state->ia->primitive_topology) ==
                MESA_PRIM_TRIANGLE_FAN;
   case MESA_SHADER_TESS_EVAL:
      return nir->info.next_stage != MESA_SHADER_GEOMETRY &&
             !nir->info.tess.point_mode &&
             nir->info.tess._primitive_mode != TESS_PRIMITIVE_ISOLINES;
   case MESA_SHADER_GEOMETRY:
      return nir->info.gs.output_primitive == MESA_PRIM_TRIANGLE_STRIP &&
             nir->info.gs.active_stream_mask == 0x1;
   default:
      return false;
   }
}

static bool
gather_passthrough_output(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   struct poly_passthrough_gs_key *key = data;

   if (intr->intrinsic != nir_intrinsic_store_output ||
       !nir_src_is_const(intr->src[1]))
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned location = sem.location + nir_src_as_uint(intr->src[1]);
   if (location >= 64)
      return false;

   unsigned components =
      nir_intrinsic_component(intr) + intr->src[0].ssa->num_components;
   key->outputs |= BITFIELD64_BIT(location);
   key->output_types[location] = nir_intrinsic_src_type(intr);
   key->output_components[location] =
      MAX2(key->output_components[location], components);
   return false;
}

static bool
store_point_primitive_id(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_emit_vertex)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_store_output(b, nir_load_primitive_id(b), nir_imm_int(b, 0),
                    .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
                    .io_semantics.num_slots = 1);
   return true;
}

static nir_shader *
kk_point_passthrough_gs(nir_shader *src)
{
   size_t key_size = poly_passthrough_gs_key_size(0);
   struct poly_passthrough_gs_key *key = calloc(1, key_size);
   if (!key)
      return NULL;

   key->prim = MESA_PRIM_TRIANGLES;
   key->clip_distance_array_size = src->info.clip_distance_array_size;
   key->cull_distance_array_size = src->info.cull_distance_array_size;
   nir_shader_intrinsics_pass(src, gather_passthrough_output, nir_metadata_all,
                              key);

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
                                                  &kk_nir_options, "kk_point");
   poly_nir_passthrough_gs(&b, key);
   free(key);

   nir_shader *gs = b.shader;
   gs->info.io_lowered = true;
   if (!(gs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID)) {
      nir_shader_intrinsics_pass(gs, store_point_primitive_id,
                                 nir_metadata_control_flow, NULL);
   }
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   return gs;
}

static uint8_t kk_partition_gs_primitive_id(nir_shader *gs);

/* Consumes nir */
static nir_shader *
kk_compile_point_emu(struct kk_physical_device *pdev, struct kk_shader *shader,
                     nir_shader *nir,
                     const struct vk_pipeline_robustness_state *robustness,
                     const struct vk_graphics_pipeline_state *state)
{
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      kk_lower_vs_vbo(nir, state, robustness);
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
               NULL);
   } else {
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_in, NULL, NULL);
      NIR_PASS(_, nir, poly_nir_lower_tes, false);
   }
   NIR_PASS(_, nir, nir_opt_constant_folding);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   nir_shader *gs = kk_point_passthrough_gs(nir);
   if (!gs) {
      ralloc_free(nir);
      return NULL;
   }

   shader->point.emu_outputs = nir->info.outputs_written;
   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
   if (nir->info.stage == MESA_SHADER_VERTEX)
      NIR_PASS(_, nir, kk_nir_lower_vertex_id_zero_base);
   bool vs = nir->info.stage == MESA_SHADER_VERTEX;
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;
   if (vs)
      NIR_PASS(_, nir, poly_nir_lower_sw_vs);

   kk_compile_msl(pdev, nir, NULL, true, &shader->point_emu, NULL);
   shader->point.has_emu = true;
   return gs;
}

/* Consumes nir */
static void
kk_compile_point_variant(struct kk_device *dev, struct kk_shader *shader,
                         nir_shader *nir,
                         const struct vk_pipeline_robustness_state *robustness,
                         const struct vk_graphics_pipeline_state *state)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);
   bool from_gs = nir->info.stage == MESA_SHADER_GEOMETRY;

   nir_shader *gs = from_gs ? nir
                            : kk_compile_point_emu(pdev, shader, nir,
                                                   robustness, state);
   if (!gs || !kk_nir_lower_gs_polygon_point(gs, !from_gs)) {
      ralloc_free(gs);
      return;
   }

   NIR_PASS(_, gs, kk_nir_lower_descriptors, robustness, 0, NULL);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   shader->point.gs_needs_main = gs->info.writes_memory || gs->xfb_info;
   if (!from_gs)
      shader->point.rast_primitive_id_slot = kk_partition_gs_primitive_id(gs);
   shader->point.rast_num_cull_distances = gs->info.cull_distance_array_size;

   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   NIR_PASS(_, gs, poly_nir_lower_gs, &count, &rast, &pre_gs,
            &shader->point.gs, true);
   ralloc_free(count);
   ralloc_free(pre_gs);

   NIR_PASS(_, rast, nir_lower_global_vars_to_local);
   kk_lower_hw_vs(rast, state, false, from_gs);
   msl_nir_lower_clip_cull_distance(rast, 0);
   NIR_PASS(_, rast, kk_nir_lower_descriptors, robustness, 0, NULL);
   kk_compile_msl(pdev, rast, state, true, &shader->point_rast,
                  &shader->point.rast_outputs_written);

   gs->info.stage = MESA_SHADER_COMPUTE;
   memset(&gs->info.cs, 0, sizeof(gs->info.cs));
   gs->xfb_info = NULL;
   kk_compile_msl(pdev, gs, state, true, &shader->point_gs_main, NULL);

   shader->info.point_variant = true;
}

static VkResult
kk_upload_gs_indices(struct kk_device *dev, struct kk_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_GEOMETRY ||
       shader->info.gs.shape != POLY_GS_SHAPE_STATIC_INDEXED ||
       shader->gs_indices)
      return VK_SUCCESS;

   uint32_t count = shader->info.gs.max_indices;
   if (count == 0 || count > ARRAY_SIZE(shader->info.gs.topology))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   VkResult result = kk_alloc_bo(dev, &shader->vk.base,
                                 count * sizeof(uint16_t), 4,
                                 &shader->gs_indices);
   if (result != VK_SUCCESS)
      return result;

   uint16_t *indices = shader->gs_indices->cpu;
   for (uint32_t i = 0; i < count; ++i) {
      uint8_t index = shader->info.gs.topology[i];
      indices[i] = index == UINT8_MAX ? UINT16_MAX : index;
   }

   return VK_SUCCESS;
}

static VkResult
kk_compile_shader(struct kk_device *dev, nir_shader *nir,
                  struct kk_shader *prev_stage, bool partition,
                  const struct vk_pipeline_robustness_state *robustness,
                  const struct vk_graphics_pipeline_state *state,
                  const VkAllocationCallbacks *pAllocator,
                  struct kk_shader **shader_out)
{
   assert(nir->info.io_lowered && "nir must have lowered io");
   struct kk_physical_device *pdev = kk_device_physical(dev);

   struct kk_shader *shader;
   VkResult result = VK_SUCCESS;

   mesa_shader_stage stage = nir->info.stage;
   shader = vk_shader_zalloc(&dev->vk, &kk_shader_ops, stage, pAllocator,
                             sizeof(*shader));
   if (shader == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   mtx_init(&shader->variants_lock, mtx_plain);
   shader->info.partial = partition;
   gather_shader_info(shader, nir, state);

   nir_shader *point_nir = NULL;
   if (kk_wants_point_variant(nir, state, partition))
      point_nir = nir_shader_clone(NULL, nir);
   shader->info.rast_num_cull_distances = shader->info.num_cull_distances;

   if (partition && stage == MESA_SHADER_FRAGMENT) {
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      if (blob.out_of_memory) {
         blob_finish(&blob);
         ralloc_free(nir);
         mtx_destroy(&shader->variants_lock);
         vk_shader_free(&dev->vk, pAllocator, &shader->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      blob_finish_get_buffer(&blob, &shader->nir_blob, &shader->nir_blob_size);
   }

   unsigned num_cull_distances =
      prev_stage ? prev_stage->info.num_cull_distances : 0;
   /* A GS lowers clip and cull distances on its rasterization VS instead; the
    * stage feeding it has none after linking */
   if (stage != MESA_SHADER_GEOMETRY)
      msl_nir_lower_clip_cull_distance(nir, num_cull_distances);

   /* When using poly to emulate tessellation or GS, the VS and TCS are turned
    * to compute shaders that will be dispatched before the draw call. The TES
    * becomes a VS, or a compute shader when a GS follows. A GS becomes a
    * compute pass plus a VS that rasterizes its output. */
   if (stage == MESA_SHADER_VERTEX) {
      /* VBO lowering needs to go here otherwise, the linking step removes all
       * inputs since we read vertex attributes from UBOs. */
      kk_lower_vs_vbo(nir, state, robustness);
      if (nir->info.next_stage == MESA_SHADER_TESS_CTRL ||
          nir->info.next_stage == MESA_SHADER_GEOMETRY) {
         /* TODO_KOSMICKRISP: Poly lowering breaks with vectorized I/O less than
          * 32-bit, which we support. Lower vertex outputs to scalar. */
         NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
                  NULL);
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         NIR_PASS(_, nir, kk_nir_lower_vertex_id_zero_base);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
         NIR_PASS(_, nir, poly_nir_lower_sw_vs);
      } else
         /* Metal's instance_id contains base_instance. When the emulation path
          * is taken, since we launch compute, they correctly get translated.
          * For the non-emulated path we need to subtract base_instance... */
         NIR_PASS(_, nir, msl_nir_lower_instance_id);
   } else if (stage == MESA_SHADER_TESS_CTRL) {
      /* TODO_KOSMICKRISP: Poly lowering breaks with vectorized I/O less than
       * 32-bit, which we support. Lower TCS inputs & outputs to scalar. */
      NIR_PASS(_, nir, nir_lower_io_to_scalar,
               nir_var_shader_in | nir_var_shader_out, NULL, NULL);

      NIR_PASS(_, nir, poly_nir_lower_tcs, false);

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      shader->info.tess.tcs_output_patch_size = nir->info.tess.tcs_vertices_out;
      shader->info.tess.tcs_per_vertex_outputs =
         poly_tcs_per_vertex_outputs(nir);
      shader->info.tess.tcs_nr_patch_outputs =
         util_last_bit(nir->info.patch_outputs_written);
      shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
   } else if (stage == MESA_SHADER_TESS_EVAL) {
      /* TODO_KOSMICKRISP: Poly lowering breaks with vectorized I/O less than
       * 32-bit, which we support. Lower TES inputs to scalar. */
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_in, NULL, NULL);

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      bool to_hw_vs = nir->info.next_stage != MESA_SHADER_GEOMETRY;
      /* This destroys info so it needs to happen after the gather */
      NIR_PASS(_, nir, poly_nir_lower_tes, to_hw_vs);
      if (!to_hw_vs) {
         shader->info.tess.tes_outputs = nir->info.outputs_written;
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
      }
   } else if (stage == MESA_SHADER_GEOMETRY) {
      nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      shader->info.gs_needs_main = nir->info.writes_memory || nir->xfb_info;
      /* Keeps work that only feeds side effects out of the rasterization VS,
       * where the GPU watchdog resets long vertex work */
      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre_gs,
               &shader->info.gs, true);

      /* Only transform feedback and pipeline statistics need these, and KK
       * exposes neither */
      ralloc_free(count);
      ralloc_free(pre_gs);

      /* poly selects the rasterized vertex through shader_temp variables,
       * localize them so msl_optimize_nir turns them into SSA */
      NIR_PASS(_, rast, nir_lower_global_vars_to_local);

      kk_lower_hw_vs(rast, state, shader->info.gs.mode == MESA_PRIM_POINTS,
                     true);
      msl_nir_lower_clip_cull_distance(rast, 0);
      /* Descriptors were lowered with the GS, this lowers the system values
       * the hardware vertex lowering added */
      NIR_PASS(_, rast, kk_nir_lower_descriptors, robustness, 0, NULL);
      kk_compile_msl(pdev, rast, state, partition,
                     &shader->msl_data[MESA_SHADER_GEOMETRY],
                     &shader->info.rast_outputs_written);

      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->xfb_info = NULL;
   }

   /* The geometry stage's own slot holds its rasterization VS */
   bool rast = nir->info.stage == MESA_SHADER_VERTEX;
   if (rast && stage == MESA_SHADER_VERTEX &&
       kk_topology_is_unknown(state, partition)) {
      shader->info.rast_point_variant = true;
      nir_shader *points = nir_shader_clone(NULL, nir);
      nir_shader_intrinsics_pass(nir, msl_nir_vs_remove_point_size_write,
                                 nir_metadata_control_flow, NULL);
      kk_compile_msl(pdev, points, state, partition, &shader->msl_points,
                     NULL);
   }

   kk_compile_msl(pdev, nir,
                  partition && stage == MESA_SHADER_FRAGMENT ? NULL : state,
                  partition,
                  stage == MESA_SHADER_GEOMETRY ? &shader->gs_main
                                                : &shader->msl_data[stage],
                  rast ? &shader->info.rast_outputs_written : NULL);

   if (point_nir)
      kk_compile_point_variant(dev, shader, point_nir, robustness, state);

   result = kk_upload_gs_indices(dev, shader);
   if (result != VK_SUCCESS) {
      kk_shader_destroy(&dev->vk, shader, pAllocator);
      return result;
   }

   *shader_out = shader;

   return result;
}

static const struct vk_pipeline_robustness_state rs_none = {
   .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
   .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
   .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,
};

VkResult
kk_compile_nir_shader(struct kk_device *dev, nir_shader *nir,
                      const VkAllocationCallbacks *alloc,
                      struct kk_shader **shader_out)
{
   const struct kk_physical_device *pdev = kk_device_physical(dev);

   assert(nir->info.stage == MESA_SHADER_COMPUTE);
   if (nir->options == NULL)
      nir->options = kk_get_nir_options((struct vk_physical_device *)&pdev->vk,
                                        nir->info.stage, &rs_none);

   struct kk_shader *shader = NULL;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   VkResult result =
      kk_compile_shader(dev, nir, NULL, false, &rs_none, NULL, alloc, &shader);
   if (result != VK_SUCCESS)
      return result;

   *shader_out = shader;

   return VK_SUCCESS;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_licm, NULL);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

static nir_shader *
get_empty_nir(struct kk_device *dev, mesa_shader_stage stage,
              const struct vk_graphics_pipeline_state *state,
              enum kk_feature_key features, bool partition)
{
   nir_shader *nir = nir_shader_create(
      NULL, stage,
      kk_get_nir_options(&kk_device_physical(dev)->vk, stage, NULL));

   nir_function *function = nir_function_create(nir, "main_entrypoint");
   function->is_entrypoint = true;
   nir_function_impl_create(function);

   const struct vk_pipeline_robustness_state no_robustness = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
      .null_uniform_buffer_descriptor = false,
      .null_storage_buffer_descriptor = false,
   };
   kk_lower_nir(dev, nir, false, partition, &no_robustness, 0u, NULL, state,
                features);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   return nir;
}

static VkResult
kk_compile_compute_pipeline(struct kk_device *device,
                            const struct msl_compile_data *data,
                            uint32_t local_size_threads,
                            mtl_compute_pipeline_state **pipe)
{
   struct kk_physical_device *pdev = kk_device_physical(device);

   mtl_library *library = mtl_new_library(
      device->mtl_compiler_handle, data->code, pdev->info.msl_version,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   if (library == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   mtl_function_descriptor *function =
      mtl_new_library_function_descriptor(library, data->entrypoint_name);
   VkResult link_result = kk_msl_helpers_link(device, function, data);
   if (link_result != VK_SUCCESS) {
      mtl_release(function);
      mtl_release(library);
      return link_result;
   }
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   const uint32_t kind = MESA_SHADER_COMPUTE;
   _mesa_blake3_update(&hash, &kind, sizeof(kind));
   _mesa_blake3_update(&hash, data->code, strlen(data->code) + 1);
   _mesa_blake3_update(&hash, data->entrypoint_name,
                       strlen(data->entrypoint_name) + 1);
   _mesa_blake3_update(&hash, &local_size_threads, sizeof(local_size_threads));
   blake3_hash digest;
   _mesa_blake3_final(&hash, digest);
   bool archive_hit = false;
   int64_t start = os_time_get_nano();
   struct kk_metal_trace previous_trace = {0};
   if (device->variant_trace_enabled) {
      struct kk_metal_trace trace = {kk_variant_metal_event, device};
      previous_trace = kk_metal_cache_set_trace(trace);
   }
   mtl_compute_pipeline_state *pipeline = device->metal_cache ?
      kk_metal_cache_new_compute_pipeline(device->metal_cache, digest, function,
                                           local_size_threads, &archive_hit) :
      mtl_new_compute_pipeline_state(device->mtl_compiler_handle, function,
                                      local_size_threads);
   if (device->variant_trace_enabled) {
      kk_metal_cache_set_trace(previous_trace);
      if (!device->metal_cache)
         kk_variant_trace(device, KK_VARIANT_NATIVE_CREATE, digest,
                          sizeof(digest), start, os_time_get_nano());
   }
   kk_log_compile(device, archive_hit ? "compute_archive" : "compute", start);
   kk_log_cache_pso(device,
                    archive_hit ? "compute_archive_hit" : "compute_fallback",
                     digest, sizeof(digest));
   mtl_release(function);
   mtl_release(library);

   if (pipeline == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   *pipe = pipeline;

   return VK_SUCCESS;
}

static mtl_depth_stencil_state *
kk_compile_ds_state(struct kk_device *device, struct kk_shader_info *info)
{
   mtl_stencil_descriptor *front = NULL;
   mtl_stencil_descriptor *back = NULL;
   mtl_depth_stencil_descriptor *descriptor =
      mtl_new_depth_stencil_descriptor();
   mtl_depth_stencil_descriptor_set_depth_write_enabled(
      descriptor, info->vs.has_depth_write);
   mtl_depth_stencil_descriptor_set_depth_compare_function(
      descriptor, info->vs.depth_compare_op);

   if (info->vs.has_stencil_test) {
      back = mtl_new_stencil_descriptor();
      mtl_stencil_descriptor_set_depth_failure_operation(
         back, info->vs.stencil_back.depth_fail);
      mtl_stencil_descriptor_set_stencil_failure_operation(
         back, info->vs.stencil_back.fail);
      mtl_stencil_descriptor_set_depth_stencil_pass_operation(
         back, info->vs.stencil_back.pass);
      mtl_stencil_descriptor_set_stencil_compare_function(
         back, info->vs.stencil_back.compare);
      mtl_stencil_descriptor_set_read_mask(back,
                                           info->vs.stencil_back.compare_mask);
      mtl_stencil_descriptor_set_write_mask(back,
                                            info->vs.stencil_back.write_mask);
      mtl_depth_stencil_descriptor_set_back_face_stencil(descriptor, back);

      front = mtl_new_stencil_descriptor();
      mtl_stencil_descriptor_set_depth_failure_operation(
         front, info->vs.stencil_front.depth_fail);
      mtl_stencil_descriptor_set_stencil_failure_operation(
         front, info->vs.stencil_front.fail);
      mtl_stencil_descriptor_set_depth_stencil_pass_operation(
         front, info->vs.stencil_front.pass);
      mtl_stencil_descriptor_set_stencil_compare_function(
         front, info->vs.stencil_front.compare);
      mtl_stencil_descriptor_set_read_mask(front,
                                           info->vs.stencil_front.compare_mask);
      mtl_stencil_descriptor_set_write_mask(front,
                                            info->vs.stencil_front.write_mask);
      mtl_depth_stencil_descriptor_set_front_face_stencil(descriptor, front);
   }

   mtl_depth_stencil_state *state =
      mtl_new_depth_stencil_state(device->mtl_handle, descriptor);

   if (front)
      mtl_release(front);
   if (back)
      mtl_release(back);
   mtl_release(descriptor);

   return state;
}

static void
kk_gather_ds_info(const struct vk_depth_stencil_state *ds, bool has_depth,
                  bool has_stencil, struct kk_shader_info *info)
{
   /* Depth */
   if (has_depth && ds->depth.test_enable) {
      info->vs.has_depth_write = ds->depth.write_enable;
      info->vs.depth_compare_op = ds->depth.compare_op;
   } else {
      info->vs.has_depth_write = false;
      info->vs.depth_compare_op = VK_COMPARE_OP_ALWAYS;
   }

   /* Stencil */
   info->vs.has_stencil_test = has_stencil && ds->stencil.test_enable;
   if (info->vs.has_stencil_test) {
      info->vs.stencil_back.depth_fail = ds->stencil.back.op.depth_fail;
      info->vs.stencil_back.fail = ds->stencil.back.op.fail;
      info->vs.stencil_back.pass = ds->stencil.back.op.pass;
      info->vs.stencil_back.compare = ds->stencil.back.op.compare;
      info->vs.stencil_back.compare_mask = ds->stencil.back.compare_mask;
      info->vs.stencil_back.write_mask = ds->stencil.back.write_mask;

      info->vs.stencil_front.depth_fail = ds->stencil.front.op.depth_fail;
      info->vs.stencil_front.fail = ds->stencil.front.op.fail;
      info->vs.stencil_front.pass = ds->stencil.front.op.pass;
      info->vs.stencil_front.compare = ds->stencil.front.op.compare;
      info->vs.stencil_front.compare_mask = ds->stencil.front.compare_mask;
      info->vs.stencil_front.write_mask = ds->stencil.front.write_mask;
   }
}

mtl_depth_stencil_state *
kk_compile_depth_stencil_state(struct kk_device *device,
                               const struct vk_depth_stencil_state *ds,
                               bool has_depth, bool has_stencil)
{
   struct kk_shader_info info = {};
   kk_gather_ds_info(ds, has_depth, has_stencil, &info);
   return kk_compile_ds_state(device, &info);
}

static void
kk_msl_copy(struct msl_compile_data *dst, const struct msl_compile_data *src)
{
   dst->code = ralloc_strdup(NULL, src->code);
   dst->entrypoint_name = ralloc_strdup(NULL, src->entrypoint_name);
}

/* Copies all msl shader data to the vertex and gathers information for pipeline
 * compilation. */
static void
gather_graphics_pipeline_create_info(
   const struct vk_graphics_pipeline_state *state, struct kk_shader **shaders,
   uint32_t shader_count)
{
   assert(state && shader_count >= 2 &&
          "state, vertex and fragment shaders are a must");
   struct kk_shader *vs = shaders[0];
   struct kk_shader_info *info = &vs->info;

   /* Render pass data */
   const struct vk_render_pass_state *rp = state->rp;
   bool has_depth = rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   bool has_stencil = rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
   {
      info->vs.color_attachment_count = rp->color_attachment_count;
      for (uint8_t i = 0u; i < MAX_DRAW_BUFFERS; ++i) {
         info->vs.rt_formats[i] = MTL_PIXEL_FORMAT_INVALID;
      }
      for (uint8_t i = 0u; i < state->rp->color_attachment_count; ++i) {
         uint8_t logical_index = kk_get_logical_color_att_index(state, i);
         if (logical_index == MESA_VK_ATTACHMENT_UNUSED) {
            continue;
         }
         VkFormat format = state->rp->color_attachment_formats[i];
         info->vs.rt_formats[logical_index] =
            format == VK_FORMAT_UNDEFINED
               ? MTL_PIXEL_FORMAT_INVALID
               : vk_format_to_mtl_pixel_format(format);
      }
      info->vs.d_format =
         has_depth ? vk_format_to_mtl_pixel_format(rp->depth_attachment_format)
                   : MTL_PIXEL_FORMAT_INVALID;
      info->vs.s_format =
         has_stencil
            ? vk_format_to_mtl_pixel_format(rp->stencil_attachment_format)
            : MTL_PIXEL_FORMAT_INVALID;

      info->vs.view_mask = state->mv->view_mask;
   }

   info->vs.has_ds = has_static_depth_stencil_state(state);
   if (info->vs.has_ds)
      kk_gather_ds_info(state->ds, has_depth, has_stencil, info);

   info->vs.has_ms = state->ms != NULL;
   info->vs.sample_count = 1u;
   if (info->vs.has_ms) {
      const struct vk_multisample_state *ms = state->ms;
      info->vs.sample_count = ms->rasterization_samples;
   }

   /* We need to store all other stage sources in the vertex too otherwise we
    * won't be able to create the whole pipeline correctly. */
   struct kk_shader *gs = NULL;
   for (uint32_t i = 1; i < shader_count; ++i) {
      struct kk_shader *s = shaders[i];
      mesa_shader_stage stage = s->info.stage;
      kk_msl_copy(&vs->msl_data[stage], &s->msl_data[stage]);

      info->vs.additional_stages_bits |= BITFIELD_BIT(stage);

      if (s->info.stage == MESA_SHADER_TESS_CTRL)
         info->vs.tess_local_thread_size = s->info.tess.tcs_output_patch_size;

      if (s->info.stage == MESA_SHADER_GEOMETRY) {
         kk_msl_copy(&vs->gs_main, &s->gs_main);
         gs = s;
      }
   }

   uint8_t topology = state->ia->primitive_topology;
   if (info->vs.additional_stages_bits & BITFIELD_BIT(MESA_SHADER_TESS_CTRL)) {
      /* TODO_KOSMICKRISP Hackity hack */
      struct kk_shader *tesc = shaders[1];
      struct kk_shader *tese = shaders[2];

      struct kk_tess_info info =
         kk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (info.points) {
         topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      } else if (info.mode == TESS_PRIMITIVE_ISOLINES) {
         topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      } else {
         topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      }
   }

   if (gs) {
      /* Only the topology class matters here */
      enum mesa_prim prim = u_reduced_prim(gs->info.gs.mode);
      if (prim == MESA_PRIM_POINTS)
         topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      else if (prim == MESA_PRIM_LINES)
         topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      else
         topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   }

   info->vs.topology =
      vk_primitive_topology_to_mtl_primitive_topology_class(topology);
}

static VkResult
kk_new_function(struct kk_device *device, const struct msl_compile_data *data,
                mtl_library **lib_out, mtl_function_descriptor **fn_out)
{
   struct kk_physical_device *pdev = kk_device_physical(device);

   mtl_library *library = mtl_new_library(
      device->mtl_compiler_handle, data->code, pdev->info.msl_version,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   if (library == NULL)
      return VK_ERROR_INVALID_SHADER_NV;

   mtl_function_descriptor *function =
      mtl_new_library_function_descriptor(library, data->entrypoint_name);
   VkResult result = kk_msl_helpers_link(device, function, data);
   if (result != VK_SUCCESS) {
      mtl_release(function);
      mtl_release(library);
      return result;
   }
   *lib_out = library;
   *fn_out = function;
   return VK_SUCCESS;
}

static VkResult
kk_build_prerender(struct kk_device *device, struct kk_shader *vs,
                   const struct msl_compile_data **rast_out)
{
   VkResult result = VK_SUCCESS;
   struct kk_pipeline_handles *pipe = &vs->pipeline;
   mesa_shader_stage vs_stage = MESA_SHADER_VERTEX;

   /* When tessellation or geometry shaders are present, the pair that will be
    * used for the Metal render pipeline will be the fragment shader and the
    * previous stage. This means that any shader before that needs to be a
    * compute pipeline to be launched before the render pipeline. */
   uint32_t stages_bits = vs->info.vs.additional_stages_bits &
                          ~BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   pipe->gfx.pre_render_count = util_bitcount(stages_bits);
   for (uint32_t i = 0u; i < pipe->gfx.pre_render_count; ++i) {
      uint32_t local_thread_size = vs_stage == MESA_SHADER_TESS_CTRL
                                      ? vs->info.vs.tess_local_thread_size
                                      : KK_POLY_WG_SIZE;
      result = kk_compile_compute_pipeline(device, &vs->msl_data[vs_stage],
                                           local_thread_size,
                                           &pipe->gfx.pre_render[i]);

      if (result != VK_SUCCESS)
         return VK_ERROR_INVALID_SHADER_NV;
      vs_stage = u_bit_scan(&stages_bits);
   }

   /* The GS compute pass is the last pre-render pipeline, its rasterization VS
    * pairs with the FS. */
   if (vs->info.vs.additional_stages_bits &
       BITFIELD_BIT(MESA_SHADER_GEOMETRY)) {
      result = kk_compile_compute_pipeline(
         device, &vs->gs_main, KK_POLY_WG_SIZE,
         &pipe->gfx.pre_render[pipe->gfx.pre_render_count++]);
      if (result != VK_SUCCESS)
         return VK_ERROR_INVALID_SHADER_NV;
   }

   *rast_out = &vs->msl_data[vs_stage];
   return VK_SUCCESS;
}

static bool
kk_msl_native_primitive_id(const struct msl_compile_data *data)
{
   return strstr(data->code, "[[primitive_id]]") != NULL;
}

static void
kk_hash_msl_function(const struct msl_compile_data *data, blake3_hash digest)
{
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   _mesa_blake3_update(&hash, data->code, strlen(data->code) + 1);
   _mesa_blake3_update(&hash, data->entrypoint_name,
                        strlen(data->entrypoint_name) + 1);
   _mesa_blake3_final(&hash, digest);
}

static VkResult
kk_compile_graphics_pipeline(struct kk_device *device, struct kk_shader *vs)
{
   struct kk_pipeline_handles *pipe = &vs->pipeline;

   const struct msl_compile_data *vs_data;
   VkResult result = kk_build_prerender(device, vs, &vs_data);
   if (result != VK_SUCCESS)
      return result;

   mtl_library *vertex_library, *fragment_library;
   mtl_function_descriptor *vertex_function, *fragment_function;
   result =
      kk_new_function(device, vs_data, &vertex_library, &vertex_function);
   if (result != VK_SUCCESS)
      return result;

   result = kk_new_function(device, &vs->msl_data[MESA_SHADER_FRAGMENT],
                            &fragment_library, &fragment_function);
   if (result != VK_SUCCESS)
      goto destroy_vertex;

   struct kk_pso_key key;
   kk_pso_key_init(&key);
   kk_hash_msl_function(vs_data, key.rast);
   kk_hash_msl_function(&vs->msl_data[MESA_SHADER_FRAGMENT], key.fs);
   key.topology_class = vs->info.vs.topology;

   /* If color attachments are reordered there might be unused / invalid gaps
    * in the array so need to check them all, not just first
    * color_attachment_count entries.
    */
   for (uint8_t i = 0; i < MAX_DRAW_BUFFERS; ++i)
      key.rt_formats[i] = vs->info.vs.rt_formats[i];

   if (vs->info.vs.has_ds) {
      pipe->gfx.ds_handle = kk_compile_ds_state(device, &vs->info);
   }

   if (vs->info.vs.view_mask)
      key.amplification = util_bitcount(vs->info.vs.view_mask);

   if (vs->info.vs.has_ms)
      key.sample_count = vs->info.vs.sample_count;

   struct kk_pso_request request = {
      .lookup_only = kk_fail_on_compile,
      .native_primitive_id =
         kk_msl_native_primitive_id(&vs->msl_data[MESA_SHADER_FRAGMENT]),
   };
   pipe->gfx.render = kk_pso_cache_acquire(device, &key, vertex_function,
                                           fragment_function, &request);
   if (pipe->gfx.render == NULL && kk_fail_on_compile) {
      struct kk_render_pending *pending = calloc(1, sizeof(*pending));
      if (pending && cnd_init(&pending->ready) == thrd_success) {
         pending->compiling = true;
         pending->native_primitive_id = request.native_primitive_id;
         pending->key = key;
         pending->vertex_library = vertex_library;
         pending->vertex_function = vertex_function;
         pending->fragment_library = fragment_library;
         pending->fragment_function = fragment_function;
         vs->pending = pending;
         return VK_SUCCESS;
      }
      free(pending);
      request.lookup_only = false;
      pipe->gfx.render = kk_pso_cache_acquire(device, &key, vertex_function,
                                              fragment_function, &request);
   }
   if (pipe->gfx.render == NULL)
      result = VK_ERROR_INVALID_SHADER_NV;

   mtl_release(fragment_function);
   mtl_release(fragment_library);
destroy_vertex:
   mtl_release(vertex_function);
   mtl_release(vertex_library);

   return result;
}

static VkResult
kk_init_fast_prerast(struct kk_device *device, struct kk_shader *vs)
{
   const struct msl_compile_data *rast;
   VkResult result = kk_build_prerender(device, vs, &rast);
   if (result != VK_SUCCESS)
      return result;

   result = kk_new_function(device, rast, &vs->fast.lib, &vs->fast.fn);
   if (result != VK_SUCCESS)
      return result;

   _mesa_blake3_compute(rast->code, strlen(rast->code), vs->fast.hash);
   return VK_SUCCESS;
}

static void
kk_identity_scalar(struct mesa_blake3 *hash, uint64_t value)
{
   uint8_t bytes[8];
   for (unsigned i = 0; i < sizeof(bytes); i++)
      bytes[i] = value >> (8 * i);
   _mesa_blake3_update(hash, bytes, sizeof(bytes));
}

static void
kk_identity_string(struct mesa_blake3 *hash, const char *string)
{
   size_t size = string ? strlen(string) + 1 : 0;
   kk_identity_scalar(hash, size);
   if (size)
      _mesa_blake3_update(hash, string, size);
}

static void
kk_identity_msl(struct mesa_blake3 *hash, const struct msl_compile_data *data)
{
   kk_identity_string(hash, data->code);
   kk_identity_string(hash, data->entrypoint_name);
}

void
kk_shader_variant_identity(struct kk_shader *shader, blake3_hash output)
{
   if (!p_atomic_read(&shader->variant_identity_ready)) {
      mtx_lock(&shader->variants_lock);
      if (!p_atomic_read(&shader->variant_identity_ready)) {
         struct mesa_blake3 hash;
         _mesa_blake3_init(&hash);
         kk_identity_scalar(&hash, 1);
         _mesa_blake3_update(&hash, shader->fast.hash, sizeof(blake3_hash));
         kk_identity_scalar(&hash, shader->nir_blob_size);
         if (shader->nir_blob_size)
            _mesa_blake3_update(&hash, shader->nir_blob, shader->nir_blob_size);
         for (unsigned i = 0; i < MESA_SHADER_STAGES; i++)
            kk_identity_msl(&hash, &shader->msl_data[i]);
         kk_identity_msl(&hash, &shader->gs_main);
         kk_identity_msl(&hash, &shader->msl_points);
         kk_identity_msl(&hash, &shader->msl_emu);
         kk_identity_msl(&hash, &shader->point_emu);
         kk_identity_msl(&hash, &shader->point_gs_main);
         kk_identity_msl(&hash, &shader->point_rast);
         const struct kk_shader_info *info = &shader->info;
         kk_identity_scalar(&hash, info->stage);
         kk_identity_scalar(&hash, info->uses_per_draw_data);
         kk_identity_scalar(&hash, info->partial);
         kk_identity_scalar(&hash, info->shader_object);
         kk_identity_scalar(&hash, info->so_hw);
         kk_identity_scalar(&hash, info->so_emu);
         kk_identity_scalar(&hash, info->num_cull_distances);
         kk_identity_scalar(&hash, info->rast_num_cull_distances);
         kk_identity_scalar(&hash, info->rast_primitive_id_slot);
         kk_identity_scalar(&hash, info->rast_point_variant);
         kk_identity_scalar(&hash, info->point_variant);
         kk_identity_scalar(&hash, info->rast_outputs_written);
         switch (info->stage) {
         case MESA_SHADER_FRAGMENT:
            kk_identity_scalar(&hash, info->fs.uses_flat_varyings);
            kk_identity_scalar(&hash, info->fs.ms_static);
            kk_identity_scalar(&hash, info->fs.writes_depth);
            kk_identity_scalar(&hash, info->fs.writes_stencil);
            kk_identity_scalar(&hash, info->fs.dual_source_blend);
            kk_identity_scalar(&hash, info->fs.reads_primitive_id);
            kk_identity_scalar(&hash, info->fs.color_outputs);
            kk_identity_scalar(&hash, info->fs.inputs_read);
            kk_identity_scalar(&hash, info->fs.flat_inputs);
            for (unsigned i = 0; i < MAX_DRAW_BUFFERS; i++)
               kk_identity_scalar(&hash, info->fs.output_base_type[i]);
            break;
         case MESA_SHADER_VERTEX:
            kk_identity_scalar(&hash, info->vs.additional_stages_bits);
            kk_identity_scalar(&hash, info->vs.tess_local_thread_size);
            kk_identity_scalar(&hash, info->vs.attribs_read);
            kk_identity_scalar(&hash, info->vs.dynamic_vi);
            kk_identity_scalar(&hash, info->vs.sample_count);
            kk_identity_scalar(&hash, info->vs.outputs_written);
            kk_identity_scalar(&hash, info->vs.topology);
            kk_identity_scalar(&hash, info->vs.view_mask);
            for (unsigned i = 0; i < MAX_DRAW_BUFFERS; i++)
               kk_identity_scalar(&hash, info->vs.rt_formats[i]);
            kk_identity_scalar(&hash, info->vs.d_format);
            kk_identity_scalar(&hash, info->vs.s_format);
            kk_identity_scalar(&hash, info->vs.depth_compare_op);
            kk_identity_scalar(&hash, info->vs.stencil_front.depth_fail);
            kk_identity_scalar(&hash, info->vs.stencil_front.fail);
            kk_identity_scalar(&hash, info->vs.stencil_front.pass);
            kk_identity_scalar(&hash, info->vs.stencil_front.compare);
            kk_identity_scalar(&hash, info->vs.stencil_front.compare_mask);
            kk_identity_scalar(&hash, info->vs.stencil_front.write_mask);
            kk_identity_scalar(&hash, info->vs.stencil_back.depth_fail);
            kk_identity_scalar(&hash, info->vs.stencil_back.fail);
            kk_identity_scalar(&hash, info->vs.stencil_back.pass);
            kk_identity_scalar(&hash, info->vs.stencil_back.compare);
            kk_identity_scalar(&hash, info->vs.stencil_back.compare_mask);
            kk_identity_scalar(&hash, info->vs.stencil_back.write_mask);
            kk_identity_scalar(&hash, info->vs.color_attachment_count);
            kk_identity_scalar(&hash, info->vs.has_ms);
            kk_identity_scalar(&hash, info->vs.has_ds);
            kk_identity_scalar(&hash, info->vs.has_depth_write);
            kk_identity_scalar(&hash, info->vs.has_stencil_test);
            break;
         case MESA_SHADER_TESS_CTRL:
         case MESA_SHADER_TESS_EVAL:
            kk_identity_scalar(&hash, info->tess.tcs_per_vertex_outputs);
            kk_identity_scalar(&hash, info->tess.tcs_output_stride);
            kk_identity_scalar(&hash, info->tess.tcs_output_patch_size);
            kk_identity_scalar(&hash, info->tess.tcs_nr_patch_outputs);
            kk_identity_scalar(&hash, info->tess.info.mode);
            kk_identity_scalar(&hash, info->tess.info.spacing);
            kk_identity_scalar(&hash, info->tess.info.points);
            kk_identity_scalar(&hash, info->tess.info.ccw);
            kk_identity_scalar(&hash, info->tess.tes_outputs);
            break;
         case MESA_SHADER_GEOMETRY:
            kk_identity_scalar(&hash, info->gs.mode);
            kk_identity_scalar(&hash, info->gs.count_words);
            kk_identity_scalar(&hash, info->gs.max_indices);
            kk_identity_scalar(&hash, info->gs.xfb);
            kk_identity_scalar(&hash, info->gs.prefix_sum);
            kk_identity_scalar(&hash, info->gs.multistream);
            kk_identity_scalar(&hash, info->gs.shape);
            kk_identity_scalar(&hash, info->gs_needs_main);
            break;
         case MESA_SHADER_COMPUTE:
            kk_identity_scalar(&hash, info->cs.local_size.x);
            kk_identity_scalar(&hash, info->cs.local_size.y);
            kk_identity_scalar(&hash, info->cs.local_size.z);
            break;
         default:
            break;
         }
         _mesa_blake3_final(&hash, shader->variant_identity);
         p_atomic_set(&shader->variant_identity_ready, 1);
      }
      mtx_unlock(&shader->variants_lock);
   }
   memcpy(output, shader->variant_identity, sizeof(blake3_hash));
}

static void
kk_variant_shader_key(struct kk_device *dev, struct kk_shader *shader,
                      const void *key, size_t size, blake3_hash identity)
{
   if (!dev->variant_trace_enabled)
      return;
   struct mesa_blake3 state;
   _mesa_blake3_init(&state);
   _mesa_blake3_update(&state, shader->fast.hash, sizeof(blake3_hash));
   _mesa_blake3_update(&state, key, size);
   _mesa_blake3_final(&state, identity);
}

const struct kk_fs_variant *
kk_shader_get_rast(struct kk_device *dev, struct kk_shader *vs, bool points)
{
   if (!points || !vs->info.rast_point_variant)
      return &vs->fast;

   const struct kk_fs_variant *rast = &vs->fast_points;
   blake3_hash identity = {0};
   kk_variant_shader_key(dev, vs, &points, sizeof(points), identity);
   if (p_atomic_read(&vs->fast_points.fn)) {
      kk_variant_trace(dev, KK_VARIANT_READY_HIT, identity, sizeof(identity),
                       0, 0);
      return rast;
   }
   if (dev->variant_trace_enabled &&
       mtx_trylock(&vs->variants_lock) != thrd_success) {
      int64_t wait_start = os_time_get_nano();
      mtx_lock(&vs->variants_lock);
      kk_variant_trace(dev, KK_VARIANT_RASTER_WAIT, identity, sizeof(identity),
                       wait_start, os_time_get_nano());
   } else if (!dev->variant_trace_enabled) {
      mtx_lock(&vs->variants_lock);
   }
   if (vs->fast_points.fn == NULL) {
      int64_t start = os_time_get_nano();
      mtl_library *lib = NULL;
      mtl_function_descriptor *fn = NULL;
      VkResult result = kk_new_function(dev, &vs->msl_points, &lib, &fn);
      kk_log_compile(dev, "rast_points", start);
      kk_variant_trace(dev, KK_VARIANT_RASTER_PREPARE, identity,
                       sizeof(identity), start, os_time_get_nano());
      if (result == VK_SUCCESS) {
         _mesa_blake3_compute(vs->msl_points.code,
                              strlen(vs->msl_points.code),
                              vs->fast_points.hash);
         vs->fast_points.lib = lib;
         p_atomic_set(&vs->fast_points.fn, fn);
      } else {
         rast = NULL;
      }
   } else {
      kk_variant_trace(dev, KK_VARIANT_READY_HIT, identity, sizeof(identity),
                       0, 0);
   }
   mtx_unlock(&vs->variants_lock);

   return rast;
}

bool
kk_shader_rast_ready(struct kk_shader *vs, bool points)
{
   if (!points || !vs->info.rast_point_variant)
      return true;
   return p_atomic_read(&vs->fast_points.fn) != NULL;
}

static uint32_t
kk_fs_link_key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct kk_fs_link_key));
}

static bool
kk_fs_link_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct kk_fs_link_key)) == 0;
}

static VkResult
kk_init_fast_fs(struct kk_device *device, struct kk_shader *fs)
{
   const struct msl_compile_data *data = &fs->msl_data[MESA_SHADER_FRAGMENT];
   VkResult result = kk_new_function(device, data, &fs->fast.lib, &fs->fast.fn);
   if (result != VK_SUCCESS)
      return result;

   _mesa_blake3_compute(data->code, strlen(data->code), fs->fast.hash);
   fs->fast.native_primitive_id = kk_msl_native_primitive_id(data);

   fs->variants =
      _mesa_hash_table_create(NULL, kk_fs_link_key_hash, kk_fs_link_key_equal);
   return fs->variants ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
}

static VkResult
kk_init_partition(struct kk_device *device,
                  const struct vk_graphics_pipeline_state *state,
                  struct kk_shader **shaders, uint32_t shader_count)
{
   if (shaders[0]->info.stage == MESA_SHADER_FRAGMENT)
      return kk_init_fast_fs(device, shaders[0]);

   struct kk_shader *vs = shaders[0];
   for (uint32_t i = 1u; i < shader_count; ++i) {
      struct kk_shader *s = shaders[i];
      mesa_shader_stage stage = s->info.stage;
      kk_msl_copy(&vs->msl_data[stage], &s->msl_data[stage]);
      vs->info.vs.additional_stages_bits |= BITFIELD_BIT(stage);

      if (stage == MESA_SHADER_TESS_CTRL)
         vs->info.vs.tess_local_thread_size =
            s->info.tess.tcs_output_patch_size;

      if (stage == MESA_SHADER_GEOMETRY)
         kk_msl_copy(&vs->gs_main, &s->gs_main);
   }

   struct kk_shader *last = shaders[shader_count - 1u];
   vs->info.rast_outputs_written = last->info.rast_outputs_written;
   vs->info.rast_num_cull_distances = last->info.num_cull_distances;
   vs->info.rast_primitive_id_slot = last->info.rast_primitive_id_slot;

   return kk_init_fast_prerast(device, vs);
}

static bool
move_primitive_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned *slot = data;

   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_PRIMITIVE_ID)
      return false;

   sem.location = *slot;
   nir_intrinsic_set_io_semantics(intr, sem);

   /* Metal requires the vertex output and fragment input types to match */
   if (intr->intrinsic == nir_intrinsic_store_output)
      nir_intrinsic_set_src_type(intr, nir_type_uint32);
   else
      nir_intrinsic_set_dest_type(intr, nir_type_uint32);
   return true;
}

/* Metal's primitive_id attribute numbers the primitives the GS rasterization
 * pass draws, not the value the GS writes. Pass that value as a flat varying
 * in a slot neither shader uses instead. */
static void
kk_forward_gs_primitive_id(nir_shader *gs, nir_shader *fs)
{
   uint64_t used = gs->info.outputs_written | fs->info.inputs_read;
   unsigned slot = VARYING_SLOT_VAR31;
   while (slot >= VARYING_SLOT_VAR0 && (used & BITFIELD64_BIT(slot)))
      --slot;

   if (!(gs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) ||
       slot < VARYING_SLOT_VAR0)
      return;

   /* Makes progress only if the FS reads the primitive ID */
   const struct nir_lower_sysvals_to_varyings_options sysvals = {
      .primitive_id = true,
   };
   bool progress = false;
   NIR_PASS(progress, fs, nir_lower_sysvals_to_varyings, &sysvals);
   if (!progress)
      return;

   nir_shader_intrinsics_pass(gs, move_primitive_id, nir_metadata_all, &slot);
   nir_shader_intrinsics_pass(fs, move_primitive_id, nir_metadata_all, &slot);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   nir_shader_gather_info(fs, nir_shader_get_entrypoint(fs));
}

static nir_def *
build_patch_id(nir_builder *b)
{
   return nir_load_per_vertex_input(
      b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, 0),
      .dest_type = nir_type_uint32,
      .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
      .io_semantics.num_slots = 1);
}

static bool
load_patch_id(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_primitive_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, build_patch_id(b));
   return true;
}

static bool
load_forwarded_patch_id(nir_builder *b, nir_intrinsic_instr *intr,
                        UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_primitive_id)
      return false;

   STATIC_ASSERT(VARYING_SLOT_PRIMITIVE_ID < 32);

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *vp = nir_load_vertex_param_buffer_poly(b);
   nir_def *outputs = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, vp, offsetof(struct poly_vertex_params, outputs)),
      .align_mul = 4);
   nir_def *forwarded = nir_test_mask(b, outputs, VARYING_BIT_PRIMITIVE_ID);
   nir_push_if(b, forwarded);
   nir_def *patch_id = build_patch_id(b);
   nir_pop_if(b, NULL);
   nir_def *id = nir_if_phi(b, patch_id, &intr->def);
   nir_foreach_use_safe(src, &intr->def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src) != nir_def_instr(id))
         nir_src_rewrite(src, id);
   }
   return true;
}

static void
kk_store_tes_primitive_id(nir_shader *tes)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(tes);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0),
                    .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
                    .io_semantics.num_slots = 1);
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_shader_gather_info(tes, impl);
}

/* After tessellation, the GS primitive ID is the patch index the tessellation
 * shaders read, but poly numbers the tessellated primitives. Pass the TES
 * primitive ID to the GS as an input. */
static void
kk_forward_tes_primitive_id(nir_shader *tes, nir_shader *gs)
{
   if (!nir_shader_intrinsics_pass(gs, load_patch_id, nir_metadata_control_flow,
                                   NULL))
      return;

   kk_store_tes_primitive_id(tes);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
}

static uint8_t
kk_partition_gs_primitive_id(nir_shader *gs)
{
   uint64_t written = gs->info.outputs_written;
   if (!(written & VARYING_BIT_PRIMITIVE_ID))
      return 0u;

   unsigned slot = VARYING_SLOT_VAR31;
   while (slot >= VARYING_SLOT_VAR0 && (written & BITFIELD64_BIT(slot)))
      --slot;
   if (slot < VARYING_SLOT_VAR0)
      return 0u;

   nir_shader_intrinsics_pass(gs, move_primitive_id, nir_metadata_all, &slot);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   return slot;
}

static VkResult
kk_so_init(struct kk_device *dev, struct kk_shader *shader)
{
   VkResult result = VK_SUCCESS;
   mesa_shader_stage stage = shader->info.stage;
   struct msl_compile_data *data = &shader->msl_data[stage];

   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      if (shader->info.so_emu) {
         result = kk_compile_compute_pipeline(
            dev, shader->info.so_hw ? &shader->msl_emu : data, KK_POLY_WG_SIZE,
            &shader->so_cs);
         if (result != VK_SUCCESS)
            return result;
      }
      if (shader->info.so_hw) {
         result =
            kk_new_function(dev, data, &shader->fast.lib, &shader->fast.fn);
         if (result == VK_SUCCESS)
            _mesa_blake3_compute(data->code, strlen(data->code),
                                 shader->fast.hash);
      }
      return result;
   case MESA_SHADER_TESS_CTRL:
      return kk_compile_compute_pipeline(
         dev, data, shader->info.tess.tcs_output_patch_size, &shader->so_cs);
   case MESA_SHADER_GEOMETRY:
      result = kk_upload_gs_indices(dev, shader);
      if (result != VK_SUCCESS)
         return result;
      result = kk_compile_compute_pipeline(dev, &shader->gs_main,
                                           KK_POLY_WG_SIZE, &shader->so_cs);
      if (result != VK_SUCCESS)
         return result;
      result = kk_new_function(dev, data, &shader->fast.lib, &shader->fast.fn);
      if (result == VK_SUCCESS)
         _mesa_blake3_compute(data->code, strlen(data->code),
                              shader->fast.hash);
      return result;
   case MESA_SHADER_FRAGMENT:
      return kk_init_fast_fs(dev, shader);
   default:
      UNREACHABLE("Unsupported shader object stage");
   }
}

static struct kk_shader *
kk_create_point_gs(struct kk_device *dev, struct kk_shader *owner)
{
   struct kk_shader *gs = vk_shader_zalloc(
      &dev->vk, &kk_shader_ops, MESA_SHADER_GEOMETRY, NULL, sizeof(*gs));
   if (gs == NULL)
      return NULL;

   mtx_init(&gs->variants_lock, mtx_plain);
   gs->info.stage = MESA_SHADER_GEOMETRY;
   gs->info.partial = true;
   gs->info.shader_object = true;
   gs->info.uses_per_draw_data = true;
   gs->info.gs = owner->point.gs;
   gs->info.gs_needs_main = owner->point.gs_needs_main;
   gs->info.num_cull_distances = owner->point.rast_num_cull_distances;
   gs->info.rast_num_cull_distances = owner->point.rast_num_cull_distances;
   gs->info.rast_outputs_written = owner->point.rast_outputs_written;
   gs->info.rast_primitive_id_slot =
      owner->info.stage == MESA_SHADER_GEOMETRY
         ? owner->info.rast_primitive_id_slot
         : owner->point.rast_primitive_id_slot;
   kk_msl_copy(&gs->gs_main, &owner->point_gs_main);
   kk_msl_copy(&gs->msl_data[MESA_SHADER_GEOMETRY], &owner->point_rast);

   int64_t start = os_time_get_nano();
   VkResult result = kk_so_init(dev, gs);
   kk_log_compile(dev, "point_gs", start);
   if (result != VK_SUCCESS) {
      kk_shader_destroy(&dev->vk, gs, NULL);
      return NULL;
   }

   return gs;
}

struct kk_shader *
kk_shader_get_point_gs(struct kk_device *dev, struct kk_shader *shader)
{
   if (!shader->info.point_variant)
      return NULL;

   mtx_lock(&shader->variants_lock);
   if (shader->point_gs == NULL && !shader->point_failed) {
      shader->point_gs = kk_create_point_gs(dev, shader);
      shader->point_failed = shader->point_gs == NULL;
   }
   struct kk_shader *gs = shader->point_gs;
   mtx_unlock(&shader->variants_lock);

   return gs;
}

mtl_compute_pipeline_state *
kk_shader_get_point_cs(struct kk_device *dev, struct kk_shader *shader)
{
   if (!shader->info.point_variant || !shader->point.has_emu)
      return NULL;

   mtx_lock(&shader->variants_lock);
   if (shader->point_cs == NULL && !shader->point_failed) {
      int64_t start = os_time_get_nano();
      VkResult result = kk_compile_compute_pipeline(
         dev, &shader->point_emu, KK_POLY_WG_SIZE, &shader->point_cs);
      kk_log_compile(dev, "point_emu", start);
      shader->point_failed = result != VK_SUCCESS;
   }
   mtl_compute_pipeline_state *cs = shader->point_cs;
   mtx_unlock(&shader->variants_lock);

   return cs;
}

static VkResult
kk_compile_so_variant(struct kk_device *dev,
                      const struct vk_shader_compile_info *info,
                      nir_shader *nir, bool emulated,
                      enum kk_feature_key features,
                      const VkAllocationCallbacks *pAllocator,
                      struct kk_shader **shader_out)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);
   const struct vk_graphics_pipeline_state *state = &kk_unknown_state;
   mesa_shader_stage stage = nir->info.stage;

   msl_preprocess_nir_workarounds(nir, pdev->settings.disabled_workarounds);
   kk_lower_nir(dev, nir, emulated, true, info->robustness,
                info->set_layout_count, info->set_layouts, state, features);

   uint8_t primitive_id_slot = 0u;
   if (stage == MESA_SHADER_GEOMETRY)
      primitive_id_slot = kk_partition_gs_primitive_id(nir);

   nir_opt_varyings_bulk(&nir, 1u, true, UINT32_MAX, UINT32_MAX, nir_opts,
                         NULL);

   if (emulated && stage == MESA_SHADER_TESS_EVAL) {
      kk_store_tes_primitive_id(nir);
   } else if (stage == MESA_SHADER_GEOMETRY &&
              nir_shader_intrinsics_pass(nir, load_forwarded_patch_id,
                                         nir_metadata_none, NULL)) {
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }

   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL) {
      if (!emulated)
         nir->info.next_stage = MESA_SHADER_FRAGMENT;
      else if (stage == MESA_SHADER_VERTEX &&
               (info->next_stage_mask &
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT))
         nir->info.next_stage = MESA_SHADER_TESS_CTRL;
      else
         nir->info.next_stage = MESA_SHADER_GEOMETRY;
   }

   VkResult result = kk_compile_shader(dev, nir, NULL, true, info->robustness,
                                       state, pAllocator, shader_out);
   if (result != VK_SUCCESS)
      return result;

   (*shader_out)->info.rast_primitive_id_slot = primitive_id_slot;
   (*shader_out)->info.shader_object = true;
   return VK_SUCCESS;
}

static VkResult
kk_compile_shader_object(struct kk_device *dev,
                         const struct vk_shader_compile_info *info,
                         enum kk_feature_key features,
                         const VkAllocationCallbacks *pAllocator,
                         struct vk_shader **shader_out)
{
   nir_shader *nir = info->nir;
   mesa_shader_stage stage = nir->info.stage;
   VkShaderStageFlags emu_stages =
      stage == MESA_SHADER_VERTEX
         ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
              VK_SHADER_STAGE_GEOMETRY_BIT
      : stage == MESA_SHADER_TESS_EVAL ? VK_SHADER_STAGE_GEOMETRY_BIT
                                       : 0;
   bool emu = info->next_stage_mask & emu_stages;
   bool hw = emu_stages &&
             (!emu || (info->next_stage_mask & VK_SHADER_STAGE_FRAGMENT_BIT));

   nir_shader *emu_nir = NULL;
   if (emu && hw)
      emu_nir = nir_shader_clone(NULL, nir);
   else if (emu)
      emu_nir = nir, nir = NULL;

   struct kk_shader *shader = NULL, *emu_shader = NULL;
   VkResult result = VK_SUCCESS;
   if (nir) {
      result = kk_compile_so_variant(dev, info, nir, false, features,
                                     pAllocator, &shader);
      if (result != VK_SUCCESS) {
         ralloc_free(emu_nir);
         return result;
      }
   }

   if (emu_nir) {
      result = kk_compile_so_variant(dev, info, emu_nir, true, features,
                                     pAllocator, &emu_shader);
      if (result != VK_SUCCESS) {
         if (shader)
            kk_shader_destroy(&dev->vk, shader, pAllocator);
         return result;
      }
   }

   if (shader && emu_shader) {
      shader->msl_emu = emu_shader->msl_data[stage];
      memset(&emu_shader->msl_data[stage], 0, sizeof(struct msl_compile_data));
      shader->info.vs.outputs_written = emu_shader->info.vs.outputs_written;
      if (stage == MESA_SHADER_TESS_EVAL)
         shader->info.tess.tes_outputs = emu_shader->info.tess.tes_outputs;
      kk_shader_destroy(&dev->vk, emu_shader, pAllocator);
   } else if (emu_shader) {
      shader = emu_shader;
   }

   shader->info.so_hw = hw;
   shader->info.so_emu = emu;

   result = kk_so_init(dev, shader);
   if (result != VK_SUCCESS) {
      kk_shader_destroy(&dev->vk, shader, pAllocator);
      return result;
   }

   *shader_out = &shader->vk;
   return VK_SUCCESS;
}

static VkResult
kk_compile_shaders(struct vk_device *device, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *enabled_features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   VkResult result = VK_SUCCESS;
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   if (enabled_features == NULL)
      enabled_features = &device->enabled_features;
   enum kk_feature_key features = kk_make_feature_key(enabled_features);

   if (state == NULL && infos[0].stage != MESA_SHADER_COMPUTE) {
      for (uint32_t i = 0u; i < shader_count; ++i) {
         result = kk_compile_shader_object(dev, &infos[i], features,
                                           pAllocator, &shaders_out[i]);
         if (result != VK_SUCCESS) {
            for (uint32_t j = 0u; j < i; ++j)
               vk_shader_destroy(device, shaders_out[j], pAllocator);
            for (uint32_t j = i + 1u; j < shader_count; ++j)
               ralloc_free(infos[j].nir);
            memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
            return result;
         }
      }
      return VK_SUCCESS;
   }

   bool has_fs = infos[shader_count - 1u].stage == MESA_SHADER_FRAGMENT;
   bool partition = state && !(has_fs && shader_count > 1u);
   bool has_gs_producer = has_fs && shader_count > 1u &&
                          infos[shader_count - 2u].stage ==
                             MESA_SHADER_GEOMETRY;

   if ((KK_DEBUG(PARTITION) || has_gs_producer ||
        (state && kk_state_needs_link(state)) ||
        kk_state_may_draw_points(state)) &&
       state && has_fs && shader_count > 1u) {
      result = kk_compile_shaders(device, shader_count - 1u, infos, state,
                                  enabled_features, pAllocator, shaders_out);
      if (result != VK_SUCCESS)
         return result;

      result = kk_compile_shaders(device, 1u, &infos[shader_count - 1u],
                                  state, enabled_features, pAllocator,
                                  &shaders_out[shader_count - 1u]);
      if (result != VK_SUCCESS) {
         for (uint32_t i = 0u; i < shader_count - 1u; ++i)
            vk_shader_destroy(device, shaders_out[i], pAllocator);
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
      }
      return result;
   }

   nir_shader *nir_shaders[shader_count];
   struct kk_shader *shaders[shader_count];

   /* Lower shaders, notably lowering IO. This is a prerequisite for intershader
    * optimization. */
   const struct vk_pipeline_robustness_state *vertex_robustness = &rs_none;
   for (uint32_t i = 0u; i < shader_count; ++i) {
      const struct vk_shader_compile_info *info = &infos[i];
      nir_shader *nir = info->nir;

      /* Stages before the tessellator or a GS run as compute */
      bool emulated_stage =
         info->next_stage_mask & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                  VK_SHADER_STAGE_GEOMETRY_BIT);

      msl_preprocess_nir_workarounds(nir, pdev->settings.disabled_workarounds);
      kk_lower_nir(dev, nir, emulated_stage, partition, info->robustness,
                   info->set_layout_count, info->set_layouts, state, features);

      if (nir->info.stage == MESA_SHADER_VERTEX)
         vertex_robustness = info->robustness;

      nir_shaders[i] = nir;
   }

   uint8_t gs_primitive_id_slot = 0u;
   if (partition) {
      uint32_t link_count = has_fs ? 1u : shader_count;
      if (!has_fs &&
          nir_shaders[link_count - 1u]->info.stage == MESA_SHADER_GEOMETRY) {
         gs_primitive_id_slot =
            kk_partition_gs_primitive_id(nir_shaders[link_count - 1u]);
      }

      nir_opt_varyings_bulk(nir_shaders, link_count, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);
      nir_opt_varyings_bulk(nir_shaders, link_count, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);

      if (link_count >= 2u &&
          nir_shaders[link_count - 2u]->info.stage == MESA_SHADER_TESS_EVAL &&
          nir_shaders[link_count - 1u]->info.stage == MESA_SHADER_GEOMETRY) {
         kk_forward_tes_primitive_id(nir_shaders[link_count - 2u],
                                     nir_shaders[link_count - 1u]);
      }
   } else {
      /* nir_opt_varyings_bulk removes a GS primitive ID output that the FS
       * does not read as an input. The FS is last and a GS right before it. */
      if (shader_count >= 2u &&
          nir_shaders[shader_count - 2u]->info.stage == MESA_SHADER_GEOMETRY) {
         kk_forward_gs_primitive_id(nir_shaders[shader_count - 2u],
                                    nir_shaders[shader_count - 1u]);
      }

      nir_opt_varyings_bulk(nir_shaders, shader_count, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);
      /* Second pass is required because some dEQP-VK.glsl.matrix.sub.dynamic.*
       * would fail otherwise due to vertex outputting vec4 while fragments
       * reading vec3 when in reality only vec3 is needed. */
      nir_opt_varyings_bulk(nir_shaders, shader_count, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);

      /* After linking, so nir_opt_varyings leaves the new slot alone */
      if (shader_count >= 3u &&
          nir_shaders[shader_count - 3u]->info.stage ==
             MESA_SHADER_TESS_EVAL &&
          nir_shaders[shader_count - 2u]->info.stage == MESA_SHADER_GEOMETRY) {
         kk_forward_tes_primitive_id(nir_shaders[shader_count - 3u],
                                     nir_shaders[shader_count - 2u]);
      }
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      struct kk_shader *prev_stage = i > 0 ? shaders[i - 1] : NULL;
      result = kk_compile_shader(dev, nir_shaders[i], prev_stage, partition,
                                 vertex_robustness, state, pAllocator,
                                 &shaders[i]);

      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            kk_shader_destroy(&dev->vk, shaders[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(nir_shaders[j]);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }

      if (shaders[i]->info.stage == MESA_SHADER_GEOMETRY)
         shaders[i]->info.rast_primitive_id_slot = gs_primitive_id_slot;
   }

   if (shaders[0]->vk.stage == MESA_SHADER_COMPUTE) {
      struct kk_shader *s = shaders[0];
      uint32_t local_size_threads = s->info.cs.local_size.x *
                                    s->info.cs.local_size.y *
                                    s->info.cs.local_size.z;
      result =
         kk_compile_compute_pipeline(dev, &s->msl_data[MESA_SHADER_COMPUTE],
                                     local_size_threads, &s->pipeline.cs);
   } else if (partition) {
      result = kk_init_partition(dev, state, shaders, shader_count);
   } else {
      gather_graphics_pipeline_create_info(state, shaders, shader_count);
      result = kk_compile_graphics_pipeline(dev, shaders[0]);
   }

   if (result != VK_SUCCESS) {
      for (uint32_t i = 0; i < shader_count; i++) {
         kk_shader_destroy(&dev->vk, shaders[i], pAllocator);
      }

      /* Memset the output array */
      memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
   } else {
      memcpy(shaders_out, shaders, shader_count * sizeof(*shaders_out));
   }

   return result;
}

/* A GS and the VS holding its pipeline also carry the GS compute pass. */
static bool
kk_has_gs_main(const struct kk_shader_info *info)
{
   return info->stage == MESA_SHADER_GEOMETRY ||
          (info->stage == MESA_SHADER_VERTEX &&
           (info->vs.additional_stages_bits &
            BITFIELD_BIT(MESA_SHADER_GEOMETRY)));
}

static void
kk_msl_serialize(const struct msl_compile_data *data, struct blob *blob)
{
   uint32_t entrypoint_length = strlen(data->entrypoint_name) + 1;
   uint32_t code_length = strlen(data->code) + 1;
   blob_write_uint32(blob, entrypoint_length);
   blob_write_uint32(blob, code_length);
   blob_write_bytes(blob, data->entrypoint_name, entrypoint_length);
   blob_write_bytes(blob, data->code, code_length);
}

static bool
kk_shader_serialize(struct vk_device *vk_dev, const struct vk_shader *vk_shader,
                    struct blob *blob)
{
   struct kk_shader *shader = container_of(vk_shader, struct kk_shader, vk);

   blob_write_uint32(blob, KK_SHADER_BINARY_MAGIC);
   blob_write_uint32(blob, KK_SHADER_BINARY_VERSION);
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   kk_msl_serialize(&shader->msl_data[shader->info.stage], blob);
   if (shader->info.stage == MESA_SHADER_VERTEX) {
      u_foreach_bit(stage, shader->info.vs.additional_stages_bits) {
         kk_msl_serialize(&shader->msl_data[stage], blob);
      }
   }
   if (kk_has_gs_main(&shader->info))
      kk_msl_serialize(&shader->gs_main, blob);

   if (shader->info.rast_point_variant)
      kk_msl_serialize(&shader->msl_points, blob);

   if (shader->info.so_hw && shader->info.so_emu)
      kk_msl_serialize(&shader->msl_emu, blob);

   if (shader->info.point_variant) {
      blob_write_bytes(blob, &shader->point, sizeof(shader->point));
      if (shader->point.has_emu)
         kk_msl_serialize(&shader->point_emu, blob);
      kk_msl_serialize(&shader->point_gs_main, blob);
      kk_msl_serialize(&shader->point_rast, blob);
   }

   if (shader->info.partial && shader->info.stage == MESA_SHADER_FRAGMENT) {
      blob_write_uint64(blob, shader->nir_blob_size);
      blob_write_bytes(blob, shader->nir_blob, shader->nir_blob_size);
   }

   return !blob->out_of_memory;
}

static VkResult
kk_msl_deserialize(struct blob_reader *blob, struct msl_compile_data *data)
{
   const uint32_t entrypoint_length = blob_read_uint32(blob);
   const uint32_t code_length = blob_read_uint32(blob);
   /* Check lengths before handing strings to the Metal compiler. */
   size_t remaining = blob->end - blob->current;
   if (blob->overrun || entrypoint_length == 0 || code_length == 0 ||
       entrypoint_length > remaining ||
       code_length > remaining - entrypoint_length)
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

   data->entrypoint_name = ralloc_array(NULL, char, entrypoint_length);
   if (data->entrypoint_name == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   data->code = ralloc_array(NULL, char, code_length);
   if (data->code == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   blob_copy_bytes(blob, (void *)data->entrypoint_name, entrypoint_length);
   blob_copy_bytes(blob, (void *)data->code, code_length);

   if (blob->overrun || data->entrypoint_name[entrypoint_length - 1] != '\0' ||
       data->code[code_length - 1] != '\0' ||
       strlen(data->entrypoint_name) != entrypoint_length - 1 ||
       strlen(data->code) != code_length - 1)
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

   return VK_SUCCESS;
}

static VkResult
kk_deserialize_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                      uint32_t binary_version,
                      const VkAllocationCallbacks *pAllocator,
                      struct vk_shader **shader_out)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);

   uint32_t magic = blob_read_uint32(blob);
   uint32_t version = blob_read_uint32(blob);
   if (blob->overrun || binary_version != KK_SHADER_BINARY_VERSION ||
       magic != KK_SHADER_BINARY_MAGIC || version != KK_SHADER_BINARY_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   struct kk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));
   const uint32_t graphics_stages = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
      BITFIELD_BIT(MESA_SHADER_TESS_EVAL) | BITFIELD_BIT(MESA_SHADER_GEOMETRY) |
      BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   if (blob->overrun || (unsigned)info.stage > MESA_SHADER_COMPUTE ||
       (info.stage == MESA_SHADER_VERTEX &&
        (info.vs.additional_stages_bits & ~graphics_stages)))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   bool so_split = info.stage == MESA_SHADER_VERTEX ||
                   info.stage == MESA_SHADER_TESS_EVAL;
   if ((info.shader_object &&
        (info.stage == MESA_SHADER_COMPUTE || !info.partial ||
         (so_split && !info.so_hw && !info.so_emu))) ||
       ((info.so_hw || info.so_emu) && (!so_split || !info.shader_object)))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   struct kk_shader *shader = vk_shader_zalloc(
      &dev->vk, &kk_shader_ops, info.stage, pAllocator, sizeof(*shader));
   if (shader == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   mtx_init(&shader->variants_lock, mtx_plain);

   shader->info = info;
   VkResult result = kk_msl_deserialize(blob, &shader->msl_data[info.stage]);
   if (result != VK_SUCCESS)
      goto fail;

   if (shader->info.stage == MESA_SHADER_VERTEX) {
      u_foreach_bit(stage, shader->info.vs.additional_stages_bits) {
         result = kk_msl_deserialize(blob, &shader->msl_data[stage]);
         if (result != VK_SUCCESS)
            goto fail;
      }
   }

   if (kk_has_gs_main(&info)) {
      result = kk_msl_deserialize(blob, &shader->gs_main);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (info.rast_point_variant) {
      result = kk_msl_deserialize(blob, &shader->msl_points);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (info.so_hw && info.so_emu) {
      result = kk_msl_deserialize(blob, &shader->msl_emu);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (info.point_variant) {
      blob_copy_bytes(blob, &shader->point, sizeof(shader->point));
      if (blob->overrun || !info.partial ||
          (info.stage != MESA_SHADER_VERTEX &&
           info.stage != MESA_SHADER_TESS_EVAL &&
           info.stage != MESA_SHADER_GEOMETRY) ||
          shader->point.has_emu != (info.stage != MESA_SHADER_GEOMETRY)) {
         result = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         goto fail;
      }
      if (shader->point.has_emu) {
         result = kk_msl_deserialize(blob, &shader->point_emu);
         if (result != VK_SUCCESS)
            goto fail;
      }
      result = kk_msl_deserialize(blob, &shader->point_gs_main);
      if (result != VK_SUCCESS)
         goto fail;
      result = kk_msl_deserialize(blob, &shader->point_rast);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (info.partial && info.stage == MESA_SHADER_FRAGMENT) {
      const uint64_t size = blob_read_uint64(blob);
      const size_t remaining = blob->end - blob->current;
      if (blob->overrun || size == 0 || size > remaining) {
         result = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         goto fail;
      }

      shader->nir_blob = malloc(size);
      if (shader->nir_blob == NULL) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      shader->nir_blob_size = size;
      blob_copy_bytes(blob, shader->nir_blob, size);
   }

   result = kk_upload_gs_indices(dev, shader);
   if (result != VK_SUCCESS)
      goto fail;

   if (info.stage == MESA_SHADER_COMPUTE) {
      uint32_t local_size_threads = shader->info.cs.local_size.x *
                                    shader->info.cs.local_size.y *
                                    shader->info.cs.local_size.z;
      result = kk_compile_compute_pipeline(
         dev, &shader->msl_data[MESA_SHADER_COMPUTE], local_size_threads,
         &shader->pipeline.cs);
   } else if (info.shader_object) {
      result = kk_so_init(dev, shader);
   } else if (info.partial && info.stage == MESA_SHADER_VERTEX) {
      result = kk_init_fast_prerast(dev, shader);
   } else if (info.partial && info.stage == MESA_SHADER_FRAGMENT) {
      result = kk_init_fast_fs(dev, shader);
   } else if (info.stage == MESA_SHADER_VERTEX) {
      result = kk_compile_graphics_pipeline(dev, shader);
   }

   if (result != VK_SUCCESS)
      goto fail;

   *shader_out = &shader->vk;

   return VK_SUCCESS;

fail:
   kk_shader_destroy(&dev->vk, shader, pAllocator);
   return vk_error(dev, result);
}

static bool
zero_missing_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   const uint64_t *missing = data;
   nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   if (!(*missing & BITFIELD64_BIT(io.location)))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_zero(b, intr->def.num_components,
                                            intr->def.bit_size));
   return true;
}

static VkResult
kk_compile_fs_variant(struct kk_device *dev, struct kk_shader *fs,
                      const struct kk_fs_link_key *key,
                      struct kk_fs_variant *variant)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   struct disk_cache *cache =
      dev->vk.disable_internal_cache ? NULL : pdev->vk.disk_cache;
   cache_key disk_key;
   if (cache) {
      blake3_hasher hash;
      _mesa_blake3_init(&hash);
      _mesa_blake3_update(&hash, fs->nir_blob, fs->nir_blob_size);
      _mesa_blake3_update(&hash, key, sizeof(*key));
      blake3_hash digest;
      _mesa_blake3_final(&hash, digest);
      disk_cache_compute_key(cache, digest, sizeof(digest), disk_key);
      size_t size;
      void *cached = disk_cache_get(cache, disk_key, &size);
      if (cached) {
         struct blob_reader blob;
         blob_reader_init(&blob, cached, size);
         struct msl_compile_data data = {0};
         VkResult result = kk_msl_deserialize(&blob, &data);
         if (result == VK_SUCCESS && blob.current == blob.end) {
            result = kk_new_function(dev, &data, &variant->lib, &variant->fn);
            if (result == VK_SUCCESS) {
               _mesa_blake3_compute(data.code, strlen(data.code),
                                    variant->hash);
               variant->shader_blend = key->shader_blend;
               variant->native_primitive_id = kk_msl_native_primitive_id(&data);
            }
         } else {
            result = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;
         }
         ralloc_free(data.code);
         ralloc_free(data.entrypoint_name);
         free(cached);
         if (result == VK_SUCCESS) {
            if (KK_DEBUG(PSO))
               mesa_logi("FS variant disk cache hit");
            return result;
         }
      }
   }
   if (KK_DEBUG(PSO))
      mesa_logi("FS variant compile");

   struct blob_reader reader;
   blob_reader_init(&reader, fs->nir_blob, fs->nir_blob_size);
   nir_shader *nir = nir_deserialize(NULL, &kk_nir_options, &reader);
   if (nir == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   if (key->missing_inputs) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, zero_missing_input,
               nir_metadata_control_flow, (void *)&key->missing_inputs);
   }

   if (key->primitive_id_slot) {
      const struct nir_lower_sysvals_to_varyings_options sysvals = {
         .primitive_id = true,
      };
      bool progress = false;
      NIR_PASS(progress, nir, nir_lower_sysvals_to_varyings, &sysvals);
      unsigned slot = key->primitive_id_slot;
      if (progress)
         nir_shader_intrinsics_pass(nir, move_primitive_id, nir_metadata_all,
                                    &slot);
   }

   struct vk_render_pass_state rp = {
      .color_attachment_count = MAX_DRAW_BUFFERS,
      .depth_attachment_format =
         key->remove_depth ? VK_FORMAT_UNDEFINED : VK_FORMAT_D32_SFLOAT,
   };
   memcpy(rp.color_attachment_formats, key->rt_formats,
          sizeof(key->rt_formats));

   struct vk_color_blend_state cb = {
      .logic_op_enable = key->logic_op_enable,
      .logic_op = key->logic_op,
      .attachment_count = MAX_DRAW_BUFFERS,
   };
   memcpy(cb.attachments, key->blend, sizeof(key->blend));

   const struct vk_graphics_pipeline_state state = {
      .rp = &rp,
      .cb = key->shader_blend ? &cb : NULL,
   };

   if (key->alpha_to_coverage || key->alpha_to_one) {
      NIR_PASS(_, nir, msl_nir_lower_multisample_alpha,
               key->alpha_to_coverage, key->alpha_to_one);
   }

   if (state.cb)
      kk_lower_fs_blend(nir, &state);

   enum pipe_format rts[MAX_DRAW_BUFFERS];
   for (uint32_t i = 0u; i < MAX_DRAW_BUFFERS; ++i)
      rts[i] = vk_format_to_pipe_format(key->rt_formats[i]);
   NIR_PASS(_, nir, msl_nir_fs_force_output_signedness, rts);

   if (key->remove_depth)
      NIR_PASS(_, nir, msl_nir_fs_remove_depth_write);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, kk_nir_swizzle_fragment_output,
            nir_metadata_control_flow, (void *)&state);
   NIR_PASS(_, nir, msl_nir_fs_io_types);

   if (key->sample_mask) {
      if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(7)) &&
          !nir->info.fs.early_fragment_tests) {
         nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_at(nir_after_impl(entrypoint));

         nir_def *sample_id = nir_load_sample_id(&b);
         nir_def *sample_bit = nir_ishl(&b, nir_imm_int(&b, 1), sample_id);
         nir_def *sample_mask_bit =
            nir_iand(&b, nir_load_sample_mask_in(&b), sample_bit);
         nir_discard_if(&b, nir_ieq_imm(&b, sample_mask_bit, 0u));
      }
      NIR_PASS(_, nir, msl_lower_dynamic_sample_mask);
   }

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, kk_nir_lower_descriptors, &rs_none, 0, NULL);
   msl_nir_lower_clip_cull_distance(nir, key->num_cull_distances);

   struct msl_compile_data data = {0};
   kk_compile_msl(pdev, nir, &state, true, &data, NULL);

   VkResult result = kk_new_function(dev, &data, &variant->lib, &variant->fn);
   if (result == VK_SUCCESS) {
      _mesa_blake3_compute(data.code, strlen(data.code), variant->hash);
      variant->shader_blend = key->shader_blend;
      variant->native_primitive_id = kk_msl_native_primitive_id(&data);
      if (cache) {
         struct blob blob;
         blob_init(&blob);
         kk_msl_serialize(&data, &blob);
         if (!blob.out_of_memory)
            disk_cache_put(cache, disk_key, blob.data, blob.size, NULL);
         blob_finish(&blob);
      }
   }

   ralloc_free(data.code);
   ralloc_free(data.entrypoint_name);
   return result;
}

VkResult
kk_shader_get_fs_variant(struct kk_device *dev, struct kk_shader *fs,
                         const struct kk_fs_link_key *key,
                         const struct kk_fs_variant **variant_out,
                         const struct kk_pso_request *request)
{
   uint32_t hash = kk_fs_link_key_hash(key);
   blake3_hash identity = {0};
   kk_variant_shader_key(dev, fs, key, sizeof(*key), identity);

   if (dev->variant_trace_enabled &&
       mtx_trylock(&fs->variants_lock) != thrd_success) {
      int64_t wait_start = os_time_get_nano();
      mtx_lock(&fs->variants_lock);
      kk_variant_trace(dev, KK_VARIANT_FS_WAIT, identity, sizeof(identity),
                       wait_start, os_time_get_nano());
   } else if (!dev->variant_trace_enabled) {
      mtx_lock(&fs->variants_lock);
   }
   struct hash_entry *he =
      _mesa_hash_table_search_pre_hashed(fs->variants, hash, key);
   bool lookup_only = request && request->lookup_only;
   if (lookup_only &&
       (!he || ((struct kk_fs_variant_entry *)he->data)->compiling)) {
      mtx_unlock(&fs->variants_lock);
      return VK_NOT_READY;
   }
   if (he) {
      struct kk_fs_variant_entry *entry = he->data;
      if (entry->compiling) {
         int64_t start = os_time_get_nano();
         struct mtl_compile_phase phase =
            kk_pso_phase_begin(dev, "fs_variant_wait", key, sizeof(*key));
         while (entry->compiling)
            cnd_wait(&entry->ready, &fs->variants_lock);
         mtl_compile_phase_end(phase, true);
         kk_log_compile(dev, "fs_variant_wait", start);
         kk_variant_trace(dev, KK_VARIANT_FS_WAIT, identity, sizeof(identity),
                          start, os_time_get_nano());
      } else if (entry->result == VK_SUCCESS) {
         kk_variant_trace(dev, KK_VARIANT_READY_HIT, identity, sizeof(identity),
                          0, 0);
      }
      VkResult result = entry->result;
      mtx_unlock(&fs->variants_lock);
      if (result == VK_SUCCESS)
         *variant_out = &entry->variant;
      return result;
   }

   struct kk_fs_variant_entry *entry =
      rzalloc(fs->variants, struct kk_fs_variant_entry);
   if (!entry) {
      mtx_unlock(&fs->variants_lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   entry->key = *key;
   entry->compiling = true;
   if (cnd_init(&entry->ready) != thrd_success) {
      ralloc_free(entry);
      mtx_unlock(&fs->variants_lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (!_mesa_hash_table_insert_pre_hashed(fs->variants, hash, &entry->key,
                                           entry)) {
      cnd_destroy(&entry->ready);
      ralloc_free(entry);
      mtx_unlock(&fs->variants_lock);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   mtx_unlock(&fs->variants_lock);

   struct kk_fs_variant variant = {0};
   int64_t start = os_time_get_nano();
   struct mtl_compile_phase phase =
      kk_pso_phase_begin(dev, "fs_generate", key, sizeof(*key));
   VkResult result = kk_compile_fs_variant(dev, fs, key, &variant);
   mtl_compile_phase_end(phase, result == VK_SUCCESS);
   kk_log_compile(dev, "fs_variant", start);
   int64_t end = os_time_get_nano();
   kk_variant_trace(dev, KK_VARIANT_FS_PREPARE, identity, sizeof(identity),
                    start, end);
   kk_variant_trace(dev, result == VK_SUCCESS ? KK_VARIANT_COMPLETE :
                                             KK_VARIANT_FAILURE,
                    identity, sizeof(identity), end, end);

   mtx_lock(&fs->variants_lock);
   if (result == VK_SUCCESS)
      entry->variant = variant;
   entry->result = result;
   entry->compiling = false;
   cnd_broadcast(&entry->ready);
   mtx_unlock(&fs->variants_lock);

   if (result == VK_SUCCESS)
      *variant_out = &entry->variant;
   return result;
}

struct kk_shader *
kk_device_get_empty_fs(struct kk_device *dev)
{
   struct kk_shader *empty_fs = p_atomic_read(&dev->empty_fs);
   if (empty_fs)
      return empty_fs;

   simple_mtx_lock(&dev->empty_fs_lock);
   if (dev->empty_fs == NULL) {
      struct vk_multiview_state mv = {0};
      struct vk_render_pass_state rp = {0};
      const struct vk_graphics_pipeline_state state = {
         .mv = &mv,
         .rp = &rp,
      };
      nir_shader *nir =
         get_empty_nir(dev, MESA_SHADER_FRAGMENT, &state, 0, true);

      struct kk_shader *fs = NULL;
      if (kk_compile_shader(dev, nir, NULL, true, &rs_none, &state, NULL,
                            &fs) == VK_SUCCESS) {
         if (kk_init_fast_fs(dev, fs) == VK_SUCCESS)
            p_atomic_set(&dev->empty_fs, fs);
         else
            kk_shader_destroy(&dev->vk, fs, NULL);
      }
   }
   simple_mtx_unlock(&dev->empty_fs_lock);

   return dev->empty_fs;
}

void
kk_device_finish_empty_fs(struct kk_device *dev)
{
   if (dev->empty_fs)
      kk_shader_destroy(&dev->vk, dev->empty_fs, NULL);
   dev->empty_fs = NULL;
}

void
kk_cmd_bind_compute_shader(struct kk_cmd_buffer *cmd, struct kk_shader *shader)
{
   cmd->state.shaders[MESA_SHADER_COMPUTE] = shader;
}

static void
kk_cmd_bind_graphics_shader(struct kk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct kk_shader *shader)
{
   cmd->state.shaders[stage] = shader;
   cmd->state.dirty_shaders |= BITFIELD_BIT(stage);
   cmd->state.gfx.dirty |= KK_DIRTY_PSO;

   /* Relevant pipeline data is only stored in vertex shaders */
   if (stage != MESA_SHADER_VERTEX || shader == NULL)
      return;

   bool requires_dynamic_depth_stencil =
      shader->info.partial || shader->pipeline.gfx.ds_handle == NULL;
   if (!cmd->state.gfx.is_depth_stencil_dynamic &&
       requires_dynamic_depth_stencil) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   }

   if (cmd->state.gfx.is_depth_stencil_dynamic) {
      /* If we are switching from dynamic to static, we need to clean up
       * temporary state. Otherwise, leave the existing dynamic state
       * untouched.
       */
      if (!requires_dynamic_depth_stencil) {
         mtl_release(cmd->state.gfx.depth_stencil_state);
         cmd->state.gfx.depth_stencil_state = shader->pipeline.gfx.ds_handle;
      }
   } else
      cmd->state.gfx.depth_stencil_state = shader->pipeline.gfx.ds_handle;
   cmd->state.gfx.is_depth_stencil_dynamic = requires_dynamic_depth_stencil;
   cmd->state.gfx.dirty |= KK_DIRTY_VB;

   if (!shader->info.partial)
      cmd->state.gfx.pipeline_sample_count = shader->info.vs.sample_count;
}

static void
kk_cmd_bind_shaders(struct vk_command_buffer *cmd_buffer, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct kk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct kk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct kk_shader *shader = container_of(shaders[i], struct kk_shader, vk);

      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL)
         kk_cmd_bind_compute_shader(cmd, shader);
      else
         kk_cmd_bind_graphics_shader(cmd, stages[i], shader);
   }
}

static VkResult
kk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   return vk_outarray_status(&out);
}

static VkResult
kk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   /* TODO_KOSMICKRISP */
   return VK_SUCCESS;
}

static VkResult
kk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   /* TODO_KOSMICKRISP */
   return VK_SUCCESS;
}

static void
kk_vk_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                     const VkAllocationCallbacks *pAllocator)
{
   kk_shader_destroy(vk_dev, container_of(vk_shader, struct kk_shader, vk),
                     pAllocator);
}

static const struct vk_shader_ops kk_shader_ops = {
   .destroy = kk_vk_shader_destroy,
   .serialize = kk_shader_serialize,
   .get_executable_properties = kk_shader_get_executable_properties,
   .get_executable_statistics = kk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      kk_shader_get_executable_internal_representations,
};

static VkResult
kk_compile_shaders_logged(struct vk_device *device, uint32_t shader_count,
                          struct vk_shader_compile_info *infos,
                          const struct vk_graphics_pipeline_state *state,
                          const struct vk_features *enabled_features,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shaders_out)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   enum kk_compile_origin origin = kk_compile_origin;
   kk_compile_origin = KK_COMPILE_ORIGIN_CREATE;
   int64_t start = os_time_get_nano();
   VkResult result = kk_compile_shaders(device, shader_count, infos, state,
                                        enabled_features, pAllocator,
                                        shaders_out);
   int64_t end = os_time_get_nano();
   kk_log_compile(dev, "shaders", start);
   for (uint32_t i = 0; i < shader_count; i++) {
      if (infos[i].cache_trace)
         kk_log_cache_shader(dev, infos[i].cache_trace,
                             result == VK_SUCCESS ? "shaders"
                                                  : "shaders_failed",
                             start, end);
   }
   kk_compile_origin = origin;
   return result;
}

static VkResult
kk_deserialize_shader_logged(struct vk_device *vk_dev,
                             struct blob_reader *blob,
                             uint32_t binary_version,
                             const VkAllocationCallbacks *pAllocator,
                             struct vk_shader **shader_out)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);
   enum kk_compile_origin origin = kk_compile_origin;
   kk_compile_origin = KK_COMPILE_ORIGIN_CREATE;
   int64_t start = os_time_get_nano();
   VkResult result = kk_deserialize_shader(vk_dev, blob, binary_version,
                                           pAllocator, shader_out);
   kk_log_compile(dev, "deserialize", start);
   kk_compile_origin = origin;
   return result;
}

static void
kk_shader_cache_trace(struct vk_device *device,
                       const struct vk_shader_cache_trace *trace,
                       const char *outcome)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   kk_log_cache_shader(dev, trace, outcome, 0, 0);
}

static void
kk_graphics_pipeline_created(struct vk_device *vk_dev,
                             const struct vk_graphics_pipeline_state *state,
                             uint32_t shader_count,
                             struct vk_shader *const *shaders)
{
   struct kk_device *dev = container_of(vk_dev, struct kk_device, vk);
   for (uint32_t i = 0; i < shader_count; ++i) {
      if (!shaders[i] || shaders[i]->stage != MESA_SHADER_VERTEX)
         continue;
      struct kk_shader *vs = container_of(shaders[i], struct kk_shader, vk);
      if (!vs->pending)
         continue;
      kk_pso_schedule_render(dev, vs);
      if (kk_fail_on_compile && kk_pso_render_pending(dev, vs))
         kk_compile_required = true;
   }
   kk_prewarm_pipeline(vk_dev, state, shader_count, shaders);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache,
                           uint32_t createInfoCount,
                           const VkGraphicsPipelineCreateInfo *pCreateInfos,
                           const VkAllocationCallbacks *pAllocator,
                           VkPipeline *pPipelines)
{
   VkResult first_error_or_success = VK_SUCCESS;
   memset(pPipelines, 0, createInfoCount * sizeof(*pPipelines));

   for (uint32_t i = 0; i < createInfoCount; i++) {
      const VkPipelineCreateFlags2KHR flags =
         vk_graphics_pipeline_create_flags(&pCreateInfos[i]);
      kk_fail_on_compile =
         flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR;
      kk_compile_required = false;
      VkResult result = vk_common_CreateGraphicsPipelines(
         device, pipelineCache, 1, &pCreateInfos[i], pAllocator,
         &pPipelines[i]);
      if (result == VK_SUCCESS && kk_compile_required) {
         vk_common_DestroyPipeline(device, pPipelines[i], pAllocator);
         pPipelines[i] = VK_NULL_HANDLE;
         result = VK_PIPELINE_COMPILE_REQUIRED;
      }
      kk_fail_on_compile = false;
      kk_compile_required = false;

      if (result == VK_SUCCESS)
         continue;
      if (first_error_or_success == VK_SUCCESS)
         first_error_or_success = result;
      if (result != VK_PIPELINE_COMPILE_REQUIRED)
         return result;
      if (flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
         return result;
   }

   return first_error_or_success;
}

const struct vk_device_shader_ops kk_device_shader_ops = {
   .get_nir_options = kk_get_nir_options,
   .get_spirv_options = kk_get_spirv_options,
   .preprocess_nir = kk_preprocess_nir,
   .hash_state = kk_hash_graphics_state,
   .compile = kk_compile_shaders_logged,
   .shader_cache_trace = kk_shader_cache_trace,
   .deserialize = kk_deserialize_shader_logged,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .graphics_pipeline_created = kk_graphics_pipeline_created,
   .cmd_bind_shaders = kk_cmd_bind_shaders,
};
