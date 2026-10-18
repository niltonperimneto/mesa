/*
 * Copyright © 2026 KosmicKrisp Contributors
 * SPDX-License-Identifier: MIT
 */

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#include "kk_image.h"
#include "kk_image_view.h"

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #cond);                                                       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const uint32_t vs_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002bu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000000u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x0000001du, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00030005u, 0x00000009u, 0x00736f70u, 0x00060005u, 0x0000000cu, 0x565f6c67u,
   0x65747265u, 0x646e4978u, 0x00007865u, 0x00060005u, 0x0000001bu, 0x505f6c67u,
   0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000001bu, 0x00000000u,
   0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000001bu, 0x00000001u,
   0x505f6c67u, 0x746e696fu, 0x657a6953u, 0x00000000u, 0x00070006u, 0x0000001bu,
   0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu, 0x00070006u,
   0x0000001bu, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu,
   0x00030005u, 0x0000001du, 0x00000000u, 0x00040047u, 0x0000000cu, 0x0000000bu,
   0x0000002au, 0x00030047u, 0x0000001bu, 0x00000002u, 0x00050048u, 0x0000001bu,
   0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000001bu, 0x00000001u,
   0x0000000bu, 0x00000001u, 0x00050048u, 0x0000001bu, 0x00000002u, 0x0000000bu,
   0x00000003u, 0x00050048u, 0x0000001bu, 0x00000003u, 0x0000000bu, 0x00000004u,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000002u,
   0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u, 0x00040015u, 0x0000000au,
   0x00000020u, 0x00000001u, 0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au,
   0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000001u, 0x0004002bu, 0x0000000au,
   0x0000000eu, 0x00000001u, 0x0004002bu, 0x0000000au, 0x00000010u, 0x00000002u,
   0x00040017u, 0x00000017u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000018u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000018u, 0x00000019u, 0x00000001u,
   0x0004001cu, 0x0000001au, 0x00000006u, 0x00000019u, 0x0006001eu, 0x0000001bu,
   0x00000017u, 0x00000006u, 0x0000001au, 0x0000001au, 0x00040020u, 0x0000001cu,
   0x00000003u, 0x0000001bu, 0x0004003bu, 0x0000001cu, 0x0000001du, 0x00000003u,
   0x0004002bu, 0x0000000au, 0x0000001eu, 0x00000000u, 0x0004002bu, 0x00000006u,
   0x00000020u, 0x40000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u,
   0x0004002bu, 0x00000006u, 0x00000025u, 0x00000000u, 0x00040020u, 0x00000029u,
   0x00000003u, 0x00000017u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u,
   0x00000007u, 0x0004003du, 0x0000000au, 0x0000000du, 0x0000000cu, 0x000500c4u,
   0x0000000au, 0x0000000fu, 0x0000000du, 0x0000000eu, 0x000500c7u, 0x0000000au,
   0x00000011u, 0x0000000fu, 0x00000010u, 0x0004006fu, 0x00000006u, 0x00000012u,
   0x00000011u, 0x0004003du, 0x0000000au, 0x00000013u, 0x0000000cu, 0x000500c7u,
   0x0000000au, 0x00000014u, 0x00000013u, 0x00000010u, 0x0004006fu, 0x00000006u,
   0x00000015u, 0x00000014u, 0x00050050u, 0x00000007u, 0x00000016u, 0x00000012u,
   0x00000015u, 0x0003003eu, 0x00000009u, 0x00000016u, 0x0004003du, 0x00000007u,
   0x0000001fu, 0x00000009u, 0x0005008eu, 0x00000007u, 0x00000021u, 0x0000001fu,
   0x00000020u, 0x00050050u, 0x00000007u, 0x00000023u, 0x00000022u, 0x00000022u,
   0x00050083u, 0x00000007u, 0x00000024u, 0x00000021u, 0x00000023u, 0x00050051u,
   0x00000006u, 0x00000026u, 0x00000024u, 0x00000000u, 0x00050051u, 0x00000006u,
   0x00000027u, 0x00000024u, 0x00000001u, 0x00070050u, 0x00000017u, 0x00000028u,
   0x00000026u, 0x00000027u, 0x00000025u, 0x00000022u, 0x00050041u, 0x00000029u,
   0x0000002au, 0x0000001du, 0x0000001eu, 0x0003003eu, 0x0000002au, 0x00000028u,
   0x000100fdu, 0x00010038u,
};

static const uint32_t fs_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000013u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x00020011u, 0x00000028u, 0x0006000bu, 0x00000001u, 0x4c534c47u,
   0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u,
   0x00030010u, 0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u,
   0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00040005u, 0x00000009u,
   0x6f635f6fu, 0x00726f6cu, 0x00040005u, 0x0000000cu, 0x6e695f75u, 0x00747570u,
   0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000cu,
   0x00000021u, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000022u, 0x00000000u,
   0x00040047u, 0x0000000cu, 0x0000002bu, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000006u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000002u, 0x00000000u, 0x00040020u, 0x0000000bu, 0x00000000u,
   0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x00040015u,
   0x0000000eu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu,
   0x00000000u, 0x00040017u, 0x00000010u, 0x0000000eu, 0x00000002u, 0x0005002cu,
   0x00000010u, 0x00000011u, 0x0000000fu, 0x0000000fu, 0x00050036u, 0x00000002u,
   0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
   0x0000000au, 0x0000000du, 0x0000000cu, 0x00050062u, 0x00000007u, 0x00000012u,
   0x0000000du, 0x00000011u, 0x0003003eu, 0x00000009u, 0x00000012u, 0x000100fdu,
   0x00010038u,
};

int
main(int argc, char **argv)
{
   if (argc != 2) {
      fprintf(stderr, "Usage: %s <path-to-libvulkan_kosmickrisp.dylib>\n",
              argv[0]);
      return 1;
   }

   void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
   if (!library) {
      fprintf(stderr, "dlopen failed: %s\n", dlerror());
      return 1;
   }

   PFN_vkGetInstanceProcAddr get_inst_proc =
      (PFN_vkGetInstanceProcAddr)dlsym(library, "vk_icdGetInstanceProcAddr");
   CHECK(get_inst_proc != NULL);

   PFN_vkCreateInstance create_instance =
      (PFN_vkCreateInstance)get_inst_proc(NULL, "vkCreateInstance");
   CHECK(create_instance != NULL);

   VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "kk-linear-attachment-test",
      .apiVersion = VK_API_VERSION_1_4,
   };
   VkInstanceCreateInfo inst_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   VkInstance instance = VK_NULL_HANDLE;
   CHECK(create_instance(&inst_info, NULL, &instance) == VK_SUCCESS);

#define INST_PROC(name)                                                        \
   PFN_##name fn_##name = (PFN_##name)get_inst_proc(instance, #name);          \
   CHECK(fn_##name != NULL)

   INST_PROC(vkDestroyInstance);
   INST_PROC(vkEnumeratePhysicalDevices);
   INST_PROC(vkGetPhysicalDeviceFormatProperties2);
   INST_PROC(vkGetPhysicalDeviceImageFormatProperties2);
   INST_PROC(vkGetPhysicalDeviceMemoryProperties);
   INST_PROC(vkCreateDevice);
   INST_PROC(vkGetDeviceProcAddr);

   uint32_t pdev_count = 0;
   VkResult res = fn_vkEnumeratePhysicalDevices(instance, &pdev_count, NULL);
   if (res == VK_SUCCESS && pdev_count == 0) {
      fn_vkDestroyInstance(instance, NULL);
      dlclose(library);
      return 77;
   }
   CHECK(res == VK_SUCCESS && pdev_count > 0);

   VkPhysicalDevice *pdevs = calloc(pdev_count, sizeof(*pdevs));
   CHECK(pdevs != NULL);
   CHECK(fn_vkEnumeratePhysicalDevices(instance, &pdev_count, pdevs) ==
         VK_SUCCESS);
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);

   /* 1. Verify MR !44990 format & image format property queries for LINEAR
    * color attachments while ensuring LINEAR depth/stencil remains rejected.
    */
   VkFormatProperties3 fmt_props3 = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
   };
   VkFormatProperties2 fmt_props2 = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
      .pNext = &fmt_props3,
   };
   fn_vkGetPhysicalDeviceFormatProperties2(pdev, VK_FORMAT_R8G8B8A8_UNORM,
                                           &fmt_props2);
   CHECK((fmt_props2.formatProperties.linearTilingFeatures &
          VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0);
   CHECK((fmt_props2.formatProperties.linearTilingFeatures &
          VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0);
   CHECK((fmt_props3.linearTilingFeatures &
          VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) != 0);
   CHECK((fmt_props3.linearTilingFeatures & VK_FORMAT_FEATURE_2_BLIT_DST_BIT) !=
         0);

   VkPhysicalDeviceImageFormatInfo2 img_fmt_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   VkImageFormatProperties2 img_fmt_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   CHECK(fn_vkGetPhysicalDeviceImageFormatProperties2(
            pdev, &img_fmt_info, &img_fmt_props) == VK_SUCCESS);

   VkPhysicalDeviceImageFormatInfo2 ds_linear_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_D32_SFLOAT,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
   };
   CHECK(fn_vkGetPhysicalDeviceImageFormatProperties2(pdev, &ds_linear_info,
                                                      &img_fmt_props) ==
         VK_ERROR_FORMAT_NOT_SUPPORTED);

   /* Create logical device with dynamicRendering + dynamicRenderingLocalRead */
   float queue_priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR local_read = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR,
      .dynamicRenderingLocalRead = VK_TRUE,
   };
   VkPhysicalDeviceDynamicRenderingFeatures dyn_rendering = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .pNext = &local_read,
      .dynamicRendering = VK_TRUE,
   };
   const char *dev_exts[] = {
      VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME,
   };
   VkDeviceCreateInfo dev_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &dyn_rendering,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice device = VK_NULL_HANDLE;
   CHECK(fn_vkCreateDevice(pdev, &dev_info, NULL, &device) == VK_SUCCESS);

#define DEV_PROC(name)                                                         \
   PFN_##name fn_##name = (PFN_##name)fn_vkGetDeviceProcAddr(device, #name);   \
   CHECK(fn_##name != NULL)

   DEV_PROC(vkDestroyDevice);
   DEV_PROC(vkGetDeviceQueue);
   DEV_PROC(vkCreateImage);
   DEV_PROC(vkDestroyImage);
   DEV_PROC(vkGetImageMemoryRequirements);
   DEV_PROC(vkGetImageSubresourceLayout);
   DEV_PROC(vkAllocateMemory);
   DEV_PROC(vkFreeMemory);
   DEV_PROC(vkMapMemory);
   DEV_PROC(vkUnmapMemory);
   DEV_PROC(vkBindImageMemory);
   DEV_PROC(vkCreateImageView);
   DEV_PROC(vkDestroyImageView);
   DEV_PROC(vkCreateDescriptorSetLayout);
   DEV_PROC(vkDestroyDescriptorSetLayout);
   DEV_PROC(vkCreateDescriptorPool);
   DEV_PROC(vkDestroyDescriptorPool);
   DEV_PROC(vkAllocateDescriptorSets);
   DEV_PROC(vkUpdateDescriptorSets);
   DEV_PROC(vkCreatePipelineLayout);
   DEV_PROC(vkDestroyPipelineLayout);
   DEV_PROC(vkCreateShaderModule);
   DEV_PROC(vkDestroyShaderModule);
   DEV_PROC(vkCreateGraphicsPipelines);
   DEV_PROC(vkDestroyPipeline);
   DEV_PROC(vkCreateCommandPool);
   DEV_PROC(vkDestroyCommandPool);
   DEV_PROC(vkAllocateCommandBuffers);
   DEV_PROC(vkResetCommandBuffer);
   DEV_PROC(vkBeginCommandBuffer);
   DEV_PROC(vkEndCommandBuffer);
   DEV_PROC(vkCmdBeginRendering);
   DEV_PROC(vkCmdEndRendering);
   DEV_PROC(vkCmdBindPipeline);
   DEV_PROC(vkCmdBindDescriptorSets);
   DEV_PROC(vkCmdSetRenderingInputAttachmentIndices);
   DEV_PROC(vkCmdDraw);
   DEV_PROC(vkCmdClearColorImage);
   DEV_PROC(vkCmdBlitImage);
   DEV_PROC(vkQueueSubmit);
   DEV_PROC(vkQueueWaitIdle);

   VkPhysicalDeviceMemoryProperties mem_props;
   fn_vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
   uint32_t host_mem_type = UINT32_MAX;
   for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
      if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
         host_mem_type = i;
         break;
      }
   }
   CHECK(host_mem_type != UINT32_MAX);

   /* 2. Verify MR !45325 Commit 1 + MR !44990 (kk-input-attachment-2d):
    * A single-layer LINEAR image with COLOR_ATTACHMENT | TRANSFER_DST | SAMPLED
    * stays MTL_TEXTURE_TYPE_2D, and creating a 2D VkImageView on it does NOT
    * allocate an input view when INPUT_ATTACHMENT_BIT is absent.
    * When INPUT_ATTACHMENT_BIT IS set on a LINEAR view, it allocates a 2D
    * input view (input_2d_gpu_resource_id != 0, input_gpu_resource_id == 0).
    */
   const uint32_t width = 16;
   const uint32_t height = 16;
   VkImageCreateInfo linear_img_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage linear_image = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImage(device, &linear_img_info, NULL, &linear_image) ==
         VK_SUCCESS);

   struct kk_image *kk_lin_img = kk_image_from_handle(linear_image);
   CHECK(kk_lin_img->planes[0].layout.type == MTL_TEXTURE_TYPE_2D);

   VkMemoryRequirements mem_reqs;
   fn_vkGetImageMemoryRequirements(device, linear_image, &mem_reqs);
   VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_reqs.size,
      .memoryTypeIndex = host_mem_type,
   };
   VkDeviceMemory linear_mem = VK_NULL_HANDLE;
   CHECK(fn_vkAllocateMemory(device, &alloc_info, NULL, &linear_mem) ==
         VK_SUCCESS);
   CHECK(fn_vkBindImageMemory(device, linear_image, linear_mem, 0) ==
         VK_SUCCESS);

   VkImageViewCreateInfo linear_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = linear_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange =
         {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
   };
   VkImageView linear_view = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImageView(device, &linear_view_info, NULL, &linear_view) ==
         VK_SUCCESS);

   struct kk_image_view *kk_lin_view = kk_image_view_from_handle(linear_view);
   CHECK(kk_lin_view->planes[0].mtl_handle_input == NULL);
   CHECK(kk_lin_view->planes[0].input_gpu_resource_id == 0);
   CHECK(kk_lin_view->planes[0].input_2d_gpu_resource_id == 0);
   CHECK(kk_lin_view->planes[0].mtl_handle_render != NULL);
   CHECK(kk_lin_view->planes[0].mtl_handle_sampled != NULL);
   CHECK(kk_lin_view->planes[0].sampled_gpu_resource_id != 0);

   /* Verify that on a LINEAR image with INPUT_ATTACHMENT_BIT:
    * - A view restricted to COLOR_ATTACHMENT_BIT has no mtl_handle_input.
    * - A view with INPUT_ATTACHMENT_BIT sets input_2d_gpu_resource_id != 0 and
    *   keeps input_gpu_resource_id == 0 (2D view instead of 2D_ARRAY view).
    */
   VkImageCreateInfo linear_ia_img_info = linear_img_info;
   linear_ia_img_info.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   VkImage linear_ia_image = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImage(device, &linear_ia_img_info, NULL,
                          &linear_ia_image) == VK_SUCCESS);
   struct kk_image *kk_lin_ia_img = kk_image_from_handle(linear_ia_image);
   CHECK(kk_lin_ia_img->planes[0].layout.type == MTL_TEXTURE_TYPE_2D);

   VkDeviceMemory linear_ia_mem = VK_NULL_HANDLE;
   CHECK(fn_vkAllocateMemory(device, &alloc_info, NULL, &linear_ia_mem) ==
         VK_SUCCESS);
   CHECK(fn_vkBindImageMemory(device, linear_ia_image, linear_ia_mem, 0) ==
         VK_SUCCESS);

   VkImageViewUsageCreateInfo color_only_usage = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
   };
   VkImageViewCreateInfo color_only_view_info = linear_view_info;
   color_only_view_info.pNext = &color_only_usage;
   color_only_view_info.image = linear_ia_image;
   VkImageView color_only_view = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImageView(device, &color_only_view_info, NULL,
                              &color_only_view) == VK_SUCCESS);
   struct kk_image_view *kk_color_only_view =
      kk_image_view_from_handle(color_only_view);
   CHECK(kk_color_only_view->planes[0].mtl_handle_input == NULL);
   CHECK(kk_color_only_view->planes[0].input_gpu_resource_id == 0);
   CHECK(kk_color_only_view->planes[0].input_2d_gpu_resource_id == 0);
   CHECK(kk_color_only_view->planes[0].mtl_handle_sampled == NULL);
   CHECK(kk_color_only_view->planes[0].mtl_handle_render != NULL);

   VkImageViewUsageCreateInfo ia_only_usage = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .usage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
   };
   VkImageViewCreateInfo linear_ia_view_info = linear_view_info;
   linear_ia_view_info.pNext = &ia_only_usage;
   linear_ia_view_info.image = linear_ia_image;
   VkImageView linear_ia_view = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImageView(device, &linear_ia_view_info, NULL,
                              &linear_ia_view) == VK_SUCCESS);
   struct kk_image_view *kk_lin_ia_view =
      kk_image_view_from_handle(linear_ia_view);
   CHECK(kk_lin_ia_view->planes[0].mtl_handle_sampled == NULL);
   CHECK(kk_lin_ia_view->planes[0].sampled_gpu_resource_id == 0);
   CHECK(kk_lin_ia_view->planes[0].mtl_handle_input != NULL);
   CHECK(kk_lin_ia_view->planes[0].input_gpu_resource_id == 0);
   CHECK(kk_lin_ia_view->planes[0].input_2d_gpu_resource_id != 0);

   /* 3. Verify MR !45325 Commit 2:
    * An OPTIMAL image view with ONLY VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT (no
    * SAMPLED_BIT) has sampled_gpu_resource_id == 0 and input_gpu_resource_id !=
    * 0, and updating a VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT descriptor set with
    * both OPTIMAL and LINEAR input-attachment-only views succeeds.
    */
   VkImageCreateInfo opt_ia_img_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage opt_ia_image = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImage(device, &opt_ia_img_info, NULL, &opt_ia_image) ==
         VK_SUCCESS);

   VkMemoryRequirements opt_reqs;
   fn_vkGetImageMemoryRequirements(device, opt_ia_image, &opt_reqs);
   VkMemoryAllocateInfo opt_alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = opt_reqs.size,
      .memoryTypeIndex = host_mem_type,
   };
   VkDeviceMemory opt_mem = VK_NULL_HANDLE;
   CHECK(fn_vkAllocateMemory(device, &opt_alloc_info, NULL, &opt_mem) ==
         VK_SUCCESS);
   CHECK(fn_vkBindImageMemory(device, opt_ia_image, opt_mem, 0) == VK_SUCCESS);

   VkImageViewCreateInfo ia_view_info = linear_view_info;
   ia_view_info.pNext = &ia_only_usage;
   ia_view_info.image = opt_ia_image;
   VkImageView ia_view = VK_NULL_HANDLE;
   CHECK(fn_vkCreateImageView(device, &ia_view_info, NULL, &ia_view) ==
         VK_SUCCESS);

   struct kk_image_view *kk_ia_view = kk_image_view_from_handle(ia_view);
   CHECK(kk_ia_view->planes[0].mtl_handle_sampled == NULL);
   CHECK(kk_ia_view->planes[0].sampled_gpu_resource_id == 0);
   CHECK(kk_ia_view->planes[0].mtl_handle_input != NULL);
   CHECK(kk_ia_view->planes[0].input_gpu_resource_id != 0);
   CHECK(kk_ia_view->planes[0].input_2d_gpu_resource_id == 0);

   VkDescriptorSetLayoutBinding ia_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dsl_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &ia_binding,
   };
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   CHECK(fn_vkCreateDescriptorSetLayout(device, &dsl_info, NULL, &dsl) ==
         VK_SUCCESS);

   VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      .descriptorCount = 2,
   };
   VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   VkDescriptorPool desc_pool = VK_NULL_HANDLE;
   CHECK(fn_vkCreateDescriptorPool(device, &pool_info, NULL, &desc_pool) ==
         VK_SUCCESS);

   VkDescriptorSetLayout dsls[2] = {dsl, dsl};
   VkDescriptorSetAllocateInfo ds_alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = desc_pool,
      .descriptorSetCount = 2,
      .pSetLayouts = dsls,
   };
   VkDescriptorSet desc_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   CHECK(fn_vkAllocateDescriptorSets(device, &ds_alloc_info, desc_sets) ==
         VK_SUCCESS);

   VkDescriptorImageInfo opt_desc_img_info = {
      .imageView = ia_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkDescriptorImageInfo lin_desc_img_info = {
      .imageView = linear_ia_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = desc_sets[0],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
         .pImageInfo = &opt_desc_img_info,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = desc_sets[1],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
         .pImageInfo = &lin_desc_img_info,
      },
   };
   fn_vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

   /* 4. Verify MR !44990 + !45325 end-to-end GPU operations on a LINEAR color
    * attachment:
    *   (a) Dynamic rendering clear (COLOR_ATTACHMENT_BIT)
    *   (b) vkCmdClearColorImage internal color view path (kk_cmd_clear.c)
    *   (c) vkCmdBlitImage into the LINEAR image (BLIT_DST_BIT)
    *   (d) subpassLoad from a LINEAR input attachment via the 2D
    *       input_2d_gpu_resource_id descriptor path (when input attachment
    *       index is mapped to VK_ATTACHMENT_UNUSED so it executes the
    *       non-framebuffer-fetch lower_input_attachment_select 2D texel fetch)
    */
   void *mapped = NULL;
   CHECK(fn_vkMapMemory(device, linear_mem, 0, VK_WHOLE_SIZE, 0, &mapped) ==
         VK_SUCCESS);
   memset(mapped, 0, (size_t)mem_reqs.size);

   VkQueue queue = VK_NULL_HANDLE;
   fn_vkGetDeviceQueue(device, 0, 0, &queue);

   VkCommandPoolCreateInfo cmd_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cmd_pool = VK_NULL_HANDLE;
   CHECK(fn_vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool) ==
         VK_SUCCESS);

   VkCommandBufferAllocateInfo cmd_alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
   CHECK(fn_vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd_buf) ==
         VK_SUCCESS);

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(fn_vkBeginCommandBuffer(cmd_buf, &begin_info) == VK_SUCCESS);

   VkRenderingAttachmentInfo color_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = linear_view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {.float32 = {0.0f, 1.0f, 0.0f, 1.0f}}},
   };
   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {.offset = {0, 0}, .extent = {width, height}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_att,
   };
   fn_vkCmdBeginRendering(cmd_buf, &rendering_info);
   fn_vkCmdEndRendering(cmd_buf);
   CHECK(fn_vkEndCommandBuffer(cmd_buf) == VK_SUCCESS);

   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd_buf,
   };
   CHECK(fn_vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) ==
         VK_SUCCESS);
   CHECK(fn_vkQueueWaitIdle(queue) == VK_SUCCESS);

   VkImageSubresource subres = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .arrayLayer = 0,
   };
   VkSubresourceLayout subres_layout;
   fn_vkGetImageSubresourceLayout(device, linear_image, &subres,
                                  &subres_layout);

   const uint8_t *base = (const uint8_t *)mapped + subres_layout.offset;
   for (uint32_t y = 0; y < height; y++) {
      const uint32_t *row =
         (const uint32_t *)(base + y * subres_layout.rowPitch);
      for (uint32_t x = 0; x < width; x++) {
         CHECK(row[x] == 0xff00ff00u);
      }
   }

   /* (b) vkCmdClearColorImage on the LINEAR image and (c) vkCmdBlitImage from
    * an OPTIMAL source image into the bottom half of the LINEAR destination
    * image.
    */
   VkImageView opt_color_view = VK_NULL_HANDLE;
   VkImageViewCreateInfo opt_color_view_info = linear_view_info;
   opt_color_view_info.image = opt_ia_image;
   opt_color_view_info.pNext = &color_only_usage;
   CHECK(fn_vkCreateImageView(device, &opt_color_view_info, NULL,
                              &opt_color_view) == VK_SUCCESS);

   CHECK(fn_vkResetCommandBuffer(cmd_buf, 0) == VK_SUCCESS);
   CHECK(fn_vkBeginCommandBuffer(cmd_buf, &begin_info) == VK_SUCCESS);

   /* Clear OPTIMAL image to red (1, 0, 0, 1) via dynamic rendering */
   VkRenderingAttachmentInfo opt_clear_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = opt_color_view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {.float32 = {1.0f, 0.0f, 0.0f, 1.0f}}},
   };
   VkRenderingInfo opt_render_info = rendering_info;
   opt_render_info.pColorAttachments = &opt_clear_att;
   fn_vkCmdBeginRendering(cmd_buf, &opt_render_info);
   fn_vkCmdEndRendering(cmd_buf);

   /* Clear LINEAR image to blue (0, 0, 1, 1) via vkCmdClearColorImage */
   VkClearColorValue blue_clear = {.float32 = {0.0f, 0.0f, 1.0f, 1.0f}};
   VkImageSubresourceRange clear_range = linear_view_info.subresourceRange;
   fn_vkCmdClearColorImage(cmd_buf, linear_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &blue_clear, 1,
                           &clear_range);

   /* Blit OPTIMAL red image into the bottom half (y in [height/2, height)) of
    * the LINEAR image (exercises VK_FORMAT_FEATURE_2_BLIT_DST_BIT on LINEAR).
    */
   VkImageBlit blit_region = {
      .srcSubresource =
         {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      .srcOffsets = {{0, 0, 0}, {(int32_t)width, (int32_t)(height / 2), 1}},
      .dstSubresource =
         {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      .dstOffsets =
         {
            {0, (int32_t)(height / 2), 0},
            {(int32_t)width, (int32_t)height, 1},
         },
   };
   fn_vkCmdBlitImage(
      cmd_buf, opt_ia_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear_image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region, VK_FILTER_NEAREST);
   CHECK(fn_vkEndCommandBuffer(cmd_buf) == VK_SUCCESS);

   CHECK(fn_vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) ==
         VK_SUCCESS);
   CHECK(fn_vkQueueWaitIdle(queue) == VK_SUCCESS);

   for (uint32_t y = 0; y < height; y++) {
      const uint32_t *row =
         (const uint32_t *)(base + y * subres_layout.rowPitch);
      uint32_t expected = (y < height / 2) ? 0xffff0000u : 0xff0000ffu;
      for (uint32_t x = 0; x < width; x++) {
         CHECK(row[x] == expected);
      }
   }

   /* (d) End-to-end subpassLoad from linear_ia_view (2D input attachment view
    * on a LINEAR image) into linear_view (LINEAR color attachment).
    * Fill linear_ia_image with a per-pixel gradient in host memory, map the
    * input attachment index to VK_ATTACHMENT_UNUSED so the shader executes the
    * descriptor-backed 2D texel fetch branch (lower_input_attachment_select),
    * draw a fullscreen triangle into linear_view, and verify every pixel.
    */
   void *mapped_ia = NULL;
   CHECK(fn_vkMapMemory(device, linear_ia_mem, 0, VK_WHOLE_SIZE, 0,
                        &mapped_ia) == VK_SUCCESS);
   VkSubresourceLayout ia_subres_layout;
   fn_vkGetImageSubresourceLayout(device, linear_ia_image, &subres,
                                  &ia_subres_layout);
   uint8_t *ia_base = (uint8_t *)mapped_ia + ia_subres_layout.offset;
   for (uint32_t y = 0; y < height; y++) {
      uint32_t *row = (uint32_t *)(ia_base + y * ia_subres_layout.rowPitch);
      for (uint32_t x = 0; x < width; x++) {
         uint8_t r = (uint8_t)(x * 15u + 3u);
         uint8_t g = (uint8_t)(y * 13u + 7u);
         uint8_t b = (uint8_t)((x + y) * 7u + 11u);
         row[x] = 0xff000000u | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r;
      }
   }

   VkShaderModuleCreateInfo vs_mod_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vs_spv),
      .pCode = vs_spv,
   };
   VkShaderModuleCreateInfo fs_mod_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(fs_spv),
      .pCode = fs_spv,
   };
   VkShaderModule vs_mod = VK_NULL_HANDLE, fs_mod = VK_NULL_HANDLE;
   CHECK(fn_vkCreateShaderModule(device, &vs_mod_info, NULL, &vs_mod) ==
         VK_SUCCESS);
   CHECK(fn_vkCreateShaderModule(device, &fs_mod_info, NULL, &fs_mod) ==
         VK_SUCCESS);

   VkPipelineLayoutCreateInfo pl_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl_layout = VK_NULL_HANDLE;
   CHECK(fn_vkCreatePipelineLayout(device, &pl_layout_info, NULL, &pl_layout) ==
         VK_SUCCESS);

   uint32_t unused_ia_idx = VK_ATTACHMENT_UNUSED;
   VkRenderingInputAttachmentIndexInfo ia_idx_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentInputIndices = &unused_ia_idx,
   };
   VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo pl_rendering_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .pNext = &ia_idx_info,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_fmt,
   };
   VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs_mod,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs_mod,
         .pName = "main",
      },
   };
   VkPipelineVertexInputStateCreateInfo vi_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport viewport = {0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f};
   VkRect2D scissor = {{0, 0}, {width, height}};
   VkPipelineViewportStateCreateInfo vp_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   VkPipelineRasterizationStateCreateInfo rs_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cb_att = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cb_att,
   };
   VkGraphicsPipelineCreateInfo gp_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &pl_rendering_info,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi_state,
      .pInputAssemblyState = &ia_state,
      .pViewportState = &vp_state,
      .pRasterizationState = &rs_state,
      .pMultisampleState = &ms_state,
      .pColorBlendState = &cb_state,
      .layout = pl_layout,
   };
   VkPipeline pipeline = VK_NULL_HANDLE;
   CHECK(fn_vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_info, NULL,
                                      &pipeline) == VK_SUCCESS);

   CHECK(fn_vkResetCommandBuffer(cmd_buf, 0) == VK_SUCCESS);
   CHECK(fn_vkBeginCommandBuffer(cmd_buf, &begin_info) == VK_SUCCESS);
   fn_vkCmdBeginRendering(cmd_buf, &rendering_info);
   fn_vkCmdSetRenderingInputAttachmentIndices(cmd_buf, &ia_idx_info);
   fn_vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   fn_vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pl_layout, 0, 1, &desc_sets[1], 0, NULL);
   fn_vkCmdDraw(cmd_buf, 3, 1, 0, 0);
   fn_vkCmdEndRendering(cmd_buf);
   CHECK(fn_vkEndCommandBuffer(cmd_buf) == VK_SUCCESS);

   CHECK(fn_vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) ==
         VK_SUCCESS);
   CHECK(fn_vkQueueWaitIdle(queue) == VK_SUCCESS);

   for (uint32_t y = 0; y < height; y++) {
      const uint32_t *row =
         (const uint32_t *)(base + y * subres_layout.rowPitch);
      for (uint32_t x = 0; x < width; x++) {
         uint8_t r = (uint8_t)(x * 15u + 3u);
         uint8_t g = (uint8_t)(y * 13u + 7u);
         uint8_t b = (uint8_t)((x + y) * 7u + 11u);
         uint32_t expected =
            0xff000000u | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r;
         CHECK(row[x] == expected);
      }
   }

   fn_vkDestroyPipeline(device, pipeline, NULL);
   fn_vkDestroyPipelineLayout(device, pl_layout, NULL);
   fn_vkDestroyShaderModule(device, vs_mod, NULL);
   fn_vkDestroyShaderModule(device, fs_mod, NULL);
   fn_vkUnmapMemory(device, linear_ia_mem);
   fn_vkDestroyImageView(device, linear_ia_view, NULL);
   fn_vkDestroyImageView(device, color_only_view, NULL);
   fn_vkDestroyImage(device, linear_ia_image, NULL);
   fn_vkFreeMemory(device, linear_ia_mem, NULL);
   fn_vkDestroyImageView(device, opt_color_view, NULL);
   fn_vkUnmapMemory(device, linear_mem);
   fn_vkDestroyCommandPool(device, cmd_pool, NULL);
   fn_vkDestroyDescriptorPool(device, desc_pool, NULL);
   fn_vkDestroyDescriptorSetLayout(device, dsl, NULL);
   fn_vkDestroyImageView(device, ia_view, NULL);
   fn_vkDestroyImage(device, opt_ia_image, NULL);
   fn_vkFreeMemory(device, opt_mem, NULL);
   fn_vkDestroyImageView(device, linear_view, NULL);
   fn_vkDestroyImage(device, linear_image, NULL);
   fn_vkFreeMemory(device, linear_mem, NULL);
   fn_vkDestroyDevice(device, NULL);
   fn_vkDestroyInstance(instance, NULL);
   dlclose(library);
   return 0;
}
