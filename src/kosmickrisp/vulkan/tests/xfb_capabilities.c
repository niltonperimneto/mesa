/* SPDX-License-Identifier: MIT */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

int
main(int argc, char **argv)
{
   if (argc != 3)
      return 1;
   bool research = strcmp(argv[2], "research") == 0;
   void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
   if (!library) {
      fprintf(stderr, "%s\n", dlerror());
      return 1;
   }
   PFN_vkGetInstanceProcAddr get = dlsym(library, "vk_icdGetInstanceProcAddr");
   if (!get)
      return 1;
   PFN_vkCreateInstance create =
      (PFN_vkCreateInstance)get(NULL, "vkCreateInstance");
   if (!create)
      return 1;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app};
   VkInstance instance;
   if (create(&info, NULL, &instance) != VK_SUCCESS)
      return 1;
   PFN_vkDestroyInstance destroy =
      (PFN_vkDestroyInstance)get(instance, "vkDestroyInstance");
   PFN_vkEnumeratePhysicalDevices enumerate =
      (PFN_vkEnumeratePhysicalDevices)get(instance,
                                          "vkEnumeratePhysicalDevices");
   PFN_vkEnumerateDeviceExtensionProperties extensions =
      (PFN_vkEnumerateDeviceExtensionProperties)get(
         instance, "vkEnumerateDeviceExtensionProperties");
   PFN_vkGetPhysicalDeviceFeatures2 features =
      (PFN_vkGetPhysicalDeviceFeatures2)get(instance,
                                            "vkGetPhysicalDeviceFeatures2");
   if (!destroy || !enumerate || !extensions || !features)
      return 1;
   uint32_t count = 0;
   VkResult result = enumerate(instance, &count, NULL);
   if (result == VK_SUCCESS && !count) {
      destroy(instance, NULL);
      return 77;
   }
   if (result != VK_SUCCESS)
      return 1;
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices || enumerate(instance, &count, devices) != VK_SUCCESS)
      return 1;
   uint32_t extension_count = 0;
   if (extensions(devices[0], NULL, &extension_count, NULL) != VK_SUCCESS)
      return 1;
   VkExtensionProperties *properties =
      calloc(extension_count, sizeof(*properties));
   if (!properties ||
       extensions(devices[0], NULL, &extension_count, properties) != VK_SUCCESS)
      return 1;
   bool xfb = false, gpl = false, shader_object = false;
   for (uint32_t i = 0; i < extension_count; ++i) {
      xfb |= strcmp(properties[i].extensionName,
                    VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME) == 0;
      gpl |= strcmp(properties[i].extensionName,
                    VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) == 0;
      shader_object |= strcmp(properties[i].extensionName,
                              VK_EXT_SHADER_OBJECT_EXTENSION_NAME) == 0;
   }
   VkPhysicalDeviceShaderObjectFeaturesEXT so = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
      .pNext = &so};
   VkPhysicalDeviceTransformFeedbackFeaturesEXT feedback = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
      .pNext = &library_features};
   VkPhysicalDeviceFeatures2 queried = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &feedback};
   features(devices[0], &queried);
   bool valid = xfb == research && gpl != research &&
                shader_object != research &&
                !!feedback.transformFeedback == research &&
                !!feedback.geometryStreams == research &&
                !!library_features.graphicsPipelineLibrary != research &&
                !!so.shaderObject != research;
   free(properties);
   free(devices);
   destroy(instance, NULL);
   dlclose(library);
   if (!valid)
      fprintf(stderr, "Inconsistent production/research capability contract\n");
   return valid ? 0 : 1;
}
