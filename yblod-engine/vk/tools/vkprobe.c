// Probe: what the Vulkan DV engine needs on this box (read-only, headless).
// Build: gcc -O1 -o vkprobe vkprobe.c $(pkg-config --cflags --libs libdrm libva libva-drm) -lvulkan
// Results for the N150 are in docs/vulkan-engine.md, "Capabilities on the N150".
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include <xf86drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_) { printf("%s -> %d\n", #x, r_); return 1; } } while (0)

static void mods(VkPhysicalDevice pd, VkFormat f, const char *name)
{
    VkDrmFormatModifierPropertiesListEXT list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
    VkFormatProperties2 p = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &list};
    vkGetPhysicalDeviceFormatProperties2(pd, f, &p);
    VkDrmFormatModifierPropertiesEXT m[64];
    list.pDrmFormatModifierProperties = m;
    if (list.drmFormatModifierCount > 64) list.drmFormatModifierCount = 64;
    vkGetPhysicalDeviceFormatProperties2(pd, f, &p);
    printf("format %s (%d): optimal 0x%x linear 0x%x\n", name, f,
           p.formatProperties.optimalTilingFeatures, p.formatProperties.linearTilingFeatures);
    for (uint32_t i = 0; i < list.drmFormatModifierCount; i++)
        printf("  mod 0x%016llx planes %u features 0x%x%s%s\n",
               (unsigned long long)m[i].drmFormatModifier, m[i].drmFormatModifierPlaneCount,
               m[i].drmFormatModifierTilingFeatures,
               (m[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? " STORAGE" : "",
               (m[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? " SAMPLED" : "");
}

int main(void)
{
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 4;
    VkPhysicalDevice pds[4];
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &drv};
    VkPhysicalDeviceVulkan13Properties p13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, &sg};
    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
    vkGetPhysicalDeviceProperties2(pd, &props);
    printf("device %s api %u.%u.%u driver %s %s\n", props.properties.deviceName,
           VK_API_VERSION_MAJOR(props.properties.apiVersion), VK_API_VERSION_MINOR(props.properties.apiVersion),
           VK_API_VERSION_PATCH(props.properties.apiVersion), drv.driverName, drv.driverInfo);
    printf("subgroup size %u stages 0x%x ops 0x%x; min/max subgroup %u/%u; maxComputeSharedMemory %u; maxWorkgroupInvocations %u\n",
           sg.subgroupSize, sg.supportedStages, sg.supportedOperations, p13.minSubgroupSize, p13.maxSubgroupSize,
           props.properties.limits.maxComputeSharedMemorySize, props.properties.limits.maxComputeWorkGroupInvocations);
    printf("timestampPeriod %f, timestampComputeAndGraphics %u\n", props.properties.limits.timestampPeriod,
           props.properties.limits.timestampComputeAndGraphics);

    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f);
    printf("int64 %u int16 %u float64 %u sb16 %u sb8 %u int8 %u timeline %u bda %u vkmm %u shaderFloat16 %u\n",
           f.features.shaderInt64, f.features.shaderInt16, f.features.shaderFloat64,
           f11.storageBuffer16BitAccess, f12.storageBuffer8BitAccess, f12.shaderInt8,
           f12.timelineSemaphore, f12.bufferDeviceAddress, f12.vulkanMemoryModel, f12.shaderFloat16);
    VkPhysicalDeviceFloatControlsProperties fc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &fc};
    vkGetPhysicalDeviceProperties2(pd, &props2);
    printf("float controls: denormPreserve32 %u RTE32 %u signedZeroInfNan32 %u\n",
           fc.shaderDenormPreserveFloat32, fc.shaderRoundingModeRTEFloat32, fc.shaderSignedZeroInfNanPreserveFloat32);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne, sizeof(*ext));
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
    const char *want[] = {"VK_EXT_external_memory_dma_buf", "VK_EXT_image_drm_format_modifier",
        "VK_KHR_external_semaphore_fd", "VK_KHR_external_memory_fd", "VK_EXT_queue_family_foreign",
        "VK_KHR_shader_float_controls", "VK_EXT_physical_device_drm", "VK_KHR_video_decode_queue",
        "VK_KHR_video_decode_h265", "VK_KHR_shader_subgroup_extended_types", "VK_EXT_shader_atomic_float",
        "VK_KHR_shader_float_controls2", "VK_EXT_external_memory_host", "VK_KHR_8bit_storage", NULL};
    for (int w = 0; want[w]; w++) {
        int found = 0;
        for (uint32_t i = 0; i < ne; i++) if (!strcmp(ext[i].extensionName, want[w])) found = 1;
        printf("ext %-40s %s\n", want[w], found ? "yes" : "NO");
    }

    uint32_t nq = 8;
    VkQueueFamilyProperties qf[8];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    for (uint32_t i = 0; i < nq; i++)
        printf("queue family %u flags 0x%x count %u ts bits %u\n", i, qf[i].queueFlags, qf[i].queueCount, qf[i].timestampValidBits);

    VkExternalSemaphoreHandleTypeFlagBits types[] = {VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                                                     VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
    for (int t = 0; t < 2; t++) for (int timeline = 0; timeline < 2; timeline++) {
        VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY};
        VkPhysicalDeviceExternalSemaphoreInfo si = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO, &st, types[t]};
        VkExternalSemaphoreProperties sp = {VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
        vkGetPhysicalDeviceExternalSemaphoreProperties(pd, &si, &sp);
        printf("semaphore %s %s: export 0x%x compat 0x%x features 0x%x\n", t ? "OPAQUE_FD" : "SYNC_FD",
               timeline ? "timeline" : "binary", sp.exportFromImportedHandleTypes, sp.compatibleHandleTypes, sp.externalSemaphoreFeatures);
    }

    mods(pd, VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16, "P010 2-plane");
    mods(pd, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, "NV12 2-plane");
    mods(pd, VK_FORMAT_R16_UNORM, "R16_UNORM");
    mods(pd, VK_FORMAT_R16_UINT, "R16_UINT");
    mods(pd, VK_FORMAT_R16G16_UINT, "R16G16_UINT");
    mods(pd, VK_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM");
    mods(pd, VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM");
    mods(pd, VK_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT");
    mods(pd, VK_FORMAT_R32_UINT, "R32_UINT");

    // VA: what a decoder P010 surface exports as.
    int fd = open("/dev/dri/renderD128", O_RDWR);
    uint64_t cap = 0;
    printf("DRM_CAP_SYNCOBJ %d, ", drmGetCap(fd, DRM_CAP_SYNCOBJ, &cap) ? -1 : (int)cap);
    printf("DRM_CAP_SYNCOBJ_TIMELINE %d\n", drmGetCap(fd, DRM_CAP_SYNCOBJ_TIMELINE, &cap) ? -1 : (int)cap);
    VADisplay va = vaGetDisplayDRM(fd);
    int major, minor;
    if (vaInitialize(va, &major, &minor)) { printf("vaInitialize failed\n"); return 1; }
    printf("VA %d.%d %s\n", major, minor, vaQueryVendorString(va));
    unsigned fourccs[2] = {VA_FOURCC_P010, VA_FOURCC_NV12};
    unsigned rts[2] = {VA_RT_FORMAT_YUV420_10, VA_RT_FORMAT_YUV420};
    for (int k = 0; k < 2; k++) {
        VASurfaceAttrib attr[2] = {
            {VASurfaceAttribPixelFormat, VA_SURFACE_ATTRIB_SETTABLE, {VAGenericValueTypeInteger, {.i = (int)fourccs[k]}}},
            {VASurfaceAttribUsageHint, VA_SURFACE_ATTRIB_SETTABLE, {VAGenericValueTypeInteger, {.i = VA_SURFACE_ATTRIB_USAGE_HINT_DECODER}}},
        };
        VASurfaceID s;
        if (vaCreateSurfaces(va, rts[k], 3840, 2160, &s, 1, attr, 2)) { printf("vaCreateSurfaces failed\n"); continue; }
        for (int layers = 0; layers < 2; layers++) {
            VADRMPRIMESurfaceDescriptor d;
            uint32_t flags = VA_EXPORT_SURFACE_READ_ONLY | (layers ? VA_EXPORT_SURFACE_SEPARATE_LAYERS : VA_EXPORT_SURFACE_COMPOSED_LAYERS);
            if (vaExportSurfaceHandle(va, s, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2, flags, &d)) { printf("export failed\n"); continue; }
            printf("VA %s %s: %ux%u fourcc %.4s objects %u layers %u\n", k ? "NV12" : "P010", layers ? "separate" : "composed",
                   d.width, d.height, (char *)&d.fourcc, d.num_objects, d.num_layers);
            for (uint32_t o = 0; o < d.num_objects; o++)
                printf("  object %u size %u modifier 0x%016llx\n", o, d.objects[o].size, (unsigned long long)d.objects[o].drm_format_modifier);
            for (uint32_t l = 0; l < d.num_layers; l++) {
                printf("  layer %u format %.4s planes %u:", l, (char *)&d.layers[l].drm_format, d.layers[l].num_planes);
                for (uint32_t p = 0; p < d.layers[l].num_planes; p++)
                    printf(" [obj %u off %u pitch %u]", d.layers[l].object_index[p], d.layers[l].offset[p], d.layers[l].pitch[p]);
                printf("\n");
            }
            for (uint32_t o = 0; o < d.num_objects; o++) close(d.objects[o].fd);
        }
        vaDestroySurfaces(va, &s, 1);
    }
    vaTerminate(va);
    return 0;
}
