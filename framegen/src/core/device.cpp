#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "core/device.hpp"
#include "core/image.hpp"
#include "core/instance.hpp"
#include "common/exception.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LSFG_FRAMEGEN_LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, "lsfg-vk-framegen", __VA_ARGS__)
#define LSFG_FRAMEGEN_LOGW(...) \
    __android_log_print(ANDROID_LOG_WARN, "lsfg-vk-framegen", __VA_ARGS__)
#else
#define LSFG_FRAMEGEN_LOGI(...) do {} while (0)
#define LSFG_FRAMEGEN_LOGW(...) do {} while (0)
#endif

// Build-fingerprint stamp. The string changes whenever this file is edited
// (via __DATE__/__TIME__), which forces NDK/CMake to detect a real source-level
// change rather than reusing the cached `lsfg-vk-framegen.a`. The variable is
// referenced from Device::Device so the linker can't dead-strip it; the value
// also lands in the runtime log so users/devs can confirm a fresh build is on
// the device by grepping for "framegen build stamp".
namespace LSFG::Core {
extern const char* kFramegenBuildStamp;
const char* kFramegenBuildStamp =
    "framegen build stamp: " __DATE__ " " __TIME__;
}

using namespace LSFG::Core;

const std::vector<const char*> requiredExtensions = {
#ifndef __ANDROID__
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore_fd",
#else
    // On Android we share via AHardwareBuffer, not opaque FDs.
    "VK_ANDROID_external_memory_android_hardware_buffer",
    "VK_KHR_external_memory",                  // base ext, dependency
    "VK_KHR_sampler_ycbcr_conversion",         // dependency of AHB ext
    "VK_KHR_dedicated_allocation",             // required for dedicated AHB import
    "VK_KHR_get_memory_requirements2",         // dependency
    "VK_KHR_bind_memory2",                     // dependency
    "VK_KHR_maintenance1",                     // dependency
#endif
};

namespace {

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const auto& extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) return true;
    }
    return false;
}

} // namespace

const Image& Device::getFallbackDescriptorImage() const {
    return *this->fallbackDescriptorImage;
}

Device::Device(const Instance& instance, uint64_t deviceUUID) {
    // First line of any framegen device init: prints the build stamp at INFO.
    // Functionality-irrelevant by design — its only purpose is to give field
    // testers a single grep-able marker that proves the loaded .so contains
    // the storage-image / DEVICE_LOST fix series. If the user hits DEVICE_LOST
    // and this line is absent from logcat, their APK was linked against a
    // stale cached `lsfg-vk-framegen.a` and a clean rebuild is required.
    LSFG_FRAMEGEN_LOGI("Entering Device::Device — %s", kFramegenBuildStamp);

    // get all physical devices
    uint32_t deviceCount{};
    auto res = vkEnumeratePhysicalDevices(instance.handle(), &deviceCount, nullptr);
    if (res != VK_SUCCESS || deviceCount == 0)
        throw LSFG::vulkan_error(res, "Failed to enumerate physical devices");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    res = vkEnumeratePhysicalDevices(instance.handle(), &deviceCount, devices.data());
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Failed to get physical devices");

    // get device by uuid
    std::optional<VkPhysicalDevice> physicalDevice;
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        const uint64_t uuid =
            static_cast<uint64_t>(properties.vendorID) << 32 | properties.deviceID;
        if (deviceUUID == uuid || deviceUUID == 0x1463ABAC) {
            physicalDevice = device;
            break;
        }
    }
    if (!physicalDevice)
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
            "Could not find physical device with UUID");

    // find queue family indices
    uint32_t familyCount{};
    vkGetPhysicalDeviceQueueFamilyProperties(*physicalDevice, &familyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(*physicalDevice, &familyCount, queueFamilies.data());

    std::optional<uint32_t> computeFamilyIdx;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            computeFamilyIdx = i;
    }
    if (!computeFamilyIdx)
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "No compute queue family found");

    uint32_t extensionCount{};
    res = vkEnumerateDeviceExtensionProperties(*physicalDevice, nullptr, &extensionCount, nullptr);
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Failed to enumerate device extensions");
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    res = vkEnumerateDeviceExtensionProperties(*physicalDevice, nullptr,
        &extensionCount, availableExtensions.data());
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Failed to get device extensions");

    std::vector<const char*> enabledExtensions;
    enabledExtensions.reserve(requiredExtensions.size() + 1);
    for (const char* extension : requiredExtensions) {
        if (!hasExtension(availableExtensions, extension)) {
            throw LSFG::vulkan_error(VK_ERROR_EXTENSION_NOT_PRESENT,
                std::string("Missing required device extension: ") + extension);
        }
        enabledExtensions.push_back(extension);
    }

    const bool hasRobustness2 =
        hasExtension(availableExtensions, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    if (hasRobustness2) {
        enabledExtensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    }

    // Probe FP16 support on this physical device. The LSFG-Android port can
    // load precompiled SPIR-V FP16 shader variants from Lossless.dll (resource
    // IDs 304..351) which carry `OpCapability Float16`. Vulkan rejects those at
    // vkCreateShaderModule time unless the device was created with the
    // shaderFloat16 feature explicitly enabled. We probe and unconditionally
    // enable it when supported — there's no downside on FP32-only sessions and
    // it lets the FP16 path "just work" when the user toggles it on.
    //
    // Also probe the core 1.0 storage-image features. The LSFG compute shader
    // chain reads from and writes to R16G16B16A16_SFLOAT storage images
    // (gamma/delta/generate); on Mali (Bifrost/Valhall) this is rejected at
    // dispatch time and surfaces as VK_ERROR_DEVICE_LOST on the first present
    // unless `shaderStorageImageExtendedFormats` is explicitly enabled at
    // device-create time. The same applies to image read/write without an
    // explicit `format` qualifier in SPIR-V — without
    // `shaderStorageImageReadWithoutFormat` / `WriteWithoutFormat` the driver
    // is allowed to UB the dispatch. Enable each only when the physical device
    // advertises it (the validation layers reject create_device with features
    // the device doesn't support, and several PowerVR/Adreno revisions only
    // expose a subset).
    VkPhysicalDeviceShaderFloat16Int8Features fp16Probe{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    VkPhysicalDeviceFeatures2 featsProbe{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &fp16Probe,
    };
    vkGetPhysicalDeviceFeatures2(*physicalDevice, &featsProbe);
    const bool hasFloat16 = fp16Probe.shaderFloat16 == VK_TRUE;
    const VkPhysicalDeviceFeatures& probedCore = featsProbe.features;

    // Build the core feature struct we'll request. We must only request what
    // the device supports — anything else fails vkCreateDevice. All four
    // fields below are required by the LSFG shader chain on Mali; absence
    // would manifest as DEVICE_LOST on first dispatch.
    VkPhysicalDeviceFeatures enabledCoreFeatures{};
    enabledCoreFeatures.shaderStorageImageExtendedFormats =
        probedCore.shaderStorageImageExtendedFormats;
    enabledCoreFeatures.shaderStorageImageReadWithoutFormat =
        probedCore.shaderStorageImageReadWithoutFormat;
    enabledCoreFeatures.shaderStorageImageWriteWithoutFormat =
        probedCore.shaderStorageImageWriteWithoutFormat;
    // shaderInt16 commonly accompanies FP16 paths; harmless on FP32-only sessions
    // and required by some Lossless.dll FP16 shader variants.
    enabledCoreFeatures.shaderInt16 = probedCore.shaderInt16;

    // Diagnostic: log which storage-image features were probed vs. enabled.
    // Tagged "lsfg-vk-framegen" so it's easy to correlate with session logs.
    // On Mali-G57 this is the load-bearing line: if any of the *Format* fields
    // shows probed=0, the device cannot legally execute LSFG's compute chain
    // and presentContext will hit DEVICE_LOST regardless of this fix.
    //
    // The build stamp below is referenced (and printed) here so the linker
    // can't dead-strip it and so users running an older cached .so can verify
    // by log-grep whether the build they have actually contains this fix.
    LSFG_FRAMEGEN_LOGI("%s", kFramegenBuildStamp);
    LSFG_FRAMEGEN_LOGI(
        "Device features probe: storageImageExtendedFormats=%d, "
        "storageImageReadWithoutFormat=%d, storageImageWriteWithoutFormat=%d, "
        "shaderInt16=%d, shaderFloat16=%d, robustness2=%d",
        (int)probedCore.shaderStorageImageExtendedFormats,
        (int)probedCore.shaderStorageImageReadWithoutFormat,
        (int)probedCore.shaderStorageImageWriteWithoutFormat,
        (int)probedCore.shaderInt16,
        (int)hasFloat16,
        (int)hasRobustness2);
    if (probedCore.shaderStorageImageExtendedFormats != VK_TRUE) {
        // Hard warning: this is the feature LSFG's R16G16B16A16_SFLOAT storage
        // image accesses depend on. Without it, dispatch is undefined behavior
        // on Mali/Adreno and surfaces as DEVICE_LOST on first present. There's
        // no shader-side workaround we can apply at runtime; logging it loudly
        // gives upstream / users a clear pointer to the actual root cause.
        LSFG_FRAMEGEN_LOGW(
            "shaderStorageImageExtendedFormats=FALSE — LSFG compute chain "
            "expects R16G16B16A16_SFLOAT storage; first dispatch will likely "
            "DEVICE_LOST on this device");
    }

    // create logical device
    const float queuePriority{1.0F}; // highest priority
    VkPhysicalDeviceRobustness2FeaturesEXT robustness2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
        .nullDescriptor = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = hasRobustness2 ? &robustness2 : nullptr,
        .synchronization2 = VK_TRUE
    };
    // shaderFloat16 is exposed in core Vulkan 1.2 — same struct we already
    // chain. Setting it conditionally avoids regressing devices that don't
    // advertise the feature (the validation layers reject create_device when
    // requested features are unsupported).
    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &features13,
        .shaderFloat16 = hasFloat16 ? VK_TRUE : VK_FALSE,
        .timelineSemaphore = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE
    };
    // Use VkPhysicalDeviceFeatures2 in pNext (mutually exclusive with
    // pEnabledFeatures per spec) so we can chain the core features alongside
    // the 1.2/1.3/robustness2 structs above.
    VkPhysicalDeviceFeatures2 enabledFeatures2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features12,
        .features = enabledCoreFeatures,
    };
    const VkDeviceQueueCreateInfo computeQueueDesc{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = *computeFamilyIdx,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };
    const VkDeviceCreateInfo deviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabledFeatures2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &computeQueueDesc,
        .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data()
        // pEnabledFeatures intentionally NULL: enabledFeatures2 carries them
        // via pNext (the spec forbids both being non-null).
    };
    VkDevice deviceHandle{};
    res = vkCreateDevice(*physicalDevice, &deviceCreateInfo, nullptr, &deviceHandle);
    if (res != VK_SUCCESS || deviceHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Failed to create logical device");

    volkLoadDevice(deviceHandle);

    // get compute queue
    VkQueue queueHandle{};
    vkGetDeviceQueue(deviceHandle, *computeFamilyIdx, 0, &queueHandle);

    // store in shared ptr
    this->computeQueue = queueHandle;
    this->computeFamilyIdx = *computeFamilyIdx;
    this->physicalDevice = *physicalDevice;
    this->nullDescriptorSupported = hasRobustness2;
    this->device = std::shared_ptr<VkDevice>(
        new VkDevice(deviceHandle),
        [](VkDevice* device) {
            vkDestroyDevice(*device, nullptr);
        }
    );
    if (!this->nullDescriptorSupported) {
        this->fallbackDescriptorImage = std::make_shared<Core::Image>(*this,
            VkExtent2D{1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    }
}
