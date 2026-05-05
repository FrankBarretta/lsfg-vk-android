#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "core/semaphore.hpp"
#include "core/device.hpp"
#include "common/exception.hpp"

#include <optional>
#include <cstdint>
#include <memory>
#include <stdexcept>

using namespace LSFG::Core;

Semaphore::Semaphore(const Core::Device& device, std::optional<uint32_t> initial) {
    // Standard Timeline Semaphore Creation
    const VkSemaphoreTypeCreateInfo typeInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = initial.value_or(0)
    };
    const VkSemaphoreCreateInfo desc{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = initial.has_value() ? &typeInfo : nullptr,
        .flags = 0
    };

    VkSemaphore semaphoreHandle{};
    auto res = vkCreateSemaphore(device.handle(), &desc, nullptr, &semaphoreHandle);
    if (res != VK_SUCCESS || semaphoreHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Unable to create semaphore");

    this->isTimeline = initial.has_value();
    this->semaphore = std::shared_ptr<VkSemaphore>(
        new VkSemaphore(semaphoreHandle),
        [dev = device.handle()](VkSemaphore* h) {
            vkDestroySemaphore(dev, *h, nullptr);
            delete h;
        }
    );
}

Semaphore::Semaphore(const Core::Device& device, int fd) {
    // 1. Create a Binary Semaphore specifically for External Sync FD
    const VkExportSemaphoreCreateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
    };

    const VkSemaphoreCreateInfo desc{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &exportInfo,
        .flags = 0
    };

    VkSemaphore semaphoreHandle{};
    auto res = vkCreateSemaphore(device.handle(), &desc, nullptr, &semaphoreHandle);
    if (res != VK_SUCCESS || semaphoreHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Unable to create binary semaphore for FD import");

    // 2. Import the FD using the TEMPORARY flag
    // On Android/Turnip, SYNC_FD imports MUST be temporary to replace the payload
    auto vkImportSemaphoreFdKHR = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device.handle(), "vkImportSemaphoreFdKHR"));

    if (!vkImportSemaphoreFdKHR)
        throw std::runtime_error("Vulkan: Could not find vkImportSemaphoreFdKHR");

    const VkImportSemaphoreFdInfoKHR importInfo{
        .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
        .pNext = nullptr,
        .semaphore = semaphoreHandle,
        .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = fd // The driver takes ownership and will close(fd)
    };

    res = vkImportSemaphoreFdKHR(device.handle(), &importInfo);
    if (res != VK_SUCCESS) {
        vkDestroySemaphore(device.handle(), semaphoreHandle, nullptr);
        throw LSFG::vulkan_error(res, "Unable to import semaphore from fd");
    }

    this->isTimeline = false;
    this->semaphore = std::shared_ptr<VkSemaphore>(
        new VkSemaphore(semaphoreHandle),
        [dev = device.handle()](VkSemaphore* h) {
            vkDestroySemaphore(dev, *h, nullptr);
            delete h;
        }
    );
}

// Added Export function so LSFG can send a signal back to the Layer
int Semaphore::exportFd(const Core::Device& device) const {
    auto vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device.handle(), "vkGetSemaphoreFdKHR"));

    if (!vkGetSemaphoreFdKHR) return -1;

    int fd = -1;
    const VkSemaphoreGetFdInfoKHR getInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .pNext = nullptr,
        .semaphore = this->handle(),
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
    };

    auto res = vkGetSemaphoreFdKHR(device.handle(), &getInfo, &fd);
    return (res == VK_SUCCESS) ? fd : -1;
}

void Semaphore::signal(const Core::Device& device, uint64_t value) const {
    if (!this->isTimeline)
        throw std::logic_error("Cannot signal a binary semaphore via signal() - use Queue submission");

    const VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = this->handle(),
        .value = value
    };
    auto res = vkSignalSemaphore(device.handle(), &signalInfo);
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Unable to signal timeline semaphore");
}

bool Semaphore::wait(const Core::Device& device, uint64_t value, uint64_t timeout) const {
    if (!this->isTimeline)
        throw std::logic_error("Cannot wait on a binary semaphore via wait() - use Queue wait");

    VkSemaphore sem = this->handle();
    const VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &sem,
        .pValues = &value
    };
    auto res = vkWaitSemaphores(device.handle(), &waitInfo, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
        throw LSFG::vulkan_error(res, "Unable to wait for timeline semaphore");

    return res == VK_SUCCESS;
}
