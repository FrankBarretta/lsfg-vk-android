#include "mini/semaphore.hpp"
#include "common/exception.hpp"
#include "layer.hpp"

#include <vulkan/vulkan_core.h>

#include <memory>

using namespace Mini ;

Semaphore::Semaphore(VkDevice device) {
    const VkExportSemaphoreCreateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
    };
    const VkSemaphoreCreateInfo desc{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &exportInfo
    };

    VkSemaphore semaphoreHandle{};
    auto res = Layer::ovkCreateSemaphore(device, &desc, nullptr, &semaphoreHandle);

    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Unable to create binary semaphore");

    this->semaphore = std::shared_ptr<VkSemaphore>(
        new VkSemaphore(semaphoreHandle),
        [dev = device](VkSemaphore* h) {
            Layer::ovkDestroySemaphore(dev, *h, nullptr);
        }
    );
}

void Semaphore::exportSyncFd(VkDevice device, int* fd) {
    const VkSemaphoreGetFdInfoKHR fdInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .semaphore = *this->semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
    };

    // FIXED: Correct check for function pointer existence to silence warning
    if (&Layer::ovkGetSemaphoreFdKHR == nullptr) {
        throw std::runtime_error("ovkGetSemaphoreFdKHR is NULL");
    }

    auto res = Layer::ovkGetSemaphoreFdKHR(device, &fdInfo, fd);
    if (res != VK_SUCCESS || *fd < 0) {
        *fd = -1;
    }
}
