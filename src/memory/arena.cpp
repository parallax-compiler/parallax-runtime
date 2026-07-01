#include "parallax/arena.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace parallax {

namespace {
inline VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) & ~(a - 1);
}
}  // namespace

UnifiedArena::~UnifiedArena() {
    destroy();
}

bool UnifiedArena::initialize(VulkanBackend* backend, VkDeviceSize capacity) {
    backend_ = backend;
    if (!backend_ || backend_->device() == VK_NULL_HANDLE) {
        std::cerr << "[UnifiedArena] No valid Vulkan device" << std::endl;
        return false;
    }

    // Allow overriding the arena size from the environment for large workloads.
    if (const char* env = std::getenv("PARALLAX_ARENA_SIZE")) {
        unsigned long long mb = std::strtoull(env, nullptr, 10);
        if (mb > 0) capacity = static_cast<VkDeviceSize>(mb) * 1024 * 1024;
    }
    capacity_ = align_up(capacity, kDefaultAlignment);

    const DeviceCapabilities& caps = backend_->capabilities();
    const bool use_bda = caps.buffer_device_address;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = capacity_;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (use_bda) {
        buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(backend_->device(), &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        std::cerr << "[UnifiedArena] Failed to create arena buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(backend_->device(), buffer_, &req);

    VkMemoryAllocateFlagsInfo flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    // Prefer DEVICE_LOCAL memory for the buffer the kernels run against. On an
    // integrated/UMA device that type is ALSO host-visible, so we map it directly and
    // there is no staging (uma_ == true; this is the lavapipe path — identical to
    // before). On a discrete GPU device-local memory is not host-visible, so the host
    // writes go to a separate staging buffer and we migrate around launches.
    // PARALLAX_FORCE_STAGING forces the staging path even on UMA to exercise it.
    const bool force_staging = std::getenv("PARALLAX_FORCE_STAGING") != nullptr;
    uint32_t uma_type = force_staging ? UINT32_MAX : find_memory_type(
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (uma_type != UINT32_MAX) {
        uma_ = true;
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = uma_type;
        if (use_bda) alloc_info.pNext = &flags_info;
        if (vkAllocateMemory(backend_->device(), &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to allocate " << capacity_ << " bytes (UMA)" << std::endl;
            destroy();
            return false;
        }
        vkBindBufferMemory(backend_->device(), buffer_, memory_, 0);
        if (vkMapMemory(backend_->device(), memory_, 0, capacity_, 0, &host_base_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to map arena memory" << std::endl;
            destroy();
            return false;
        }
    } else {
        // Discrete path: device-local device buffer + host-visible staging buffer.
        uma_ = false;
        uint32_t dev_type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (dev_type == UINT32_MAX) {
            std::cerr << "[UnifiedArena] No device-local memory type available" << std::endl;
            destroy();
            return false;
        }
        VkMemoryAllocateInfo dev_alloc{};
        dev_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        dev_alloc.allocationSize = req.size;
        dev_alloc.memoryTypeIndex = dev_type;
        if (use_bda) dev_alloc.pNext = &flags_info;
        if (vkAllocateMemory(backend_->device(), &dev_alloc, nullptr, &memory_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to allocate device-local memory" << std::endl;
            destroy();
            return false;
        }
        vkBindBufferMemory(backend_->device(), buffer_, memory_, 0);

        // Host-visible staging buffer the host reads/writes through.
        VkBufferCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sci.size = capacity_;
        sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        sci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(backend_->device(), &sci, nullptr, &staging_buffer_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to create staging buffer" << std::endl;
            destroy();
            return false;
        }
        VkMemoryRequirements sreq{};
        vkGetBufferMemoryRequirements(backend_->device(), staging_buffer_, &sreq);
        uint32_t stage_type = find_memory_type(
            sreq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stage_type == UINT32_MAX) {
            std::cerr << "[UnifiedArena] No host-visible staging memory type" << std::endl;
            destroy();
            return false;
        }
        VkMemoryAllocateInfo salloc{};
        salloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        salloc.allocationSize = sreq.size;
        salloc.memoryTypeIndex = stage_type;
        if (vkAllocateMemory(backend_->device(), &salloc, nullptr, &staging_memory_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to allocate staging memory" << std::endl;
            destroy();
            return false;
        }
        vkBindBufferMemory(backend_->device(), staging_buffer_, staging_memory_, 0);
        if (vkMapMemory(backend_->device(), staging_memory_, 0, capacity_, 0, &host_base_) != VK_SUCCESS) {
            std::cerr << "[UnifiedArena] Failed to map staging memory" << std::endl;
            destroy();
            return false;
        }

        // A dedicated command pool + fence for the migration copies.
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = backend_->compute_queue_family();
        vkCreateCommandPool(backend_->device(), &pci, nullptr, &xfer_pool_);
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = xfer_pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        vkAllocateCommandBuffers(backend_->device(), &cbi, &xfer_cmd_);
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(backend_->device(), &fci, nullptr, &xfer_fence_);
    }

    if (use_bda) {
        VkBufferDeviceAddressInfo addr_info{};
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = buffer_;
        device_address_ = vkGetBufferDeviceAddress(backend_->device(), &addr_info);
    }

    bump_ = 0;
    high_water_ = 0;
    std::cout << "[UnifiedArena] Ready: " << (capacity_ / (1024 * 1024)) << " MiB, host_base="
              << host_base_ << " device_address=0x" << std::hex << device_address_ << std::dec
              << " (uma=" << uma_ << " int64=" << caps.shader_int64
              << " float64=" << caps.shader_float64
              << " bda=" << caps.buffer_device_address << ")" << std::endl;
    return true;
}

void UnifiedArena::copy_range(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    if (size == 0 || xfer_cmd_ == VK_NULL_HANDLE) return;
    vkResetCommandBuffer(xfer_cmd_, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(xfer_cmd_, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(xfer_cmd_, src, dst, 1, &region);
    vkEndCommandBuffer(xfer_cmd_);
    vkResetFences(backend_->device(), 1, &xfer_fence_);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &xfer_cmd_;
    vkQueueSubmit(backend_->compute_queue(), 1, &si, xfer_fence_);
    vkWaitForFences(backend_->device(), 1, &xfer_fence_, VK_TRUE, UINT64_MAX);
}

void UnifiedArena::flush_to_device() {
    if (uma_ || high_water_ == 0) return;
    copy_range(staging_buffer_, buffer_, high_water_);
}

void UnifiedArena::invalidate_from_device() {
    if (uma_ || high_water_ == 0) return;
    copy_range(buffer_, staging_buffer_, high_water_);
}

void UnifiedArena::destroy() {
    VkDevice dev = backend_ ? backend_->device() : VK_NULL_HANDLE;
    if (host_base_) {
        // host_base_ maps the device memory (UMA) or the staging memory (discrete).
        vkUnmapMemory(dev, uma_ ? memory_ : staging_memory_);
        host_base_ = nullptr;
    }
    if (xfer_fence_ != VK_NULL_HANDLE) { vkDestroyFence(dev, xfer_fence_, nullptr); xfer_fence_ = VK_NULL_HANDLE; }
    if (xfer_pool_ != VK_NULL_HANDLE) { vkDestroyCommandPool(dev, xfer_pool_, nullptr); xfer_pool_ = VK_NULL_HANDLE; xfer_cmd_ = VK_NULL_HANDLE; }
    if (staging_buffer_ != VK_NULL_HANDLE) { vkDestroyBuffer(dev, staging_buffer_, nullptr); staging_buffer_ = VK_NULL_HANDLE; }
    if (staging_memory_ != VK_NULL_HANDLE) { vkFreeMemory(dev, staging_memory_, nullptr); staging_memory_ = VK_NULL_HANDLE; }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(dev, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(dev, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    device_address_ = 0;
    uma_ = true;
    free_list_.clear();
    live_.clear();
    bump_ = high_water_ = capacity_ = 0;
}

void* UnifiedArena::allocate(std::size_t size, std::size_t align) {
    if (size == 0 || !host_base_) return nullptr;
    const VkDeviceSize a = std::max<VkDeviceSize>(align, kDefaultAlignment);
    const VkDeviceSize need = align_up(size, a);

    std::lock_guard<std::mutex> lock(mutex_);

    // First-fit reuse of a previously freed block.
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        const VkDeviceSize aligned_off = align_up(it->offset, a);
        const VkDeviceSize pad = aligned_off - it->offset;
        if (it->size >= pad + need) {
            VkDeviceSize off = aligned_off;
            // Shrink/erase the free block (no coalescing yet; padding is leaked
            // back to the block tail if any remains).
            VkDeviceSize remainder = it->size - pad - need;
            if (remainder > 0) {
                it->offset = off + need;
                it->size = remainder;
            } else {
                free_list_.erase(it);
            }
            void* p = static_cast<char*>(host_base_) + off;
            live_[p] = {off, need};
            return p;
        }
    }

    // Bump-allocate from the never-used tail.
    VkDeviceSize off = align_up(bump_, a);
    if (off + need > capacity_) {
        std::cerr << "[UnifiedArena] Out of memory: requested " << need
                  << ", available " << (capacity_ - off) << std::endl;
        return nullptr;
    }
    bump_ = off + need;
    high_water_ = std::max(high_water_, bump_);
    void* p = static_cast<char*>(host_base_) + off;
    live_[p] = {off, need};
    return p;
}

void UnifiedArena::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(ptr);
    if (it == live_.end()) {
        // Not an arena pointer (or double free) — ignore defensively.
        return;
    }
    free_list_.push_back(it->second);
    live_.erase(it);
}

bool UnifiedArena::contains(const void* ptr) const {
    if (!host_base_ || !ptr) return false;
    const char* base = static_cast<const char*>(host_base_);
    const char* p = static_cast<const char*>(ptr);
    return p >= base && p < base + capacity_;
}

VkDeviceSize UnifiedArena::offset_of(const void* ptr) const {
    return static_cast<VkDeviceSize>(static_cast<const char*>(ptr) -
                                     static_cast<const char*>(host_base_));
}

uint32_t UnifiedArena::find_memory_type(uint32_t type_filter,
                                        VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(backend_->physical_device(), &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

}  // namespace parallax
