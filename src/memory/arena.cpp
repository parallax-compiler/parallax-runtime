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

    detect_capabilities();

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = capacity_;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(backend_->device(), &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        std::cerr << "[UnifiedArena] Failed to create arena buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(backend_->device(), buffer_, &req);

    uint32_t mem_type = find_memory_type(
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        std::cerr << "[UnifiedArena] No host-visible|coherent memory type available" << std::endl;
        destroy();
        return false;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = mem_type;

    if (vkAllocateMemory(backend_->device(), &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
        std::cerr << "[UnifiedArena] Failed to allocate " << capacity_ << " bytes" << std::endl;
        destroy();
        return false;
    }

    vkBindBufferMemory(backend_->device(), buffer_, memory_, 0);

    if (vkMapMemory(backend_->device(), memory_, 0, capacity_, 0, &host_base_) != VK_SUCCESS) {
        std::cerr << "[UnifiedArena] Failed to map arena memory" << std::endl;
        destroy();
        return false;
    }

    bump_ = 0;
    high_water_ = 0;
    std::cout << "[UnifiedArena] Ready: " << (capacity_ / (1024 * 1024)) << " MiB, host_base="
              << host_base_ << " (int64=" << caps_.shader_int64
              << " float64=" << caps_.shader_float64
              << " bda=" << caps_.buffer_device_address << ")" << std::endl;
    return true;
}

void UnifiedArena::destroy() {
    if (memory_ != VK_NULL_HANDLE && host_base_) {
        vkUnmapMemory(backend_->device(), memory_);
        host_base_ = nullptr;
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(backend_->device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(backend_->device(), memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
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

void UnifiedArena::detect_capabilities() {
    VkPhysicalDevice pd = backend_->physical_device();

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(pd, &feats);
    caps_.shader_int64 = feats.shaderInt64 == VK_TRUE;
    caps_.shader_float64 = feats.shaderFloat64 == VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &bda;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    caps_.buffer_device_address = bda.bufferDeviceAddress == VK_TRUE;
}

}  // namespace parallax
