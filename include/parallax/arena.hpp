#pragma once

// UnifiedArena — the foundation of Parallax's software unified memory.
//
// A single large Vulkan buffer backed by one device allocation, persistently
// mapped to a host pointer. Every unified-memory allocation is a sub-region
// (offset) of this one arena. Keeping all allocations in one contiguous region
// is what makes the next two steps tractable:
//   * a single VK_KHR_buffer_device_address for the whole arena (Phase 1b), and
//   * whole-arena migration / coherence (Phase 1e).
// See PARALLAX_STDPAR_COMPLIANCE_PLAN.md (Phase 1).

#include "parallax/vulkan_backend.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace parallax {

// Device capabilities relevant to codegen and the memory model. Detected from
// the physical device at arena init and later consumed by the compiler so it
// never silently emits unsupported types (e.g. Int64 truncation).
struct DeviceCapabilities {
    bool shader_int64 = false;
    bool shader_float64 = false;
    bool buffer_device_address = false;
};

class UnifiedArena {
public:
    static constexpr VkDeviceSize kDefaultCapacity = 256ull * 1024 * 1024;  // 256 MiB
    static constexpr VkDeviceSize kDefaultAlignment = 256;                  // GPU-friendly

    UnifiedArena() = default;
    ~UnifiedArena();

    UnifiedArena(const UnifiedArena&) = delete;
    UnifiedArena& operator=(const UnifiedArena&) = delete;

    // Create the arena (one buffer + one mapped allocation). Returns false on
    // any Vulkan failure; the object is left destroyed.
    bool initialize(VulkanBackend* backend, VkDeviceSize capacity = kDefaultCapacity);
    void destroy();

    // Sub-allocate `size` bytes, aligned to max(align, kDefaultAlignment).
    // Returns nullptr when the arena is exhausted.
    void* allocate(std::size_t size, std::size_t align = kDefaultAlignment);
    void  deallocate(void* ptr);

    // Pointer queries.
    bool         contains(const void* ptr) const;
    VkDeviceSize offset_of(const void* ptr) const;  // precondition: contains(ptr)

    // Accessors.
    void*                     host_base() const { return host_base_; }
    VkBuffer                  buffer() const { return buffer_; }
    VkDeviceSize              capacity() const { return capacity_; }
    VkDeviceSize              used() const { return high_water_; }
    const DeviceCapabilities& capabilities() const { return caps_; }
    bool                      valid() const { return host_base_ != nullptr; }

private:
    struct Block {
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    void     detect_capabilities();

    VulkanBackend* backend_ = nullptr;
    VkBuffer       buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void*          host_base_ = nullptr;
    VkDeviceSize   capacity_ = 0;
    VkDeviceSize   bump_ = 0;        // next never-used offset
    VkDeviceSize   high_water_ = 0;  // bytes ever handed out (bump_ minus reuse)
    std::vector<Block>                      free_list_;    // reusable freed blocks
    std::unordered_map<void*, Block>        live_;         // ptr -> {offset,size}
    DeviceCapabilities                      caps_;
    mutable std::mutex                      mutex_;
};

}  // namespace parallax
