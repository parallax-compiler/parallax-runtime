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

// DeviceCapabilities is declared in vulkan_backend.hpp and detected by the
// backend at device-creation time (the bufferDeviceAddress feature must be
// enabled before the logical device is created).

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

    // Phase 3 whole-heap adoption: import the process-wide heap pool (heap_pool.cpp) as
    // ONE device buffer via VK_EXT_external_memory_host, so every heap pointer is
    // GPU-addressable with no copy. allocate/deallocate/contains/offset_of then delegate
    // to the pool. Returns false if the device lacks the extension or the import fails
    // (the caller then falls back to initialize()). host memory is coherent, so uma()==true.
    bool initialize_from_pool(VulkanBackend* backend);

    void destroy();

    // Sub-allocate `size` bytes, aligned to max(align, kDefaultAlignment).
    // Returns nullptr when the arena is exhausted.
    void* allocate(std::size_t size, std::size_t align = kDefaultAlignment);
    void  deallocate(void* ptr);

    // Pointer queries.
    bool         contains(const void* ptr) const;
    VkDeviceSize offset_of(const void* ptr) const;  // precondition: contains(ptr)

    // Software unified memory across a discrete (non-host-visible) device.
    // On an integrated/UMA device the arena buffer is itself host-visible, so the
    // host pointer aliases device memory and these are no-ops (uma() == true). On a
    // discrete GPU the host writes into a host-visible STAGING buffer and the kernels
    // run against a separate DEVICE_LOCAL buffer; flush copies host->device before a
    // launch and invalidate copies device->host after it. (Whole used-range copies
    // for now; dirty-range tracking is the next optimization.)
    void flush_to_device();       // host staging -> device (before kernels read)
    void invalidate_from_device(); // device -> host staging (after kernels write)
    bool uma() const { return uma_; }

    // Accessors.
    void*                     host_base() const { return host_base_; }
    VkBuffer                  buffer() const { return buffer_; }
    VkDeviceSize              capacity() const { return capacity_; }
    VkDeviceSize              used() const { return high_water_; }
    // GPU base address of the arena (Phase 2 relocates host pointers against this).
    // Zero when the device lacks buffer_device_address.
    VkDeviceAddress           device_address() const { return device_address_; }
    const DeviceCapabilities& capabilities() const { return backend_->capabilities(); }
    bool                      valid() const { return host_base_ != nullptr; }

private:
    struct Block {
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    void     copy_range(VkBuffer src, VkBuffer dst, VkDeviceSize size);  // staging xfer

    VulkanBackend*  backend_ = nullptr;
    VkBuffer        buffer_ = VK_NULL_HANDLE;        // DEVICE buffer kernels bind to
    VkDeviceMemory  memory_ = VK_NULL_HANDLE;
    void*           host_base_ = nullptr;            // host pointer (device map or staging)
    VkDeviceAddress device_address_ = 0;

    // Discrete-GPU staging (only allocated when the device buffer is NOT host-visible,
    // or when PARALLAX_FORCE_STAGING forces this path to exercise it on UMA hardware).
    bool            uma_ = true;                     // device memory is host-visible
    bool            pool_backed_ = false;             // arena imported the heap pool (Phase 3)
    VkBuffer        staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory  staging_memory_ = VK_NULL_HANDLE;
    VkCommandPool   xfer_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer xfer_cmd_ = VK_NULL_HANDLE;
    VkFence         xfer_fence_ = VK_NULL_HANDLE;
    VkDeviceSize    capacity_ = 0;
    VkDeviceSize    bump_ = 0;        // next never-used offset
    VkDeviceSize    high_water_ = 0;  // bytes ever handed out (bump_ minus reuse)
    std::vector<Block>                      free_list_;    // reusable freed blocks
    std::unordered_map<void*, Block>        live_;         // ptr -> {offset,size}
    mutable std::mutex                      mutex_;
};

}  // namespace parallax
