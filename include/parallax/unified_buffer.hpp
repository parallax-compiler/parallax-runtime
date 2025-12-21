#ifndef PARALLAX_UNIFIED_BUFFER_HPP
#define PARALLAX_UNIFIED_BUFFER_HPP

#include "parallax/vulkan_backend.hpp"
#include <cstddef>
#include <unordered_map>
#include <memory>
#include <vector>
#include <cstdint>

namespace parallax {

enum class SyncDirection {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST
};

// Block-level dirty tracking (4KB blocks)
constexpr size_t BLOCK_SIZE = 4096;

struct DirtyBlock {
    size_t block_index;
    bool dirty_on_host = false;
    bool dirty_on_device = false;
    
    void mark_host_dirty() { dirty_on_host = true; dirty_on_device = false; }
    void mark_device_dirty() { dirty_on_device = true; dirty_on_host = false; }
    void clear_dirty() { dirty_on_host = false; dirty_on_device = false; }
    bool needs_host_to_device() const { return dirty_on_host && !dirty_on_device; }
    bool needs_device_to_host() const { return dirty_on_device && !dirty_on_host; }
};

struct UnifiedBuffer {
    void* host_ptr = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    bool is_mapped = false;
    
    // Dirty tracking
    std::vector<DirtyBlock> blocks;
    size_t num_blocks = 0;
    
    void init_dirty_tracking() {
        num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.resize(num_blocks);
        for (size_t i = 0; i < num_blocks; i++) {
            blocks[i].block_index = i;
            blocks[i].mark_host_dirty(); // Initially dirty on host
        }
    }
    
    void mark_range_dirty_on_host(size_t offset, size_t length) {
        size_t start_block = offset / BLOCK_SIZE;
        size_t end_block = (offset + length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (size_t i = start_block; i < end_block && i < num_blocks; i++) {
            blocks[i].mark_host_dirty();
        }
    }
    
    void mark_all_dirty_on_device() {
        for (auto& block : blocks) {
            block.mark_device_dirty();
        }
    }
};

class MemoryManager {
public:
    explicit MemoryManager(VulkanBackend* backend);
    ~MemoryManager();
    
    // Disable copy
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    // Unified memory allocation
    void* allocate(size_t size);
    void deallocate(void* ptr);
    
    // Synchronization (deprecated - for compatibility)
    void sync(void* ptr, SyncDirection direction);
    
    // Automatic coherence before kernel launch
    void sync_before_kernel(void* ptr);
    void sync_after_kernel(void* ptr);
    
    // Get buffer for kernel launch
    VkBuffer get_buffer(void* ptr);
    
private:
    bool create_buffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped_ptr);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    void transfer_dirty_blocks(UnifiedBuffer* buffer, SyncDirection direction);
    
    VulkanBackend* backend_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    
    std::unordered_map<void*, std::unique_ptr<UnifiedBuffer>> buffers_;
};

} // namespace parallax

#endif // PARALLAX_UNIFIED_BUFFER_HPP
