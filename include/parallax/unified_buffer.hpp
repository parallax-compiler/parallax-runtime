#ifndef PARALLAX_UNIFIED_BUFFER_HPP
#define PARALLAX_UNIFIED_BUFFER_HPP

#include "parallax/vulkan_backend.hpp"
#include <cstddef>
#include <unordered_map>
#include <memory>

namespace parallax {

enum class SyncDirection {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST
};

struct UnifiedBuffer {
    void* host_ptr = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    bool is_mapped = false;
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
    
    // Synchronization
    void sync(void* ptr, SyncDirection direction);
    
    // Get buffer for kernel launch
    VkBuffer get_buffer(void* ptr);
    
private:
    bool create_buffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped_ptr);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    
    VulkanBackend* backend_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    
    std::unordered_map<void*, std::unique_ptr<UnifiedBuffer>> buffers_;
};

} // namespace parallax

#endif // PARALLAX_UNIFIED_BUFFER_HPP
