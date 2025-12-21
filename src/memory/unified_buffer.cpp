#include "parallax/unified_buffer.hpp"
#include <iostream>
#include <cstring>

namespace parallax {

MemoryManager::MemoryManager(VulkanBackend* backend) : backend_(backend) {
    // Create command pool for memory operations
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = backend_->compute_queue_family();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(backend_->device(), &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return;
    }
    
    // Allocate command buffer
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(backend_->device(), &alloc_info, &command_buffer_);
}

MemoryManager::~MemoryManager() {
    // Free all buffers
    for (auto& [ptr, buffer] : buffers_) {
        if (buffer->is_mapped) {
            vkUnmapMemory(backend_->device(), buffer->memory);
        }
        vkDestroyBuffer(backend_->device(), buffer->buffer, nullptr);
        vkFreeMemory(backend_->device(), buffer->memory, nullptr);
    }
    
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(backend_->device(), command_pool_, nullptr);
    }
}

void* MemoryManager::allocate(size_t size) {
    auto buffer = std::make_unique<UnifiedBuffer>();
    buffer->size = size;
    
    if (!create_buffer(size, buffer->buffer, buffer->memory, buffer->host_ptr)) {
        std::cerr << "Failed to create unified buffer" << std::endl;
        return nullptr;
    }
    
    buffer->is_mapped = true;
    void* ptr = buffer->host_ptr;
    buffers_[ptr] = std::move(buffer);
    
    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) {
        std::cerr << "Attempt to free unknown pointer" << std::endl;
        return;
    }
    
    auto& buffer = it->second;
    if (buffer->is_mapped) {
        vkUnmapMemory(backend_->device(), buffer->memory);
    }
    vkDestroyBuffer(backend_->device(), buffer->buffer, nullptr);
    vkFreeMemory(backend_->device(), buffer->memory, nullptr);
    
    buffers_.erase(it);
}

void MemoryManager::sync(void* ptr, SyncDirection direction) {
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) {
        std::cerr << "Sync on unknown pointer" << std::endl;
        return;
    }
    
    // For host-visible memory, just flush/invalidate
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = it->second->memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    
    if (direction == SyncDirection::HOST_TO_DEVICE) {
        vkFlushMappedMemoryRanges(backend_->device(), 1, &range);
    } else {
        vkInvalidateMappedMemoryRanges(backend_->device(), 1, &range);
    }
}

VkBuffer MemoryManager::get_buffer(void* ptr) {
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) {
        return VK_NULL_HANDLE;
    }
    return it->second->buffer;
}

bool MemoryManager::create_buffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped_ptr) {
    // Create buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(backend_->device(), &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return false;
    }
    
    // Get memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(backend_->device(), buffer, &mem_requirements);
    
    // Allocate memory (host-visible and device-local if possible)
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        mem_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    if (vkAllocateMemory(backend_->device(), &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory" << std::endl;
        vkDestroyBuffer(backend_->device(), buffer, nullptr);
        return false;
    }
    
    // Bind buffer to memory
    vkBindBufferMemory(backend_->device(), buffer, memory, 0);
    
    // Map memory
    if (vkMapMemory(backend_->device(), memory, 0, size, 0, &mapped_ptr) != VK_SUCCESS) {
        std::cerr << "Failed to map memory" << std::endl;
        vkFreeMemory(backend_->device(), memory, nullptr);
        vkDestroyBuffer(backend_->device(), buffer, nullptr);
        return false;
    }
    
    return true;
}

uint32_t MemoryManager::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(backend_->physical_device(), &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    std::cerr << "Failed to find suitable memory type" << std::endl;
    return 0;
}

} // namespace parallax
