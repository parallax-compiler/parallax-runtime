#ifndef PARALLAX_KERNEL_LAUNCHER_HPP
#define PARALLAX_KERNEL_LAUNCHER_HPP

#include "parallax/vulkan_backend.hpp"
#include "parallax/unified_buffer.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace parallax {

struct PipelineData {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
};

class KernelLauncher {
public:
    KernelLauncher(VulkanBackend* backend, MemoryManager* memory_manager);
    ~KernelLauncher();
    
    // Load SPIR-V kernel with name
    bool load_kernel(const std::string& name, const uint32_t* spirv_code, size_t spirv_size);
    
    // Launch kernel by name (vector_multiply specific)
    bool launch(const std::string& kernel_name, void* buffer, size_t count, float multiplier);

    // Generic launch (buffer + count)
    bool launch(const std::string& kernel_name, void* buffer, size_t count);

    // Transform launch (in/out buffers + count)
    bool launch_transform(const std::string& kernel_name, void* in_buffer, void* out_buffer, size_t count);

    // NEW V2: Launch with captured parameters (for function objects)
    bool launch_with_captures(
        const std::string& kernel_name,
        void* buffer,
        size_t count,
        void* captures,
        size_t capture_size
    );

    // Synchronize all pending operations
    void sync();
    
private:
    VulkanBackend* backend_;
    MemoryManager* memory_manager_;
    
    // Pipeline cache
    std::unordered_map<std::string, PipelineData> pipelines_;
    
    // Descriptor Set Cache: key is (descriptor_set_layout, buffer_ptr)
    struct CacheKey {
        VkDescriptorSetLayout layout;
        void* buffer;
        bool operator==(const CacheKey& other) const {
            return layout == other.layout && buffer == other.buffer;
        }
    };
    struct CacheHash {
        std::size_t operator()(const CacheKey& k) const {
            return std::hash<void*>{}(k.buffer) ^ (std::hash<uint64_t>{}((uint64_t)k.layout) << 1);
        }
    };
    std::unordered_map<CacheKey, VkDescriptorSet, CacheHash> descriptor_cache_;
    
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool fence_signaled_ = true;
};

} // namespace parallax

#endif // PARALLAX_KERNEL_LAUNCHER_HPP
