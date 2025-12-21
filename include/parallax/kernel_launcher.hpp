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
    
    // Launch kernel by name
    bool launch(const std::string& kernel_name, void* buffer, size_t count, float multiplier);
    
private:
    VulkanBackend* backend_;
    MemoryManager* memory_manager_;
    
    // Pipeline cache
    std::unordered_map<std::string, PipelineData> pipelines_;
    
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace parallax

#endif // PARALLAX_KERNEL_LAUNCHER_HPP
