#ifndef PARALLAX_KERNEL_LAUNCHER_HPP
#define PARALLAX_KERNEL_LAUNCHER_HPP

#include "parallax/vulkan_backend.hpp"
#include "parallax/unified_buffer.hpp"
#include <vector>

namespace parallax {

class KernelLauncher {
public:
    KernelLauncher(VulkanBackend* backend, MemoryManager* memory);
    ~KernelLauncher();
    
    // Load SPIR-V kernel
    bool load_spirv(const uint32_t* spirv_code, size_t spirv_size);
    
    // Launch kernel
    bool launch(void* input_buffer, void* output_buffer, uint32_t count, float multiplier);
    
private:
    bool create_descriptor_set_layout();
    bool create_pipeline();
    bool create_descriptor_pool();
    bool allocate_descriptor_sets();
    
    VulkanBackend* backend_;
    MemoryManager* memory_;
    
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
};

} // namespace parallax

#endif // PARALLAX_KERNEL_LAUNCHER_HPP
