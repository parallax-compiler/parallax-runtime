#include "parallax/kernel_launcher.hpp"
#include <iostream>
#include <fstream>
#include <cstring>

namespace parallax {

KernelLauncher::KernelLauncher(VulkanBackend* backend, MemoryManager* memory_manager)
    : backend_(backend), memory_manager_(memory_manager) {
    
    // Create descriptor pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 16; // Support up to 16 buffers
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 16;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    
    if (vkCreateDescriptorPool(backend_->device(), &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
    }
    
    // Create command pool
    VkCommandPoolCreateInfo cmd_pool_info{};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = backend_->compute_queue_family();
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(backend_->device(), &cmd_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
    }
    
    // Allocate command buffer
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(backend_->device(), &alloc_info, &command_buffer_);
    
    // Create fence for synchronization
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    vkCreateFence(backend_->device(), &fence_info, nullptr, &fence_);
}

KernelLauncher::~KernelLauncher() {
    // Clean up pipelines
    for (auto& [hash, pipeline_data] : pipelines_) {
        vkDestroyPipeline(backend_->device(), pipeline_data.pipeline, nullptr);
        vkDestroyPipelineLayout(backend_->device(), pipeline_data.layout, nullptr);
        vkDestroyDescriptorSetLayout(backend_->device(), pipeline_data.descriptor_set_layout, nullptr);
        vkDestroyShaderModule(backend_->device(), pipeline_data.shader_module, nullptr);
    }
    
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(backend_->device(), fence_, nullptr);
    }
    
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(backend_->device(), command_pool_, nullptr);
    }
    
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(backend_->device(), descriptor_pool_, nullptr);
    }
}

bool KernelLauncher::load_kernel(const std::string& name, const uint32_t* spirv_code, size_t spirv_size) {
    // Create shader module
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_size;
    create_info.pCode = spirv_code;
    
    VkShaderModule shader_module;
    if (vkCreateShaderModule(backend_->device(), &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module for " << name << std::endl;
        return false;
    }
    
    // Create descriptor set layout (1 storage buffer binding)
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    
    VkDescriptorSetLayout descriptor_set_layout;
    if (vkCreateDescriptorSetLayout(backend_->device(), &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        vkDestroyShaderModule(backend_->device(), shader_module, nullptr);
        return false;
    }
    
    // Create pipeline layout with push constants
    VkPushConstantRange push_constant{};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = 16; // 4 uint32_t values
    
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    
    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(backend_->device(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        vkDestroyDescriptorSetLayout(backend_->device(), descriptor_set_layout, nullptr);
        vkDestroyShaderModule(backend_->device(), shader_module, nullptr);
        return false;
    }
    
    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shader_stage{};
    shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage.module = shader_module;
    shader_stage.pName = "main";
    
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage;
    pipeline_info.layout = pipeline_layout;
    
    VkPipeline pipeline;
    if (vkCreateComputePipelines(backend_->device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        vkDestroyPipelineLayout(backend_->device(), pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(backend_->device(), descriptor_set_layout, nullptr);
        vkDestroyShaderModule(backend_->device(), shader_module, nullptr);
        return false;
    }
    
    // Store pipeline data
    PipelineData data;
    data.pipeline = pipeline;
    data.layout = pipeline_layout;
    data.descriptor_set_layout = descriptor_set_layout;
    data.shader_module = shader_module;
    
    pipelines_[name] = data;
    
    std::cout << "Loaded kernel: " << name << std::endl;
    return true;
}

bool KernelLauncher::launch(const std::string& kernel_name, void* buffer, size_t count, float multiplier) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }
    
    auto& pipeline_data = it->second;
    
    // Get Vulkan buffer
    VkBuffer vk_buffer = memory_manager_->get_buffer(buffer);
    if (vk_buffer == VK_NULL_HANDLE) {
        std::cerr << "Invalid buffer" << std::endl;
        return false;
    }
    
    // Sync dirty blocks before kernel
    memory_manager_->sync_before_kernel(buffer);
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
    
    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set" << std::endl;
        return false;
    }
    
    // Update descriptor set
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = vk_buffer;
    buffer_info.offset = 0;
    buffer_info.range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffer_info;
    
    vkUpdateDescriptorSets(backend_->device(), 1, &write, 0, nullptr);
    
    // Wait for previous operations
    vkWaitForFences(backend_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(backend_->device(), 1, &fence_);
    
    // Record command buffer
    vkResetCommandBuffer(command_buffer_, 0);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.layout, 0, 1, &descriptor_set, 0, nullptr);
    
    // Push constants: count and multiplier
    uint32_t push_data[4] = {static_cast<uint32_t>(count), 0, 0, 0};
    std::memcpy(&push_data[1], &multiplier, sizeof(float));
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);
    
    // Dispatch compute shader (256 threads per workgroup)
    uint32_t workgroup_count = (count + 255) / 256;
    vkCmdDispatch(command_buffer_, workgroup_count, 1, 1);
    
    vkEndCommandBuffer(command_buffer_);
    
    // Submit command buffer
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    
    if (vkQueueSubmit(backend_->compute_queue(), 1, &submit_info, fence_) != VK_SUCCESS) {
        std::cerr << "Failed to submit command buffer" << std::endl;
        vkFreeDescriptorSets(backend_->device(), descriptor_pool_, 1, &descriptor_set);
        return false;
    }
    
    // Wait for completion
    vkWaitForFences(backend_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    
    // Sync after kernel
    memory_manager_->sync_after_kernel(buffer);
    
    // Free descriptor set
    vkFreeDescriptorSets(backend_->device(), descriptor_pool_, 1, &descriptor_set);
    
    return true;
}

} // namespace parallax
