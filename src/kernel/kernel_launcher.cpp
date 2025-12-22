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
    pool_size.descriptorCount = 2048; // Support more buffers/sets
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1024;
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
    // Debug: Dump SPIR-V if requested or always for now
    std::cerr << "SPIR-V Dump for " << name << " (" << spirv_size << " bytes):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(spirv_size / 4, 512); ++i) {
        std::cerr << "  [" << i << "] 0x" << std::hex << spirv_code[i] << std::dec << "\n";
    }
    std::cerr << std::flush;

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
    
    // Create descriptor set layout (Up to 2 storage buffer bindings for in/out)
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    for (int i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
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
    
    // Check descriptor cache
    CacheKey key{pipeline_data.descriptor_set_layout, buffer};
    VkDescriptorSet descriptor_set;
    if (descriptor_cache_.count(key)) {
        descriptor_set = descriptor_cache_[key];
    } else {
        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
        
        if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set" << std::endl;
            return false;
        }
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
    descriptor_cache_[key] = descriptor_set;
    
    // Wait for previous operations if any
    sync();
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
        return false;
    }
    
    fence_signaled_ = false;
    
    // NOTE: sync_after_kernel is removed to avoid host-device roundtrip.
    // User must call sync() explicitly if they want to access the buffer.
    
    return true;
}

void KernelLauncher::sync() {
    if (!fence_signaled_) {
        vkWaitForFences(backend_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
        fence_signaled_ = true;
    }
}

bool KernelLauncher::launch(const std::string& kernel_name, void* buffer, size_t count) {
    // Reuse specific implementation with dummy multiplier
    return launch(kernel_name, buffer, count, 1.0f);
}

bool KernelLauncher::launch_transform(const std::string& kernel_name, void* in_buffer, void* out_buffer, size_t count) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }
    
    auto& pipeline_data = it->second;
    
    // Get Vulkan buffers
    VkBuffer vk_in = memory_manager_->get_buffer(in_buffer);
    VkBuffer vk_out = memory_manager_->get_buffer(out_buffer);
    
    if (vk_in == VK_NULL_HANDLE || vk_out == VK_NULL_HANDLE) {
        std::cerr << "Invalid buffers" << std::endl;
        return false;
    }
    
    // Sync dirty blocks before kernel
    memory_manager_->sync_before_kernel(in_buffer);
    memory_manager_->sync_before_kernel(out_buffer);
    
    // Check descriptor cache (multi-buffer key is complex, for MVP we use a simple combination)
    // Actually, for transform we have 2 buffers. We'll use the out_buffer as the key part for now
    // or better, a combined key. For now, let's just use the out_buffer since layout is fixed.
    CacheKey key{pipeline_data.descriptor_set_layout, out_buffer};
    VkDescriptorSet descriptor_set;
    if (descriptor_cache_.count(key)) {
        descriptor_set = descriptor_cache_[key];
    } else {
        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
        
        if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set" << std::endl;
            return false;
        }
        
        // Update descriptor set (2 buffers)
        std::vector<VkDescriptorBufferInfo> buffer_infos(2);
        buffer_infos[0].buffer = vk_in;
        buffer_infos[0].offset = 0;
        buffer_infos[0].range = VK_WHOLE_SIZE;
        
        buffer_infos[1].buffer = vk_out;
        buffer_infos[1].offset = 0;
        buffer_infos[1].range = VK_WHOLE_SIZE;
        
        std::vector<VkWriteDescriptorSet> writes(2);
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        
        vkUpdateDescriptorSets(backend_->device(), 2, writes.data(), 0, nullptr);
        descriptor_cache_[key] = descriptor_set;
    }
    
    // Wait for previous operations if any
    sync();
    vkResetFences(backend_->device(), 1, &fence_);
    
    // Record command buffer (same logic as launch)
    vkResetCommandBuffer(command_buffer_, 0);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.layout, 0, 1, &descriptor_set, 0, nullptr);
    
    // Push constants: count
    uint32_t push_data[4] = {static_cast<uint32_t>(count), 0, 0, 0};
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);
    
    // Dispatch
    uint32_t workgroup_count = (count + 255) / 256;
    vkCmdDispatch(command_buffer_, workgroup_count, 1, 1);
    
    vkEndCommandBuffer(command_buffer_);
    
    // Submit
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    
    if (vkQueueSubmit(backend_->compute_queue(), 1, &submit_info, fence_) != VK_SUCCESS) {
        std::cerr << "Failed to submit command buffer" << std::endl;
        return false;
    }
    
    fence_signaled_ = false;
    return true;
}

} // namespace parallax
