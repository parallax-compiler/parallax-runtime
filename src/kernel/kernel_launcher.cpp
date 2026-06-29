#include "parallax/kernel_launcher.hpp"
#include "parallax/runtime.hpp"
#include "parallax/arena.hpp"
#include <iostream>
#include <fstream>
#include <cstring>

namespace parallax {

namespace {
// Push-constant block shared by all compute dispatches. Mirrors the compiler's
// setup_push_constants layout: count @0, host_base @8, dev_base @16. Ordinary
// kernels read only count; pointer-chasing kernels relocate stored host pointers
// with gpu = dev_base + (host_ptr - host_base), so the arena bases travel here.
struct PushBlock {
    uint32_t count;
    uint32_t pad;
    uint64_t host_base;
    uint64_t dev_base;
};
static_assert(sizeof(PushBlock) == 24, "push-constant block must be 24 bytes");

PushBlock make_push_block(size_t count) {
    PushBlock pc{};
    pc.count = static_cast<uint32_t>(count);
    UnifiedArena* arena = get_global_arena();
    if (arena) {
        pc.host_base = reinterpret_cast<uint64_t>(arena->host_base());
        pc.dev_base = static_cast<uint64_t>(arena->device_address());
    }
    return pc;
}
}  // namespace

KernelLauncher::KernelLauncher(VulkanBackend* backend, MemoryManager* memory_manager)
    : backend_(backend), memory_manager_(memory_manager) {
    
    // Create descriptor pool with support for both storage and uniform buffers
    std::vector<VkDescriptorPoolSize> pool_sizes(2);
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 6144; // up to 3 storage bindings per set
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 2048; // For captures buffers

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
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

void KernelLauncher::retire_transient_buffers() {
    if (transient_buffers_.empty()) return;
    // These scratch buffers are referenced by cached descriptor sets, so they live
    // for the launcher's lifetime and are reclaimed here at teardown. If the device
    // has already been torn down (static-destruction order), just drop the handles.
    if (!backend_ || backend_->device() == VK_NULL_HANDLE) {
        transient_buffers_.clear();
        return;
    }
    vkDeviceWaitIdle(backend_->device());
    for (auto& [buf, mem] : transient_buffers_) {
        if (buf != VK_NULL_HANDLE) vkDestroyBuffer(backend_->device(), buf, nullptr);
        if (mem != VK_NULL_HANDLE) vkFreeMemory(backend_->device(), mem, nullptr);
    }
    transient_buffers_.clear();
}

KernelLauncher::~KernelLauncher() {
    retire_transient_buffers();
    std::cout << "[KernelLauncher] Destructor: Cleaning up " << pipelines_.size() << " pipelines" << std::endl;

    // Clean up pipelines
    for (auto& [hash, pipeline_data] : pipelines_) {
        try {
            if (pipeline_data.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(backend_->device(), pipeline_data.pipeline, nullptr);
            }
            if (pipeline_data.layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(backend_->device(), pipeline_data.layout, nullptr);
            }
            if (pipeline_data.descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(backend_->device(), pipeline_data.descriptor_set_layout, nullptr);
            }
            if (pipeline_data.shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(backend_->device(), pipeline_data.shader_module, nullptr);
            }
        } catch (...) {
            std::cerr << "[KernelLauncher] Error freeing pipeline" << std::endl;
        }
    }

    pipelines_.clear();

    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(backend_->device(), fence_, nullptr);
    }

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(backend_->device(), command_pool_, nullptr);
    }

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(backend_->device(), descriptor_pool_, nullptr);
    }

    std::cout << "[KernelLauncher] Destructor: Cleanup complete" << std::endl;
}

bool KernelLauncher::load_kernel(const std::string& name, const uint32_t* spirv_code, size_t spirv_size) {
    // Debug: Dump SPIR-V header only (first 10 words) to avoid output buffer issues
    std::cerr << "SPIR-V Dump for " << name << " (" << spirv_size << " bytes):" << std::endl;
    std::cerr << "  Header (first 10 words): ";
    for (size_t i = 0; i < std::min<size_t>(spirv_size / 4, 10); ++i) {
        std::cerr << "0x" << std::hex << spirv_code[i] << std::dec;
        if (i + 1 < std::min<size_t>(spirv_size / 4, 10)) std::cerr << " ";
    }
    std::cerr << std::endl;

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
    
    // Create descriptor set layout
    // Binding 0-1: Storage buffers for data (in/out)
    // Binding 2: Uniform buffer for captures
    // Binding 3: Storage buffer for a third array (compaction scatter: positions).
    //   Existing kernels never declare/access binding 3, so per the Vulkan spec they
    //   need not write it — the 2-storage dispatch paths are unaffected (additive).
    std::vector<VkDescriptorSetLayoutBinding> bindings(4);
    for (int i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Binding 2: Uniform buffer for captures
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Binding 3: third storage buffer (compaction only).
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

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

    // Create pipeline layout with push constants. Layout (matches the compiler's
    // setup_push_constants): { uint count @0, uint64 host_base @8, uint64 dev_base
    // @16 }. Ordinary kernels only read count@0; pointer-chasing kernels also read
    // the arena bases to relocate stored host pointers. The range is sized for the
    // superset so a single pipeline layout serves both.
    VkPushConstantRange push_constant{};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = 24;
    
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

bool KernelLauncher::launch(const std::string& kernel_name, void* buffer, size_t count, float multiplier, size_t elem_size) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }

    auto& pipeline_data = it->second;

    // Phase 2c: if the data is arena-backed (e.g. via allocation interposition),
    // bind the arena buffer directly at the allocation's offset (zero-copy unified
    // memory) instead of copying it into a fresh Vulkan buffer.
    parallax::UnifiedArena* arena = parallax::get_global_arena();
    const bool arena_backed = arena && buffer && arena->contains(buffer);

    VkBuffer vk_buffer = VK_NULL_HANDLE;
    VkDeviceSize data_offset = 0;
    VkDeviceSize data_range = static_cast<VkDeviceSize>(count) * elem_size;

    if (arena_backed) {
        vk_buffer = arena->buffer();
        data_offset = arena->offset_of(buffer);
        std::cout << "[KernelLauncher] Zero-copy arena bind (offset=" << data_offset
                  << ", range=" << data_range << ")" << std::endl;
    } else {
        vk_buffer = memory_manager_->get_buffer(buffer);
        if (vk_buffer == VK_NULL_HANDLE && buffer != nullptr) {
            std::cerr << "[KernelLauncher] Auto-registering buffer (elem_size=" << elem_size << ")" << std::endl;
            memory_manager_->register_external_buffer(buffer, data_range);
            vk_buffer = memory_manager_->get_buffer(buffer);
        }
        if (vk_buffer == VK_NULL_HANDLE) {
            std::cerr << "Invalid buffer after auto-registration" << std::endl;
            return false;
        }
        // Sync dirty blocks before kernel (host-coherent arena needs no sync).
        memory_manager_->sync_before_kernel(buffer);
    }

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
    buffer_info.offset = data_offset;
    buffer_info.range = data_range;

    // Create a small dummy uniform buffer for captures (empty for now)
    VkBuffer dummy_uniform_buffer = VK_NULL_HANDLE;
    VkDeviceMemory dummy_uniform_memory = VK_NULL_HANDLE;
    VkBufferCreateInfo uniform_buf_info{};
    uniform_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    uniform_buf_info.size = 64; // Small buffer for captures
    uniform_buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    uniform_buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateBuffer(backend_->device(), &uniform_buf_info, nullptr, &dummy_uniform_buffer);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(backend_->device(), dummy_uniform_buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = mem_reqs.size;
    alloc.memoryTypeIndex = memory_manager_->find_memory_type(mem_reqs.memoryTypeBits,
                                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(backend_->device(), &alloc, nullptr, &dummy_uniform_memory);
    vkBindBufferMemory(backend_->device(), dummy_uniform_buffer, dummy_uniform_memory, 0);

    VkDescriptorBufferInfo captures_buffer_info{};
    captures_buffer_info.buffer = dummy_uniform_buffer;
    captures_buffer_info.offset = 0;
    captures_buffer_info.range = VK_WHOLE_SIZE;

    // Write both storage buffer (binding 0) and uniform buffer (binding 2)
    std::vector<VkWriteDescriptorSet> writes(2);
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &buffer_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 2; // Captures uniform buffer
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &captures_buffer_info;

    vkUpdateDescriptorSets(backend_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    descriptor_cache_[key] = descriptor_set;

    transient_buffers_.emplace_back(dummy_uniform_buffer, dummy_uniform_memory);
    
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

    // Push constants: count + arena bases (for pointer-chasing relocation).
    PushBlock push = make_push_block(count);
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

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

bool KernelLauncher::launch(const std::string& kernel_name, void* buffer, size_t count, size_t elem_size) {
    // Reuse specific implementation with dummy multiplier
    return launch(kernel_name, buffer, count, 1.0f, elem_size);
}

bool KernelLauncher::launch_transform(const std::string& kernel_name, void* in_buffer, void* out_buffer, size_t count, size_t elem_size, size_t out_elem_size) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }

    auto& pipeline_data = it->second;

    // Auto-register buffers if not already registered. The output element size may
    // differ from the input (e.g. float -> double), so size each buffer separately.
    if (out_elem_size == 0) out_elem_size = elem_size;
    size_t in_size = count * elem_size;
    size_t out_size = count * out_elem_size;
    VkBuffer vk_in = memory_manager_->get_buffer(in_buffer);
    if (vk_in == VK_NULL_HANDLE && in_buffer != nullptr) {
        std::cerr << "[KernelLauncher] Auto-registering input buffer" << std::endl;
        memory_manager_->register_external_buffer(in_buffer, in_size);
        vk_in = memory_manager_->get_buffer(in_buffer);
    }

    VkBuffer vk_out = memory_manager_->get_buffer(out_buffer);
    if (vk_out == VK_NULL_HANDLE && out_buffer != nullptr) {
        std::cerr << "[KernelLauncher] Auto-registering output buffer" << std::endl;
        memory_manager_->register_external_buffer(out_buffer, out_size);
        vk_out = memory_manager_->get_buffer(out_buffer);
    }

    if (vk_in == VK_NULL_HANDLE || vk_out == VK_NULL_HANDLE) {
        std::cerr << "Invalid buffers after auto-registration" << std::endl;
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
        
        // Update descriptor set (2 storage buffers + 1 uniform buffer for captures)
        std::vector<VkDescriptorBufferInfo> buffer_infos(2);
        buffer_infos[0].buffer = vk_in;
        buffer_infos[0].offset = 0;
        buffer_infos[0].range = VK_WHOLE_SIZE;

        buffer_infos[1].buffer = vk_out;
        buffer_infos[1].offset = 0;
        buffer_infos[1].range = VK_WHOLE_SIZE;

        // Create dummy uniform buffer for captures
        VkBuffer dummy_uniform_buffer = VK_NULL_HANDLE;
        VkDeviceMemory dummy_uniform_memory = VK_NULL_HANDLE;
        VkBufferCreateInfo uniform_buf_info{};
        uniform_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        uniform_buf_info.size = 64; // Small buffer
        uniform_buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        uniform_buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(backend_->device(), &uniform_buf_info, nullptr, &dummy_uniform_buffer);

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(backend_->device(), dummy_uniform_buffer, &mem_reqs);

        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = mem_reqs.size;
        alloc.memoryTypeIndex = memory_manager_->find_memory_type(mem_reqs.memoryTypeBits,
                                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(backend_->device(), &alloc, nullptr, &dummy_uniform_memory);
        vkBindBufferMemory(backend_->device(), dummy_uniform_buffer, dummy_uniform_memory, 0);

        VkDescriptorBufferInfo captures_buffer_info{};
        captures_buffer_info.buffer = dummy_uniform_buffer;
        captures_buffer_info.offset = 0;
        captures_buffer_info.range = VK_WHOLE_SIZE;

        std::vector<VkWriteDescriptorSet> writes(3);
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        // Binding 2: Captures uniform buffer
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptor_set;
        writes[2].dstBinding = 2;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &captures_buffer_info;

        vkUpdateDescriptorSets(backend_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        descriptor_cache_[key] = descriptor_set;

        transient_buffers_.emplace_back(dummy_uniform_buffer, dummy_uniform_memory);
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

    // Push constants: count + arena bases (for pointer-chasing relocation).
    PushBlock push = make_push_block(count);
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

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

// NEW V2: Launch kernel with captured parameters (for function objects)
bool KernelLauncher::launch_with_captures(
    const std::string& kernel_name,
    void* buffer,
    size_t count,
    void* captures,
    size_t capture_size,
    size_t elem_size) {

    (void)elem_size;  // data buffer is expected to be pre-registered on this path
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }

    auto& pipeline_data = it->second;

    // Get Vulkan buffer for main data
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

        // Find decomposed vector pointers in captures struct
        // Heuristic: Decomposed vectors are stored as {T* data, size_t size} pairs
        // Look for pointer + size_t patterns, or standalone pointers
        std::vector<void*> captured_buffers;
        if (captures != nullptr && capture_size >= sizeof(void*)) {
            const size_t num_fields = capture_size / sizeof(void*);
            void** ptr_array = static_cast<void**>(captures);

            std::cout << "[KernelLauncher] Analyzing captures struct: size=" << capture_size
                      << " bytes, num_fields=" << num_fields << std::endl;

            for (size_t i = 0; i < num_fields; ++i) {
                void* potential_ptr = ptr_array[i];

                // Check if there's a next field that could be a size
                size_t potential_size = 0;
                if (i + 1 < num_fields) {
                    potential_size = reinterpret_cast<size_t*>(ptr_array)[i + 1];
                }

                std::cout << "[KernelLauncher]   Field " << i << ": ptr=" << potential_ptr;
                if (i + 1 < num_fields) {
                    std::cout << ", next_value=" << potential_size;
                }
                std::cout << std::endl;

                // Check if this looks like a decomposed vector: pointer + reasonable size
                if (potential_ptr != nullptr && i + 1 < num_fields && potential_size > 0 && potential_size < 1000000000) {
                    VkBuffer test_buffer = memory_manager_->get_buffer(potential_ptr);

                    std::cout << "[KernelLauncher]   Looks like decomposed vector. VkBuffer="
                              << (void*)test_buffer << std::endl;

                    // If not registered, try to register it
                    if (test_buffer == VK_NULL_HANDLE) {
                        std::cout << "[KernelLauncher] Auto-registering captured buffer at offset "
                                  << (i * sizeof(void*)) << ", size=" << potential_size << " elements" << std::endl;

                        // Estimate element size - assume float or struct (try common sizes)
                        // For now, assume the size field is in elements, and element is at least 4 bytes
                        size_t byte_size = potential_size;
                        if (potential_size < 1000000) {  // Likely in elements, not bytes
                            byte_size = potential_size * sizeof(float) * 4;  // Assume Body-like struct (~28 bytes)
                        }

                        memory_manager_->register_external_buffer(potential_ptr, byte_size);
                        test_buffer = memory_manager_->get_buffer(potential_ptr);

                        std::cout << "[KernelLauncher] After registration: VkBuffer="
                                  << (void*)test_buffer << std::endl;
                    }

                    if (test_buffer != VK_NULL_HANDLE) {
                        captured_buffers.push_back(potential_ptr);
                        std::cout << "[KernelLauncher] Found captured buffer pointer at offset "
                                  << (i * sizeof(void*)) << std::endl;
                        i++;  // Skip the size field
                    }
                }
                // Fallback: check if this is just a standalone buffer pointer (not a vector decomposition)
                else if (potential_ptr != nullptr) {
                    VkBuffer test_buffer = memory_manager_->get_buffer(potential_ptr);
                    if (test_buffer != VK_NULL_HANDLE) {
                        captured_buffers.push_back(potential_ptr);
                        std::cout << "[KernelLauncher] Found standalone captured buffer pointer at offset "
                                  << (i * sizeof(void*)) << std::endl;
                    }
                }
            }
        }

        // Prepare descriptor writes
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> buffer_infos;

        // Reserve space to prevent reallocation (important! pointers must remain valid)
        buffer_infos.reserve(3);
        writes.reserve(3);

        // Binding 0: Main buffer (storage buffer)
        buffer_infos.push_back({vk_buffer, 0, VK_WHOLE_SIZE});
        VkWriteDescriptorSet write0{};
        write0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write0.dstSet = descriptor_set;
        write0.dstBinding = 0;
        write0.dstArrayElement = 0;
        write0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write0.descriptorCount = 1;
        write0.pBufferInfo = &buffer_infos[0];
        writes.push_back(write0);

        // Binding 1: Captured buffer (storage buffer) - if any
        if (!captured_buffers.empty()) {
            VkBuffer captured_vk_buffer = memory_manager_->get_buffer(captured_buffers[0]);
            buffer_infos.push_back({captured_vk_buffer, 0, VK_WHOLE_SIZE});

            VkWriteDescriptorSet write1{};
            write1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write1.dstSet = descriptor_set;
            write1.dstBinding = 1;
            write1.dstArrayElement = 0;
            write1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write1.descriptorCount = 1;
            write1.pBufferInfo = &buffer_infos[1];
            writes.push_back(write1);

            std::cout << "[KernelLauncher] Binding captured buffer to binding 1" << std::endl;
        }

        // Binding 2: Captures uniform buffer
        VkBuffer captures_uniform_buffer = VK_NULL_HANDLE;
        VkDeviceMemory captures_uniform_memory = VK_NULL_HANDLE;

        VkBufferCreateInfo uniform_buf_info{};
        uniform_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        uniform_buf_info.size = std::max<size_t>(capture_size, 64); // At least 64 bytes
        uniform_buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        uniform_buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(backend_->device(), &uniform_buf_info, nullptr, &captures_uniform_buffer);

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(backend_->device(), captures_uniform_buffer, &mem_reqs);

        VkMemoryAllocateInfo mem_alloc{};
        mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mem_alloc.allocationSize = mem_reqs.size;
        mem_alloc.memoryTypeIndex = memory_manager_->find_memory_type(mem_reqs.memoryTypeBits,
                                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(backend_->device(), &mem_alloc, nullptr, &captures_uniform_memory);
        vkBindBufferMemory(backend_->device(), captures_uniform_buffer, captures_uniform_memory, 0);

        // Copy captures data to uniform buffer
        if (capture_size > 0 && captures != nullptr) {
            void* mapped_data;
            vkMapMemory(backend_->device(), captures_uniform_memory, 0, capture_size, 0, &mapped_data);
            std::memcpy(mapped_data, captures, capture_size);
            vkUnmapMemory(backend_->device(), captures_uniform_memory);
        }

        buffer_infos.push_back({captures_uniform_buffer, 0, VK_WHOLE_SIZE});
        VkWriteDescriptorSet write2{};
        write2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write2.dstSet = descriptor_set;
        write2.dstBinding = 2;
        write2.dstArrayElement = 0;
        write2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write2.descriptorCount = 1;
        write2.pBufferInfo = &buffer_infos[buffer_infos.size() - 1];  // Use correct index
        writes.push_back(write2);

        vkUpdateDescriptorSets(backend_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        descriptor_cache_[key] = descriptor_set;

        transient_buffers_.emplace_back(captures_uniform_buffer, captures_uniform_memory);
    }

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

    // Push constants: count + arena bases (for pointer-chasing relocation).
    PushBlock push = make_push_block(count);
    vkCmdPushConstants(command_buffer_, pipeline_data.layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0,
                      sizeof(push), &push);

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
    return true;
}

bool KernelLauncher::dispatch_reduce_level(PipelineData& pipeline_data,
                                           VkBuffer src_buf, VkDeviceSize src_off, VkDeviceSize src_range,
                                           VkBuffer dst_buf, VkDeviceSize dst_off, VkDeviceSize dst_range,
                                           uint32_t count, uint32_t groups) {
    // Allocate a fresh descriptor set for this level (src/dst differ each level).
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "[reduce] Failed to allocate descriptor set" << std::endl;
        return false;
    }

    VkDescriptorBufferInfo infos[2]{};
    infos[0].buffer = src_buf; infos[0].offset = src_off; infos[0].range = src_range;
    infos[1].buffer = dst_buf; infos[1].offset = dst_off; infos[1].range = dst_range;

    // Binding 2 (captures uniform) is unused by the reduce kernel but present in
    // the shared descriptor layout; bind a small dummy so the set is complete.
    VkBuffer dummy_buf = VK_NULL_HANDLE;
    VkDeviceMemory dummy_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo ubi{};
    ubi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubi.size = 64;
    ubi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(backend_->device(), &ubi, nullptr, &dummy_buf);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(backend_->device(), dummy_buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_manager_->find_memory_type(
        mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(backend_->device(), &mai, nullptr, &dummy_mem);
    vkBindBufferMemory(backend_->device(), dummy_buf, dummy_mem, 0);
    VkDescriptorBufferInfo dummy_info{dummy_buf, 0, VK_WHOLE_SIZE};
    transient_buffers_.emplace_back(dummy_buf, dummy_mem);

    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &dummy_info;
    vkUpdateDescriptorSets(backend_->device(), 3, writes, 0, nullptr);

    sync();
    vkResetFences(backend_->device(), 1, &fence_);
    vkResetCommandBuffer(command_buffer_, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_data.layout, 0, 1, &descriptor_set, 0, nullptr);
    PushBlock push = make_push_block(count);
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(command_buffer_, groups, 1, 1);
    vkEndCommandBuffer(command_buffer_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    if (vkQueueSubmit(backend_->compute_queue(), 1, &submit_info, fence_) != VK_SUCCESS) {
        std::cerr << "[reduce] Failed to submit command buffer" << std::endl;
        return false;
    }
    fence_signaled_ = false;
    sync();  // each level depends on the previous one's output
    return true;
}

bool KernelLauncher::launch_reduce(const std::string& kernel_name, void* data, size_t count,
                                   size_t elem_size, void* out_result) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }
    auto& pipeline_data = it->second;

    if (count == 0) { std::memset(out_result, 0, elem_size); return true; }
    if (count == 1) { std::memcpy(out_result, data, elem_size); return true; }

    UnifiedArena* arena = get_global_arena();
    if (!arena || !arena->valid()) {
        std::cerr << "[reduce] requires the unified arena (scratch buffers)" << std::endl;
        return false;
    }

    // Two ping-pong scratch buffers in the arena, each large enough for one
    // level's partials (first level is the largest).
    size_t first_groups = (count + 255) / 256;
    void* scratch[2];
    scratch[0] = arena->allocate(first_groups * elem_size, 256);
    scratch[1] = arena->allocate(((first_groups + 255) / 256 + 1) * elem_size, 256);
    if (!scratch[0] || !scratch[1]) {
        std::cerr << "[reduce] arena scratch allocation failed" << std::endl;
        return false;
    }

    void*  src = data;
    size_t n = count;
    int    toggle = 0;
    while (n > 1) {
        uint32_t groups = static_cast<uint32_t>((n + 255) / 256);
        void* dst = scratch[toggle];

        // Resolve src binding (arena zero-copy, else register external buffer).
        VkBuffer src_buf; VkDeviceSize src_off; VkDeviceSize src_range = n * elem_size;
        if (arena->contains(src)) {
            src_buf = arena->buffer();
            src_off = arena->offset_of(src);
        } else {
            VkBuffer reg = memory_manager_->get_buffer(src);
            if (reg == VK_NULL_HANDLE) {
                memory_manager_->register_external_buffer(src, n * elem_size);
                reg = memory_manager_->get_buffer(src);
            }
            if (reg == VK_NULL_HANDLE) { std::cerr << "[reduce] bad src buffer" << std::endl; return false; }
            memory_manager_->sync_before_kernel(src);
            src_buf = reg; src_off = 0; src_range = VK_WHOLE_SIZE;
        }

        // dst is always arena scratch -> zero-copy bind at its offset.
        VkBuffer dst_buf = arena->buffer();
        VkDeviceSize dst_off = arena->offset_of(dst);
        VkDeviceSize dst_range = static_cast<VkDeviceSize>(groups) * elem_size;

        if (!dispatch_reduce_level(pipeline_data, src_buf, src_off, src_range,
                                   dst_buf, dst_off, dst_range,
                                   static_cast<uint32_t>(n), groups)) {
            return false;
        }

        src = dst;
        n = groups;
        toggle ^= 1;
    }

    // src now holds the single reduced element in host-coherent arena memory.
    std::memcpy(out_result, src, elem_size);
    arena->deallocate(scratch[0]);
    arena->deallocate(scratch[1]);
    return true;
}

bool KernelLauncher::launch_scan(const std::string& scan_kernel, const std::string& add_kernel,
                                 void* data, size_t count, size_t elem_size) {
    auto sit = pipelines_.find(scan_kernel);
    auto ait = pipelines_.find(add_kernel);
    if (sit == pipelines_.end() || ait == pipelines_.end()) {
        std::cerr << "[scan] kernel not found" << std::endl;
        return false;
    }
    if (count <= 1) return true;  // inclusive scan of <=1 element is the identity

    UnifiedArena* arena = get_global_arena();
    if (!arena || !arena->valid()) { std::cerr << "[scan] requires arena" << std::endl; return false; }

    const uint32_t num_blocks = static_cast<uint32_t>((count + 255) / 256);
    if (num_blocks > 256) {
        std::cerr << "[scan] count exceeds the single-level block-sum limit (256*256)" << std::endl;
        return false;
    }

    // Resolve the in-place data buffer (arena zero-copy, else register + upload).
    const bool data_in_arena = arena->contains(data);
    VkBuffer data_buf; VkDeviceSize data_off; VkDeviceSize data_range = count * elem_size;
    if (data_in_arena) {
        data_buf = arena->buffer();
        data_off = arena->offset_of(data);
    } else {
        VkBuffer reg = memory_manager_->get_buffer(data);
        if (reg == VK_NULL_HANDLE) { memory_manager_->register_external_buffer(data, data_range); reg = memory_manager_->get_buffer(data); }
        if (reg == VK_NULL_HANDLE) { std::cerr << "[scan] bad data buffer" << std::endl; return false; }
        memory_manager_->sync_before_kernel(data);
        data_buf = reg; data_off = 0; data_range = VK_WHOLE_SIZE;
    }

    void* blocksums = arena->allocate(num_blocks * elem_size, 256);
    void* dummy = arena->allocate(elem_size, 256);
    if (!blocksums || !dummy) { std::cerr << "[scan] scratch alloc failed" << std::endl; return false; }
    VkBuffer bs_buf = arena->buffer();
    VkDeviceSize bs_off = arena->offset_of(blocksums);
    VkDeviceSize bs_range = static_cast<VkDeviceSize>(num_blocks) * elem_size;

    // Pass 1: per-block inclusive scan of data (in place) + write block totals.
    if (!dispatch_reduce_level(sit->second, data_buf, data_off, data_range,
                               bs_buf, bs_off, bs_range,
                               static_cast<uint32_t>(count), num_blocks))
        return false;

    if (num_blocks > 1) {
        // Pass 2: inclusive scan of the block sums (single workgroup, in place).
        VkBuffer dm_buf = arena->buffer();
        VkDeviceSize dm_off = arena->offset_of(dummy);
        if (!dispatch_reduce_level(sit->second, bs_buf, bs_off, bs_range,
                                   dm_buf, dm_off, elem_size, num_blocks, 1))
            return false;
        // Pass 3: add each block's exclusive offset (= scanned blocksums[wg-1]).
        if (!dispatch_reduce_level(ait->second, data_buf, data_off, data_range,
                                   bs_buf, bs_off, bs_range,
                                   static_cast<uint32_t>(count), num_blocks))
            return false;
    }

    if (!data_in_arena) memory_manager_->sync_after_kernel(data);  // download in-place result
    arena->deallocate(blocksums);
    arena->deallocate(dummy);
    return true;
}

// Push block for the bitonic stage: { uint count, uint k, uint j } (12 bytes, fits
// the shared 24-byte push range). k/j select the compare-exchange schedule.
namespace { struct SortPush { uint32_t count; uint32_t k; uint32_t j; }; }

bool KernelLauncher::dispatch_sort_stage(PipelineData& pipeline_data,
                                         VkBuffer data_buf, VkDeviceSize data_off, VkDeviceSize data_range,
                                         uint32_t count, uint32_t k, uint32_t j, uint32_t groups) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "[sort] Failed to allocate descriptor set" << std::endl;
        return false;
    }

    // The kernel uses only binding 0; bind the same buffer at 1 and a dummy uniform
    // at 2 so the shared 3-binding layout is fully populated (validation-clean).
    VkDescriptorBufferInfo infos[2]{};
    infos[0].buffer = data_buf; infos[0].offset = data_off; infos[0].range = data_range;
    infos[1] = infos[0];

    VkBuffer dummy_buf = VK_NULL_HANDLE;
    VkDeviceMemory dummy_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo ubi{};
    ubi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubi.size = 64;
    ubi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(backend_->device(), &ubi, nullptr, &dummy_buf);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(backend_->device(), dummy_buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_manager_->find_memory_type(
        mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(backend_->device(), &mai, nullptr, &dummy_mem);
    vkBindBufferMemory(backend_->device(), dummy_buf, dummy_mem, 0);
    VkDescriptorBufferInfo dummy_info{dummy_buf, 0, VK_WHOLE_SIZE};
    transient_buffers_.emplace_back(dummy_buf, dummy_mem);

    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &dummy_info;
    vkUpdateDescriptorSets(backend_->device(), 3, writes, 0, nullptr);

    sync();
    vkResetFences(backend_->device(), 1, &fence_);
    vkResetCommandBuffer(command_buffer_, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_data.layout, 0, 1, &descriptor_set, 0, nullptr);
    SortPush push{count, k, j};
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(command_buffer_, groups, 1, 1);
    vkEndCommandBuffer(command_buffer_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    if (vkQueueSubmit(backend_->compute_queue(), 1, &submit_info, fence_) != VK_SUCCESS) {
        std::cerr << "[sort] Failed to submit command buffer" << std::endl;
        return false;
    }
    fence_signaled_ = false;
    sync();  // each stage depends on the previous stage's writes
    return true;
}

bool KernelLauncher::launch_sort(const std::string& kernel_name, void* data, size_t count,
                                 size_t elem_size) {
    auto it = pipelines_.find(kernel_name);
    if (it == pipelines_.end()) { std::cerr << "[sort] kernel not found" << std::endl; return false; }
    if (count <= 1) return true;  // already sorted

    // MVP: bitonic sort requires a power-of-two element count.
    if ((count & (count - 1)) != 0) {
        std::cerr << "[sort] count " << count << " is not a power of two (MVP limit)" << std::endl;
        return false;
    }

    UnifiedArena* arena = get_global_arena();
    const bool data_in_arena = arena && arena->valid() && arena->contains(data);
    VkBuffer data_buf; VkDeviceSize data_off; VkDeviceSize data_range = count * elem_size;
    if (data_in_arena) {
        data_buf = arena->buffer();
        data_off = arena->offset_of(data);
    } else {
        VkBuffer reg = memory_manager_->get_buffer(data);
        if (reg == VK_NULL_HANDLE) { memory_manager_->register_external_buffer(data, data_range); reg = memory_manager_->get_buffer(data); }
        if (reg == VK_NULL_HANDLE) { std::cerr << "[sort] bad data buffer" << std::endl; return false; }
        memory_manager_->sync_before_kernel(data);
        data_buf = reg; data_off = 0; data_range = VK_WHOLE_SIZE;
    }

    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t groups = (n + 255) / 256;
    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            if (!dispatch_sort_stage(it->second, data_buf, data_off, data_range, n, k, j, groups))
                return false;
        }
    }

    if (!data_in_arena) memory_manager_->sync_after_kernel(data);
    return true;
}

bool KernelLauncher::dispatch_scatter(PipelineData& pipeline_data,
                                      VkBuffer in_buf, VkDeviceSize in_off, VkDeviceSize in_range,
                                      VkBuffer out_buf, VkDeviceSize out_off, VkDeviceSize out_range,
                                      VkBuffer pos_buf, VkDeviceSize pos_off, VkDeviceSize pos_range,
                                      uint32_t count, uint32_t groups, uint32_t num_true) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline_data.descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    if (vkAllocateDescriptorSets(backend_->device(), &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "[scatter] Failed to allocate descriptor set" << std::endl;
        return false;
    }

    VkDescriptorBufferInfo in_info{in_buf, in_off, in_range};
    VkDescriptorBufferInfo out_info{out_buf, out_off, out_range};
    VkDescriptorBufferInfo pos_info{pos_buf, pos_off, pos_range};

    // Dummy uniform for binding 2 (unused by the scatter kernel but in the layout).
    VkBuffer dummy_buf = VK_NULL_HANDLE;
    VkDeviceMemory dummy_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo ubi{};
    ubi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubi.size = 64;
    ubi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(backend_->device(), &ubi, nullptr, &dummy_buf);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(backend_->device(), dummy_buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_manager_->find_memory_type(
        mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(backend_->device(), &mai, nullptr, &dummy_mem);
    vkBindBufferMemory(backend_->device(), dummy_buf, dummy_mem, 0);
    VkDescriptorBufferInfo dummy_info{dummy_buf, 0, VK_WHOLE_SIZE};
    transient_buffers_.emplace_back(dummy_buf, dummy_mem);

    VkWriteDescriptorSet writes[4]{};
    auto storage_write = [&](int idx, uint32_t binding, VkDescriptorBufferInfo* bi) {
        writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[idx].dstSet = descriptor_set;
        writes[idx].dstBinding = binding;
        writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[idx].descriptorCount = 1;
        writes[idx].pBufferInfo = bi;
    };
    storage_write(0, 0, &in_info);
    storage_write(1, 1, &out_info);
    storage_write(2, 3, &pos_info);
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = descriptor_set;
    writes[3].dstBinding = 2;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &dummy_info;
    vkUpdateDescriptorSets(backend_->device(), 4, writes, 0, nullptr);

    sync();
    vkResetFences(backend_->device(), 1, &fence_);
    vkResetCommandBuffer(command_buffer_, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_data.layout, 0, 1, &descriptor_set, 0, nullptr);
    // Push { count, num_true }: copy_if's scatter reads only count; the partition
    // scatter also reads num_true (the size of the kept block).
    uint32_t push[2] = {count, num_true};
    vkCmdPushConstants(command_buffer_, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);
    vkCmdDispatch(command_buffer_, groups, 1, 1);
    vkEndCommandBuffer(command_buffer_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    if (vkQueueSubmit(backend_->compute_queue(), 1, &submit_info, fence_) != VK_SUCCESS) {
        std::cerr << "[scatter] Failed to submit command buffer" << std::endl;
        return false;
    }
    fence_signaled_ = false;
    sync();
    return true;
}

bool KernelLauncher::launch_compact(const std::string& flags_kernel, const std::string& scan_kernel,
                                    const std::string& add_kernel, const std::string& scatter_kernel,
                                    void* input, void* output, size_t count, size_t elem_size,
                                    bool elem_is_float, size_t* out_kept) {
    auto fit = pipelines_.find(flags_kernel);
    auto scit = pipelines_.find(scatter_kernel);
    if (fit == pipelines_.end() || scit == pipelines_.end()) {
        std::cerr << "[compact] kernel not found" << std::endl;
        return false;
    }
    if (out_kept) *out_kept = 0;
    if (count == 0) return true;

    UnifiedArena* arena = get_global_arena();
    if (!arena || !arena->valid()) { std::cerr << "[compact] requires arena" << std::endl; return false; }

    // Resolve input/output: arena zero-copy when possible, else register the external
    // buffer (upload the input; download the output afterwards). The arena often only
    // initializes at the first launch, so allocator-backed vectors built earlier take
    // this register path — same as launch_scan/sort/reduce. positions is arena scratch.
    const VkDeviceSize range = count * elem_size;
    const bool in_arena = arena->contains(input);
    const bool out_arena = arena->contains(output);
    VkBuffer in_buf, out_buf; VkDeviceSize in_off, out_off;
    VkDeviceSize in_range = range, out_range = range;
    if (in_arena) {
        in_buf = arena->buffer(); in_off = arena->offset_of(input);
    } else {
        VkBuffer reg = memory_manager_->get_buffer(input);
        if (reg == VK_NULL_HANDLE) { memory_manager_->register_external_buffer(input, range); reg = memory_manager_->get_buffer(input); }
        if (reg == VK_NULL_HANDLE) { std::cerr << "[compact] bad input buffer" << std::endl; return false; }
        memory_manager_->sync_before_kernel(input);
        in_buf = reg; in_off = 0; in_range = VK_WHOLE_SIZE;
    }
    if (out_arena) {
        out_buf = arena->buffer(); out_off = arena->offset_of(output);
    } else {
        VkBuffer reg = memory_manager_->get_buffer(output);
        if (reg == VK_NULL_HANDLE) { memory_manager_->register_external_buffer(output, range); reg = memory_manager_->get_buffer(output); }
        if (reg == VK_NULL_HANDLE) { std::cerr << "[compact] bad output buffer" << std::endl; return false; }
        out_buf = reg; out_off = 0; out_range = VK_WHOLE_SIZE;
    }

    // positions scratch (also receives the flags, scanned in place into positions).
    void* positions = arena->allocate(count * elem_size, 256);
    if (!positions) { std::cerr << "[compact] scratch alloc failed" << std::endl; return false; }
    VkBuffer pos_buf = arena->buffer();
    VkDeviceSize pos_off = arena->offset_of(positions);

    const uint32_t groups = static_cast<uint32_t>((count + 255) / 256);

    // 1. flags: input -> positions (1.0 if kept, else 0.0).
    if (!dispatch_reduce_level(fit->second, in_buf, in_off, in_range, pos_buf, pos_off, range,
                               static_cast<uint32_t>(count), groups)) {
        arena->deallocate(positions); return false;
    }

    // 2. inclusive scan of the flags, in place -> positions[i] = #kept in [0..i].
    if (!launch_scan(scan_kernel, add_kernel, positions, count, elem_size)) {
        arena->deallocate(positions); return false;
    }

    // 3. kept count = the last inclusive-scan value (arena is host-mapped). The
    // positions buffer holds the element type, so read float or int32 accordingly.
    size_t kept = elem_is_float
                      ? static_cast<size_t>(static_cast<float*>(positions)[count - 1])
                      : static_cast<size_t>(static_cast<int32_t*>(positions)[count - 1]);
    if (out_kept) *out_kept = kept;

    // 4. scatter into the output. num_true (= kept) is read only by the partition
    // scatter (which writes every element); copy_if's scatter ignores it.
    if (!dispatch_scatter(scit->second, in_buf, in_off, in_range, out_buf, out_off, out_range,
                          pos_buf, pos_off, range, static_cast<uint32_t>(count), groups,
                          static_cast<uint32_t>(kept))) {
        arena->deallocate(positions); return false;
    }

    if (!out_arena) memory_manager_->sync_after_kernel(output);  // download compacted result
    arena->deallocate(positions);
    return true;
}

} // namespace parallax
