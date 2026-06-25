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
    bool launch(const std::string& kernel_name, void* buffer, size_t count, float multiplier,
                size_t elem_size = sizeof(float));

    // Generic launch (buffer + count). elem_size is the per-element byte size of
    // the data buffer (e.g. sizeof(float)=4, sizeof(int64_t)=8); defaults to float
    // for backward compatibility.
    bool launch(const std::string& kernel_name, void* buffer, size_t count,
                size_t elem_size = sizeof(float));

    // Transform launch (in/out buffers + count)
    bool launch_transform(const std::string& kernel_name, void* in_buffer, void* out_buffer,
                          size_t count, size_t elem_size = sizeof(float));

    // NEW V2: Launch with captured parameters (for function objects)
    bool launch_with_captures(
        const std::string& kernel_name,
        void* buffer,
        size_t count,
        void* captures,
        size_t capture_size,
        size_t elem_size = sizeof(float)
    );

    // Parallel reduction (Phase 3). Reduces `count` elements of the data buffer to
    // a single scalar by dispatching the workgroup-reduction kernel iteratively
    // (data -> partials -> ... -> one element), ping-ponging through arena scratch.
    // The single reduced value (elem_size bytes) is written to out_result. The
    // kernel applies the '+' identity, so the caller combines any init separately.
    bool launch_reduce(const std::string& kernel_name, void* data, size_t count,
                       size_t elem_size, void* out_result);

    // Synchronize all pending operations
    void sync();

private:
    // One level of the iterative reduction: bind src@0 / dst@1, dispatch `groups`
    // workgroups over `count` elements. src/dst are already resolved to a VkBuffer
    // plus byte offset/range (arena zero-copy or registered external buffer).
    bool dispatch_reduce_level(PipelineData& pipeline_data,
                               VkBuffer src_buf, VkDeviceSize src_off, VkDeviceSize src_range,
                               VkBuffer dst_buf, VkDeviceSize dst_off, VkDeviceSize dst_range,
                               uint32_t count, uint32_t groups);

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

    // Per-launch scratch buffers (capture uniforms). Retired and destroyed by
    // retire_transient_buffers() once prior work has completed, and at teardown,
    // so they are not leaked past device destruction.
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> transient_buffers_;
    void retire_transient_buffers();
};

} // namespace parallax

#endif // PARALLAX_KERNEL_LAUNCHER_HPP
