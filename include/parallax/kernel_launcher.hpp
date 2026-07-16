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

    // Transform launch (in/out buffers + count). out_elem_size may differ from the
    // input element size (e.g. a float -> double map); 0 means "same as in".
    bool launch_transform(const std::string& kernel_name, void* in_buffer, void* out_buffer,
                          size_t count, size_t elem_size = sizeof(float), size_t out_elem_size = 0);

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

    // Inclusive prefix scan (Phase 5). Scans `data` in place: per-block scan
    // (scan_kernel) writing block totals, scan of those totals, then add the
    // exclusive block offsets back (add_kernel). MVP: count <= 256*256 (the block
    // sums fit one workgroup). Scratch is taken from the arena.
    bool launch_scan(const std::string& scan_kernel, const std::string& add_kernel,
                     void* data, size_t count, size_t elem_size);

    // Exclusive prefix scan (Phase 5), fully on-GPU. Runs the inclusive scan over
    // `input` (in place; caller passes a scratch copy) via scan_kernel+add_kernel, then
    // the shift_kernel writes `output[i] = init + (i>0 ? incl[i-1] : 0)`. `init` points
    // at elem_size bytes (the caller's init value). Default '+'. Same count limit as scan.
    bool launch_exclusive_scan(const std::string& scan_kernel, const std::string& add_kernel,
                               const std::string& shift_kernel, void* input, void* output,
                               size_t count, size_t elem_size, const void* init);

    // Bitonic sort (Phase 5). Sorts `data` in place ascending. The kernel is a
    // global compare-exchange stage dispatched O(log^2 n) times over the (k,j)
    // schedule. MVP: count must be a power of two (the caller pads otherwise).
    bool launch_sort(const std::string& kernel_name, void* data, size_t count,
                     size_t elem_size);

    // Stream compaction / copy_if (Phase 5). flags_kernel writes 1/0 per element
    // (predicate), scan_kernel+add_kernel turn that into output positions, and
    // scatter_kernel writes each kept element to output[pos-1]. Returns the number
    // of kept elements via out_kept. input/output/scratch are arena-backed.
    bool launch_compact(const std::string& flags_kernel, const std::string& scan_kernel,
                        const std::string& add_kernel, const std::string& scatter_kernel,
                        void* input, void* output, size_t count, size_t elem_size,
                        bool elem_is_float, size_t* out_kept);
    // Partition reuses launch_compact: passing the partition scatter kernel (which
    // reads num_true and writes every element) turns compaction into a partition.

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

    // One bitonic compare-exchange stage: bind `data` at binding 0 (and a dummy at
    // 1/2 to complete the shared layout), push { count, k, j }, dispatch `groups`
    // workgroups. The kernel swaps each in-pair element with its i^j partner.
    bool dispatch_sort_stage(PipelineData& pipeline_data,
                             VkBuffer data_buf, VkDeviceSize data_off, VkDeviceSize data_range,
                             uint32_t count, uint32_t k, uint32_t j, uint32_t groups);

    // Exclusive-scan finalize: bind in@0 (inclusive scan), out@1 (+ dummy uniform@2),
    // push { uint count@0, elem init@8 } (init_bytes = elem_size), dispatch `groups`
    // workgroups. Writes out[i] = init + (i>0 ? in[i-1] : 0).
    bool dispatch_exclusive_shift(PipelineData& pipeline_data,
                                  VkBuffer in_buf, VkDeviceSize in_off, VkDeviceSize in_range,
                                  VkBuffer out_buf, VkDeviceSize out_off, VkDeviceSize out_range,
                                  uint32_t count, const void* init, size_t init_bytes,
                                  uint32_t groups);

    // Compaction scatter: bind input@0, output@1, positions@3 (+ dummy uniform@2),
    // push { count }, dispatch `groups` workgroups. Writes each kept element to its
    // compacted slot. Needs the 3-storage descriptor layout (binding 3).
    bool dispatch_scatter(PipelineData& pipeline_data,
                          VkBuffer in_buf, VkDeviceSize in_off, VkDeviceSize in_range,
                          VkBuffer out_buf, VkDeviceSize out_off, VkDeviceSize out_range,
                          VkBuffer pos_buf, VkDeviceSize pos_off, VkDeviceSize pos_range,
                          uint32_t count, uint32_t groups, uint32_t num_true = 0);

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
