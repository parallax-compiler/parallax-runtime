// Phase 2 de-risk: dereference an arena-allocated buffer on the GPU through
// buffer_device_address, using the relocation formula the compiler will emit.
// Builds a descriptor-less compute pipeline (the shader addresses memory purely
// by physical address) and checks the result. Skips cleanly without a device or
// without buffer_device_address support.

#include "parallax/runtime.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef RELOCATE_SPV
#define RELOCATE_SPV "relocate.spv"
#endif

namespace {
struct PushConstants {
    uint64_t host_ptr;
    uint64_t host_base;
    uint64_t dev_base;
    uint32_t count;
};

std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto size = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}
}  // namespace

int main() {
    auto* backend = parallax::get_global_backend();
    auto* arena = parallax::get_global_arena();
    if (!backend || !arena) {
        std::printf("SKIP: no Vulkan device / arena\n");
        return 0;
    }
    if (!arena->capabilities().buffer_device_address || arena->device_address() == 0) {
        std::printf("SKIP: device lacks buffer_device_address\n");
        return 0;
    }

    VkDevice device = backend->device();
    const uint32_t N = 64;

    auto* data = static_cast<float*>(arena->allocate(N * sizeof(float)));
    for (uint32_t i = 0; i < N; ++i) data[i] = 1.0f;

    std::vector<uint32_t> spv = read_spv(RELOCATE_SPV);
    if (spv.empty()) { std::fprintf(stderr, "FAIL: could not read %s\n", RELOCATE_SPV); return 1; }

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4;
    smci.pCode = spv.data();
    VkShaderModule module;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
        std::fprintf(stderr, "FAIL: shader module\n"); return 1;
    }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &plci, nullptr, &layout);

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) {
        std::fprintf(stderr, "FAIL: compute pipeline\n"); return 1;
    }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = backend->compute_queue_family();
    VkCommandPool pool;
    vkCreateCommandPool(device, &cpi, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbai, &cmd);

    PushConstants pc;
    pc.host_ptr = reinterpret_cast<uint64_t>(data);
    pc.host_base = reinterpret_cast<uint64_t>(arena->host_base());
    pc.dev_base = static_cast<uint64_t>(arena->device_address());
    pc.count = N;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmd, (N + 255) / 256, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(backend->compute_queue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(backend->compute_queue());

    int rc = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (data[i] != 3.0f) {
            std::fprintf(stderr, "FAIL: data[%u]=%f expected 3.0\n", i, data[i]);
            rc = 1;
            break;
        }
    }
    if (rc == 0) {
        std::printf("PASS: GPU dereferenced an arena pointer via relocation "
                    "(host_ptr=0x%llx host_base=0x%llx dev_base=0x%llx) -> 3.0\n",
                    (unsigned long long)pc.host_ptr, (unsigned long long)pc.host_base,
                    (unsigned long long)pc.dev_base);
    }

    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    return rc;
}
