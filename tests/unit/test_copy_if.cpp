// Phase 5 de-risk: prove stream compaction (copy_if = flags + scan + scatter). Keeps
// elements > 0.5: input has 0.0 at even indices (dropped) and the index value at odd
// indices (kept), so the compacted output must be [1,3,5,...] and the kept count N/2.

#include "parallax/runtime.hpp"
#include "parallax/runtime.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef FLAGS_SPV
#define FLAGS_SPV "flags.spv"
#endif
#ifndef SCATTER_SPV
#define SCATTER_SPV "scatter.spv"
#endif
#ifndef SCAN_SPV
#define SCAN_SPV "scan.spv"
#endif
#ifndef SCAN_ADD_SPV
#define SCAN_ADD_SPV "scan_add.spv"
#endif

namespace {
std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto size = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}
parallax_kernel_t load(const char* path) {
    auto spv = read_spv(path);
    if (spv.empty()) return nullptr;
    return parallax_kernel_load(spv.data(), spv.size());
}
}  // namespace

int main() {
    auto* backend = parallax::get_global_backend();
    auto* arena = parallax::get_global_arena();
    if (!backend || !arena || !arena->valid()) { std::printf("SKIP: no device/arena\n"); return 0; }

    parallax_kernel_t flags_k = load(FLAGS_SPV);
    parallax_kernel_t scan_k  = load(SCAN_SPV);
    parallax_kernel_t add_k   = load(SCAN_ADD_SPV);
    parallax_kernel_t scat_k  = load(SCATTER_SPV);
    if (!flags_k || !scan_k || !add_k || !scat_k) { std::fprintf(stderr, "FAIL: load kernels\n"); return 1; }

    const uint32_t N = 1000;  // spans 4 scan blocks
    auto* input  = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    auto* output = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    if (!input || !output) { std::fprintf(stderr, "FAIL: arena alloc\n"); return 1; }
    for (uint32_t i = 0; i < N; ++i) {
        input[i] = (i % 2 == 0) ? 0.0f : static_cast<float>(i);  // keep the odd indices
        output[i] = -1.0f;
    }

    size_t kept = parallax_copy_if(flags_k, scan_k, add_k, scat_k, input, output, N, sizeof(float), 1);

    int rc = 0;
    if (kept != N / 2) { std::fprintf(stderr, "FAIL: kept=%zu expected %u\n", kept, N / 2); rc = 1; }
    // Compacted output must be the odd-index values in order: 1, 3, 5, ..., 999.
    for (uint32_t j = 0; rc == 0 && j < N / 2; ++j) {
        float expected = static_cast<float>(2 * j + 1);
        if (output[j] != expected) {
            std::fprintf(stderr, "FAIL: out[%u]=%.1f expected %.1f\n", j, output[j], expected);
            rc = 1;
        }
    }
    if (rc == 0) std::printf("copy_if: kept=%zu out[0]=%.1f out[%u]=%.1f (expected %u, 1, 999)\n",
                             kept, output[0], N / 2 - 1, output[N / 2 - 1], N / 2);
    if (rc == 0) std::printf("PASS: GPU stream compaction (copy_if) is correct\n");
    return rc;
}
