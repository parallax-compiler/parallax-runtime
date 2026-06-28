// Phase 5 de-risk: prove the multi-block inclusive prefix scan. Loads the per-block
// scan + add-offsets kernels, scans an arena buffer of ones, and checks the result
// is [1,2,3,...,N] across block boundaries. Skips cleanly without a device/arena.

#include "parallax/runtime.hpp"
#include "parallax/runtime.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

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
}  // namespace

int main() {
    auto* backend = parallax::get_global_backend();
    auto* arena = parallax::get_global_arena();
    if (!backend || !arena || !arena->valid()) { std::printf("SKIP: no device/arena\n"); return 0; }

    std::vector<uint32_t> scan_spv = read_spv(SCAN_SPV);
    std::vector<uint32_t> add_spv = read_spv(SCAN_ADD_SPV);
    if (scan_spv.empty() || add_spv.empty()) { std::fprintf(stderr, "FAIL: read spv\n"); return 1; }

    parallax_kernel_t scan_k = parallax_kernel_load(scan_spv.data(), scan_spv.size());
    parallax_kernel_t add_k  = parallax_kernel_load(add_spv.data(), add_spv.size());
    if (!scan_k || !add_k) { std::fprintf(stderr, "FAIL: load kernels\n"); return 1; }

    // N spans 4 blocks (1000 = 3*256 + 232) to exercise the block-offset add.
    const uint32_t N = 1000;
    auto* data = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    if (!data) { std::fprintf(stderr, "FAIL: arena alloc\n"); return 1; }
    for (uint32_t i = 0; i < N; ++i) data[i] = 1.0f;

    parallax_scan(scan_k, add_k, data, N, sizeof(float));

    // Inclusive scan of all-ones is [1, 2, 3, ..., N].
    int rc = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (data[i] != static_cast<float>(i + 1)) {
            std::fprintf(stderr, "FAIL: scan[%u]=%.1f expected %u\n", i, data[i], i + 1);
            rc = 1;
            break;
        }
    }
    if (rc == 0) std::printf("scan result: data[0]=%.1f data[%u]=%.1f (expected 1, %u)\n",
                             data[0], N - 1, data[N - 1], N);
    if (rc == 0) std::printf("PASS: GPU inclusive prefix scan is correct across blocks\n");
    return rc;
}
