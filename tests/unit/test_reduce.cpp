// Phase 3 de-risk: prove the parallel-reduction primitive end-to-end. Loads the
// workgroup-reduction kernel (reduce.comp -> reduce.spv), fills an arena buffer
// with a known pattern, and checks parallax_reduce() returns the exact sum via
// iterative multi-level GPU dispatch. Skips cleanly without a device/arena.

#include "parallax/runtime.hpp"
#include "parallax/runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef REDUCE_SPV
#define REDUCE_SPV "reduce.spv"
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
    if (!backend || !arena || !arena->valid()) {
        std::printf("SKIP: no Vulkan device / arena\n");
        return 0;
    }

    std::vector<uint32_t> spv = read_spv(REDUCE_SPV);
    if (spv.empty()) { std::fprintf(stderr, "FAIL: could not read %s\n", REDUCE_SPV); return 1; }

    parallax_kernel_t kernel = parallax_kernel_load(spv.data(), spv.size());
    if (!kernel) { std::fprintf(stderr, "FAIL: could not load reduce kernel\n"); return 1; }

    // N deliberately not a multiple of 256 to exercise the tail-padding path, and
    // a multiple of 8 so the i%8 pattern sums exactly (625*28 = 17500, < 2^24).
    const uint32_t N = 5000;
    auto* data = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    if (!data) { std::fprintf(stderr, "FAIL: arena alloc\n"); return 1; }

    float expected = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i % 8);
        expected += data[i];
    }

    float result = -1.0f;
    parallax_reduce(kernel, data, N, sizeof(float), &result);

    std::printf("reduce result=%.1f expected=%.1f\n", result, expected);
    if (result != expected) {
        std::fprintf(stderr, "FAIL: reduce mismatch (got %.3f, want %.3f)\n", result, expected);
        return 1;
    }
    std::printf("PASS: GPU parallel reduction is exact\n");
    return 0;
}
