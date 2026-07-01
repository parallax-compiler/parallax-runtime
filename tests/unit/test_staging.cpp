// Discrete-GPU memory model de-risk: force the staging path (device-local device
// buffer + host-visible staging + explicit host<->device migration around launches)
// even on UMA hardware, and prove a multi-pass GPU reduction still returns the exact
// result THROUGH the migration. If flush/invalidate were wrong the device would
// reduce un-migrated garbage. Skips cleanly without a device.

#include "parallax/runtime.hpp"
#include "parallax/runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    // Force the discrete-GPU staging path BEFORE the arena initializes (it inits
    // lazily at the first backend/arena access below).
    setenv("PARALLAX_FORCE_STAGING", "1", 1);

    auto* backend = parallax::get_global_backend();
    auto* arena = parallax::get_global_arena();
    if (!backend || !arena || !arena->valid()) { std::printf("SKIP: no device/arena\n"); return 0; }

    if (arena->uma()) {
        std::fprintf(stderr, "FAIL: PARALLAX_FORCE_STAGING did not engage the staging path\n");
        return 1;
    }
    std::printf("staging engaged: arena->uma()=0 (device-local buffer + host staging)\n");

    std::vector<uint32_t> spv = read_spv(REDUCE_SPV);
    if (spv.empty()) { std::fprintf(stderr, "FAIL: read spv\n"); return 1; }
    parallax_kernel_t kernel = parallax_kernel_load(spv.data(), spv.size());
    if (!kernel) { std::fprintf(stderr, "FAIL: load kernel\n"); return 1; }

    // Multi-block, non-multiple-of-256 so the reduction runs several device passes
    // (each keeping data on the device buffer) with one flush in and one invalidate out.
    const uint32_t N = 5000;
    auto* data = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    if (!data) { std::fprintf(stderr, "FAIL: arena alloc\n"); return 1; }
    float expected = 0.0f;
    for (uint32_t i = 0; i < N; ++i) { data[i] = static_cast<float>(i % 8); expected += data[i]; }

    float result = -1.0f;
    parallax_reduce(kernel, data, N, sizeof(float), &result);

    std::printf("staged reduce result=%.1f expected=%.1f\n", result, expected);
    if (result != expected) {
        std::fprintf(stderr, "FAIL: staged reduction mismatch (migration bug)\n");
        return 1;
    }
    std::printf("PASS: GPU reduction correct through host<->device staging migration\n");
    return 0;
}
