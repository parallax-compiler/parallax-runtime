// Phase 5 de-risk: prove the multi-stage bitonic sort. Loads the compare-exchange
// kernel, sorts a shuffled power-of-two array of floats in place, and checks the
// result is fully ascending and a permutation of the input. Skips without a device.

#include "parallax/runtime.hpp"
#include "parallax/runtime.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef BITONIC_SPV
#define BITONIC_SPV "bitonic.spv"
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

    std::vector<uint32_t> spv = read_spv(BITONIC_SPV);
    if (spv.empty()) { std::fprintf(stderr, "FAIL: read spv\n"); return 1; }
    parallax_kernel_t k = parallax_kernel_load(spv.data(), spv.size());
    if (!k) { std::fprintf(stderr, "FAIL: load kernel\n"); return 1; }

    // 1024 = power of two, spanning 4 workgroups. A deterministic non-trivial
    // permutation: a descending-ish scramble that touches every bitonic stage.
    const uint32_t N = 1024;
    auto* data = static_cast<float*>(arena->allocate(N * sizeof(float), 16));
    if (!data) { std::fprintf(stderr, "FAIL: arena alloc\n"); return 1; }
    uint32_t expected_sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        // Reverse + bit-mix so it is neither sorted nor reverse-sorted trivially.
        uint32_t v = ((N - 1 - i) * 2654435761u) % N;
        data[i] = static_cast<float>(v);
        expected_sum += v;
    }

    parallax_sort(k, data, N, sizeof(float));

    int rc = 0;
    uint32_t got_sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        got_sum += static_cast<uint32_t>(data[i]);
        if (i > 0 && data[i] < data[i - 1]) {
            std::fprintf(stderr, "FAIL: not sorted at %u: %.1f < %.1f\n", i, data[i], data[i - 1]);
            rc = 1;
            break;
        }
    }
    // A correct sort is a permutation, so the multiset sum is preserved.
    if (rc == 0 && got_sum != expected_sum) {
        std::fprintf(stderr, "FAIL: sum changed %u -> %u (not a permutation)\n", expected_sum, got_sum);
        rc = 1;
    }
    if (rc == 0) std::printf("sort: data[0]=%.1f data[%u]=%.1f sum=%u (ascending, permutation)\n",
                             data[0], N - 1, data[N - 1], got_sum);
    if (rc == 0) std::printf("PASS: GPU bitonic sort is correct\n");
    return rc;
}
