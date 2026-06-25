// Verifies that, with the interposition library linked, C++ heap allocations are
// served from the unified-memory arena (so they are GPU-addressable, zero-copy).
// Skips cleanly when no Vulkan device is available.

#include "parallax/runtime.hpp"

#include <cstdio>
#include <vector>

extern "C" int parallax_arena_contains(const void* ptr);

int main() {
    if (!parallax::get_global_arena()) {
        std::printf("SKIP: no Vulkan device / arena\n");
        return 0;
    }

    // A bare operator new[] should land in the arena.
    int* p = new int[256];
    const bool new_in_arena = parallax_arena_contains(p) != 0;
    std::printf("new int[256] -> %p in_arena=%d\n", static_cast<void*>(p), new_in_arena);
    delete[] p;

    // std::vector uses operator new under the hood -> arena-backed storage.
    std::vector<float> v(1000, 1.0f);
    const bool vec_in_arena = parallax_arena_contains(v.data()) != 0;
    std::printf("vector.data() -> %p in_arena=%d\n",
                static_cast<void*>(v.data()), vec_in_arena);

    if (!new_in_arena || !vec_in_arena) {
        std::printf("FAIL: heap allocations were not routed to the arena\n");
        return 1;
    }
    std::printf("PASS: operator new routed to the unified arena (zero-copy ready)\n");
    return 0;
}
