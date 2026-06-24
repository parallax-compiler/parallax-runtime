// Unit test for UnifiedArena. Runs against whatever Vulkan device is present
// (lavapipe in CI). Skips cleanly (exit 0) when no device is available.

#include "parallax/arena.hpp"
#include "parallax/vulkan_backend.hpp"

#include <cstdint>
#include <cstdio>

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: %s\n", msg);      \
            return 1;                                     \
        }                                                 \
    } while (0)

int main() {
    parallax::VulkanBackend backend;
    if (!backend.initialize()) {
        std::printf("SKIP: no Vulkan device available\n");
        return 0;  // environment without a GPU/software driver — not a failure
    }

    parallax::UnifiedArena arena;
    CHECK(arena.initialize(&backend, 16ull * 1024 * 1024), "arena init");
    CHECK(arena.valid(), "arena valid");

    const auto& caps = arena.capabilities();
    std::printf("caps: int64=%d float64=%d bda=%d\n",
                caps.shader_int64, caps.shader_float64, caps.buffer_device_address);

    // Allocate two blocks; verify containment, ordering, alignment.
    auto* a = static_cast<int*>(arena.allocate(1000 * sizeof(int)));
    auto* b = static_cast<float*>(arena.allocate(500 * sizeof(float)));
    CHECK(a && b, "allocate returned non-null");
    CHECK(arena.contains(a) && arena.contains(b), "contains() for arena pointers");
    CHECK(arena.offset_of(b) > arena.offset_of(a), "second alloc has higher offset");
    CHECK((reinterpret_cast<uintptr_t>(a) % 256) == 0, "block a is 256-aligned");
    CHECK((reinterpret_cast<uintptr_t>(b) % 256) == 0, "block b is 256-aligned");

    // Host read/write through the mapped pointer.
    for (int i = 0; i < 1000; ++i) a[i] = i * 3;
    for (int i = 0; i < 1000; ++i) CHECK(a[i] == i * 3, "host read-back matches write");

    // Free + reallocate should reuse the freed block.
    const auto off_a = arena.offset_of(a);
    arena.deallocate(a);
    auto* c = static_cast<int*>(arena.allocate(1000 * sizeof(int)));
    CHECK(c != nullptr, "realloc after free");
    CHECK(arena.offset_of(c) == off_a, "freed block was reused");

    // A non-arena pointer must not be reported as contained.
    int on_stack = 42;
    CHECK(!arena.contains(&on_stack), "stack pointer not contained");

    // Exhaustion returns nullptr rather than crashing.
    CHECK(arena.allocate(64ull * 1024 * 1024) == nullptr, "over-capacity alloc returns null");

    std::printf("PASS: UnifiedArena basic operations\n");
    return 0;
}
