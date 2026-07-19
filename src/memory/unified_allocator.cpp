#include "parallax/runtime.hpp"
#include "parallax/runtime.h"
#include "parallax/unified_buffer.hpp"
#include "parallax/vulkan_backend.hpp"
#include "parallax/arena.hpp"
#include <cstdlib>
#include <memory>
#include <iostream>

namespace parallax {

// Global runtime state (exposed for testing)
static std::unique_ptr<VulkanBackend> g_backend;
static std::unique_ptr<MemoryManager> g_memory_manager;
static std::unique_ptr<UnifiedArena> g_arena;
static bool g_initialized = false;

static bool g_initializing = false;
static bool g_arena_initializing = false;

static bool ensure_initialized() {
    if (g_initialized) return true;
    // Vulkan initialization itself allocates; with allocation interposition active
    // those allocations re-enter here. Report "not ready" on re-entry so they fall
    // back to the system allocator instead of re-initializing (and corrupting) the
    // half-built backend.
    if (g_initializing) return false;
    g_initializing = true;

    g_backend = std::make_unique<VulkanBackend>();
    if (!g_backend->initialize()) {
        // Vulkan unavailable - CPU fallback will be used
        g_backend.reset();
        g_initializing = false;
        return false;
    }

    g_memory_manager = std::make_unique<MemoryManager>(g_backend.get());
    g_initialized = true;
    g_initializing = false;
    return true;
}

VulkanBackend* get_global_backend() {
    ensure_initialized();
    return g_backend.get();
}

MemoryManager* get_global_memory_manager() {
    ensure_initialized();
    return g_memory_manager.get();
}

UnifiedArena* get_global_arena() {
    if (g_arena) return g_arena.get();
    // Creating the arena (buffer + bookkeeping) allocates; under interposition that
    // re-enters here. Report "not ready" on re-entry so those allocations fall back
    // to the system allocator rather than recursively building another arena.
    if (g_arena_initializing) return nullptr;
    if (!ensure_initialized()) return nullptr;

    g_arena_initializing = true;
    auto arena = std::make_unique<UnifiedArena>();
    // Phase 3: prefer adopting the whole heap pool as the device buffer (so every heap
    // pointer is GPU-addressable with no copy). Falls back to the legacy per-arena buffer
    // when VK_EXT_external_memory_host is absent or the import fails, or when forced.
    const bool force_legacy = std::getenv("PARALLAX_FORCE_LEGACY_ARENA") != nullptr;
    bool ok = (!force_legacy && arena->initialize_from_pool(g_backend.get()));
    if (!ok) ok = arena->initialize(g_backend.get());
    if (!ok) {
        g_arena_initializing = false;
        return nullptr;
    }
    g_arena = std::move(arena);
    g_arena_initializing = false;
    return g_arena.get();
}

} // namespace parallax

extern "C" {

// C accessors used by the opt-in allocation-interposition library (Phase 1d).
void* parallax_arena_alloc(size_t size, size_t align) {
    auto* arena = parallax::get_global_arena();
    void* p = arena ? arena->allocate(size, align ? align : 16) : nullptr;
    // The funnels' staging path calls this; the zero-copy fast path does NOT. The marker
    // lets gates confirm staging was skipped for pool-resident data.
    static const bool dbg = std::getenv("PARALLAX_DEBUG") != nullptr;
    if (dbg) std::cerr << "[parallax_arena_alloc] staging " << size << " bytes -> " << p << std::endl;
    return p;
}

void parallax_arena_free(void* ptr) {
    auto* arena = parallax::get_global_arena();
    if (arena) arena->deallocate(ptr);
}

int parallax_arena_contains(const void* ptr) {
    auto* arena = parallax::get_global_arena();
    return (arena && arena->contains(ptr)) ? 1 : 0;
}

}  // extern "C"

using namespace parallax;

void* parallax_umalloc(size_t size, unsigned flags) {
    if (ensure_initialized() && g_memory_manager) {
        return g_memory_manager->allocate(size);
    }
    // CPU fallback - standard malloc
    return std::malloc(size);
}

void parallax_ufree(void* ptr) {
    if (g_memory_manager) {
        g_memory_manager->deallocate(ptr);
    } else {
        std::free(ptr);
    }
}
