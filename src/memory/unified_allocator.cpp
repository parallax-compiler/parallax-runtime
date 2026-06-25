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

static bool ensure_initialized() {
    if (g_initialized) return true;
    
    g_backend = std::make_unique<VulkanBackend>();
    if (!g_backend->initialize()) {
        // Vulkan unavailable - CPU fallback will be used
        g_backend.reset();
        return false;
    }
    
    g_memory_manager = std::make_unique<MemoryManager>(g_backend.get());
    g_initialized = true;
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
    if (!ensure_initialized()) return nullptr;
    if (!g_arena) {
        g_arena = std::make_unique<UnifiedArena>();
        if (!g_arena->initialize(g_backend.get())) {
            g_arena.reset();
            return nullptr;
        }
    }
    return g_arena.get();
}

} // namespace parallax

extern "C" {

// C accessors used by the opt-in allocation-interposition library (Phase 1d).
void* parallax_arena_alloc(size_t size, size_t align) {
    auto* arena = parallax::get_global_arena();
    return arena ? arena->allocate(size, align ? align : 16) : nullptr;
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
