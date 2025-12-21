#include "parallax/runtime.hpp"
#include "parallax/unified_buffer.hpp"
#include "parallax/vulkan_backend.hpp"
#include <cstdlib>
#include <memory>
#include <iostream>

namespace parallax {

// Global runtime state (exposed for testing)
static std::unique_ptr<VulkanBackend> g_backend;
static std::unique_ptr<MemoryManager> g_memory_manager;
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

} // namespace parallax

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
