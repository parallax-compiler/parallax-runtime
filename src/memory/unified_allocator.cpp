#include "parallax/runtime.h"
#include "parallax/unified_buffer.hpp"
#include "parallax/vulkan_backend.hpp"
#include <cstdlib>
#include <memory>
#include <iostream>

// Global runtime state
static std::unique_ptr<parallax::VulkanBackend> g_backend;
static std::unique_ptr<parallax::MemoryManager> g_memory_manager;
static bool g_initialized = false;

static bool ensure_initialized() {
    if (g_initialized) return true;
    
    g_backend = std::make_unique<parallax::VulkanBackend>();
    if (!g_backend->initialize()) {
        std::cerr << "Parallax: Vulkan initialization failed, using CPU fallback" << std::endl;
        g_backend.reset();
        return false;
    }
    
    g_memory_manager = std::make_unique<parallax::MemoryManager>(g_backend.get());
    g_initialized = true;
    return true;
}

void* parallax_umalloc(size_t size, unsigned flags) {
    if (ensure_initialized() && g_memory_manager) {
        return g_memory_manager->allocate(size);
    }
    // CPU fallback
    return std::malloc(size);
}

void parallax_ufree(void* ptr) {
    if (g_memory_manager) {
        g_memory_manager->deallocate(ptr);
    } else {
        std::free(ptr);
    }
}

void parallax_sync(void* ptr, int direction) {
    if (g_memory_manager) {
        auto dir = (direction == 0) ? parallax::SyncDirection::HOST_TO_DEVICE 
                                    : parallax::SyncDirection::DEVICE_TO_HOST;
        g_memory_manager->sync(ptr, dir);
    }
}

