#ifndef PARALLAX_RUNTIME_HPP
#define PARALLAX_RUNTIME_HPP

#include "parallax/vulkan_backend.hpp"
#include "parallax/unified_buffer.hpp"
#include "parallax/arena.hpp"

namespace parallax {

// Global accessors for the shared runtime state
VulkanBackend* get_global_backend();
MemoryManager* get_global_memory_manager();

// Lazily-created process-wide unified-memory arena. Returns nullptr if no Vulkan
// device is available. Used by the opt-in allocation interposition (Phase 1d).
UnifiedArena* get_global_arena();

} // namespace parallax

#endif // PARALLAX_RUNTIME_HPP
