#ifndef PARALLAX_RUNTIME_HPP
#define PARALLAX_RUNTIME_HPP

#include "parallax/vulkan_backend.hpp"
#include "parallax/unified_buffer.hpp"

namespace parallax {

// Global accessors for the shared runtime state
VulkanBackend* get_global_backend();
MemoryManager* get_global_memory_manager();

} // namespace parallax

#endif // PARALLAX_RUNTIME_HPP
