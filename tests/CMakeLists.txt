# Unit tests
add_executable(test_vulkan_backend
    unit/test_vulkan_backend.cpp
)

target_link_libraries(test_vulkan_backend
    PRIVATE parallax-runtime
)

# Register test
add_test(NAME VulkanBackend COMMAND test_vulkan_backend)

# UnifiedArena (software unified-memory foundation)
add_executable(test_arena unit/test_arena.cpp)
target_link_libraries(test_arena PRIVATE parallax-runtime)
add_test(NAME UnifiedArena COMMAND test_arena)

