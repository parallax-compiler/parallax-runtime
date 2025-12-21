# Unit tests
add_executable(test_vulkan_backend
    unit/test_vulkan_backend.cpp
)

target_link_libraries(test_vulkan_backend
    PRIVATE parallax-runtime
)

# Register test
add_test(NAME VulkanBackend COMMAND test_vulkan_backend)

