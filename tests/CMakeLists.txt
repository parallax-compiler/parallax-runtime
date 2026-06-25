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

# Allocation interposition (operator new -> arena)
add_executable(test_interpose unit/test_interpose.cpp)
target_link_libraries(test_interpose PRIVATE parallax-interpose parallax-runtime)
add_test(NAME Interpose COMMAND test_interpose)

# Phase 2: buffer_device_address pointer relocation. Requires a GLSL->SPIR-V
# compiler to build the buffer_reference shader; skipped if not present.
find_program(GLSLANG glslangValidator)
if(GLSLANG)
    set(RELOCATE_SPV ${CMAKE_CURRENT_BINARY_DIR}/relocate.spv)
    add_custom_command(
        OUTPUT ${RELOCATE_SPV}
        COMMAND ${GLSLANG} -V --target-env vulkan1.2
                ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/relocate.comp -o ${RELOCATE_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/relocate.comp
        COMMENT "Compiling relocate.comp -> relocate.spv")
    add_custom_target(relocate_spv DEPENDS ${RELOCATE_SPV})

    add_executable(test_physptr unit/test_physptr.cpp)
    add_dependencies(test_physptr relocate_spv)
    target_compile_definitions(test_physptr PRIVATE RELOCATE_SPV="${RELOCATE_SPV}")
    target_link_libraries(test_physptr PRIVATE parallax-runtime)
    add_test(NAME PhysPtrRelocation COMMAND test_physptr)
else()
    message(STATUS "glslangValidator not found; skipping PhysPtrRelocation test")
endif()

