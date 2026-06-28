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

    # Phase 3: parallel reduction primitive (workgroup tree reduction).
    set(REDUCE_SPV ${CMAKE_CURRENT_BINARY_DIR}/reduce.spv)
    add_custom_command(
        OUTPUT ${REDUCE_SPV}
        COMMAND ${GLSLANG} -V --target-env vulkan1.2
                ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/reduce.comp -o ${REDUCE_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/reduce.comp
        COMMENT "Compiling reduce.comp -> reduce.spv")
    add_custom_target(reduce_spv DEPENDS ${REDUCE_SPV})

    add_executable(test_reduce unit/test_reduce.cpp)
    add_dependencies(test_reduce reduce_spv)
    target_compile_definitions(test_reduce PRIVATE REDUCE_SPV="${REDUCE_SPV}")
    target_link_libraries(test_reduce PRIVATE parallax-runtime)
    add_test(NAME ParallelReduce COMMAND test_reduce)

    # Phase 5: inclusive prefix scan (per-block scan + add block offsets).
    set(SCAN_SPV ${CMAKE_CURRENT_BINARY_DIR}/scan.spv)
    set(SCAN_ADD_SPV ${CMAKE_CURRENT_BINARY_DIR}/scan_add.spv)
    add_custom_command(OUTPUT ${SCAN_SPV}
        COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/scan.comp -o ${SCAN_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/scan.comp COMMENT "Compiling scan.comp")
    add_custom_command(OUTPUT ${SCAN_ADD_SPV}
        COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/scan_add.comp -o ${SCAN_ADD_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/scan_add.comp COMMENT "Compiling scan_add.comp")
    add_custom_target(scan_spv DEPENDS ${SCAN_SPV} ${SCAN_ADD_SPV})

    add_executable(test_scan unit/test_scan.cpp)
    add_dependencies(test_scan scan_spv)
    target_compile_definitions(test_scan PRIVATE SCAN_SPV="${SCAN_SPV}" SCAN_ADD_SPV="${SCAN_ADD_SPV}")
    target_link_libraries(test_scan PRIVATE parallax-runtime)
    add_test(NAME ParallelScan COMMAND test_scan)

    # Phase 5: bitonic sort (global compare-exchange stage).
    set(BITONIC_SPV ${CMAKE_CURRENT_BINARY_DIR}/bitonic.spv)
    add_custom_command(OUTPUT ${BITONIC_SPV}
        COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/bitonic.comp -o ${BITONIC_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/bitonic.comp COMMENT "Compiling bitonic.comp")
    add_custom_target(bitonic_spv DEPENDS ${BITONIC_SPV})

    add_executable(test_sort unit/test_sort.cpp)
    add_dependencies(test_sort bitonic_spv)
    target_compile_definitions(test_sort PRIVATE BITONIC_SPV="${BITONIC_SPV}")
    target_link_libraries(test_sort PRIVATE parallax-runtime)
    add_test(NAME ParallelSort COMMAND test_sort)
else()
    message(STATUS "glslangValidator not found; skipping PhysPtrRelocation/ParallelReduce tests")
endif()

