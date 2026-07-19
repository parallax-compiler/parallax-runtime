cmake_minimum_required(VERSION 3.20)
project(parallax-runtime VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(PARALLAX_ENABLE_VALIDATION "Enable Vulkan validation layers" ON)
option(PARALLAX_BUILD_TESTS "Build test programs" ON)

# Find Vulkan
find_package(Vulkan 1.3 REQUIRED)

# Library sources
set(RUNTIME_SOURCES
    src/memory/unified_allocator.cpp
    src/memory/unified_buffer.cpp
    src/memory/arena.cpp
    src/memory/heap_pool.cpp
    src/kernel/cache.cpp
    src/kernel/kernel_launcher.cpp
    src/backend/vulkan/device.cpp
)

# Main library
add_library(parallax-runtime SHARED ${RUNTIME_SOURCES})

target_include_directories(parallax-runtime 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(parallax-runtime 
    PUBLIC 
        Vulkan::Vulkan
)

# Validation layers in debug mode
if(PARALLAX_ENABLE_VALIDATION)
    target_compile_definitions(parallax-runtime PRIVATE PARALLAX_ENABLE_VALIDATION)
endif()

# Tests
if(PARALLAX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS parallax-runtime
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/parallax
    DESTINATION include
)

# (The old opt-in parallax-interpose SHARED lib — operator-new-only, link-order dependent,
# stranded pre-init allocations — has been removed. parallax-heap below supersedes it:
# strong operator new + full malloc family, routed into the Vulkan-free heap pool.)

# Phase 2 whole-heap capture (nvc++-style). A STATIC library the compiler wrapper links
# into the program: its strong operator new / malloc-family overrides route every
# allocation into the heap pool (px_pool_*, which live in parallax-runtime), so all heap
# data is arena-importable and GPU-addressable. Static so the strong symbols win at the
# program's final link; px_pool_* resolve from the runtime shared lib.
add_library(parallax-heap STATIC src/memory/heap_capture.cpp)
target_include_directories(parallax-heap PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
install(TARGETS parallax-heap ARCHIVE DESTINATION lib)

