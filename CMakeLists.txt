cmake_minimum_required(VERSION 3.20)
project(parallax-runtime VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan 1.3 REQUIRED)

# Library
add_library(parallax-runtime SHARED
    src/memory/unified_allocator.cpp
    src/kernel/cache.cpp
    src/backend/vulkan/device.cpp
)

target_include_directories(parallax-runtime PUBLIC include)
target_link_libraries(parallax-runtime PUBLIC Vulkan::Vulkan)

# Tests
enable_testing()
add_subdirectory(tests)
