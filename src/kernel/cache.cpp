#include "parallax/runtime.h"
#include "parallax/runtime.hpp"
#include "parallax/kernel_launcher.hpp"
#include <memory>
#include <string>
#include <cstdarg>
#include <iostream>
#include <atomic>

namespace {
    // Global kernel launcher instance
    static std::unique_ptr<parallax::KernelLauncher> g_kernel_launcher;
    static std::atomic<uint64_t> g_kernel_counter{0};

    // Kernel handle is just a heap-allocated string containing the kernel name
    struct KernelHandle {
        std::string name;
    };

    bool ensure_kernel_launcher_initialized() {
        if (g_kernel_launcher) return true;

        auto* backend = parallax::get_global_backend();
        auto* memory_manager = parallax::get_global_memory_manager();

        if (!backend || !memory_manager) {
            std::cerr << "[Parallax] Runtime not initialized" << std::endl;
            return false;
        }

        g_kernel_launcher = std::make_unique<parallax::KernelLauncher>(backend, memory_manager);
        std::cout << "[Parallax] KernelLauncher initialized" << std::endl;
        return true;
    }
}

parallax_kernel_t parallax_kernel_load(const unsigned int* spirv, size_t words) {
    if (!ensure_kernel_launcher_initialized()) {
        std::cerr << "[parallax_kernel_load] Failed to initialize launcher" << std::endl;
        return nullptr;
    }

    // Generate unique kernel name
    uint64_t id = g_kernel_counter.fetch_add(1);
    std::string kernel_name = "kernel_" + std::to_string(id);

    std::cout << "[parallax_kernel_load] Loading kernel: " << kernel_name
              << " (" << words << " SPIR-V words)" << std::endl;

    // Load kernel (size in bytes = words * 4)
    bool success = g_kernel_launcher->load_kernel(kernel_name, spirv, words * 4);

    if (!success) {
        std::cerr << "[parallax_kernel_load] Failed to load kernel" << std::endl;
        return nullptr;
    }

    // Create handle
    auto* handle = new KernelHandle{kernel_name};
    std::cout << "[parallax_kernel_load] Successfully loaded kernel: " << kernel_name << std::endl;
    return reinterpret_cast<parallax_kernel_t>(handle);
}

void parallax_kernel_launch(parallax_kernel_t kernel, ...) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_kernel_launch] Invalid kernel or launcher not initialized" << std::endl;
        return;
    }

    auto* handle = reinterpret_cast<KernelHandle*>(kernel);

    // Extract variadic arguments: (void* buffer, size_t count)
    va_list args;
    va_start(args, kernel);
    void* buffer = va_arg(args, void*);
    size_t count = va_arg(args, size_t);
    va_end(args);

    std::cout << "[parallax_kernel_launch] Launching kernel: " << handle->name
              << " with buffer=" << buffer << ", count=" << count << std::endl;

    // Launch kernel
    bool success = g_kernel_launcher->launch(handle->name, buffer, count);

    if (!success) {
        std::cerr << "[parallax_kernel_launch] Failed to launch kernel" << std::endl;
        return;
    }

    // Wait for completion
    std::cout << "[parallax_kernel_launch] Waiting for kernel completion..." << std::endl;
    g_kernel_launcher->sync();

    // Sync back to host
    auto* memory_manager = parallax::get_global_memory_manager();
    if (memory_manager) {
        memory_manager->sync_after_kernel(buffer);
    }

    std::cout << "[parallax_kernel_launch] Kernel completed successfully" << std::endl;
}

void parallax_kernel_launch_transform(parallax_kernel_t kernel, ...) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_kernel_launch_transform] Invalid kernel or launcher not initialized" << std::endl;
        return;
    }

    auto* handle = reinterpret_cast<KernelHandle*>(kernel);

    // Extract variadic arguments: (void* in_buffer, void* out_buffer, size_t count)
    va_list args;
    va_start(args, kernel);
    void* in_buffer = va_arg(args, void*);
    void* out_buffer = va_arg(args, void*);
    size_t count = va_arg(args, size_t);
    va_end(args);

    std::cout << "[parallax_kernel_launch_transform] Launching kernel: " << handle->name
              << " with in_buffer=" << in_buffer
              << ", out_buffer=" << out_buffer
              << ", count=" << count << std::endl;

    // Launch transform kernel (separate input/output buffers)
    bool success = g_kernel_launcher->launch_transform(handle->name, in_buffer, out_buffer, count);

    if (!success) {
        std::cerr << "[parallax_kernel_launch_transform] Failed to launch kernel" << std::endl;
        return;
    }

    // Wait for completion
    std::cout << "[parallax_kernel_launch_transform] Waiting for kernel completion..." << std::endl;
    g_kernel_launcher->sync();

    // Sync output buffer back to host
    auto* memory_manager = parallax::get_global_memory_manager();
    if (memory_manager) {
        memory_manager->sync_after_kernel(out_buffer);
    }

    std::cout << "[parallax_kernel_launch_transform] Kernel completed successfully" << std::endl;
}
