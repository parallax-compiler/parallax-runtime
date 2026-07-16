#include "parallax/runtime.h"
#include "parallax/runtime.hpp"
#include "parallax/kernel_launcher.hpp"
#include <memory>
#include <string>
#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <atomic>
#include <unordered_map>

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

namespace {
    // Layer A funnel registry. Keyed by the device_invoke instantiation's
    // __PRETTY_FUNCTION__ (the plugin emits the identical string). Registrars run
    // at static-init time, possibly before the backend exists, so we only record
    // the SPIR-V pointer here and load lazily on the first lookup.
    struct FunnelEntry {
        const unsigned int* spirv;
        size_t words;
        parallax_kernel_t handle;
        bool loaded;
    };
    std::unordered_map<std::string, FunnelEntry>& funnel_registry() {
        static std::unordered_map<std::string, FunnelEntry> reg;
        return reg;
    }
}

void parallax_kernel_register(const char* key, const unsigned int* spirv, size_t words) {
    if (!key) return;
    funnel_registry()[key] = FunnelEntry{spirv, words, nullptr, false};
    if (std::getenv("PARALLAX_DEBUG"))
        std::cerr << "[parallax_kernel_register] " << key << " (" << words << " words)\n";
}

parallax_kernel_t parallax_kernel_lookup(const char* key) {
    if (!key) return nullptr;
    auto& reg = funnel_registry();
    auto it = reg.find(key);
    if (it == reg.end()) {
        if (std::getenv("PARALLAX_DEBUG"))
            std::cerr << "[parallax_kernel_lookup] MISS: " << key << "\n";
        return nullptr;
    }
    if (!it->second.loaded) {
        it->second.handle = parallax_kernel_load(it->second.spirv, it->second.words);
        it->second.loaded = true;
    }
    return it->second.handle;
}

void parallax_kernel_launch(parallax_kernel_t kernel, ...) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_kernel_launch] Invalid kernel or launcher not initialized" << std::endl;
        return;
    }

    auto* handle = reinterpret_cast<KernelHandle*>(kernel);

    // Extract variadic arguments: (void* buffer, size_t count, size_t elem_size)
    va_list args;
    va_start(args, kernel);
    void* buffer = va_arg(args, void*);
    size_t count = va_arg(args, size_t);
    size_t elem_size = va_arg(args, size_t);
    va_end(args);

    std::cout << "[parallax_kernel_launch] Launching kernel: " << handle->name
              << " with buffer=" << buffer << ", count=" << count
              << ", elem_size=" << elem_size << std::endl;

    // Launch kernel
    bool success = g_kernel_launcher->launch(handle->name, buffer, count, elem_size);

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

    // Extract variadic arguments: (void* in_buffer, void* out_buffer, size_t count, size_t elem_size)
    va_list args;
    va_start(args, kernel);
    void* in_buffer = va_arg(args, void*);
    void* out_buffer = va_arg(args, void*);
    size_t count = va_arg(args, size_t);
    size_t elem_size = va_arg(args, size_t);
    va_end(args);

    std::cout << "[parallax_kernel_launch_transform] Launching kernel: " << handle->name
              << " with in_buffer=" << in_buffer
              << ", out_buffer=" << out_buffer
              << ", count=" << count << std::endl;

    // Launch transform kernel (separate input/output buffers)
    bool success = g_kernel_launcher->launch_transform(handle->name, in_buffer, out_buffer, count, elem_size);

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

// Transform with distinct input/output element sizes (e.g. float -> double). Fixed
// signature (not variadic) so the in/out sizes are unambiguous.
void parallax_kernel_launch_transform2(parallax_kernel_t kernel, void* in_buffer,
                                       void* out_buffer, size_t count,
                                       size_t in_elem_size, size_t out_elem_size) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_kernel_launch_transform2] Invalid kernel or launcher" << std::endl;
        return;
    }
    auto* handle = reinterpret_cast<KernelHandle*>(kernel);
    std::cout << "[parallax_kernel_launch_transform2] Launching kernel: " << handle->name
              << " in_elem=" << in_elem_size << " out_elem=" << out_elem_size
              << " count=" << count << std::endl;
    if (!g_kernel_launcher->launch_transform(handle->name, in_buffer, out_buffer, count,
                                             in_elem_size, out_elem_size)) {
        std::cerr << "[parallax_kernel_launch_transform2] Failed to launch kernel" << std::endl;
        return;
    }
    g_kernel_launcher->sync();
    auto* mm = parallax::get_global_memory_manager();
    if (mm) mm->sync_after_kernel(out_buffer);
    std::cout << "[parallax_kernel_launch_transform2] Kernel completed successfully" << std::endl;
}

// NEW V2: Kernel launch with captured parameters (for function objects)
void parallax_kernel_launch_with_captures(
    parallax_kernel_t kernel,
    void* buffer,
    size_t count,
    void* captures,
    size_t capture_size,
    size_t elem_size) {

    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_kernel_launch_with_captures] Invalid kernel or launcher not initialized" << std::endl;
        return;
    }

    auto* handle = reinterpret_cast<KernelHandle*>(kernel);

    std::cout << "[parallax_kernel_launch_with_captures] Launching kernel: " << handle->name
              << " with buffer=" << buffer
              << ", count=" << count
              << ", captures=" << captures
              << ", capture_size=" << capture_size << std::endl;

    // Launch kernel with captures
    bool success = g_kernel_launcher->launch_with_captures(
        handle->name, buffer, count, captures, capture_size, elem_size);

    if (!success) {
        std::cerr << "[parallax_kernel_launch_with_captures] Failed to launch kernel" << std::endl;
        return;
    }

    // Wait for completion
    std::cout << "[parallax_kernel_launch_with_captures] Waiting for kernel completion..." << std::endl;
    g_kernel_launcher->sync();

    // Sync buffer back to host
    auto* memory_manager = parallax::get_global_memory_manager();
    if (memory_manager) {
        memory_manager->sync_after_kernel(buffer);
    }

    std::cout << "[parallax_kernel_launch_with_captures] Kernel completed successfully" << std::endl;
}

void parallax_reduce(parallax_kernel_t kernel, void* data, size_t count,
                     size_t elem_size, void* result) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_reduce] Invalid kernel or launcher not initialized" << std::endl;
        return;
    }
    auto* handle = reinterpret_cast<KernelHandle*>(kernel);
    std::cout << "[parallax_reduce] Reducing kernel: " << handle->name
              << " count=" << count << " elem_size=" << elem_size << std::endl;
    if (!g_kernel_launcher->launch_reduce(handle->name, data, count, elem_size, result)) {
        std::cerr << "[parallax_reduce] reduction failed" << std::endl;
    }
}

void parallax_scan(parallax_kernel_t scan_kernel, parallax_kernel_t add_kernel,
                   void* data, size_t count, size_t elem_size) {
    if (!scan_kernel || !add_kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_scan] invalid kernels or launcher" << std::endl;
        return;
    }
    auto* sh = reinterpret_cast<KernelHandle*>(scan_kernel);
    auto* ah = reinterpret_cast<KernelHandle*>(add_kernel);
    std::cout << "[parallax_scan] scan=" << sh->name << " add=" << ah->name
              << " count=" << count << " elem_size=" << elem_size << std::endl;
    if (!g_kernel_launcher->launch_scan(sh->name, ah->name, data, count, elem_size)) {
        std::cerr << "[parallax_scan] scan failed" << std::endl;
    }
}

void parallax_exclusive_scan(parallax_kernel_t scan_kernel, parallax_kernel_t add_kernel,
                             parallax_kernel_t shift_kernel, void* input, void* output,
                             size_t count, size_t elem_size, const void* init) {
    if (!scan_kernel || !add_kernel || !shift_kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_exclusive_scan] invalid kernels or launcher" << std::endl;
        return;
    }
    auto* sh = reinterpret_cast<KernelHandle*>(scan_kernel);
    auto* ah = reinterpret_cast<KernelHandle*>(add_kernel);
    auto* hh = reinterpret_cast<KernelHandle*>(shift_kernel);
    std::cout << "[parallax_exclusive_scan] scan=" << sh->name << " shift=" << hh->name
              << " count=" << count << " elem_size=" << elem_size << std::endl;
    if (!g_kernel_launcher->launch_exclusive_scan(sh->name, ah->name, hh->name,
                                                  input, output, count, elem_size, init)) {
        std::cerr << "[parallax_exclusive_scan] scan failed" << std::endl;
    }
}

void parallax_sort(parallax_kernel_t kernel, void* data, size_t count, size_t elem_size) {
    if (!kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_sort] invalid kernel or launcher" << std::endl;
        return;
    }
    auto* h = reinterpret_cast<KernelHandle*>(kernel);
    std::cout << "[parallax_sort] kernel=" << h->name << " count=" << count
              << " elem_size=" << elem_size << std::endl;
    if (!g_kernel_launcher->launch_sort(h->name, data, count, elem_size)) {
        std::cerr << "[parallax_sort] sort failed" << std::endl;
    }
}

size_t parallax_copy_if(parallax_kernel_t flags_kernel, parallax_kernel_t scan_kernel,
                        parallax_kernel_t add_kernel, parallax_kernel_t scatter_kernel,
                        void* input, void* output, size_t count, size_t elem_size,
                        int elem_is_float) {
    if (!flags_kernel || !scan_kernel || !add_kernel || !scatter_kernel || !g_kernel_launcher) {
        std::cerr << "[parallax_copy_if] invalid kernels or launcher" << std::endl;
        return 0;
    }
    auto* fh = reinterpret_cast<KernelHandle*>(flags_kernel);
    auto* sh = reinterpret_cast<KernelHandle*>(scan_kernel);
    auto* ah = reinterpret_cast<KernelHandle*>(add_kernel);
    auto* xh = reinterpret_cast<KernelHandle*>(scatter_kernel);
    std::cout << "[parallax_copy_if] count=" << count << " elem_size=" << elem_size << std::endl;
    size_t kept = 0;
    if (!g_kernel_launcher->launch_compact(fh->name, sh->name, ah->name, xh->name,
                                           input, output, count, elem_size,
                                           elem_is_float != 0, &kept)) {
        std::cerr << "[parallax_copy_if] compaction failed" << std::endl;
        return 0;
    }
    return kept;
}

bool parallax_register_buffer(void* ptr, size_t size) {
    auto* memory_manager = parallax::get_global_memory_manager();
    if (!memory_manager) {
        std::cerr << "[parallax_register_buffer] Memory manager not initialized" << std::endl;
        return false;
    }
    
    return memory_manager->register_external_buffer(ptr, size);
}
