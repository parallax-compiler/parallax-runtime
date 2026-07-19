// Parallax heap pool (Phase 2) — a Vulkan-free, process-wide allocation pool.
//
// One large anonymous mmap reservation (default 8 GiB, MAP_NORESERVE so physical
// pages are only backed on touch) served by a real coalescing allocator. This is the
// substrate for the nvc++-style whole-heap memory model: the compiler wrapper links a
// strong-symbol allocation shim (heap_capture.cpp) that routes every operator new /
// malloc in the program here, so ALL heap data lands in one contiguous region. At the
// first kernel launch the arena imports that same region as a device buffer
// (VK_EXT_external_memory_host), making every heap pointer GPU-addressable with no copy.
//
// Deliberately depends on NOTHING (no Vulkan, no C++ runtime allocation): the pool
// allocates only via mmap and static storage, so the allocation shim can call it during
// static initialization — before Vulkan, before main — without reentrancy.

#ifndef PARALLAX_HEAP_POOL_HPP
#define PARALLAX_HEAP_POOL_HPP

#include <cstddef>

extern "C" {

// Allocate `size` bytes aligned to at least `align` (align is rounded up to a power of
// two >= 16). Returns nullptr only if the reservation is exhausted (the caller then
// falls back to the system allocator). Thread-safe.
void* px_pool_alloc(size_t size, size_t align);

// Free a pointer previously returned by px_pool_alloc/px_pool_realloc. Must NOT be
// called on a foreign pointer (callers gate on px_pool_contains first). Thread-safe.
void px_pool_free(void* ptr);

// Resize: preserves min(old,new) bytes. nullptr ptr => alloc; 0 size => free + nullptr.
void* px_pool_realloc(void* ptr, size_t new_size);

// Usable byte count for a pool pointer (>= the requested size).
size_t px_pool_usable_size(const void* ptr);

// 1 iff `ptr` lies inside the pool reservation (a pure range check — this is how the
// allocation shim decides pool-free vs system-free, and how the arena decides which
// pointers are already device-resident). 0 if the pool was never initialized.
int px_pool_contains(const void* ptr);

// Base address of the reservation (page-aligned), or nullptr if uninitialized. The
// arena imports [base, base+reservation) as one device buffer.
void* px_pool_base(void);

// Total reservation size in bytes (what the arena imports), or 0 if uninitialized.
size_t px_pool_reservation(void);

// Highest byte offset ever handed out (informational / diagnostics).
size_t px_pool_high_water(void);

// Force initialization now (idempotent). Returns 1 on success, 0 if mmap failed.
// Normally lazy at first px_pool_alloc; the arena may call this to learn base/size.
int px_pool_init(void);

// Walk the block list asserting header/footer/alignment/tiling invariants; returns 1 if
// consistent. Enabled per-op when PARALLAX_HEAP_CHECK=1. For tests/debugging.
int px_pool_check(void);

}  // extern "C"

#endif  // PARALLAX_HEAP_POOL_HPP
