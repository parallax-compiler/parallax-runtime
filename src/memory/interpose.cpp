// Parallax allocation interposition (Phase 1d) — OPT-IN.
//
// Link this library into a program (e.g. -lparallax-interpose) to route all C++
// heap allocations through the unified-memory arena, so container storage is
// GPU-addressable with no host->device copy. Allocations made while the arena is
// still initializing (Vulkan setup itself allocates), or that don't fit the
// arena, fall back to the system allocator. operator delete routes arena pointers
// back to the arena and everything else to free().
//
// Only operator new/delete are interposed (this is what std::vector and the rest
// of the C++ standard library use). Raw malloc/free interposition is intentionally
// left out — it is far more invasive and unnecessary for the C++ stdpar path.

#include <cstddef>
#include <cstdlib>
#include <new>

extern "C" {
void* parallax_arena_alloc(size_t size, size_t align);
void  parallax_arena_free(void* ptr);
int   parallax_arena_contains(const void* ptr);
}

namespace {

// Reentrancy guard: the arena's own bookkeeping (and Vulkan initialization) call
// operator new; those must use the system allocator instead of recursing.
thread_local int g_in_alloc = 0;

void* px_alloc(std::size_t n, std::size_t align) {
    if (n == 0) n = 1;
    if (g_in_alloc == 0) {
        ++g_in_alloc;
        void* p = parallax_arena_alloc(n, align);
        --g_in_alloc;
        if (p) return p;
    }
    return std::malloc(n);
}

void px_free(void* p) {
    if (!p) return;
    if (g_in_alloc == 0) {
        ++g_in_alloc;
        const bool in_arena = parallax_arena_contains(p) != 0;
        --g_in_alloc;
        if (in_arena) {
            ++g_in_alloc;
            parallax_arena_free(p);
            --g_in_alloc;
            return;
        }
    }
    std::free(p);
}

}  // namespace

void* operator new(std::size_t n) {
    void* p = px_alloc(n, alignof(std::max_align_t));
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    void* p = px_alloc(n, static_cast<std::size_t>(a));
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    return px_alloc(n, alignof(std::max_align_t));
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return px_alloc(n, alignof(std::max_align_t));
}

void operator delete(void* p) noexcept { px_free(p); }
void operator delete[](void* p) noexcept { px_free(p); }
void operator delete(void* p, std::size_t) noexcept { px_free(p); }
void operator delete[](void* p, std::size_t) noexcept { px_free(p); }
void operator delete(void* p, std::align_val_t) noexcept { px_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { px_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { px_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { px_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { px_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { px_free(p); }
