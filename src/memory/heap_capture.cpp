// Parallax link-time heap capture (Phase 2) — the nvc++-style whole-heap model.
//
// Built as a STATIC library (parallax-heap). When the compiler wrapper links it into a
// program, these STRONG global definitions of operator new / delete and the malloc family
// override the C++/C runtime's, routing every allocation the program makes into the
// Parallax heap pool (heap_pool.cpp). All heap data then lives in one contiguous mmap
// region that the arena imports as a device buffer, so any heap pointer is GPU-addressable
// with no copy — exactly what nvc++ gets from CUDA managed memory, but only for code
// linked into the program (prebuilt shared libraries keep their own allocations).
//
// Linux/glibc only (matches nvc++'s posture and avoids dlsym reentrancy): foreign pointers
// — allocated before the pool existed, or by a prebuilt .so's internal allocator — are
// routed to the glibc internal allocator via __libc_*. On other platforms this TU is empty
// (the capture mechanism simply is not linked in).

#include "parallax/heap_pool.hpp"

#if defined(__linux__) && defined(__GLIBC__)

#include <cstddef>
#include <cstring>
#include <new>

// glibc internal allocator — used for pointers that did not come from the pool. Calling
// the public free()/realloc() here would recurse into our own strong definitions.
extern "C" {
void* __libc_malloc(size_t);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void  __libc_free(void*);
}

namespace {
inline void* pool_or_libc(size_t n, size_t align) {
    if (n == 0) n = 1;
    void* p = px_pool_alloc(n, align);
    return p ? p : __libc_malloc(n);   // pool exhausted -> system heap (still freeable: contains() is false)
}
inline void route_free(void* p) {
    if (!p) return;
    if (px_pool_contains(p)) px_pool_free(p);
    else                     __libc_free(p);
}
}  // namespace

// ---- operator new / delete (all standard variants) --------------------------------
void* operator new(std::size_t n) {
    void* p = pool_or_libc(n, alignof(std::max_align_t));
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    void* p = pool_or_libc(n, static_cast<std::size_t>(a));
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    return pool_or_libc(n, alignof(std::max_align_t));
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return pool_or_libc(n, alignof(std::max_align_t));
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    return pool_or_libc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    return pool_or_libc(n, static_cast<std::size_t>(a));
}

void operator delete(void* p) noexcept                                  { route_free(p); }
void operator delete[](void* p) noexcept                                { route_free(p); }
void operator delete(void* p, std::size_t) noexcept                     { route_free(p); }
void operator delete[](void* p, std::size_t) noexcept                   { route_free(p); }
void operator delete(void* p, std::align_val_t) noexcept                { route_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { route_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { route_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { route_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept           { route_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept         { route_free(p); }

// ---- C malloc family --------------------------------------------------------------
extern "C" {

void* malloc(size_t n) { return pool_or_libc(n, 16); }

void free(void* p) { route_free(p); }

void* calloc(size_t nmemb, size_t size) {
    size_t n = nmemb * size;
    if (nmemb != 0 && n / nmemb != size) return nullptr;   // overflow
    void* p = px_pool_alloc(n ? n : 1, 16);
    if (!p) return __libc_calloc(nmemb, size);
    std::memset(p, 0, n);
    return p;
}

void* realloc(void* p, size_t n) {
    if (!p) return pool_or_libc(n, 16);
    if (px_pool_contains(p)) {
        if (n == 0) { px_pool_free(p); return nullptr; }
        return px_pool_realloc(p, n);
    }
    return __libc_realloc(p, n);   // foreign block stays on the system heap
}

void* aligned_alloc(size_t align, size_t n) { return px_pool_alloc(n ? n : 1, align); }
void* memalign(size_t align, size_t n)      { return px_pool_alloc(n ? n : 1, align); }
void* valloc(size_t n)                       { return px_pool_alloc(n ? n : 1, 4096); }
void* pvalloc(size_t n)                      { return px_pool_alloc(n ? n : 1, 4096); }

int posix_memalign(void** out, size_t align, size_t n) {
    if (align < sizeof(void*) || (align & (align - 1))) return 22;  // EINVAL
    void* p = px_pool_alloc(n ? n : 1, align);
    if (!p) return 12;  // ENOMEM
    *out = p;
    return 0;
}

size_t malloc_usable_size(void* p) {
    if (p && px_pool_contains(p)) return px_pool_usable_size(p);
    return 0;   // foreign / null: conservative
}

}  // extern "C"

#endif  // __linux__ && __GLIBC__ — non-glibc: capture not linked; parallax_heap_contains
        // is provided by the runtime (heap_pool.cpp).
