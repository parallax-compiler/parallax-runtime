// Parallax heap pool (Phase 2) — see include/parallax/heap_pool.hpp.
//
// A boundary-tag, segregated-fit allocator with coalescing over one anonymous mmap
// reservation. Depends on nothing but mmap + static storage, so the allocation shim can
// call it during static initialization (the pool itself never calls operator new, so
// there is no reentrancy to guard against).
//
// Block layout (all block bases 16-aligned, all sizes multiples of 16):
//   [ header: size_t (bsize | ALLOC_BIT) ][ payload … ][ footer: size_t (bsize) ]
// A free block additionally stores {next,prev} free-list links in the first 16 payload
// bytes (hence MIN_BSIZE = header + 16 + footer = 32). The footer (pure size, no flag)
// lets block_free coalesce with the previous physical block. The whole reservation is
// always tiled by blocks, so physical neighbours are found by pointer arithmetic.
//
// A returned pointer p is aligned to the request and has a back-pointer to its block
// header stored in the sizeof(void*) bytes immediately before it, so free()/usable_size()
// recover the block for ANY alignment uniformly.

#include "parallax/heap_pool.hpp"

#include <sys/mman.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr size_t ALIGN     = 16;
constexpr size_t HDR       = 8;
constexpr size_t FTR       = 8;
constexpr size_t MIN_BSIZE = 32;   // header + 2 links + footer
constexpr size_t ALLOC_BIT = 1;
constexpr int    NBINS     = 44;   // floor_log2 size classes up to 2^47

inline size_t round_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

// State — trivially/constinit-constructible only (no dynamic init), so operator new can
// touch it before any C++ constructor runs.
std::atomic_flag  g_lock = ATOMIC_FLAG_INIT;
std::atomic<char*> g_base{nullptr};
std::atomic<size_t> g_reservation{0};
size_t g_high = 0;
bool   g_init_done = false;
bool   g_init_failed = false;
char*  g_bins[NBINS] = {};
int    g_check = 0;   // PARALLAX_HEAP_CHECK cache (set at init)

struct Guard {
    Guard()  { while (g_lock.test_and_set(std::memory_order_acquire)) {} }
    ~Guard() { g_lock.clear(std::memory_order_release); }
};

// --- block accessors (B = block base) ---
inline size_t& hdr(char* B)        { return *reinterpret_cast<size_t*>(B); }
inline size_t  bsize(char* B)      { return hdr(B) & ~ALLOC_BIT; }
inline bool    allocated(char* B)  { return (hdr(B) & ALLOC_BIT) != 0; }
inline void    set_block(char* B, size_t bs, bool alloc) {
    hdr(B) = bs | (alloc ? ALLOC_BIT : 0);
    *reinterpret_cast<size_t*>(B + bs - FTR) = bs;   // footer: pure size
}
inline char*&  fl_next(char* B) { return *reinterpret_cast<char**>(B + HDR); }
inline char*&  fl_prev(char* B) { return *reinterpret_cast<char**>(B + HDR + sizeof(char*)); }

inline int bin_index(size_t bs) {
    int lg = 63 - __builtin_clzll((unsigned long long)bs);
    int idx = lg - 4;
    if (idx < 0) idx = 0;
    if (idx >= NBINS) idx = NBINS - 1;
    return idx;
}

void bin_insert(char* B) {
    int i = bin_index(bsize(B));
    char* head = g_bins[i];
    fl_prev(B) = nullptr;
    fl_next(B) = head;
    if (head) fl_prev(head) = B;
    g_bins[i] = B;
}

void bin_remove(char* B) {
    char* p = fl_prev(B);
    char* n = fl_next(B);
    if (p) fl_next(p) = n; else g_bins[bin_index(bsize(B))] = n;
    if (n) fl_prev(n) = p;
}

bool ensure_init_locked() {
    if (g_init_done) return !g_init_failed;
    g_init_done = true;
    if (const char* e = std::getenv("PARALLAX_HEAP_CHECK")) g_check = (e[0] && e[0] != '0');
    size_t res = 8ull << 30;   // 8 GiB default
    if (const char* e = std::getenv("PARALLAX_HEAP_POOL")) {
        long long v = atoll(e);   // MiB
        if (v > 0) res = (size_t)v << 20;
    }
    res = round_up(res, 4096);
    void* p = mmap(nullptr, res, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { g_init_failed = true; return false; }
    for (int i = 0; i < NBINS; ++i) g_bins[i] = nullptr;
    set_block((char*)p, res, /*alloc=*/false);
    bin_insert((char*)p);
    g_reservation.store(res, std::memory_order_relaxed);
    g_base.store((char*)p, std::memory_order_release);   // publish last
    return true;
}

// Allocate a block of at least `need` bytes (already >= MIN_BSIZE, multiple of ALIGN).
char* block_alloc(size_t need) {
    char* base = g_base.load(std::memory_order_relaxed);
    for (int i = bin_index(need); i < NBINS; ++i) {
        for (char* B = g_bins[i]; B; B = fl_next(B)) {
            size_t bs = bsize(B);
            if (bs >= need) {
                bin_remove(B);
                if (bs - need >= MIN_BSIZE) {          // split
                    set_block(B, need, true);
                    char* R = B + need;
                    set_block(R, bs - need, false);
                    bin_insert(R);
                } else {
                    set_block(B, bs, true);            // use whole block
                }
                size_t end = (size_t)(B - base) + bsize(B);
                if (end > g_high) g_high = end;
                return B;
            }
        }
    }
    return nullptr;   // reservation exhausted
}

void block_free(char* B) {
    char* base = g_base.load(std::memory_order_relaxed);
    size_t res = g_reservation.load(std::memory_order_relaxed);
    size_t bs = bsize(B);
    char* next = B + bs;                                // forward coalesce
    if (next < base + res && !allocated(next)) {
        bin_remove(next);
        bs += bsize(next);
    }
    if (B > base) {                                     // backward coalesce
        size_t prev_bs = *reinterpret_cast<size_t*>(B - FTR);   // prev footer = its size
        char* prev = B - prev_bs;
        if (prev >= base && !allocated(prev)) {
            bin_remove(prev);
            bs += bsize(prev);
            B = prev;
        }
    }
    set_block(B, bs, false);
    bin_insert(B);
}

int check_locked() {
    char* base = g_base.load(std::memory_order_relaxed);
    size_t res = g_reservation.load(std::memory_order_relaxed);
    if (!base) return 1;
    size_t off = 0;
    while (off < res) {
        char* B = base + off;
        size_t bs = bsize(B);
        if (bs < MIN_BSIZE || (bs & (ALIGN - 1)) != 0) return 0;
        if (off + bs > res) return 0;
        if (*reinterpret_cast<size_t*>(B + bs - FTR) != bs) return 0;   // footer mismatch
        off += bs;
    }
    return off == res ? 1 : 0;   // exact tiling
}

}  // namespace

extern "C" {

void* px_pool_alloc(size_t size, size_t align) {
    if (align < ALIGN) align = ALIGN;
    if (align & (align - 1)) { size_t a = ALIGN; while (a < align) a <<= 1; align = a; }
    if (size == 0) size = 1;
    Guard g;
    if (!ensure_init_locked()) return nullptr;
    // Payload must hold: sizeof(void*) back-pointer slack + alignment slack + size.
    size_t need_payload = size + align + sizeof(void*);
    size_t need_bsize = round_up(HDR + need_payload + FTR, ALIGN);
    if (need_bsize < MIN_BSIZE) need_bsize = MIN_BSIZE;
    char* B = block_alloc(need_bsize);
    if (!B) return nullptr;
    uintptr_t raw = reinterpret_cast<uintptr_t>(B + HDR) + sizeof(void*);
    uintptr_t p = (raw + (align - 1)) & ~(static_cast<uintptr_t>(align) - 1);
    *reinterpret_cast<char**>(p - sizeof(void*)) = B;   // back-pointer to header
    if (g_check) check_locked();
    return reinterpret_cast<void*>(p);
}

void px_pool_free(void* ptr) {
    if (!ptr) return;
    Guard g;
    char* B = *reinterpret_cast<char**>(reinterpret_cast<char*>(ptr) - sizeof(void*));
    block_free(B);
    if (g_check) check_locked();
}

size_t px_pool_usable_size(const void* ptr) {
    if (!ptr) return 0;
    Guard g;
    char* B = *reinterpret_cast<char* const*>(reinterpret_cast<const char*>(ptr) - sizeof(void*));
    char* end = B + bsize(B) - FTR;
    return static_cast<size_t>(end - reinterpret_cast<const char*>(ptr));
}

void* px_pool_realloc(void* ptr, size_t new_size) {
    if (!ptr) return px_pool_alloc(new_size, ALIGN);
    if (new_size == 0) { px_pool_free(ptr); return nullptr; }
    size_t old = px_pool_usable_size(ptr);
    if (new_size <= old) return ptr;                     // shrink/fit in place
    void* np = px_pool_alloc(new_size, ALIGN);
    if (!np) return nullptr;
    std::memcpy(np, ptr, old);
    px_pool_free(ptr);
    return np;
}

int px_pool_contains(const void* ptr) {
    char* base = g_base.load(std::memory_order_acquire);
    if (!base) return 0;
    size_t res = g_reservation.load(std::memory_order_relaxed);
    const char* p = reinterpret_cast<const char*>(ptr);
    return (p >= base && p < base + res) ? 1 : 0;
}

void*  px_pool_base(void)        { return g_base.load(std::memory_order_acquire); }
size_t px_pool_reservation(void) { return g_reservation.load(std::memory_order_relaxed); }

size_t px_pool_high_water(void) { Guard g; return g_high; }

int px_pool_init(void) { Guard g; return ensure_init_locked() ? 1 : 0; }

int px_pool_check(void) { Guard g; return check_locked(); }

// Public heap query (declared in runtime.h). Lives with the pool so it is available from
// the runtime shared lib regardless of whether the capture shim was linked.
int parallax_heap_contains(const void* ptr) { return px_pool_contains(ptr); }

}  // extern "C"
