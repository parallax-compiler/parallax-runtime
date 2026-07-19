// Heap pool + link-time capture (Phase 2/4C). Exercises the Vulkan-free allocator that
// backs the whole-heap memory model: alignment, contains/usable, realloc data preservation,
// a checksum-verified concurrent stress, and — on a platform where the capture shim is
// active — that operator new / malloc land in the pool (parallax_heap_contains).

#include "parallax/heap_pool.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" int parallax_heap_contains(const void*);

int main() {
    setenv("PARALLAX_HEAP_POOL", "256", 1);   // 256 MiB
    setenv("PARALLAX_HEAP_CHECK", "1", 1);
    assert(px_pool_init());

    // Alignment + contains + usable.
    for (size_t a : {size_t(16), size_t(64), size_t(4096)}) {
        void* p = px_pool_alloc(1000, a);
        assert(p && ((uintptr_t)p % a) == 0);
        assert(px_pool_contains(p) && !px_pool_contains((void*)0x1234));
        assert(px_pool_usable_size(p) >= 1000);
        std::memset(p, 0xAB, 1000);
        px_pool_free(p);
    }

    // realloc preserves data.
    char* q = (char*)px_pool_alloc(100, 16);
    for (int i = 0; i < 100; ++i) q[i] = (char)i;
    q = (char*)px_pool_realloc(q, 5000);
    for (int i = 0; i < 100; ++i) assert(q[i] == (char)i);
    px_pool_free(q);

    // Concurrent checksum stress.
    std::atomic<long> fails{0};
    auto work = [&](int id) {
        std::mt19937 rng(id * 7919 + 1);
        struct A { void* p; size_t n; uint8_t s; };
        std::vector<A> live;
        for (int i = 0; i < 40000; ++i) {
            if ((rng() % 3) && live.size() > 20) {
                size_t k = rng() % live.size(); A a = live[k];
                for (size_t j = 0; j < a.n; ++j) if (((uint8_t*)a.p)[j] != a.s) { ++fails; break; }
                px_pool_free(a.p); live[k] = live.back(); live.pop_back();
            } else {
                size_t n = 1 + (rng() % 4096); uint8_t s = (uint8_t)(id * 16 + (rng() & 0xF));
                void* p = px_pool_alloc(n, 1u << (rng() % 5));
                if (p) { std::memset(p, s, n); live.push_back({p, n, s}); }
            }
        }
        for (A& a : live) { for (size_t j = 0; j < a.n; ++j) if (((uint8_t*)a.p)[j] != a.s) ++fails; px_pool_free(a.p); }
    };
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) ts.emplace_back(work, i);
    for (auto& t : ts) t.join();
    assert(fails.load() == 0);
    assert(px_pool_check());

    // If the capture shim is linked (Linux/glibc), heap allocations land in the pool.
    void* n1 = ::operator new(4096);
    void* m1 = std::malloc(4096);
    std::printf("heap-capture: new=%d malloc=%d (0 on non-glibc where capture is inactive)\n",
                parallax_heap_contains(n1), parallax_heap_contains(m1));
    ::operator delete(n1);
    std::free(m1);

    std::printf("HEAP OK: alignment, realloc, 4-thread stress, invariants\n");
    return 0;
}
