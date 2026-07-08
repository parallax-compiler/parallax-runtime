#ifndef PARALLAX_STDPAR_HPP
#define PARALLAX_STDPAR_HPP

// Layer A: library-dispatch interception (the nvc++/dpc++ shape).
//
// Instead of the compiler plugin pattern-matching `std::for_each(par, ...)` call
// sites in the AST (fragile: shapes vary, and clang's traversal of generic-lambda
// instantiations is non-deterministic), interception here rides ordinary C++
// overload resolution. Every parallel algorithm funnels into ONE stable template,
// `parallax::detail::device_invoke<T, F>`. The plugin keys on that single named
// instantiation (always concrete, enumerable, deterministic) rather than on
// arbitrary call shapes.
//
// For each device_invoke<T,F> instantiation the plugin: (1) compiles F applied to
// an element into a SPIR-V kernel, and (2) appends a registrar keyed by this
// function's __PRETTY_FUNCTION__. At runtime the funnel looks that key up; a hit
// dispatches to the GPU, a miss runs the ISO sequential loop (correct either way).

#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <parallax/runtime.h>

namespace parallax {

// Parallax execution policies. `std::execution::par` interception is layered on
// top of this later; the explicit policy is the robust core the plugin keys on.
struct parallel_policy {};
struct parallel_unsequenced_policy {};
inline constexpr parallel_policy par{};
inline constexpr parallel_unsequenced_policy par_unseq{};

namespace detail {

// The single stable funnel. One instantiation per (element type T, functor F).
// The plugin compiles this instantiation's `f(data[i])` body to SPIR-V and
// registers it under __PRETTY_FUNCTION__ (which uniquely names this T,F pair and
// which the plugin computes identically). noinline keeps the instantiation a real,
// enumerable symbol; the body is the host fallback when no kernel is registered.
template <class T, class F>
__attribute__((noinline)) void device_invoke(T* data, std::size_t n, F f) {
    static parallax_kernel_t k = parallax_kernel_lookup(__PRETTY_FUNCTION__);
    if (k) {
        // Stage the data through the unified arena (host-mapped, GPU-addressable):
        // copy in, launch zero-copy on the arena buffer, copy back. This is the
        // software-UM path that is correct on both UMA and discrete GPUs; a later
        // optimization can bind arena-backed containers zero-copy to skip the copies.
        void* ab = parallax_arena_alloc(n * sizeof(T), alignof(T));
        if (ab) {
            std::memcpy(ab, data, n * sizeof(T));
            if constexpr (std::is_empty_v<F>) {
                parallax_kernel_launch(k, ab, n, sizeof(T));
            } else {
                parallax_kernel_launch_with_captures(k, ab, n,
                                                     static_cast<void*>(&f), sizeof(F),
                                                     sizeof(T));
            }
            std::memcpy(data, ab, n * sizeof(T));
            parallax_arena_free(ab);
            return;
        }
    }
    // Host fallback (ISO semantics) — also the path when codegen bailed.
    for (std::size_t i = 0; i < n; ++i) f(data[i]);
}

} // namespace detail

// ---- Algorithm surface (map skeleton) -------------------------------------

template <class Policy, class It, class F>
void for_each(Policy, It first, It last, F f) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) return;
    detail::device_invoke(&*first, n, f);
}

} // namespace parallax

#endif // PARALLAX_STDPAR_HPP
