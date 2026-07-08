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
#include <string>
#include <iterator>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <limits>
#include <execution>
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

// Map in->out funnel: out[i] = f(in[i]). The plugin sees the functor returns non-void
// and generates a two-buffer transform kernel automatically. MVP: captureless functor
// (the pSTL-Bench transform kernel is a standalone lambda); a capturing transform op
// stays on the host until a transform-with-captures runtime path exists.
template <class Tin, class Tout, class F>
__attribute__((noinline)) void device_transform(const Tin* in, Tout* out, std::size_t n, F f) {
    static parallax_kernel_t k = parallax_kernel_lookup(__PRETTY_FUNCTION__);
    if (k && std::is_empty_v<F>) {
        void* ai = parallax_arena_alloc(n * sizeof(Tin), alignof(Tin));
        void* ao = parallax_arena_alloc(n * sizeof(Tout), alignof(Tout));
        if (ai && ao) {
            std::memcpy(ai, in, n * sizeof(Tin));
            parallax_kernel_launch_transform2(k, ai, ao, n, sizeof(Tin), sizeof(Tout));
            std::memcpy(out, ao, n * sizeof(Tout));
            parallax_arena_free(ao);
            parallax_arena_free(ai);
            return;
        }
    }
    for (std::size_t i = 0; i < n; ++i) out[i] = f(in[i]);
}

// Fold funnel (value-returning): reduce with the default '+' op. The plugin generates
// the fixed workgroup tree-reduction kernel for T (no per-element functor). The runtime
// returns the pure GPU sum; we combine the caller's init on the host (matches the old
// std::reduce path). MVP: default '+' only (custom-op reduce stays on the host).
template <class T>
__attribute__((noinline)) T device_reduce(const T* data, std::size_t n, T init) {
    static parallax_kernel_t k = parallax_kernel_lookup(__PRETTY_FUNCTION__);
    if (k) {
        void* ab = parallax_arena_alloc(n * sizeof(T), alignof(T));
        if (ab) {
            std::memcpy(ab, data, n * sizeof(T));
            T gpu{};
            parallax_reduce(k, ab, n, sizeof(T), &gpu);
            parallax_arena_free(ab);
            return init + gpu;
        }
    }
    T acc = init;
    for (std::size_t i = 0; i < n; ++i) acc = acc + data[i];
    return acc;
}

// Sort funnel (in-place, ascending, default '<'). The plugin generates the fixed
// bitonic compare-exchange kernel for T; the runtime dispatches it over the bitonic
// schedule (which requires a power-of-two count, so we pad through the arena with the
// type max — padding sorts to the end, leaving the first n as the sorted values).
template <class T>
__attribute__((noinline)) void device_sort(T* data, std::size_t n) {
    static parallax_kernel_t k = parallax_kernel_lookup(__PRETTY_FUNCTION__);
    if (k) {
        std::size_t m = 1;
        while (m < n) m <<= 1;                 // next power of two
        void* ab = parallax_arena_alloc(m * sizeof(T), alignof(T));
        if (ab) {
            T* pad = static_cast<T*>(ab);
            std::memcpy(pad, data, n * sizeof(T));
            for (std::size_t i = n; i < m; ++i) pad[i] = (std::numeric_limits<T>::max)();
            parallax_sort(k, pad, m, sizeof(T));
            std::memcpy(data, pad, n * sizeof(T));
            parallax_arena_free(ab);
            return;
        }
    }
    std::sort(data, data + n);
}

// Inclusive-scan funnel (default '+'). The plugin generates TWO fixed kernels for T
// (per-block Hillis-Steele scan + add-block-offsets), registered under this
// instantiation's __PRETTY_FUNCTION__ with ":scan"/":add" suffixes (both sides append
// the same suffix). Stage in->arena, scan in place, copy out.
template <class T>
__attribute__((noinline)) void device_scan(const T* in, T* out, std::size_t n) {
    static parallax_kernel_t ks =
        parallax_kernel_lookup((std::string(__PRETTY_FUNCTION__) + ":scan").c_str());
    static parallax_kernel_t ka =
        parallax_kernel_lookup((std::string(__PRETTY_FUNCTION__) + ":add").c_str());
    if (ks && ka) {
        void* ab = parallax_arena_alloc(n * sizeof(T), alignof(T));
        if (ab) {
            std::memcpy(ab, in, n * sizeof(T));
            parallax_scan(ks, ka, ab, n, sizeof(T));
            std::memcpy(out, ab, n * sizeof(T));
            parallax_arena_free(ab);
            return;
        }
    }
    T acc{};
    for (std::size_t i = 0; i < n; ++i) { acc = acc + in[i]; out[i] = acc; }
}

} // namespace detail

namespace detail {

// Only parallel policies offload; sequenced/unsequenced stay on the CPU. Recognizes
// both our policies and the std::execution ones (so transparently-routed
// std::for_each(std::execution::par,...) is gated correctly).
template <class P>
inline constexpr bool is_offload_policy_v =
    std::is_same_v<std::remove_cvref_t<P>, parallel_policy> ||
    std::is_same_v<std::remove_cvref_t<P>, parallel_unsequenced_policy>
#if defined(__cpp_lib_execution)
    || std::is_same_v<std::remove_cvref_t<P>, std::execution::parallel_policy>
    || std::is_same_v<std::remove_cvref_t<P>, std::execution::parallel_unsequenced_policy>
#endif
    ;

} // namespace detail

// ---- Algorithm surface (map skeleton) -------------------------------------

template <class Policy, class It, class F>
void for_each(Policy&&, It first, It last, F f) {
    if constexpr (detail::is_offload_policy_v<Policy>) {
        auto n = static_cast<std::size_t>(std::distance(first, last));
        if (n == 0) return;
        detail::device_invoke(&*first, n, f);
    } else {
        // seq / unseq -> plain serial std::for_each on the CPU (3-arg form is not
        // routed by the plugin, so no recursion).
        std::for_each(first, last, f);
    }
}

template <class Policy, class InIt, class OutIt, class F>
OutIt transform(Policy&&, InIt first, InIt last, OutIt d_first, F f) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if constexpr (detail::is_offload_policy_v<Policy>) {
        if (n) detail::device_transform(&*first, &*d_first, n, f);
    } else {
        std::transform(first, last, d_first, f);  // 4-arg form is not routed -> no recursion
        return std::next(d_first, static_cast<typename std::iterator_traits<OutIt>::difference_type>(n));
    }
    return std::next(d_first, static_cast<typename std::iterator_traits<OutIt>::difference_type>(n));
}

template <class Policy, class It, class T>
T reduce(Policy&&, It first, It last, T init) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if constexpr (detail::is_offload_policy_v<Policy>) {
        if (n == 0) return init;
        // Reduce in the ELEMENT type on the GPU (identity 0), then host-combine the
        // init. Keeps the kernel monomorphic in the element type even when the init type
        // T differs (e.g. float elements, double init) — approximate vs std::reduce's
        // per-step accumulation, exact when T == element type (the common case).
        using E = typename std::iterator_traits<It>::value_type;
        E gpu = detail::device_reduce<E>(&*first, n, E{});
        return init + static_cast<T>(gpu);
    } else {
        return std::reduce(first, last, init);  // 3-arg form is not routed -> no recursion
    }
}

// fill = a map that writes a captured constant. Reuses the device_invoke funnel with a
// value-capturing setter (captures now bind correctly), so no new kernel shape is needed.
template <class Policy, class It, class V>
void fill(Policy&&, It first, It last, const V& value) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if constexpr (detail::is_offload_policy_v<Policy>) {
        using E = typename std::iterator_traits<It>::value_type;
        if (n) { E v = static_cast<E>(value); detail::device_invoke(&*first, n, [v](E& x){ x = v; }); }
    } else {
        std::fill(first, last, value);  // 3-arg form is not routed -> no recursion
    }
}

template <class Policy, class It>
void sort(Policy&&, It first, It last) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if constexpr (detail::is_offload_policy_v<Policy>) {
        if (n) detail::device_sort(&*first, n);
    } else {
        std::sort(first, last);  // 2-arg form is not routed -> no recursion
    }
}

template <class Policy, class It, class OutIt>
OutIt inclusive_scan(Policy&&, It first, It last, OutIt d_first) {
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if constexpr (detail::is_offload_policy_v<Policy>) {
        if (n) detail::device_scan(&*first, &*d_first, n);
    } else {
        std::inclusive_scan(first, last, d_first);  // 3-arg form is not routed
    }
    return std::next(d_first, static_cast<typename std::iterator_traits<OutIt>::difference_type>(n));
}

} // namespace parallax

#endif // PARALLAX_STDPAR_HPP
