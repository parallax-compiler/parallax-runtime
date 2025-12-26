#include <new>
#include <type_traits>
#ifndef PARALLAX_ALLOCATOR_HPP
#define PARALLAX_ALLOCATOR_HPP

#include "parallax/runtime.h"
#include <cstddef>
#include <limits>

namespace parallax {

/**
 * STL-compatible allocator that uses Parallax unified memory
 *
 * Usage:
 *   std::vector<float, parallax::allocator<float>> data(1000);
 *   std::for_each(std::execution::par, data.begin(), data.end(),
 *                 [](float& x) { x *= 2.0f; });
 */
template<typename T>
class allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    allocator() noexcept = default;

    template<typename U>
    allocator(const allocator<U>&) noexcept {}

    T* allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        void* ptr = parallax_umalloc(n * sizeof(T), 0);
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_type) noexcept {
        parallax_ufree(ptr);
    }

    template<typename U>
    struct rebind {
        using other = allocator<U>;
    };
};

template<typename T, typename U>
bool operator==(const allocator<T>&, const allocator<U>&) noexcept {
    return true;
}

template<typename T, typename U>
bool operator!=(const allocator<T>&, const allocator<U>&) noexcept {
    return false;
}

} // namespace parallax

#endif // PARALLAX_ALLOCATOR_HPP
