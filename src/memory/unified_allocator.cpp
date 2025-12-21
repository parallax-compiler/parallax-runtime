#include "parallax/runtime.h"
#include <cstdlib>

void* parallax_umalloc(size_t size, unsigned flags) {
    // Stub implementation
    return std::malloc(size);
}

void parallax_ufree(void* ptr) {
    std::free(ptr);
}

void parallax_sync(void* ptr, int direction) {
    // Stub
}
