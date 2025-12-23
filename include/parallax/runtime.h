#ifndef PARALLAX_RUNTIME_H
#define PARALLAX_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parallax C API - See docs for full documentation */

typedef struct parallax_runtime* parallax_runtime_t;
typedef struct parallax_kernel* parallax_kernel_t;

/* Memory management */
void* parallax_umalloc(size_t size, unsigned flags);
void parallax_ufree(void* ptr);

/* Kernel execution */
parallax_kernel_t parallax_kernel_load(const unsigned int* spirv, size_t words);
void parallax_kernel_launch(parallax_kernel_t kernel, ...);
void parallax_kernel_launch_transform(parallax_kernel_t kernel, ...);

/* NEW V2: Kernel execution with captured parameters */
void parallax_kernel_launch_with_captures(
    parallax_kernel_t kernel,
    void* buffer,
    size_t count,
    void* captures,
    size_t capture_size
);

#ifdef __cplusplus
}
#endif

#endif /* PARALLAX_RUNTIME_H */
