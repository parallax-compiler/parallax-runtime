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
/* Transform with distinct input/output element sizes (e.g. float -> double). */
void parallax_kernel_launch_transform2(parallax_kernel_t kernel, void* in_buffer,
                                       void* out_buffer, size_t count,
                                       size_t in_elem_size, size_t out_elem_size);

/* Parallel reduction (Phase 3). Reduces `count` elements of `data` to a single
 * scalar using the loaded workgroup-reduction kernel, writing elem_size bytes to
 * `result`. The kernel uses the '+' identity (0); the caller combines any init.
 */
void parallax_reduce(parallax_kernel_t kernel, void* data, size_t count,
                     size_t elem_size, void* result);

/* Inclusive prefix scan (Phase 5). Scans `data` in place using two kernels: a
 * per-block scan and an add-block-offsets pass. */
void parallax_scan(parallax_kernel_t scan_kernel, parallax_kernel_t add_kernel,
                   void* data, size_t count, size_t elem_size);

/* Bitonic sort (Phase 5). Sorts `data` in place ascending; `count` must be a power
 * of two (the caller pads otherwise). The kernel is a global compare-exchange stage
 * dispatched over the bitonic (k, j) schedule. */
void parallax_sort(parallax_kernel_t kernel, void* data, size_t count, size_t elem_size);

/* NEW V2: Kernel execution with captured parameters */
void parallax_kernel_launch_with_captures(
    parallax_kernel_t kernel,
    void* buffer,
    size_t count,
    void* captures,
    size_t capture_size,
    size_t elem_size
);

#ifdef __cplusplus
}
#endif

#endif /* PARALLAX_RUNTIME_H */

/* Register external buffer (e.g., from std::vector) for GPU use */
bool parallax_register_buffer(void* ptr, size_t size);
