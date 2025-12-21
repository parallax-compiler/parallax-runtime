#ifndef PARALLAX_RUNTIME_H
#define PARALLAX_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

/* Parallax C API - See docs for full documentation */

typedef struct parallax_runtime* parallax_runtime_t;
typedef struct parallax_kernel* parallax_kernel_t;

/* Memory management */
void* parallax_umalloc(size_t size, unsigned flags);
void parallax_ufree(void* ptr);
void parallax_sync(void* ptr, int direction);

/* Kernel execution */
parallax_kernel_t parallax_kernel_load(const unsigned int* spirv, size_t words);
void parallax_kernel_launch(parallax_kernel_t kernel, ...);

#ifdef __cplusplus
}
#endif

#endif /* PARALLAX_RUNTIME_H */
