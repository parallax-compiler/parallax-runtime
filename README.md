# Parallax Runtime

**Vulkan execution engine for GPU-offloaded C++ Standard Parallel Algorithms**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.2%2B-red.svg)](https://www.vulkan.org/)

> **Status**: actively developed. Correctness is verified end-to-end in CI on **lavapipe**
> (the software Vulkan rasterizer) via the runtime ctest suite plus the compiler's
> integration probe. Real-hardware performance tuning is on the roadmap, not yet the focus.

## What is Parallax Runtime?

Parallax Runtime is the Vulkan-based execution engine behind the
[Parallax Compiler](https://github.com/parallax-compiler/parallax-compiler). The compiler
turns `std::execution::par` algorithms into SPIR-V kernels; the runtime loads and dispatches
them, and provides the **software unified-memory** model that makes plain `std::vector` data
addressable on the GPU. It provides:

- **Software unified memory (`UnifiedArena`)** — a single host-mapped, GPU-addressable arena.
  On integrated/UMA devices it maps directly (zero-copy); on discrete GPUs it keeps a
  device-local buffer with host staging and migrates on demand. Host pointer values stored in
  the data are relocated in-shader (`gpu = dev_base + (host − host_base)`).
- **Funnel kernel cache** — kernels the compiler embeds register themselves at static-init
  under a string key; the runtime lazy-loads them on first use and falls back to the CPU on a
  miss (so semantics are always correct).
- **Primitive library** — workgroup tree-reduction, Hillis-Steele prefix scan, bitonic sort,
  and scan-scatter stream compaction, all type-parametric over 32/64-bit float and integer.
- **Cross-vendor GPU support** — any Vulkan 1.2+ device (NVIDIA, AMD, Intel, lavapipe).

## Features

- ✅ **Software UM** — one arena; zero-copy on UMA, device-local + staging/migration on discrete
- ✅ **Funnel dispatch** — string-keyed kernel registry with a correct CPU fallback on a miss
- ✅ **Primitive skeletons** — reduce (+ custom binary op), inclusive/exclusive scan, sort,
  copy_if/partition/unique compaction
- ✅ **Captures & multi-buffer** — scalar/struct capture uniforms, 2-buffer transforms
- ✅ **Pointer relocation** — dereference host pointers stored in unified memory (BDA / physical
  storage buffer)
- ✅ **Cross-vendor** — any Vulkan 1.2+ device; verified on lavapipe in CI

## Installation

### Prerequisites

- **CMake** 3.20+
- **C++20 compiler** (GCC 11+, Clang 14+)
- **Vulkan SDK** 1.2+ ([Download](https://vulkan.lunarg.com/)) and a Vulkan 1.2+ device
  (or lavapipe / `LIBGL_ALWAYS_SOFTWARE=1` for a software device)

### Build from Source

```bash
git clone https://github.com/parallax-compiler/parallax-runtime.git
cd parallax-runtime
cmake -S . -B out -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out -j
```

### macOS note

The runtime builds on macOS, but kernel **execution** is verified only where a Vulkan
compute device is available (Linux + lavapipe in CI, or real hardware). On macOS the toolchain
is typically used for compiler/codegen work; run the offload tests in CI or on a Vulkan host.

## Quick Start

The runtime is normally driven by the compiler, not called directly. With the Parallax Clang
plugin, **unmodified** `std::execution::par` code offloads — no `parallax::` names, no manual
allocator, no explicit sync:

```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>

int main() {
    std::vector<float> data(1'000'000, 1.0f);

    // Map: offloads to the GPU (identical CPU semantics without a device).
    std::for_each(std::execution::par, data.begin(), data.end(),
                  [](float& x) { x = x * 2.0f + 1.0f; });

    // Fold: offloads a workgroup tree-reduction.
    float sum = std::reduce(std::execution::par, data.begin(), data.end(), 0.0f);

    std::printf("data[0]=%.1f sum=%.1f\n", data[0], sum);   // 3.0, 3000000.0
    return 0;
}
```

Data is staged through the arena and synchronized automatically. See the
[compiler README](https://github.com/parallax-compiler/parallax-compiler) for the build flags
(`PARALLAX_TRANSPARENT=1` two-pass build, or the `parallax-cxx` wrapper).

## Architecture

```
┌─────────────────────────────────────────────────┐
│  C++ app: std::algo(std::execution::par, …)      │
│  → parallax::  funnel  → parallax::detail::device_*│  (compiler-routed)
└──────────────┬───────────────────────────────────┘
               │  embedded SPIR-V self-registers by key
┌──────────────▼───────────────────────────────────┐
│  Parallax Runtime                                 │
│  ┌─────────────────────────────────────────────┐ │
│  │ Kernel cache (register / lookup by key)     │ │
│  │  - lazy pipeline load; CPU fallback on miss │ │
│  ├─────────────────────────────────────────────┤ │
│  │ KernelLauncher                              │ │
│  │  - launch / with_captures / transform2      │ │
│  │  - reduce · scan · exclusive_scan · sort    │ │
│  │  - copy_if / partition / unique (compaction)│ │
│  ├─────────────────────────────────────────────┤ │
│  │ UnifiedArena (software UM)                   │ │
│  │  - UMA: host-mapped, zero-copy               │ │
│  │  - discrete: device-local + staging/migrate  │ │
│  │  - in-shader pointer relocation (BDA)        │ │
│  ├─────────────────────────────────────────────┤ │
│  │ Vulkan backend (device, queues, cmd buffers) │ │
│  └─────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  Vulkan 1.2+ driver (NVIDIA / AMD / Intel / lavapipe)│
└───────────────────────────────────────────────────┘
```

## API Reference

Applications use the compiler and never touch these directly, but the C ABI
(`include/parallax/runtime.h`) is the contract the embedded kernels use:

```c
#include <parallax/runtime.h>

/* Software-UM arena */
void*  parallax_arena_alloc(size_t size, size_t align);
void   parallax_arena_free(void* ptr);
int    parallax_arena_contains(const void* ptr);

/* Funnel kernel registry (embedded SPIR-V registers itself; the funnel looks it up) */
void             parallax_kernel_register(const char* key, const unsigned int* spirv, size_t words);
parallax_kernel_t parallax_kernel_lookup(const char* key);

/* Element-wise launches */
void parallax_kernel_launch(parallax_kernel_t k, void* buf, size_t count, size_t elem_size);
void parallax_kernel_launch_with_captures(parallax_kernel_t k, void* buf, size_t count,
                                          void* captures, size_t capture_size, size_t elem_size);
void parallax_kernel_launch_transform2(parallax_kernel_t k, void* in, void* out, size_t count,
                                       size_t in_elem, size_t out_elem);
void parallax_kernel_launch_transform2_captures(parallax_kernel_t k, void* in, void* out, size_t count,
                                                size_t in_elem, size_t out_elem,
                                                void* captures, size_t capture_size);

/* Primitives */
void   parallax_reduce(parallax_kernel_t k, void* data, size_t count, size_t elem_size, void* result);
void   parallax_scan(parallax_kernel_t scan_k, parallax_kernel_t add_k,
                     void* data, size_t count, size_t elem_size);
void   parallax_exclusive_scan(parallax_kernel_t scan_k, parallax_kernel_t add_k,
                               parallax_kernel_t shift_k, void* in, void* out,
                               size_t count, size_t elem_size, const void* init);
void   parallax_sort(parallax_kernel_t k, void* data, size_t count, size_t elem_size);
size_t parallax_copy_if(parallax_kernel_t flags_k, parallax_kernel_t scan_k,
                        parallax_kernel_t add_k, parallax_kernel_t scatter_k,
                        void* in, void* out, size_t count, size_t elem_size, int elem_is_float);
```

## Supported operations

| Capability | Status | Notes |
|---|---|---|
| Element-wise map (`for_each`/`fill`/`generate`) | ✅ | in-place; capture uniforms |
| Transform (`in→out`) | ✅ | capturing ops; differing in/out element sizes |
| Reduce / transform_reduce | ✅ | default `+`, or a custom binary op |
| Inclusive / exclusive scan | ✅ | multi-block, default `+` |
| Sort | ✅ | bitonic, ascending (default `<`) |
| Compaction (`copy_if`/`partition`/`unique`) | ✅ | flags → scan → scatter |
| Element types | ✅ | 32/64-bit float and integer |
| Software UM (UMA zero-copy) | ✅ | host-mapped arena |
| Discrete GPU (device-local + staging) | ✅ mechanism | migration works; real-HW perf not yet tuned |
| Pointer relocation (host ptr in data) | ✅ | BDA / physical storage buffer |
| Custom sort comparator / scan op | ⏳ | default `<` / `+` only for now |

## Testing

```bash
cmake --build out -j
cd out && ctest --output-on-failure
```

CI runs the suite on lavapipe:

- `VulkanBackend`, `UnifiedArena`, `Interpose`, `PhysPtrRelocation`
- `ParallelReduce`, `ParallelScan`, `ParallelSort`, `ParallelCopyIf`
- `StagingMigration` (discrete-path software-UM migration, forced on UMA)

The compiler repo's integration probe additionally exercises the full offload pipeline
(plugin → SPIR-V → dispatch → correctness-vs-CPU) end to end on lavapipe.

## Troubleshooting

**"No GPU found"** — check `vulkaninfo`; for a software device use lavapipe with
`LIBGL_ALWAYS_SOFTWARE=1` (and `VK_ICD_FILENAMES` pointing at the lavapipe ICD).

**Wrong results / no offload** — set `PARALLAX_DEBUG=1` to see kernel loads
(`Successfully loaded kernel`) vs a `MISS` (which means the algorithm ran on the CPU
fallback). `PARALLAX_FORCE_STAGING=1` exercises the discrete-GPU migration path on a UMA
device.

## Roadmap

- **Discrete-GPU performance** — dirty-range migration (copy only changed regions), real-HW
  perf validation (mechanism is proven on lavapipe; perf is not CI-observable)
- **Custom ops** — sort comparators and scan/segment operators
- **More skeletons** — search, merge, set operations, `min`/`max_element`,
  `transform_{in,ex}clusive_scan`
- **Optimizations** — subgroup reductions, specialization-constant workgroup sizing

## License

Apache 2.0 © Parallax Contributors

## Acknowledgments

- **LLVM/Clang** — compiler infrastructure
- **Vulkan** / **SPIR-V Tools** — cross-platform GPU compute + validation
- **pSTL-Bench** — parallel-STL benchmarking suite used for end-to-end validation
