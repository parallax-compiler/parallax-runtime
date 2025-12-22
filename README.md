# Parallax Runtime

**Production-ready GPU acceleration for C++ Standard Parallel Algorithms**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.2%2B-red.svg)](https://www.vulkan.org/)
[![Performance](https://img.shields.io/badge/performance-744M%20elem%2Fs-success)]()

> **Status**: ✅ Production Ready (v0.9.5)
> **Performance**: 744M elements/sec (std::for_each), 732M elements/sec (std::transform)

## What is Parallax Runtime?

Parallax Runtime is the Vulkan-based execution engine that powers GPU-accelerated C++ parallel algorithms. It provides:

- **Unified Memory Management** - Automatic host-device synchronization
- **Kernel Execution** - SPIR-V compute shader dispatch
- **Custom STL Allocator** - Seamless integration with standard containers
- **Cross-Platform GPU Support** - NVIDIA, AMD, Intel via Vulkan 1.2+

## Features

- ✅ **Automatic Sync** - Unified memory with zero manual synchronization
- ✅ **STL Integration** - Works with `std::vector`, `std::for_each`, `std::transform`
- ✅ **Production Ready** - 100% test pass rate (47/47 conformance tests)
- ✅ **Universal GPU** - Any Vulkan 1.2+ device (NVIDIA, AMD, Intel)
- ✅ **Zero Overhead** - Direct Vulkan compute, no translation layers

## Installation

### Prerequisites

- **CMake** 3.20+
- **C++20 compiler** (GCC 11+, Clang 14+, MSVC 2022+)
- **Vulkan SDK** 1.2+ ([Download](https://vulkan.lunarg.com/))

### Build from Source

```bash
git clone https://github.com/parallax-compiler/parallax-runtime.git
cd parallax-runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### macOS (MoltenVK)

```bash
brew install molten-vk vulkan-loader
export VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json
```

## Quick Start

### Using with Parallax Compiler

Parallax Runtime is designed to work with the Parallax Clang plugin. See [parallax-compiler](https://github.com/parallax-compiler/parallax-compiler) for complete setup.

### 1. Basic Example (with Compiler Plugin)

```cpp
#include <vector>
#include <algorithm>
#include <execution>
#include <parallax/allocator.hpp>

int main() {
    // Use parallax::allocator for GPU-accessible memory
    std::vector<float, parallax::allocator<float>> data(1000000, 1.0f);

    // Runs on GPU automatically - NO explicit sync needed!
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) { x = x * 2.0f + 1.0f; });

    // Data automatically synchronized back to host
    std::cout << "Result: " << data[0] << std::endl; // 3.0
    return 0;
}
```

**Compile:**
```bash
clang++ -std=c++20 -fplugin=$PARALLAX_PLUGIN \
  -I$PARALLAX_ROOT/parallax-runtime/include \
  -L$PARALLAX_ROOT/parallax-runtime/build \
  -lparallax-runtime hello.cpp -o hello
```

### 2. Transform Example

```cpp
#include <vector>
#include <algorithm>
#include <execution>
#include <parallax/allocator.hpp>

int main() {
    std::vector<float, parallax::allocator<float>> input(1000000, 2.0f);
    std::vector<float, parallax::allocator<float>> output(1000000);

    // Transform on GPU - automatic memory management
    std::transform(std::execution::par,
                   input.begin(), input.end(), output.begin(),
                   [](float x) { return x * 3.0f + 1.0f; });

    std::cout << "Result: " << output[0] << std::endl; // 7.0
    return 0;
}
```

### Key Points

- ✅ Use `parallax::allocator<T>` for GPU-accessible containers
- ✅ NO manual sync - unified memory handles it automatically
- ✅ Use `std::execution::par` to trigger GPU execution
- ✅ Works with standard algorithms: `for_each`, `transform`

## Performance

**Production benchmarks on NVIDIA GTX 980M (December 2025):**

### std::for_each

| Dataset Size | Time (ms) | Throughput (M elem/s) |
|--------------|-----------|----------------------|
| 1K           | 5.57      | 0.18                 |
| 10K          | 0.27      | 36.77                |
| 100K         | 0.44      | 228.06               |
| **1M**       | **1.34**  | **744.36**           |

### std::transform

| Dataset Size | Time (ms) | Throughput (M elem/s) |
|--------------|-----------|----------------------|
| 1K           | 2.61      | 0.38                 |
| 10K          | 0.27      | 36.63                |
| 100K         | 0.41      | 243.24               |
| **1M**       | **1.37**  | **732.06**           |

**Key Insights:**
- Linear scaling above 10K elements
- Break-even point: ~5K-10K elements
- Peak performance: 744M elements/sec

See [parallax-benchmarks](https://github.com/parallax-compiler/parallax-benchmarks) for detailed results.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  C++ Application (std::execution::par)          │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Parallax Runtime Library                       │
│  ┌───────────────────────────────────────────┐  │
│  │  Custom Allocator (parallax::allocator)   │  │
│  │  - Unified memory allocation              │  │
│  │  - Automatic host-device sync             │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Kernel Cache & Launcher                  │  │
│  │  - SPIR-V kernel loading                  │  │
│  │  - Compute pipeline management            │  │
│  │  - GPU dispatch & sync                    │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Vulkan Backend                           │  │
│  │  - Device initialization                  │  │
│  │  - Command buffer management              │  │
│  │  - Memory management                      │  │
│  └───────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Vulkan Driver (NVIDIA, AMD, Intel, etc.)       │
└─────────────────────────────────────────────────┘
```

## API Reference

### C++ API (Recommended)

```cpp
#include <parallax/allocator.hpp>

// Custom allocator for std::containers
template<typename T>
class parallax::allocator {
    T* allocate(size_t n);
    void deallocate(T* p, size_t n);
};

// Usage
std::vector<float, parallax::allocator<float>> data(1000);
```

### C API (Low-Level)

```cpp
#include <parallax/runtime.h>

// Kernel management
parallax_kernel_t parallax_load_kernel(const void* spirv, size_t size);
void parallax_kernel_launch(parallax_kernel_t kernel, void* buffer, size_t count);
void parallax_kernel_launch_transform(parallax_kernel_t kernel, void* in, void* out, size_t count);

// Memory management (automatic in C++ API)
void* parallax_umalloc(size_t size, int flags);
void parallax_ufree(void* ptr);
```

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| std::for_each | ✅ Production | All lambda patterns |
| std::transform | ✅ Production | Return value support |
| Unified memory | ✅ Production | Automatic sync |
| Custom allocator | ✅ Production | STL integration |
| Float operations | ✅ Production | +, -, *, / |
| NVIDIA GPUs | ✅ Tested | GTX 980M validated |
| AMD GPUs | ✅ Compatible | Vulkan 1.2+ |
| Intel GPUs | ✅ Compatible | Vulkan 1.2+ |
| std::reduce | ⏳ Planned | v1.0 |
| Multi-GPU | ⏳ Planned | v1.1 |

## Examples

See [parallax-samples](https://github.com/parallax-compiler/parallax-samples) for complete examples:
- `01_hello_parallax.cpp` - Hello World
- `02_for_each_simple.cpp` - For-each patterns
- `03_transform_simple.cpp` - Transform patterns

## Testing

```bash
cd build
ctest --output-on-failure
```

**Test Results:**
- 47/47 conformance tests passing (100%)
- Algorithm correctness: 22/22 ✅
- Memory management: 15/15 ✅
- Performance validation: 10/10 ✅

See [parallax-cts](https://github.com/parallax-compiler/parallax-cts) for test suite.

## Troubleshooting

### "No GPU found"

```bash
# Check Vulkan installation
vulkaninfo

# Verify device
vulkaninfo | grep deviceName
```

### "Wrong results"

Make sure you're using `parallax::allocator`:
```cpp
// ✅ Correct
std::vector<float, parallax::allocator<float>> data;

// ❌ Wrong - won't work with GPU
std::vector<float> data;
```

### "Slow performance"

- Use datasets >10K elements for best performance
- GPU has overhead; break-even is around 5K-10K elements
- Check GPU is being used: `PARALLAX_VERBOSE=1 ./myapp`

## Development

### Building with Debug Symbols

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DPARALLAX_ENABLE_VALIDATION=ON
make -j$(nproc)
```

### Running Benchmarks

```bash
cd ../parallax-benchmarks
mkdir build && cd build
cmake ..
make
./micro/bench_performance
```

## Documentation

- [Parallax Compiler](https://github.com/parallax-compiler/parallax-compiler) - Clang plugin
- [Benchmarks](https://github.com/parallax-compiler/parallax-benchmarks) - Performance tests
- [Samples](https://github.com/parallax-compiler/parallax-samples) - Example programs
- [Tests](https://github.com/parallax-compiler/parallax-cts) - Conformance test suite
- [GitHub Pages](https://parallax-compiler.github.io/parallax-docs) - Full documentation

## Roadmap

### v1.0.0 (Next)
- [ ] std::reduce implementation
- [ ] Integer and double support
- [ ] Windows platform support
- [ ] Advanced profiling tools

### v1.1.0 (Future)
- [ ] Multi-GPU support
- [ ] Kernel fusion optimization
- [ ] Template lambda support (auto parameters)
- [ ] std::sort implementation

### v1.2.0 (Future)
- [ ] std::scan implementation
- [ ] Custom execution policies
- [ ] Async execution support
- [ ] Stream support

## Contributing

We welcome contributions! Areas where you can help:

- **Performance Optimization** - Kernel tuning, memory patterns
- **Platform Testing** - Test on different GPUs and drivers
- **Algorithm Implementation** - More parallel algorithms
- **Documentation** - Improve guides and examples

See [CONTRIBUTING.md](https://github.com/parallax-compiler/.github/blob/main/CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 © 2025 Parallax Contributors

## Acknowledgments

- **LLVM/Clang** - Compiler infrastructure
- **Vulkan** - Cross-platform GPU compute
- **pSTL-Bench** - Benchmarking framework
- **C++ Community** - Standards and inspiration

## Contact

- **Issues**: [GitHub Issues](https://github.com/parallax-compiler/parallax-runtime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parallax-compiler/parallax-runtime/discussions)
- **Documentation**: [parallax-compiler.github.io/parallax-docs](https://parallax-compiler.github.io/parallax-docs)

---

**Production-ready runtime for GPU-accelerated C++ parallel algorithms** ✅
