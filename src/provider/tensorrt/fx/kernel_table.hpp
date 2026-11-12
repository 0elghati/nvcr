#pragma once
// nvcr.fx.v1 kernels compiled ahead of time from Triton (kernels/aot_compile.py) and launched
// through the CUDA driver API. The table is generated; this header declares it and the launcher.

#include <cuda.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace nvcr::tensorrt::fx {

struct KernelEntry {
    const char* key;
    const unsigned char* cubin;
    std::size_t size;
    const char* entry;
    int shared_bytes;
    int warps;
};

extern const KernelEntry kernel_entries[];
extern const std::size_t kernel_entry_count;
extern const char kernel_target[];
extern const char kernel_triton_version[];

// A loaded kernel in the current CUDA context. Loading is cached per (context, key).
struct Kernel {
    CUfunction function{};
    int shared_bytes{};
    int warps{};
};

// Returns CUDA_ERROR_NOT_FOUND when the table has no such key.
CUresult load_kernel(std::string_view key, Kernel& out);

// Launches with Triton's argument convention: the kernel's own arguments, then its two scratch pointers (null).
CUresult launch(const Kernel& kernel, unsigned grid_x, unsigned grid_y, unsigned grid_z, CUstream stream,
                void** args, int nargs);

std::string kernel_error(CUresult status);

}  // namespace nvcr::tensorrt::fx
