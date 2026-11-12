#pragma once

// nvcr.fx.v1 elementwise and data-movement kernels (CUDA) for the TensorRT
// provider, and the failure word. Integer arithmetic only; results equal
// src/deterministic/fx.cpp. A kernel that meets one of the profile's failure
// conditions (docs/spec/numerics-fx-v1.md section 5) sets the word; the
// session reads and clears it after each stage.

#include <cuda_runtime.h>

#include <cstdint>

namespace nvcr::tensorrt::fx {

// Each session owns a failure word and installs it on its thread around the
// enqueues of its stages (TensorRT calls plugin enqueue on the thread that
// calls enqueueV3), so sessions never see each other's failures. Returns the
// word installed before.
std::uint32_t* install_failure_word(std::uint32_t* word) noexcept;

// The installed word, or with none installed (plan building) a per-device
// scratch word that nobody reads.
cudaError_t failure_word(std::uint32_t** word);

// Sets the word when amax[index] (a kernel's reported maximum) exceeds limit.
cudaError_t check_max(const std::int32_t* amax, int index, std::int32_t limit, std::uint32_t* word,
                      cudaStream_t stream);

struct Shape final {
    int channels{0};
    int height{0};
    int width{0};
};

// rsr(a * b, shift); b has a's channels or one, and a's size or 1x1.
cudaError_t mul(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, Shape a_shape, Shape b_shape,
                int shift, std::uint32_t* word, cudaStream_t stream);
cudaError_t affine(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int64_t alpha,
                   std::int64_t beta, int shift, std::uint32_t* word, cudaStream_t stream);
cudaError_t add(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::int64_t count,
                std::uint32_t* word, cudaStream_t stream);
cudaError_t sat_table(const std::int32_t* x, std::int32_t* out, std::int64_t count, const std::int32_t* table,
                      std::int64_t n, std::int32_t top, cudaStream_t stream);
cudaError_t delta(const std::int32_t* x, std::int32_t* out, std::int64_t count, const std::int16_t* dc,
                  std::int64_t h, int shift, cudaStream_t stream);
cudaError_t clamp(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int32_t lo, std::int32_t hi,
                  cudaStream_t stream);
// out = rows[q] for q = qp[0]; a q outside [0, row_count) sets the word.
cudaError_t param_row(const std::int32_t* qp, const std::int64_t* rows, std::int32_t* out, int channels,
                      int row_count, std::uint32_t* word, cudaStream_t stream);
cudaError_t crop(const std::int32_t* x, std::int32_t* out, Shape in, int out_height, int out_width,
                 cudaStream_t stream);
// PyTorch pixel_shuffle (up) and pixel_unshuffle; `in` describes the input.
cudaError_t shuffle(const std::int32_t* x, std::int32_t* out, Shape in, int r, bool up, cudaStream_t stream);
// Taps (c, i, j) of a k x k window, stride s; 3x3 pads 1, 2x2 pads 0; zero outside.
cudaError_t im2col(const std::int32_t* x, std::int32_t* out, Shape in, int k, int stride, cudaStream_t stream);

}  // namespace nvcr::tensorrt::fx
