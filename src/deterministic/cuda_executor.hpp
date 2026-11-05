#pragma once

// CUDA executor for det-fp32-v1 programs. It produces exactly the bits of
// run_cpu on any NVIDIA GPU: every kernel performs the per-element operation
// sequence that program.hpp specifies, with _rn intrinsics only, no split-K
// reductions, and no floating-point tensor-core instructions.

#include "program.hpp"

#include <memory>
#include <span>
#include <string_view>
#include <vector>

struct CUstream_st;

namespace nvcr::deterministic {

// A batch-1 NCHW FP16 tensor in device memory, named as in the program.
struct HalfDeviceTensor final {
    std::string_view name;
    Extent extent;
    void* data{nullptr};
};

class CudaExecutor final {
public:
    CudaExecutor(CudaExecutor&&) noexcept;
    CudaExecutor& operator=(CudaExecutor&&) noexcept;
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;
    ~CudaExecutor();

    // Uploads the program's weights to the current device. The program must
    // outlive the executor. With a stream, all execution is ordered on it;
    // otherwise the executor owns a stream.
    [[nodiscard]] static Result<CudaExecutor> create(
        const Program& program, NumericsProfile profile = NumericsProfile::fp32_v1,
        CUstream_st* stream = nullptr);

    // Enqueues one execution on device-resident FP16 tensors. Inputs widen
    // exactly; outputs round to nearest even. Returns once the work is
    // enqueued; the output buffers are complete in stream order.
    [[nodiscard]] Result<void> run_half(
        std::span<const HalfDeviceTensor> inputs, std::span<const HalfDeviceTensor> outputs);

    // Host-to-host convenience path used by conformance tests.
    [[nodiscard]] Result<std::vector<Tensor>> run(std::span<const Tensor> inputs);

    // Device time in milliseconds for `iterations` back-to-back executions on
    // inputs already resident on the device (the first run warms the pool).
    [[nodiscard]] Result<double> time(std::span<const Tensor> inputs, int iterations);

private:
    struct Impl;
    explicit CudaExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr::deterministic
