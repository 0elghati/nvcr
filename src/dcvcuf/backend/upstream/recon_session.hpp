#pragma once

// Experimental, codec-local bridge to the pinned upstream fused implementation.
// Built only by the C7 probe; Torch/CUDA do not enter NVCR's public contracts.
#include "nvcr/provider/experimental/session.hpp"
#include <ATen/ATen.h>
#include <map>

namespace nvcr::dcvcuf {
using Weights = std::map<std::string, at::Tensor>;
std::shared_ptr<provider::experimental::IProviderSession> make_recon_session(
    const Weights& weights, int device);
provider::experimental::ExecutableStageDescriptor recon_descriptor(int height, int width);
// Probe-only transfers; tensor layout at the interface is contiguous NHWC FP16.
void upload(const provider::experimental::BufferHandle&, const at::Tensor&);
at::Tensor download(const provider::experimental::BufferHandle&,
                    const std::vector<std::int64_t>& shape);
}  // namespace nvcr::dcvcuf
