#pragma once

// Experimental, codec-local bridge to the pinned upstream fused implementation.
// Built only by the C7 probe; Torch/CUDA do not enter NVCR's public contracts.
#include "nvcr/provider/experimental/session.hpp"

#include <ATen/ATen.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace nvcr::dcvcuf {

using Weights = std::map<std::string, at::Tensor>;

std::shared_ptr<provider::experimental::IProviderSession> make_recon_session(
    int device);

provider::experimental::ExecutableStageDescriptor recon_descriptor(
    int height, int width, std::string artifact_path,
    std::vector<std::byte> expected_digest);
provider::experimental::ExecutableStageDescriptor decoder_descriptor(
    int height, int width, std::string artifact_path,
    std::vector<std::byte> expected_digest);

// Probe-only transfers; tensor layout at the interface is contiguous NHWC FP16.
void upload(const provider::experimental::BufferHandle&, const at::Tensor&);
at::Tensor download(const provider::experimental::BufferHandle&,
                    const std::vector<std::int64_t>& shape);

}  // namespace nvcr::dcvcuf
