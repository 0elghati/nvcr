#pragma once

// Microsoft MLVC (Multi-platform Learned Video Codec) codec adapter.
//
// The adapter runs MLVC's published ONNX model bundles unchanged through any
// provider session that loads ONNX stages into host-addressable buffers (the
// ONNX Runtime providers). Configuration mapping:
//   artifacts.intra_engine_path  MLVC bundle directory (bundle_manifest.json)
//   codec.intra_qp               MLVC Q index, 0..63
//   codec.gop_size               MLVC I period in coded frames, or
//                                use_model_iframe_period for the bundle's own
//   provider.id                  e.g. onnxruntime-cpu, onnxruntime-qnn

#include "nvcr/codec/descriptor.hpp"

#include <cstdint>
#include <limits>

namespace nvcr::mlvc {

// codec.gop_size value that keeps the I period published in the bundle
// (1024 for MLVC-S, 64 for MLVC).
inline constexpr std::uint32_t use_model_iframe_period = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] codec::CodecDescriptor codec_descriptor();

// Registers the "mlvc" codec adapter. Idempotent.
void register_codec();

}  // namespace nvcr::mlvc
