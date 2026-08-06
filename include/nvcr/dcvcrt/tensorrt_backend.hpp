#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/provider/provider_api.hpp"

#include <memory>

namespace nvcr::dcvcrt {

// Available only when NVCR is configured with NVCR_ENABLE_TENSORRT=ON.
// The returned backend owns engines, execution contexts, CUDA streams, and
// binding buffers. No TensorRT type is exposed through this public boundary.
[[nodiscard]] Result<std::unique_ptr<CodecBackend>> make_tensorrt_backend();

// Register the TensorRT execution-provider entry in the global registry.
// Safe to call multiple times; re-registration is idempotent.
void register_tensorrt_provider();

// Returns the TensorRT provider descriptor for callers that need it directly.
[[nodiscard]] provider::ProviderDescriptor tensorrt_provider_descriptor();

}  // namespace nvcr::dcvcrt

