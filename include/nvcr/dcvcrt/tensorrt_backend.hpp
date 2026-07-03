#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/dcvcrt/backend.hpp"

#include <memory>

namespace nvcr::dcvcrt {

// Available only when NVCR is configured with NVCR_ENABLE_TENSORRT=ON.
// The returned backend owns engines, execution contexts, CUDA streams, and
// binding buffers. No TensorRT type is exposed through this public boundary.
[[nodiscard]] Result<std::unique_ptr<CodecBackend>> make_tensorrt_backend();

}  // namespace nvcr::dcvcrt

