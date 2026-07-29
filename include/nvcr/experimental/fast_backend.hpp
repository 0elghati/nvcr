#pragma once

#include "nvcr/codec/backend.hpp"

namespace nvcr::experimental {

[[nodiscard]] Result<std::unique_ptr<codec::CodecBackend>> make_fast_backend();

}  // namespace nvcr::experimental
