#pragma once

#include "nvcr/codec/adapter.hpp"

#include <memory>

namespace nvcr::dcvcrt {

// Factory for the DCVC-RT codec adapter.
[[nodiscard]] Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter();

}  // namespace nvcr::dcvcrt
