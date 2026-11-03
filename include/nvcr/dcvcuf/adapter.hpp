#pragma once

#include "nvcr/codec/adapter.hpp"

namespace nvcr::dcvcuf {

[[nodiscard]] Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter();

}  // namespace nvcr::dcvcuf
