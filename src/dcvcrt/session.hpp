#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/codec/session.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::dcvcrt {

[[nodiscard]] Result<codec::Sessions> make_sessions(
    RuntimeConfiguration configuration,
    std::unique_ptr<codec::CodecBackend> backend);

}  // namespace nvcr::dcvcrt
