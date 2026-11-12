#pragma once

// DCVC-RT under nvcr.fx.v1, on any provider session that implements the profile.

#include "nvcr/codec/backend.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <memory>

namespace nvcr::dcvcrt {

[[nodiscard]] Result<std::unique_ptr<codec::CodecBackend>> make_fx_backend(
    std::shared_ptr<provider::experimental::IProviderSession> session);

}  // namespace nvcr::dcvcrt
