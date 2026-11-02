#pragma once

#include "nvcr/codec/session.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <memory>

namespace nvcr::mlvc {

// Loads the MLVC bundle named by the configuration and returns codec-owned
// encoder and decoder sessions that execute it through provider_session.
[[nodiscard]] Result<codec::Sessions> make_sessions(
    const RuntimeConfiguration& configuration,
    std::shared_ptr<provider::experimental::IProviderSession> provider_session);

}  // namespace nvcr::mlvc
