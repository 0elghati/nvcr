#pragma once

// DCVC-UF under nvcr.fx.v1, on any provider session that implements the profile.

#include "execution_session.hpp"

#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::dcvcuf {

[[nodiscard]] Result<std::shared_ptr<ExecutionSession>> make_fx_session(
    std::shared_ptr<provider::experimental::IProviderSession> provider, const RuntimeConfiguration& configuration);

}  // namespace nvcr::dcvcuf
