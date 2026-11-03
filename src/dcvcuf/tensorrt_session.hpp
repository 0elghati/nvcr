#pragma once

#include "execution_session.hpp"

#include <memory>

namespace nvcr::dcvcuf {

// Wraps a TensorRT provider session opened on a DCVC-UF stage bundle; returns
// null when the session is another provider's.
[[nodiscard]] Result<std::shared_ptr<ExecutionSession>> make_tensorrt_session(
    std::shared_ptr<provider::experimental::IProviderSession> provider);

}  // namespace nvcr::dcvcuf
