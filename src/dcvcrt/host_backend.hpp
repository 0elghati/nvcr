#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <memory>

namespace nvcr::dcvcrt {

// Runs DCVC-RT through any provider session whose buffers expose host bytes.
// The DCVC-RT orchestration and host_ops own the inter-stage transforms; the
// provider session only loads and executes stages. Each stage is loaded with
// artifact.path set to the bundle root and artifact.component_id naming the
// stage (for example "i_analysis"); providers resolve their own file format.
[[nodiscard]] Result<std::unique_ptr<codec::CodecBackend>> make_host_backend(
    std::shared_ptr<provider::experimental::IProviderSession> provider_session);

}  // namespace nvcr::dcvcrt
