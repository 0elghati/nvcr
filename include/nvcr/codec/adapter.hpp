#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/codec/descriptor.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::provider::experimental {
class IProviderSession;
}

namespace nvcr::codec {

// A codec adapter composes codec semantics with a provider-owned execution
// session while preserving the common backend runtime.
class ICodecAdapter {
public:
    virtual ~ICodecAdapter() = default;

    [[nodiscard]] virtual CodecDescriptor descriptor() const = 0;
    [[nodiscard]] virtual CodecCapabilities capabilities() const = 0;
    [[nodiscard]] virtual OptionSchema encoder_options() const = 0;
    [[nodiscard]] virtual OptionSchema decoder_options() const = 0;

    // Creates runtime components for the given session configuration.
    [[nodiscard]] virtual Result<Components>
    create_components(
        const RuntimeConfiguration& configuration,
        std::shared_ptr<provider::experimental::IProviderSession> provider_session) = 0;
};

}  // namespace nvcr::codec
