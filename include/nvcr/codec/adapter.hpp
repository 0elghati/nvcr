#pragma once

#include "nvcr/codec/descriptor.hpp"
#include "nvcr/codec/session.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::provider::experimental {
class IProviderSession;
}

namespace nvcr::codec {

// A codec adapter composes codec-owned encoder/decoder sessions with a
// provider-owned execution session.
class ICodecAdapter {
public:
    virtual ~ICodecAdapter() = default;

    [[nodiscard]] virtual CodecDescriptor descriptor() const = 0;
    [[nodiscard]] virtual CodecCapabilities capabilities() const = 0;
    [[nodiscard]] virtual OptionSchema encoder_options() const = 0;
    [[nodiscard]] virtual OptionSchema decoder_options() const = 0;

    // Creates initialized codec-owned sessions for the given configuration.
    [[nodiscard]] virtual Result<Sessions>
    create_sessions(
        const RuntimeConfiguration& configuration,
        std::shared_ptr<provider::experimental::IProviderSession> provider_session) = 0;
};

}  // namespace nvcr::codec
