#pragma once

#include "nvcr/codec/descriptor.hpp"
#include "nvcr/codec/runtime.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::runtime {
class RuntimeServices;
}

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
        const RuntimeConfiguration&,
        std::shared_ptr<provider::experimental::IProviderSession>) {
        return Error(
            ErrorCode::not_implemented,
            "codec adapter does not implement session construction",
            "codec-adapter");
    }
    // 1.0.x construction hook retained for rebuilt clients and adapters.
    [[nodiscard]] virtual Result<Components> create_components(
        const RuntimeConfiguration&, const runtime::RuntimeServices&) {
        return Error(
            ErrorCode::not_implemented,
            "codec adapter does not implement component construction",
            "codec-adapter");
    }

};

}  // namespace nvcr::codec
