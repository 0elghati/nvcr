#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/codec/descriptor.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <memory>

namespace nvcr::runtime {
class RuntimeServices;
}

namespace nvcr::codec {

// Transitional adapter boundary used to separate codec semantics from
// execution-provider construction while preserving the existing backend runtime.
class ICodecAdapter {
public:
    virtual ~ICodecAdapter() = default;

    [[nodiscard]] virtual CodecDescriptor descriptor() const = 0;
    [[nodiscard]] virtual CodecCapabilities capabilities() const = 0;
    [[nodiscard]] virtual OptionSchema encoder_options() const = 0;
    [[nodiscard]] virtual OptionSchema decoder_options() const = 0;

    // Applies codec-owned defaults before the generic runtime validates the
    // configuration. Callers select the adapter first; an adapter may then
    // choose its default model, stream identity, and execution provider.
    // Explicit caller values must be preserved unless they are incompatible.
    [[nodiscard]] virtual Result<void>
    apply_defaults(RuntimeConfiguration& configuration) const {
        if (configuration.codec_id.empty()) configuration.codec_id = descriptor().id;
        if (configuration.model_id.empty()) configuration.model_id = descriptor().id;
        if (configuration.bitstream_model_id.empty()) {
            configuration.bitstream_model_id = descriptor().id;
        }
        return {};
    }

    // Creates runtime components for the given session configuration.
    [[nodiscard]] virtual Result<Components>
    create_components(
        const RuntimeConfiguration& configuration,
        const runtime::RuntimeServices& services) = 0;
};

}  // namespace nvcr::codec
