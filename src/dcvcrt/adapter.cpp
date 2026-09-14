#include "nvcr/dcvcrt/adapter.hpp"

#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/runtime/registry.hpp"

#if defined(NVCR_HAS_TENSORRT)
#include "nvcr/dcvcrt/tensorrt_backend.hpp"
#endif

#include <memory>

namespace nvcr::dcvcrt {
namespace {

class DcvcrtAdapter final : public codec::ICodecAdapter {
public:
    [[nodiscard]] codec::CodecDescriptor descriptor() const override {
        return codec_descriptor();
    }

    [[nodiscard]] codec::CodecCapabilities capabilities() const override {
        register_codec();
        const auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->capabilities : codec::CodecCapabilities{};
    }

    [[nodiscard]] codec::OptionSchema encoder_options() const override {
        register_codec();
        const auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->encoder_options : codec::OptionSchema{};
    }

    [[nodiscard]] codec::OptionSchema decoder_options() const override {
        register_codec();
        const auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->decoder_options : codec::OptionSchema{};
    }

    [[nodiscard]] Result<codec::Components>
    create_components(
        const RuntimeConfiguration& configuration,
        std::shared_ptr<provider::experimental::IProviderSession>
            provider_session) override {
#if defined(NVCR_HAS_TENSORRT)
        static_cast<void>(configuration);
        auto backend = make_tensorrt_backend(std::move(provider_session));
        if (!backend) return backend.error();
        codec::Components components;
        components.codec = std::move(backend.value());
        return components;
#else
        static_cast<void>(configuration);
        static_cast<void>(provider_session);
        return Error(
            ErrorCode::dependency_unavailable,
            "DCVC-RT has no production execution provider in this build",
            "dcvcrt");
#endif
    }
};

}  // namespace

Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter() {
    return std::unique_ptr<codec::ICodecAdapter>(std::make_unique<DcvcrtAdapter>());
}

}  // namespace nvcr::dcvcrt
