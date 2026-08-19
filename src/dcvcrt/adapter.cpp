#include "nvcr/dcvcrt/adapter.hpp"

#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/runtime/registry.hpp"

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

    [[nodiscard]] Result<void>
    apply_defaults(RuntimeConfiguration& configuration) const override {
        if (!configuration.codec_id.empty() && configuration.codec_id != codec_descriptor().id) {
            return Error(
                ErrorCode::invalid_argument,
                "DCVC-RT adapter was selected for a different codec id",
                "dcvcrt");
        }
        configuration.codec_id = codec_descriptor().id;
        if (configuration.model_id.empty()) configuration.model_id = "dcvcrt-cvpr2025";
        if (configuration.bitstream_model_id.empty()) configuration.bitstream_model_id = "dcvcrt";
        if (configuration.provider_id.empty()) configuration.provider_id = "tensorrt";
        return {};
    }

    [[nodiscard]] Result<codec::Components>
    create_components(
        const RuntimeConfiguration& configuration,
        const runtime::RuntimeServices& services) override {
        register_execution_providers();
        return services.create_components(configuration);
    }
};

}  // namespace

Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter() {
    return std::unique_ptr<codec::ICodecAdapter>(std::make_unique<DcvcrtAdapter>());
}

}  // namespace nvcr::dcvcrt
