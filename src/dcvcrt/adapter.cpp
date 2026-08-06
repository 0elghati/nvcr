#include "nvcr/dcvcrt/adapter.hpp"

#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/dcvcrt/tensorrt_backend.hpp"
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

    [[nodiscard]] Result<codec::Components>
    create_components(const RuntimeConfiguration&) override {
        auto backend = make_tensorrt_backend();
        if (!backend) {
            return backend.error();
        }
        codec::Components components;
        components.codec = std::move(backend.value());
        return components;
    }
};

}  // namespace

Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter() {
    return std::unique_ptr<codec::ICodecAdapter>(std::make_unique<DcvcrtAdapter>());
}

}  // namespace nvcr::dcvcrt
