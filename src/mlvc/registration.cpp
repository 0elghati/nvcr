#include "nvcr/mlvc/codec.hpp"

#include "nvcr/codec/adapter.hpp"
#include "nvcr/runtime/registry.hpp"

#include "session.hpp"

#include <memory>
#include <string>
#include <utility>

namespace nvcr::mlvc {
namespace {

class MlvcAdapter final : public codec::ICodecAdapter {
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

    [[nodiscard]] Result<codec::Sessions> create_sessions(
        const RuntimeConfiguration& configuration,
        std::shared_ptr<provider::experimental::IProviderSession> provider_session) override {
        // MLVC parts are ONNX graphs with host-side glue; the TensorRT session
        // loads DCVC-RT engine bundles only.
        if (configuration.provider.id == "tensorrt") {
            return Error(ErrorCode::incompatible_target,
                         "MLVC runs through the ONNX Runtime providers, not tensorrt",
                         "mlvc");
        }
        return make_sessions(configuration, std::move(provider_session));
    }
};

}  // namespace

codec::CodecDescriptor codec_descriptor() {
    return {"mlvc", "MLVC (Microsoft Multi-platform Learned Video Codec)", 1};
}

void register_codec() {
    using namespace nvcr::codec;
    runtime::Registry::instance().register_codec({
        codec_descriptor(),
        CodecCapabilities{
            .supports_intra = true,
            .supports_predicted = true,
            .supports_bidirectional = false,
            .supports_hierarchical = false,
            .supports_delayed_output = false,
            .min_qp = 0,
            .max_qp = 63,
        },
        OptionSchema{{
            {"mlvc.q_index", "uint",
             "MLVC Q index (0 = lowest quality, 63 = highest)", "32", "0", "63", false},
            {"mlvc.iframe_period", "uint",
             "Coded frames between I frames (bundle default if unset)", {}, "1", "4294967294",
             false},
            {"mlvc.bundle", "path",
             "MLVC ONNX model bundle directory (bundle_manifest.json)", {}, {}, {}, true},
            {"video.width", "uint", "Frame width; even, at most the largest bundle canvas",
             {}, "2", "16384", false},
            {"video.height", "uint", "Frame height; even, at most the largest bundle canvas",
             {}, "2", "16384", false},
            {"video.pixel_format", "string", "Input pixel format", "yuv420p8", {}, {}, false},
        }},
        OptionSchema{{
            {"mlvc.bundle", "path",
             "MLVC ONNX model bundle containing the stream's canvas", {}, {}, {}, true},
        }},
        []() -> std::unique_ptr<codec::ICodecAdapter> {
            return std::make_unique<MlvcAdapter>();
        },
    });
}

}  // namespace nvcr::mlvc
