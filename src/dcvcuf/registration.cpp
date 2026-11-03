#include "nvcr/dcvcuf/backend.hpp"

#ifdef NVCR_HAS_DCVCUF_NATIVE
#include "native_session.hpp"
#endif

#include "nvcr/dcvcuf/adapter.hpp"
#include "nvcr/runtime/registry.hpp"

#include <cstdlib>
#include <string>

namespace nvcr::dcvcuf {
namespace {

constexpr const char* kProviderId = "dcvcuf-upstream-cuda";
constexpr const char* kModelId = "dcvcuf-cvpr2026-hts";
constexpr const char* kBundleFile = "dcvcuf-cvpr2026-hts.pt";
constexpr const char* kBundleEnvironment = "NVCR_DCVCUF_BUNDLE";

}  // namespace

codec::CodecDescriptor codec_descriptor() {
    return {
        "dcvc-uf",
        "DCVC-UF (CVPR 2026 HT-S)",
        1,
    };
}

void register_codec() {
    runtime::Registry::instance().register_codec({
        codec_descriptor(),
        codec::CodecCapabilities{
            .supports_intra = true,
            .supports_predicted = true,
            .supports_bidirectional = false,
            .supports_hierarchical = true,
            .supports_delayed_output = true,
            .min_qp = 0,
            .max_qp = 63,
        },
        codec::OptionSchema{{
            {"dcvc_uf.quality_index", "uint",
             "DCVC-UF image and HT-S quality index", "32", "0", "63", false},
            {"dcvc_uf.intra_period", "uint",
             "1 for all-intra or a multiple of the HT-S eight-frame delay",
             "32", "1", "65535", false},
            {"video.width", "uint", "Coded frame width in pixels",
             {}, "2", "7680", false},
            {"video.height", "uint", "Coded frame height in pixels",
             {}, "2", "4320", false},
            {"video.pixel_format", "string", "Input pixel format",
             "yuv420p8", {}, {}, false},
            {"dcvcuf_cuda.device_id", "int",
             "CUDA device index", "0", "0", "31", false},
        }},
        codec::OptionSchema{{
            {"dcvcuf_cuda.device_id", "int",
             "CUDA device index", "0", "0", "31", false},
        }},
        []() -> std::unique_ptr<codec::ICodecAdapter> {
            auto adapter = make_adapter();
            return adapter ? std::move(adapter.value()) : nullptr;
        },
    });
}

void register_execution_provider() {
#ifdef NVCR_HAS_DCVCUF_NATIVE
    runtime::Registry::instance().register_provider({
        provider::ProviderDescriptor{
            provider::ProviderKind::cuda_fused,
            kProviderId,
            "cvpr2026-pinned",
            "Pinned DCVC-UF fused CUDA provider",
        },
        provider::ProviderCapabilities{
            .supports_fp16 = true,
            .supports_int8 = false,
            .supports_dynamic_shapes = true,
            .supports_cuda_graphs = true,
            .target_device = "cuda",
        },
        {},
        {},
        make_native_session,
    });
#endif
}

Result<RuntimeConfiguration> runtime_configuration(std::filesystem::path bundle) {
    if (bundle.empty()) {
        if (const char* value = std::getenv(kBundleEnvironment))
            bundle = value;
    }
    // A TensorRT stage bundle (engine_manifest.json) runs on the tensorrt provider.
    const bool tensorrt = std::filesystem::is_regular_file(bundle / "engine_manifest.json");
    if (std::filesystem::is_directory(bundle) && !tensorrt)
        bundle /= kBundleFile;
    if (!tensorrt && !std::filesystem::is_regular_file(bundle)) {
        return Error(
            ErrorCode::missing_artifact,
            std::string("DCVC-UF requires a weight bundle path or ") + kBundleEnvironment,
            "dcvcuf");
    }
    RuntimeConfiguration configuration;
    // The converted bundle carries both image and HT-S weights; this codec's
    // provider reads it from the single artifact slot.
    configuration.artifacts.intra_engine_path = bundle;
    configuration.codec.id = codec_descriptor().id;
    configuration.provider.id = tensorrt ? "tensorrt" : kProviderId;
    configuration.artifacts.model_id = kModelId;
    configuration.stream.bitstream_model_id = kModelId;
    return configuration;
}

}  // namespace nvcr::dcvcuf
