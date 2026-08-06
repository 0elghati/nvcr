#include "nvcr/dcvcrt/backend.hpp"

namespace nvcr::dcvcrt {

codec::CodecDescriptor codec_descriptor() {
    return {
        "dcvc-rt",
        "DCVC-RT (Deep Video Compression Real-Time)",
        1,
    };
}

void register_codec() {
    using namespace nvcr::codec;
    runtime::Registry::instance().register_codec({
        codec_descriptor(),
        // DCVC-RT v1 capabilities.
        CodecCapabilities{
            .supports_intra        = true,
            .supports_predicted    = true,
            .supports_bidirectional = false,
            .supports_hierarchical  = false,
            .supports_delayed_output = false,
            .min_qp = 0,
            .max_qp = 63,
        },
        // Encoder option schema (namespaced dcvc_rt.*).
        OptionSchema{{
            {"dcvc_rt.quality_index", "uint",
             "QP / quality index (0 = lowest quality, 63 = highest)",
             "32", "0", "63", false},
            {"dcvc_rt.intra_period", "uint",
             "GOP size: frames between forced intra frames",
             "32", "1", "65535", false},
            {"dcvc_rt.model_profile", "string",
             "Engine profile name, e.g. '720p-fp16' (inferred from dimensions if unset)",
             "", {}, {}, false},
            {"dcvc_rt.integerized", "bool",
             "Use INT8-quantized engines (experimental, not for v1 release)",
             "false", {}, {}, false},
            {"video.width",  "uint", "Coded frame width in pixels",
             {}, "2", "7680", false},
            {"video.height", "uint", "Coded frame height in pixels",
             {}, "2", "4320", false},
            {"video.pixel_format", "string", "Input pixel format",
             "yuv420p8", {}, {}, false},
            {"tensorrt.device_id", "int",
             "CUDA device index", "0", "0", "31", false},
            {"tensorrt.execution_mode", "string",
             "TensorRT execution policy: automatic | low_memory | performance",
             "automatic", {}, {}, false},
        }},
        // Decoder option schema (mostly mirrors encoder minus encode-only fields).
        OptionSchema{{
            {"dcvc_rt.model_profile", "string",
             "Engine profile name (inferred from stream dimensions if unset)",
             "", {}, {}, false},
            {"tensorrt.device_id", "int",
             "CUDA device index", "0", "0", "31", false},
            {"tensorrt.execution_mode", "string",
             "TensorRT execution policy: automatic | low_memory | performance",
             "automatic", {}, {}, false},
        }},
        {},
    });
}

}  // namespace nvcr::dcvcrt
