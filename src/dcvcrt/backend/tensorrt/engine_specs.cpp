#include "engine_specs.hpp"

namespace nvcr::dcvcrt {
namespace {

constexpr std::array<TensorSpec, 3> analysis_tensors{{
    {"frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"q_enc", nvinfer1::TensorIOMode::kINPUT, {1, 368, 1, 1}},
    {"y", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> hyper_analysis_tensors{{
    {"y_padded", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"z", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> hyper_synthesis_tensors{{
    {"z_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"params_padded", nvinfer1::TensorIOMode::kOUTPUT, {1, 514, -1, -1}},
    {"common_padded", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> spatial_prior_tensors{{
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 512, -1, -1}},
    {"scales_means", nvinfer1::TensorIOMode::kOUTPUT, {1, 512, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> synthesis_tensors{{
    {"y_hat", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_dec", nvinfer1::TensorIOMode::kINPUT, {1, 368, 1, 1}},
    {"frame_hat", nvinfer1::TensorIOMode::kOUTPUT, {1, 3, -1, -1}},
}};

constexpr std::array<TensorSpec, 4> p_reference_frame_tensors{{
    {"reference_frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"q_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 4> p_reference_feature_tensors{{
    {"reference_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 4> p_analysis_tensors{{
    {"frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_encoder", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"y", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> p_hyper_analysis_tensors{{
    {"y_padded", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"z", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> p_prior_tensors{{
    {"z_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"params", nvinfer1::TensorIOMode::kOUTPUT, {1, 384, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> p_spatial_prior_tensors{{
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 512, -1, -1}},
    {"scales_means", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 6> p_synthesis_tensors{{
    {"y_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_decoder", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"q_recon", nvinfer1::TensorIOMode::kINPUT, {1, 320, 1, 1}},
    {"frame_hat", nvinfer1::TensorIOMode::kOUTPUT, {1, 3, -1, -1}},
    {"feature", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};

}  // namespace

const std::array<EngineSpec, 14> engine_specs{{
    {"i_analysis.plan", analysis_tensors},
    {"i_hyper_analysis.plan", hyper_analysis_tensors},
    {"i_hyper_synthesis.plan", hyper_synthesis_tensors},
    {"i_spatial_prior_1.plan", spatial_prior_tensors},
    {"i_spatial_prior_2.plan", spatial_prior_tensors},
    {"i_spatial_prior_3.plan", spatial_prior_tensors},
    {"i_synthesis.plan", synthesis_tensors},
    {"p_reference_frame.plan", p_reference_frame_tensors},
    {"p_reference_feature.plan", p_reference_feature_tensors},
    {"p_analysis.plan", p_analysis_tensors},
    {"p_hyper_analysis.plan", p_hyper_analysis_tensors},
    {"p_prior.plan", p_prior_tensors},
    {"p_spatial_prior.plan", p_spatial_prior_tensors},
    {"p_synthesis.plan", p_synthesis_tensors},
}};

}  // namespace nvcr::dcvcrt
