#pragma once

// MLVC model bundles as written by MLVC's `convert.py bundle`
// (video/conversion/_model_bundler.py). NVCR consumes a bundle unchanged:
// bundle_manifest.json, one metadata_<W>x<H>.json per canvas, the Gaussian and
// bit-estimator PMF tables, and MLVCEncoder/MLVCDecoder ONNX parts sharing
// weights.bin.

#include "nvcr/common/error.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nvcr::mlvc {

enum class Precision : std::uint8_t {
    fp16,
    fp32,
};

// Integer PMF tables quantized to 16 bits by the exporter.
struct PmfTable final {
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> table;
};

struct GaussianPmf final {
    PmfTable pmf;
    std::uint32_t scale_levels{};
};

struct BitEstimatorPmf final {
    PmfTable pmf;
    std::uint32_t qp_num{};
    std::uint32_t channels{};
};

// bundle_manifest.json model_metadata for one canvas.
struct ModelParameters final {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t qp_num{};
    std::uint32_t total_qp_num{};
    std::vector<std::uint32_t> frame_index_map;
    std::vector<std::uint32_t> qp_shift;
    std::uint32_t feature_channels{};
    std::uint32_t latent_channels{};
    std::uint32_t hyperprior_channels{};
    std::uint32_t downsample_feature{};
    std::uint32_t downsample_latent{};
    std::uint32_t downsample_hyperprior{};
    std::uint32_t y_scale_repeat{};
    std::optional<std::uint32_t> iframe_period;
    std::optional<std::uint32_t> reset_period;
    std::uint32_t ltr_start_idx{};
    std::optional<std::uint32_t> ltr_period;
};

struct ModelPart final {
    std::string component_id;  // ONNX file stem, e.g. MLVCEncoder_320x192
    std::filesystem::path model_path;
    std::filesystem::path weights_path;
    std::string sha256;  // SHA-256 of the model bytes followed by the weights bytes
};

struct BundleModel final {
    std::string canvas_id;  // "<W>x<H>", the bundle's model id
    std::string stream_identity;  // bundle name plus canvas; written into NVAU
    ModelParameters parameters;
    Precision precision{Precision::fp16};
    ModelPart encoder;
    ModelPart decoder;
    std::shared_ptr<const GaussianPmf> gaussian;
    std::shared_ptr<const BitEstimatorPmf> bit_estimator;
};

struct Bundle final {
    std::filesystem::path root;
    std::string name;
    std::string target_device;
    std::vector<BundleModel> models;  // ascending canvas area
};

[[nodiscard]] Result<Bundle> load_bundle(const std::filesystem::path& root);

// MLVC fits a frame natively, else with height and width swapped, and never
// scales. NVCR picks the smallest canvas that fits natively, then the smallest
// that fits swapped. A non-empty canvas_id selects that canvas explicitly.
[[nodiscard]] Result<const BundleModel*> select_model(
    const Bundle& bundle,
    std::uint32_t frame_width,
    std::uint32_t frame_height,
    std::string_view canvas_id = {});

// Recomputes the bundle registry digests for both parts of one canvas.
[[nodiscard]] Result<void> verify_digests(const BundleModel& model);

}  // namespace nvcr::mlvc
