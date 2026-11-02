#include "bundle.hpp"

#include "../common/json.hpp"
#include "../common/sha256.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <tuple>
#include <utility>

namespace nvcr::mlvc {
namespace {

namespace fs = std::filesystem;
using detail::json::Value;

constexpr std::string_view subsystem = "mlvc.bundle";
// The only encoder/decoder interface MLVC's public split models declare
// (video/conversion/types.py); FP32 exports keep the same name.
constexpr std::string_view interface_type = "fp16_scale_sending_no_reset_1p";
constexpr std::string_view split_type = "dmc61sbr_e1d1";
constexpr std::string_view encoder_part = "MLVCEncoder";
constexpr std::string_view decoder_part = "MLVCDecoder";
constexpr std::array<std::string_view, 3> encoder_inputs{"x", "ref_feature", "q_index_shifted"};
constexpr std::array<std::string_view, 4> encoder_outputs{"feature", "z_raw", "y_raw_0", "y_raw_1"};
constexpr std::array<std::string_view, 5> decoder_inputs{
    "z_raw", "y_raw_0", "y_raw_1", "ref_feature", "q_index_shifted"};
constexpr std::array<std::string_view, 2> decoder_outputs{"x_hat", "feature"};

Error bundle_error(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), std::string(subsystem));
}

Result<const Value*> member(const Value& object, std::string_view key, std::string_view where) {
    const auto* value = object.find(key);
    if (value == nullptr) {
        return bundle_error(std::string(where) + " is missing '" + std::string(key) + "'");
    }
    return value;
}

Result<const Value*> object_member(
    const Value& object, std::string_view key, std::string_view where) {
    auto value = member(object, key, where);
    if (!value) return value.error();
    if (value.value()->kind != Value::Kind::object) {
        return bundle_error(std::string(where) + "." + std::string(key) + " must be an object");
    }
    return value;
}

Result<std::string> string_member(
    const Value& object, std::string_view key, std::string_view where) {
    auto value = member(object, key, where);
    if (!value) return value.error();
    if (value.value()->kind != Value::Kind::string) {
        return bundle_error(std::string(where) + "." + std::string(key) + " must be a string");
    }
    return value.value()->string_value;
}

Result<std::uint32_t> as_uint(const Value& value, std::string_view where) {
    if (value.kind != Value::Kind::number || !value.integer_value ||
        *value.integer_value < 0 ||
        *value.integer_value > std::numeric_limits<std::uint32_t>::max()) {
        return bundle_error(std::string(where) + " must be a non-negative integer");
    }
    return static_cast<std::uint32_t>(*value.integer_value);
}

Result<std::uint32_t> uint_member(
    const Value& object, std::string_view key, std::string_view where) {
    auto value = member(object, key, where);
    if (!value) return value.error();
    return as_uint(*value.value(), std::string(where) + "." + std::string(key));
}

Result<std::optional<std::uint32_t>> optional_uint_member(
    const Value& object, std::string_view key, std::string_view where) {
    const auto* value = object.find(key);
    if (value == nullptr || value->is_null()) return std::optional<std::uint32_t>{};
    auto parsed = as_uint(*value, std::string(where) + "." + std::string(key));
    if (!parsed) return parsed.error();
    return std::optional<std::uint32_t>(parsed.value());
}

template <typename Integer>
Result<std::vector<Integer>> int_array_member(
    const Value& object, std::string_view key, std::string_view where) {
    auto value = member(object, key, where);
    if (!value) return value.error();
    const auto name = std::string(where) + "." + std::string(key);
    if (value.value()->kind != Value::Kind::array) return bundle_error(name + " must be an array");
    std::vector<Integer> output;
    output.reserve(value.value()->array_value.size());
    for (const auto& element : value.value()->array_value) {
        if (element.kind != Value::Kind::number || !element.integer_value ||
            *element.integer_value < std::numeric_limits<Integer>::min() ||
            *element.integer_value > std::numeric_limits<Integer>::max()) {
            return bundle_error(name + " must contain integers in range");
        }
        output.push_back(static_cast<Integer>(*element.integer_value));
    }
    return output;
}

// Bundle members are plain file names in the bundle root.
Result<fs::path> bundle_file(const fs::path& root, const std::string& name) {
    if (name.empty() || name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos || name == "." || name == "..") {
        return bundle_error("bundle file name is not a plain file name: " + name);
    }
    auto path = root / name;
    std::error_code error;
    if (!fs::is_regular_file(path, error)) {
        return Error(ErrorCode::missing_artifact, "bundle file is missing: " + path.string(),
                     std::string(subsystem));
    }
    return path;
}

Result<PmfTable> parse_pmf(const Value& document, std::string_view where) {
    PmfTable pmf;
    auto lengths = int_array_member<std::int32_t>(document, "pmf_lengths", where);
    if (!lengths) return lengths.error();
    auto offsets = int_array_member<std::int32_t>(document, "pmf_offsets", where);
    if (!offsets) return offsets.error();
    auto table = int_array_member<std::int32_t>(document, "pmf_table", where);
    if (!table) return table.error();
    pmf.lengths = std::move(lengths.value());
    pmf.offsets = std::move(offsets.value());
    pmf.table = std::move(table.value());
    std::int64_t total = 0;
    for (const auto length : pmf.lengths) {
        if (length < 2) return bundle_error(std::string(where) + " has a PMF shorter than two");
        total += length;
    }
    if (pmf.lengths.empty() || pmf.lengths.size() != pmf.offsets.size() ||
        total != static_cast<std::int64_t>(pmf.table.size())) {
        return bundle_error(std::string(where) + " PMF table sizes are inconsistent");
    }
    return pmf;
}

Result<std::shared_ptr<const GaussianPmf>> load_gaussian(const fs::path& path) {
    auto document = detail::json::parse_file(path, subsystem);
    if (!document) return document.error();
    const auto where = path.filename().string();
    auto pmf = parse_pmf(document.value(), where);
    if (!pmf) return pmf.error();
    auto levels = uint_member(document.value(), "scale_levels", where);
    if (!levels) return levels.error();
    const auto* index_space = document.value().find("index_space");
    // The upsample scale decoder sends distribution indices, not scales.
    if (index_space == nullptr || index_space->kind != Value::Kind::boolean ||
        !index_space->boolean_value) {
        return bundle_error(where + " must use index_space");
    }
    if (levels.value() == 0U || pmf.value().lengths.size() != levels.value()) {
        return bundle_error(where + " has one PMF per scale level");
    }
    return std::make_shared<const GaussianPmf>(
        GaussianPmf{std::move(pmf.value()), levels.value()});
}

Result<std::shared_ptr<const BitEstimatorPmf>> load_bit_estimator(const fs::path& path) {
    auto document = detail::json::parse_file(path, subsystem);
    if (!document) return document.error();
    const auto where = path.filename().string();
    auto pmf = parse_pmf(document.value(), where);
    if (!pmf) return pmf.error();
    auto qp_num = uint_member(document.value(), "qp_num", where);
    if (!qp_num) return qp_num.error();
    auto channels = uint_member(document.value(), "channels", where);
    if (!channels) return channels.error();
    if (static_cast<std::uint64_t>(qp_num.value()) * channels.value() !=
        pmf.value().lengths.size()) {
        return bundle_error(where + " PMF count is not qp_num * channels");
    }
    return std::make_shared<const BitEstimatorPmf>(
        BitEstimatorPmf{std::move(pmf.value()), qp_num.value(), channels.value()});
}

Result<ModelParameters> parse_parameters(const Value& metadata, std::string_view where) {
    ModelParameters parameters;
    struct Field {
        std::string_view key;
        std::uint32_t* target;
    };
    const std::array<Field, 12> fields{{
        {"model_width", &parameters.width},
        {"model_height", &parameters.height},
        {"qp_num", &parameters.qp_num},
        {"total_qp_num", &parameters.total_qp_num},
        {"feature_channels", &parameters.feature_channels},
        {"latent_channels", &parameters.latent_channels},
        {"hyperprior_channels", &parameters.hyperprior_channels},
        {"downsample_feature", &parameters.downsample_feature},
        {"downsample_latent", &parameters.downsample_latent},
        {"downsample_hyperprior", &parameters.downsample_hyperprior},
        {"y_scale_repeat", &parameters.y_scale_repeat},
        {"ltr_start_idx", &parameters.ltr_start_idx},
    }};
    for (const auto& field : fields) {
        auto value = uint_member(metadata, field.key, where);
        if (!value) return value.error();
        *field.target = value.value();
    }
    auto frame_index_map = int_array_member<std::uint32_t>(metadata, "frame_index_map", where);
    if (!frame_index_map) return frame_index_map.error();
    auto qp_shift = int_array_member<std::uint32_t>(metadata, "qp_shift", where);
    if (!qp_shift) return qp_shift.error();
    parameters.frame_index_map = std::move(frame_index_map.value());
    parameters.qp_shift = std::move(qp_shift.value());
    for (const auto& [key, target] : {
             std::pair{std::string_view("iframe_period"), &parameters.iframe_period},
             std::pair{std::string_view("reset_period"), &parameters.reset_period},
             std::pair{std::string_view("ltr_period"), &parameters.ltr_period}}) {
        auto value = optional_uint_member(metadata, key, where);
        if (!value) return value.error();
        *target = value.value();
    }

    auto scale_decoder = string_member(metadata, "scale_decoder_type", where);
    if (!scale_decoder) return scale_decoder.error();
    if (scale_decoder.value() != "upsample") {
        return Error(ErrorCode::not_implemented,
                     std::string(where) + " uses unsupported scale decoder " +
                         scale_decoder.value(),
                     std::string(subsystem));
    }
    // NVCR converts pixels exactly as MLVC does only for the published unit
    // pixel range.
    const auto* pixel_range = metadata.find("pixel_range");
    if (pixel_range == nullptr || pixel_range->kind != Value::Kind::number ||
        pixel_range->number_value != 1.0) {
        return Error(ErrorCode::not_implemented,
                     std::string(where) + " must use pixel_range 1.0",
                     std::string(subsystem));
    }
    for (const auto* key : {"image_layout", "feature_layout"}) {
        const auto* layout = metadata.find(key);
        if (layout != nullptr && !(layout->kind == Value::Kind::string &&
                                   layout->string_value == "nchw")) {
            return Error(ErrorCode::not_implemented,
                         std::string(where) + " uses a non-NCHW " + key,
                         std::string(subsystem));
        }
    }

    const auto& p = parameters;
    const auto latent_stride = static_cast<std::uint64_t>(p.downsample_latent);
    const bool shapes_valid =
        p.width > 0U && p.height > 0U && p.downsample_latent > 0U &&
        p.downsample_feature > 0U && p.downsample_hyperprior > 0U &&
        p.width % latent_stride == 0U && p.height % latent_stride == 0U &&
        p.width % p.downsample_feature == 0U && p.height % p.downsample_feature == 0U &&
        p.feature_channels > 0U && p.latent_channels > 0U && p.latent_channels % 2U == 0U &&
        p.y_scale_repeat > 0U && p.latent_channels % p.y_scale_repeat == 0U &&
        p.latent_channels / p.y_scale_repeat <= p.hyperprior_channels;
    if (!shapes_valid) return bundle_error(std::string(where) + " has inconsistent dimensions");
    if (p.qp_num == 0U || p.frame_index_map.empty() || p.qp_shift.empty()) {
        return bundle_error(std::string(where) + " has an empty Q schedule");
    }
    std::uint32_t maximum_shift = 0;
    for (const auto index : p.frame_index_map) {
        if (index >= p.qp_shift.size()) {
            return bundle_error(std::string(where) + " frame_index_map exceeds qp_shift");
        }
        maximum_shift = std::max(maximum_shift, p.qp_shift[index]);
    }
    if (static_cast<std::uint64_t>(p.qp_num) - 1U + maximum_shift >= p.total_qp_num) {
        return bundle_error(std::string(where) + " shifted Q exceeds total_qp_num");
    }
    if ((p.iframe_period && *p.iframe_period == 0U) ||
        (p.reset_period && *p.reset_period == 0U) || (p.ltr_period && *p.ltr_period == 0U)) {
        return bundle_error(std::string(where) + " has a zero period");
    }
    return parameters;
}

// Bundles exported before MLVC 64b10d4 list fields as strings; later ones as
// {name, layout}. A missing layout means NCHW.
Result<void> check_fields(
    const Value& part, std::string_view key, std::span<const std::string_view> expected,
    std::string_view where) {
    auto fields = member(part, key, where);
    if (!fields) return fields.error();
    const auto name = std::string(where) + "." + std::string(key);
    if (fields.value()->kind != Value::Kind::array) return bundle_error(name + " must be an array");
    std::set<std::string, std::less<>> names;
    for (const auto& field : fields.value()->array_value) {
        if (field.kind == Value::Kind::string) {
            names.insert(field.string_value);
            continue;
        }
        auto field_name = string_member(field, "name", name);
        if (!field_name) return field_name.error();
        const auto* layout = field.find("layout");
        if (layout != nullptr && !layout->is_null() &&
            !(layout->kind == Value::Kind::string && layout->string_value == "nchw")) {
            return Error(ErrorCode::not_implemented,
                         name + " field " + field_name.value() + " is not NCHW",
                         std::string(subsystem));
        }
        names.insert(field_name.value());
    }
    if (names.size() != expected.size() ||
        !std::all_of(expected.begin(), expected.end(),
                     [&](std::string_view field) { return names.contains(field); })) {
        return bundle_error(name + " does not match the MLVC split interface");
    }
    return {};
}

Result<Precision> check_conversion_metadata(
    const fs::path& path, const ModelParameters& parameters) {
    auto document = detail::json::parse_file(path, subsystem);
    if (!document) return document.error();
    const auto where = path.filename().string();
    auto params = object_member(document.value(), "params", where);
    if (!params) return params.error();
    auto split = object_member(*params.value(), "split_model_params", where + ".params");
    if (!split) return split.error();
    auto split_name = string_member(*split.value(), "split_type", where);
    if (!split_name) return split_name.error();
    if (split_name.value() != split_type) {
        return Error(ErrorCode::not_implemented,
                     where + " uses unsupported split " + split_name.value(),
                     std::string(subsystem));
    }
    auto width = uint_member(*split.value(), "model_width", where);
    if (!width) return width.error();
    auto height = uint_member(*split.value(), "model_height", where);
    if (!height) return height.error();
    if (width.value() != parameters.width || height.value() != parameters.height) {
        return bundle_error(where + " canvas does not match the bundle manifest");
    }

    auto parts = object_member(document.value(), "model_parts_metadata", where);
    if (!parts) return parts.error();
    if (parts.value()->object_value.size() != 2U) {
        return bundle_error(where + " must describe exactly the encoder and decoder");
    }
    std::optional<std::string> precision;
    for (const auto& [part_name, inputs, outputs] : {
             std::tuple{encoder_part, std::span<const std::string_view>(encoder_inputs),
                        std::span<const std::string_view>(encoder_outputs)},
             std::tuple{decoder_part, std::span<const std::string_view>(decoder_inputs),
                        std::span<const std::string_view>(decoder_outputs)}}) {
        const auto part_where = where + "." + std::string(part_name);
        auto part = object_member(*parts.value(), part_name, where);
        if (!part) return part.error();
        auto part_precision = string_member(*part.value(), "precision", part_where);
        if (!part_precision) return part_precision.error();
        if (precision && *precision != part_precision.value()) {
            return bundle_error(where + " parts use different precisions");
        }
        precision = part_precision.value();
        auto checked = check_fields(*part.value(), "input_fields", inputs, part_where);
        if (!checked) return checked.error();
        checked = check_fields(*part.value(), "output_fields", outputs, part_where);
        if (!checked) return checked.error();
    }
    if (*precision == "fp16") return Precision::fp16;
    if (*precision == "fp32") return Precision::fp32;
    return Error(ErrorCode::incompatible_precision,
                 where + " uses unsupported precision " + *precision, std::string(subsystem));
}

bool hex_digest(std::string_view value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

Result<ModelPart> parse_part(
    const fs::path& root, const Value& manifest, const Value& model_parts,
    std::string_view part_name, std::string_view where) {
    auto part = object_member(model_parts, part_name, where);
    if (!part) return part.error();
    const auto* function_name = part.value()->find("function_name");
    if (function_name != nullptr && !function_name->is_null()) {
        return bundle_error(std::string(where) + " ONNX parts cannot name a function");
    }
    auto registry_id = string_member(*part.value(), "registry_id", where);
    if (!registry_id) return registry_id.error();
    auto registry = object_member(manifest, "model_registry", "bundle_manifest.json");
    if (!registry) return registry.error();
    auto entry = object_member(*registry.value(), registry_id.value(), "model_registry");
    if (!entry) return entry.error();
    auto model_name = string_member(*entry.value(), "model_path", registry_id.value());
    if (!model_name) return model_name.error();
    auto weights_name = string_member(*entry.value(), "weights_path", registry_id.value());
    if (!weights_name) return weights_name.error();
    auto digest = string_member(*entry.value(), "sha256", registry_id.value());
    if (!digest) return digest.error();
    if (!hex_digest(digest.value())) {
        return bundle_error(registry_id.value() + " has an invalid sha256");
    }

    ModelPart output;
    auto model_path = bundle_file(root, model_name.value());
    if (!model_path) return model_path.error();
    auto weights_path = bundle_file(root, weights_name.value());
    if (!weights_path) return weights_path.error();
    output.model_path = std::move(model_path.value());
    output.weights_path = std::move(weights_path.value());
    output.sha256 = std::move(digest.value());
    if (output.model_path.extension() != ".onnx") {
        return Error(ErrorCode::not_implemented,
                     registry_id.value() + " is not an ONNX part", std::string(subsystem));
    }
    output.component_id = output.model_path.stem().string();
    const bool simple_id = std::all_of(
        output.component_id.begin(), output.component_id.end(), [](char character) {
            return (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') || character == '_';
        });
    if (output.component_id.empty() || !simple_id) {
        return bundle_error("ONNX part name is not an identifier: " + model_name.value());
    }
    return output;
}

}  // namespace

Result<Bundle> load_bundle(const fs::path& root) {
    auto manifest_path = bundle_file(root, "bundle_manifest.json");
    if (!manifest_path) return manifest_path.error();
    auto manifest = detail::json::parse_file(manifest_path.value(), subsystem);
    if (!manifest) return manifest.error();
    const auto& document = manifest.value();
    constexpr std::string_view where = "bundle_manifest.json";

    Bundle bundle;
    bundle.root = root;
    auto name = string_member(document, "bundle_name", where);
    if (!name) return name.error();
    auto model_type = string_member(document, "model_type", where);
    if (!model_type) return model_type.error();
    auto target = string_member(document, "target_device", where);
    if (!target) return target.error();
    if (model_type.value() != "onnx") {
        return Error(ErrorCode::incompatible_target,
                     "NVCR runs ONNX MLVC bundles; this bundle is " + model_type.value(),
                     std::string(subsystem));
    }
    bundle.name = std::move(name.value());
    bundle.target_device = std::move(target.value());

    auto manifests = object_member(document, "model_manifests", where);
    if (!manifests) return manifests.error();
    auto metadata = object_member(document, "model_metadata", where);
    if (!metadata) return metadata.error();
    std::map<fs::path, std::shared_ptr<const GaussianPmf>> gaussian_cache;
    std::map<fs::path, std::shared_ptr<const BitEstimatorPmf>> bit_cache;

    for (const auto& [canvas_id, model_manifest] : manifests.value()->object_value) {
        const auto model_where = std::string("model_manifests.") + canvas_id;
        if (model_manifest.kind != Value::Kind::object) {
            return bundle_error(model_where + " must be an object");
        }
        for (const auto* key : {"encoder_interface_type", "decoder_interface_type"}) {
            auto type = string_member(model_manifest, key, model_where);
            if (!type) return type.error();
            if (type.value() != interface_type) {
                return Error(ErrorCode::not_implemented,
                             model_where + " uses unsupported interface " + type.value(),
                             std::string(subsystem));
            }
        }
        const auto* scale_model = model_manifest.find("scale_decoder_model_path");
        if (scale_model != nullptr && !scale_model->is_null()) {
            return Error(ErrorCode::not_implemented,
                         model_where + " requires a scale decoder model",
                         std::string(subsystem));
        }

        BundleModel model;
        model.canvas_id = canvas_id;
        model.stream_identity = bundle.name + "-" + canvas_id;
        auto model_metadata = object_member(*metadata.value(), canvas_id, "model_metadata");
        if (!model_metadata) return model_metadata.error();
        auto parameters = parse_parameters(*model_metadata.value(), "model_metadata." + canvas_id);
        if (!parameters) return parameters.error();
        model.parameters = std::move(parameters.value());
        if (canvas_id != std::to_string(model.parameters.width) + "x" +
                std::to_string(model.parameters.height)) {
            return bundle_error(model_where + " id does not match its canvas");
        }

        auto metadata_name = string_member(model_manifest, "metadata_path", model_where);
        if (!metadata_name) return metadata_name.error();
        auto metadata_path = bundle_file(root, metadata_name.value());
        if (!metadata_path) return metadata_path.error();
        auto precision = check_conversion_metadata(metadata_path.value(), model.parameters);
        if (!precision) return precision.error();
        model.precision = precision.value();

        auto gaussian_name = string_member(model_manifest, "gaussian_pmf_path", model_where);
        if (!gaussian_name) return gaussian_name.error();
        auto gaussian_path = bundle_file(root, gaussian_name.value());
        if (!gaussian_path) return gaussian_path.error();
        auto& gaussian = gaussian_cache[gaussian_path.value()];
        if (!gaussian) {
            auto loaded = load_gaussian(gaussian_path.value());
            if (!loaded) return loaded.error();
            gaussian = std::move(loaded.value());
        }
        model.gaussian = gaussian;

        auto bit_name = string_member(model_manifest, "bit_estimator_pmf_path", model_where);
        if (!bit_name) return bit_name.error();
        auto bit_path = bundle_file(root, bit_name.value());
        if (!bit_path) return bit_path.error();
        auto& bit_estimator = bit_cache[bit_path.value()];
        if (!bit_estimator) {
            auto loaded = load_bit_estimator(bit_path.value());
            if (!loaded) return loaded.error();
            bit_estimator = std::move(loaded.value());
        }
        model.bit_estimator = bit_estimator;
        if (model.bit_estimator->channels != model.parameters.hyperprior_channels ||
            model.bit_estimator->qp_num < model.parameters.qp_num) {
            return bundle_error(model_where + " bit-estimator PMFs do not match the model");
        }

        auto parts = object_member(model_manifest, "model_parts", model_where);
        if (!parts) return parts.error();
        if (parts.value()->object_value.size() != 2U) {
            return bundle_error(model_where + " must have exactly an encoder and a decoder");
        }
        auto encoder = parse_part(root, document, *parts.value(), encoder_part, model_where);
        if (!encoder) return encoder.error();
        auto decoder = parse_part(root, document, *parts.value(), decoder_part, model_where);
        if (!decoder) return decoder.error();
        model.encoder = std::move(encoder.value());
        model.decoder = std::move(decoder.value());
        bundle.models.push_back(std::move(model));
    }
    if (bundle.models.empty()) return bundle_error("bundle has no models");
    std::sort(bundle.models.begin(), bundle.models.end(),
              [](const BundleModel& left, const BundleModel& right) {
                  const auto left_area =
                      static_cast<std::uint64_t>(left.parameters.width) * left.parameters.height;
                  const auto right_area =
                      static_cast<std::uint64_t>(right.parameters.width) * right.parameters.height;
                  return left_area != right_area ? left_area < right_area :
                                                   left.canvas_id < right.canvas_id;
              });
    return bundle;
}

Result<const BundleModel*> select_model(
    const Bundle& bundle,
    std::uint32_t frame_width,
    std::uint32_t frame_height,
    std::string_view canvas_id) {
    const auto fits = [&](const BundleModel& model, bool rotated) {
        const auto width = rotated ? frame_height : frame_width;
        const auto height = rotated ? frame_width : frame_height;
        return width <= model.parameters.width && height <= model.parameters.height;
    };
    if (!canvas_id.empty()) {
        for (const auto& model : bundle.models) {
            if (model.canvas_id != canvas_id) continue;
            if (!fits(model, false) && !fits(model, true)) {
                return Error(ErrorCode::invalid_argument,
                             std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                                 " does not fit MLVC canvas " + model.canvas_id,
                             std::string(subsystem));
            }
            return &model;
        }
        return Error(ErrorCode::missing_artifact,
                     "bundle has no MLVC canvas " + std::string(canvas_id),
                     std::string(subsystem));
    }
    for (const bool rotated : {false, true}) {
        for (const auto& model : bundle.models) {
            if (fits(model, rotated)) return &model;
        }
    }
    return Error(ErrorCode::invalid_argument,
                 std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                     " is larger than every MLVC canvas in the bundle; MLVC does not scale",
                 std::string(subsystem));
}

Result<void> verify_digests(const BundleModel& model) {
    for (const auto* part : {&model.encoder, &model.decoder}) {
        const std::array<fs::path, 2> files{part->model_path, part->weights_path};
        auto digest = detail::sha256_files(files, subsystem);
        if (!digest) return digest.error();
        if (digest.value() != part->sha256) {
            return Error(ErrorCode::digest_mismatch,
                         "MLVC bundle digest mismatch: " + part->model_path.string(),
                         std::string(subsystem));
        }
    }
    return {};
}

}  // namespace nvcr::mlvc
