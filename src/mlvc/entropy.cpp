#include "entropy.hpp"

#include "host_ops.hpp"

#include <msrtc_rans/EntropyCoder.h>

#include <exception>
#include <new>
#include <string>
#include <system_error>
#include <utility>

namespace nvcr::mlvc {
namespace {

constexpr std::string_view subsystem = "mlvc.entropy";
constexpr auto variant = msrtc_rans::RansVariant::RansByte;
constexpr int symbol_bits = 16;
constexpr int bypass_bits = 2;

Error rans_error(ErrorCode code, std::string_view what, const std::error_code& error) {
    return Error(code, std::string(what) + ": " + error.message(), std::string(subsystem));
}

template <typename T>
msrtc_rans::span<const T> view(const std::vector<T>& values) {
    return {values.data(), values.size()};
}

}  // namespace

struct EntropyCoder::Impl final {
    msrtc_rans::EntropyEncoder gaussian_encoder;
    msrtc_rans::EntropyDecoder gaussian_decoder;
    msrtc_rans::EntropyEncoder z_encoder;
    msrtc_rans::EntropyDecoder z_decoder;
    ModelParameters parameters;
    std::uint32_t scale_levels{};
    std::uint32_t z_qp_num{};
    host_ops::LatentShape shape;

    [[nodiscard]] std::vector<std::int32_t> z_indices(std::uint32_t q_index) const {
        // BitEstimator._build_indices: q * channels + c, broadcast over H and W.
        std::vector<std::int32_t> indices(shape.z_elements());
        const std::size_t plane = static_cast<std::size_t>(shape.z_height) * shape.z_width;
        for (std::size_t channel = 0; channel < shape.z_channels; ++channel) {
            const auto index = static_cast<std::int32_t>(q_index * shape.z_channels + channel);
            std::fill_n(indices.begin() + static_cast<std::ptrdiff_t>(channel * plane), plane, index);
        }
        return indices;
    }
};

EntropyCoder::EntropyCoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
EntropyCoder::~EntropyCoder() = default;

Result<std::unique_ptr<EntropyCoder>> EntropyCoder::create(const BundleModel& model) {
    try {
        auto impl = std::make_unique<Impl>();
        const auto& gaussian = model.gaussian->pmf;
        const auto& bit = model.bit_estimator->pmf;
        for (const auto& [error, what] : {
                 std::pair{impl->gaussian_encoder.Initialize(
                               variant, view(gaussian.lengths), view(gaussian.offsets),
                               view(gaussian.table), symbol_bits, bypass_bits),
                           "Gaussian encoder"},
                 std::pair{impl->gaussian_decoder.Initialize(
                               variant, view(gaussian.lengths), view(gaussian.offsets),
                               view(gaussian.table), symbol_bits, bypass_bits),
                           "Gaussian decoder"},
                 std::pair{impl->z_encoder.Initialize(
                               variant, view(bit.lengths), view(bit.offsets), view(bit.table),
                               symbol_bits, bypass_bits),
                           "bit-estimator encoder"},
                 std::pair{impl->z_decoder.Initialize(
                               variant, view(bit.lengths), view(bit.offsets), view(bit.table),
                               symbol_bits, bypass_bits),
                           "bit-estimator decoder"}}) {
            if (error) return rans_error(ErrorCode::invalid_argument, what, error);
        }
        impl->parameters = model.parameters;
        impl->scale_levels = model.gaussian->scale_levels;
        impl->z_qp_num = model.bit_estimator->qp_num;
        impl->shape = host_ops::latent_shape(model.parameters);
        return std::unique_ptr<EntropyCoder>(new EntropyCoder(std::move(impl)));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "unable to create MLVC entropy coder",
                     std::string(subsystem));
    }
}

Result<std::vector<std::byte>> EntropyCoder::encode(
    const LatentSymbols& symbols, std::uint32_t q_index) const {
    const auto& shape = impl_->shape;
    if (q_index >= impl_->z_qp_num || symbols.z.size() != shape.z_elements() ||
        symbols.y_0.size() != shape.y_elements() || symbols.y_1.size() != shape.y_elements()) {
        return Error(ErrorCode::invalid_argument, "MLVC latent symbols do not match the model",
                     std::string(subsystem));
    }
    try {
        std::vector<float> z(symbols.z.begin(), symbols.z.end());
        const auto scales = host_ops::extract_scales(z, impl_->parameters, impl_->scale_levels);
        const auto z_indices = impl_->z_indices(q_index);

        msrtc_rans::HeapResizableBuffer buffer;
        msrtc_rans::RansEncoderStream stream;
        if (auto error = stream.Initialize(variant, buffer)) {
            return rans_error(ErrorCode::internal_error, "rANS stream", error);
        }
        if (auto error = impl_->gaussian_encoder.Encode(
                stream, view(scales.scales_1), view(symbols.y_1))) {
            return rans_error(ErrorCode::internal_error, "y_raw_1", error);
        }
        if (auto error = impl_->gaussian_encoder.Encode(
                stream, view(scales.scales_0), view(symbols.y_0))) {
            return rans_error(ErrorCode::internal_error, "y_raw_0", error);
        }
        if (auto error = impl_->z_encoder.Encode(stream, view(z_indices), view(symbols.z))) {
            return rans_error(ErrorCode::internal_error, "z_raw", error);
        }
        const auto bytes = stream.Flush();
        return std::vector<std::byte>(bytes.begin(), bytes.end());
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "MLVC entropy encoding ran out of memory",
                     std::string(subsystem));
    }
}

Result<LatentSymbols> EntropyCoder::decode(
    std::span<const std::byte> payload, std::uint32_t q_index) const {
    const auto& shape = impl_->shape;
    if (q_index >= impl_->z_qp_num) {
        return Error(ErrorCode::malformed_bitstream, "MLVC Q index is outside the PMF table",
                     std::string(subsystem));
    }
    try {
        msrtc_rans::RansDecoderStream stream;
        if (auto error = stream.Initialize(variant)) {
            return rans_error(ErrorCode::internal_error, "rANS stream", error);
        }
        if (auto error = stream.Open({payload.data(), payload.size()})) {
            return rans_error(ErrorCode::malformed_bitstream, "MLVC payload", error);
        }
        LatentSymbols symbols;
        symbols.z.resize(shape.z_elements());
        symbols.y_0.resize(shape.y_elements());
        symbols.y_1.resize(shape.y_elements());
        const auto z_indices = impl_->z_indices(q_index);
        if (auto error = impl_->z_decoder.Decode(
                {symbols.z.data(), symbols.z.size()}, view(z_indices), stream)) {
            return rans_error(ErrorCode::malformed_bitstream, "z_raw", error);
        }
        std::vector<float> z(symbols.z.begin(), symbols.z.end());
        const auto scales = host_ops::extract_scales(z, impl_->parameters, impl_->scale_levels);
        if (auto error = impl_->gaussian_decoder.Decode(
                {symbols.y_0.data(), symbols.y_0.size()}, view(scales.scales_0), stream)) {
            return rans_error(ErrorCode::malformed_bitstream, "y_raw_0", error);
        }
        if (auto error = impl_->gaussian_decoder.Decode(
                {symbols.y_1.data(), symbols.y_1.size()}, view(scales.scales_1), stream)) {
            return rans_error(ErrorCode::malformed_bitstream, "y_raw_1", error);
        }
        if (!stream.CheckEOF()) {
            return Error(ErrorCode::malformed_bitstream,
                         "MLVC payload has data after the last symbol",
                         std::string(subsystem));
        }
        return symbols;
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "MLVC entropy decoding ran out of memory",
                     std::string(subsystem));
    }
}

}  // namespace nvcr::mlvc
