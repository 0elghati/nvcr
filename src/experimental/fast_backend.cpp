#include "nvcr/experimental/fast_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace nvcr::experimental {
namespace {

constexpr std::string_view subsystem = "experimental.fast";
constexpr std::array<std::byte, 4> payload_magic{
    std::byte{'N'}, std::byte{'V'}, std::byte{'M'}, std::byte{'1'}};
constexpr std::uint16_t payload_version = 1;
constexpr std::uint16_t predicted_flag = 1U;
constexpr std::uint16_t supported_flags = predicted_flag;
constexpr std::uint16_t motion_block_size = 32;
constexpr std::size_t coefficient_tile_size = 256;
constexpr std::size_t header_bytes = 36;

struct MotionVector final {
    std::int8_t x{};
    std::int8_t y{};
};

struct PlaneGeometry final {
    std::size_t offset{};
    std::uint32_t width{};
    std::uint32_t height{};
    bool chroma{};
};

constexpr std::array<PlaneGeometry, 3> plane_geometries(
    std::uint32_t width, std::uint32_t height) {
    const auto luma_size = static_cast<std::size_t>(width) * height;
    const auto chroma_size = static_cast<std::size_t>(width / 2U) * (height / 2U);
    return {{
        {0, width, height, false},
        {luma_size, width / 2U, height / 2U, true},
        {luma_size + chroma_size, width / 2U, height / 2U, true},
    }};
}

std::uint8_t byte_value(std::byte value) {
    return std::to_integer<std::uint8_t>(value);
}

void write_u16(std::vector<std::byte>& output, std::uint16_t value) {
    output.push_back(static_cast<std::byte>(value & 0xFFU));
    output.push_back(static_cast<std::byte>((value >> 8U) & 0xFFU));
}

void write_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
    }
}

std::uint16_t read_u16(std::span<const std::byte> bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(byte_value(bytes[offset])) |
        (static_cast<std::uint16_t>(byte_value(bytes[offset + 1])) << 8U));
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        value |= static_cast<std::uint32_t>(
                     byte_value(bytes[offset + shift / 8U]))
                 << shift;
    }
    return value;
}

std::uint16_t quantization_step(std::uint32_t qp) {
    // QP 20 starts at step 2 and every six QP values doubles the step.
    const auto shift = qp <= 20U ? 0U : std::min((qp - 20U + 5U) / 6U, 7U);
    return static_cast<std::uint16_t>(2U << shift);
}

int quantize(int residual, int step) {
    const auto magnitude = std::abs(residual);
    const auto rounded = (magnitude + step / 2) / step;
    return residual < 0 ? -rounded : rounded;
}

std::uint16_t zigzag(int value) {
    return static_cast<std::uint16_t>(
        value >= 0 ? value * 2 : (-value * 2) - 1);
}

int unzigzag(std::uint16_t value) {
    return (value & 1U) == 0U
               ? static_cast<int>(value / 2U)
               : -static_cast<int>(value / 2U) - 1;
}

std::uint8_t reconstructed_sample(int prediction, int coefficient, int step) {
    return static_cast<std::uint8_t>(
        std::clamp(prediction + coefficient * step, 0, 255));
}

int spatial_prediction(
    std::span<const std::byte> reconstructed,
    std::uint32_t width,
    std::uint32_t x,
    std::uint32_t y) {
    const auto index = static_cast<std::size_t>(y) * width + x;
    if (x == 0U && y == 0U) return 128;
    if (y == 0U) return byte_value(reconstructed[index - 1U]);
    if (x == 0U) return byte_value(reconstructed[index - width]);

    const auto left = static_cast<int>(byte_value(reconstructed[index - 1U]));
    const auto top = static_cast<int>(byte_value(reconstructed[index - width]));
    const auto top_left =
        static_cast<int>(byte_value(reconstructed[index - width - 1U]));
    if (top_left >= std::max(left, top)) return std::min(left, top);
    if (top_left <= std::min(left, top)) return std::max(left, top);
    return left + top - top_left;
}

template <typename Function>
void parallel_rows(std::uint32_t row_count, Function function) {
    const auto available = std::max(1U, std::thread::hardware_concurrency());
    const auto worker_count = std::min(row_count, available);
    if (worker_count < 2U || row_count < 8U) {
        for (std::uint32_t row = 0; row < row_count; ++row) function(row);
        return;
    }

    std::atomic<std::uint32_t> next_row{0};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const auto row = next_row.fetch_add(1U, std::memory_order_relaxed);
                if (row >= row_count) return;
                function(row);
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

std::uint64_t motion_cost(
    std::span<const std::byte> source,
    std::span<const std::byte> reference,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t block_x,
    std::uint32_t block_y,
    int dx,
    int dy) {
    const auto x_end = std::min(block_x + motion_block_size, width);
    const auto y_end = std::min(block_y + motion_block_size, height);
    if (static_cast<int>(block_x) + dx < 0 ||
        static_cast<int>(block_y) + dy < 0 ||
        static_cast<int>(x_end) + dx > static_cast<int>(width) ||
        static_cast<int>(y_end) + dy > static_cast<int>(height)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    std::uint64_t cost = 0;
    for (auto y = block_y; y < y_end; y += 4U) {
        const auto source_row = static_cast<std::size_t>(y) * width;
        const auto reference_row =
            static_cast<std::size_t>(static_cast<int>(y) + dy) * width;
        for (auto x = block_x; x < x_end; x += 4U) {
            const auto source_value = static_cast<int>(byte_value(source[source_row + x]));
            const auto reference_value = static_cast<int>(
                byte_value(reference[reference_row + static_cast<int>(x) + dx]));
            cost += static_cast<std::uint64_t>(std::abs(source_value - reference_value));
        }
    }
    return cost;
}

std::vector<MotionVector> estimate_motion(
    std::span<const std::byte> source,
    std::span<const std::byte> reference,
    std::uint32_t width,
    std::uint32_t height) {
    const auto block_columns =
        (width + motion_block_size - 1U) / motion_block_size;
    const auto block_rows =
        (height + motion_block_size - 1U) / motion_block_size;
    std::vector<MotionVector> vectors(
        static_cast<std::size_t>(block_columns) * block_rows);

    parallel_rows(block_rows, [&](std::uint32_t block_row) {
        const auto y = block_row * motion_block_size;
        for (std::uint32_t block_column = 0; block_column < block_columns;
             ++block_column) {
            const auto x = block_column * motion_block_size;
            int best_x = 0;
            int best_y = 0;
            auto best_cost = motion_cost(
                source, reference, width, height, x, y, best_x, best_y);

            for (int dy = -8; dy <= 8; dy += 4) {
                for (int dx = -8; dx <= 8; dx += 4) {
                    const auto cost =
                        motion_cost(source, reference, width, height, x, y, dx, dy);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_x = dx;
                        best_y = dy;
                    }
                }
            }
            for (const int radius : {2, 1}) {
                const auto center_x = best_x;
                const auto center_y = best_y;
                for (int dy = -radius; dy <= radius; dy += radius) {
                    for (int dx = -radius; dx <= radius; dx += radius) {
                        const auto candidate_x = center_x + dx;
                        const auto candidate_y = center_y + dy;
                        const auto cost = motion_cost(
                            source, reference, width, height, x, y,
                            candidate_x, candidate_y);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_x = candidate_x;
                            best_y = candidate_y;
                        }
                    }
                }
            }
            vectors[static_cast<std::size_t>(block_row) * block_columns +
                    block_column] = {
                static_cast<std::int8_t>(best_x),
                static_cast<std::int8_t>(best_y)};
        }
    });
    return vectors;
}

int inter_prediction(
    std::span<const std::byte> reference,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t x,
    std::uint32_t y,
    bool chroma,
    std::span<const MotionVector> motion,
    std::uint32_t luma_block_columns) {
    const auto luma_x = chroma ? x * 2U : x;
    const auto luma_y = chroma ? y * 2U : y;
    const auto block_x = luma_x / motion_block_size;
    const auto block_y = luma_y / motion_block_size;
    const auto vector = motion[
        static_cast<std::size_t>(block_y) * luma_block_columns + block_x];
    const auto dx = chroma ? static_cast<int>(vector.x) / 2
                           : static_cast<int>(vector.x);
    const auto dy = chroma ? static_cast<int>(vector.y) / 2
                           : static_cast<int>(vector.y);
    const auto reference_x = std::clamp(static_cast<int>(x) + dx, 0,
                                        static_cast<int>(width) - 1);
    const auto reference_y = std::clamp(static_cast<int>(y) + dy, 0,
                                        static_cast<int>(height) - 1);
    return byte_value(reference[
        static_cast<std::size_t>(reference_y) * width + reference_x]);
}

std::vector<std::uint16_t> encode_intra_plane(
    std::span<const std::byte> source,
    std::span<std::byte> reconstructed,
    std::uint32_t width,
    std::uint32_t height,
    int step) {
    std::vector<std::uint16_t> coefficients(source.size());
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto index = static_cast<std::size_t>(y) * width + x;
            const auto prediction = spatial_prediction(reconstructed, width, x, y);
            const auto coefficient =
                quantize(static_cast<int>(byte_value(source[index])) - prediction, step);
            coefficients[index] = zigzag(coefficient);
            reconstructed[index] = static_cast<std::byte>(
                reconstructed_sample(prediction, coefficient, step));
        }
    }
    return coefficients;
}

std::vector<std::uint16_t> encode_inter_plane(
    std::span<const std::byte> source,
    std::span<const std::byte> reference,
    std::span<std::byte> reconstructed,
    std::uint32_t width,
    std::uint32_t height,
    bool chroma,
    std::span<const MotionVector> motion,
    std::uint32_t luma_block_columns,
    int step) {
    std::vector<std::uint16_t> coefficients(source.size());
    parallel_rows(height, [&](std::uint32_t y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto index = static_cast<std::size_t>(y) * width + x;
            const auto prediction = inter_prediction(
                reference, width, height, x, y, chroma, motion,
                luma_block_columns);
            const auto coefficient =
                quantize(static_cast<int>(byte_value(source[index])) - prediction, step);
            coefficients[index] = zigzag(coefficient);
            reconstructed[index] = static_cast<std::byte>(
                reconstructed_sample(prediction, coefficient, step));
        }
    });
    return coefficients;
}

std::vector<std::byte> pack_coefficients(
    std::span<const std::uint16_t> coefficients) {
    std::vector<std::byte> output;
    output.reserve(coefficients.size());
    for (std::size_t tile_start = 0; tile_start < coefficients.size();
         tile_start += coefficient_tile_size) {
        const auto tile_size =
            std::min(coefficient_tile_size, coefficients.size() - tile_start);
        std::uint16_t maximum = 0;
        for (std::size_t index = 0; index < tile_size; ++index) {
            maximum = std::max(maximum, coefficients[tile_start + index]);
        }
        const auto bits = static_cast<std::uint8_t>(std::bit_width(maximum));
        output.push_back(static_cast<std::byte>(bits));
        if (bits == 0U) continue;

        std::uint64_t accumulator = 0;
        unsigned accumulator_bits = 0;
        for (std::size_t index = 0; index < tile_size; ++index) {
            accumulator |=
                static_cast<std::uint64_t>(coefficients[tile_start + index])
                << accumulator_bits;
            accumulator_bits += bits;
            while (accumulator_bits >= 8U) {
                output.push_back(static_cast<std::byte>(accumulator & 0xFFU));
                accumulator >>= 8U;
                accumulator_bits -= 8U;
            }
        }
        if (accumulator_bits != 0U) {
            output.push_back(static_cast<std::byte>(accumulator & 0xFFU));
        }
    }
    return output;
}

Result<std::vector<std::uint16_t>> unpack_coefficients(
    std::span<const std::byte> payload,
    std::size_t coefficient_count) {
    try {
        std::vector<std::uint16_t> coefficients(coefficient_count);
        std::size_t payload_offset = 0;
        for (std::size_t tile_start = 0; tile_start < coefficient_count;
             tile_start += coefficient_tile_size) {
            if (payload_offset >= payload.size()) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "truncated fast coefficient tile",
                    std::string(subsystem));
            }
            const auto tile_size =
                std::min(coefficient_tile_size, coefficient_count - tile_start);
            const auto bits = byte_value(payload[payload_offset++]);
            if (bits > 9U) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "invalid fast coefficient width",
                    std::string(subsystem));
            }
            if (bits == 0U) continue;

            const auto packed_bytes = (tile_size * bits + 7U) / 8U;
            if (packed_bytes > payload.size() - payload_offset) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "truncated fast coefficient data",
                    std::string(subsystem));
            }
            const auto packed = payload.subspan(payload_offset, packed_bytes);
            const auto mask = static_cast<std::uint16_t>((1U << bits) - 1U);
            for (std::size_t index = 0; index < tile_size; ++index) {
                const auto bit_offset = index * bits;
                const auto byte_offset = bit_offset / 8U;
                const auto shift = static_cast<unsigned>(bit_offset % 8U);
                auto window = static_cast<std::uint16_t>(
                    byte_value(packed[byte_offset]));
                if (shift + bits > 8U) {
                    window = static_cast<std::uint16_t>(
                        window | static_cast<std::uint16_t>(
                                     static_cast<std::uint16_t>(
                                         byte_value(packed[byte_offset + 1U]))
                                     << 8U));
                }
                coefficients[tile_start + index] =
                    static_cast<std::uint16_t>((window >> shift) & mask);
            }
            payload_offset += packed_bytes;
        }
        if (payload_offset != payload.size()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "fast coefficient plane has trailing data",
                std::string(subsystem));
        }
        return coefficients;
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "fast coefficient allocation failed",
            std::string(subsystem));
    }
}

void decode_intra_plane(
    std::span<const std::uint16_t> coefficients,
    std::span<std::byte> reconstructed,
    std::uint32_t width,
    std::uint32_t height,
    int step) {
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto index = static_cast<std::size_t>(y) * width + x;
            const auto prediction = spatial_prediction(reconstructed, width, x, y);
            reconstructed[index] = static_cast<std::byte>(
                reconstructed_sample(
                    prediction, unzigzag(coefficients[index]), step));
        }
    }
}

void decode_inter_plane(
    std::span<const std::uint16_t> coefficients,
    std::span<const std::byte> reference,
    std::span<std::byte> reconstructed,
    std::uint32_t width,
    std::uint32_t height,
    bool chroma,
    std::span<const MotionVector> motion,
    std::uint32_t luma_block_columns,
    int step) {
    parallel_rows(height, [&](std::uint32_t y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto index = static_cast<std::size_t>(y) * width + x;
            const auto prediction = inter_prediction(
                reference, width, height, x, y, chroma, motion,
                luma_block_columns);
            reconstructed[index] = static_cast<std::byte>(
                reconstructed_sample(
                    prediction, unzigzag(coefficients[index]), step));
        }
    });
}

Result<void> validate_reference(
    const codec::SequenceStateView& state,
    std::uint32_t width,
    std::uint32_t height) {
    if (state.reference_frame == nullptr ||
        state.reference_frame->pixel_format() != PixelFormat::yuv420p8 ||
        state.reference_frame->width() != width ||
        state.reference_frame->height() != height) {
        return Error(
            ErrorCode::invalid_state,
            "predicted fast frame requires a matching YUV420P8 reference",
            std::string(subsystem));
    }
    return {};
}

class FastBackend final : public codec::CodecBackend {
public:
    Result<void> initialize(const RuntimeConfiguration& configuration) override {
        qp_ = std::min(configuration.intra_qp, 63U);
        step_ = quantization_step(qp_);
        return {};
    }

    Result<codec::CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const codec::SequenceStateView& state) override {
        if (frame.pixel_format() != PixelFormat::yuv420p8) {
            return Error(
                ErrorCode::invalid_argument,
                "experimental fast backend accepts only YUV420P8",
                std::string(subsystem));
        }
        const auto predicted = frame_type == FrameType::predicted;
        if (predicted) {
            auto valid = validate_reference(state, frame.width(), frame.height());
            if (!valid) return valid.error();
        }

        try {
            auto reconstructed = Frame::create(
                frame.width(), frame.height(), PixelFormat::yuv420p8,
                frame.timestamp());
            if (!reconstructed) return reconstructed.error();

            const auto geometries = plane_geometries(frame.width(), frame.height());
            const auto luma_size =
                static_cast<std::size_t>(frame.width()) * frame.height();
            const auto block_columns =
                (frame.width() + motion_block_size - 1U) / motion_block_size;
            std::vector<MotionVector> motion;
            if (predicted) {
                motion = estimate_motion(
                    frame.data().first(luma_size),
                    state.reference_frame->data().first(luma_size),
                    frame.width(), frame.height());
            }

            std::array<std::vector<std::byte>, 3> plane_payloads;
            for (std::size_t plane_index = 0; plane_index < geometries.size();
                 ++plane_index) {
                const auto& plane = geometries[plane_index];
                const auto plane_size =
                    static_cast<std::size_t>(plane.width) * plane.height;
                const auto source = frame.data().subspan(plane.offset, plane_size);
                auto output =
                    reconstructed.value().data().subspan(plane.offset, plane_size);
                std::vector<std::uint16_t> coefficients;
                if (predicted) {
                    const auto reference = state.reference_frame->data().subspan(
                        plane.offset, plane_size);
                    coefficients = encode_inter_plane(
                        source, reference, output, plane.width, plane.height,
                        plane.chroma, motion, block_columns, step_);
                } else {
                    coefficients = encode_intra_plane(
                        source, output, plane.width, plane.height, step_);
                }
                plane_payloads[plane_index] = pack_coefficients(coefficients);
                if (plane_payloads[plane_index].size() >
                    std::numeric_limits<std::uint32_t>::max()) {
                    return Error(
                        ErrorCode::resource_exhausted,
                        "fast coefficient plane is too large",
                        std::string(subsystem));
                }
            }

            if (motion.size() > std::numeric_limits<std::uint32_t>::max()) {
                return Error(
                    ErrorCode::resource_exhausted,
                    "fast motion field is too large",
                    std::string(subsystem));
            }
            std::vector<std::byte> payload;
            const auto motion_bytes = motion.size() * 2U;
            payload.reserve(
                header_bytes + motion_bytes + plane_payloads[0].size() +
                plane_payloads[1].size() + plane_payloads[2].size());
            payload.insert(payload.end(), payload_magic.begin(), payload_magic.end());
            write_u16(payload, payload_version);
            write_u16(payload, predicted ? predicted_flag : 0U);
            write_u32(payload, frame.width());
            write_u32(payload, frame.height());
            write_u16(payload, step_);
            write_u16(payload, motion_block_size);
            write_u32(payload, static_cast<std::uint32_t>(motion.size()));
            for (const auto& plane_payload : plane_payloads) {
                write_u32(payload, static_cast<std::uint32_t>(plane_payload.size()));
            }
            for (const auto vector : motion) {
                payload.push_back(static_cast<std::byte>(
                    static_cast<std::uint8_t>(vector.x)));
                payload.push_back(static_cast<std::byte>(
                    static_cast<std::uint8_t>(vector.y)));
            }
            for (const auto& plane_payload : plane_payloads) {
                payload.insert(
                    payload.end(), plane_payload.begin(), plane_payload.end());
            }
            return codec::CodecEncodeResult{
                std::move(payload), std::move(reconstructed.value()), {}, qp_};
        } catch (const std::bad_alloc&) {
            return Error(
                ErrorCode::resource_exhausted,
                "experimental fast encode allocation failed",
                std::string(subsystem));
        } catch (const std::exception& exception) {
            return Error(
                ErrorCode::internal_error,
                std::string("experimental fast encode failed: ") + exception.what(),
                std::string(subsystem));
        }
    }

    Result<codec::CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType frame_type,
        Timestamp timestamp,
        const codec::SequenceStateView& state) override {
        if (payload.size() < header_bytes ||
            !std::equal(payload_magic.begin(), payload_magic.end(), payload.begin())) {
            return Error(
                ErrorCode::malformed_bitstream,
                "invalid experimental fast payload",
                std::string(subsystem));
        }
        const auto version = read_u16(payload, 4);
        const auto flags = read_u16(payload, 6);
        if (version != payload_version || (flags & ~supported_flags) != 0U) {
            return Error(
                ErrorCode::malformed_bitstream,
                "unsupported experimental fast payload version or flags",
                std::string(subsystem));
        }
        const auto predicted = (flags & predicted_flag) != 0U;
        if (predicted != (frame_type == FrameType::predicted)) {
            return Error(
                ErrorCode::malformed_bitstream,
                "fast payload and packet frame types disagree",
                std::string(subsystem));
        }

        const auto width = read_u32(payload, 8);
        const auto height = read_u32(payload, 12);
        const auto step = read_u16(payload, 16);
        const auto block_size = read_u16(payload, 18);
        const auto motion_count = read_u32(payload, 20);
        const std::array<std::uint32_t, 3> plane_sizes{
            read_u32(payload, 24),
            read_u32(payload, 28),
            read_u32(payload, 32)};
        auto frame_size = frame_size_bytes(width, height, PixelFormat::yuv420p8);
        if (!frame_size) {
            return Error(
                ErrorCode::malformed_bitstream,
                "invalid fast payload dimensions",
                std::string(subsystem));
        }
        if (step < 2U || step > 256U || block_size != motion_block_size) {
            return Error(
                ErrorCode::malformed_bitstream,
                "invalid fast quantizer or motion block size",
                std::string(subsystem));
        }

        const auto block_columns =
            (width + motion_block_size - 1U) / motion_block_size;
        const auto block_rows =
            (height + motion_block_size - 1U) / motion_block_size;
        const auto expected_motion_count = static_cast<std::uint64_t>(
            predicted ? static_cast<std::uint64_t>(block_columns) * block_rows : 0U);
        if (motion_count != expected_motion_count) {
            return Error(
                ErrorCode::malformed_bitstream,
                "fast payload motion field size is invalid",
                std::string(subsystem));
        }
        const auto encoded_size =
            static_cast<std::uint64_t>(header_bytes) +
            static_cast<std::uint64_t>(motion_count) * 2U +
            static_cast<std::uint64_t>(plane_sizes[0]) + plane_sizes[1] +
            plane_sizes[2];
        if (encoded_size != payload.size()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "fast payload section sizes disagree",
                std::string(subsystem));
        }
        if (predicted) {
            auto valid = validate_reference(state, width, height);
            if (!valid) return valid.error();
        }

        try {
            std::vector<MotionVector> motion(motion_count);
            std::size_t offset = header_bytes;
            for (auto& vector : motion) {
                vector.x = static_cast<std::int8_t>(byte_value(payload[offset++]));
                vector.y = static_cast<std::int8_t>(byte_value(payload[offset++]));
            }
            std::array<std::span<const std::byte>, 3> plane_payloads;
            for (std::size_t plane_index = 0; plane_index < plane_sizes.size();
                 ++plane_index) {
                plane_payloads[plane_index] =
                    payload.subspan(offset, plane_sizes[plane_index]);
                offset += plane_sizes[plane_index];
            }

            auto frame = Frame::create(width, height, PixelFormat::yuv420p8, timestamp);
            if (!frame) return frame.error();
            const auto geometries = plane_geometries(width, height);
            for (std::size_t plane_index = 0; plane_index < geometries.size();
                 ++plane_index) {
                const auto& plane = geometries[plane_index];
                const auto plane_size =
                    static_cast<std::size_t>(plane.width) * plane.height;
                auto coefficients =
                    unpack_coefficients(plane_payloads[plane_index], plane_size);
                if (!coefficients) return coefficients.error();
                auto output = frame.value().data().subspan(plane.offset, plane_size);
                if (predicted) {
                    const auto reference = state.reference_frame->data().subspan(
                        plane.offset, plane_size);
                    decode_inter_plane(
                        coefficients.value(), reference, output, plane.width,
                        plane.height, plane.chroma, motion, block_columns, step);
                } else {
                    decode_intra_plane(
                        coefficients.value(), output, plane.width, plane.height,
                        step);
                }
            }
            return codec::CodecDecodeResult{std::move(frame.value()), {}};
        } catch (const std::bad_alloc&) {
            return Error(
                ErrorCode::resource_exhausted,
                "experimental fast decode allocation failed",
                std::string(subsystem));
        } catch (const std::exception& exception) {
            return Error(
                ErrorCode::internal_error,
                std::string("experimental fast decode failed: ") + exception.what(),
                std::string(subsystem));
        }
    }

    Result<void> flush() override { return {}; }
    void reset() noexcept override {}

private:
    std::uint32_t qp_{32};
    std::uint16_t step_{8};
};

}  // namespace

Result<std::unique_ptr<codec::CodecBackend>> make_fast_backend() {
    try {
        return std::unique_ptr<codec::CodecBackend>(std::make_unique<FastBackend>());
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate experimental fast backend",
            std::string(subsystem));
    }
}

}  // namespace nvcr::experimental
