#pragma once

// Running a codec's stages on a generic provider session: provider buffers,
// a pool that reuses them across frames, and one call per stage. Shared by the
// codecs that run under nvcr.fx.v1, whatever provider implements it.

#include "stream_numerics.hpp"

#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace nvcr::codec::fx {

namespace api = provider::experimental;

struct Buffer final {
    api::BufferHandle handle;
    std::array<std::int64_t, 4> shape{};
    api::TensorDataType type{api::TensorDataType::int32};
    bool host{false};

    [[nodiscard]] std::size_t count() const {
        return static_cast<std::size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
    }
    [[nodiscard]] std::size_t bytes() const;
    [[nodiscard]] std::span<std::int32_t> ints() const {
        return {reinterpret_cast<std::int32_t*>(handle->host_bytes().data()), count()};
    }
    [[nodiscard]] std::span<std::uint16_t> halves() const {
        return {reinterpret_cast<std::uint16_t*>(handle->host_bytes().data()), count()};
    }
};

// host: the codec reads or writes the values; otherwise device memory is preferred.
[[nodiscard]] Result<Buffer> allocate(api::IProviderSession& session, std::array<std::int64_t, 4> shape, bool host,
                                      api::TensorDataType type = api::TensorDataType::int32);

// Storage is reused across frames: a pooled buffer is free when only the pool holds it.
class BufferPool final {
public:
    [[nodiscard]] Result<Buffer> take(api::IProviderSession& session, std::array<std::int64_t, 4> shape, bool host,
                                      api::TensorDataType type = api::TensorDataType::int32);

private:
    std::vector<Buffer> pool_;
};

struct Bind final {
    std::string_view name;
    const Buffer* buffer;
};

// A codec's nvcr.fx.v1 stage: its program and the program's tensors (empty names pad).
struct StageTensors final {
    std::string_view component;
    std::array<std::string_view, 7> inputs;
    std::array<std::string_view, 5> outputs;
};

// The codec's fx stages from the programs in directory, bound by component,
// and the stream identity over their digests and the entropy rules the codec
// keeps next to them (entropy_file).
[[nodiscard]] Result<NormativeStages> normative_stages(const RuntimeConfiguration& configuration,
                                                       std::string_view codec_id,
                                                       const std::filesystem::path& directory,
                                                       std::span<const StageTensors> stages,
                                                       std::string_view entropy_file);

// Runs a stage and waits for it.
[[nodiscard]] Result<void> run(api::IProviderSession& session, const api::ExecutableStage& stage,
                               const std::vector<Bind>& in, const std::vector<Bind>& out);

}  // namespace nvcr::codec::fx
