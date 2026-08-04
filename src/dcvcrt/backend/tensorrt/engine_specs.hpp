#pragma once

#include <NvInfer.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace nvcr::dcvcrt {

struct TensorSpec final {
    std::string_view name;
    nvinfer1::TensorIOMode mode;
    std::array<std::int64_t, 4> dimensions;
};

struct EngineSpec final {
    std::string_view filename;
    std::span<const TensorSpec> tensors;
};

extern const std::array<EngineSpec, 14> engine_specs;

}  // namespace nvcr::dcvcrt
