#pragma once

#include "nvcr/codec/descriptor.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <filesystem>

namespace nvcr::dcvcuf {

[[nodiscard]] codec::CodecDescriptor codec_descriptor();
void register_codec();
void register_execution_provider();

// Selects this codec, its pinned provider, and its model identity for a
// converted weight bundle. An empty path falls back to NVCR_DCVCUF_BUNDLE; a
// directory selects the default bundle file inside it. Callers apply generic
// options (device, QP, GOP, logging) afterwards.
[[nodiscard]] Result<RuntimeConfiguration> runtime_configuration(
    std::filesystem::path bundle);

}  // namespace nvcr::dcvcuf
