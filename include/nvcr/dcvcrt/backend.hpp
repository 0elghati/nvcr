#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/codec/descriptor.hpp"
#include "nvcr/runtime/registry.hpp"

namespace nvcr::dcvcrt {

using CodecBackend = codec::CodecBackend;
using CodecDecodeResult = codec::CodecDecodeResult;
using CodecEncodeResult = codec::CodecEncodeResult;
using Components = codec::Components;

// Register the DCVC-RT codec adapter entry in the global registry.
// Safe to call multiple times; re-registration is idempotent.
void register_codec();

// Register built-in execution-provider entries available in this build.
// Safe to call multiple times; re-registration is idempotent.
void register_execution_providers();

// Returns the DCVC-RT codec descriptor for callers that need it directly.
[[nodiscard]] codec::CodecDescriptor codec_descriptor();

}  // namespace nvcr::dcvcrt
