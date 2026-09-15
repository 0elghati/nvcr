#pragma once

#include "nvcr/runtime/runtime.hpp"

namespace nvcr::codec {

// Compatibility name only. Codec semantics live in adapter-owned sessions.
using Runtime = ::nvcr::Runtime;

}  // namespace nvcr::codec
