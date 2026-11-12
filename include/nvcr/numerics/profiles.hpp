#pragma once

#include <string_view>

// Numerics profile ids. A codec lists the profiles it runs under; a provider
// claims the exact ones it implements. native is the provider's own
// arithmetic: no stage is normative and the stream carries no numerics identity.
namespace nvcr::numerics {

inline constexpr std::string_view native = "native";
inline constexpr std::string_view intconv_v1 = "nvcr.intconv.v1";

}  // namespace nvcr::numerics
