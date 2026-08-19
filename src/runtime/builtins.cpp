#include "nvcr/runtime/registry.hpp"

#if defined(NVCR_HAS_DCVCRT)
#include "nvcr/dcvcrt/backend.hpp"
#endif

namespace nvcr::runtime {

void register_builtin_components() {
#if defined(NVCR_HAS_DCVCRT)
    dcvcrt::register_codec();
    dcvcrt::register_execution_providers();
#endif
}

}  // namespace nvcr::runtime
