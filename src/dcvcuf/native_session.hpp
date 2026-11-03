#pragma once

#include "execution_session.hpp"

#include <memory>

namespace nvcr::dcvcuf {

[[nodiscard]] Result<std::shared_ptr<provider::experimental::IProviderSession>>
make_native_session(const RuntimeConfiguration& configuration);

}  // namespace nvcr::dcvcuf
