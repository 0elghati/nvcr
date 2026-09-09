#pragma once

#include "nvcr/provider/experimental/session.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace nvcr::test_support {

void register_test_provider();

[[nodiscard]] std::shared_ptr<provider::experimental::IProviderSession>
make_test_provider_session();

[[nodiscard]] Result<void> write_test_buffer(
    const provider::experimental::BufferHandle& buffer,
    std::size_t offset,
    std::span<const std::byte> bytes);

[[nodiscard]] Result<std::vector<std::byte>> read_test_buffer(
    const provider::experimental::BufferHandle& buffer,
    std::size_t offset,
    std::size_t size);

}  // namespace nvcr::test_support
