#include "fx_stages.hpp"

#include <string>

namespace nvcr::codec::fx {

std::size_t Buffer::bytes() const {
    return count() * (type == api::TensorDataType::int32 ? 4U : type == api::TensorDataType::float16 ? 2U : 1U);
}

Result<Buffer> allocate(api::IProviderSession& session, std::array<std::int64_t, 4> shape, bool host,
                        api::TensorDataType type) {
    Buffer buffer{nullptr, shape, type, host};
    const std::array<api::MemoryDomain, 3> domains{api::MemoryDomain::provider_device, api::MemoryDomain::pinned_host,
                                                   api::MemoryDomain::host};
    for (std::size_t i = host ? 1 : 0; i < domains.size(); ++i) {
        auto allocated = session.allocate(buffer.bytes(), domains[i]);
        if (allocated && (!host || !allocated.value()->host_bytes().empty())) {
            buffer.handle = std::move(allocated.value());
            return buffer;
        }
    }
    return Error(ErrorCode::backend_error, "cannot allocate a provider buffer", "codec.fx");
}

Result<Buffer> BufferPool::take(api::IProviderSession& session, std::array<std::int64_t, 4> shape, bool host,
                                api::TensorDataType type) {
    Buffer wanted{nullptr, shape, type, host};
    for (const auto& pooled : pool_) {
        if (pooled.handle.use_count() == 1 && pooled.host == host && pooled.bytes() == wanted.bytes()) {
            wanted.handle = pooled.handle;
            return wanted;
        }
    }
    auto made = allocate(session, shape, host, type);
    if (made) pool_.push_back(made.value());
    return made;
}

Result<void> run(api::IProviderSession& session, const api::ExecutableStage& stage, const std::vector<Bind>& in,
                 const std::vector<Bind>& out) {
    std::vector<api::TensorView> views;
    for (const auto* list : {&in, &out}) {
        for (const auto& b : *list) {
            views.push_back({b.name, b.buffer->type, {b.buffer->shape.begin(), b.buffer->shape.end()},
                             list == &in ? api::TensorAccess::read : api::TensorAccess::write,
                             {b.buffer->handle, 0, b.buffer->bytes()}});
        }
    }
    auto completion = session.submit(stage, views, {});
    if (!completion) return completion.error();
    return completion.value()->wait();
}

}  // namespace nvcr::codec::fx
