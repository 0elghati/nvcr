#include "fx_stages.hpp"

#include "../common/sha256.hpp"
#include "../deterministic/fx_program.hpp"
#include "nvcr/numerics/profiles.hpp"

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

Result<NormativeStages> normative_stages(const RuntimeConfiguration& configuration, std::string_view codec_id,
                                        const std::filesystem::path& directory, std::span<const StageTensors> stages,
                                        std::string_view entropy_file) {
    NormativeStages result;
    std::vector<NormativeStageIdentity> identities;
    for (const auto& [component, inputs, outputs] : stages) {
        auto digest = deterministic::fx::program_digest(directory, component);
        if (!digest) return digest.error();
        auto bytes = detail::sha256_bytes(digest.value());
        if (!bytes) return Error(ErrorCode::internal_error, "program digest is not SHA-256", "codec.fx");
        identities.push_back(NormativeStageIdentity{std::string(component), *bytes});
        api::ExecutableStageDescriptor descriptor;
        descriptor.stage_id = std::string(component);
        descriptor.artifact.codec_id = std::string(codec_id);
        descriptor.artifact.model_set_id = configuration.artifacts.model_id;
        descriptor.artifact.component_id = std::string(component);
        descriptor.artifact.provider_id = configuration.provider.id;
        descriptor.artifact.precision = "int32";
        descriptor.artifact.path = directory.string();
        descriptor.artifact.expected_digest.assign(bytes->begin(), bytes->end());
        descriptor.numerics_profile = std::string(numerics::fx_v1);
        const auto add = [&](std::string_view name, api::TensorAccess access) {
            if (name.empty()) return;
            descriptor.tensors.push_back(api::TensorContract{std::string(name), api::TensorDataType::int32, access,
                                                             {{1, 1}, {1, 65535}, {1, 65535}, {1, 65535}}});
        };
        for (const auto name : inputs) add(name, api::TensorAccess::read);
        for (const auto name : outputs) add(name, api::TensorAccess::write);
        result.descriptors.push_back(std::move(descriptor));
    }
    auto entropy = detail::sha256_file(directory / entropy_file, "codec.fx");
    if (!entropy) return entropy.error();
    identities.push_back(NormativeStageIdentity{"fx_entropy", *detail::sha256_bytes(entropy.value())});
    auto identity = NumericsIdentityIO::make(std::string(numerics::fx_v1), identities);
    if (!identity) return identity.error();
    result.identity = std::move(identity.value());
    return result;
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
