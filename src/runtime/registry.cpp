#include "nvcr/runtime/registry.hpp"

#include <algorithm>

namespace nvcr::runtime {

Registry& Registry::instance() {
    static Registry instance;
    return instance;
}

void Registry::register_codec(CodecEntry entry) {
    std::scoped_lock lock(mutex_);
    // Replace existing entry with the same id so re-registration is idempotent.
    auto it = std::find_if(codecs_.begin(), codecs_.end(),
        [&](const CodecEntry& e) { return e.descriptor.id == entry.descriptor.id; });
    if (it != codecs_.end()) {
        *it = std::move(entry);
    } else {
        codecs_.push_back(std::move(entry));
    }
}

void Registry::register_provider(ProviderEntry entry) {
    std::scoped_lock lock(mutex_);
    auto it = std::find_if(providers_.begin(), providers_.end(),
        [&](const ProviderEntry& e) {
            return e.descriptor.id == entry.descriptor.id;
        });
    if (it != providers_.end()) {
        *it = std::move(entry);
    } else {
        providers_.push_back(std::move(entry));
    }
}

std::vector<CodecEntry> Registry::codecs() const {
    std::scoped_lock lock(mutex_);
    return codecs_;
}

std::vector<ProviderEntry> Registry::providers() const {
    std::scoped_lock lock(mutex_);
    return providers_;
}

std::optional<CodecEntry> Registry::find_codec(std::string_view id) const {
    std::scoped_lock lock(mutex_);
    for (const auto& entry : codecs_) {
        if (entry.descriptor.id == id) return entry;
    }
    return std::nullopt;
}

Result<std::unique_ptr<codec::ICodecAdapter>> Registry::create_codec(
    std::string_view id) const {
    auto entry = find_codec(id);
    if (!entry || !entry->factory) {
        return Error(
            ErrorCode::dependency_unavailable,
            "codec has no registered adapter factory: " + std::string(id),
            "registry");
    }
    auto adapter = entry->factory();
    if (!adapter) {
        return Error(
            ErrorCode::internal_error,
            "registered codec adapter factory returned null",
            "registry");
    }
    return adapter;
}

std::optional<ProviderEntry> Registry::find_provider(std::string_view id) const {
    std::scoped_lock lock(mutex_);
    for (const auto& entry : providers_) {
        if (entry.descriptor.id == id) return entry;
    }
    return std::nullopt;
}

bool Registry::compatible(
    std::string_view codec_id,
    std::string_view provider_id) const {
    std::scoped_lock lock(mutex_);
    const bool codec_found = std::any_of(codecs_.begin(), codecs_.end(),
        [&](const CodecEntry& e) { return e.descriptor.id == codec_id; });
    if (!codec_found) return false;
    const bool provider_found = std::any_of(providers_.begin(), providers_.end(),
        [&](const ProviderEntry& e) { return e.descriptor.id == provider_id; });
    return provider_found;
}

// ---------------------------------------------------------------------------
// RuntimeServices
// ---------------------------------------------------------------------------

Result<std::shared_ptr<provider::IExecutable>>
RuntimeServices::resolve(const provider::ArtifactDescriptor& artifact) const {
    // Try preferred provider first, then any compatible provider.
    auto try_provider = [&](const ProviderEntry& entry)
        -> std::optional<Result<std::shared_ptr<provider::IExecutable>>> {
        if (!entry.factory) return std::nullopt;
        auto instance = entry.factory();
        if (!instance) return std::nullopt;
        if (!instance->supports(artifact)) return std::nullopt;
        return instance->load(artifact);
    };

    {
        std::scoped_lock lock(registry_.mutex_);
        // Preferred first.
        for (const auto& entry : registry_.providers_) {
            if (entry.descriptor.id == preferred_provider_id_) {
                if (auto result = try_provider(entry)) return std::move(*result);
            }
        }
        // Any compatible provider as fallback.
        for (const auto& entry : registry_.providers_) {
            if (entry.descriptor.id == preferred_provider_id_) continue;
            if (auto result = try_provider(entry)) return std::move(*result);
        }
    }

    return Error(
        ErrorCode::missing_artifact,
        "no compatible provider found for artifact: " + artifact.component_id,
        "registry");
}

Result<std::shared_ptr<provider::IExecutable>> RuntimeServices::resolve(
    const artifacts::Resolver& resolver,
    const artifacts::ArtifactRequest& request) const {
    auto selected = resolver.resolve(request);
    if (!selected) return selected.error();
    return resolve(selected.value().candidate.artifact);
}

}  // namespace nvcr::runtime
