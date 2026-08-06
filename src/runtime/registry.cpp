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
    const std::string& provider_id = artifact.provider_id.empty()
        ? preferred_provider_id_
        : artifact.provider_id;
    if (provider_id.empty()) {
        return Error(
            ErrorCode::missing_provider,
            "artifact does not identify an execution provider",
            "registry");
    }

    std::scoped_lock lock(registry_.mutex_);
    const auto provider = std::find_if(
        registry_.providers_.begin(), registry_.providers_.end(),
        [&](const ProviderEntry& entry) { return entry.descriptor.id == provider_id; });
    if (provider == registry_.providers_.end() || !provider->factory) {
        return Error(
            ErrorCode::missing_provider,
            "execution provider is not registered: " + provider_id,
            "registry");
    }

    auto instance = provider->factory();
    if (!instance) {
        return Error(
            ErrorCode::dependency_unavailable,
            "execution provider factory returned null: " + provider_id,
            "registry");
    }
    if (!instance->supports(artifact)) {
        return Error(
            ErrorCode::incompatible_target,
            "execution provider does not support artifact: " + artifact.component_id,
            "registry");
    }
    return instance->load(artifact);
}

Result<std::shared_ptr<provider::IExecutable>> RuntimeServices::resolve(
    const artifacts::Resolver& resolver,
    const artifacts::ArtifactRequest& request) const {
    auto selected = resolver.resolve(request);
    if (!selected) return selected.error();
    return resolve(selected.value().candidate.artifact);
}

}  // namespace nvcr::runtime
