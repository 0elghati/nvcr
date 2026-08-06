#pragma once

// Static codec / execution-provider registry (Phase 3 of M-EXT).
//
// Registration is static: implementations call register_codec() /
// register_provider() at program start (e.g. from a namespace-scope variable
// or an explicit init() call).  No dynamic shared-library loading occurs in
// v1.  The registry is process-global, thread-safe for reads after all
// registrations complete.
//
// Clients query the registry to enumerate or locate implementations.
// The NVCR CLI and Runtime use it to drive "nvcr codec list/describe",
// "nvcr provider list/describe", and "nvcr compatibility check".

#include "nvcr/codec/descriptor.hpp"
#include "nvcr/provider/provider_api.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::runtime {

// ---------------------------------------------------------------------------
// Codec registry entry
// ---------------------------------------------------------------------------

struct CodecEntry {
    codec::CodecDescriptor descriptor;
    codec::CodecCapabilities capabilities;
    codec::OptionSchema encoder_options;
    codec::OptionSchema decoder_options;
};

// ---------------------------------------------------------------------------
// Provider registry entry
// ---------------------------------------------------------------------------

struct ProviderEntry {
    provider::ProviderDescriptor descriptor;
    provider::ProviderCapabilities capabilities;
    // Factory creates a live provider instance on demand.
    std::function<std::shared_ptr<provider::IExecutionProvider>()> factory;
};

// ---------------------------------------------------------------------------
// Registry (singleton)
// ---------------------------------------------------------------------------

class Registry final {
public:
    static Registry& instance();

    // Registration — call once per codec/provider before any query.
    void register_codec(CodecEntry entry);
    void register_provider(ProviderEntry entry);

    // Queries — safe to call concurrently after all registrations complete.
    [[nodiscard]] std::vector<CodecEntry> codecs() const;
    [[nodiscard]] std::vector<ProviderEntry> providers() const;

    [[nodiscard]] std::optional<CodecEntry>
    find_codec(std::string_view id) const;

    [[nodiscard]] std::optional<ProviderEntry>
    find_provider(std::string_view id) const;

    // Returns true when the codec and provider are both registered and the
    // provider reports it can support at least one of the codec's declared
    // artifact descriptors.
    [[nodiscard]] bool compatible(
        std::string_view codec_id,
        std::string_view provider_id) const;

private:
    friend class RuntimeServices;
    Registry() = default;
    mutable std::mutex mutex_;
    std::vector<CodecEntry> codecs_;
    std::vector<ProviderEntry> providers_;
};

// ---------------------------------------------------------------------------
// RuntimeServices — codec adapter -> runtime bridge
// ---------------------------------------------------------------------------
//
// Passed to ICodecAdapter::create_encoder / create_decoder so that the adapter
// can request executable model components through the registry without
// depending on TensorRT or any concrete provider type.

class RuntimeServices final {
public:
    explicit RuntimeServices(Registry& registry, std::string preferred_provider_id)
        : registry_(registry),
          preferred_provider_id_(std::move(preferred_provider_id)) {}

    // Locate a loaded executable for the given artifact specification.
    // The registry selects the preferred provider if available; falls back to
    // any compatible provider otherwise.  Returns missing_artifact when no
    // compatible local artifact is found.
    [[nodiscard]] Result<std::shared_ptr<provider::IExecutable>>
    resolve(const provider::ArtifactDescriptor& artifact) const;

    [[nodiscard]] const Registry& registry() const noexcept { return registry_; }
    [[nodiscard]] std::string_view preferred_provider() const noexcept {
        return preferred_provider_id_;
    }

private:
    Registry& registry_;
    std::string preferred_provider_id_;
};

}  // namespace nvcr::runtime
