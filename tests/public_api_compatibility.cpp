#include <nvcr/nvcr.hpp>

#include <memory>
#include <type_traits>

namespace {

class LegacyBackend final : public nvcr::codec::CodecBackend {
public:
    nvcr::Result<void> initialize(
        const nvcr::RuntimeConfiguration& configuration) override {
        if (configuration.gop_size != 8 || configuration.codec.gop_size != 8 ||
            configuration.log_level != nvcr::LogLevel::off ||
            configuration.runtime.log_level != nvcr::LogLevel::off) {
            return nvcr::Error(
                nvcr::ErrorCode::invalid_argument,
                "legacy configuration fields were not reconciled",
                "compatibility-test");
        }
        return {};
    }

    nvcr::Result<nvcr::codec::CodecEncodeResult> encode(
        const nvcr::Frame&, nvcr::FrameType,
        const nvcr::codec::SequenceStateView&) override {
        return nvcr::Error(
            nvcr::ErrorCode::not_implemented, "not exercised", "compatibility-test");
    }

    nvcr::Result<nvcr::codec::CodecDecodeResult> decode(
        std::span<const std::byte>, nvcr::FrameType, nvcr::Timestamp,
        const nvcr::codec::SequenceStateView&) override {
        return nvcr::Error(
            nvcr::ErrorCode::not_implemented, "not exercised", "compatibility-test");
    }

    nvcr::Result<void> flush() override { return {}; }
    void reset() noexcept override {}
};

class LegacyAdapter final : public nvcr::codec::ICodecAdapter {
public:
    nvcr::codec::CodecDescriptor descriptor() const override {
        return {"legacy-codec", "Legacy codec", 1};
    }

    nvcr::codec::CodecCapabilities capabilities() const override { return {}; }
    nvcr::codec::OptionSchema encoder_options() const override { return {}; }
    nvcr::codec::OptionSchema decoder_options() const override { return {}; }

    nvcr::Result<nvcr::codec::Components> create_components(
        const nvcr::RuntimeConfiguration&,
        const nvcr::runtime::RuntimeServices&) override {
        return nvcr::codec::Components{std::make_unique<LegacyBackend>()};
    }
};

}  // namespace

int main() {
    static_assert(std::is_aggregate_v<nvcr::RuntimeConfiguration>);

    nvcr::RuntimeConfiguration configuration;
    configuration.gop_size = 8;
    configuration.log_level = nvcr::LogLevel::off;

    auto runtime = nvcr::Runtime::create(
        configuration,
        nvcr::codec::Components{std::make_unique<LegacyBackend>()});
    if (!runtime) return 1;

    nvcr::runtime::ProviderEntry legacy_provider{
        {nvcr::provider::ProviderKind::cpu_reference, "legacy", "1", "Legacy"},
        {},
        {},
        [](const nvcr::RuntimeConfiguration&) -> nvcr::Result<nvcr::codec::Components> {
            return nvcr::codec::Components{std::make_unique<LegacyBackend>()};
        },
    };
    static_cast<void>(legacy_provider);

    LegacyAdapter adapter;
    nvcr::runtime::RuntimeServices services(
        nvcr::runtime::Registry::instance(), "legacy");
    auto components = adapter.create_components(configuration, services);
    return components ? 0 : 1;
}
