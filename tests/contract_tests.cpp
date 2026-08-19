#include "support/test_codec/test_codec.hpp"
#include "support/test_provider/test_provider.hpp"

#if defined(NVCR_TEST_HAS_DCVCRT)
#include <nvcr/dcvcrt/backend.hpp>
#endif
#if defined(NVCR_TEST_HAS_TENSORRT) && defined(NVCR_TEST_HAS_DCVCRT)
#include <nvcr/dcvcrt/tensorrt_backend.hpp>
#endif
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void registry_contracts() {
    nvcr::runtime::register_builtin_components();
    nvcr::test_support::register_test_codec();
    nvcr::test_support::register_test_provider();

    const auto& registry = nvcr::runtime::Registry::instance();
    const auto codecs = registry.codecs();
    const auto providers = registry.providers();

    expect(!codecs.empty(), "codec registry is non-empty");
    expect(!providers.empty(), "provider registry is non-empty");
#if defined(NVCR_TEST_HAS_DCVCRT)
    expect(registry.find_codec("dcvc-rt").has_value(), "dcvc-rt is registered");
#endif
    expect(registry.find_codec("test-codec").has_value(), "test codec is registered");
    auto registered_adapter = registry.create_codec("test-codec");
    expect(registered_adapter.has_value(), "registered test codec creates an adapter");
    if (registered_adapter) {
        expect(registered_adapter.value()->descriptor().id == "test-codec",
               "registered adapter matches registry descriptor");
    }
#if defined(NVCR_TEST_HAS_TENSORRT)
    expect(registry.find_provider("tensorrt").has_value(), "tensorrt is registered");
#endif
    expect(registry.find_provider("test-cpu").has_value(), "test provider is registered");
    expect(registry.compatible("test-codec", "test-cpu"), "test codec/provider compatibility");
    expect(!registry.compatible("missing-codec", "test-cpu"), "missing codec is incompatible");

    for (const auto& codec_entry : codecs) {
        expect(!codec_entry.descriptor.id.empty(), "registered codec has an id");
        expect(codec_entry.descriptor.api_version > 0U, "registered codec has an API version");
        expect(!codec_entry.encoder_options.declarations.empty(),
               "registered codec exposes encoder options");
        if (codec_entry.factory) {
            auto adapter = codec_entry.factory();
            expect(adapter != nullptr, "registered codec factory creates an adapter");
            if (adapter) {
                expect(adapter->descriptor().id == codec_entry.descriptor.id,
                       "registered adapter id matches registry id");
            }
        }
    }
    for (const auto& provider_entry : providers) {
        expect(!provider_entry.descriptor.id.empty(), "registered provider has an id");
        expect(static_cast<bool>(provider_entry.factory), "registered provider has a factory");
        auto provider = provider_entry.factory();
        expect(provider != nullptr, "registered provider factory creates a provider");
        if (!provider) continue;
        for (const auto& codec_entry : codecs) {
            nvcr::provider::ArtifactDescriptor artifact;
            artifact.codec_id = codec_entry.descriptor.id;
            artifact.model_set_id = "contract-model";
            artifact.component_id = "contract-component";
            artifact.provider_id = provider_entry.descriptor.id;
            artifact.precision = provider_entry.capabilities.supports_fp16 ? "fp16" : "int8";
            artifact.path = "contract-artifact";
            expect(provider->supports(artifact), "provider accepts its registered artifact identity");
        }
    }
}

void runtime_services_contract() {
    nvcr::runtime::RuntimeServices services(
        nvcr::runtime::Registry::instance(),
        "test-cpu");

    nvcr::provider::ArtifactDescriptor descriptor;
    descriptor.codec_id = "test-codec";
    descriptor.model_set_id = "test-model";
    descriptor.component_id = "identity";
    descriptor.provider_id = "test-cpu";
    descriptor.precision = "int8";
    descriptor.path = "in-memory";
        expect(nvcr::current_software_version.major == 0U, "software version is explicit");
        expect(descriptor.codec_api_version.major == 1U, "artifact codec API version is explicit");
        expect(descriptor.provider_api_version.major == 1U, "artifact provider API version is explicit");
        expect(descriptor.manifest_schema_version.major == 2U,
            "artifact manifest schema version is explicit");

    auto executable = services.resolve(descriptor);
    expect(executable.has_value(), "runtime services resolves test provider executable");
    if (!executable) return;

    std::vector<std::byte> in{std::byte{1}, std::byte{2}, std::byte{3}};
    std::vector<std::byte> out(3);
    nvcr::provider::TensorBindings inputs{{"x", in, {3}, "int8"}};
    nvcr::provider::TensorBindings outputs{{"y", out, {3}, "int8"}};
    auto ran = executable.value()->execute(inputs, outputs, {});
    expect(ran.has_value(), "test provider executable runs");
    expect(out == in, "test provider executable copies tensor bytes deterministically");

    descriptor.provider_id = "no-such-provider";
    auto missing = services.resolve(descriptor);
    expect(!missing.has_value(), "runtime services reports missing provider artifact");
    if (!missing) {
         expect(missing.error().code() == nvcr::ErrorCode::missing_provider,
             "missing provider maps to missing_provider error");
    }

    descriptor.component_id.clear();
    descriptor.provider_id = "test-cpu";
    auto invalid_load = services.resolve(descriptor);
    expect(!invalid_load.has_value(), "test provider rejects incomplete artifact identity");
    if (!invalid_load) {
        expect(invalid_load.error().code() == nvcr::ErrorCode::invalid_argument,
               "incomplete artifact identity maps to invalid_argument");
    }

    descriptor.component_id = "identity";
    auto executable_again = services.resolve(descriptor);
    expect(executable_again.has_value(), "test provider resolves a valid artifact again");
    if (executable_again) {
        std::vector<std::byte> short_output(2);
        nvcr::provider::TensorBindings bad_outputs{{"y", short_output, {2}, "int8"}};
        auto rejected = executable_again.value()->execute(inputs, bad_outputs, {});
        expect(!rejected.has_value(), "test provider rejects mismatched tensor bindings");
        if (!rejected) {
            expect(rejected.error().code() == nvcr::ErrorCode::invalid_argument,
                   "mismatched tensor bindings map to invalid_argument");
        }
    }

    nvcr::RuntimeConfiguration configuration;
    configuration.provider_id = "test-cpu";
    auto components = services.create_components(configuration);
    expect(components.has_value(), "runtime services creates provider-owned codec components");
    if (components) {
        expect(components.value().codec != nullptr, "provider-owned components include a codec backend");
    }

    configuration.provider_id = "no-such-provider";
    auto missing_components = services.create_components(configuration);
    expect(!missing_components.has_value(), "runtime services rejects missing component provider");
    if (!missing_components) {
        expect(missing_components.error().code() == nvcr::ErrorCode::missing_provider,
               "missing component provider maps to missing_provider");
    }
}

nvcr::Result<nvcr::Frame> make_frame(std::byte seed) {
    std::vector<std::byte> data(12U);
    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<std::byte>(
            std::to_integer<unsigned char>(seed) + static_cast<unsigned char>(index));
    }
    return nvcr::Frame::copy_from(
        4U, 2U, nvcr::PixelFormat::yuv420p8, data, nvcr::Timestamp{7});
}

void codec_adapter_contract() {
    auto adapter = nvcr::test_support::make_test_codec_adapter();
    expect(adapter != nullptr, "test codec adapter is constructible");
    if (!adapter) return;

    expect(adapter->descriptor().id == "test-codec", "test adapter descriptor id");
    expect(adapter->capabilities().supports_delayed_output,
           "test adapter advertises delayed output");
    expect(adapter->encoder_options().declarations.size() == 1U,
           "test adapter exposes encoder option schema");
    expect(adapter->decoder_options().declarations.size() == 1U,
           "test adapter exposes decoder option schema");

    auto frame = make_frame(std::byte{0x20});
    expect(frame.has_value(), "test contract frame is constructible");
    if (!frame) return;

    nvcr::runtime::RuntimeServices services(
        nvcr::runtime::Registry::instance(),
        "test-cpu");
    auto components = adapter->create_components({}, services);
    expect(components.has_value(), "test adapter creates codec components");
    if (!components) return;
    expect(components.value().codec != nullptr, "test adapter returns a codec backend");
    if (!components.value().codec) return;
    expect(components.value().codec->initialize({}).has_value(), "test backend initializes");
    auto encoded = components.value().codec->encode(
        frame.value(), nvcr::FrameType::intra, {});
    expect(encoded.has_value(), "test backend encodes a frame");
    if (!encoded) return;
    auto decoded = components.value().codec->decode(
        encoded.value().payload, nvcr::FrameType::intra, frame.value().timestamp(), {});
    expect(decoded.has_value(), "test backend decodes its private payload");
    if (decoded) {
        expect(decoded.value().frame.data().size() == frame.value().data().size(),
               "test backend preserves frame size");
        expect(std::equal(
                   decoded.value().frame.data().begin(), decoded.value().frame.data().end(),
                   frame.value().data().begin()),
               "test backend preserves frame bytes");
    }

    nvcr::RuntimeConfiguration runtime_configuration;
    auto defaults = adapter->apply_defaults(runtime_configuration);
    expect(defaults.has_value(), "test adapter applies runtime defaults");
    expect(runtime_configuration.codec_id == "test-codec", "adapter selects its codec id");
    expect(runtime_configuration.provider_id == "test-cpu", "adapter selects its provider");
    auto runtime_components = adapter->create_components(runtime_configuration, services);
    expect(runtime_components.has_value(), "test adapter creates runtime components");
    if (!defaults || !runtime_components) return;
    auto runtime = nvcr::Runtime::create(
        std::move(runtime_configuration), std::move(runtime_components.value()));
    expect(runtime.has_value(), "generic runtime creates a selected test codec");
    if (runtime) {
        expect(runtime.value().send_frame(frame.value()).has_value(),
               "generic runtime accepts a test-codec frame");
        auto packet = runtime.value().receive_access_unit();
        expect(packet.has_value(), "generic runtime emits a test-codec access unit");
    }
}

void session_lifecycle_contract() {
    auto encoder = nvcr::test_support::make_test_encoder_session(1U);
    auto decoder = nvcr::test_support::make_test_decoder_session();
    auto first = make_frame(std::byte{0x10});
    auto second = make_frame(std::byte{0x40});
    auto third = make_frame(std::byte{0x70});
    expect(first && second && third, "session contract frames are constructible");
    if (!first || !second || !third) return;

    expect(encoder->send_frame(first.value()).has_value(), "encoder accepts first frame");
    auto delayed = encoder->receive_access_unit();
    expect(!delayed && delayed.error().code() == nvcr::ErrorCode::try_again,
           "encoder delays first output");

    expect(encoder->send_frame(second.value()).has_value(), "encoder accepts second frame");
    auto first_packet = encoder->receive_access_unit();
    expect(first_packet.has_value(), "encoder emits delayed first access unit");
    expect(encoder->send_frame(third.value()).has_value(), "encoder accepts third frame");
    auto second_packet = encoder->receive_access_unit();
    expect(second_packet.has_value(), "encoder emits queued second access unit");
    expect(encoder->flush().has_value(), "encoder flushes delayed state");
    auto third_packet = encoder->receive_access_unit();
    expect(third_packet.has_value(), "encoder emits final access unit during flush");
    auto end = encoder->receive_access_unit();
    expect(!end && end.error().code() == nvcr::ErrorCode::end_of_stream,
           "encoder reports end of stream after drain");

    if (first_packet && second_packet && third_packet) {
        for (const auto* packet : {&first_packet.value(), &second_packet.value(), &third_packet.value()}) {
            auto access_unit = nvcr::AccessUnitIO::deserialize(packet->data());
            expect(access_unit.has_value(), "session output uses the common access-unit envelope");
            if (access_unit) {
                expect(access_unit.value().model_id == "test-codec",
                       "session output preserves codec-private model identity");
            }
            expect(decoder->send_access_unit(*packet).has_value(),
                   "decoder accepts test codec access unit");
        }
    }
    for (unsigned index = 0; index < 3U; ++index) {
        auto frame = decoder->receive_frame();
        expect(frame.has_value(), "decoder emits a lossless reconstructed frame");
    }
    expect(decoder->flush().has_value(), "decoder flushes state");
    auto decoder_end = decoder->receive_frame();
    expect(!decoder_end && decoder_end.error().code() == nvcr::ErrorCode::end_of_stream,
           "decoder reports end of stream after drain");

    expect(encoder->reset().has_value(), "encoder resets state");
    expect(decoder->reset().has_value(), "decoder resets state");
    expect(encoder->send_frame(first.value()).has_value(), "encoder accepts frame after reset");
}

}  // namespace

int main() {
    registry_contracts();
    runtime_services_contract();
    codec_adapter_contract();
    session_lifecycle_contract();
    if (failures == 0) {
        std::cout << "NVCR codec/provider contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
