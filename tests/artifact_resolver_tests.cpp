#include <nvcr/nvcr.hpp>
#include "support/test_provider/test_provider.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

nvcr::artifacts::ArtifactCandidate candidate(
    std::string provider_id = "test-cpu",
    std::string path = "artifact.plan") {
    nvcr::artifacts::ArtifactCandidate output;
    output.artifact.codec_id = "test-codec";
    output.artifact.model_set_id = "test-model";
    output.artifact.component_id = "identity";
    output.artifact.provider_id = std::move(provider_id);
    output.artifact.precision = "fp16";
    output.artifact.path = std::move(path);
    output.target_profile_id = "target-a";
    output.target_device = "test-device";
    output.compute_capability = "8.9";
    output.hardware_compatibility = nvcr::artifacts::HardwareCompatibility::exact;
    output.digest = {std::byte{1}, std::byte{2}, std::byte{3}};
    return output;
}

nvcr::artifacts::ArtifactRequest request() {
    nvcr::artifacts::ArtifactRequest output;
    output.codec_id = "test-codec";
    output.model_set_id = "test-model";
    output.component_id = "identity";
    output.target_profile_id = "target-a";
    output.target_device = "test-device";
    output.compute_capability = "8.9";
    output.precision = "fp16";
    output.expected_digest = {std::byte{1}, std::byte{2}, std::byte{3}};
    output.provider_preference = {"test-cpu", "tensorrt"};
    return output;
}

void resolves_best_candidate() {
    nvcr::artifacts::Resolver resolver;
    resolver.add(candidate("tensorrt", "tensorrt.plan"));
    resolver.add(candidate("test-cpu", "cpu.plan"));
    auto result = resolver.resolve(request());
    expect(result.has_value(), "resolver selects a compatible artifact");
    if (!result) return;
    expect(result.value().candidate.artifact.provider_id == "test-cpu",
           "provider preference selects test-cpu");
    expect(result.value().candidate.artifact.path == "cpu.plan",
           "resolver returns selected artifact path");
}

void resolves_without_provider_preference() {
       nvcr::artifacts::Resolver resolver;
       resolver.add(candidate("test-cpu", "z-cpu.plan"));
       resolver.add(candidate("tensorrt", "a-tensorrt.plan"));
       auto requested = request();
       requested.provider_id.clear();
       requested.provider_preference.clear();
       auto result = resolver.resolve(requested);
       expect(result.has_value(), "resolver accepts an empty provider preference");
       if (result) {
              expect(result.value().candidate.artifact.path == "a-tensorrt.plan",
                        "empty provider preference uses stable artifact ordering");
       }
}

void reports_missing_identity() {
    nvcr::artifacts::Resolver resolver;
    resolver.add(candidate());

    auto missing_codec = request();
    missing_codec.codec_id = "missing-codec";
    auto codec_result = resolver.resolve(missing_codec);
    expect(!codec_result && codec_result.error().code() == nvcr::ErrorCode::missing_codec,
           "resolver reports missing codec");

    auto missing_model = request();
    missing_model.model_set_id = "missing-model";
    auto model_result = resolver.resolve(missing_model);
    expect(!model_result && model_result.error().code() == nvcr::ErrorCode::missing_model_set,
           "resolver reports missing model set");

    auto missing_component = request();
    missing_component.component_id = "missing-component";
    auto component_result = resolver.resolve(missing_component);
    expect(!component_result && component_result.error().code() == nvcr::ErrorCode::missing_artifact,
           "resolver reports missing component artifact");

    auto missing_provider = request();
    missing_provider.provider_id = "missing-provider";
    auto provider_result = resolver.resolve(missing_provider);
    expect(!provider_result && provider_result.error().code() == nvcr::ErrorCode::missing_provider,
           "resolver reports missing provider");
}

void reports_compatibility_failures() {
    auto verify = [](nvcr::artifacts::ArtifactCandidate candidate_value,
                     nvcr::artifacts::ArtifactRequest request_value,
                     nvcr::ErrorCode expected,
                     const char* message) {
        nvcr::artifacts::Resolver resolver;
        resolver.add(std::move(candidate_value));
        auto result = resolver.resolve(request_value);
        expect(!result && result.error().code() == expected, message);
    };

    auto version_candidate = candidate();
    version_candidate.artifact.codec_api_version.major = 2;
    verify(std::move(version_candidate), request(), nvcr::ErrorCode::incompatible_version,
           "resolver reports incompatible API version");

    auto precision_candidate = candidate();
    precision_candidate.artifact.precision = "int8";
    verify(std::move(precision_candidate), request(), nvcr::ErrorCode::incompatible_precision,
           "resolver reports incompatible precision");

    auto target_candidate = candidate();
    target_candidate.target_profile_id = "other-target";
    target_candidate.target_device = "other-device";
    target_candidate.compute_capability = "9.0";
    verify(std::move(target_candidate), request(), nvcr::ErrorCode::incompatible_target,
           "resolver reports incompatible target");

    auto digest_request = request();
    digest_request.expected_digest = {std::byte{9}};
    verify(candidate(), std::move(digest_request), nvcr::ErrorCode::digest_mismatch,
           "resolver reports digest mismatch");

    auto restricted_candidate = candidate();
    restricted_candidate.license_restricted = true;
    verify(std::move(restricted_candidate), request(), nvcr::ErrorCode::license_restricted,
           "resolver reports license restriction");

    auto unavailable_candidate = candidate();
    unavailable_candidate.available = false;
    verify(std::move(unavailable_candidate), request(), nvcr::ErrorCode::missing_artifact,
           "resolver reports unavailable artifact");
}

void supports_compatibility_classes() {
    auto compatibility_candidate = candidate();
    compatibility_candidate.target_profile_id = "other-target";
    compatibility_candidate.hardware_compatibility =
        nvcr::artifacts::HardwareCompatibility::same_compute_capability;
    nvcr::artifacts::Resolver resolver;
    resolver.add(std::move(compatibility_candidate));
    auto result = resolver.resolve(request());
    expect(result.has_value(), "same-compute-capability artifact resolves");
    if (result) {
        expect(result.value().target_match_rank == 1U,
               "same-compute-capability match has the expected rank");
    }

       auto ampere_candidate = candidate();
       ampere_candidate.target_profile_id = "other-target";
       ampere_candidate.hardware_compatibility =
              nvcr::artifacts::HardwareCompatibility::ampere_plus;
       nvcr::artifacts::Resolver ampere_resolver;
       ampere_resolver.add(std::move(ampere_candidate));
       auto ampere_result = ampere_resolver.resolve(request());
       expect(ampere_result.has_value(), "ampere-plus artifact resolves for Ampere targets");
       if (ampere_result) {
              expect(ampere_result.value().target_match_rank == 2U,
                        "ampere-plus match has the expected rank");
       }
}

void allows_explicit_license_override() {
       auto restricted = candidate();
       restricted.license_restricted = true;
       auto requested = request();
       requested.allow_license_restricted = true;
       nvcr::artifacts::Resolver resolver;
       resolver.add(std::move(restricted));
       expect(resolver.resolve(requested).has_value(),
                 "explicit license override resolves a restricted artifact");
}

void reports_catalog_runtime_mismatches() {
       auto architecture_candidate = candidate();
       architecture_candidate.architecture = "x86_64";
       auto architecture_request = request();
       architecture_request.architecture = "aarch64";
       nvcr::artifacts::Resolver architecture_resolver;
       architecture_resolver.add(std::move(architecture_candidate));
       auto architecture_result = architecture_resolver.resolve(architecture_request);
       expect(!architecture_result &&
                        architecture_result.error().code() == nvcr::ErrorCode::incompatible_target,
                 "resolver rejects incompatible catalog architecture");

       auto runtime_candidate = candidate();
       runtime_candidate.runtime_version = "10.9.0";
       runtime_candidate.architecture = "x86_64";
       runtime_candidate.cuda_runtime_version = 12060U;
       auto runtime_request = request();
       runtime_request.runtime_version = "10.8.0";
       runtime_request.architecture = "x86_64";
       runtime_request.cuda_runtime_version = 12080U;
       nvcr::artifacts::Resolver runtime_resolver;
       runtime_resolver.add(std::move(runtime_candidate));
       auto runtime_result = runtime_resolver.resolve(runtime_request);
       expect(!runtime_result &&
                        runtime_result.error().code() == nvcr::ErrorCode::incompatible_version,
                 "resolver rejects incompatible provider runtime version");

       auto compatible_candidate = candidate();
       compatible_candidate.architecture = "x86_64";
       compatible_candidate.cuda_runtime_version = 12060U;
       auto compatible_request = request();
       compatible_request.architecture = "x86_64";
       compatible_request.cuda_runtime_version = 12080U;
       nvcr::artifacts::Resolver compatible_resolver;
       compatible_resolver.add(std::move(compatible_candidate));
       expect(compatible_resolver.resolve(compatible_request).has_value(),
                 "desktop catalog accepts an older CUDA runtime in the same major family");
}

void resolves_through_runtime_services() {
       nvcr::test_support::register_test_provider();
       nvcr::artifacts::Resolver resolver;
       auto artifact = candidate("test-cpu", "runtime-artifact");
       artifact.artifact.precision = "int8";
       resolver.add(artifact);

       auto requested = request();
       requested.provider_id = "test-cpu";
       requested.precision = "int8";
       nvcr::runtime::RuntimeServices services(
              nvcr::runtime::Registry::instance(), "test-cpu");
       auto executable = services.resolve(resolver, requested);
       expect(executable.has_value(), "runtime services loads resolver-selected artifact");
       if (executable) {
              expect(executable.value()->descriptor().path == "runtime-artifact",
                        "runtime services preserves selected artifact identity");
       }
}

void honors_resolver_selected_provider() {
       nvcr::test_support::register_test_provider();
       nvcr::artifacts::Resolver resolver;
       auto artifact = candidate("unregistered-provider", "runtime-artifact");
       artifact.artifact.precision = "int8";
       resolver.add(artifact);

       auto requested = request();
       requested.provider_id = "unregistered-provider";
       requested.precision = "int8";
       nvcr::runtime::RuntimeServices services(
              nvcr::runtime::Registry::instance(), "test-cpu");
       auto executable = services.resolve(resolver, requested);
       expect(!executable && executable.error().code() == nvcr::ErrorCode::missing_provider,
              "runtime services does not fall back after resolver selects a provider");
}

}  // namespace

int main() {
    resolves_best_candidate();
        resolves_without_provider_preference();
    reports_missing_identity();
    reports_compatibility_failures();
    supports_compatibility_classes();
               allows_explicit_license_override();
       reports_catalog_runtime_mismatches();
       resolves_through_runtime_services();
               honors_resolver_selected_provider();
    if (failures == 0) {
        std::cout << "NVCR artifact resolver tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
