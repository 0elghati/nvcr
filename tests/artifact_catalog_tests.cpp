#include <nvcr/nvcr.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr std::string_view valid_catalog = R"json({
  "schema": "nvcr.engine-catalog.v1",
  "assets": [
    {
      "backend": "dcvcrt",
      "model_profile_id": "dcvcrt-cvpr2025",
      "target_profile_id": "target-a",
      "profile": "720p",
      "precision": "fp16",
      "operating_system": "linux",
      "architecture": "x86_64",
      "device_name": "Test Device",
      "filename": "nvcr-engines-target-a-dcvcrt-cvpr2025-720p.tar.gz",
      "size_bytes": 123,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000001",
      "compute_capability_major": 8,
      "compute_capability_minor": 9,
      "multiprocessor_count": 46,
      "cuda_runtime_version": 12060,
      "tensorrt_version_major": 10,
      "tensorrt_version_minor": 9,
      "tensorrt_version_patch": 0,
      "hardware_compatibility": "exact"
    }
  ]
})json";

void accepts_valid_catalog() {
    auto catalog = nvcr::artifacts::Catalog::from_json(valid_catalog);
    expect(catalog.has_value(), "valid catalog parses");
    if (!catalog) return;
    expect(catalog.value().schema() == "nvcr.engine-catalog.v1", "catalog schema is preserved");
    expect(catalog.value().entries().size() == 1U, "catalog has one typed entry");
    if (catalog.value().entries().empty()) return;
    const auto& entry = catalog.value().entries().front();
    expect(entry.codec_id == "dcvcrt", "codec identity is ingested");
    expect(entry.engine_profile_id == "720p", "engine profile identity is ingested");
    expect(entry.tensorrt_version_minor == 9U, "TensorRT version is ingested");

    nvcr::artifacts::Resolver resolver;
    nvcr::artifacts::CatalogLoadOptions options;
    options.codec_id = "dcvc-rt";
    options.provider_id = "tensorrt";
    auto appended = catalog.value().append_candidates(resolver, options);
    expect(appended.has_value(), "catalog converts to resolver candidates");
    expect(resolver.candidates().size() == 1U, "resolver receives one catalog candidate");
    if (resolver.candidates().empty()) return;
    const auto& candidate = resolver.candidates().front();
    expect(candidate.artifact.codec_id == "dcvc-rt", "caller controls codec identity mapping");
        expect(candidate.artifact.component_id == "engine-bundle",
            "catalog assigns the explicit bundle component id");
        expect(candidate.artifact.engine_profile_id == "720p",
            "catalog preserves the engine profile identity");
    expect(candidate.artifact.provider_id == "tensorrt", "provider identity is preserved");
    expect(candidate.license_restricted, "external catalog assets remain license-restricted by default");
    expect(!candidate.available, "catalog metadata does not claim a local artifact is available");

    const auto root = std::filesystem::temp_directory_path() / "nvcr-catalog-loader-assets";
    std::filesystem::create_directories(root);
    {
        std::ofstream archive(root / entry.filename, std::ios::binary);
        archive << "placeholder; byte validation remains a separate gate";
    }
    nvcr::artifacts::Resolver local_resolver;
    options.artifact_root = root;
    options.mark_license_restricted = false;
    expect(catalog.value().append_candidates(local_resolver, options).has_value(),
           "catalog converts local asset metadata to resolver candidates");
    if (local_resolver.candidates().empty()) {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        return;
    }
    auto local_request = nvcr::artifacts::ArtifactRequest{};
    local_request.codec_id = "dcvc-rt";
    local_request.model_set_id = "dcvcrt-cvpr2025";
    local_request.component_id = "engine-bundle";
    local_request.engine_profile_id = "720p";
    local_request.provider_id = "tensorrt";
    local_request.target_profile_id = "target-a";
    local_request.target_device = "Test Device";
    local_request.compute_capability = "8.9";
    local_request.operating_system = "linux";
    local_request.architecture = "x86_64";
    local_request.runtime_version = "10.9.0";
    local_request.cuda_runtime_version = 12060U;
    local_request.precision = "fp16";
    local_request.expected_digest = local_resolver.candidates().front().digest;
    auto resolved = local_resolver.resolve(local_request);
    expect(resolved.has_value(), "ingested local catalog asset resolves");
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);

    std::string canonical_catalog(valid_catalog);
    const auto replace_key = [](std::string& document,
                                std::string_view legacy,
                                std::string_view canonical) {
        const auto position = document.find(legacy);
        if (position != std::string::npos) {
            document.replace(position, legacy.size(), canonical);
        }
    };
    replace_key(canonical_catalog, "\"backend\"", "\"codec_id\"");
    replace_key(canonical_catalog, "\"profile\"", "\"engine_profile_id\"");
    auto canonical = nvcr::artifacts::Catalog::from_json(canonical_catalog);
    expect(canonical.has_value(), "canonical catalog identity keys parse");
    if (canonical && !canonical.value().entries().empty()) {
        expect(canonical.value().entries().front().codec_id == "dcvcrt",
               "canonical codec identity is ingested");
        expect(canonical.value().entries().front().engine_profile_id == "720p",
               "canonical engine profile identity is ingested");
    }
}

void rejects_invalid_catalogs() {
    auto wrong_schema = nvcr::artifacts::Catalog::from_json(
        R"json({"schema":"other.v1","assets":[]})json");
    expect(!wrong_schema, "wrong catalog schema is rejected");

    auto malformed = nvcr::artifacts::Catalog::from_json(
        R"json({"schema":"nvcr.engine-catalog.v1","assets":[)json");
    expect(!malformed, "malformed JSON is rejected");

    auto leading_zero = nvcr::artifacts::Catalog::from_json(
        R"json({"schema":"nvcr.engine-catalog.v1","assets":[],"extra":01})json");
    expect(!leading_zero, "invalid JSON number syntax is rejected");

    std::string bad_filename(valid_catalog);
    const auto filename = bad_filename.find("nvcr-engines-target-a");
    bad_filename.replace(filename, 26U, "not-an-engine-archive");
    auto filename_result = nvcr::artifacts::Catalog::from_json(bad_filename);
    expect(!filename_result, "identity-inconsistent filename is rejected");

    std::string duplicate(valid_catalog);
    const auto asset_end = duplicate.rfind("  ]\n}");
    duplicate.insert(asset_end, ",\n    " + duplicate.substr(duplicate.find("    {"), duplicate.find("    }\n") - duplicate.find("    {") + 7U));
    auto duplicate_result = nvcr::artifacts::Catalog::from_json(duplicate);
    expect(!duplicate_result, "duplicate catalog identity is rejected");
}

void loads_catalog_file() {
    const auto path = std::filesystem::temp_directory_path() / "nvcr-catalog-loader-test.json";
    {
        std::ofstream output(path);
        output << valid_catalog;
    }
    auto catalog = nvcr::artifacts::Catalog::from_file(path);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    expect(catalog.has_value(), "catalog file loads from disk");
}

}  // namespace

int main() {
    accepts_valid_catalog();
    rejects_invalid_catalogs();
    loads_catalog_file();
    if (failures == 0) {
        std::cout << "NVCR artifact catalog tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
