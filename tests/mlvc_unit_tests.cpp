// Model-free MLVC tests: record syntax, frame scheduling, reference pruning,
// Q shifts, canvas placement, scale extraction, host conversions, the JSON
// reader, and bundle validation on a synthetic bundle.

#include "bundle.hpp"
#include "host_ops.hpp"
#include "record.hpp"
#include "sequence.hpp"

#include "../common/json.hpp"
#include "../common/sha256.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace nvcr::mlvc;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<std::byte> bytes(std::initializer_list<unsigned> values) {
    std::vector<std::byte> output;
    for (const auto value : values) output.push_back(static_cast<std::byte>(value));
    return output;
}

void record_tests() {
    const auto payload = bytes({1, 2, 3});
    auto record = serialize_record(21, payload);
    check(record && record.value() == bytes({21, 0, 0, 0, 3, 0, 0, 0, 1, 2, 3}),
          "record is little-endian q, length, payload");
    auto parsed = parse_record(record.value());
    check(parsed && parsed.value().q_index == 21 && parsed.value().payload.size() == 3U,
          "record round trips");
    auto dropped = serialize_record(dropped_frame_q_index, {});
    check(dropped && parse_record(dropped.value()) &&
              parse_record(dropped.value()).value().q_index == -1,
          "dropped record round trips");
    check(!serialize_record(dropped_frame_q_index, payload), "dropped record with payload rejected");
    check(!parse_record(bytes({21, 0, 0, 0, 3, 0, 0})), "truncated header rejected");
    check(!parse_record(bytes({21, 0, 0, 0, 4, 0, 0, 0, 1, 2, 3})), "short payload rejected");
    check(!parse_record(bytes({21, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3})), "trailing bytes rejected");
    check(!parse_record(bytes({0xfe, 0xff, 0xff, 0xff, 0, 0, 0, 0})), "q below -1 rejected");
    check(!parse_record(bytes({0xff, 0xff, 0xff, 0xff, 1, 0, 0, 0, 9})),
          "dropped record with payload rejected on parse");
}

ModelParameters mlvc_s() {
    ModelParameters model;
    model.width = 320;
    model.height = 192;
    model.qp_num = 64;
    model.total_qp_num = 72;
    model.frame_index_map = {0, 1, 0, 2, 0, 2, 0, 2};
    model.qp_shift = {0, 8, 4};
    model.feature_channels = 96;
    model.latent_channels = 48;
    model.hyperprior_channels = 48;
    model.downsample_feature = 8;
    model.downsample_latent = 16;
    model.downsample_hyperprior = 4;
    model.y_scale_repeat = 4;
    model.iframe_period = 1024;
    model.ltr_start_idx = 8;
    model.ltr_period = 64;
    return model;
}

void scheduler_tests() {
    FrameScheduler scheduler(LoopParameters::from_model(mlvc_s()));
    std::vector<FramePlan> plans;
    for (int index = 0; index < 130; ++index) {
        plans.push_back(scheduler.plan());
        scheduler.commit(plans.back());
    }
    check(plans[0].frame_class == FrameClass::intra && !plans[0].reference, "frame 0 is intra");
    check(plans[1].frame_class == FrameClass::predicted && plans[1].reference == 0U,
          "P frames reference the previous coded frame");
    check(plans[8].mark_as_ltr && plans[8].frame_class == FrameClass::predicted &&
              plans[8].reference == 7U,
          "first LTR mark is an ordinary P frame");
    check(plans[64].frame_class == FrameClass::ltr_recovery && plans[64].reference == 8U &&
              plans[64].mark_as_ltr,
          "frame 64 recovers from LTR 8 and is marked");
    check(plans[65].reference == 64U, "frame after recovery references it");
    check(plans[128].frame_class == FrameClass::ltr_recovery && plans[128].reference == 64U,
          "frame 128 recovers from LTR 64");
    check(plans[8].droppable() == false && plans[9].droppable(), "only plain P frames drop");

    LoopParameters short_period = LoopParameters::from_model(mlvc_s());
    short_period.iframe_period = 16;
    short_period.reset_period = 32;
    FrameScheduler periodic(short_period);
    std::vector<FramePlan> periodic_plans;
    for (int index = 0; index < 20; ++index) {
        periodic_plans.push_back(periodic.plan());
        periodic.commit(periodic_plans.back());
    }
    check(periodic_plans[16].frame_class == FrameClass::intra &&
              periodic_plans[16].coded_index == 0U,
          "I period restarts the coded index");
    check(periodic_plans[17].reference == 0U, "P after the second I references it");
    check(!periodic_plans[16].mark_as_ltr && periodic_plans[8].mark_as_ltr,
          "LTR marks restart after an I frame");

    LoopParameters with_reset = LoopParameters::from_model(mlvc_s());
    with_reset.reset_period = 4;
    FrameScheduler resetting(with_reset);
    std::vector<FramePlan> reset_plans;
    for (int index = 0; index < 8; ++index) {
        reset_plans.push_back(resetting.plan());
        resetting.commit(reset_plans.back());
    }
    check(reset_plans[3].feature_reset && reset_plans[7].feature_reset &&
              !reset_plans[4].feature_reset,
          "feature reset when (index + 1) % reset_period == 0");

    FrameScheduler decoder(LoopParameters::from_model(mlvc_s()));
    decoder.commit(decoder.plan(true));
    decoder.commit(decoder.plan(false));
    const auto forced = decoder.plan(true);
    check(forced.frame_class == FrameClass::intra && forced.coded_index == 0U,
          "a stream intra flag restarts the sequence");
}

void reference_tests() {
    ReferenceStore<int> store;
    store.save(0, 0, false);
    store.save(1, 1, false);
    store.save(2, 2, false);
    check(store.size() == 1U && store.find(2) && !store.find(1), "one short-term reference kept");
    for (const std::uint64_t index : {8U, 64U, 128U, 192U}) store.save(index, 0, true);
    check(store.find(64) && store.find(128) && store.find(192) && !store.find(8),
          "three most recent long-term references kept");
    check(store.find(2), "long-term pruning keeps the short-term reference");
    store.save(193, 0, false);
    check(!store.find(2) && store.find(193) && store.size() == 4U, "capacity is 1 + 3");
}

void q_shift_tests() {
    const auto model = mlvc_s();
    const std::array<std::uint32_t, 9> expected{8, 0, 4, 0, 4, 0, 4, 0, 8};
    for (std::uint64_t index = 0; index < expected.size(); ++index) {
        check(shifted_q_index(model, 21, index) == 21U + expected[index],
              "Q shift follows qp_shift[frame_index_map[(index + 1) % 8]]");
    }
}

void placement_tests() {
    const auto model = mlvc_s();
    auto native = host_ops::place(176, 144, model);
    check(native && !native.value().transposed, "QCIF fits natively");
    ModelParameters wide = model;
    wide.width = 640;
    wide.height = 368;
    auto rotated = host_ops::place(360, 640, wide);
    check(rotated && rotated.value().transposed, "portrait frame fits transposed");
    check(!host_ops::place(700, 400, wide), "oversized frame rejected");
    check(!host_ops::place(175, 144, model), "odd width rejected");

    for (const auto& [width, height, canvas] : {std::tuple{6U, 4U, std::pair{16U, 16U}},
                                                 std::tuple{4U, 6U, std::pair{6U, 4U}}}) {
        ModelParameters small = model;
        small.width = canvas.first;
        small.height = canvas.second;
        auto placement = host_ops::place(width, height, small);
        check(placement.has_value(), "small frame placed");
        std::vector<std::byte> frame(width * height * 3U / 2U);
        for (std::size_t index = 0; index < frame.size(); ++index) {
            frame[index] = static_cast<std::byte>((index * 37U + 11U) % 256U);
        }
        for (const auto precision : {Precision::fp16, Precision::fp32}) {
            const std::size_t element = precision == Precision::fp16 ? 2U : 4U;
            std::vector<std::byte> tensor(3U * small.width * small.height * element);
            host_ops::prepare_input(frame, placement.value(), precision, tensor);
            // Upsampled chroma has four equal samples, so the 2x2 mean and the
            // 8-bit rounding return the source exactly.
            std::vector<std::byte> output(frame.size());
            host_ops::finish_output(tensor, placement.value(), precision, output);
            check(output == frame, "input preparation and output finishing invert");
        }
    }
}

void conversion_tests() {
    check(host_ops::to_half(1.0F) == 0x3c00U, "1.0 converts exactly");
    check(host_ops::to_half(65520.0F) == 0x7c00U, "overflow rounds to infinity");
    check(host_ops::to_half(1.0F + 0x1p-11F) == 0x3c00U, "ties round to even");
    check(host_ops::to_half(1.0F + 3.0F * 0x1p-11F) == 0x3c02U, "ties round up to even");
    check(host_ops::to_float(0x0001U) == 0x1p-24F, "smallest subnormal");
}

void scale_tests() {
    ModelParameters model;
    model.width = 32;
    model.height = 32;
    model.latent_channels = 4;
    model.hyperprior_channels = 2;
    model.downsample_latent = 16;
    model.downsample_hyperprior = 2;
    model.y_scale_repeat = 2;
    const std::vector<float> z{-3.0F, 200.0F};  // [2, 1, 1]
    const auto scales = host_ops::extract_scales(z, model, 128);
    // s[c] = min(|z[c / 2]|, 127): channels 0-1 -> 3, channels 2-3 -> 127.
    // scales_0 takes channel c' where (h + w) is even, c' + 2 elsewhere.
    const std::vector<std::int32_t> expected_0{3, 127, 127, 3, 3, 127, 127, 3};
    const std::vector<std::int32_t> expected_1{127, 3, 3, 127, 127, 3, 3, 127};
    check(scales.scales_0 == expected_0 && scales.scales_1 == expected_1,
          "scale indices follow the checkerboard packing");
}

void json_tests() {
    auto value = nvcr::detail::json::parse(
        R"({"a": [1, -2, 0.5, 1e3, null, true], "b": {"c": "xA"}})", "test");
    check(value.has_value(), "valid JSON parses");
    if (value) {
        const auto& array = value.value().find("a")->array_value;
        check(array[0].integer_value == 1 && array[1].integer_value == -2 &&
                  !array[2].integer_value && array[2].number_value == 0.5 &&
                  !array[3].integer_value && array[3].number_value == 1000.0 &&
                  array[4].is_null() && array[5].boolean_value,
              "numbers keep integer identity");
        check(value.value().find("b")->find("c")->string_value == "xA", "escapes decode");
    }
    check(!nvcr::detail::json::parse("{} x", "test"), "trailing data rejected");
    check(!nvcr::detail::json::parse(R"({"a": 1, "a": 2})", "test"), "duplicate key rejected");
    check(!nvcr::detail::json::parse("[01]", "test"), "leading zero rejected");
    check(!nvcr::detail::json::parse(std::string(100, '[') + std::string(100, ']'), "test"),
          "deep nesting rejected");
}

void write(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

std::string metadata_json(const std::string& layout) {
    const auto fields = [&](std::initializer_list<const char*> names) {
        std::string output = "[";
        for (const auto* name : names) {
            if (output.size() > 1U) output += ",";
            output += layout.empty() ? std::string("\"") + name + "\"" :
                std::string(R"({"name":")") + name + R"(","layout":")" + layout + "\"}";
        }
        return output + "]";
    };
    return std::string(R"({"params":{"split_model_params":{"split_type":"dmc61sbr_e1d1",)") +
        R"("model_width":32,"model_height":32}},"model_parts_metadata":{)" +
        R"("MLVCEncoder":{"precision":"fp16","input_fields":)" +
        fields({"x", "ref_feature", "q_index_shifted"}) + R"(,"output_fields":)" +
        fields({"feature", "z_raw", "y_raw_0", "y_raw_1"}) + "}," +
        R"("MLVCDecoder":{"precision":"fp16","input_fields":)" +
        fields({"z_raw", "y_raw_0", "y_raw_1", "ref_feature", "q_index_shifted"}) +
        R"(,"output_fields":)" + fields({"x_hat", "feature"}) + "}}}";
}

void write_bundle(const fs::path& root, const std::string& layout, const std::string& interface) {
    fs::remove_all(root);
    fs::create_directories(root);
    write(root / "MLVCEncoder_32x32.onnx", "encoder");
    write(root / "MLVCDecoder_32x32.onnx", "decoder");
    write(root / "weights.bin", "weights");
    write(root / "metadata_32x32.json", metadata_json(layout));
    write(root / "gaussian_pmf.json",
          R"({"pmf_lengths":[2,2],"pmf_offsets":[0,0],"pmf_table":[1,65535,65535,1],)"
          R"("scale_min":0.11,"scale_max":16.0,"scale_levels":2,"index_space":true})");
    write(root / "bit_estimator_pmf.json",
          R"({"pmf_lengths":[2,2],"pmf_offsets":[0,0],"pmf_table":[1,65535,65535,1],)"
          R"("qp_num":1,"channels":2})");
    std::string digests[2];
    for (int part = 0; part < 2; ++part) {
        const std::array<fs::path, 2> files{
            root / (part == 0 ? "MLVCEncoder_32x32.onnx" : "MLVCDecoder_32x32.onnx"),
            root / "weights.bin"};
        digests[part] = nvcr::detail::sha256_files(files, "test").value();
    }
    write(root / "bundle_manifest.json",
          std::string(R"({"bundle_name":"synthetic","model_type":"onnx","target_device":"generic",)") +
              R"("model_registry":{"e":{"model_path":"MLVCEncoder_32x32.onnx","weights_path":"weights.bin","sha256":")" +
              digests[0] + R"("},"d":{"model_path":"MLVCDecoder_32x32.onnx","weights_path":"weights.bin","sha256":")" +
              digests[1] + R"("}},"model_manifests":{"32x32":{"metadata_path":"metadata_32x32.json",)" +
              R"("scale_decoder_model_path":null,"gaussian_pmf_path":"gaussian_pmf.json",)" +
              R"("bit_estimator_pmf_path":"bit_estimator_pmf.json","model_parts":{)" +
              R"("MLVCEncoder":{"registry_id":"e","function_name":null},)" +
              R"("MLVCDecoder":{"registry_id":"d","function_name":null}},)" +
              R"("encoder_interface_type":")" + interface + R"(","decoder_interface_type":")" +
              interface + R"("}},"model_metadata":{"32x32":{"model_width":32,"model_height":32,)" +
              R"("pixel_range":1.0,"qp_num":1,"total_qp_num":9,"frame_index_map":[0,1],)" +
              R"("qp_shift":[0,8],"feature_channels":4,"latent_channels":4,)" +
              R"("hyperprior_channels":2,"downsample_feature":8,"downsample_latent":16,)" +
              R"("downsample_hyperprior":2,"scale_decoder_type":"upsample","y_scale_repeat":2,)" +
              R"("iframe_period":1024,"reset_period":null,"ltr_start_idx":8,"ltr_period":64}}})");
}

void bundle_tests() {
    const auto root = fs::temp_directory_path() / "nvcr-mlvc-unit-bundle";
    write_bundle(root, "", "fp16_scale_sending_no_reset_1p");
    auto bundle = load_bundle(root);
    check(bundle.has_value(), "legacy string-field bundle loads");
    if (bundle) {
        const auto& model = bundle.value().models.front();
        check(model.stream_identity == "synthetic-32x32" && model.encoder.component_id ==
                  "MLVCEncoder_32x32",
              "bundle identity and part names");
        check(verify_digests(model).has_value(), "registry digests verify");
        check(select_model(bundle.value(), 32, 30).has_value(), "frame fits the canvas");
        check(!select_model(bundle.value(), 34, 34), "larger frame rejected");
        write(root / "weights.bin", "tampered");
        auto mismatch = verify_digests(model);
        check(!mismatch && mismatch.error().code() == nvcr::ErrorCode::digest_mismatch,
              "tampered weights detected");
    }
    write_bundle(root, "nchw", "fp16_scale_sending_no_reset_1p");
    check(load_bundle(root).has_value(), "layout-tagged NCHW bundle loads");
    write_bundle(root, "nhwc", "fp16_scale_sending_no_reset_1p");
    auto nhwc = load_bundle(root);
    check(!nhwc && nhwc.error().code() == nvcr::ErrorCode::not_implemented,
          "NHWC bundle fails closed");
    write_bundle(root, "", "another_interface");
    check(!load_bundle(root), "unknown interface rejected");
    fs::remove_all(root);
}

}  // namespace

int main() {
    record_tests();
    scheduler_tests();
    reference_tests();
    q_shift_tests();
    placement_tests();
    conversion_tests();
    scale_tests();
    json_tests();
    bundle_tests();
    if (failures != 0) {
        std::cerr << failures << " MLVC unit check(s) failed\n";
        return 1;
    }
    std::cout << "MLVC unit tests passed\n";
    return 0;
}
