#include "orchestration.hpp"

#include <cstdint>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

nvcr::codec::SequenceStateView state(
    std::uint64_t frame_index, std::uint64_t generation) {
    nvcr::codec::SequenceStateView value;
    value.frame_index = frame_index;
    value.generation = generation;
    return value;
}

}  // namespace

int main() {
    nvcr::dcvcrt::PredictedOrchestration orchestration;
    nvcr::dcvcrt::ReferenceState reference;

    auto missing = orchestration.select_frame(32, state(1, 7), reference);
    if (missing || missing.error().code() != nvcr::ErrorCode::invalid_state) {
        return fail("missing reference state was accepted");
    }

    reference.commit(state(0, 7), false);
    auto first_predicted = orchestration.select_frame(32, state(1, 7), reference);
    if (!first_predicted || first_predicted.value().qp != 40 ||
        !first_predicted.value().use_frame_reference) {
        return fail("first predicted frame decision is incorrect");
    }

    auto missing_feature = orchestration.validate_decode_reference(
        false, state(1, 7), reference);
    if (missing_feature ||
        missing_feature.error().code() != nvcr::ErrorCode::invalid_state) {
        return fail("missing feature reference was accepted");
    }

    reference.commit(state(1, 7), true);
    auto feature_reference = orchestration.select_frame(32, state(2, 7), reference);
    if (!feature_reference || feature_reference.value().qp != 32 ||
        feature_reference.value().use_frame_reference) {
        return fail("feature reference decision is incorrect");
    }
    if (!orchestration.validate_decode_reference(
            false, state(2, 7), reference)) {
        return fail("available feature reference was rejected");
    }

    reference.commit(state(64, 7), true);
    auto periodic_frame = orchestration.select_frame(32, state(65, 7), reference);
    if (!periodic_frame || periodic_frame.value().qp != 40 ||
        !periodic_frame.value().use_frame_reference) {
        return fail("periodic frame-reference decision is incorrect");
    }

    auto invalid_qp = orchestration.select_frame(64, state(65, 7), reference);
    if (invalid_qp ||
        invalid_qp.error().code() != nvcr::ErrorCode::backend_error) {
        return fail("invalid base P-frame QP was accepted");
    }

    auto wrong_generation = orchestration.select_frame(32, state(65, 8), reference);
    if (wrong_generation ||
        wrong_generation.error().code() != nvcr::ErrorCode::invalid_state) {
        return fail("reference state from another generation was accepted");
    }

    reference.reset();
    if (reference.matches(state(0, 0)) || reference.has_feature()) {
        return fail("reference reset retained codec state");
    }

    return 0;
}
