#include "payload.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool bytes_equal(std::span<const std::byte> left, std::span<const std::byte> right) {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin());
}

}  // namespace

int main() {
    const std::vector<std::byte> rans{
        std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40}};

    auto intra_bytes = nvcr::dcvcrt::make_intra_payload(1280, 720, 32, true, rans);
    auto intra = nvcr::dcvcrt::parse_intra_payload(intra_bytes);
    if (!intra) return fail("valid intra payload was rejected");
    if (intra.value().width != 1280 || intra.value().height != 720 ||
        intra.value().qp != 32 || !intra.value().two_coders ||
        !bytes_equal(intra.value().rans, rans)) {
        return fail("intra payload fields did not round trip");
    }

    auto predicted_bytes =
        nvcr::dcvcrt::make_predicted_payload(640, 360, 40, false, true, rans);
    auto predicted = nvcr::dcvcrt::parse_predicted_payload(predicted_bytes);
    if (!predicted) return fail("valid predicted payload was rejected");
    if (predicted.value().width != 640 || predicted.value().height != 360 ||
        predicted.value().qp != 40 || predicted.value().two_coders ||
        !predicted.value().use_frame_reference ||
        !bytes_equal(predicted.value().rans, rans)) {
        return fail("predicted payload fields did not round trip");
    }

    auto truncated = nvcr::dcvcrt::parse_intra_payload(
        std::span<const std::byte>(intra_bytes).first(8));
    if (truncated) return fail("truncated intra payload was accepted");

    auto invalid_qp = nvcr::dcvcrt::make_predicted_payload(640, 360, 72, false, false, rans);
    if (nvcr::dcvcrt::parse_predicted_payload(invalid_qp)) {
        return fail("out-of-range predicted QP was accepted");
    }

    auto odd_dimensions = nvcr::dcvcrt::make_intra_payload(641, 360, 32, false, rans);
    if (nvcr::dcvcrt::parse_intra_payload(odd_dimensions)) {
        return fail("odd intra dimensions were accepted");
    }

    return 0;
}
