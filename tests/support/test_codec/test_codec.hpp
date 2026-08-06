#pragma once

#include "nvcr/codec/adapter.hpp"
#include "nvcr/codec/session.hpp"

#include <cstddef>
#include <memory>

namespace nvcr::test_support {

void register_test_codec();

[[nodiscard]] std::unique_ptr<codec::ICodecAdapter> make_test_codec_adapter();

[[nodiscard]] std::unique_ptr<IEncoderSession>
make_test_encoder_session(std::size_t delay_frames = 1U);

[[nodiscard]] std::unique_ptr<IDecoderSession> make_test_decoder_session();

}  // namespace nvcr::test_support
