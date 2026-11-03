// Bitwise parity of NVCR's DCVC-UF glue kernels (src/dcvcuf/backend/tensorrt)
// with the pinned upstream kernels compiled into the native DCVC-UF build.
// Both run on identical channels-last FP16 inputs on the current stream; every
// output must match bit for bit. Unary paths sweep all 65,536 FP16 patterns.

#include "def_elementwise.h"

#include "glue.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace glue = nvcr::dcvcuf::glue;

int failures = 0;
int checks = 0;

const auto half_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
const auto bool_options = at::TensorOptions().dtype(at::kBool).device(at::kCUDA);

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void expect_same(const std::string& name, const at::Tensor& upstream, const at::Tensor& port) {
    ++checks;
    torch::cuda::synchronize();
    const auto a = upstream.contiguous().view({-1}).view(at::kByte).cpu();
    const auto b = port.contiguous().view({-1}).view(at::kByte).cpu();
    if (a.numel() != b.numel() || !torch::equal(a, b)) {
        const auto differing = a.numel() == b.numel() ? (a != b).sum().item<std::int64_t>() : -1;
        std::cerr << "FAILED " << name << ": " << differing << " differing bytes of "
                  << a.numel() << '\n';
        ++failures;
    }
}

void expect_ok(const std::string& name, cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "FAILED " << name << ": " << cudaGetErrorString(status) << '\n';
        ++failures;
    }
}

// Channels-last values mixing a normal spread, exact .5 rounding boundaries,
// large magnitudes and small values.
at::Tensor values(int C, int H, int W, double scale, std::uint64_t seed) {
    torch::manual_seed(seed);
    auto x = torch::randn({1, C, H, W}, at::TensorOptions().dtype(at::kFloat)) * scale;
    auto boundary = torch::randint(-130, 130, {1, C, H, W}).to(at::kFloat) + 0.5;
    auto pick = torch::rand({1, C, H, W});
    x = torch::where(pick < 0.15, boundary, x);
    x = torch::where((pick >= 0.15) & (pick < 0.2), x * 1e-4, x);
    x = torch::where((pick >= 0.2) & (pick < 0.23), x * 300.0, x);
    return x.to(half_options).contiguous(at::MemoryFormat::ChannelsLast);
}

at::Tensor positive(int C, int H, int W, std::uint64_t seed) {
    return values(C, H, W, 2.0, seed).abs().contiguous(at::MemoryFormat::ChannelsLast);
}

at::Tensor mask(int C, int H, int W, std::uint64_t seed) {
    torch::manual_seed(seed);
    return (torch::rand({1, C, H, W}) < 0.5).to(bool_options).contiguous(
        at::MemoryFormat::ChannelsLast);
}

at::Tensor like(const at::Tensor& x) { return torch::empty_like(x); }

// Every FP16 bit pattern, as a [1, C, H, W] channels-last tensor.
at::Tensor all_halves(int C) {
    auto bits = torch::arange(0, 65536, at::TensorOptions().dtype(at::kInt))
                    .to(at::kShort).view(at::kHalf);
    return bits.view({1, C, 65536 / C / 16, 16}).to(at::kCUDA).contiguous(
        at::MemoryFormat::ChannelsLast);
}

glue::HalfTensor view(at::Tensor& tensor) {
    const bool nhwc = tensor.is_contiguous(at::MemoryFormat::ChannelsLast);
    return {static_cast<std::uint16_t*>(tensor.data_ptr()), static_cast<int>(tensor.size(1)),
            static_cast<int>(tensor.size(2)), static_cast<int>(tensor.size(3)),
            nhwc ? glue::Layout::nhwc : glue::Layout::nchw};
}

glue::ConstHalfTensor cview(const at::Tensor& tensor) {
    auto mutable_tensor = tensor;
    auto v = view(mutable_tensor);
    return {v.data, v.channels, v.height, v.width, v.layout};
}

glue::MaskTensor mview(const at::Tensor& tensor) {
    return {static_cast<const bool*>(tensor.data_ptr()), static_cast<int>(tensor.size(1)),
            static_cast<int>(tensor.size(2)), static_cast<int>(tensor.size(3)),
            tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ? glue::Layout::nhwc :
                                                                    glue::Layout::nchw};
}

const std::uint16_t* hp(const at::Tensor& tensor) {
    return static_cast<const std::uint16_t*>(tensor.data_ptr());
}

struct Shape {
    int C;
    int H;
    int W;
};

void test_shape(const Shape& s, std::uint64_t seed) {
    const auto tag = "[" + std::to_string(s.C) + "x" + std::to_string(s.H) + "x" +
                     std::to_string(s.W) + "]";
    const auto y = values(s.C, s.H, s.W, 20.0, seed);
    const auto scales = positive(s.C, s.H, s.W, seed + 1);
    const auto means = values(s.C, s.H, s.W, 3.0, seed + 2);
    const auto m = mask(s.C, s.H, s.W, seed + 3);
    const auto q4 = positive(s.C, s.H, s.W, seed + 4);
    const auto q1 = positive(s.C, 1, 1, seed + 5).view({s.C});

    {
        auto a = like(y);
        auto b = like(y);
        add_and_multiply_broadcast_cuda(y, means, q1, a);
        expect_ok("add_and_multiply_broadcast", glue::add_and_multiply_broadcast(
            view(b), cview(y), cview(means), hp(q1), stream()));
        expect_same("add_and_multiply_broadcast" + tag, a, b);
    }
    {
        auto a = y.clone(at::MemoryFormat::ChannelsLast);
        auto b = y.clone(at::MemoryFormat::ChannelsLast);
        add_and_multiply_with_clamp_min_inplace_cuda(a, means, q4);
        expect_ok("add_and_multiply_with_clamp_min_inplace",
                  glue::add_and_multiply_with_clamp_min_inplace(view(b), cview(means), cview(q4),
                                                                stream()));
        expect_same("add_and_multiply_with_clamp_min_inplace" + tag, a, b);
    }
    {
        auto a = y.clone(at::MemoryFormat::ChannelsLast);
        auto b = y.clone(at::MemoryFormat::ChannelsLast);
        divide_with_clamp_min_inplace_cuda(a, q4);
        expect_ok("divide_with_clamp_min_inplace",
                  glue::divide_with_clamp_min_inplace(view(b), cview(q4), stream()));
        expect_same("divide_with_clamp_min_inplace" + tag, a, b);
    }
    {
        auto a = like(y);
        auto b = like(y);
        multiply_with_broadcast_cuda(y, q1, a);
        expect_ok("multiply_with_broadcast",
                  glue::multiply_with_broadcast(view(b), cview(y), hp(q1), stream()));
        expect_same("multiply_with_broadcast" + tag, a, b);
    }
    for (const float threshold : {0.0F, 0.15F}) {
        const auto ttag = tag + " thr=" + std::to_string(threshold);
        const int N = s.C * s.H * s.W;
        {
            auto ua = torch::empty({N}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
            auto uc = torch::empty({N}, bool_options);
            auto pa = like(ua);
            auto pc = like(uc);
            build_index_dec_cuda(scales, threshold, at::nullopt, 0, ua, uc);
            expect_ok("build_index_dec", glue::build_index_dec(
                cview(scales), threshold, pa.data_ptr<std::uint8_t>(), pc.data_ptr<bool>(),
                stream()));
            expect_same("build_index_dec.index" + ttag, ua, pa);
            expect_same("build_index_dec.condition" + ttag, uc, pc);
        }
        const auto symbols = torch::round(y.to(at::kFloat).clamp(-100, 100)).to(at::kHalf)
                                 .contiguous(at::MemoryFormat::ChannelsLast);
        auto ui = torch::empty({N}, at::TensorOptions().dtype(at::kShort).device(at::kCUDA));
        auto uc = torch::empty({N}, bool_options);
        {
            auto pi = like(ui);
            auto pc = like(uc);
            build_index_enc_cuda(symbols, scales, threshold, at::nullopt, 0, ui, uc);
            expect_ok("build_index_enc", glue::build_index_enc(
                cview(symbols), cview(scales), threshold, pi.data_ptr<std::int16_t>(),
                pc.data_ptr<bool>(), stream()));
            expect_same("build_index_enc.index" + ttag, ui, pi);
            expect_same("build_index_enc.condition" + ttag, uc, pc);
        }
        {
            auto out_u = torch::zeros_like(ui);
            auto total_u = torch::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
            auto num_u = torch::zeros({1024, 1024}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
            auto out_p = torch::zeros_like(ui);
            auto total_p = torch::zeros_like(total_u);
            auto num_p = torch::zeros_like(num_u);
            conditional_index_part1_cuda(ui, uc, out_u, total_u, num_u);
            expect_ok("conditional_index_part1", glue::conditional_index_part1(
                ui.data_ptr(), 2, true, uc.data_ptr<bool>(), N, out_p.data_ptr(),
                total_p.data_ptr<int>(), num_p.data_ptr<int>(), stream()));
            expect_same("conditional_index_part1.total" + ttag, total_u, total_p);
            expect_same("conditional_index_part1.out" + ttag, out_u, out_p);

            // Flat in memory (HWC) order, as upstream reads its channels-last buffer.
            auto y8 = symbols.to(at::kChar).permute({0, 2, 3, 1}).reshape({-1});
            for (const bool to_half : {true, false}) {
                auto ru = to_half ? torch::empty({N}, half_options) :
                                    torch::empty({N}, at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
                auto rp = like(ru);
                conditional_recover_with_type_conversion_cuda(
                    y8, uc, num_u, to_half ? at::kHalf : at::kChar, ru);
                expect_ok("conditional_recover", glue::conditional_recover(
                    y8.data_ptr<std::int8_t>(), uc.data_ptr<bool>(), num_u.data_ptr<int>(), N,
                    to_half, rp.data_ptr(), stream()));
                expect_same(std::string("conditional_recover.") + (to_half ? "half" : "int8") + ttag,
                            ru, rp);
            }
        }
        {
            auto uq = like(y), uh = like(y), us = like(y);
            auto pq = like(y), ph = like(y), ps = like(y);
            process_with_mask_cuda(y, scales, means, m, threshold, 0, uq, uh, us);
            expect_ok("process_with_mask", glue::process_with_mask(
                view(pq), view(ph), view(ps), cview(y), cview(scales), cview(means), mview(m),
                threshold, stream()));
            expect_same("process_with_mask.y_q" + ttag, uq, pq);
            expect_same("process_with_mask.y_hat" + ttag, uh, ph);
            expect_same("process_with_mask.s_hat" + ttag, us, ps);
        }
        {
            auto uq = like(y), uh = like(y), pq = like(y), ph = like(y);
            process_with_mask_no_scale_cuda(y, scales, means, m, threshold, 0, uq, uh);
            expect_ok("process_with_mask_no_scale", glue::process_with_mask_no_scale(
                view(pq), view(ph), cview(y), cview(scales), cview(means), mview(m), threshold,
                stream()));
            expect_same("process_with_mask_no_scale.y_q" + ttag, uq, pq);
            expect_same("process_with_mask_no_scale.y_hat" + ttag, uh, ph);
        }
        {
            auto uq = means.clone(at::MemoryFormat::ChannelsLast);
            auto uh = scales.clone(at::MemoryFormat::ChannelsLast);
            auto pq = means.clone(at::MemoryFormat::ChannelsLast);
            auto ph = scales.clone(at::MemoryFormat::ChannelsLast);
            process_with_mask_no_scale_add_inplace_cuda(y, scales, means, m, uq, uh, threshold, 0);
            expect_ok("process_with_mask_no_scale_add_inplace",
                      glue::process_with_mask_no_scale_add_inplace(
                          view(pq), view(ph), cview(y), cview(scales), cview(means), mview(m),
                          threshold, stream()));
            expect_same("process_with_mask_no_scale_add_inplace.y_q" + ttag, uq, pq);
            expect_same("process_with_mask_no_scale_add_inplace.y_hat" + ttag, uh, ph);
        }
        {
            auto uq = means.clone(at::MemoryFormat::ChannelsLast);
            auto uh = scales.clone(at::MemoryFormat::ChannelsLast);
            auto pq = means.clone(at::MemoryFormat::ChannelsLast);
            auto ph = scales.clone(at::MemoryFormat::ChannelsLast);
            process_with_mask_no_scale_add_and_multiply_inplace_cuda(
                y, scales, means, m, uq, uh, q4, threshold, 0);
            expect_ok("process_with_mask_no_scale_add_and_multiply_inplace",
                      glue::process_with_mask_no_scale_add_and_multiply_inplace(
                          view(pq), view(ph), cview(y), cview(scales), cview(means), mview(m),
                          cview(q4), threshold, stream()));
            expect_same("process_with_mask_no_scale_add_and_multiply_inplace.y_q" + ttag, uq, pq);
            expect_same("process_with_mask_no_scale_add_and_multiply_inplace.y_hat" + ttag, uh, ph);
        }
    }
    {
        auto a = like(y), b = like(y);
        restore_y_cuda(y, means, m, a);
        expect_ok("restore_y", glue::restore_y(view(b), cview(y), cview(means), mview(m), stream()));
        expect_same("restore_y" + tag, a, b);
        auto c = scales.clone(at::MemoryFormat::ChannelsLast);
        auto d = scales.clone(at::MemoryFormat::ChannelsLast);
        restore_y_and_add_inplace_cuda(y, means, m, c);
        expect_ok("restore_y_and_add_inplace", glue::restore_y_and_add_inplace(
            view(d), cview(y), cview(means), mview(m), stream()));
        expect_same("restore_y_and_add_inplace" + tag, c, d);
        auto e = scales.clone(at::MemoryFormat::ChannelsLast);
        auto f = scales.clone(at::MemoryFormat::ChannelsLast);
        restore_y_and_add_multiply_inplace_cuda(y, means, m, e, q4);
        expect_ok("restore_y_and_add_multiply_inplace", glue::restore_y_and_add_multiply_inplace(
            view(f), cview(y), cview(means), mview(m), cview(q4), stream()));
        expect_same("restore_y_and_add_multiply_inplace" + tag, e, f);
    }
    if (s.C % 4 == 0) {
        const int Q = s.C / 4;
        const auto y_quarter = values(Q, s.H, s.W, 20.0, seed + 6);
        const auto q_full = positive(s.C, 1, 1, seed + 7).view({s.C});
        auto a = like(y), b = like(y);
        restore_y_4x_cuda(y_quarter, means, m, a);
        expect_ok("restore_y_4x", glue::restore_y_4x(
            view(b), cview(y_quarter), cview(means), mview(m), stream()));
        expect_same("restore_y_4x" + tag, a, b);
        auto c = scales.clone(at::MemoryFormat::ChannelsLast);
        auto d = scales.clone(at::MemoryFormat::ChannelsLast);
        restore_y_4x_and_add_inplace_cuda(y_quarter, means, m, c);
        expect_ok("restore_y_4x_and_add_inplace", glue::restore_y_4x_and_add_inplace(
            view(d), cview(y_quarter), cview(means), mview(m), stream()));
        expect_same("restore_y_4x_and_add_inplace" + tag, c, d);
        auto e = like(y), f = like(y);
        restore_y_4x_add_and_multiply_broadcast_cuda(y_quarter, means, m, scales, q_full, e);
        expect_ok("restore_y_4x_add_and_multiply_broadcast",
                  glue::restore_y_4x_add_and_multiply_broadcast(
                      view(f), cview(y_quarter), cview(means), mview(m), cview(scales),
                      hp(q_full), stream()));
        expect_same("restore_y_4x_add_and_multiply_broadcast" + tag, e, f);

        auto r = torch::empty({1, Q, s.H, s.W}, half_options).contiguous(at::MemoryFormat::ChannelsLast);
        auto t = like(r);
        single_part_for_reading_4x_cuda(y, m, r);
        expect_ok("single_part_for_reading_4x", glue::single_part_for_reading_4x(
            view(t), cview(y), mview(m), stream()));
        expect_same("single_part_for_reading_4x" + tag, r, t);
        auto u = like(r), v = like(r);
        single_part_for_writing_4x_cuda(y, u);
        expect_ok("single_part_for_writing_4x",
                  glue::single_part_for_writing_4x(view(v), cview(y), stream()));
        expect_same("single_part_for_writing_4x" + tag, u, v);
    }
    {
        for (const auto [pad_b, pad_r] : {std::pair{0, 0}, std::pair{3, 1}, std::pair{2, 2}}) {
            auto a = torch::empty({1, s.C, s.H + pad_b, s.W + pad_r}, half_options)
                         .contiguous(at::MemoryFormat::ChannelsLast);
            auto b = like(a);
            replicate_pad_cuda(y, pad_b, pad_r, a);
            expect_ok("replicate_pad", glue::replicate_pad(view(b), cview(y), pad_b, pad_r, stream()));
            expect_same("replicate_pad" + tag + " pad=" + std::to_string(pad_b) + "," +
                        std::to_string(pad_r), a, b);
        }
        if (s.H > 1 && s.W > 1) {
            auto a = torch::empty({1, s.C, s.H - 1, s.W - 1}, half_options)
                         .contiguous(at::MemoryFormat::ChannelsLast);
            auto b = like(a);
            slice_cuda(y, s.H - 1, s.W - 1, a);
            expect_ok("slice", glue::slice(view(b), cview(y), stream()));
            expect_same("slice" + tag, a, b);
        }
    }
}

void test_shuffles(std::uint64_t seed) {
    for (const int factor : {2, 4, 8}) {
        const auto x = values(3, 16 * factor, 8 * factor, 0.4, seed + factor);
        auto a = torch::empty({1, 3 * factor * factor, 16, 8}, half_options)
                     .contiguous(at::MemoryFormat::ChannelsLast);
        auto b = like(a);
        pixel_unshuffle_cuda(x, factor, a);
        expect_ok("pixel_unshuffle", glue::pixel_unshuffle(view(b), cview(x), factor, stream()));
        expect_same("pixel_unshuffle x" + std::to_string(factor), a, b);
    }
    for (const auto [pad_b, pad_r] : {std::pair{0, 0}, std::pair{8, 0}, std::pair{4, 12}}) {
        const auto x = values(3, 64 - pad_b, 96 - pad_r, 0.4, seed + 20);
        auto a = torch::empty({1, 192, 8, 12}, half_options).contiguous(at::MemoryFormat::ChannelsLast);
        auto b = like(a);
        pad_and_unshuffle_8_cuda(x, pad_b, pad_r, a);
        expect_ok("pad_and_unshuffle_8", glue::pad_and_unshuffle_8(view(b), cview(x), pad_b, pad_r, stream()));
        expect_same("pad_and_unshuffle_8 pad=" + std::to_string(pad_b) + "," + std::to_string(pad_r), a, b);
    }
    for (const bool clamp : {false, true}) {
        for (const int channels : {192, 512}) {
            const auto x = values(channels, 6, 10, 0.6, seed + channels);
            auto a = torch::empty({1, channels / 64, 48, 80}, half_options)
                         .contiguous(at::MemoryFormat::ChannelsLast);
            auto b = like(a);
            pixel_shuffle_8_cuda(x, clamp, a);
            expect_ok("pixel_shuffle_8", glue::pixel_shuffle_8(view(b), cview(x), clamp, stream()));
            expect_same("pixel_shuffle_8 C=" + std::to_string(channels) +
                        (clamp ? " clamp" : ""), a, b);
        }
    }
    {
        const auto x = values(64, 6, 10, 1.0, seed + 30);
        auto a = torch::empty({1, 16, 12, 20}, half_options).contiguous(at::MemoryFormat::ChannelsLast);
        auto b = like(a);
        pixel_shuffle_2_cuda(x, a);
        expect_ok("pixel_shuffle_2", glue::pixel_shuffle_2(view(b), cview(x), stream()));
        expect_same("pixel_shuffle_2", a, b);
    }
}

void test_exhaustive() {
    // Every FP16 pattern through round_z, scale-to-index and int8 conversion.
    const auto z = all_halves(128);
    const auto N = static_cast<int>(z.numel());
    {
        auto uh = like(z);
        auto u8 = torch::empty({N}, at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
        auto ph = like(z);
        auto p8 = like(u8);
        round_z_cuda(z, uh, u8);
        expect_ok("round_z", glue::round_z(hp(z), N, static_cast<std::uint16_t*>(ph.data_ptr()),
                                           p8.data_ptr<std::int8_t>(), stream()));
        expect_same("round_z.z_hat all fp16", uh, ph);
        expect_same("round_z.int8 all fp16", u8, p8);
    }
    for (const float threshold : {0.0F, 0.15F}) {
        auto ua = torch::empty({N}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
        auto uc = torch::empty({N}, bool_options);
        auto pa = like(ua);
        auto pc = like(uc);
        build_index_dec_cuda(z, threshold, at::nullopt, 0, ua, uc);
        expect_ok("build_index_dec", glue::build_index_dec(
            cview(z), threshold, pa.data_ptr<std::uint8_t>(), pc.data_ptr<bool>(), stream()));
        expect_same("build_index_dec.index all fp16 thr=" + std::to_string(threshold), ua, pa);
        expect_same("build_index_dec.condition all fp16 thr=" + std::to_string(threshold), uc, pc);
    }
    {
        auto x8 = torch::arange(-128, 128, at::TensorOptions().dtype(at::kInt)).to(at::kChar)
                      .repeat({4}).to(at::kCUDA);
        for (const bool to_half : {true, false}) {
            auto u = to_half ? torch::empty({1024}, half_options) :
                               torch::empty({1024}, at::TensorOptions().dtype(at::kShort).device(at::kCUDA));
            auto p = like(u);
            int8_to_dtype_cuda(x8, to_half ? at::kHalf : at::kShort, 1024, 1, 1, u);
            expect_ok("int8_to_dtype", glue::int8_to_dtype(
                x8.data_ptr<std::int8_t>(), 1024, to_half, p.data_ptr(), stream()));
            expect_same(std::string("int8_to_dtype ") + (to_half ? "half" : "int16"), u, p);
        }
    }
}

}  // namespace

int main() {
    c10::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool(false, 0));
    test_exhaustive();
    const std::vector<Shape> shapes{{256, 16, 16}, {256, 17, 30}, {128, 5, 7}, {512, 34, 60}};
    for (std::size_t index = 0; index < shapes.size(); ++index) test_shape(shapes[index], 100 * index);
    test_shuffles(7);
    if (failures != 0) {
        std::cerr << failures << " of " << checks << " parity checks failed\n";
        return 1;
    }
    std::cout << "DCVC-UF glue kernels match upstream bit for bit (" << checks << " checks)\n";
    return 0;
}
