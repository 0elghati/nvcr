#pragma once

// Worker threads shared by the CPU reference executors (nvcr.intconv.v1 and
// nvcr.fx.v1). Each index must write disjoint outputs, so the split cannot
// change a result.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace nvcr::deterministic {

// Clears flush-to-zero/denormals-are-zero for the calling thread so CPU results
// keep denormals exactly like the GPU kernels.
class DenormalGuard final {
public:
    DenormalGuard() {
#if defined(__x86_64__) || defined(_M_X64)
        saved_ = _mm_getcsr();
        _mm_setcsr(saved_ & ~0x8040U);
#endif
    }
    ~DenormalGuard() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr(saved_);
#endif
    }
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;

private:
    [[maybe_unused]] unsigned int saved_{0};
};

template <typename Function>
void parallel_for(std::int32_t count, Function function) {
    const auto workers = static_cast<std::int32_t>(std::clamp<unsigned>(
        std::thread::hardware_concurrency(), 1U, 16U));
    if (workers <= 1 || count < 4) {
        for (std::int32_t index = 0; index < count; ++index) function(index);
        return;
    }
    // Indexes are taken one at a time, so faster cores take more of them.
    std::atomic<std::int32_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (std::int32_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            const DenormalGuard guard;
            for (std::int32_t index = next++; index < count; index = next++) function(index);
        });
    }
    for (auto& thread : threads) thread.join();
}

}  // namespace nvcr::deterministic
