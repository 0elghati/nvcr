#pragma once

// Exact DCVC-RT scale-to-entropy-table mapping of the dcvc-rt.exact profiles
// (issue #216).
//
// The native mapping, floor((ln(clamp(s, 0.11, 16)) - ln 0.11) * 25.516), uses
// logf, whose last-bit result differs between targets and fast-math builds,
// so scales near a table boundary map to different tables. The exact mapping
// counts the boundaries a clamped scale has reached:
//
//   index = #{ i in 1..127 : clamp(s) >= T[i] }
//
// where T[i] is the smallest float32 not below exp(L + i / R), computed in
// exact real arithmetic from the native float32 constants L = -2.2072749 and
// R = 25.516415. Comparisons are exact, so every target agrees.

#define NVCR_DCVCRT_SCALE_THRESHOLDS \
    0x1.d49178p-4F, \
    0x1.e74bd6p-4F, \
    0x1.fac5d2p-4F, \
    0x1.07838cp-3F, \
    0x1.120bcep-3F, \
    0x1.1cffd6p-3F, \
    0x1.2863eep-3F, \
    0x1.343c94p-3F, \
    0x1.408e7p-3F, \
    0x1.4d5e5ap-3F, \
    0x1.5ab15ep-3F, \
    0x1.688cb4p-3F, \
    0x1.76f5d4p-3F, \
    0x1.85f268p-3F, \
    0x1.958852p-3F, \
    0x1.a5bdb4p-3F, \
    0x1.b698fp-3F, \
    0x1.c820a4p-3F, \
    0x1.da5bb6p-3F, \
    0x1.ed5152p-3F, \
    0x1.008476p-2F, \
    0x1.0ac522p-2F, \
    0x1.156eb6p-2F, \
    0x1.208562p-2F, \
    0x1.2c0d84p-2F, \
    0x1.380ba4p-2F, \
    0x1.448478p-2F, \
    0x1.517ceap-2F, \
    0x1.5efa12p-2F, \
    0x1.6d014p-2F, \
    0x1.7b97f4p-2F, \
    0x1.8ac3fp-2F, \
    0x1.9a8b28p-2F, \
    0x1.aaf3dp-2F, \
    0x1.bc045ep-2F, \
    0x1.cdc386p-2F, \
    0x1.e03844p-2F, \
    0x1.f369dap-2F, \
    0x1.03afeap-1F, \
    0x1.0e1104p-1F, \
    0x1.18dc52p-1F, \
    0x1.241612p-1F, \
    0x1.2fc2aep-1F, \
    0x1.3be6bep-1F, \
    0x1.488708p-1F, \
    0x1.55a88p-1F, \
    0x1.635054p-1F, \
    0x1.7183e2p-1F, \
    0x1.8048bep-1F, \
    0x1.8fa4b8p-1F, \
    0x1.9f9dd8p-1F, \
    0x1.b03a6ap-1F, \
    0x1.c180f2p-1F, \
    0x1.d3783ep-1F, \
    0x1.e6275ep-1F, \
    0x1.f995aap-1F, \
    0x1.06e564p+0F, \
    0x1.116754p+0F, \
    0x1.1c54c8p+0F, \
    0x1.27b20ap+0F, \
    0x1.338394p+0F, \
    0x1.3fce0cp+0F, \
    0x1.4c9646p+0F, \
    0x1.59e148p+0F, \
    0x1.67b45p+0F, \
    0x1.7614c8p+0F, \
    0x1.85085cp+0F, \
    0x1.9494ecp+0F, \
    0x1.a4c094p+0F, \
    0x1.b591b2p+0F, \
    0x1.c70eep+0F, \
    0x1.d93f02p+0F, \
    0x1.ec293cp+0F, \
    0x1.ffd5p+0F, \
    0x1.0a2506p+1F, \
    0x1.14c832p+1F, \
    0x1.1fd838p+1F, \
    0x1.2b596ep+1F, \
    0x1.37505ap+1F, \
    0x1.43c1b2p+1F, \
    0x1.50b25cp+1F, \
    0x1.5e276cp+1F, \
    0x1.6c262ep+1F, \
    0x1.7ab422p+1F, \
    0x1.89d702p+1F, \
    0x1.9994cp+1F, \
    0x1.a9f39p+1F, \
    0x1.baf9ep+1F, \
    0x1.ccae62p+1F, \
    0x1.df180cp+1F, \
    0x1.f23e1cp+1F, \
    0x1.03140ep+2F, \
    0x1.0d6eeep+2F, \
    0x1.1833c2p+2F, \
    0x1.2366c4p+2F, \
    0x1.2f0c5ep+2F, \
    0x1.3b2926p+2F, \
    0x1.47c1dap+2F, \
    0x1.54db72p+2F, \
    0x1.627b14p+2F, \
    0x1.70a61ap+2F, \
    0x1.7f621ap+2F, \
    0x1.8eb4dcp+2F, \
    0x1.9ea466p+2F, \
    0x1.af36fep+2F, \
    0x1.c0732ap+2F, \
    0x1.d25facp+2F, \
    0x1.e50396p+2F, \
    0x1.f86638p+2F, \
    0x1.06479cp+3F, \
    0x1.10c33cp+3F, \
    0x1.1baa22p+3F, \
    0x1.270092p+3F, \
    0x1.32cb04p+3F, \
    0x1.3f0e1ap+3F, \
    0x1.4bcea8p+3F, \
    0x1.5911b2p+3F, \
    0x1.66dc6cp+3F, \
    0x1.753444p+3F, \
    0x1.841eep+3F, \
    0x1.93a21ap+3F, \
    0x1.a3c40ep+3F, \
    0x1.b48b12p+3F, \
    0x1.c5fdc2p+3F, \
    0x1.d822fap+3F, \
    0x1.eb01dap+3F, \
    0x1.fea1dp+3F

#include <cstdint>

namespace nvcr::dcvcrt {

inline constexpr float exact_scale_min = 0.11F;
inline constexpr float exact_scale_max = 16.0F;
inline constexpr float exact_scale_thresholds[127] = {NVCR_DCVCRT_SCALE_THRESHOLDS};

// Binary search over the thresholds; NaN maps to table 0.
inline std::uint8_t exact_scale_index(float scale) noexcept {
    if (!(scale >= exact_scale_min)) scale = exact_scale_min;
    if (scale > exact_scale_max) scale = exact_scale_max;
    int low = 0;
    int high = 127;  // count of thresholds <= scale lies in [low, high]
    while (low < high) {
        const int middle = (low + high + 1) / 2;
        if (exact_scale_thresholds[middle - 1] <= scale) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return static_cast<std::uint8_t>(low);
}

}  // namespace nvcr::dcvcrt
