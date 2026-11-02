#pragma once

// MLVC frame scheduling and reference state, following MLVC's conversion
// frame loop (video/conversion/_frame_loop.py) and reference manager
// (video/conversion/_split_model/_base_split_model.py).

#include "bundle.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace nvcr::mlvc {

enum class FrameClass : std::uint8_t {
    intra,
    predicted,
    ltr_recovery,
};

struct LoopParameters final {
    std::optional<std::uint32_t> iframe_period;
    std::optional<std::uint32_t> reset_period;
    std::uint32_t ltr_start_idx{};
    std::optional<std::uint32_t> ltr_period;
    bool proactive_ltr_recovery{true};

    [[nodiscard]] static LoopParameters from_model(const ModelParameters& model) {
        return {model.iframe_period, model.reset_period, model.ltr_start_idx, model.ltr_period, true};
    }
};

struct FramePlan final {
    FrameClass frame_class{FrameClass::intra};
    std::uint64_t coded_index{};  // MLVC cur_frame_idx
    std::optional<std::uint64_t> reference;  // coded index of the reference feature
    bool mark_as_ltr{};
    bool feature_reset{};

    // Only a plain P frame that changes no reference state may be dropped.
    [[nodiscard]] bool droppable() const noexcept {
        return frame_class == FrameClass::predicted && !mark_as_ltr && !feature_reset;
    }
};

class FrameScheduler final {
public:
    explicit FrameScheduler(LoopParameters parameters) : parameters_(parameters) {}

    // Plans the next coded frame. The encoder lets the I period decide; the
    // decoder passes the stream's intra flag through force_intra.
    [[nodiscard]] FramePlan plan(std::optional<bool> stream_intra = std::nullopt) const {
        FramePlan plan;
        bool intra = !started_ ||
            (parameters_.iframe_period && coded_index_ % *parameters_.iframe_period == 0U);
        if (stream_intra) intra = *stream_intra;
        plan.coded_index = intra ? 0U : coded_index_;
        const auto latest_ltr = intra ? std::nullopt : latest_ltr_;
        const auto index = plan.coded_index;

        plan.feature_reset =
            parameters_.reset_period && (index + 1U) % *parameters_.reset_period == 0U;
        plan.mark_as_ltr = parameters_.ltr_period && *parameters_.ltr_period > 0U &&
            (index == parameters_.ltr_start_idx ||
             (index > parameters_.ltr_start_idx && index % *parameters_.ltr_period == 0U));

        if (intra) {
            plan.frame_class = FrameClass::intra;
        } else if (parameters_.proactive_ltr_recovery && plan.mark_as_ltr && latest_ltr) {
            plan.frame_class = FrameClass::ltr_recovery;
            plan.reference = latest_ltr;
        } else {
            plan.frame_class = FrameClass::predicted;
            plan.reference = index - 1U;
        }
        return plan;
    }

    // Records a coded frame. Dropped frames are never committed.
    void commit(const FramePlan& plan) noexcept {
        started_ = true;
        if (plan.frame_class == FrameClass::intra) latest_ltr_.reset();
        if (plan.mark_as_ltr) latest_ltr_ = plan.coded_index;
        coded_index_ = plan.coded_index + 1U;
    }

    void reset() noexcept {
        started_ = false;
        coded_index_ = 0;
        latest_ltr_.reset();
    }

    [[nodiscard]] bool started() const noexcept { return started_; }
    [[nodiscard]] const LoopParameters& parameters() const noexcept { return parameters_; }

private:
    LoopParameters parameters_;
    bool started_{false};
    std::uint64_t coded_index_{0};
    std::optional<std::uint64_t> latest_ltr_;
};

// Keeps the most recent short-term reference and the three most recent
// long-term references, as MLVC's BaseReferenceManager._prune does.
template <typename Feature>
class ReferenceStore final {
public:
    static constexpr std::size_t short_term_capacity = 1;
    static constexpr std::size_t long_term_capacity = 3;

    void clear() noexcept { entries_.clear(); }

    void save(std::uint64_t coded_index, Feature feature, bool long_term) {
        entries_.insert_or_assign(coded_index, Entry{std::move(feature), long_term});
        prune(false, short_term_capacity);
        prune(true, long_term_capacity);
    }

    [[nodiscard]] const Feature* find(std::uint64_t coded_index) const noexcept {
        const auto found = entries_.find(coded_index);
        return found == entries_.end() ? nullptr : &found->second.feature;
    }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry final {
        Feature feature;
        bool long_term{};
    };

    void prune(bool long_term, std::size_t capacity) {
        std::vector<std::uint64_t> matching;
        for (const auto& [index, entry] : entries_) {
            if (entry.long_term == long_term) matching.push_back(index);
        }
        // entries_ is ordered, so the oldest indices come first.
        if (matching.size() <= capacity) return;
        for (auto it = matching.begin();
             it != std::prev(matching.end(), static_cast<std::ptrdiff_t>(capacity)); ++it) {
            entries_.erase(*it);
        }
    }

    std::map<std::uint64_t, Entry> entries_;
};

// q_index_shifted = q + qp_shift[frame_index_map[(cur_frame_idx + 1) % len]]
// (_base_split_model.py _get_q_index_shift).
[[nodiscard]] inline std::uint32_t shifted_q_index(
    const ModelParameters& model, std::uint32_t q_index, std::uint64_t coded_index) {
    const auto& map = model.frame_index_map;
    return q_index + model.qp_shift[map[(coded_index + 1U) % map.size()]];
}

}  // namespace nvcr::mlvc
