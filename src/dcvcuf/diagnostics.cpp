#include "diagnostics.hpp"

#include <atomic>

namespace nvcr::dcvcuf::diagnostics {
namespace {

std::atomic<Observer*> installed{nullptr};

}  // namespace

void set_observer(Observer* value) noexcept { installed.store(value, std::memory_order_release); }

Observer* observer() noexcept { return installed.load(std::memory_order_acquire); }

}  // namespace nvcr::dcvcuf::diagnostics
