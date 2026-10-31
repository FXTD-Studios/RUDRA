#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace rudra {
// Shared by storage and rendering; changing the reconstruction requires a new
// identifier. Unknown values never select the default policy.
enum class HeadroomPolicy : std::uint8_t { Disabled, Bt1886V1 };
inline constexpr std::string_view headroom_policy_id(HeadroomPolicy policy) noexcept {
    switch (policy) {
        case HeadroomPolicy::Disabled: return "disabled";
        case HeadroomPolicy::Bt1886V1: return "bt1886-headroom-v1";
    }
    return {};
}
inline constexpr std::optional<HeadroomPolicy> parse_headroom_policy(std::string_view id) noexcept {
    if (id == "disabled") return HeadroomPolicy::Disabled;
    if (id == "bt1886-headroom-v1") return HeadroomPolicy::Bt1886V1;
    return std::nullopt;
}
}  // namespace rudra
