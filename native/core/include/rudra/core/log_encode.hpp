#pragma once
// Scene-referred log encodings for ProRes delivery (9 Oct 2026):
//   ACEScct  Academy S-2016-001, in ACES AP1 primaries.
//   LogC4    ARRI LogC4 Specification (2022), in ARRI Wide Gamut 4.
// The input is RUDRA's scene-linear master (Rec.2020, diffuse white = 1.0,
// 18% grey = 0.18), the same pixels the ACES EXR carries, before any peak or
// knee. Native only: rudra/delivery/profiles.py has no log formats.

#include <cstdint>
#include <optional>
#include <string_view>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

enum class LogCurve : std::uint8_t { AcesCct, LogC4 };

double acescct_encode(double linear) noexcept;
double acescct_decode(double code) noexcept;
double logc4_encode(double linear) noexcept;
double logc4_decode(double code) noexcept;

const char* log_curve_name(LogCurve c) noexcept;      // "acescct", "logc4"
const char* log_curve_label(LogCurve c) noexcept;     // "ACEScct", "ARRI LogC4"
Primaries log_curve_primaries(LogCurve c) noexcept;   // Ap1, Awg4
const char* log_gamut_label(LogCurve c) noexcept;     // "ACES AP1", "ARRI Wide Gamut 4"
std::optional<LogCurve> log_curve_from_name(std::string_view name) noexcept;

// Network units (1.0 = 10 000 nits, Rec.2020 linear) -> code values in [0, 1]
// in the curve's own gamut. Code values outside [0, 1] are clamped (ACEScct
// tops out at 222.9 x diffuse white, LogC4 at 469.8 x).
PlanarBuffer encode_log(const PlanarBuffer& rgb_normalized, LogCurve curve);

}  // namespace rudra
