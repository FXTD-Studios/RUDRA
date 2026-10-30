#pragma once
// SMPTE timecode (28 Oct 2026): what a colorist conforms by. A movie's start
// timecode (its tmcd track or stream tag) carried to every frame of a master,
// non-drop "HH:MM:SS:FF" or drop-frame "HH:MM:SS;FF" (29.97 and 59.94 only),
// and the OpenEXR timeCode attribute's packing (SMPTE 12M BCD, as OpenEXR's
// TimeCode class packs it for TV60).

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rudra {

struct Timecode {
    int hours = 0, minutes = 0, seconds = 0, frames = 0;
    bool drop = false;   // written with ';' (or '.') before the frames
};

// The frame rate a timecode counts in: 23.976 -> 24, 29.97 -> 30, 59.94 -> 60.
int timecode_base(double fps) noexcept;
// Whether drop-frame counting applies at this rate (29.97 or 59.94).
bool drop_frame_rate(double fps) noexcept;

// "01:00:00:00", "00:59:59;29" (also '.' or ',' before the frames); nullopt if malformed.
std::optional<Timecode> parse_timecode(std::string_view text);
std::string format_timecode(const Timecode& tc);

// Frame count since 00:00:00:00 at `base` frames a second (drop-frame skips
// frame numbers 0 and 1, or 0-3 at 60, every minute except each tenth).
std::optional<long long> timecode_to_frames(const Timecode& tc, int base);
Timecode frames_to_timecode(long long frames, int base, bool drop);

// `start` advanced by `offset` frames at `fps`; "" when start is not a timecode.
std::string timecode_at(std::string_view start, long long offset, double fps);

// OpenEXR timeCode: timeAndFlags (BCD, drop flag in bit 6) and userData (0).
std::array<std::uint32_t, 2> exr_timecode(const Timecode& tc);

}  // namespace rudra
