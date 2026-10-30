#include "rudra/core/timecode.hpp"

#include <cmath>
#include <cstdio>

namespace rudra {

int timecode_base(double fps) noexcept {
    if (!(fps > 0)) return 24;
    return static_cast<int>(std::lround(fps));
}

bool drop_frame_rate(double fps) noexcept {
    return std::abs(fps - 30000.0 / 1001.0) < 1e-3 || std::abs(fps - 60000.0 / 1001.0) < 1e-3;
}

std::optional<Timecode> parse_timecode(std::string_view t) {
    // HH:MM:SS[:;.,]FF, two digits each (hours may be one or two).
    int part[4] = {0, 0, 0, 0};
    int idx = 0, digits = 0;
    Timecode tc;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const char c = t[i];
        if (c >= '0' && c <= '9') {
            if (++digits > 3) return std::nullopt;
            part[idx] = part[idx] * 10 + (c - '0');
        } else if (c == ':' || c == ';' || c == '.' || c == ',') {
            if (digits == 0 || idx == 3) return std::nullopt;
            if (idx == 2 && c != ':') tc.drop = true;
            else if (c != ':') return std::nullopt;
            ++idx;
            digits = 0;
        } else {
            return std::nullopt;
        }
    }
    if (idx != 3 || digits == 0) return std::nullopt;
    tc.hours = part[0];
    tc.minutes = part[1];
    tc.seconds = part[2];
    tc.frames = part[3];
    if (tc.hours > 23 || tc.minutes > 59 || tc.seconds > 59 || tc.frames > 119) return std::nullopt;
    return tc;
}

std::string format_timecode(const Timecode& tc) {
    char b[32];
    std::snprintf(b, sizeof b, "%02d:%02d:%02d%c%02d", tc.hours, tc.minutes, tc.seconds, tc.drop ? ';' : ':', tc.frames);
    return b;
}

std::optional<long long> timecode_to_frames(const Timecode& tc, int base) {
    if (base <= 0 || tc.frames >= base) return std::nullopt;
    const long long total_minutes = 60LL * tc.hours + tc.minutes;
    long long n = (total_minutes * 60 + tc.seconds) * base + tc.frames;
    if (tc.drop) {
        if (base != 30 && base != 60) return std::nullopt;
        const int dropped = base / 15;   // 2 at 30, 4 at 60
        if (tc.seconds == 0 && tc.minutes % 10 != 0 && tc.frames < dropped) return std::nullopt;   // no such label
        n -= dropped * (total_minutes - total_minutes / 10);
    }
    return n;
}

Timecode frames_to_timecode(long long n, int base, bool drop) {
    if (base <= 0) base = 24;
    const long long day = 24LL * 3600 * base - (drop && (base == 30 || base == 60) ? (base / 15) * (24LL * 60 - 24 * 6) : 0);
    n %= day;
    if (n < 0) n += day;
    Timecode tc;
    tc.drop = drop && (base == 30 || base == 60);
    if (tc.drop) {
        const long long dropped = base / 15;
        const long long per10 = 10LL * 60 * base - 9 * dropped;   // frames in ten minutes
        const long long per1 = 60LL * base - dropped;             // frames in a minute that drops
        const long long d = n / per10, m = n % per10;
        n += 9 * dropped * d + (m > dropped ? dropped * ((m - dropped) / per1) : 0);
    }
    tc.frames = int(n % base);
    tc.seconds = int((n / base) % 60);
    tc.minutes = int((n / (60LL * base)) % 60);
    tc.hours = int(n / (3600LL * base));
    return tc;
}

std::string timecode_at(std::string_view start, long long offset, double fps) {
    auto tc = parse_timecode(start);
    if (!tc) return {};
    const int base = timecode_base(fps);
    const bool drop = tc->drop && (base == 30 || base == 60);
    tc->drop = drop;
    auto n = timecode_to_frames(*tc, base);
    if (!n) return {};
    return format_timecode(frames_to_timecode(*n + offset, base, drop));
}

std::array<std::uint32_t, 2> exr_timecode(const Timecode& tc) {
    auto bcd = [](int v) { return std::uint32_t((v / 10) << 4 | (v % 10)); };
    std::uint32_t t = 0;
    t |= bcd(tc.frames % 100) & 0x3f;           // bits 0-5: frames (units, tens)
    if (tc.drop) t |= 1u << 6;                   // bit 6: drop frame
    t |= (bcd(tc.seconds) & 0x7f) << 8;          // bits 8-14: seconds
    t |= (bcd(tc.minutes) & 0x7f) << 16;         // bits 16-22: minutes
    t |= (bcd(tc.hours) & 0x3f) << 24;           // bits 24-29: hours
    return {t, 0u};
}

}  // namespace rudra
