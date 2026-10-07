#include "rudra/core/masks.hpp"

#include <algorithm>
#include <cmath>

#include "rudra/platform/png8.hpp"

namespace rudra {

bool MaskSet::empty() const noexcept {
    for (const auto& p : planes)
        if (p) return false;
    return true;
}

std::vector<int> MaskSet::bands() const {
    std::vector<int> out;
    for (int b = 0; b < kMaxMaskBands; ++b)
        if (planes[std::size_t(b)]) out.push_back(b);
    return out;
}

double MaskSet::coverage(int band) const noexcept {
    if (!has(band)) return 0.0;
    const auto& p = *planes[std::size_t(band)];
    if (p.empty()) return 0.0;
    double s = 0.0;
    for (std::uint8_t v : p) s += v;
    return s / (255.0 * double(p.size()));
}

float MaskSet::weight(int band, int x, int y, int frame_w, int frame_h) const noexcept {
    if (!has(band) || width <= 0 || height <= 0) return 1.0f;
    const auto& p = *planes[std::size_t(band)];
    if (frame_w == width && frame_h == height) return float(p[std::size_t(y) * std::size_t(width) + std::size_t(x)]) / 255.0f;
    // Bilinear on the plane's pixel centres: the frame pixel's centre mapped
    // into the plane, clamped at the edges, as a GPU sampler with
    // clamp-to-edge does.
    const double u = (double(x) + 0.5) * double(width) / double(frame_w) - 0.5;
    const double v = (double(y) + 0.5) * double(height) / double(frame_h) - 0.5;
    const int x0 = std::clamp(int(std::floor(u)), 0, width - 1), y0 = std::clamp(int(std::floor(v)), 0, height - 1);
    const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const double fx = std::clamp(u - std::floor(u), 0.0, 1.0), fy = std::clamp(v - std::floor(v), 0.0, 1.0);
    auto at = [&](int xx, int yy) { return double(p[std::size_t(yy) * std::size_t(width) + std::size_t(xx)]); };
    const double top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    const double bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    // Rounded to a code: the GPU reads the resampled mask as RGBA8, so the
    // two agree to the bit.
    return float(std::lround(top * (1.0 - fy) + bot * fy)) / 255.0f;
}

void MaskSet::weights(int x, int y, int frame_w, int frame_h, float out[kMaxMaskBands]) const noexcept {
    for (int b = 0; b < kMaxMaskBands; ++b) out[b] = weight(b, x, y, frame_w, frame_h);
}

std::vector<std::uint8_t> MaskSet::rgba8() const {
    const std::size_t n = std::size_t(std::max(width, 0)) * std::size_t(std::max(height, 0));
    std::vector<std::uint8_t> out(n * 4, 255);
    for (int b = 0; b < kMaxMaskBands; ++b) {
        if (!planes[std::size_t(b)] || planes[std::size_t(b)]->size() != n) continue;
        const auto& p = *planes[std::size_t(b)];
        for (std::size_t i = 0; i < n; ++i) out[i * 4 + std::size_t(b)] = p[i];
    }
    return out;
}

bool MaskSet::operator==(const MaskSet& o) const {
    if (width != o.width || height != o.height) return false;
    for (int b = 0; b < kMaxMaskBands; ++b) {
        const auto &a = planes[std::size_t(b)], &c = o.planes[std::size_t(b)];
        if (a == c) continue;
        if (!a || !c || *a != *c) return false;
    }
    return true;
}

std::shared_ptr<MaskPlane> empty_plane(int width, int height) {
    return std::make_shared<MaskPlane>(std::size_t(std::max(width, 0)) * std::size_t(std::max(height, 0)), std::uint8_t(0));
}

namespace {

void stamp(MaskPlane& m, int w, int h, double x, double y, const Brush& brush) {
    const double r = std::max(brush.size_px, 1.0) * 0.5;
    const double hard = 1.0 - std::clamp(brush.softness, 0.0, 1.0);
    const double flow = std::clamp(brush.flow, 0.0, 1.0);
    const int x0 = std::max(0, int(std::floor(x - r))), x1 = std::min(w - 1, int(std::ceil(x + r)));
    const int y0 = std::max(0, int(std::floor(y - r))), y1 = std::min(h - 1, int(std::ceil(y + r)));
    for (int yy = y0; yy <= y1; ++yy)
        for (int xx = x0; xx <= x1; ++xx) {
            const double d = std::hypot(double(xx) - x, double(yy) - y) / r;
            if (d > 1.0) continue;
            const double e = hard >= 1.0 ? 1.0 : std::min(1.0, (1.0 - d) / (1.0 - hard + 1e-6));
            const double a = e * e * (3.0 - 2.0 * e) * flow;
            std::uint8_t& v = m[std::size_t(yy) * std::size_t(w) + std::size_t(xx)];
            const double cur = v;
            const double next = brush.erase ? cur - a * cur : cur + a * (255.0 - cur);
            v = std::uint8_t(std::clamp(std::lround(next), 0L, 255L));
        }
}

}  // namespace

void paint_stroke(MaskPlane& plane, int width, int height, double x0, double y0, double x1, double y1,
                  const Brush& brush) {
    if (width <= 0 || height <= 0 || plane.size() != std::size_t(width) * std::size_t(height)) return;
    const double step = std::max(brush.size_px, 1.0) * 0.15;
    const int n = int(std::ceil(std::hypot(x1 - x0, y1 - y0) / step));   // 0: one stamp
    for (int k = 0; k <= n; ++k) {
        const double t = n ? double(k) / double(n) : 0.0;
        stamp(plane, width, height, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, brush);
    }
}

void invert_plane(MaskPlane& plane) {
    for (auto& v : plane) v = std::uint8_t(255 - v);
}

Result<void> write_mask_set(const std::filesystem::path& path, const MaskSet& set) {
    if (set.empty() || set.width <= 0 || set.height <= 0)
        return make_error(ErrorCode::InvalidArgument, "There is no mask to write.", path.string());
    Png8 img;
    img.width = set.width;
    img.height = set.height;
    img.channels = 4;
    img.data = set.rgba8();
    return write_png8(path, img);
}

Result<MaskSet> read_mask_set(const std::filesystem::path& path) {
    auto img = read_png8(path);
    if (!img) return img.error();
    MaskSet set;
    set.width = img->width;
    set.height = img->height;
    const std::size_t n = std::size_t(img->width) * std::size_t(img->height);
    const int ch = img->channels;
    // Grey (1 or 2 channels): band 0 from the grey; RGB/RGBA: a band per channel.
    const int bands = ch <= 2 ? 1 : ch;
    for (int b = 0; b < std::min(bands, kMaxMaskBands); ++b) {
        auto plane = std::make_shared<MaskPlane>(n);
        bool all_on = true;
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint8_t v = img->data[i * std::size_t(ch) + std::size_t(b)];
            (*plane)[i] = v;
            if (v != 255) all_on = false;
        }
        if (!all_on) set.planes[std::size_t(b)] = std::move(plane);
    }
    return set;
}

}  // namespace rudra
