#include "rudra/core/fields.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace rudra {

PlanarBuffer resize_bilinear(const PlanarBuffer& in, int height, int width) {
    const int ih = in.height(), iw = in.width();
    if (ih == height && iw == width) return in;
    PlanarBuffer out(in.channels(), height, width);
    // Source coordinates of each output row and column, on pixel centres.
    struct Tap {
        int a, b;
        float t;
    };
    auto taps = [](int n_out, int n_in) {
        std::vector<Tap> v(static_cast<std::size_t>(n_out));
        const double scale = double(n_in) / double(n_out);
        for (int i = 0; i < n_out; ++i) {
            const double x = std::clamp((i + 0.5) * scale - 0.5, 0.0, double(n_in - 1));
            const int a = int(std::floor(x));
            const int b = std::min(a + 1, n_in - 1);
            v[std::size_t(i)] = {a, b, float(x - a)};
        }
        return v;
    };
    const auto ty = taps(height, ih), tx = taps(width, iw);
    for (int c = 0; c < in.channels(); ++c) {
        const float* src = in.plane(c);
        float* dst = out.plane(c);
        for (int y = 0; y < height; ++y) {
            const Tap& r = ty[std::size_t(y)];
            const float* r0 = src + std::size_t(r.a) * std::size_t(iw);
            const float* r1 = src + std::size_t(r.b) * std::size_t(iw);
            float* row = dst + std::size_t(y) * std::size_t(width);
            for (int x = 0; x < width; ++x) {
                const Tap& k = tx[std::size_t(x)];
                const float top = r0[k.a] + (r0[k.b] - r0[k.a]) * k.t;
                const float bot = r1[k.a] + (r1[k.b] - r1[k.a]) * k.t;
                row[x] = top + (bot - top) * r.t;
            }
        }
    }
    return out;
}

Fields resize_fields(const Fields& in, int height, int width) {
    return {resize_bilinear(in.residual, height, width), resize_bilinear(in.highlight, height, width),
            resize_bilinear(in.shadow, height, width)};
}

}  // namespace rudra
