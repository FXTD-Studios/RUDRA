#include <rudra/core/resample.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <thread>
#include <vector>

namespace rudra {

bool valid_output_size(const std::string& choice) {
    return choice == "source" || choice == "2x" || choice == "uhd" || choice == "hd";
}

namespace {
int even(double v) { return std::max(2, 2 * int(std::lround(v / 2.0 - 1e-9))); }
}  // namespace

namespace {
OutputSize asked_size(const std::string& choice, int width, int height) {
    if (width <= 0 || height <= 0) return {width, height};
    if (choice == "2x") return {2 * width, 2 * height};
    if (choice == "uhd" || choice == "hd") {
        const int w = choice == "uhd" ? 3840 : 1920;
        return {w, even(double(height) * w / width)};
    }
    return {width, height};
}
}  // namespace

bool output_size_fits(const std::string& choice, int width, int height) {
    if (choice == "source" || width <= 0 || height <= 0) return true;
    const OutputSize o = asked_size(choice, width, height);
    return o.width <= kMaxOutputSide && o.height <= kMaxOutputSide;
}

OutputSize output_size(const std::string& choice, int width, int height) {
    if (!output_size_fits(choice, width, height)) return {width, height};
    return asked_size(choice, width, height);
}

std::string output_size_note(const std::string& choice, int src_width, int src_height) {
    const OutputSize o = output_size(choice, src_width, src_height);
    const bool up = o.width > src_width;
    const std::string label = choice == "2x" ? "2x" : choice == "uhd" ? "UHD" : choice == "hd" ? "HD" : "source";
    return std::to_string(o.width) + " x " + std::to_string(o.height) + " from " + std::to_string(src_width) + " x " +
           std::to_string(src_height) + " (" + label + ", " +
           (up ? "Lanczos 3 with anti-ringing" : o.width == src_width ? "unchanged" : "tent, no negative weights") +
           "; a resize, nothing invented)";
}

namespace {

// One output sample along an axis: the taps it reads and, for an upscale, the two source
// samples it is held between.
struct Taps {
    int first = 0;
    std::vector<double> w;
    int lo = -1, hi = -1;
};

double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double p = std::numbers::pi * x;
    return std::sin(p) / p;
}

std::vector<Taps> plan_axis(int src, int dst) {
    std::vector<Taps> plan(static_cast<std::size_t>(dst));
    const double scale = double(src) / double(dst);
    auto clampi = [src](int i) { return std::clamp(i, 0, src - 1); };
    for (int o = 0; o < dst; ++o) {
        const double c = (o + 0.5) * scale - 0.5;
        Taps& t = plan[std::size_t(o)];
        std::vector<std::pair<int, double>> raw;
        if (scale <= 1.0) {   // upscale: Lanczos 3, held between the two nearest
            const int f = int(std::floor(c));
            for (int i = f - 2; i <= f + 3; ++i) {
                const double d = c - i;
                if (std::abs(d) < 3.0) raw.emplace_back(i, sinc(d) * sinc(d / 3.0));
            }
            t.lo = clampi(f);
            t.hi = clampi(f + 1);
        } else {   // downscale: a tent the width of the step
            const int a = int(std::ceil(c - scale)), b = int(std::floor(c + scale));
            for (int i = a; i <= b; ++i) {
                const double wgt = 1.0 - std::abs(i - c) / scale;
                if (wgt > 0.0) raw.emplace_back(i, wgt);
            }
            if (raw.empty()) raw.emplace_back(int(std::lround(c)), 1.0);
        }
        // Edges repeat the last sample: fold each tap onto its clamped index.
        int lo_i = src, hi_i = -1;
        for (auto& [i, _] : raw) {
            i = clampi(i);
            lo_i = std::min(lo_i, i);
            hi_i = std::max(hi_i, i);
        }
        t.first = lo_i;
        t.w.assign(std::size_t(hi_i - lo_i + 1), 0.0);
        double sum = 0.0;
        for (const auto& [i, wgt] : raw) {
            t.w[std::size_t(i - lo_i)] += wgt;
            sum += wgt;
        }
        for (double& wgt : t.w) wgt /= sum;
    }
    return plan;
}

template <class F>
void rows_in_parallel(int rows, F&& f) {
    const int n = std::clamp(int(std::thread::hardware_concurrency()), 1, 32);
    if (n == 1 || rows < 64) {
        f(0, rows);
        return;
    }
    std::vector<std::thread> pool;
    const int step = (rows + n - 1) / n;
    for (int a = 0; a < rows; a += step) pool.emplace_back([&f, a, b = std::min(rows, a + step)] { f(a, b); });
    for (auto& t : pool) t.join();
}

template <class T>
T apply(const Taps& t, const T* in, std::ptrdiff_t stride) {
    double acc = 0.0;
    for (std::size_t k = 0; k < t.w.size(); ++k) acc += t.w[k] * double(in[(t.first + std::ptrdiff_t(k)) * stride]);
    if (t.lo >= 0) {
        const double a = double(in[t.lo * stride]), b = double(in[t.hi * stride]);
        acc = std::clamp(acc, std::min(a, b), std::max(a, b));
    }
    return T(acc);
}

// One plane, h x w -> oh x ow: across the rows, then down the columns.
template <class T>
void resample_plane(const T* in, int h, int w, T* out, int oh, int ow) {
    std::vector<T> mid;
    const T* across = in;
    if (ow != w) {
        const auto px = plan_axis(w, ow);
        mid.resize(std::size_t(h) * ow);
        rows_in_parallel(h, [&](int a, int b) {
            for (int y = a; y < b; ++y)
                for (int x = 0; x < ow; ++x)
                    mid[std::size_t(y) * ow + x] = apply(px[std::size_t(x)], in + std::size_t(y) * w, 1);
        });
        across = mid.data();
    }
    if (oh == h) {
        std::copy_n(across, std::size_t(h) * ow, out);
        return;
    }
    const auto py = plan_axis(h, oh);
    rows_in_parallel(oh, [&](int a, int b) {
        for (int y = a; y < b; ++y)
            for (int x = 0; x < ow; ++x) out[std::size_t(y) * ow + x] = apply(py[std::size_t(y)], across + x, ow);
    });
}

}  // namespace

PlanarBuffer resample_hdr(const PlanarBuffer& x, int height, int width) {
    if (x.empty() || (height == x.height() && width == x.width())) return x;
    PlanarBuffer out(x.channels(), height, width);
    for (int c = 0; c < x.channels(); ++c) resample_plane(x.plane(c), x.height(), x.width(), out.plane(c), height, width);
    return out;
}

NitsFrame resample_hdr(const NitsFrame& x, int height, int width) {
    if (x.plane_size() == 0 || (height == x.height() && width == x.width())) return x;
    NitsFrame out(height, width);
    for (int c = 0; c < 3; ++c) resample_plane(x.plane(c), x.height(), x.width(), out.plane(c), height, width);
    return out;
}

}  // namespace rudra
