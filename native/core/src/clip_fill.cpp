#include "rudra/core/clip_fill.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include "rudra/core/baseline.hpp"
#include "parallel.hpp"

namespace rudra {
namespace {

// The SDR's own luma weights: the baseline is in the source's primaries.
constexpr float kLuma[3] = {0.2126f, 0.7152f, 0.0722f};
constexpr float kTiny = 1e-8f;
constexpr float kInf = std::numeric_limits<float>::infinity();

float smooth01(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

float sigmoidf(float x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }

float luma_of(const float v[3]) noexcept { return kLuma[0] * v[0] + kLuma[1] * v[1] + kLuma[2] * v[2]; }

// Two-pass chamfer distance (1, sqrt 2) from the seeds (dist 0), carrying the
// seed's label. Pixels never reached keep kInf and label -1.
void chamfer(std::vector<float>& dist, std::vector<int>& label, int h, int w) {
    constexpr float kD = 1.41421356f;
    auto relax = [&](std::size_t i, std::size_t j, float step) {
        const float d = dist[j] + step;
        if (d < dist[i]) {
            dist[i] = d;
            label[i] = label[j];
        }
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            if (x > 0) relax(i, i - 1, 1.0f);
            if (y > 0) {
                relax(i, i - std::size_t(w), 1.0f);
                if (x > 0) relax(i, i - std::size_t(w) - 1, kD);
                if (x + 1 < w) relax(i, i - std::size_t(w) + 1, kD);
            }
        }
    for (int y = h - 1; y >= 0; --y)
        for (int x = w - 1; x >= 0; --x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            if (x + 1 < w) relax(i, i + 1, 1.0f);
            if (y + 1 < h) {
                relax(i, i + std::size_t(w), 1.0f);
                if (x + 1 < w) relax(i, i + std::size_t(w) + 1, kD);
                if (x > 0) relax(i, i + std::size_t(w) - 1, kD);
            }
        }
}

constexpr std::size_t kMaxFillPlanes = 8;

// fn(i) for i in [0, n), split over a few threads. Every use writes only its
// own index, so the result does not depend on the thread count.
template <class F>
void for_each_index(std::size_t n, F&& fn) {
    detail::parallel_chunks(n, detail::worker_count(n), [&](int, std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) fn(i);
    });
}

struct Level {
    int h = 0, w = 0;
    std::vector<std::vector<float>> v;
    std::vector<float> wt;
};

}  // namespace

void fill_missing(std::vector<std::vector<float>>& planes, const std::vector<float>& weight, int height, int width,
                  int relax_iterations) {
    const std::size_t n = std::size_t(height) * std::size_t(width);
    const std::size_t k = planes.size();
    if (k == 0 || n == 0) return;
    assert(k <= kMaxFillPlanes);
    bool any_known = false, any_missing = false;
    for (float wt : weight) (wt > 0.0f ? any_known : any_missing) = true;
    if (!any_missing) return;
    if (!any_known) {
        for (auto& p : planes) std::fill(p.begin(), p.end(), 0.0f);
        return;
    }

    // Push: 2x2 weighted means, level by level, down to one pixel (or until
    // nothing is missing). Level 0 is the caller's planes, read in place.
    std::vector<Level> levels;
    auto push = [k](int fh, int fw, auto value, auto wt_of) {
        Level c;
        c.h = (fh + 1) / 2;
        c.w = (fw + 1) / 2;
        const std::size_t cn = std::size_t(c.h) * std::size_t(c.w);
        c.v.assign(k, std::vector<float>(cn, 0.0f));
        c.wt.assign(cn, 0.0f);
        for_each_index(cn, [&](std::size_t ci) {
                const int y = int(ci / std::size_t(c.w)), x = int(ci % std::size_t(c.w));
                double W = 0.0;
                double acc[kMaxFillPlanes] = {};
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int fy = 2 * y + dy, fx = 2 * x + dx;
                        if (fy >= fh || fx >= fw) continue;
                        const std::size_t fi = std::size_t(fy) * std::size_t(fw) + std::size_t(fx);
                        const double wt = wt_of(fi);
                        if (wt <= 0.0) continue;
                        W += wt;
                        for (std::size_t p = 0; p < k; ++p) acc[p] += wt * value(p, fi);
                    }
                if (W > 0.0)
                    for (std::size_t p = 0; p < k; ++p) c.v[p][ci] = float(acc[p] / W);
                c.wt[ci] = float(std::min(W, 1.0));
            });
        return c;
    };
    levels.push_back(push(height, width, [&](std::size_t p, std::size_t i) { return planes[p][i]; },
                          [&](std::size_t i) { return double(std::clamp(weight[i], 0.0f, 1.0f)); }));
    while (levels.back().h > 1 || levels.back().w > 1) {
        const Level& f = levels.back();
        if (std::all_of(f.wt.begin(), f.wt.end(), [](float v) { return v >= 1.0f; })) break;
        levels.push_back(push(f.h, f.w, [&f](std::size_t p, std::size_t i) { return f.v[p][i]; },
                              [&f](std::size_t i) { return double(f.wt[i]); }));
    }

    // Jacobi sweeps over the listed pixels of a plane set (the rest fixed).
    auto relax = [k](std::vector<std::vector<float>>& v, int h, int w, const std::vector<std::size_t>& miss,
                     int iterations) {
        std::vector<float> next(miss.size());
        for (std::size_t p = 0; p < k; ++p) {
            auto& pl = v[p];
            for (int it = 0; it < iterations; ++it) {
                for_each_index(miss.size(), [&](std::size_t m) {
                    const std::size_t i = miss[m];
                    const int y = int(i / std::size_t(w)), x = int(i % std::size_t(w));
                    float s = 0.0f;
                    int cnt = 0;
                    if (x > 0) { s += pl[i - 1]; ++cnt; }
                    if (x + 1 < w) { s += pl[i + 1]; ++cnt; }
                    if (y > 0) { s += pl[i - std::size_t(w)]; ++cnt; }
                    if (y + 1 < h) { s += pl[i + std::size_t(w)]; ++cnt; }
                    next[m] = cnt ? s / float(cnt) : pl[i];
                });
                for (std::size_t m = 0; m < miss.size(); ++m) pl[miss[m]] = next[m];
            }
        }
    };
    // Pull: each level's missing share from the coarser one, bilinear, then
    // a few sweeps there over what was missing: relaxed coarse to fine, so a
    // handful per level does what hundreds at full size would.
    auto pull = [k](std::vector<std::vector<float>>& v, std::vector<float>& wt, int fh, int fw, const Level& c,
                    std::vector<std::size_t>& miss) {
        miss.clear();
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x) {
                const std::size_t fi = std::size_t(y) * std::size_t(fw) + std::size_t(x);
                const float w0 = std::clamp(wt[fi], 0.0f, 1.0f);
                if (w0 >= 1.0f) continue;
                if (w0 <= 0.0f) miss.push_back(fi);
                const double u = (x + 0.5) * 0.5 - 0.5, vv = (y + 0.5) * 0.5 - 0.5;
                const int x0 = std::clamp(int(std::floor(u)), 0, c.w - 1), y0 = std::clamp(int(std::floor(vv)), 0, c.h - 1);
                const int x1 = std::min(x0 + 1, c.w - 1), y1 = std::min(y0 + 1, c.h - 1);
                const float fx = float(std::clamp(u - std::floor(u), 0.0, 1.0));
                const float fy = float(std::clamp(vv - std::floor(vv), 0.0, 1.0));
                const std::size_t i00 = std::size_t(y0) * std::size_t(c.w) + std::size_t(x0);
                const std::size_t i01 = std::size_t(y0) * std::size_t(c.w) + std::size_t(x1);
                const std::size_t i10 = std::size_t(y1) * std::size_t(c.w) + std::size_t(x0);
                const std::size_t i11 = std::size_t(y1) * std::size_t(c.w) + std::size_t(x1);
                for (std::size_t p = 0; p < k; ++p) {
                    const auto& cv = c.v[p];
                    const float up = (cv[i00] * (1 - fx) + cv[i01] * fx) * (1 - fy) + (cv[i10] * (1 - fx) + cv[i11] * fx) * fy;
                    v[p][fi] = w0 * v[p][fi] + (1.0f - w0) * up;
                }
            }
    };
    std::vector<std::size_t> miss;
    for (int l = int(levels.size()) - 2; l >= 0; --l) {
        Level& f = levels[std::size_t(l)];
        pull(f.v, f.wt, f.h, f.w, levels[std::size_t(l) + 1], miss);
        std::fill(f.wt.begin(), f.wt.end(), 1.0f);
        relax(f.v, f.h, f.w, miss, 4);
    }
    std::vector<float> wt0(weight);
    pull(planes, wt0, height, width, levels[0], miss);
    relax(planes, height, width, miss, relax_iterations);
}

namespace {

// How far from anything clipped the fill looks (px): the glow ring is at most
// 32 px, so a margin past it is all the clean picture the fill ever reads.
constexpr int kMargin = 40;

// Where a crop sits in the frame it came from.
struct FrameRef {
    std::size_t pixels = 0;
    int x0 = 0, y0 = 0, width = 0, height = 0;
};

float clip_weight(float code, float soft, float hard) noexcept {
    code = std::clamp(code, 0.0f, 1.0f);
    return code > soft ? smooth01((code - soft) / (hard - soft)) : 0.0f;
}

ClipFill reconstruct_crop(const SdrImage& sdr, float corpus_ev, const ClipFillOptions& opt, const FrameRef& frame) {
    ClipFill out;
    const int h = sdr.height(), w = sdr.width();
    const std::size_t n = std::size_t(h) * std::size_t(w);
    const PlanarBuffer& s = sdr.buffer();
    const float soft = float(opt.soft_code), hard = float(std::max(opt.clip_code, opt.soft_code + 1e-4));

    std::vector<float> wc[3];
    for (int c = 0; c < 3; ++c) {
        wc[c].resize(n);
        const float* p = s.plane(c);
        for (std::size_t i = 0; i < n; ++i) wc[c][i] = clip_weight(p[i], soft, hard);
    }

    // The work area: within kMargin of anything clipped. Nothing outside it
    // is read, so the baseline and the logs are computed only inside.
    std::vector<float> d_work(n, kInf);
    std::vector<int> unused(n, -1);
    for (std::size_t i = 0; i < n; ++i)
        if (wc[0][i] > 0.0f || wc[1][i] > 0.0f || wc[2][i] > 0.0f) d_work[i] = 0.0f;
    chamfer(d_work, unused, h, w);
    std::vector<std::uint8_t> work(n);
    for (std::size_t i = 0; i < n; ++i) work[i] = d_work[i] <= float(kMargin);

    // The analytic baseline (no curve) and each pixel's class.
    std::vector<float> b[3];
    for (int c = 0; c < 3; ++c) {
        b[c].assign(n, 0.0f);
        const float* p = s.plane(c);
        for_each_index(n, [&](std::size_t i) {
            if (work[i]) b[c][i] = baseline_value(std::clamp(p[i], 0.0f, 1.0f), corpus_ev);
        });
    }
    enum : std::uint8_t { kClean, kPartial, kFull };
    std::vector<std::uint8_t> cls(n);
    std::size_t n_partial = 0, n_full = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // Clean: no channel near white. Full: every channel at or past
        // clip_code, nothing left to go on. Partial: the rest, where at least
        // one channel still says something (weight 1 - w).
        int touched = 0, gone = 0;
        for (int c = 0; c < 3; ++c) {
            touched += wc[c][i] > 0.0f;
            gone += wc[c][i] >= 1.0f;
        }
        cls[i] = touched == 0 ? kClean : gone == 3 ? kFull : kPartial;
        n_partial += cls[i] == kPartial;
        n_full += cls[i] == kFull;
    }
    out.partial_pct = 100.0 * double(n_partial) / double(frame.pixels);
    out.full_pct = 100.0 * double(n_full) / double(frame.pixels);

    // Targets start at the baseline.
    std::vector<float> t[3] = {b[0], b[1], b[2]};

    // Stage 1: chroma (log2 channel over log2 luma) of the clean pixels,
    // carried into everything else; the partly clipped pixels' missing
    // channels from the ones they kept.
    if (n_partial > 0 && opt.partial) {
        std::vector<std::vector<float>> q(3, std::vector<float>(n, 0.0f));
        std::vector<float> wt(n, 0.0f);
        for_each_index(n, [&](std::size_t i) {
            if (cls[i] != kClean || !work[i]) return;
            const float v[3] = {b[0][i], b[1][i], b[2][i]};
            const float ly = std::log2(std::max(luma_of(v), kTiny));
            for (int c = 0; c < 3; ++c) q[std::size_t(c)][i] = std::log2(std::max(v[c], kTiny)) - ly;
            wt[i] = 1.0f;
        });
        fill_missing(q, wt, h, w);
        for_each_index(n, [&](std::size_t i) {
            if (cls[i] != kPartial) return;
            double est = 0.0, kept = 0.0;
            for (int c = 0; c < 3; ++c) {
                const double k = 1.0 - double(wc[c][i]);
                est += k * (std::log2(std::max(b[c][i], kTiny)) - q[std::size_t(c)][i]);
                kept += k;
            }
            est /= kept;   // log2 luma the kept channels say, each by how far below white it is
            for (int c = 0; c < 3; ++c) {
                if (wc[c][i] == 0.0f) continue;
                const double lb = std::log2(std::max(b[c][i], kTiny));
                const double lift = std::clamp(est + q[std::size_t(c)][i] - lb, 0.0, opt.max_partial_ev) * wc[c][i];
                t[c][i] = float(double(b[c][i]) * std::exp2(lift));
            }
        });
    }

    // Stages 2 and 3: the fully clipped regions.
    if (n_full > 0) {
        // Luma and chroma of everything not fully clipped, filled across.
        std::vector<std::vector<float>> f(4, std::vector<float>(n, 0.0f));   // log2 Y, q r, g, b
        std::vector<float> known(n, 0.0f);
        for_each_index(n, [&](std::size_t i) {
            if (cls[i] == kFull || !work[i]) return;
            const float v[3] = {t[0][i], t[1][i], t[2][i]};
            const float ly = std::log2(std::max(luma_of(v), kTiny));
            f[0][i] = ly;
            for (int c = 0; c < 3; ++c) f[std::size_t(c) + 1][i] = std::log2(std::max(v[c], kTiny)) - ly;
            known[i] = 1.0f;
        });
        const bool any_known = n_full < n;
        if (any_known) fill_missing(f, known, h, w);
        else {
            const float ceil = baseline_value(1.0f, corpus_ev);
            std::fill(f[0].begin(), f[0].end(), std::log2(ceil));
        }

        // Regions: 8-connected components of the fully clipped pixels.
        std::vector<int> label(n, -1);
        std::vector<std::size_t> stack;
        int regions = 0;
        for (std::size_t i0 = 0; i0 < n; ++i0) {
            if (cls[i0] != kFull || label[i0] >= 0) continue;
            label[i0] = regions;
            stack.push_back(i0);
            while (!stack.empty()) {
                const std::size_t i = stack.back();
                stack.pop_back();
                const int y = int(i / std::size_t(w)), x = int(i % std::size_t(w));
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = y + dy, xx = x + dx;
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                        const std::size_t j = std::size_t(yy) * std::size_t(w) + std::size_t(xx);
                        if (cls[j] == kFull && label[j] < 0) {
                            label[j] = regions;
                            stack.push_back(j);
                        }
                    }
            }
            ++regions;
        }

        // Inside: distance to the nearest pixel that is not fully clipped.
        std::vector<float> d_in(n, kInf);
        std::vector<int> dummy(n, -1);
        for (std::size_t i = 0; i < n; ++i)
            if (cls[i] != kFull) d_in[i] = 0.0f;
        if (any_known) chamfer(d_in, dummy, h, w);
        // Outside: distance to the nearest region, and which.
        std::vector<float> d_out(n, kInf);
        std::vector<int> near(n, -1);
        for (std::size_t i = 0; i < n; ++i)
            if (cls[i] == kFull) {
                d_out[i] = 0.0f;
                near[i] = label[i];
            }
        chamfer(d_out, near, h, w);

        std::vector<ClipRegion> reg(static_cast<std::size_t>(regions));
        std::vector<double> radius(reg.size(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            if (label[i] < 0) continue;
            ClipRegion& r = reg[std::size_t(label[i])];
            ++r.pixels;
            const double d = any_known ? d_in[i] : double(std::max(h, w));
            if (d > radius[std::size_t(label[i])] || r.pixels == 1) {
                radius[std::size_t(label[i])] = d;
                r.x = int(i % std::size_t(w));
                r.y = int(i / std::size_t(w));
            }
        }
        // The ring outside each region: its edge level, and the slope of the
        // glow (log2 Y against distance) as a least-squares line.
        struct Ring { double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, edge_n = 0, edge_sum = 0; };
        std::vector<Ring> ring(reg.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (cls[i] == kFull || near[i] < 0) continue;
            const std::size_t r = std::size_t(near[i]);
            const double span = std::clamp(radius[r], 3.0, 32.0);
            const double d = d_out[i];
            if (d > span) continue;
            Ring& g = ring[r];
            const double ly = f[0][i];
            if (d <= 1.5) { g.edge_n += 1; g.edge_sum += ly; }
            // The slope from clean pixels only: near white the inverse is
            // steep and 8-bit codes are far apart in it, so a ring through
            // the touched band reads quantisation as glow.
            if (opt.slope_from_clean && cls[i] != kClean) continue;
            g.n += 1; g.sx += d; g.sy += ly; g.sxx += d * d; g.sxy += d * ly;
        }
        // What clipped is at least the ceiling: the level a fill starts from.
        const double ly_ceiling = std::log2(double(baseline_value(1.0f, corpus_ev)));
        std::vector<double> dome(reg.size(), 0.0), edge_ly(reg.size(), 0.0), centre_nits(reg.size(), 0.0);
        for (std::size_t r = 0; r < reg.size(); ++r) {
            ClipRegion& R = reg[r];
            const Ring& g = ring[r];
            R.radius_px = radius[r];
            R.fraction = double(R.pixels) / double(frame.pixels);
            R.large = R.fraction >= opt.large_fraction;
            edge_ly[r] = std::max(g.edge_n > 0 ? g.edge_sum / g.edge_n : double(f[0][std::size_t(R.y) * std::size_t(w) + std::size_t(R.x)]),
                                  ly_ceiling);
            R.edge_nits = std::exp2(edge_ly[r]) * 10000.0;
            // The level the dome stands on at its deepest point: the peak is
            // set against it, so an override's nits are what the centre gets.
            {
                const std::size_t c = std::size_t(R.y) * std::size_t(w) + std::size_t(R.x);
                const float bv[3] = {b[0][c], b[1][c], b[2][c]};
                centre_nits[r] = std::exp2(std::max(double(f[0][c]), double(std::log2(std::max(luma_of(bv), kTiny))))) * 10000.0;
            }
            double slope = 0.0;
            const double var = g.n * g.sxx - g.sx * g.sx;
            if (g.n >= 3 && var > 1e-9) slope = (g.n * g.sxy - g.sx * g.sy) / var;
            R.glow_ev_per_px = std::max(0.0, -slope);
            const double rad = std::max(R.radius_px, 1.0);
            double a = opt.glow_scale * R.glow_ev_per_px * rad * 0.5;   // the dome's slope at the edge matches the glow's
            if (R.large) a = std::clamp(a, 0.0, opt.large_max_ev);
            else {
                const double most = std::max(std::log2(opt.source_max_nits / std::max(centre_nits[r], 1e-6)), 0.0);
                a = std::min(std::max(a, opt.source_min_ev), most);
            }
            dome[r] = a;
        }
        for (const ClipOverride& o : opt.overrides) {
            if (!(o.nits > 0.0)) continue;
            const int x = int(o.x * frame.width) - frame.x0, y = int(o.y * frame.height) - frame.y0;
            if (x < 0 || y < 0 || x >= w || y >= h) continue;
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            if (near[i] < 0 || d_out[i] > 4.0f) continue;   // a click within 4 px of a region picks it
            const std::size_t r = std::size_t(near[i]);
            dome[r] = std::max(0.0, std::log2(o.nits / std::max(centre_nits[r], 1e-6)));
            reg[r].overridden = true;
        }

        for_each_index(n, [&](std::size_t i) {
            if (cls[i] != kFull) return;
            const std::size_t r = std::size_t(label[i]);
            const double rad = std::max(reg[r].radius_px, 1.0);
            const double u = std::clamp((any_known ? double(d_in[i]) : rad) / rad, 0.0, 1.0);
            const float bv[3] = {b[0][i], b[1][i], b[2][i]};
            const double ly = std::max(double(f[0][i]), double(std::log2(std::max(luma_of(bv), kTiny)))) +
                              dome[r] * (2.0 * u - u * u);
            // A source's core runs toward white; a sky keeps its edge's hue.
            // The chroma is renormalised so the luma is the one asked for.
            const double keep = reg[r].large ? 1.0 : 1.0 - 0.5 * u;
            double cq[3], norm = 0.0;
            for (int c = 0; c < 3; ++c) {
                cq[c] = keep * f[std::size_t(c) + 1][i];
                norm += double(kLuma[c]) * std::exp2(cq[c]);
            }
            norm = std::log2(std::max(norm, 1e-12));
            for (int c = 0; c < 3; ++c) {
                const double lb = std::log2(std::max(b[c][i], kTiny));
                const double lt = std::max(ly + cq[c] - norm, lb);   // never below what clipped
                t[c][i] = float(double(b[c][i]) * std::exp2(lt - lb));
            }
        });
        for (std::size_t r = 0; r < reg.size(); ++r) {
            const std::size_t i = std::size_t(reg[r].y) * std::size_t(w) + std::size_t(reg[r].x);
            const float v[3] = {t[0][i], t[1][i], t[2][i]};
            reg[r].peak_nits = double(luma_of(v)) * 10000.0;
        }
        out.regions = std::move(reg);
    }

    out.gain_log2 = PlanarBuffer(3, h, w);
    for (int c = 0; c < 3; ++c) {
        float* g = out.gain_log2.plane(c);
        for_each_index(n, [&](std::size_t i) {
            if (work[i] && t[c][i] > b[c][i]) g[i] = std::log2(std::max(t[c][i], kTiny)) - std::log2(std::max(b[c][i], kTiny));
        });
    }
    return out;
}

}  // namespace

ClipFill reconstruct_clipped(const SdrImage& sdr, float corpus_ev, const ClipFillOptions& opt) {
    const int h = sdr.height(), w = sdr.width();
    if (!opt.enabled || h <= 0 || w <= 0) return {};
    // The box around everything clipped, kMargin wider: the fill runs there.
    const float soft = float(opt.soft_code), hard = float(std::max(opt.clip_code, opt.soft_code + 1e-4));
    const PlanarBuffer& s = sdr.buffer();
    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            if (clip_weight(s.plane(0)[i], soft, hard) > 0.0f || clip_weight(s.plane(1)[i], soft, hard) > 0.0f ||
                clip_weight(s.plane(2)[i], soft, hard) > 0.0f) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    if (x1 < 0) return {};   // nothing reaches soft_code
    x0 = std::max(0, x0 - kMargin);
    y0 = std::max(0, y0 - kMargin);
    x1 = std::min(w - 1, x1 + kMargin);
    y1 = std::min(h - 1, y1 + kMargin);
    const FrameRef frame{std::size_t(h) * std::size_t(w), x0, y0, w, h};
    const bool whole = x0 == 0 && y0 == 0 && x1 == w - 1 && y1 == h - 1;
    if (whole) return reconstruct_crop(sdr, corpus_ev, opt, frame);

    const int ch = y1 - y0 + 1, cw = x1 - x0 + 1;
    ClipFill part = reconstruct_crop(SdrImage(s.crop(y0, x0, ch, cw)), corpus_ev, opt, frame);
    ClipFill out;
    out.partial_pct = part.partial_pct;
    out.full_pct = part.full_pct;
    out.regions = std::move(part.regions);
    for (auto& r : out.regions) {
        r.x += x0;
        r.y += y0;
    }
    out.gain_log2 = PlanarBuffer(3, h, w);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < ch; ++y)
            std::copy_n(part.gain_log2.plane(c) + std::size_t(y) * std::size_t(cw), cw,
                        out.gain_log2.plane(c) + std::size_t(y + y0) * std::size_t(w) + std::size_t(x0));
    return out;
}

Fields clip_fill_fields(const SdrImage& sdr, const ModelConstants& model, const ClipFillOptions& options,
                        ClipFill* detail) {
    const int h = sdr.height(), w = sdr.width();
    Fields f = analytic_fields(h, w);
    ClipFill fill = reconstruct_clipped(sdr, model.corpus_ev, options);
    if (!fill.gain_log2.empty()) {
        const std::size_t n = std::size_t(h) * std::size_t(w);
        const PlanarBuffer& s = sdr.buffer();
        const float ls = model.log_scale;
        float* hl = f.highlight.plane(0);
        for_each_index(n, [&](std::size_t i) {
            float code[3];
            for (int c = 0; c < 3; ++c) code[c] = std::clamp(s.plane(c)[i], 0.0f, 1.0f);
            bool changed = false;
            for (int c = 0; c < 3; ++c) changed = changed || fill.gain_log2.plane(c)[i] > 0.0f;
            if (!changed) return;
            // composite()'s gate under the defaults (mode All, strength 1, an
            // analytic frame's shadow weight of 1), divided back out.
            const float y = 0.2126f * code[0] + 0.7152f * code[1] + 0.0722f * code[2];
            const float gate = std::max({sigmoidf((y - 0.82f) * 24.0f), sigmoidf((0.10f - y) * 24.0f), 1e-6f});
            for (int c = 0; c < 3; ++c) {
                const float g = fill.gain_log2.plane(c)[i];
                if (!(g > 0.0f)) continue;
                const float base = baseline_value(code[c], model.corpus_ev);
                const float target = base * std::exp2(g);
                f.residual.plane(c)[i] = (std::log1p(target * ls) - std::log1p(base * ls)) / gate;
            }
            hl[i] = 1.0f;
        });
    }
    if (detail) *detail = std::move(fill);
    return f;
}

}  // namespace rudra
