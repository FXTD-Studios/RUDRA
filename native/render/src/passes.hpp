#pragma once
// Internal to render/: the uniform blocks of the viewer's shaders (std140) and
// how they are filled, shared by GpuCompositor (the parity path) and
// ViewerWindow (the live viewer), so the two cannot drift apart.

#include <QFile>
#include <rhi/qrhi.h>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/view.hpp"

namespace rudra::detail {

inline QShader load_shader(const QString& name) {
    QFile f(name);
    return f.open(QIODevice::ReadOnly) ? QShader::fromSerialized(f.readAll()) : QShader();
}

// The Composite block of shaders/composite.frag.
struct CompositeUbo {
    float model[4];
    float control[4];
    float counts[4];
    float curve[288];   // 72 vec4: exposure + up to 287 knots (source_curve.hpp needs 1 + 256)
    float bands[32];
    float frame[4];     // x width, y height (3.3: the masks sample by normalised position), z exposure gain,
                        // w 1 when the anchor's target has knots (13 Oct 2026)
    float anchor[256];  // 64 vec4: anchor_target_knots, one per 8-bit code
    float region[12];   // spec rev 2 (5 Nov 2026), three vec4: [0] x on, y band, z highlight_from, w its softness;
                        // [1] x highlight strength, y shadow_below, z its softness, w shadow strength;
                        // [2] x band low, y band high, z band softness
};
static_assert(sizeof(CompositeUbo) == (3 * 4 + 288 + 32 + 4 + 256 + 12) * sizeof(float));
static_assert(288 >= 1 + kSourceCurveKnots);

// The View block of shaders/display.frag.
struct ViewUbo {
    float view[4];
    float extra[4];
    float target[4];
    float pic[12];   // three vec4 rows
    float gfx[12];
    float anchor[4];   // x on, y knee, z softness, w hold (roadmap 3.3)
    float paint[4];    // x the painted mask's channel (< 0 none), yzw its tint (3.3)
    float look[4];     // revision 6: x 2^view_ev, y 1 / gamma, z gamma on, w the SDR light bound
    float ab[4];       // revision 6: x shown, y wipe left, z wipe right (0 model, 1 baseline, 2 sdr), w zebras
};
static_assert(sizeof(ViewUbo) == 52 * sizeof(float));

// The Hold block of shaders/hold.frag (the static hold while playing).
struct HoldUbo {
    float hold[4];   // x retention^frames since the previous one shown, y 1: the shot's first frame
};

inline HoldUbo hold_ubo(double strength, bool reset) {
    HoldUbo u{};
    u.hold[0] = float(std::clamp(strength, 0.0, 1.0));
    u.hold[1] = reset ? 1.0f : 0.0f;
    return u;
}

// The Reduce block of shaders/reduce.frag.
struct ReduceUbo {
    float sizes[4];
};

inline void rows_of(const Mat3& m, float* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r * 4 + c] = float(m[std::size_t(r)][std::size_t(c)]);
}

inline CompositeUbo composite_ubo(const FrameScalars& scalars, const ModelConstants& model,
                                  const CompositeParams& params, int frame_width = 0, int frame_height = 0) {
    CompositeUbo u{};
    u.model[0] = model.log_scale;
    u.model[1] = model.max_hdr;
    u.model[2] = static_cast<float>(std::exp2(-static_cast<double>(model.corpus_ev)) * (203.0 / 10000.0));
    u.model[3] = params.strength;
    u.control[0] = float(int(params.mode));
    u.control[1] = params.preserve_outside ? 1.0f : 0.0f;
    u.control[2] = scalars.shadow_weight;
    u.control[3] = static_cast<float>(params.region_softness_stops);
    // The model's CurveHead plus the source curve, as the CPU composite feeds
    // corrected_baseline, so baselineOf() in the shader is the same function.
    const std::vector<float> curve = baseline_curve_params(scalars, params, model.corpus_ev);
    const std::size_t np = std::min<std::size_t>(curve.size(), 288);
    u.counts[0] = np >= 3 ? float(np - 1) : 0.0f;
    std::copy(curve.begin(), curve.begin() + std::ptrdiff_t(np), u.curve);
    const bool masked = params.masks && !params.masks->empty();
    int bands = 0, index = 0;
    for (const auto& band : params.regions) {
        const int band_index = index++;
        if (band.ev == 0.0 || bands == 8) continue;
        u.bands[bands * 4 + 0] = static_cast<float>(std::log2(band.low_nits));
        u.bands[bands * 4 + 1] = static_cast<float>(std::log2(band.high_nits));
        u.bands[bands * 4 + 2] = static_cast<float>(band.ev);
        u.bands[bands * 4 + 3] = masked && params.masks->has(band_index) ? float(band_index) : -1.0f;
        ++bands;
    }
    u.counts[1] = float(bands);
    u.counts[2] = masked ? 1.0f : 0.0f;
    u.frame[0] = float(std::max(frame_width, 1));
    u.frame[1] = float(std::max(frame_height, 1));
    u.frame[2] = float(exposure_gain(params.exposure_ev));   // 1: no exposure (13 Oct 2026)
    // The anchor's target follows the Source picker (13 Oct 2026); none: the sRGB reading.
    const std::vector<float> knots = anchor_target_knots(params);
    u.frame[3] = knots.size() == 256 ? 1.0f : 0.0f;
    if (knots.size() == 256) std::copy(knots.begin(), knots.end(), u.anchor);
    const ReconstructionRegion& rg = params.region;   // all 0 when off: the shader skips it
    if (rg.enabled) {
        const float region[12] = {1.0f, rg.band ? 1.0f : 0.0f, rg.highlight_from, rg.highlight_softness,
                                  rg.highlight_strength, rg.shadow_below, rg.shadow_softness, rg.shadow_strength,
                                  rg.band_low, rg.band_high, rg.band_softness, 0.0f};
        std::copy(std::begin(region), std::end(region), u.region);
    }
    return u;
}

// The masks texture's bytes at the frame's size (RGBA8, a band per channel,
// 255 for none): the set as it is when the sizes match, else resampled the
// way MaskSet::weight samples, so the GPU reads what the CPU composite does.
inline std::vector<std::uint8_t> masks_rgba8(const CompositeParams& params, int frame_width, int frame_height) {
    const std::size_t n = std::size_t(std::max(frame_width, 1)) * std::size_t(std::max(frame_height, 1));
    if (!params.masks || params.masks->empty()) return std::vector<std::uint8_t>(n * 4, 255);
    const MaskSet& m = *params.masks;
    if (m.width == frame_width && m.height == frame_height) return m.rgba8();
    std::vector<std::uint8_t> out(n * 4, 255);
    for (int y = 0; y < frame_height; ++y)
        for (int x = 0; x < frame_width; ++x) {
            float w[kMaxMaskBands];
            m.weights(x, y, frame_width, frame_height, w);
            for (int b = 0; b < kMaxMaskBands; ++b)
                out[(std::size_t(y) * std::size_t(frame_width) + std::size_t(x)) * 4 + std::size_t(b)] =
                    std::uint8_t(std::lround(std::clamp(w[b], 0.0f, 1.0f) * 255.0f));
        }
    return out;
}

// The uniforms as core/view.cpp rounds them to fp32.
inline ViewUbo view_ubo(const ViewParams& params, int width, bool have_sdr = false) {
    ViewUbo u{};
    u.view[0] = float(int(params.mode));
    u.view[1] = float(10000.0 / std::max(params.display_nits, 1e-3));
    u.view[2] = params.wipe >= 0.0 ? float(std::clamp(params.wipe, 0.0, 1.0)) : -1.0f;
    u.view[3] = float(params.wipe_half_width);
    u.extra[0] = std::log2(1.0f + float(std::max(params.diff_gain, 1.0)));
    u.extra[1] = float(width);
    u.extra[2] = params.show == ViewSource::Baseline ? 1.0f : 0.0f;
    u.extra[3] = params.show_changes ? 1.0f : 0.0f;
    u.target[0] = float(int(params.target.path));
    u.target[1] = float(std::min(params.display_nits, params.target.peak_nits));
    u.target[2] = float(params.target.unit_nits);
    rows_of(rgb_to_rgb_matrix(params.source, params.target.primaries), u.pic);
    rows_of(rgb_to_rgb_matrix(Primaries::Rec709, params.target.primaries), u.gfx);
    u.anchor[0] = params.anchor ? 1.0f : 0.0f;
    u.anchor[1] = float(params.anchor_knee);
    u.anchor[2] = float(params.anchor_softness);
    u.anchor[3] = float(params.anchor_hold);
    const bool painting = params.paint_band >= 0 && params.masks && params.masks->has(params.paint_band);
    u.paint[0] = painting ? float(params.paint_band) : -1.0f;
    for (int k = 0; k < 3; ++k) u.paint[1 + k] = params.paint_tint[std::size_t(k)];
    u.look[0] = float(std::exp2(params.view_ev));
    u.look[1] = float(1.0 / std::clamp(params.view_gamma, 0.2, 5.0));
    u.look[2] = params.view_gamma != 1.0 ? 1.0f : 0.0f;
    u.look[3] = have_sdr ? 1.0f : 0.0f;
    u.ab[0] = float(int(params.show));
    u.ab[1] = float(int(params.wipe_left));
    u.ab[2] = float(int(params.wipe_right));
    u.ab[3] = float((params.clamp_zebra ? 1 : 0) | (params.clip_zebra ? 2 : 0));
    return u;
}



// The masks texture for the display pass, from the view's set (see
// masks_rgba8 above for the composite's).
inline std::vector<std::uint8_t> view_masks_rgba8(const ViewParams& params, int frame_width, int frame_height) {
    CompositeParams cp;
    cp.masks = params.masks;
    return masks_rgba8(cp, frame_width, frame_height);
}

// The viewer packs every frame it shows on the UI thread (4 Nov 2026): the
// pixels split over a few threads, each writing its own range, so the result
// is the same whatever the count. Small images stay on one thread.
template <class F>
inline void for_pixel_ranges(std::size_t n, F&& fn) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t k = std::max<std::size_t>(1, std::min<std::size_t>({std::size_t(hw), 8, n / 65536 + 1}));
    if (k <= 1) {
        fn(std::size_t(0), n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(k - 1);
    for (std::size_t w = 1; w < k; ++w) pool.emplace_back([&, w] { fn(n * w / k, n * (w + 1) / k); });
    fn(std::size_t(0), n / k);
    for (auto& th : pool) th.join();
}

// sdr rgb + shadow in a, and residual rgb + highlight in a: the two inputs
// composite.frag fetches.
inline void interleave_inputs(const SdrImage& sdr, const Fields& fields, std::vector<float>& a, std::vector<float>& b) {
    const std::size_t n = std::size_t(sdr.width()) * std::size_t(sdr.height());
    a.resize(n * 4);   // every value is written below
    b.resize(n * 4);
    const float *s0 = sdr.buffer().plane(0), *s1 = sdr.buffer().plane(1), *s2 = sdr.buffer().plane(2);
    const float *r0 = fields.residual.plane(0), *r1 = fields.residual.plane(1), *r2 = fields.residual.plane(2);
    const float *sh = fields.shadow.plane(0), *hi = fields.highlight.plane(0);
    float *pa = a.data(), *pb = b.data();
    for_pixel_ranges(n, [=](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            float* qa = pa + i * 4;
            float* qb = pb + i * 4;
            qa[0] = s0[i], qa[1] = s1[i], qa[2] = s2[i], qa[3] = sh[i];
            qb[0] = r0[i], qb[1] = r1[i], qb[2] = r2[i], qb[3] = hi[i];
        }
    });
}

// The baseline as the display pass reads it: rgb, and the SDR's max code in
// alpha for the Invented layer (core/view.hpp sdr_max_codes; 0.5 without an
// SDR of the same size).
inline std::vector<float> baseline_rgba(const PlanarBuffer& rgb, const SdrImage* sdr) {
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 4);
    const bool have = sdr && sdr->width() == rgb.width() && sdr->height() == rgb.height();
    const float *c0 = rgb.plane(0), *c1 = rgb.plane(1), *c2 = rgb.plane(2);
    const float* m0 = have ? sdr->buffer().plane(0) : nullptr;
    const float* m1 = have ? sdr->buffer().plane(1) : nullptr;
    const float* m2 = have ? sdr->buffer().plane(2) : nullptr;
    float* po = out.data();
    for_pixel_ranges(n, [=](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            float* q = po + i * 4;
            q[0] = c0[i], q[1] = c1[i], q[2] = c2[i];
            q[3] = m0 ? std::max(std::max(m0[i], m1[i]), m2[i]) : 0.5f;
        }
    });
    return out;
}

// The same with the codes given (a precomposited picture's own); empty: as above with no SDR.
inline std::vector<float> baseline_rgba(const PlanarBuffer& rgb, const std::vector<float>& codes) {
    if (codes.size() != rgb.plane_size()) return baseline_rgba(rgb, nullptr);
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 4);
    const float *c0 = rgb.plane(0), *c1 = rgb.plane(1), *c2 = rgb.plane(2), *m = codes.data();
    float* po = out.data();
    for_pixel_ranges(n, [=](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            float* q = po + i * 4;
            q[0] = c0[i], q[1] = c1[i], q[2] = c2[i], q[3] = m[i];
        }
    });
    return out;
}

// A composite target as an RGBA32F upload: rgb, alpha 1.
inline std::vector<float> rgba_of(const PlanarBuffer& rgb) {
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 4, 1.0f);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) out[i * 4 + std::size_t(c)] = rgb.plane(c)[i];
    return out;
}

// The model target as the composite shader writes it: rgb, alpha the SDR's
// Rec.2020 luma of its linearised codes (the anchor's target, roadmap 3.3);
// alpha 1 without an SDR.
inline std::vector<float> model_rgba(const PlanarBuffer& rgb, const SdrImage* sdr,
                                     std::span<const float> anchor_knots = {}) {
    std::vector<float> out = rgba_of(rgb);
    if (!sdr) return out;
    const PlanarBuffer& s = sdr->buffer();
    const std::size_t n = rgb.plane_size();
    if (anchor_knots.empty()) {
        for (std::size_t i = 0; i < n; ++i)
            out[i * 4 + 3] = 0.2627f * srgb_to_linear(s.plane(0)[i]) + 0.6780f * srgb_to_linear(s.plane(1)[i]) +
                             0.0593f * srgb_to_linear(s.plane(2)[i]);
        return out;
    }
    // The anchor's target follows the Source picker (13 Oct 2026): render_view's light.
    for (std::size_t i = 0; i < n; ++i) {
        float t = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float v = s.plane(k)[i];
            t += kRec2020Luma[std::size_t(k)] * (srgb_to_linear(v) * std::exp2(anchor_target_log2(v, anchor_knots)));
        }
        out[i * 4 + 3] = t;
    }
    return out;
}

}  // namespace rudra::detail
