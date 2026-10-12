#include "rudra/video/predictor.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

#include "rudra/infer/tiler.hpp"

namespace rudra {

namespace {
// The frame's residual scale and curve held across the shot as the shadow
// weight is (rudra/video.py Predictor.predict, 10 Oct 2026). A model without
// those heads gives constants, and a constant holds at itself.
void hold_frame_values(FrameScalars& s, ShadowSmoother& smoother) {
    s.residual_scale = static_cast<float>(smoother.hold("residual_scale", static_cast<double>(s.residual_scale)));
    if (!s.curve_params.empty()) {
        const std::vector<double> held =
            smoother.hold("curve", std::vector<double>(s.curve_params.begin(), s.curve_params.end()));
        for (std::size_t i = 0; i < held.size(); ++i) s.curve_params[i] = static_cast<float>(held[i]);
    }
}
}  // namespace

Result<NetworkLinearImage> VideoPredictor::forward(const SdrImage& tile, const FrameScalars& scalars) {
    auto fields = backend_->tile_pass(tile, scalars);
    if (!fields) return fields.error();
    CompositeParams p;   // recovery "all", strength 1, preserve_outside, no grade
    return composite(tile, *fields, scalars, model_, p);
}

Result<std::pair<SdrImage, PlanarBuffer>> VideoPredictor::prepare(const SdrImage& rgb, const std::string& transfer,
                                                                   const std::string& primaries) const {
    auto canon = canonicalize_sdr(rgb.buffer(), transfer, "full");   // the decoder already made it full range
    if (!canon) return canon.error();
    PlanarBuffer x = std::move(*canon);
    if (primaries != "rec2020") {
        if (primaries != "rec709") return make_error(ErrorCode::InvalidArgument, "Unsupported primaries: " + primaries);
        x = srgb_code_to_rec2020(x, Primaries::Rec709);
    }
    PlanarBuffer thumb = area_resize(x, kCutThumbHeight, kCutThumbWidth);
    return std::make_pair(SdrImage(std::move(x)), std::move(thumb));
}

Result<VideoPrediction> VideoPredictor::predict_graded(const SdrImage& rgb, const std::string& transfer,
                                                       const std::string& primaries, ShadowSmoother& smoother,
                                                       const MasterRequest& q) {
    auto prep = prepare(rgb, transfer, primaries);
    if (!prep) return prep.error();
    const SdrImage& frame = prep->first;
    VideoPrediction out;
    Fields fields;
    FrameScalars scalars;
    ModelConstants model = kAnalyticConstants;
    if (q.analytic) {
        std::tie(out.shadow_weight, out.cut) = smoother.update(prep->second, 1.0);   // the cuts, for the report
        out.raw_weight = 1.0;
        out.shadow_weight = 1.0;
        fields = analytic_fields_for(frame, q);   // with the clip fill unless the request turns it off
    } else {
        if (!backend_) return make_error(ErrorCode::InvalidArgument, "This export runs the model, and no model is open");
        auto s = backend_->frame_pass(frame);
        if (!s) return s.error();
        out.raw_weight = static_cast<double>(s->shadow_weight);
        std::tie(out.shadow_weight, out.cut) = smoother.update(prep->second, out.raw_weight);
        scalars = *s;
        scalars.shadow_weight = static_cast<float>(out.shadow_weight);
        hold_frame_values(scalars, smoother);
        auto f = infer_fields(*backend_, frame, scalars, TileConfig{tile_size_, overlap_});
        if (!f) return f.error();
        fields = std::move(*f);
        model = model_;
    }
    // The anchor's hold, the one whole-frame number of the grade, held across
    // the shot in log2 (10 Oct 2026): it is a median and moves with content.
    auto settle = [&smoother](double hold) {
        if (!(hold > 0.0) || !std::isfinite(hold) || smoother.retention() == 0.0) return hold;   // off: exactly its own
        return std::exp2(smoother.hold("anchor_hold", std::log2(hold)));
    };
    auto g = grade_master(frame, fields, scalars, model, q, settle);
    if (!g) return g.error();
    PlanarBuffer hdr(3, frame.height(), frame.width());
    const auto in = g->nits.span();
    auto o = hdr.span();
    for (std::size_t i = 0; i < in.size(); ++i) o[i] = static_cast<float>(in[i] / 10000.0);
    out.hdr = NetworkLinearImage(std::move(hdr));
    return out;
}

Result<VideoPrediction> VideoPredictor::predict(const SdrImage& rgb, const std::string& transfer,
                                                const std::string& primaries, ShadowSmoother& smoother) {
    if (!backend_) return make_error(ErrorCode::InvalidArgument, "This export runs the model, and no model is open");
    auto prep = prepare(rgb, transfer, primaries);
    if (!prep) return prep.error();
    const PlanarBuffer& thumb = prep->second;
    const SdrImage& frame = prep->first;
    auto scalars = backend_->frame_pass(frame);
    if (!scalars) return scalars.error();
    VideoPrediction out;
    out.raw_weight = static_cast<double>(scalars->shadow_weight);   // 1 when the model has no shadow gate
    std::tie(out.shadow_weight, out.cut) = smoother.update(thumb, out.raw_weight);
    FrameScalars use = *scalars;
    use.shadow_weight = static_cast<float>(out.shadow_weight);
    hold_frame_values(use, smoother);

    const int h = frame.height(), w = frame.width();
    if (tile_size_ <= 0 || std::max(h, w) <= tile_size_) {
        auto whole = forward(frame, use);
        if (!whole) return whole.error();
        out.hdr = std::move(*whole);
        return out;
    }
    PlanarBuffer result(3, h, w, 0.0f), weights(1, h, w, 0.0f);
    for (const int y : video_tile_starts(h, tile_size_, overlap_)) {
        for (const int left : video_tile_starts(w, tile_size_, overlap_)) {
            const int th = std::min(tile_size_, h - y), tw = std::min(tile_size_, w - left);
            auto hdr = forward(frame.crop(y, left, th, tw), use);
            if (!hdr) return hdr.error();
            const auto blend = video_tile_blend(y, left, th, tw, h, w, overlap_);
            for (int r = 0; r < th; ++r) {
                float* wrow = weights.plane(0) + static_cast<std::size_t>(y + r) * w + left;
                const float* brow = blend.data() + static_cast<std::size_t>(r) * tw;
                for (int c = 0; c < 3; ++c) {
                    float* dst = result.plane(c) + static_cast<std::size_t>(y + r) * w + left;
                    const float* src = hdr->buffer().plane(c) + static_cast<std::size_t>(r) * tw;
                    for (int k = 0; k < tw; ++k) dst[k] += src[k] * brow[k];
                }
                for (int k = 0; k < tw; ++k) wrow[k] += brow[k];
            }
        }
    }
    const float* wp = weights.plane(0);
    for (int c = 0; c < 3; ++c) {
        float* dst = result.plane(c);
        for (std::size_t i = 0; i < result.plane_size(); ++i) dst[i] /= std::max(wp[i], 1e-6f);
    }
    out.hdr = NetworkLinearImage(std::move(result));
    return out;
}

}  // namespace rudra
