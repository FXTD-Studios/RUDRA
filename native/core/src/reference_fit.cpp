#include "rudra/core/reference_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <nlohmann/json.hpp>

#include "rudra/core/calibration.hpp"
#include "rudra/core/color.hpp"

namespace rudra {
namespace {

double median_of(std::vector<float>& v) {
    const auto mid = v.begin() + std::ptrdiff_t(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return double(*mid);
}

// Pool adjacent violators: the non-decreasing sequence closest to y in the
// weighted least-squares sense.
void isotonic(std::vector<double>& y, const std::vector<double>& w) {
    const std::size_t n = y.size();
    std::vector<double> val, wt;
    std::vector<std::size_t> len;
    for (std::size_t i = 0; i < n; ++i) {
        val.push_back(y[i]);
        wt.push_back(w[i]);
        len.push_back(1);
        while (val.size() > 1 && val[val.size() - 2] > val.back()) {
            const double tw = wt[wt.size() - 2] + wt.back();
            const double tv = (val[val.size() - 2] * wt[wt.size() - 2] + val.back() * wt.back()) / tw;
            val.pop_back(); wt.pop_back();
            const std::size_t l = len.back(); len.pop_back();
            val.back() = tv; wt.back() = tw; len.back() += l;
        }
    }
    std::size_t k = 0;
    for (std::size_t b = 0; b < val.size(); ++b)
        for (std::size_t i = 0; i < len[b]; ++i) y[k++] = val[b];
}

}  // namespace

Result<ReferenceFit> fit_reference(const SdrImage& sdr, const PlanarBuffer& reference, std::string file) {
    if (reference.channels() != 3)
        return make_error(ErrorCode::InvalidArgument, "reference must have 3 channels", std::to_string(reference.channels()));
    if (reference.width() != sdr.width() || reference.height() != sdr.height())
        return make_error(ErrorCode::InvalidArgument, "reference is not the frame's size",
                          std::to_string(reference.width()) + "x" + std::to_string(reference.height()) + " vs " +
                              std::to_string(sdr.width()) + "x" + std::to_string(sdr.height()));
    const std::size_t n = std::size_t(kSourceCurveKnots);
    std::vector<std::vector<float>> buckets(n);
    const std::size_t px = sdr.buffer().plane_size();
    const double k = double(kDiffuseWhite.v);
    for (int c = 0; c < 3; ++c) {
        const float* s = sdr.buffer().plane(c);
        const float* r = reference.plane(c);
        for (std::size_t i = 0; i < px; ++i) {
            const float v = r[i];
            if (!(v > 0.0f) || !std::isfinite(v)) continue;
            const int code = std::clamp(int(std::lround(double(s[i]) * 255.0)), 0, int(n) - 1);
            buckets[std::size_t(code)].push_back(float(std::log2(double(v) * k)));
        }
    }
    ReferenceFit fit;
    fit.file = std::move(file);
    fit.samples_per_code.assign(n, 0);
    std::vector<double> med(n, NAN);
    int first = -1, last = -1;
    for (int code = 0; code < int(n); ++code) {
        auto& b = buckets[std::size_t(code)];
        if (int(b.size()) < kReferenceMinSamples) continue;
        fit.samples_per_code[std::size_t(code)] = int(b.size());
        med[std::size_t(code)] = median_of(b);
        if (first < 0) first = code;
        last = code;
        ++fit.codes_seen;
    }
    if (fit.codes_seen == 0)
        return make_error(ErrorCode::InvalidArgument, "reference gives no code enough samples",
                          std::to_string(kReferenceMinSamples) + " needed per code");
    // Holes from their neighbours, flat outside.
    std::vector<double> t(n);
    for (int code = 0; code < int(n); ++code) {
        if (code <= first) t[std::size_t(code)] = med[std::size_t(first)];
        else if (code >= last) t[std::size_t(code)] = med[std::size_t(last)];
        else if (!std::isnan(med[std::size_t(code)])) t[std::size_t(code)] = med[std::size_t(code)];
        else {
            int a = code, b = code;
            while (std::isnan(med[std::size_t(a)])) --a;
            while (std::isnan(med[std::size_t(b)])) ++b;
            t[std::size_t(code)] = med[std::size_t(a)] + (med[std::size_t(b)] - med[std::size_t(a)]) * double(code - a) / double(b - a);
        }
    }
    // Triangular smoothing, then monotone.
    std::vector<double> sm(n), w(n);
    for (int code = 0; code < int(n); ++code) {
        double s = 0.0, ws = 0.0;
        for (int d = -kReferenceSmoothRadius; d <= kReferenceSmoothRadius; ++d) {
            const int j = std::clamp(code + d, 0, int(n) - 1);
            const double ww = double(kReferenceSmoothRadius + 1 - std::abs(d));
            s += t[std::size_t(j)] * ww;
            ws += ww;
        }
        sm[std::size_t(code)] = s / ws;
        w[std::size_t(code)] = std::max(1.0, double(fit.samples_per_code[std::size_t(code)]));
    }
    isotonic(sm, w);
    fit.target_log2_nits.resize(n);
    for (int code = 0; code < int(n); ++code) fit.target_log2_nits[std::size_t(code)] = float(sm[std::size_t(code)]);
    // The residual: every sample's distance to the target, codes 1..254.
    std::vector<float> err;
    for (int code = 1; code < int(n) - 1; ++code)
        for (float v : buckets[std::size_t(code)]) err.push_back(std::fabs(v - fit.target_log2_nits[std::size_t(code)]));
    fit.samples = (long long)err.size();
    if (!err.empty()) {
        double s = 0.0;
        for (float e : err) s += e;
        fit.residual_mean = s / double(err.size());
        const auto p = err.begin() + std::ptrdiff_t(std::min(err.size() - 1, std::size_t(double(err.size()) * 0.95)));
        std::nth_element(err.begin(), p, err.end());
        fit.residual_p95 = double(*p);
    }
    return fit;
}

std::vector<float> reference_params(const ReferenceFit& fit, SourceCurve source, float corpus_ev) {
    if (fit.empty() || int(fit.target_log2_nits.size()) != kSourceCurveKnots) return {};
    std::vector<float> out(1 + std::size_t(kSourceCurveKnots), 0.0f);
    for (int code = 0; code < kSourceCurveKnots; ++code) {
        const double have = std::max(source_code_nits(source, code, corpus_ev), 1e-9);
        out[1 + std::size_t(code)] = float(double(fit.target_log2_nits[std::size_t(code)]) - std::log2(have));
    }
    return out;
}

double reference_exposure(const ReferenceFit& fit, SourceCurve source, float corpus_ev) {
    const auto p = reference_params(fit, source, corpus_ev);
    if (p.empty()) return 0.0;
    std::vector<std::pair<double, double>> v;   // correction, weight
    double total = 0.0;
    for (int code = 0; code < kSourceCurveKnots; ++code) {
        const double w = code < int(fit.samples_per_code.size()) ? double(fit.samples_per_code[std::size_t(code)]) : 0.0;
        if (w <= 0.0) continue;
        v.emplace_back(double(p[1 + std::size_t(code)]), w);
        total += w;
    }
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double acc = 0.0;
    for (const auto& [c, w] : v) {
        acc += w;
        if (acc >= total * 0.5) return c;
    }
    return v.back().first;
}

std::string reference_summary(const ReferenceFit& fit, SourceCurve source, float corpus_ev) {
    if (fit.empty()) return "";
    char b[200];
    std::snprintf(b, sizeof b, "%lld samples over %d codes: exposure %+.2f stops and a per-code curve over the picker's; anchors are replaced.",
                  fit.samples, fit.codes_seen, reference_exposure(fit, source, corpus_ev));
    std::string s = b;
    if (fit.residual_p95 > 0.5) s += " p95 over half a stop: the grade has a local key the curve cannot carry.";
    return s;
}

std::string reference_fit_json(const ReferenceFit& fit) {
    nlohmann::ordered_json j;
    j["file"] = fit.file;
    j["samples"] = fit.samples;
    j["codes_seen"] = fit.codes_seen;
    j["residual_mean"] = fit.residual_mean;
    j["residual_p95"] = fit.residual_p95;
    j["target_log2_nits"] = fit.target_log2_nits;
    j["samples_per_code"] = fit.samples_per_code;
    return j.dump();
}

Result<ReferenceFit> reference_fit_from_json(const std::string& json_text) {
    nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return make_error(ErrorCode::InvalidArgument, "reference is not a JSON object", "");
    ReferenceFit fit;
    fit.file = j.value("file", std::string());
    fit.samples = j.value("samples", 0LL);
    fit.codes_seen = j.value("codes_seen", 0);
    fit.residual_mean = j.value("residual_mean", 0.0);
    fit.residual_p95 = j.value("residual_p95", 0.0);
    const auto t = j.find("target_log2_nits");
    if (t == j.end() || !t->is_array() || t->size() != std::size_t(kSourceCurveKnots))
        return make_error(ErrorCode::InvalidArgument, "reference target must be 256 numbers", "");
    for (const auto& v : *t) {
        if (!v.is_number() || !std::isfinite(v.get<double>()))
            return make_error(ErrorCode::InvalidArgument, "reference target must be 256 finite numbers", "");
        fit.target_log2_nits.push_back(v.get<float>());
    }
    fit.samples_per_code.assign(std::size_t(kSourceCurveKnots), 0);
    if (const auto s = j.find("samples_per_code"); s != j.end() && s->is_array() && s->size() == std::size_t(kSourceCurveKnots))
        for (std::size_t i = 0; i < s->size(); ++i) fit.samples_per_code[i] = (*s)[i].is_number() ? (*s)[i].get<int>() : 0;
    return fit;
}

}  // namespace rudra
