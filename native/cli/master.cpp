#include "master.hpp"

#include "rudra/core/clip_fill.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/media/still.hpp"

#include <nlohmann/json.hpp>

namespace rudra {

std::string with_app_defaults(const std::string& params_json, const std::filesystem::path& source, bool lossy_movie,
                              std::vector<std::string>* notes) {
    nlohmann::json j;
    try {
        j = params_json.empty() ? nlohmann::json::object() : nlohmann::json::parse(params_json);
    } catch (const std::exception&) {
        return params_json;
    }
    if (!j.is_object() || j.contains("rudra_project")) return params_json;
    if (!j.contains("source_curve")) {
        j["source_curve"] = "bt1886";
        if (notes) notes->push_back("Source curve: bt1886, the app's default (give \"source_curve\" in the parameters for another)");
    }
    if (!j.contains("lossy_source") && (lossy_movie || is_lossy_still_path(source.string()))) {
        j["lossy_source"] = true;
        if (notes) notes->push_back("Lossy source: the highlight fill smooths its compression noise, as the app does "
                                    "(\"lossy_source\": false turns it off)");
    }
    return j.dump();
}

Result<MasterResult> render_analytic_master(const std::filesystem::path& image, const MasterRequest& q,
                                            const std::filesystem::path& out) {
    auto decoded = decode_sdr_file(image);
    if (!decoded) return decoded.error();
    MasterRequest qa = q;
    qa.analytic = true;
    qa.own_codes = true;   // 24 Nov 2026: a still's own codes (core/source_curve.hpp)
    qa.checkpoint.clear();
    qa.checkpoint_sha256.clear();
    qa.source_file = image;
    qa.source_space = q.source_space == "rec709" ? primaries_id(decoded->colour.primaries) : q.source_space;
    qa.alpha = decoded->alpha;
    return write_master(decoded->rgb, decoded->bits, analytic_fields_for(decoded->rgb, qa), FrameScalars{}, kAnalyticConstants, qa,
                        out);
}

Result<MasterResult> render_master(const ModelManifest& pkg, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& q,
                                   const std::filesystem::path& out) {
    // Full resolution, untiled: the Studio's master is one pass over the frame.
    auto decoded = decode_sdr_file(image);
    if (!decoded) return decoded.error();
    // Roadmap 1.1: the still's own primaries (ICC / cICP), unless the request names one.
    const std::string space = q.source_space == "rec709" ? primaries_id(decoded->colour.primaries) : q.source_space;
    // The analytic reconstruction (the app's default): no model runs, and the
    // label the master gets is then true.
    if (q.analytic) {
        MasterRequest qa = q;
        qa.own_codes = true;   // 24 Nov 2026: a still's own codes (core/source_curve.hpp)
        qa.checkpoint.clear();
        qa.checkpoint_sha256.clear();
        qa.source_file = image;
        qa.source_space = space;
        qa.alpha = decoded->alpha;   // an RGBA still keeps its alpha in the master
        return write_master(decoded->rgb, decoded->bits, analytic_fields_for(decoded->rgb, qa), FrameScalars{}, kAnalyticConstants, qa,
                            out);
    }
    auto fr = infer_frame(backend, decoded->rgb, TileConfig{0, 0});
    if (!fr) return fr.error();
    MasterRequest qm = q;   // provenance: the package's checkpoint hash, the source, the pass
    qm.own_codes = true;   // 24 Nov 2026: a still's own codes (core/source_curve.hpp)
    qm.checkpoint_sha256 = pkg.source_sha256;
    qm.source_file = image;
    qm.tiled = fr->tiled;
    qm.source_space = space;
    qm.alpha = decoded->alpha;
    return write_master(decoded->rgb, decoded->bits, fr->fields, fr->scalars,
                        ModelConstants{pkg.log_scale, pkg.max_hdr, pkg.corpus_ev}, qm, out);
}

Result<CorrectionEntry> log_master_correction(const ModelManifest* pkg, InferenceBackend* backend,
                                              const std::filesystem::path& log_dir, const std::filesystem::path& image,
                                              const MasterRequest& q, const std::string& label) {
    // Before any work: no consent, nothing is decoded.
    auto consent = read_correction_consent(log_dir);
    if (!consent) return consent.error();
    if (!consent->opted_in || consent->withdrawn())
        return make_error(ErrorCode::Unsupported, "Correction logging is off: there is no active consent", log_dir.string());
    auto decoded = decode_sdr_file(image);
    if (!decoded) return decoded.error();
    MasterRequest qg = q;
    qg.own_codes = true;   // 24 Nov 2026: a still's own codes (core/source_curve.hpp)
    qg.source_space = q.source_space == "rec709" ? primaries_id(decoded->colour.primaries) : q.source_space;
    qg.source_file = image;
    Result<GradedMaster> g = make_error(ErrorCode::InvalidArgument, "no grade");
    if (q.analytic) {
        qg.checkpoint.clear();
        qg.checkpoint_sha256.clear();
        g = grade_master(decoded->rgb, analytic_fields_for(decoded->rgb, qg), FrameScalars{}, kAnalyticConstants, qg);
    } else {
        if (!pkg || !backend) return make_error(ErrorCode::InvalidArgument, "These settings run the model: give its package");
        auto fr = infer_frame(*backend, decoded->rgb, TileConfig{0, 0});
        if (!fr) return fr.error();
        qg.checkpoint_sha256 = pkg->source_sha256;
        qg.tiled = fr->tiled;
        g = grade_master(decoded->rgb, fr->fields, fr->scalars, ModelConstants{pkg->log_scale, pkg->max_hdr, pkg->corpus_ev}, qg);
    }
    if (!g) return g.error();
    return log_correction(log_dir, decoded->rgb, g->nits, qg, CorrectionSource{image, 0, label});
}

}  // namespace rudra
