#include "master.hpp"

#include "rudra/infer/tiler.hpp"
#include "rudra/media/still.hpp"

namespace rudra {

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
        qa.checkpoint.clear();
        qa.checkpoint_sha256.clear();
        qa.source_file = image;
        qa.source_space = space;
        return write_master(decoded->rgb, decoded->bits, analytic_fields_for(decoded->rgb, qa), FrameScalars{}, kAnalyticConstants, qa,
                            out);
    }
    auto fr = infer_frame(backend, decoded->rgb, TileConfig{0, 0});
    if (!fr) return fr.error();
    MasterRequest qm = q;   // provenance: the package's checkpoint hash, the source, the pass
    qm.checkpoint_sha256 = pkg.source_sha256;
    qm.source_file = image;
    qm.tiled = fr->tiled;
    qm.source_space = space;
    return write_master(decoded->rgb, decoded->bits, fr->fields, fr->scalars,
                        ModelConstants{pkg.log_scale, pkg.max_hdr, pkg.corpus_ev}, qm, out);
}

}  // namespace rudra
