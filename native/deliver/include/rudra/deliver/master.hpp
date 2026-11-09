#pragma once
// The master (Phase 3 step 9, from the CLI of Phase 1 step 7): a frame to a
// master EXR and its sidecar, the stages of ui/server.py _render_master after
// the network -- composite and the master chain (core), measure (core), EXR
// and sidecar (deliver) -- and the Studio's render plan and publishing
// (master_targets and run_master). Decoding the still and running the network
// are the caller's (the CLI, the app's master job): deliver sees neither media
// nor infer, as the layer rule says.

#include <optional>
#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/clip_fill.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/reference_fit.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct MasterBand {
    std::string label;
    double low_nits = 0.0, high_nits = 0.0, ev = 0.0;
};

// The Studio's master parameters, with its defaults.
struct MasterRequest {
    std::string checkpoint;                 // recorded, not loaded: the package is the model
    // The source's straight alpha in [0, 1] (1 Nov 2026), from an RGBA still or an
    // alpha movie's frame; empty for none. Written as EXR A with RGB premultiplied
    // (deliver/exr.hpp), in either container. The reconstruction never sees it.
    std::optional<PlanarBuffer> alpha;
    // The analytic reconstruction: no model ran (the fields were
    // analytic_fields()). Recorded in the EXR (rudra:reconstruction) and the
    // sidecar ("reconstruction"); left out of both when a model ran, so
    // masters made with one read as they did before.
    bool analytic = false;
    bool preserve_outside = true;
    std::string recovery_mode = "all";      // all, highlights, shadows, off
    double strength = 1.0;
    // Spec revision 2 (RUDRA 1.1, 5 Nov 2026): where the reconstruction may act
    // (core/composite.hpp). In the JSON (`reconstruction_region`), the EXR
    // (rudra:reconstructionRegion) and the sidecar only when on, so every
    // master before reads the same.
    ReconstructionRegion region;
    // RUDRA 1.1 priority 3 (5 Nov 2026): write <name>.mattes.exr beside the master: float32
    // channels rudra.highlight, rudra.shadow, rudra.changed, rudra.region (with a region) and
    // rudra.paint.<band> (painted masks), core reconstruction_mattes. In the JSON, EXR and
    // sidecar only when on.
    bool mattes = false;
    // RUDRA 1.1: where the mattes (and the regions picture) go instead of beside the master,
    // as <master stem>.mattes.exr: a movie export's frames folder holds frames only. Empty:
    // beside the master. Not in the JSON (the app sets it for the movie it is making).
    std::filesystem::path mattes_dir;
    // RUDRA 1.1 priority 4 (5 Nov 2026): <name>.ev-2.png, .ev+0.png, .ev+2.png beside the master,
    // rendered from the EXR as written (read back), and the preview-to-export check: the same
    // brackets from the in-memory picture the viewer shows, compared code for code. In the JSON,
    // EXR and sidecar only when on.
    bool brackets = false;
    std::vector<MasterBand> regions;        // empty: the neutral default bands
    double region_softness_stops = 1.0;
    bool anchor = true;
    double anchor_knee = 0.9;
    bool carry_chroma = true;
    double chroma_knee = 0.99;
    bool settle_grain = true;               // at anchor_knee
    std::string source_space = "rec709";
    std::string container = "aces";         // aces (ACES 2065-1), acescg, linear (Rec.2020) or rec709 (linear; RUDRA 1.1)
    bool exr_half = true;                  // false: 32-bit float grading master
    // Roadmap 3.1: the curve that made the SDR (core/source_curve.hpp ids).
    // "unknown" is the ACES inverse, and is left out of the JSON and the
    // sidecar so every master before 3.1 reads the same.
    std::string source_curve = "unknown";
    // Roadmap 3.2: the anchors over it; empty for none, left out of the JSON
    // and the sidecar then.
    std::vector<CalibrationPoint> calibration;
    // Roadmap 3.4: the reference match (core/reference_fit.hpp); empty for
    // none, left out of the JSON and the sidecar then. Replaces the anchors.
    ReferenceFit reference;
    // Roadmap 3.3: the painted masks (core/masks.hpp). In the JSON as
    // `masks_file`, a PNG path read on parse; written beside the master as
    // <name>.masks.png and named in the sidecar. Null for none.
    std::shared_ptr<const MaskSet> masks;
    std::string masks_file;   // where the JSON pointed, or where write_master wrote them
    // 12 Oct 2026: the clip fill (core/clip_fill.hpp) in the analytic
    // reconstruction, and the peaks set on clipped regions. On by default;
    // in the JSON only when off or with peaks. With a model it does nothing:
    // the model's own residual is what fills there.
    bool clip_fill = true;
    std::vector<ClipOverride> clip_peaks;
    // 9 Nov 2026: the clip fill's near-clip detail gain (ClipFillOptions::detail_gain),
    // 1 = off; in the JSON and the sidecar only when not 1.
    double highlight_detail = 1.0;
    // 13 Oct 2026: the master exposure in stops, with the grade (after the
    // anchor, before the Region EV bands). In the JSON and the sidecar only
    // when not 0, so every master before reads the same.
    double exposure_ev = 0.0;
    // 16 Oct 2026, provenance (roadmap 1.3): what made the master, for the
    // record. Runtime facts, never in the request JSON. The source file is
    // hashed (rudra:sourceSha256) when set; the checkpoint's SHA-256 is the
    // package manifest's source_sha256, empty for the analytic
    // reconstruction; tiled says whether the network ran in tiles.
    std::string checkpoint_sha256;
    std::filesystem::path source_file;
    bool tiled = false;
};

// The fill's options for this request (the defaults, with its peaks).
ClipFillOptions clip_fill_options(const MasterRequest& q);

// The fields of the analytic reconstruction for this request: the clip fill's
// when it is on (clip_fill_fields at kAnalyticConstants), analytic_fields when
// it is off. What every analytic path (master, movie, the app's view) uses.
Fields analytic_fields_for(const SdrImage& sdr, const MasterRequest& q, ClipFill* detail = nullptr);

struct MasterResult {
    int maxcll = 0, maxfall = 0;
    double peak_nits = 0.0;
    int width = 0, height = 0, source_bits = 0;
    std::filesystem::path exr, sidecar;
    std::filesystem::path masks, mattes;   // beside the master when written (painted masks; RUDRA 1.1 mattes)
    std::filesystem::path regions;         // RUDRA 1.1: <name>.regions.png, the pixel classes, with the mattes
    std::vector<std::filesystem::path> brackets;   // RUDRA 1.1: -2, 0, +2 EV PNGs
    int bracket_max_code_delta = -1;               // preview vs export, worst 8-bit code over the brackets; -1 none
};

// The request as the Studio's params dict (what the golden index records).
Result<MasterRequest> master_request_from_json(const std::string& json_text);
// And back: what `rudra-native master --params` reads for the same master.
std::string master_request_json(const MasterRequest& q);

// The reconstruction region (spec revision 2) as the JSON object masters,
// projects and sidecars write, and back (enabled on read; checked).
std::string reconstruction_region_json(const ReconstructionRegion& r);
Result<ReconstructionRegion> reconstruction_region_from_json(const std::string& json_text);

// The master's picture before any file: the request's composite (mode,
// strength, preserve, source curve, calibration or reference, masks), then
// the anchor, chroma and grain stages and the grade, in absolute nits in the
// frame's own primaries. write_master writes it; the graded movie export
// (video/convert) encodes it.
struct GradedMaster {
    CompositeParams params;
    NetworkLinearImage network;   // the composite, before the stages
    NitsFrame nits;               // what is delivered
    bool graded = false;          // Region EV moved something
};
// `settle_hold`: the anchor's hold through it (core/master.hpp anchor_to_sdr).
Result<GradedMaster> grade_master(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                                  const ModelConstants& model, const MasterRequest& request,
                                  const HoldSettle& settle_hold = {});

// One frame, decoded (`source_bits` its depth) and through the network
// (fields, scalars), to `out` and its .json beside it.
Result<MasterResult> write_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                  const FrameScalars& scalars, const ModelConstants& model,
                                  const MasterRequest& request, const std::filesystem::path& out);

// ---- the render plan (master_targets) -------------------------------------

struct RenderPlan {
    std::string render_dir;           // an absolute folder
    std::string render_name = "master";
    int render_count = 1;
    int frame_start = 1;
    bool sequence = false;            // render_mode "sequence": name.000001.exr ...; else name.exr
};
// The files a render will write, in order; refused (with the Studio's
// message) for a relative folder, a bad name or range, or any file or
// sidecar that exists already.
Result<std::vector<std::filesystem::path>> master_targets(const RenderPlan& plan);

// run_master's publishing: write into a staging folder beside `out`, then
// link the EXR and its sidecar into place without replacing anything.
Result<MasterResult> publish_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                    const FrameScalars& scalars, const ModelConstants& model,
                                    const MasterRequest& request, const std::filesystem::path& out);

}  // namespace rudra
