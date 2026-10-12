#pragma once
// The video predictor (Phase 4, step 3): rudra/video.py Predictor.predict on
// a model package. The frame is canonicalised and brought to Rec.2020 code
// (core/video_predict), the frame pass gives the shadow weight and residual
// scale once, the smoother holds the weight across a shot, and the frame runs
// whole or in tiles blended with the video path's own feathers; each tile is
// the network's fields through the composite with preserve_outside and
// recovery "all", which is forward(...).hdr.

#include <string>

#include "rudra/core/composite.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/core/video_predict.hpp"
#include "rudra/infer/backend.hpp"

namespace rudra {

struct VideoPrediction {
    NetworkLinearImage hdr;      // 1.0 = 10 000 nits
    double raw_weight = 1.0;     // the frame pass's shadow weight, before smoothing
    double shadow_weight = 1.0;  // what the frame was reconstructed with
    bool cut = false;
};

class VideoPredictor {
public:
    VideoPredictor(InferenceBackend& backend, const ModelConstants& model, int tile_size, int overlap)
        : backend_(&backend), model_(model), tile_size_(tile_size), overlap_(overlap) {}
    // No backend: only predict_graded with an analytic request runs.
    VideoPredictor(InferenceBackend* backend, const ModelConstants& model, int tile_size, int overlap)
        : backend_(backend), model_(model), tile_size_(tile_size), overlap_(overlap) {}

    // `rgb` is the decoded frame as full-range code values in the clip's
    // transfer; `transfer` and `primaries` are the input contract's.
    Result<VideoPrediction> predict(const SdrImage& rgb, const std::string& transfer, const std::string& primaries,
                                    ShadowSmoother& smoother);

    // The app's movie export: the frame through the EXR master's chain with
    // the app's settings (deliver/master grade_master): its recovery, source
    // curve, calibration or reference, masks, anchor, chroma, grain and
    // Region EV. With `request.analytic` no model runs (kAnalyticConstants,
    // analytic_fields_for: the clip fill unless the request turns it off); otherwise the fields come from the backend, tiled as
    // the master tiles, with the smoothed shadow weight. hdr is the delivered
    // picture in the network convention (1.0 = 10 000 nits), Rec.2020.
    Result<VideoPrediction> predict_graded(const SdrImage& rgb, const std::string& transfer,
                                           const std::string& primaries, ShadowSmoother& smoother,
                                           const MasterRequest& request);

private:
    Result<NetworkLinearImage> forward(const SdrImage& tile, const FrameScalars& scalars);
    // canonicalize_sdr and Rec.2020 code, as predict() starts; and the cut thumbnail.
    Result<std::pair<SdrImage, PlanarBuffer>> prepare(const SdrImage& rgb, const std::string& transfer,
                                                      const std::string& primaries) const;
    InferenceBackend* backend_;
    ModelConstants model_;
    int tile_size_, overlap_;
};

}  // namespace rudra
