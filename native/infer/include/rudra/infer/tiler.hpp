#pragma once
// Frame -> fields through any backend: the frame pass once, then tiles, then
// stitching exactly as training/infer_sdr2hdr.py::predict_fields does.

#include "rudra/core/tiling.hpp"
#include "rudra/infer/backend.hpp"

namespace rudra {

struct FrameResult {
    FrameScalars scalars;
    Fields fields;
    bool tiled = false;
    // Inferred from a smaller copy of the frame and upsampled (playback):
    // the engine shows it, and infers the frame again at full size once the
    // playhead stops on it.
    bool proxy = false;
};

Result<FrameResult> infer_frame(InferenceBackend& backend, const SdrImage& frame, const TileConfig& cfg);

// The tile passes and stitching alone, with scalars the caller chose (the
// video path's smoothed shadow weight). `tiled` says whether it tiled.
Result<Fields> infer_fields(InferenceBackend& backend, const SdrImage& frame, const FrameScalars& scalars,
                            const TileConfig& cfg, bool* tiled = nullptr);

}  // namespace rudra
