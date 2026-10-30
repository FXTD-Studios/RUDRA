#pragma once
// The Deliver tab's choices (27 Oct 2026): what renders the files, which
// frames, the file type, its colour and, for a movie, its bit depth.
//
//   source    "aces": the learned ACES 2.0 model kept loaded (aces_engine.hpp),
//             what the viewer shows when it is ready; "analytic": the analytic
//             reconstruction (Master EXR and the movie exports, as before).
//   output    "still" (the frame on screen), "sequence" (every loaded frame,
//             one file each) or "movie" (one MOV).
//   image     a still or a sequence: "exr32", "exr16" or "dpx10".
//   gamut     an EXR: "ap0" (ACES 2065-1) or "acescg" (AP1); the analytic
//             source's other choice is "linear" (Rec.2020 scene linear).
//   encoding  a DPX or a movie: "logc4", "acescct", "pq" (at pq_peak), "hlg" or "sdr"
//             (the learned source's HLG is ACES 2.0 HDR 1000 on Rec.2100-HLG, its SDR
//             the SDR 100 Rec.709 trim pass; 28 Oct 2026 review) (the learned
//             source's PQ is the ACES 2.0 HDR 1000-nit output transform; the
//             analytic source's is its graded PQ, to the peak and knee).
//   bits      a movie: 10 (ProRes 422 HQ or 4444) or 16 (uncompressed RGB, 'b48r').
//   ai_fill   the learned source's EXR still or sequence: also an AI-filled copy
//             (genfill_job.hpp, layer 3), with its thresholds, prompts, seed and steps.
//
// Pure functions, so the tests check every combination without a window.

#include <QJsonObject>
#include <QString>

#include "genfill_job.hpp"

namespace rudra::app {

struct DeliverChoice {
    QString source = "aces";
    QString output = "still";
    QString image = "exr32";
    QString gamut = "ap0";
    QString encoding = "logc4";
    int bits = 10;
    int pq_peak = 1000;              // the learned PQ target: 1000, 2000 or 4000 nits (ACES 2.0, P3 D65)
    QString codec = "prores422hq";   // 10-bit: "prores422hq", "prores4444", "prores4444xq" (learned) or "hevc10" (learned, PQ: HDR10 MP4)
    bool keep_exr = false;           // learned movie: the EXR masters beside it
    double fps = 24.0;               // a movie made from stills
    bool ai_fill = false;            // learned EXR still / sequence: then the AI fill of the master
    GenFillOptions ai;               // its thresholds, prompts, seed and steps
    bool operator==(const DeliverChoice&) const = default;
};

// What the analytic source can write, for a shot that is a movie, a folder of
// frames or a still: the choice moved to the nearest one it can (EXR 32-bit,
// a 16-bit movie in PQ, a folder's movie in PQ). The learned source takes all.
DeliverChoice deliver_fit(DeliverChoice c, bool aces_source, bool shot_movie, bool shot_folder);

// Whether this choice can have the AI fill: the learned source's EXR (still or sequence).
bool ai_fill_possible(const DeliverChoice& c, bool aces_source);

// The learned engine's master request fields (cli/aces_v2_deliver.hpp):
// exr, exr_gamut, log, log_formats. Frames, fps, name and start are the window's.
QJsonObject aces_deliver_fields(const DeliverChoice& c);

// The trims as the engine takes them ({} when they change nothing).
QJsonObject aces_trims_fields(double strength, double saturation, double peak_nits, double exposure_ev = 0.0,
                              double contrast = 1.0);

// The analytic movie's export format (the queue's and `rudra-native deliver`'s):
// prores422hq / prores4444, with _acescct or _logc4 for a log, or rgb16mov.
QString analytic_movie_format(const DeliverChoice& c);

// The Signal card: container, transfer, primaries.
struct DeliverSignal {
    QString container, transfer, primaries;
};
DeliverSignal deliver_signal(const DeliverChoice& c, bool aces_source);
// "PQ", "PQ 4000", "HLG", "SDR Rec.709", "LogC4", "ACEScct".
QString deliver_encoding_words(const DeliverChoice& c);

// The pipe's last step ("learned ACES 2.0 → ACES 2065-1 EXR, float32"); empty
// for the analytic EXR, whose words the pipe already has.
QString deliver_pipe_label(const DeliverChoice& c, bool aces_source);

// The Render card's file path, as Nuke's Write takes it (29 Oct 2026):
// "D:/Renders/SH010/SH010_hdr.####.exr" (also %04d), "…/SH010_hdr.mov".
struct RenderPath {
    QString dir, name, ext;
    int padding = 0;             // the #'s; 0 = no frame number (a movie)
    QString separator = ".";     // between the name and the number: ".", "_" or ""
    bool operator==(const RenderPath&) const = default;
};
RenderPath parse_render_path(const QString& path);
QString compose_render_path(const RenderPath& p);   // native separators
// The file type of this choice: "exr", "dpx" or "mov".
QString render_ext(const DeliverChoice& c);

// The Render button's words ("Master EXR" for the analytic EXR, as before).
QString deliver_button_text(const DeliverChoice& c, bool aces_source);

}  // namespace rudra::app
