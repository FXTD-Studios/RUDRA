#pragma once
// AI fill in the OFX plug-in (phase 3, 26 Nov 2026). OFX hands a plug-in one frame at a time, and AI fill
// draws a shot in windows of 33 frames, so the plug-in does it in three steps, the same in Nuke and Resolve:
//
//   1. Generate (a button): the plug-in starts collecting. Every frame the host renders at full size is
//      stored: its SDR codes (16-bit TIFF) and its master (ACEScg EXR, 1.0 = diffuse white, 0 EV), exactly
//      the picture the plug-in shows without the fill. Play or render the shot once.
//   2. When the clip's last missing frame is in, the AI fill starts beside the host, as the RUDRA app runs it
//      (rudra/genfill/ofx_job.py: AI fill v5, the region re-render, the app's settings), on the GPU, about
//      20 minutes a 100 frames of 1080p. It writes its progress to status.json.
//   3. Once it is done, every render of that clip with those settings shows the stored fill (an ACEScg EXR
//      per frame with rudra.generated, what AI drew), after Exposure and in the chosen Output.
//
// Stored per instance and per settings: <root>/<shot id>/<key>/, the key a hash of everything that changes
// the master or the fill (route, Source, Strength, the luminance limit, the prompt, the seed, the clip's
// frames). A changed setting is another key: the fill shown is never one made with other settings, and a
// frame whose codes differ from the stored ones (something upstream changed) shows the master.
//   root: RUDRA_OFX_FILL_ROOT, else %LOCALAPPDATA%\FXTD Studios\RUDRA\ofx_fill (Windows),
//         ~/Library/Application Support/FXTD Studios/RUDRA/ofx_fill (macOS), ~/.local/share/... (Linux).
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "engine_client.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::ofx_fill {

struct FillRequest {
    std::string shot_id;       // the instance's own id (a hidden parameter, made at the first Generate)
    int first = 0, last = 0;   // the clip's frames
    std::string condition;     // "bt1886" (the analytic route's master) or "aces" (the learned route's)
    std::string prompt;        // what to draw in the highlights ("" none, as the app's default)
    int seed = 0;
    int steps = 30;            // the fill's denoising steps (27 Nov 2026: the plug-in's Steps, 10 .. 50)
    std::string region_json;   // the luminance limit, deliver reconstruction_region_json ("" the whole frame)
    std::string master_key;    // every other setting that changes the master (route, Source, Strength ...)
    int frames() const { return last - first + 1; }
    std::string key() const;   // 16 hex digits
};

enum class State { None, Collecting, Running, Done, Failed, Cancelled };

struct Status {
    State state = State::None;
    int done = 0, of = 0;      // frames collected (Collecting) or filled (Running)
    std::string message;       // the job's last word (Running, Failed)
};

std::filesystem::path root();
std::filesystem::path folder(const FillRequest& r);   // root/shot_id/key
std::string frame_stem(int index);                     // "f.000123"
std::uint64_t codes_hash(const SdrImage& codes);       // xxh64 of the 8-bit codes

// The Generate button: request.json and the collect marker. Refused for a clip of more than kMaxFrames.
inline constexpr int kMaxFrames = 2000;
Result<void> begin_collect(const FillRequest& r);
Status status(const FillRequest& r);
// Stores one frame while collecting (codes and the 0 EV master, ACEScg planes). True when it was the last
// frame missing. A frame already stored from the same codes is not written again.
Result<bool> store_frame(const FillRequest& r, int frame, const SdrImage& codes, const PlanarBuffer& master_ap1);
// Starts the AI fill on the stored frames (once; a running job is left alone).
Result<void> start_job(const FillRequest& r, const ofx_engine::GenFillConfig& cfg);
// Cancel: stops a running job, or stops collecting (the stored frames are kept for the next Generate).
Result<void> cancel(const FillRequest& r);
// Clear stored fill: everything stored for this instance (refused while its fill runs).
Result<void> clear(const std::string& shot_id);
// Another setting of this shot has a finished fill (the settings changed since its Generate).
bool other_fill(const FillRequest& r);

struct FillFrame {
    PlanarBuffer ap1;         // ACEScg, 1.0 = diffuse white, 0 EV
    PlanarBuffer generated;   // rudra.generated: where AI drew, 0..1
};
// The stored fill of `frame`, when the fill is done and was made from these same codes; else nullopt and why.
std::optional<FillFrame> fill_frame(const FillRequest& r, int frame, const SdrImage& codes, std::string* why = nullptr);

// The stored master of `frame` (ACEScg, 0 EV), when it was made from these same codes: shown while the fill
// runs, so the learned engine stays off the GPU the fill uses.
std::optional<PlanarBuffer> master_frame(const FillRequest& r, int frame, const SdrImage& codes);

// One line for the plug-in's Status parameter and messages.
std::string status_text(const Status& s);

}  // namespace rudra::ofx_fill
