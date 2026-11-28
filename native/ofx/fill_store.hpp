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
#include <vector>

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
    double prompt_strength = 1.0;   // 1.3.2: the plug-in's Prompt strength, 0 .. 2 (1 with no prompt)
    std::string region_json;   // the luminance limit, deliver reconstruction_region_json ("" the whole frame)
    std::string master_key;    // every other setting that changes the master (route, Source, Strength ...)
    int frames() const { return last - first + 1; }
    std::string key() const;   // 16 hex digits
};

// rudra/genfill/run.py PROMPT_PASS == "single" (1.3.2): one region prompt at strength 1 in one transformer pass a
// step, so it costs what no prompt costs (tests/test_ofx_job.py checks the two agree).
inline constexpr bool kSinglePromptPass = true;
// What this request's fill costs against one with no prompt: 1, or kBlendedPromptCost when its prompt is blended in
// (a second pass a step; measured 1.28 to 1.38 times on the gate shots, 26 Nov 2026).
inline constexpr double kBlendedPromptCost = 1.35;
double prompt_cost(const FillRequest& r);

enum class State { None, Collecting, Running, Done, Failed, Cancelled };

struct Status {
    State state = State::None;
    int done = 0, of = 0;      // frames collected (Collecting) or filled (Running)
    std::string message;       // the job's last word (Running, Failed)
    double generated = -1.0;   // 1.3.2: the share of the clip's pixels the fill drew (Done), -1 when not said
};

std::filesystem::path root();
std::filesystem::path folder(const FillRequest& r);   // root/shot_id/key
std::string frame_stem(int index);                     // "f.000123"
std::uint64_t codes_hash(const SdrImage& codes);       // xxh64 of the 8-bit codes

// The Generate button: request.json and the collect marker. Refused for a clip of more than kMaxFrames.
inline constexpr int kMaxFrames = 2000;
Result<void> begin_collect(const FillRequest& r);
Status status(const FillRequest& r);
// The clip's frames not stored yet, as indices from r.first (1.3.2: the status says which are still to play).
std::vector<int> missing_frames(const FillRequest& r);
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

// 28 Nov 2026 (review P2): whether an AI fill this host started is drawing. The learned route does not start
// its engine then (it would load a second model onto the GPU the fill is using; ofx_process.cpp make_master).
bool fill_running();

// One line for the plug-in's Status parameter and messages.
std::string status_text(const Status& s);

// 1.3.2: the prompt read while the frames are collected (rudra/genfill/ofx_prompt.py embed, beside the host, on the
// CPU), into <root>/prompt_cache, so the fill does not wait for the text encoder after the last frame. Started by
// Generate with a prompt; one at a time (a reading that runs is left alone).
Result<void> start_prompt_embed(const FillRequest& r, const ofx_engine::GenFillConfig& cfg);
// Whether the prompt reading started last still runs (1.3.2; the tests wait for one another's).
bool prompt_embed_running();

// 1.3.2, Suggest prompt: <root>/<shot id>/suggest/ (frame.tif, suggest.json, suggest.log). The frame's codes go to
// ofx_prompt.py suggest beside the host (Florence-2 from cfg.florence when set up; on the CPU when `cpu`). Refused
// while a suggestion for this shot is still being made.
Result<void> start_suggest(const std::string& shot_id, const SdrImage& codes, const ofx_engine::GenFillConfig& cfg, bool cpu);
// The answer (suggest.json's text) once written; nullopt while it is being made; an error when none was started
// or it ended without one.
Result<std::optional<std::string>> suggest_answer(const std::string& shot_id);

}  // namespace rudra::ofx_fill
