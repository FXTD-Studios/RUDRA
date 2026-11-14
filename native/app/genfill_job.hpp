#pragma once
// The AI fill (RUDRA layer 3, 27 Oct 2026): `python -m rudra.genfill.run` over a learned
// master's EXR folder (a Wan2.1-VACE LoRA trained on RUDRA's own data fills only the
// fully clipped highlights and crushed shadows; no 8-bit code of the master's SDR view
// changes). The app runs it as a QProcess after a learned EXR render and reads its JSON
// lines: ready, progress (window / step), frame (index of N), done, error.
//
// Where things are (QSettings, else the first that exists of: <app>/models/genfill,
// <repo>/models/genfill, $RUDRA_GENFILL_DIR, then the training workstation's folders;
// Deliver > AI fill > Setup… sets and checks them, 28 Oct 2026 review):
//   genfill/python   the Python with torch, diffusers, peft, OpenEXR, OCIO  (<repo>/.venv/Scripts/python.exe)
//   genfill/repo     the RUDRA repository (rudra/genfill/run.py)            (found beside the app, upwards)
//   genfill/base     Wan2.1-VACE-1.3B-diffusers                              (<root>/Wan2.1-VACE-1.3B-diffusers)
//   genfill/lora     rudra_genfill.safetensors                               (<root>/rudra_genfill.safetensors)
// No Q_OBJECT (as elsewhere in the app): listeners are callbacks.

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "json_lines.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

class QObject;
class QProcess;

namespace rudra::app {

struct GenFillSetup {
    std::filesystem::path python, repo, base, lora;
    // 4 Nov 2026: which LoRA this is. A saved path is kept while it exists, so an
    // older LoRA (genfill v1, v2) could run as "Ready"; the identity says so.
    bool lora_is_v3 = false;
    std::filesystem::path v3_found;   // a verified v3 among the candidates, when there is one
    // 12 Nov 2026: the setup of the region re-render (v4). `lora` is then the v4 LoRA, and the
    // fill starts with the settings v4 was tested with (feather 8, the model's own brightness).
    bool rerender = false;
};

// The published AI fill v3 LoRA (docs/MODELS.md; PUBLISH_HF_MODELS.ps1).
inline constexpr const char* kGenFillV3Sha256 = "9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a";
// The v4 LoRA (research_genfill_v4_20261109), loaded only by the region re-render. Not published:
// experimental, release_authorized false (configs/experiments/research_genfill_v4_1_20261111.json).
inline constexpr const char* kGenFillV4Sha256 = "95d707eb3e5628d7515187c0752018a23182f186edb060d8652f77a64466375c";
// SHA-256 of a file, cached in QSettings by path, size and modification time
// (a LoRA is ~100 MB and the setup is asked often). Empty when unreadable.
QString genfill_file_sha256(const std::filesystem::path& file);
// Whether `file` is the published v3 LoRA (or `expected`, for tests).
bool genfill_lora_is_v3(const std::filesystem::path& file, const QString& expected = QString::fromLatin1(kGenFillV3Sha256));
// The setup's one-line status, with the LoRA's identity (the dialog and Deliver use it).
QString genfill_ready_text(const GenFillSetup& s);

struct GenFillOptions {
    int tau_high = 250;   // 8-bit code: >= is a clipped highlight to fill (256 = none)
    int tau_low = -1;     // 8-bit code: <= is a crushed shadow to fill (-1 = none; v1.1 default off)
    int seed = 0;
    int steps = 30;
    double brightness = 0.0;  // v3: the artist's offset in stops over the estimated brightness (-3..+3)
    QString prompt_highlight, prompt_shadow;
    // RUDRA 1.1 (5 Nov 2026), priority 2. Each is sent only when not its default, so a fill
    // with the defaults runs the command line it always did.
    double prompt_strength = 1.0;   // how strongly a region prompt steers its region (0..2; 1 = as v3 was gated)
    double shape_gain = 1.0;        // detail strength: how much the drawn shape varies the brightness (0..2)
    QString region_json;            // the grade's reconstruction region (deliver reconstruction_region_json); "" = none
    // RUDRA 1.1 priority 5: a frame whose fill flickers for one frame is written as the master's own
    // (--fallback spikes). Off here so a bare GenFillOptions is v3; Deliver turns it on by default.
    bool fallback = false;
    // 7 Nov 2026: the model draws only the box around what the shot fills, at its working size
    // (--focus region): more detail where the fill is. Off = v3 (the whole frame, as gated).
    bool focus_region = false;
    QString glob;         // the master folder's frames to fill ("SH010.*.exr"); "" = every EXR there
    // 12 Nov 2026, experimental: AI fill v4, region re-render. The whole bright region is drawn again
    // from the source's codes (v3 fills only the fully clipped pixels), with the v4 LoRA and the
    // amended inference of 11 Nov: tau-high 240, feather 8, the source's colour with 0.3 of the
    // model's, the level 3.3 SDR whites, 0.8 of the previous frame's luminance kept. The SDR codes
    // inside the region are NOT kept. Off = v3. `source`, `source_start` and `condition` are filled
    // in when a fill starts (the shot, its first frame in the master, the route's inverse).
    bool rerender = false;
    QString source;                 // the movie, or the folder of stills, the master was made from
    int source_start = 0;           // the source frame of the master's first frame
    QString condition = QStringLiteral("bt1886");   // "bt1886" (the analytic route) or "aces" (the learned one)
    bool operator==(const GenFillOptions&) const = default;
};

// The command line (after the Python): -m rudra.genfill.run --master ... --json.
QStringList genfill_arguments(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out);

// RUDRA 1.1 (5 Nov 2026): prompt presets and a suggestion from the shot's clipped areas
// (rudra/genfill/prompts.py, no model). The command line after the Python, for one or more
// SDR frames, and its JSON read back.
QStringList genfill_suggest_arguments(const GenFillSetup& s, const GenFillOptions& o, const QStringList& frames);
struct PromptSuggestion {
    QString id, label, text, region;   // region "highlight" or "shadow": the box it fills
    QString reason;                    // why it was suggested; empty for a plain preset
    bool cached = false;               // its embedding is in the prompt cache: no text model to load
};
struct PromptSuggestions {
    std::vector<PromptSuggestion> suggested, presets;
    double clipped_fraction = 0.0;
};
std::optional<PromptSuggestions> parse_prompt_suggestions(const QByteArray& json, QString* why = nullptr);

// A new folder beside the master: <master>_aifill, else _aifill_2, _3 ... (nothing replaced).
QString genfill_out_dir(const QString& master);

// Every place the base model and the LoRA are looked for, in order (4 Nov 2026):
// beside the app, the repository, RUDRA_GENFILL_DIR, the model download folder,
// the Hugging Face caches (HF_HUB_CACHE, HF_HOME, ~/.cache/huggingface), then
// the training workstation's folders.
std::vector<std::filesystem::path> genfill_base_candidates(const std::filesystem::path& repo);
std::vector<std::filesystem::path> genfill_lora_candidates(const std::filesystem::path& repo);

// What the automatic setup (rudra/genfill/setup.py, 4 Nov 2026) runs with: the
// repository, its Python (the saved one, else the repo's .venv), and where the
// base model goes when none is found (the model download folder).
struct GenFillInstall {
    std::filesystem::path repo, python, base;
};
GenFillInstall genfill_install_plan();

// The AI fill's own Python (7 Nov 2026): "Set up automatically" on a computer with no Python for
// it runs genfill_bootstrap (.ps1 on Windows, .sh elsewhere: a package's python/ folder or the
// repository's scripts/), which puts uv, a private Python 3.12 and the fill's environment under
// genfill_bootstrap_root() (<AppLocalData>/genfill) and then rudra.genfill.setup. Nothing on the
// system changes. genfill_bootstrap_script is empty when neither has it.
std::filesystem::path genfill_bootstrap_root();
std::filesystem::path genfill_bootstrap_python();
std::filesystem::path genfill_bootstrap_script(const std::filesystem::path& repo);

// The setup: the saved paths while they exist, else the first candidate that
// does (and it is saved); nullopt and `why` when a part is missing.
std::optional<GenFillSetup> genfill_setup(QString* why = nullptr);
// The setup a fill with these options runs with: genfill_setup(), and for the region re-render
// the v4 LoRA in place of v3's (QSettings genfill/loraV4, <root>/rudra_genfill_v4.safetensors,
// then the training workstation's run folder; only a file that is v4 by its SHA-256).
std::optional<GenFillSetup> genfill_setup_for(const GenFillOptions& o, QString* why = nullptr);
std::filesystem::path genfill_v4_lora();   // empty when none is found
// AI fill v4 (region re-render) is parked (13 Nov 2026: it passes 09V and the held-out clips, not a second
// shot). The app shows and runs it only with RUDRA_GENFILL_V4=1 set, for development.
bool genfill_v4_enabled();

class GenFillJob {
public:
    using Event = std::function<void(const QJsonObject&)>;
    using Finished = std::function<void(bool ok, const QJsonObject& last)>;

    explicit GenFillJob(QObject* parent);
    ~GenFillJob();
    GenFillJob(const GenFillJob&) = delete;
    GenFillJob& operator=(const GenFillJob&) = delete;

    // Starts it; false (and `why`) if one is running or it cannot start.
    bool start(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out,
               Event event_cb, Finished finished_cb, QString* why = nullptr);
    void cancel();                 // asks it to stop at the next step (frames written so far stay); twice = kill
    bool running() const;
    QString out_dir() const { return out_; }

private:
    void on_stdout();
    void on_finished(int code);

    QObject* parent_;
    QProcess* proc_ = nullptr;
    JsonLines lines_;
    QString out_;
    QJsonObject last_;
    QString stderr_tail_;
    Event on_event_;
    Finished on_finished_;
    bool cancelled_ = false;
};

}  // namespace rudra::app
