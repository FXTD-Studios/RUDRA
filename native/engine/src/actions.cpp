#include "rudra/engine/actions.hpp"

#include <algorithm>
#include <map>
#include <string>

namespace rudra {
namespace {

using O = ActionOrigin;
using E = EnableRule;

const std::vector<ActionSpec> kActions = {
    // File
    {"open", "Open frames…", "O", "", O::Studio, E::Always, ""},
    {"close", "Close all frames", "", "", O::Studio, E::AnyFrames, ""},
    {"master", "Master EXR", "M", "", O::Studio, E::CanMaster, ""},
    // Edit
    {"undo", "Undo", "Z", "", O::Studio, E::CanUndo, ""},
    {"redo", "Redo", "Y", "", O::Studio, E::CanRedo, ""},
    {"reset-recon", "Reset reconstruction", "", "", O::Studio, E::Always, ""},
    {"reset-regions", "Reset region EV", "", "", O::Studio, E::Always, ""},
    // Clip
    {"first", "First frame", "Home", "", O::Studio, E::ManyFrames, ""},
    {"prev", "Previous frame", ",", "", O::Studio, E::ManyFrames, ""},
    {"next", "Next frame", ".", "", O::Studio, E::ManyFrames, ""},
    {"last", "Last frame", "End", "", O::Studio, E::ManyFrames, ""},
    {"play", "Play / pause", "Space", "", O::Studio, E::ManyFrames, ""},
    // Reconstruct
    {"mode-all", "All", "", "mode:all", O::Studio, E::Always, ""},
    {"mode-highlights", "Highlights", "", "mode:highlights", O::Studio, E::Always, ""},
    {"mode-shadows", "Shadows", "", "mode:shadows", O::Studio, E::Always, ""},
    {"mode-off", "Off", "", "mode:off", O::Studio, E::Always, ""},
    {"preserve", "Preserve outside masks", "", "preserve", O::Studio, E::Always, ""},
    {"strength-down", "Weaker residual", "[", "", O::Studio, E::Always, ""},
    {"strength-up", "Stronger residual", "]", "", O::Studio, E::Always, ""},
    // Measure
    {"copy-metrics", "Copy measurements as JSON", "", "", O::Studio, E::HasMetrics, ""},
    {"copy-scopes", "Copy scope data as JSON", "", "", O::Studio, E::HasScopes, ""},
    {"remeasure", "Re-measure this frame", "", "", O::Studio, E::AnyFrames, ""},
    // Deliver
    {"container-aces", "Container: ACES 2065-1", "", "container:aces", O::Studio, E::Always, ""},
    {"container-linear", "Container: linear Rec.2020", "", "container:linear", O::Studio, E::Always, ""},
    {"copy-delivery", "Copy delivery metadata", "", "", O::Studio, E::AnyFrames, ""},
    // Window
    {"rail-left", "Frames rail", "", "railLeft", O::Studio, E::Always, ""},
    {"rail-right", "Reconstruction rail", "", "railRight", O::Studio, E::Always, ""},
    {"scopes", "Scopes", "", "scopesOpen", O::Studio, E::Always, ""},
    {"zoom-fit", "Fit to window", "", "zoom:fit", O::Studio, E::Always, ""},
    {"zoom-actual", "Actual pixels", "", "zoom:actual", O::Studio, E::Always, ""},
    // Help
    {"shortcuts", "Keyboard shortcuts", "?", "", O::Studio, E::Always, ""},
    {"about", "About RUDRA Studio", "", "", O::Studio, E::Always, ""},
    // Bound by the page's bind() rather than listed in ACTIONS: W.
    {"wipe", "Wipe", "W", "wipe", O::Studio, E::Always, ""},

    // Native only. The page opens a package with the server and keeps its
    // view modes and guides on the viewer toolbar; the app also puts them in
    // a menu, on keys the page leaves free. The view peak is the session's
    // peak slider, as on the page.
    {"open-package", "Open model package…", "", "", O::Native, E::Always, "Ctrl+Shift+O"},
    {"models", "Model packages…", "", "", O::Native, E::Always, "Ctrl+Shift+M"},
    {"recent-clear", "Clear recent", "", "", O::Native, E::Always, ""},
    {"first-run", "Check the display and the model…", "", "", O::Native, E::Always, ""},
    {"open-folder", "Open folder of frames…", "", "", O::Native, E::Always, "Ctrl+Alt+O"},
    {"quit", "Quit", "", "", O::Native, E::Always, "Ctrl+Q"},
    {"view-image", "Image", "", "view:0", O::Native, E::Always, ""},
    {"view-false-colour", "False colour", "", "view:1", O::Native, E::Always, ""},
    {"view-difference", "Difference", "", "view:2", O::Native, E::Always, ""},
    {"guides-action", "Action safe (90 %)", "", "guides:action", O::Native, E::Always, "G"},
    {"guides-title", "Title safe (80 %)", "", "guides:title", O::Native, E::Always, "Shift+G"},
    {"guides-centre", "Centre cross", "", "guides:centre", O::Native, E::Always, ""},
    {"aspect-none", "No aspect mask", "", "aspect:0", O::Native, E::Always, ""},
    {"aspect-2.39", "2.39", "", "aspect:2.39", O::Native, E::Always, ""},
    {"aspect-1.85", "1.85", "", "aspect:1.85", O::Native, E::Always, ""},
    {"aspect-16:9", "16:9", "", "aspect:16:9", O::Native, E::Always, ""},
    {"aspect-4:3", "4:3", "", "aspect:4:3", O::Native, E::Always, ""},
    {"aspect-1:1", "1:1", "", "aspect:1:1", O::Native, E::Always, ""},
    // Roadmap 3.1: the Source picker. Ids are core/source_curve.hpp's.
    {"source-unknown", "Source: unknown (ACES inverse)", "", "source:unknown", O::Native, E::Always, ""},
    {"source-bt1886", "Source: BT.1886 (display, the default)", "", "source:bt1886", O::Native, E::Always, ""},
    // 14 Oct 2026, opt-in: BT.1886 with the measured real-SDR blacks below code 40.
    {"source-bt1886_shadows", "Source: BT.1886, real-SDR shadows", "", "source:bt1886_shadows", O::Native, E::Always, ""},
    // 13 Oct 2026: the master exposure, in thirds of a stop.
    {"exposure-down", "Exposure: down a third of a stop", "", "", O::Native, E::Always, "Ctrl+Shift+Down"},
    {"exposure-up", "Exposure: up a third of a stop", "", "", O::Native, E::Always, "Ctrl+Shift+Up"},
    {"exposure-reset", "Exposure: reset to 0", "", "", O::Native, E::Always, "Ctrl+Shift+0"},
    {"source-aces", "Source: ACES", "", "source:aces", O::Native, E::Always, ""},
    {"source-hable", "Source: filmic (Hable)", "", "source:hable", O::Native, E::Always, ""},
    {"source-agx", "Source: AgX", "", "source:agx", O::Native, E::Always, ""},
    {"source-camera_log", "Source: camera log", "", "source:camera_log", O::Native, E::Always, ""},
    {"source-clip", "Source: Rec.709, plain clip", "", "source:clip", O::Native, E::Always, ""},
    // Projects (engine/project.hpp): the shot, the model and the grade in a
    // .rudra file; the app autosaves the session and can reopen it.
    // The reconstruction (the 10 Oct review): the analytic inverse by default,
    // the model only when asked for. Checked: the model runs.
    {"use-model", "Use the model (AI-assisted)", "", "useModel", O::Native, E::Always, ""},
    // The clip fill (12 Oct 2026, core/clip_fill.hpp): what the analytic
    // reconstruction puts where the SDR clipped, and the artist's peak for a
    // clipped region (armed, then a click on the picture asks for its nits).
    {"clip-fill", "Fill clipped highlights", "", "clipFill", O::Native, E::Always, ""},
    {"clip-peak", "Set the peak of a clipped region…", "", "", O::Native, E::AnyFrames, ""},
    {"clip-peaks-clear", "Clear clipped peaks", "", "", O::Native, E::AnyFrames, ""},
    // Super-whites (21 Oct 2026, docs/SUPER_WHITES_REVIEW_2026-10-16.md): a
    // limited-range BT.709 movie read with its detail above white kept
    // (bt1886-headroom-v1) instead of clipped and filled. Experimental.
    {"keep-super-whites", "Keep super-whites (experimental)", "", "superWhites", O::Native, E::Always, ""},
    // Playback with the model (10 Oct 2026): the size the model runs at
    // while playing (full where the playhead stops), and whether playback
    // keeps time by skipping frames the model has not reached.
    {"playback-side-0", "Model at full size", "", "playback:0", O::Native, E::Always, ""},
    {"playback-side-960", "Model at 960 px (faster)", "", "playback:960", O::Native, E::Always, ""},
    {"playback-side-640", "Model at 640 px (fastest)", "", "playback:640", O::Native, E::Always, ""},
    {"playback-realtime", "Keep time (skip frames when behind)", "", "playbackRealtime", O::Native, E::Always, ""},
    // Temporal consistency (10 Oct 2026): the per-frame values held within a
    // shot while playing and in movie exports.
    {"steady-shots", "Hold steady within a shot", "", "steadyShots", O::Native, E::Always, ""},
    {"open-project", "Open project…", "", "", O::Native, E::Always, "Ctrl+O"},
    {"save-project", "Save project", "", "", O::Native, E::Always, "Ctrl+S"},
    {"save-project-as", "Save project as…", "", "", O::Native, E::Always, "Ctrl+Shift+S"},
    {"reopen-session", "Reopen last session", "", "", O::Native, E::Always, ""},
    // Help for someone new to the app.
    {"getting-started", "Getting started", "", "", O::Native, E::Always, ""},
    {"user-guide", "User guide", "", "", O::Native, E::Always, "F1"},
    {"check-updates", "Check for updates…", "", "", O::Native, E::Always, ""},
    {"open-log-folder", "Open log folder", "", "", O::Native, E::Always, ""},
    // 26 Oct 2026: the learned ACES 2.0 model's masters (app/aces_master.hpp).
    {"aces-master", "Learned ACES 2.0 master…", "", "", O::Native, E::Always, "Ctrl+Shift+E"},
};

const std::vector<MenuSpec> kMenus = {
    {"File",
     {"open", "open-folder", ">Open recent", "recent-clear", "<", "open-package", "models", "close", "-", "open-project",
      "save-project", "save-project-as", "reopen-session", "-", "master", "-", "quit"}},
    {"Edit", {"undo", "redo", "-", "reset-recon", "reset-regions"}},
    {"Clip",
     {"first", "prev", "next", "last", "-", "play", "-", ">While playing", "playback-side-0", "playback-side-960",
      "playback-side-640", "-", "playback-realtime", "<", "steady-shots"}},
    {"Reconstruct",
     {"use-model", "-", "clip-fill", "clip-peak", "clip-peaks-clear", "keep-super-whites", "-", ">Source", "source-bt1886", "source-bt1886_shadows", "source-unknown", "source-aces", "source-hable", "source-agx", "source-camera_log", "source-clip", "<", "-",
      "exposure-down", "exposure-up", "exposure-reset", "-", "mode-all", "mode-highlights", "mode-shadows", "mode-off", "-", "preserve", "-", "strength-down", "strength-up"}},
    {"Measure", {"copy-metrics", "copy-scopes", "-", "remeasure"}},
    {"Deliver", {"master", "aces-master", "-", "container-aces", "container-linear", "-", "copy-delivery"}},
    {"View",
     {"view-image", "view-false-colour", "view-difference", "-", "wipe", "-", ">Guides", "guides-action",
      "guides-title", "guides-centre", "-", "aspect-none", "aspect-2.39", "aspect-1.85", "aspect-16:9", "aspect-4:3",
      "aspect-1:1", "<"}},
    {"Window", {"rail-left", "rail-right", "scopes", "-", "zoom-fit", "zoom-actual"}},
    {"Help",
     {"getting-started", "user-guide", "-", "shortcuts", "about", "first-run", "-", "check-updates", "open-log-folder"}},
};

// The keydown handler: the map, then Space, 1 to 4 and W, which it tests on
// their own.
const std::map<std::string, std::string_view, std::less<>> kKeys = {
    {" ", "play"},        {",", "prev"},        {".", "next"},          {"1", "mode-all"},
    {"2", "mode-highlights"}, {"3", "mode-shadows"}, {"4", "mode-off"},  {"?", "shortcuts"},
    {"End", "last"},      {"Home", "first"},    {"M", "master"},         {"O", "open"},
    {"P", "preserve"},    {"W", "wipe"},        {"Y", "redo"},           {"Z", "undo"},
    {"[", "strength-down"}, {"]", "strength-up"}, {"m", "master"},       {"o", "open"},
    {"p", "preserve"},    {"w", "wipe"},        {"y", "redo"},           {"z", "undo"},
};

const std::vector<ShortcutRow> kSheet = {
    {"B (hold)", "Flip to the inverse-ACES baseline"},
    {"W", "Wipe: baseline left, RUDRA right. Drag the image to move it."},
    {"← →", "Nudge the wipe (Shift for fine)"},
    {"O", "Open frames"},
    {"M", "Master EXR"},
    {", / .", "Previous / next frame"},
    {"Home / End", "First / last frame"},
    {"Space", "Play / pause"},
    {"[ / ]", "Weaker / stronger residual"},
    {"1 2 3 4", "Recovery: all, highlights, shadows, off"},
    {"P", "Preserve outside masks"},
    {"Z / Y", "Undo / redo"},
    {"?", "This list"},
};

}  // namespace

const std::vector<ActionSpec>& action_specs() { return kActions; }

const ActionSpec* find_action(std::string_view id) {
    const auto it = std::find_if(kActions.begin(), kActions.end(), [&](const ActionSpec& a) { return a.id == id; });
    return it == kActions.end() ? nullptr : &*it;
}

const std::vector<MenuSpec>& menu_specs() { return kMenus; }

std::string_view action_for_key(std::string_view key) {
    const auto it = kKeys.find(key);
    return it == kKeys.end() ? std::string_view{} : it->second;
}

std::vector<std::string_view> mapped_keys() {
    std::vector<std::string_view> k;
    for (const auto& [key, act] : kKeys) k.push_back(key);
    return k;
}

double key_nudge_step(bool shift) { return shift ? 0.01 : 0.05; }

const std::vector<ShortcutRow>& shortcut_sheet() { return kSheet; }

}  // namespace rudra
