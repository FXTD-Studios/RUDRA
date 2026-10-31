#pragma once
// The main window (Phase 3). Its menubar is the Studio's, built from
// engine/actions (the page's menus, labels, keys and check states, step 2);
// every action runs through run(id), which the session (step 3) takes over.
// The rails, toolbar and panels arrive with step 4.

#include <QMainWindow>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <QDialog>
#include <QImage>
#include <QPointer>
#include <QTimer>
#include <QTemporaryDir>
#include <QJsonObject>

#include <thread>

#include "rudra/core/composite.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/core/scopes.hpp"
#include "rudra/engine/headroom_preview.hpp"
#include "rudra/engine/master_job.hpp"
#include "rudra/engine/measure.hpp"
#include "rudra/engine/model_catalog.hpp"
#include "rudra/engine/playback_hold.hpp"
#include "rudra/engine/project.hpp"
#include "rudra/infer/self_test.hpp"
#include "rudra/engine/session.hpp"
#include "rudra/media/sequence.hpp"
#include "rudra/media/still_metadata.hpp"
#include "batch_queue.hpp"
#include "rudra/core/aces_v2_trims.hpp"
#include "deliver_choice.hpp"
#include "model_downloads.hpp"
#include "video_queues.hpp"

#include <atomic>
#include <chrono>

class QAction;
class QLabel;
namespace rudra::app { class GenFillJob; }
class QLineEdit;
class QMenu;
class QNetworkAccessManager;
class QPlainTextEdit;
class QPushButton;
class QSlider;
class QStackedWidget;

namespace rudra {
class FrameEngine;
class InferenceBackend;
struct ReadyFrame;
struct ModelConstants;
struct ViewerStatus;
class ViewerWindow;
}  // namespace rudra

namespace rudra::app {

class AcesEngine;
class CheckRow;
class IconButton;
class RegionEditor;
class SourcePanel;
class MaskPanel;
class ScopePlot;
class VectorscopeView;
class ScrubBar;
class Seg;

// The Open dialog's filter: stills, every movie the reader opens, projects.
QString open_file_filter();

class MainWindow : public QMainWindow {
public:
    // viewer: false builds the window without the QRhi viewer (tests, and
    // builds without Qt Shader Tools).
    // headroom_capable: the session and project reader accept the experimental
    // bt1886-headroom-v1 policy and movies preview through the headroom driver
    // (E1). Off in every public entry point; the workflow check and tests opt in.
    explicit MainWindow(bool viewer = true, bool headroom_capable = false);
    ~MainWindow() override;

    // A model package from a folder picker (or `preset`): use_model on it.
    void open_package(const QString& preset = {});

    // The checkpoint manager (Phase 3 step 10). The roots searched, the
    // catalog they give, and a switch that keeps the session: frames, grade,
    // undo and the frame on screen stay; the new model's fields replace the
    // old ones when they arrive. Files are verified, the backend opened and,
    // the first time this package runs on this backend, the goldens checked
    // (infer/self_test), all off the UI thread; a package that fails keeps
    // the model in use. `done` gets whether it switched and why not.
    void set_model_roots(std::vector<std::filesystem::path> roots);   // instead of package_roots() (tests)
    std::vector<std::filesystem::path> model_roots() const;
    void add_model_root(const std::filesystem::path& root);              // remembered between runs
    void rescan_models();
    const Catalog& catalog() const { return catalog_; }
    void use_model(const std::filesystem::path& package, std::optional<BackendChoice> choice = std::nullopt,
                   std::function<void(bool, QString)> done = {});
    bool loading_model() const { return loading_model_; }
    std::filesystem::path model_package() const { return manifest_ ? manifest_->root : std::filesystem::path(); }
    std::optional<BackendChoice> model_backend() const { return backend_choice_; }
    // Stand-ins for the runtimes (the tests): how a package is opened and tested.
    struct ModelHooks {
        std::function<Result<OpenedBackend>(const ModelManifest&, std::optional<BackendChoice>)> open;
        std::function<Result<SelfTestReport>(const ModelManifest&, InferenceBackend&)> test;
        bool verify_files = true;
    };
    void set_model_hooks(ModelHooks h) { hooks_ = std::move(h); }
    // What RUDRA does on a bare start: the first-run check once, then the
    // package used last (on its backend), else the catalog's pick.
    void boot();
    void open_model_manager();
    // Model downloads (29 Oct 2026, model_downloads.hpp): the list we publish, fetched
    // on demand; a finished download is added to the model folders and, for the learned
    // ACES 2.0 model, loaded when none is.
    ModelDownloads* model_downloads();
    // The learned ACES 2.0 package to load (remembered as aces/package); false and why when it cannot.
    bool use_aces_package(const std::filesystem::path& package, QString* why = nullptr);
    bool have_aces_package();           // a learned ACES 2.0 package and its engine are on this computer
    // Help > About (29 Oct 2026): the version, the model, the GPU, the links.
    void open_about();
    QString update_state() const { return update_state_; }   // "Up to date", "RUDRA x is available", "" unchecked
    // Deliver > Learned ACES 2.0 master… (aces_master.hpp): the learned ACES 2.0
    // model's masters, run by the ACES v2 rudra-native as a child process.
    void open_aces_master();
    // The learned ACES 2.0 model kept loaded (aces_engine.hpp): started at
    // launch when aces/preload is on (the default) and an engine and package are found.
    void preload_aces_engine();
    AcesEngine* aces_engine();
    // The main viewer shows the learned ACES 2.0 master of each frame, through
    // the kept-loaded engine, when it is ready and aces/viewer is on (default).
    bool aces_routed() const;
    void set_aces_viewer(bool on);
    bool aces_viewer() const { return aces_view_; }
    // The colorist's trims on the learned master (28 Oct 2026, core/aces_v2_trims.hpp):
    // in every viewer frame, Deliver render and batch job; kept between runs.
    const aces_v2::AcesTrims& aces_trims() const { return aces_trims_; }
    void set_aces_trims(const aces_v2::AcesTrims& t);
    // The Deliver tab's start timecode and reel for stills and sequences ("" = none;
    // a movie carries its own timecode and reel unless these are set).
    QString render_timecode() const;
    // The monitoring transform on the learned route (28 Oct 2026): "scene" (the
    // master's scene light, as before), "sdr100" or "hdr1000" (the ACES 2.0
    // output, in display light, with the view peak set to it). Kept between runs.
    QString aces_monitor() const { return aces_monitor_; }
    void set_aces_monitor(const QString& m);
    // The display the SDR was made for (28 Oct 2026 review): bt1886 (what the model
    // reads), srgb or gamma22 (re-encoded to BT.1886 codes first, the same light).
    QString aces_source_display() const { return aces_source_display_; }
    void set_aces_source_display(const QString& s);
    // The playback cache: how many frames are in, and whether it plays in real time now.
    int aces_cached_frames() const;
    bool aces_realtime_playing() const { return aces_rt_playing_; }
    bool aces_playing() const { return aces_playing_; }   // through the model, frame by frame
    QString aces_cache_key() const;   // the shot, trims, monitor and source display it was made with
    void aces_rt_stop();
    // Per-shot trims (28 Oct 2026 review): remembered with each shot (the movie, the
    // folder or the still); a new shot keeps the trims in hand. Copy / paste between shots.
    void copy_aces_trims();
    void paste_aces_trims();
    QString shot_trim_key() const;
    std::optional<aces_v2::AcesTrims> remembered_trims(const QString& key) const;
    // Reference stills (28 Oct 2026): the learned frame on screen kept (Ctrl+Alt+G),
    // then compared in the baseline's place (Baseline, Wipe and B show it).
    int grab_reference_still();                   // its index, -1 when there is no learned frame
    void set_reference_compare(bool on, int index = -1);
    bool reference_compare() const { return ref_compare_; }
    bool reference_shown() const { return aces_ref_shown_; }   // the frame on screen is compared with it
    int reference_count() const { return int(ref_stills_.size()); }
    const NetworkLinearImage* reference_still() const;
    const std::vector<float>& aces_codes() const { return aces_codes_; }
    QString render_reel() const;
    int aces_shown() const { return aces_shown_; }      // the frame the ACES route last showed (-1: none)
    QString frame_info() const { return frame_info_; }
    // The Deliver tab (27 Oct 2026, deliver_choice.hpp): the source (the learned
    // ACES 2.0 model when it is ready, else the analytic reconstruction), still,
    // sequence or MOV, EXR 32/16-bit or DPX 10-bit, ACES 2065-1 or ACEScg, LogC4,
    // ACEScct or PQ, and a 10- or 16-bit movie. Render runs the choice.
    const DeliverChoice& deliver_choice() const { return deliver_; }   // what was chosen
    DeliverChoice deliver_effective() const;     // what Render writes for this shot and source
    bool deliver_aces() const;                   // the learned model renders the files
    // The learned model is chosen and set up (loading, failed or ready): the
    // Deliver tab then never falls back to the analytic look (28 Oct 2026 review).
    bool deliver_learned_wanted() const;
    // One choice by its key ("source", "output", "image", "gamut", "encoding",
    // "bits", "codec", "keep_exr", "fps"; the AI fill: "ai_fill", "ai_tau_high",
    // "ai_tau_low", "ai_seed", "ai_steps", "ai_prompt_highlight", "ai_prompt_shadow"),
    // as the controls set it.
    void set_deliver(const QString& key, const QString& value);
    void deliver_render();                       // the Render button (action "master")
    bool aces_delivering() const { return aces_deliver_id_ != 0; }
    // The AI fill (layer 3, genfill_job.hpp) of a learned EXR render: runs after the
    // render when chosen, or on any master folder; a Render click while it runs stops it.
    bool genfill_running() const;
    bool start_genfill(const QString& master_dir, QString* why = nullptr);
    // The same with its own options and output base (<base>_aifill) and a
    // callback when it ends (true when done): the batch's AI fill (29 Oct 2026).
    bool start_genfill_with(const QString& master_dir, const GenFillOptions& options, const QString& out_base,
                            std::function<void(bool done, QString text)> finished, QString* why = nullptr);
    QString genfill_last_out() const { return genfill_last_out_; }
    // The review of an AI fill folder (the last one, else one picked); false and why when there is none.
    bool review_genfill(const QString& dir = {}, QString* why = nullptr);
    // The batch queue (28 Oct 2026, batch_queue.hpp): shots rendered one after
    // another through the learned ACES 2.0 model. Its file is the setting
    // batch/file, else batch_queue.json in the app's data folder.
    BatchQueue& batch_queue();
    BatchRunner& batch_runner();
    bool batch_running() const { return batch_runner_ && batch_runner_->running(); }
    void open_batch();                            // Deliver > Batch queue… (Ctrl+Shift+B)
    // The shot on screen, with the Deliver tab's choice, name and start frame
    // (Ctrl+B); `dir` overrides the Deliver tab's render folder.
    bool batch_add_current(QString* why = nullptr, const QString& dir = {});
    bool batch_start(QString* why = nullptr);
    // An EDL into the batch (edl.hpp): one shot per matched video event, only its
    // frames plus `handles`, named <event>_<reel>, as a sequence (or the Deliver
    // tab's MOV). Returns the shots added; `report` says what did not match.
    int batch_import_edl(const QString& edl, const QString& media_folder, int handles, const QString& render_dir,
                         QString* report = nullptr);
    QString render_dir() const;                   // the Deliver tab's render folder
    int render_start() const;                     // its start frame
    // What Render sends the learned engine now (empty with the reason when it cannot).
    QJsonObject aces_deliver_request(QString* why = nullptr) const;
    // The export sheet (the Pro boards): format, frames, destination, then Master
    // EXR, or for a movie an HDR10, HLG or ProRes export as a queue job.
    void open_export_sheet();
    // Phase 4 step 11: a movie as a shot, and its exports as queue jobs.
    bool shot_is_video() const { return shot_video_.has_value(); }
    const std::optional<FrameSequence>& shot_video() const { return shot_video_; }
    QString shot_title() const;                  // the movie, the folder or the still
    std::string frame_name(std::size_t i) const; // what the Frames rail and a master call frame i
    // HDR10, HLG or ProRes 422 HQ of the movie on screen, to the Deliver tab's
    // folder and name: checked against this ffmpeg, written as a queue file
    // (rudra/batch.py's format) and run; false with the reason logged when it cannot be.
    bool queue_video_export(const QString& format);
    // A folder of frames to a movie (16 Oct 2026): the frames mastered as a
    // linear Rec.2020 EXR sequence into <name>_frames beside the movie, then
    // encoded as `rudra-native deliver` encodes them (hdr10, hlg, prores422hq
    // or prores4444; PQ, no audio, the Deliver tab's fps). False with the reason
    // shown when it cannot start.
    bool shot_is_frame_folder() const { return shot_kind_ == "folder" && frames_.size() > 1; }
    bool export_frames_to_movie(const QString& format, double fps = 24.0);
    // The movie's name, without its suffix: the Deliver tab's name (else the
    // folder's) and the format, in the characters a render name may use.
    QString frames_movie_base(const QString& format) const;
    bool encoding_movie() const { return encoding_movie_; }
    void open_queue_window();
    VideoQueues& video_queues() { return *queues_; }
    void resume_queue(const std::filesystem::path& queue);
    void forget_queue(const std::filesystem::path& queue);
    void open_first_run();
    // A still, or a folder of frames: both are a sequence to the engine.
    void open_source(const QString& preset = {}, bool folder = false);
    // The page's addFiles (Open, a drop of images): appended to the frames,
    // the first of them shown ("added N frames").
    void add_files(const std::vector<std::filesystem::path>& files);
    // What a drop brings (step 11): a model package folder is used, a folder
    // is opened as a shot, image files are added.
    void open_paths(const QStringList& paths);
    // Open recent: the shots opened last, newest first, kept between runs.
    QStringList recent_sources() const;
    void fill_recent();
    // Settings kept between runs: the window's place and size, the workspace,
    // the rails, the inspector tab, the container and the Render fields.
    void save_settings() const;
    void restore_settings();
    // The sheet on screen (show_sheet), if any.
    QDialog* sheet() const { return sheet_; }
    // What Deliver > Copy writes (deliveryRecord()).
    std::string delivery_text() const;

    // The QAction for an action id (engine/actions), or nullptr.
    QAction* action(std::string_view id) const;
    // Run an action by id, as a menu item or its key does.
    void run(std::string_view id);
    // Enable each action as the page's refreshMenu() does, for the state now.
    void refresh_enabled();

    // Why an action is off in this build or at this step ("" when it is live).
    QString pending_reason(std::string_view id) const;

    // The page's shell (shell.js): the workspace and the inspector tab.
    void set_workspace(const QString& mode);     // "full" or "simple"
    QString workspace() const { return workspace_; }
    void show_tab(const QString& tab);           // "rec", "grade", "check" (Simple), "deliver"
    QString tab() const { return tab_; }
    // The tab on screen: Simple has Look (rec), Check and Deliver; Full has Reconstruct (rec), Grade and Deliver.
    QString shown_tab() const;
    // Simple's Deliver (29 Oct 2026): "master" (ACES 2065-1 EXR), "hdr10" (HEVC 10-bit PQ 1000 nits),
    // "hlg" (ProRes 422 HQ HLG) or "custom" (the full What to write card).
    void apply_deliver_preset(const QString& key);
    QString deliver_preset() const;
    // A line in the Log, as the page's log() writes it.
    void log(const QString& line);
    // The scopes of the frame on screen (drawScopes, drawVector); cleared
    // when there is none.
    void show_scopes(const ScopeData& s, std::optional<double> maxcll, const std::vector<std::uint8_t>& vector_rgba);
    void clear_scopes();

    // A frame to show and measure (the engine's, or a test's): the viewer
    // gets it now, the measurements follow on their own thread.
    // `own_shadow_weight`: the frame's own when `scalars` carries a held one (playing).
    // `pixel`: the static hold's step for the viewer (playing held; off otherwise).
    void present_frame(SdrImage sdr, Fields fields, FrameScalars scalars, ModelConstants model, FrameHeader header,
                       std::optional<float> own_shadow_weight = std::nullopt, PlaybackHold::PixelStep pixel = {});
    // Measure the frame on screen now, on this thread (the tests).
    void measure_now();
    const FrameMeasure* measurement() const { return measure_.get(); }
    // The probe at a frame pixel, as a hover with Probe on shows it; nullopt
    // clears it. `global` places the floating box.
    void probe_pixel(std::optional<std::pair<double, double>> px, QPoint global = {});
    // Calibrate (3.2): arm a slot (-1 disarms) so the next click on the
    // picture reads the SDR code there (a 5x5 mean of R, G and B) into it.
    void arm_calibration(int slot);
    void calibration_picked(double x, double y);
    // The clip fill (12 Oct 2026): arm a peak so the next click on the
    // picture picks a clipped region and asks for its nits; the options the
    // frame engine was started with, to start it again when they change.
    void arm_clip_peak(bool on);
    void clip_peak_picked(double x, double y);
    ClipFillOptions session_clip_options() const;
    // Reference (3.4): asks for the graded HDR of the current frame (or takes
    // `path`), fits it and hands the fit to the session; the message of a
    // refused file goes to the Source card's line. False when refused.
    bool load_reference(const std::filesystem::path& path);
    // Painted masks (3.3): arms a band for painting (-1 none), which puts the
    // viewer in paint mode; strokes come back through paint_stroke.
    void arm_mask(int band);
    void paint_stroke(double x0, double y0, double x1, double y1, bool erase);
    // The SDR code at a frame pixel, as the pick reads it; nullopt without a frame.
    std::optional<int> sdr_code_at(double x, double y) const;
    QWidget* probe_box() const { return probe_box_; }

    // The Deliver tab's Master EXR (the page's master()): the render plan from
    // the Render fields, then a background job over the frames. `prepare` and
    // `count` stand in for the opened frames and the network (the tests).
    void master(PrepareMasterFrame prepare = {}, std::size_t count = 0);
    bool mastering() const { return master_job_ && master_job_->running(); }
    // What a master renders with now (the session, the Deliver checks, the model).
    MasterRequest master_request() const;
    const SdrImage* frame_sdr() const;
    const std::vector<std::filesystem::path>& frames() const { return frames_; }
    ViewerWindow* viewer() const { return viewer_; }
    // The preview's long side (the Studio's 1600; 0 for full size). Applies to frames opened after it.
    void set_preview_max_side(int side) { preview_max_side_ = side; }
    int preview_max_side() const { return preview_max_side_; }
    // The frame on screen (the tests): its fields and which frame it is.
    const Fields* frame_fields() const;
    const FrameScalars* frame_scalars() const;   // as shown (held while playing)
    int current_index() const { return current_; }
    // Requested and delivered indices differ during asynchronous scrubbing.
    int presented_index() const;
    std::size_t frame_count() const { return frames_.size(); }

    // Projects (engine/project.hpp): the shot, the model and the grade in a
    // .rudra file. Open applies the grade, switches to the project's model
    // when it is another one and opens the shot; what is missing is logged and
    // the rest still opens. `adopt` makes the file the window's project (Save
    // writes it); the autosaved session is opened without.
    Project current_project() const;
    bool save_project_to(const std::filesystem::path& path);   // false, with the reason logged
    bool open_project(const std::filesystem::path& path, bool adopt = true);
    const std::filesystem::path& project_path() const { return project_path_; }
    bool project_dirty() const { return dirty_; }
    // Autosave: the session to `file` `delay_ms` after it last changed, and on
    // close. The file a previous run left becomes the one File > Reopen last
    // session opens (previous-session.rudra beside it).
    void set_autosave_path(const std::filesystem::path& file, int delay_ms = 4000);
    const std::filesystem::path& autosave_path() const { return autosave_path_; }
    const std::filesystem::path& reopen_path() const { return reopen_path_; }
    bool autosave_now();

    // Updates: the releases feed (GitHub's API; a file:// URL in the tests),
    // read without blocking. `interactive` answers in a dialog; otherwise a
    // newer release is logged and shown in the status bar. `done` gets the line logged.
    static QString default_updates_feed();
    void set_updates_feed(const QString& url) { updates_feed_ = url; }
    void check_for_updates(bool interactive, std::function<void(QString)> done = {});
    // At start: once a day unless turned off (QSettings updates/auto, or RUDRA_NO_UPDATE_CHECK).
    void maybe_check_for_updates();
    QString update_url() const { return update_url_; }   // the newer release's page, empty for none

    // The reconstruction (the 10 Oct 2026 review): the analytic inverse by
    // default, with no model run for the view or a master; the model's
    // reconstruction (AI-assisted) only when turned on (Reconstruct > Use the
    // model, kept in the settings and in a project). A package may still be
    // loaded in the background so turning it on is quick; it runs only when on.
    bool using_model() const { return use_model_; }
    void set_use_model(bool on, bool remember = true);   // remember: the default for the next start
    // The constants frames are composed with: the package's when the model
    // is on, kAnalyticConstants otherwise.
    ModelConstants frame_constants() const;

    // Playback with the model (10 Oct 2026): inference runs ahead of the
    // playhead (FrameEngine::set_playing), at `side` pixels on the long side
    // while playing (0: full size) with full size where it stops; a movie is
    // decoded in order by one ffmpeg (MovieFeeder); `realtime` keeps time by
    // skipping frames the model has not reached.
    bool playing() const { return play_.isActive() || headroom_playing_ || aces_playing_; }
    void set_playback_side(int side);
    int playback_side() const { return play_state_->side.load(); }
    void set_playback_realtime(bool on);
    // Temporal consistency (10 Oct 2026): hold the shadow weight and the
    // anchor's hold steady within a shot while playing, and in movie exports
    // (--temporal-smoothing); on by default (QSettings video/steady).
    void set_steady_shots(bool on);
    bool steady_shots() const { return steady_; }
    double shown_anchor_hold() const { return shown_hold_; }   // the hold the viewer was given
    struct PlaybackStats {
        int shown = 0, skipped = 0;
        double seconds = 0.0;
    };
    PlaybackStats playback_stats() const;
    bool frame_proxy() const { return frame_proxy_; }   // the frame on screen is a playback proxy

    // Help: the five steps, the user guide's address.
    static QString user_guide_url();
    void show_getting_started();
    void show_getting_started_once();   // once per edition of the steps (QSettings help/gettingStarted)
    // The first-run tips (29 Oct 2026): the welcome card, then four coach marks
    // the first time a shot is on screen (QSettings help/tipsWelcomed, help/tipsDone).
    void show_tips_once();
    void show_welcome_tips();
    void start_tips();                  // the four tips now (Help > Show tips); switches to Simple
    class CoachMarks* coach_marks() const { return coach_; }
    bool tips_pending() const { return tips_pending_; }

    // The page's state: grade, undo, peak, wipe, container (engine/session).
    Session& session() { return session_; }
    const Session& session() const { return session_; }
    CompositeParams composite() const { return session_.composite_params(); }
    const std::string& container() const { return session_.container; }

    // ---- Headroom preview (E1, 17 Oct 2026; experimental, capability-gated) ----
    // When the window is capable, the grade's policy is bt1886-headroom-v1 and
    // the shot is a movie, frames come from HeadroomPreviewDriver instead of the
    // legacy SDR engine: paused frames, playback, grade changes, undo/redo and
    // project restore all route through it. A refused grade or a failure is
    // shown as such; legacy pixels are never substituted. Exports refuse the
    // policy until they support it (E2). See archive/docs/HEADROOM_APP_INTEGRATION_2026-10-17.md.
    bool headroom_capable() const { return headroom_capable_; }
    bool headroom_routed() const;                       // frames come from the headroom driver
    Result<void> set_headroom_preview(bool on);         // the grade's policy: one undo step
    // Reconstruct > Keep super-whites: checks the shot, says what it turns off
    // (asked first when the window is visible), opens the window's headroom
    // support if needed (remembered for the next start) and sets the policy;
    // on again, back to the usual reading. Refusals are in problem().
    void toggle_super_whites();
    QString problem() const { return problem_; }
    const HeadroomShown* headroom_frame() const { return headroom_shown_ ? &*headroom_shown_ : nullptr; }
    QString headroom_problem() const { return headroom_problem_; }   // why nothing is shown, or empty
    HeadroomPreviewStats headroom_stats() const;
    using HeadroomPortFactory = std::function<Result<std::unique_ptr<HeadroomPreviewPort>>()>;
    void set_headroom_port_factory(HeadroomPortFactory f) { headroom_factory_ = std::move(f); }   // tests
    // E2: with the policy on, a movie export (HDR10, HLG, ProRes) runs the
    // transactional headroom export (engine/headroom_export.hpp) off the UI
    // thread instead of the legacy queue; asking again while it runs cancels it.
    bool exporting_headroom() const { return headroom_export_running_; }
    QString headroom_export_result() const { return headroom_export_result_; }   // the published file, or why not

protected:
    // The keys the page's keydown takes before its map: the wipe nudge,
    // Escape, and B held to flip (the viewer takes them itself when it has
    // the focus).
    void keyPressEvent(class QKeyEvent* e) override;
    void keyReleaseEvent(class QKeyEvent* e) override;
    void dragEnterEvent(class QDragEnterEvent* e) override;
    void dragLeaveEvent(class QDragLeaveEvent* e) override;
    void dropEvent(class QDropEvent* e) override;
    void closeEvent(class QCloseEvent* e) override;
public:
    // What would be stopped by quitting now ("the Deliver render", "the batch", ...).
    QStringList running_work() const;
    // Asks each to stop after the frame in hand (written whole); the owners wait on destruction.
    void stop_running_work();
    // Tests answer the quit question here (true = stop and quit); else a dialog.
    std::function<bool(const QString&)> quit_question;
protected:
    bool eventFilter(QObject* o, QEvent* e) override;   // a shot row in the sidebar reopens its shot

private:
    void build_menus();
    void apply_tooltips();
    void mark_dirty();
    void update_title();
    // The shot as a project names it: "folder", "movie" or "files", and its paths.
    std::string shot_kind_;
    std::vector<std::filesystem::path> shot_sources_;
    void open_shot(const std::string& kind, const std::vector<std::filesystem::path>& sources, int frame);
    struct PendingShot {
        std::string kind;
        std::vector<std::filesystem::path> sources;
        int frame = 0;
    };
    std::optional<PendingShot> pending_shot_;   // a project's shot, opened when its model is in
    // The project's model while it is not yet the one in use (loading, or
    // queued behind another load): what the project still names.
    struct ProjectModel {
        std::filesystem::path package;
        std::string backend;
    };
    std::optional<ProjectModel> project_model_;
    bool from_project_ = false;   // use_model called for project_model_ (any other call replaces it)
    void use_project_model();
    void model_settled();   // a load finished either way: a queued project model goes next
    // Before the session is replaced: false when the user cancels. Asks only
    // in a window on screen with unsaved work worth asking about.
    bool confirm_replace(bool closing);
    bool save_interactive();   // Save, or Save as for an untitled session
    std::filesystem::path project_path_, autosave_path_, reopen_path_;
    bool dirty_ = false, restoring_ = false;
    bool use_model_ = false;
    bool clip_pick_armed_ = false;
    ClipFillOptions engine_clip_;   // what start_engine gave the frames
    bool frame_proxy_ = false;
    struct PlayState {
        std::atomic<bool> playing{false};
        std::atomic<int> side{960};
    };
    std::shared_ptr<PlayState> play_state_ = std::make_shared<PlayState>();
    struct FeederSlot;
    std::shared_ptr<FeederSlot> feeder_slot_;
    bool realtime_ = true;
    bool steady_ = true;
    PlaybackHold play_hold_;
    double shown_hold_ = 1.0;
    std::chrono::steady_clock::time_point play_t0_;
    int play_from_ = 0, play_shown_ = 0, play_skipped_ = 0, play_shown_window_ = 0;
    QTimer play_status_;
    void play_tick();
    double play_fps() const;
    void start_feeder(int at);
    void stop_playback();
    void halt_playback();   // stop without the summary or the refine (a new engine, a close, a render)
    bool needs_model() const { return use_model_ && !backend_; }   // frames wait for a package
    class QLabel* mode_label_ = nullptr;   // the status bar's "Analytic" / "AI-assisted: <model>"
    void show_mode();
    QString problem_;   // why the last open or save failed, for its dialog
    QTimer autosave_timer_;
    QString updates_feed_ = default_updates_feed(), update_url_;
    QNetworkAccessManager* net_ = nullptr;
    std::unique_ptr<ModelDownloads> downloads_;
    std::optional<bool> have_aces_package_;   // cached: reset by a rescan and a download
    QString update_state_;
    void bind_handlers();
    void show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows);
    void sync_checks();
    void sync_ui();
    // Simple (29 Oct 2026): Look, Check, Deliver; what Full adds is hidden here.
    void build_simple_parts();
    void sync_simple();
    void fill_empty_recent();
    void sync_check_summary();   // Check's four numbers, from the HDR on screen
    class CoachMarks* coach_ = nullptr;
    bool tips_pending_ = false;
    bool deliver_custom_ = false;                // Simple's Deliver: Custom… picked
    std::optional<bool> simple_scopes_;          // Simple: the Scopes button's choice on this tab
    QStringList empty_recent_;                   // the recent shots the empty viewer lists
    QPushButton* view_menu_btn_ = nullptr;
    QLabel* hold_hint_ = nullptr;
    void session_changed(std::uint32_t what);
    void build_ui(bool with_viewer);
    QWidget* build_left_rail();
    QWidget* build_centre(bool with_viewer);
    QWidget* build_right_rail();
    QWidget* build_pipe();
    QWidget* build_toolbar();
    // The Pro chrome: the toolbar's shot title, the sidebar's shot rows and
    // collections, the viewer's badges, the filmstrip and its clipping lane.
    void refresh_library();
    void update_badges();
    void note_frame_in_strip(int index);
    int delivered_ = 0;
    std::optional<std::pair<QString, bool>> pending_source_;   // a shot asked for while the model loads
    QImage first_thumb_;
    QImage thumb_of_first() const;
    QPointer<QDialog> export_sheet_, queue_window_;
    std::optional<FrameSequence> shot_video_;
    std::unique_ptr<VideoQueues> queues_ = std::make_unique<VideoQueues>();
    VideoQueueOptions queue_options() const;
    void save_queues() const;

    void start_engine(std::vector<std::filesystem::path> frames, int at = 0);
    void inspect_source();
    struct LoadedModel;
    void adopt_model(std::shared_ptr<LoadedModel> loaded, const std::function<void(bool, QString)>& done);
    void set_model_pills();
    void close_frames();
    void step_to(int i);
    void toggle_play();
    void frame_ready(const ReadyFrame& f, const ModelConstants& model);
    void show_status();
    void schedule_stats();
    void run_stats();
    // `index`: the frame it measured (-1: the one on screen, not held).
    void apply_measure(std::shared_ptr<const FrameMeasure> m, int index = -1, int hold_shot = -1);
    void show_own_values();
    void update_pipe();
    // The compare line under the Compare bar: what RUDRA changed and how much
    // of it the view peak shows (core/readouts compare_text).
    void update_compare();
    void fill_rows(QWidget* ms, const std::vector<MetricRow>& rows);

    std::map<std::string, QAction*, std::less<>> actions_;
    std::map<std::string, std::function<void()>, std::less<>> handlers_;
    std::map<std::string, QString, std::less<>> pending_;

    ViewerWindow* viewer_ = nullptr;
    QString workspace_ = "full", tab_ = "rec";
    bool guides_on_ = false;
    // The widgets the state is drawn into (the page's ids).
    QWidget *rail_left_ = nullptr, *rail_right_ = nullptr, *scopes_ = nullptr, *scopes_dock_ = nullptr;
    QPushButton* scopes_btn_ = nullptr;
    QStackedWidget* viewer_stack_ = nullptr;
    QWidget* panels_ = nullptr;          // the inspector's tab page (#ipanels)
    std::vector<QWidget*> pages_;        // Reconstruct, Grade, Deliver
    SourcePanel* source_panel_ = nullptr;
    MaskPanel* mask_panel_ = nullptr;
    QSlider* anchor_knee_ = nullptr;
    QLabel* anchor_knee_val_ = nullptr;
    Seg *view_mode_ = nullptr, *view_layer_ = nullptr, *zoom_seg_ = nullptr, *mode_seg_ = nullptr, *tabs_ = nullptr,
        *ws_ = nullptr;
    CheckRow *preserve_ = nullptr, *anchor_ = nullptr, *carry_chroma_ = nullptr;
    QSlider *strength_ = nullptr, *peak_ = nullptr, *exposure_ = nullptr;
    QLabel* exposure_val_ = nullptr;
    QLabel *strength_val_ = nullptr, *peak_val_ = nullptr, *shot_count_ = nullptr, *frames_empty_ = nullptr,
           *tc_ = nullptr, *src_info_ = nullptr, *zoom_val_ = nullptr, *container_field_ = nullptr,
           *primaries_field_ = nullptr, *ckpt_ = nullptr, *device_ = nullptr, *lamp_ = nullptr,
           *region_count_ = nullptr, *view_transform_ = nullptr, *compare_line_ = nullptr;
    QWidget* compare_bar_ = nullptr;
    QPushButton *changes_btn_ = nullptr, *invented_btn_ = nullptr, *compare_fit_ = nullptr;
    double compare_fit_ev_ = 0.0;
    QPushButton *guide_btn_ = nullptr, *probe_btn_ = nullptr, *btn_master_ = nullptr, *btn_reprocess_ = nullptr;
    IconButton *i_media_ = nullptr, *i_scopes_ = nullptr, *i_inspector_ = nullptr, *btn_prev_ = nullptr,
               *btn_play_ = nullptr, *btn_next_ = nullptr;
    ScrubBar* scrub_ = nullptr;
    ScopePlot *wave_ = nullptr, *hist_ = nullptr;
    class ParadeView* parade_ = nullptr;
    // The learned route's playback cache (28 Oct 2026 review): the first play renders
    // and keeps each frame (at the playback size); once every frame of the shot is in,
    // play runs in real time at the shot's rate, for judging flicker.
    std::vector<std::shared_ptr<const NetworkLinearImage>> aces_cache_;
    QString aces_cache_key_;
    std::size_t aces_cache_bytes_ = 0;
    QTimer* aces_rt_timer_ = nullptr;
    bool aces_rt_playing_ = false;
    RegionEditor* regions_ = nullptr;
    VectorscopeView* vector_ = nullptr;
    QLineEdit* seq_path_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    std::vector<QWidget*> notes_;
    // The frame on screen and its measurements (computeStats).
    struct Current {
        SdrImage sdr;
        Fields fields;
        FrameScalars scalars;
        ModelConstants model;
        std::shared_ptr<const NetworkLinearImage> baseline;
        SourceCurve baseline_source = SourceCurve::Unknown;   // the source the baseline was built with (3.1)
        std::vector<CalibrationPoint> baseline_calibration;    // and the anchors (3.2)
        ReferenceFit baseline_reference;                        // and the reference (3.4)
        FrameHeader header;
        int index = -1;   // the frame's index in the shot
        float own_shadow_weight = 1.0f;   // before the playback hold
        int hold_shot = -1;               // PlaybackHold::shot() when it was shown
    };
    std::shared_ptr<const Current> current_frame_;
    std::shared_ptr<const FrameMeasure> measure_;
    QTimer stats_timer_;
    int stats_gen_ = 0;
    bool stats_running_ = false, stats_again_ = false;
    class ClipBar* clip_bar_ = nullptr;
    QWidget* probe_box_ = nullptr;
    std::unique_ptr<MasterJob> master_job_;
    std::shared_ptr<std::mutex> backend_mutex_ = std::make_shared<std::mutex>();   // the engine and the master share it
    void master_finished(const MasterOutcome& o, const QString& folder);
    struct FramesMovie {
        QString format;                      // hdr10, hlg, prores422hq, prores4444
        std::filesystem::path frames_dir, output;
        std::string name;
        double fps = 24.0;
    };
    std::optional<FramesMovie> frames_movie_;   // set while its frames are being mastered
    void encode_frames_movie(FramesMovie job);
    std::thread encode_worker_;
    std::atomic<bool> encode_cancel_{false};
    bool encoding_movie_ = false;
    bool probe_on_ = false;
    Session session_;
    QTimer play_;
    int current_ = 0;
    bool waiting_ = false;
    bool frame_ready_ok_ = false;
    QString frame_info_;
    QString runtimes_;
    Catalog catalog_;
    std::optional<std::vector<std::filesystem::path>> roots_override_;
    ModelHooks hooks_;
    bool loading_model_ = false;
    std::thread model_worker_;
    std::thread stats_worker_;   // the measurement off the UI thread; joined, never left running
    std::optional<BackendChoice> backend_choice_;
    QPointer<QDialog> manager_, first_run_, sheet_, aces_master_;
    std::shared_ptr<AcesEngine> aces_engine_;   // shared_ptr: the type stays incomplete here
    QMenu* recent_menu_ = nullptr;
    void remember_source(const QString& path);
    void copy_text(const QString& label, const std::string& text);
    void set_drop_hot(bool hot);
    std::unique_ptr<ModelManifest> manifest_;
    std::unique_ptr<InferenceBackend> backend_;   // used only from the engine's worker
    std::vector<std::filesystem::path> frames_;
    std::unique_ptr<FrameEngine> engine_;         // declared after the backend: destroyed first
    int engine_gen_ = 0;                          // which engine a delivered frame came from
    // Each frame's size as decoded, before the preview downscale (the header's source_resolution).
    struct SourceSizes {
        std::mutex mu;
        std::vector<std::pair<int, int>> size;
        std::vector<std::optional<StillColour>> colour;
    };
    std::shared_ptr<SourceSizes> source_sizes_;
    int preview_max_side_ = 1600;                 // media/still.hpp kPreviewMaxSide

    // Headroom preview (E1). The driver is destroyed before the engine.
    bool headroom_capable_ = false;
    bool headroom_playing_ = false;
    // The learned ACES 2.0 route of the main viewer.
    bool aces_view_ = true;
    bool aces_busy_ = false, aces_playing_ = false;
    int aces_next_ = -1;            // a frame asked for while one was running
    int aces_shown_ = -1;
    std::unique_ptr<QTemporaryDir> aces_tmp_;
    void aces_request(int index);
    DeliverChoice deliver_;
    aces_v2::AcesTrims aces_trims_;
    QString aces_monitor_ = "scene";
    QString aces_source_display_ = "bt1886";
    std::optional<aces_v2::AcesTrims> trims_clipboard_;
    QString trims_shot_;   // the shot whose trims are in hand
    std::vector<float> aces_codes_;
    struct RefStill {
        QString label;
        std::shared_ptr<const NetworkLinearImage> image;
    };
    std::vector<RefStill> ref_stills_;
    int ref_index_ = -1;
    bool ref_compare_ = false, aces_ref_shown_ = false;
    void sync_reference_ui();
    QTimer* aces_trim_timer_ = nullptr;   // a slider drag asks for the frame again, once it settles
    void sync_aces_trims();
    // The ACES route's frame for the readouts and scopes (the viewer's convention).
    std::shared_ptr<const NetworkLinearImage> aces_model_img_, aces_base_img_;
    std::unique_ptr<BatchQueue> batch_;
    std::unique_ptr<BatchRunner> batch_runner_;
    QPointer<QDialog> batch_window_;
    QString render_name_for_fill_;
    // The Render card's Nuke-style path (29 Oct 2026): its padding and separator.
    int render_padding_ = 4;
    QString render_separator_ = ".";
    bool syncing_render_file_ = false;
    void render_file_edited();      // the path typed: the folder, name, padding follow
    void browse_render_file();
    void sync_render_file();        // the folder, name or file type changed: the path follows   // the Deliver render's name when it wrote into a shared folder
    void batch_changed();
    int aces_deliver_id_ = 0;       // the engine's master request running for the Deliver tab
    std::shared_ptr<GenFillJob> genfill_;   // shared_ptr: the type stays incomplete here
    QString genfill_last_out_;
    void sync_deliver();            // the Deliver tab's controls and Signal card from deliver_
    void load_deliver_settings();
    void aces_deliver();
    void aces_deliver_finished(const QJsonObject& o, const QString& out);
    void aces_after_frame(const QJsonObject& o, int index, const QString& model_path, const QString& base_path);
    HeadroomPortFactory headroom_factory_;
    std::unique_ptr<HeadroomPreviewDriver> headroom_;
    std::optional<HeadroomShown> headroom_shown_;
    QString headroom_problem_;
    void route_headroom();                       // after anything that decides or feeds the route
    void headroom_request(int index, bool play);
    void headroom_present(const HeadroomShown& f);   // inside the bridge's present: no driver calls
    void headroom_after_present();
    void headroom_failed(const Error& e);
    void headroom_finished();
    void stop_headroom();
    void clear_legacy_frame();
    bool refuse_headroom_export(const QString& what);
    bool start_headroom_export(const QString& format);
    std::thread headroom_export_worker_;
    std::atomic<bool> headroom_export_cancel_{false};
    bool headroom_export_running_ = false;
    QString headroom_export_result_;
};

}  // namespace rudra::app
