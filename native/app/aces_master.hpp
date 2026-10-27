#pragma once
// Deliver > Learned ACES 2.0 master… (26 Oct 2026).
//
// The learned ACES 2.0 model (core/aces_v2.hpp, a package written by
// tools/export_native_v2.py) as a job the app starts: one 32-bit float ACEScg
// EXR per still with the rudra.reconstructed channel and a provenance.json,
// the same masters `rudra-native aces-v2` writes. The app runs that command
// (built with OpenColorIO 2.5.2 and LibTorch / ONNX Runtime, which the app
// itself does not link) as a child process: the window stays responsive, a
// cancel stops it, and the masters are byte for byte the verified CLI's.
//
// The helpers below hold no widgets, so the tests run them directly.

#include <QDialog>
#include <QProcessEnvironment>
#include <QStringList>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProcess;
class QProgressBar;
class QPushButton;
class QSlider;
class QTemporaryDir;

namespace rudra::app {

class MainWindow;
class AcesEngine;

// An ACES v2 package the app found (aces_v2::read_manifest accepted it).
struct AcesPackageInfo {
    std::filesystem::path root;
    std::string g1_sha256, checkpoint_sha256;
    bool has_onnx = false;
    bool luminance_guard = false;      // appearance_constraint.forbid_new_negative_y
    bool release_authorized = false;
};
// Each root itself and its sub-folders (one level), in order, without repeats.
std::vector<AcesPackageInfo> find_aces_packages(const std::vector<std::filesystem::path>& roots);

// What `rudra-native aces-v2` takes: a folder's stills (sorted), one still, or
// one video (every frame, <stem>_<NNNNNN>.exr).
bool is_aces_still(const std::filesystem::path& p);
bool is_aces_video(const std::filesystem::path& p);
std::vector<std::filesystem::path> collect_aces_stills(const std::filesystem::path& input);
// The first still whose EXR name another already takes (case-insensitive), if any.
std::optional<std::filesystem::path> aces_name_collision(const std::vector<std::filesystem::path>& stills);

// Where the job runs. The label is what the dialog lists.
struct AcesRunTarget {
    std::string key, label, runtime, device;
};
const std::vector<AcesRunTarget>& aces_run_targets();

struct AcesRunSettings {
    std::filesystem::path cli, package, input, out;
    std::string runtime = "libtorch", device = "cuda";
    bool luminance_guard = true;
    int ocio_threads = 8;
    // Camera-log copies beside the EXRs (cli/aces_v2_log.hpp): "" for none, else
    // "logc4"/"acescct", and the formats ("tiff16", "dpx10", "prores4444xq"...).
    std::string log_curve, log_formats;
};
QStringList aces_cli_arguments(const AcesRunSettings& s);
// What the "Also write" list offers: key (log_formats), label. Key "" = EXR only.
struct AcesLogChoice {
    std::string key, label;
};
const std::vector<AcesLogChoice>& aces_log_choices();

// One line the command prints per finished frame:
//   "<source> -> <exr>  <ms> ms  remaining_display_errors=<n>"
struct AcesProgressLine {
    std::string source, exr;
    double milliseconds = 0;
    std::size_t remaining_display_errors = 0;
};
std::optional<AcesProgressLine> parse_aces_progress(const std::string& line);

// The run's provenance.json, summed up for the dialog.
struct AcesRunSummary {
    std::size_t frames = 0, remaining_display_errors = 0, new_negative_luminance = 0;
    bool appearance_pass = false, luminance_guard = false;
    double mean_ms = 0, mean_reconstructed_fraction = 0;
    std::string device, runtime;
    std::string log_curve;                 // "ARRI LogC4" when camera-log copies were written
    std::vector<std::string> log_files;    // what they are ("16-bit TIFF per frame (lossless)", ...)
};
Result<AcesRunSummary> read_aces_provenance(const std::filesystem::path& out_dir);
std::string aces_summary_text(const AcesRunSummary& s);

// The child's environment: the app's, with the folders in the setting
// aces/path_prepend (QStringList) put first on PATH, e.g. LibTorch's lib/
// for a development rudra-native whose runtime DLLs are not beside it.
QProcessEnvironment aces_child_environment();

// The command beside the app (rudra-native[.exe]); empty if there is none.
std::filesystem::path default_aces_cli();
// Whether that command has the aces-v2 master (runs "aces-v2 --help": its own usage, exit 64), and if
// not, what happened (did not start, a DLL missing, timed out, other output).
struct AcesCliProbe {
    bool ok = false;
    QString detail;
};
AcesCliProbe probe_aces_cli(const std::filesystem::path& cli, int timeout_ms = 30000);
bool cli_has_aces_v2(const std::filesystem::path& cli, int timeout_ms = 30000);

class AcesMasterDialog : public QDialog {
public:
    // roots: where packages are looked for; input: the frames to suggest;
    // engine: the app's loaded model (none: the dialog keeps one of its own).
    AcesMasterDialog(QWidget* parent, std::vector<std::filesystem::path> roots, std::filesystem::path input,
                     AcesEngine* engine = nullptr);
    ~AcesMasterDialog() override;

    void refresh_packages();
    void set_input(const std::filesystem::path& input);
    void set_cli(const std::filesystem::path& cli);
    bool ready(QString* why = nullptr) const;   // everything set for Start
    void start();
    void cancel();
    bool running() const;
    AcesEngine* engine() const { return engine_; }
    // The preview: which frame, how bright, what is shown (model, plain, split, mask, source).
    void preview_frame(int index);
    void set_preview_ev(double ev);
    void set_preview_mode(const QString& key);
    bool preview_busy() const { return preview_busy_ || view_busy_; }

private:
    void ensure_engine();
    void on_engine_change();
    void request_preview();
    void request_view();
    void start_with_engine(const AcesRunSettings& s);
    void update_state();
    void on_output();
    void on_finished(int code, bool crashed);
    void show_package();
    AcesRunSettings settings() const;

    std::vector<std::filesystem::path> roots_;
    std::vector<AcesPackageInfo> packages_;
    std::vector<std::filesystem::path> stills_;   // the stills, or the one video
    std::size_t total_frames_ = 0;                // frames to master (a video's frame count)
    bool video_ = false;
    QString frames_text(std::size_t n) const;
    QComboBox *package_ = nullptr, *target_ = nullptr, *log_format_ = nullptr;
    QLabel *package_note_ = nullptr, *frames_note_ = nullptr, *cli_note_ = nullptr, *status_ = nullptr;
    QLineEdit *input_ = nullptr, *out_ = nullptr, *cli_ = nullptr;
    QCheckBox* guard_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    QPushButton *start_ = nullptr, *cancel_ = nullptr, *open_out_ = nullptr;
    QProcess* proc_ = nullptr;
    std::string pending_;          // stdout not yet split into lines
    std::size_t done_ = 0;
    bool cancelled_ = false, cli_ok_ = false;
    bool hold_ = false;            // a run's summary is on show
    // The model kept loaded, and the preview through it.
    AcesEngine* engine_ = nullptr;
    std::unique_ptr<AcesEngine> own_engine_;
    std::unique_ptr<QTemporaryDir> tmp_;
    QLabel *engine_note_ = nullptr, *image_ = nullptr, *preview_stats_ = nullptr, *ev_label_ = nullptr;
    QPushButton* reload_ = nullptr;
    QComboBox *preview_frame_ = nullptr, *mode_ = nullptr;
    QSlider* ev_ = nullptr;
    bool preview_busy_ = false, view_busy_ = false, preview_again_ = false, view_again_ = false;
    bool engine_master_ = false;
    int views_ = 0;
};

}  // namespace rudra::app
