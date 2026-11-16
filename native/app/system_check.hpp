#pragma once
// Check this computer (16 Nov 2026, 1.2.5). The 1.2.4 testers' machines needed things nobody told
// them about: one opened the retired Beta 2 model an older version had left behind, one had to find
// the model downloads by hand, and AI fill said its base model was not installed only when it was
// asked to run. This window is where all of that is said and done, once:
//
//   - whether this computer can run RUDRA (system, GPU, memory, disk, ffmpeg, the model list), each
//     line OK, a note, a warning or a blocker, in plain words;
//   - what RUDRA can install and what is missing: the learned ACES 2.0 model (recommended, ticked),
//     AI fill's model and its base model and Python (optional, only where the GPU can run them),
//     each with its size and licence and one button; "Install selected" downloads the ticked ones,
//     one after the other, while the user goes on working;
//   - what RUDRA put right by itself: the retired Beta 2 packages it no longer opens (they can be
//     moved aside here), and "Use the model" turned off when nothing could load.
//
// It opens after the welcome on the first start and after an update, and any time from the
// toolbar's Check button (a dot on it when something wants a look) and Help > Check this computer.
// Nothing is downloaded until the user asks for it (docs/PRIVACY.md); the model list is read while
// the window is open, as File > Model packages does.

#include <QDialog>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <map>
#include <optional>
#include <vector>

#include "rudra/engine/model_catalog.hpp"

class QCheckBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QVBoxLayout;

namespace rudra::app {

class MainWindow;
class ModelDownloads;

// ---- The retired Beta 2 models -------------------------------------------------------------
// sdr2hdr_* and the shadow and temporal checkpoints of Beta 2: non-commercial, and retired since
// 1.2.0 (the learned ACES 2.0 model replaced them). 1.0.0 to 1.0.2 put sdr2hdr_image_v8 beside the
// app, and an update installed over it left it there, so a later start could open it again.
// Named as tools/check_package_weights.py names them.
bool is_beta2_name(const QString& name);
bool is_beta2_package(const std::filesystem::path& package);
// Takes them out of the catalog (the model list, the start's pick) and returns their folders.
std::vector<std::filesystem::path> drop_beta2(Catalog& c);
// Moves a retired package to <its folder's parent>/.retired-beta2/ (a dot folder: never scanned),
// where it can be put back by hand. False, and why, when it cannot be moved (a folder only an
// administrator can change, say); it is not opened either way.
bool archive_beta2(const std::filesystem::path& package, QString* said);

// ---- This computer -------------------------------------------------------------------------
enum class CheckLevel { Ok, Info, Warn, Fail };

struct CheckLine {
    QString id;      // "os", "gpu", "aifill", "memory", "disk", "ffmpeg", "network"
    QString title;   // "System"
    QString value;   // "Windows 11 (build 26100), x64"
    QString note;    // what it means for the user; empty when there is nothing to add
    CheckLevel level = CheckLevel::Ok;
};

struct GpuFact {
    QString name;
    double vram_gb = 0.0;   // dedicated memory; 0 when none is reported
};

// What the check reads of the machine. The tests fill one in by hand.
struct MachineFacts {
    bool windows = false, mac = false;
    QString os;                                   // "Windows 11", "macOS 15.1"
    int windows_build = 0;                        // 19045: Windows 10 22H2; 22000 and up: Windows 11
    QString cpu_arch;                             // QSysInfo::currentCpuArchitecture(): "x86_64", "arm64"
    std::optional<std::vector<GpuFact>> gpus;     // nullopt: not read on this system or build
    double ram_gb = 0.0;                          // 0: not read
    double free_gb = -1.0;                        // where models are kept; -1: not read
    QString models_root;
    bool ffmpeg = true;
};

MachineFacts read_machine_facts(const QString& models_root);
std::vector<CheckLine> judge_machine(const MachineFacts& f);
bool machine_blocked(const std::vector<CheckLine>& lines);
// What AI fill needs of the machine (a 16 GB NVIDIA GPU; on a Mac, 16 GB of memory): empty when it
// has it, else why not, in a sentence.
QString aifill_machine_problem(const MachineFacts& f);
// genfill_setup()'s reason as a user reads it: what is missing, and where it is set up (this window),
// instead of a path and a script name ("no Wan2.1-VACE-1.3B model at ... (run scripts/genfill_setup.sh)").
QString genfill_missing_text(const QString& why);

// ---- The window ----------------------------------------------------------------------------
class SystemCheckDialog : public QDialog {
public:
    // first_run: opened by the welcome; closing it then shows the getting-started tips.
    SystemCheckDialog(MainWindow* w, bool first_run);
    void refresh();                          // the machine read again, every row rebuilt
    void set_facts(const MachineFacts& f);   // the tests: a machine of their own
    QString report() const;                  // Copy report: what the window says, as text
    void install_selected();                 // the ticked downloads, one after the other
    QStringList queued() const { return queue_; }
    const std::vector<CheckLine>& machine() const { return machine_; }

protected:
    void done(int r) override;

private:
    void build_parts();
    void download_next();
    void on_downloads_changed();
    void update_progress();

    MainWindow* w_;
    ModelDownloads* md_;
    bool first_run_;
    std::optional<MachineFacts> facts_override_;
    MachineFacts facts_;
    std::vector<CheckLine> machine_;
    QStringList queue_;
    QString current_;                        // the id this window started downloading, "" when none
    QStringList results_;                    // what the downloads it started came to
    QString parts_shape_;                    // the rows' states: rebuilt only when they change
    QVBoxLayout* machine_box_ = nullptr;
    QVBoxLayout* parts_box_ = nullptr;
    QLabel* summary_ = nullptr;
    QLabel* progress_text_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPushButton* install_ = nullptr;
    std::map<QString, QCheckBox*> ticks_;    // model id -> its box
};

}  // namespace rudra::app
