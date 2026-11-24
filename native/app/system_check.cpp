#include "system_check.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QOperatingSystemVersion>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QStandardPaths>
#include <QStorageInfo>
#include <QStyle>
#include <QSysInfo>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <system_error>

#include "genfill_job.hpp"
#include "genfill_review.hpp"
#include "main_window.hpp"
#include "model_downloads.hpp"
#include "paths.hpp"
#include "widgets.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/gpu_adapters.hpp"
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

QLabel* text(const QString& t, const QString& role, QWidget* parent, const QString& id = {}) {
    auto* l = new QLabel(t, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    l->setWordWrap(true);
    l->setTextInteractionFlags(Qt::TextSelectableByMouse);
    return l;
}

QString level_key(CheckLevel l) {
    switch (l) {
        case CheckLevel::Ok: return QStringLiteral("ok");
        case CheckLevel::Info: return QStringLiteral("info");
        case CheckLevel::Warn: return QStringLiteral("warn");
        case CheckLevel::Fail: return QStringLiteral("fail");
    }
    return {};
}

QString level_word(CheckLevel l) {
    switch (l) {
        case CheckLevel::Ok: return QStringLiteral("OK");
        case CheckLevel::Info: return QStringLiteral("Note");
        case CheckLevel::Warn: return QStringLiteral("Warning");
        case CheckLevel::Fail: return QStringLiteral("Cannot run");
    }
    return {};
}

// The rows of a rebuild replace the old ones at once: taken out of the window (setParent) before they
// are deleted, so nothing finds a row of the last build by its name in between (deleteLater waits for
// the event loop).
void clear_layout(QVBoxLayout* box) {
    while (QLayoutItem* it = box->takeAt(0)) {
        if (QWidget* w = it->widget()) {
            w->hide();
            w->setParent(nullptr);
            w->deleteLater();
        }
        delete it;
    }
}

// The AI fill model files (kind "genfill"): v4 is shown only where v4 is turned on.
bool is_v4(const RemoteModel& m) {
    return m.id.contains(QStringLiteral("v4"), Qt::CaseInsensitive) || m.folder.contains(QStringLiteral("v4"), Qt::CaseInsensitive);
}

bool listed_here(const RemoteModel& m) {
    if (m.channel == QStringLiteral("legacy")) return false;
    if (m.kind == QStringLiteral("aces_v2")) return true;
    if (m.kind == QStringLiteral("genfill")) return !is_v4(m) || genfill_v4_enabled();
    if (m.kind == QStringLiteral("florence")) return true;   // 1.2.6: Suggest a prompt's Florence-2 (MIT)
    return false;   // reconstruction packages: optional, in File > Model packages
}

}  // namespace

// ---- The retired Beta 2 models -----------------------------------------------------------------

bool is_beta2_name(const QString& name) {
    const QString n = name.toLower();
    for (const char* marker : {"sdr2hdr_", "shadow_v1", "shadow_s2", "shadow_s3", "temporal_v1"})
        if (n.contains(QLatin1String(marker))) return true;
    return false;
}

bool is_beta2_package(const fs::path& package) {
    if (package.empty()) return false;
    if (is_beta2_name(from_path(package.filename()))) return true;
    const auto m = read_manifest(package);   // a package renamed: its manifest still says what it is
    return m && (is_beta2_name(QString::fromStdString(m->name)) || is_beta2_name(QString::fromStdString(m->source_file)));
}

std::vector<fs::path> drop_beta2(Catalog& c) {
    std::vector<fs::path> retired;
    std::vector<CatalogEntry> kept;
    std::optional<std::size_t> chosen;
    for (std::size_t i = 0; i < c.entries.size(); ++i) {
        CatalogEntry& e = c.entries[i];
        const bool beta2 = is_beta2_name(from_path(e.package.filename())) ||
                           (e.manifest && (is_beta2_name(QString::fromStdString(e.manifest->name)) ||
                                           is_beta2_name(QString::fromStdString(e.manifest->source_file))));
        if (beta2) {
            retired.push_back(e.package);
            continue;
        }
        if (c.chosen && *c.chosen == i) chosen = kept.size();
        kept.push_back(std::move(e));
    }
    c.entries = std::move(kept);
    c.chosen = chosen;   // a retired default is no default: pick() takes the first readable one left
    return retired;
}

bool archive_beta2(const fs::path& package, QString* said) {
    const QString from = from_path(package);
    const QString name = from_path(package.filename());
    const QString shelf = QDir(from_path(package.parent_path())).filePath(QStringLiteral(".retired-beta2"));
    QString to = QDir(shelf).filePath(name);
    for (int n = 2; QFileInfo::exists(to); ++n) to = QDir(shelf).filePath(name + QStringLiteral(" (%1)").arg(n));
    if (!QDir().mkpath(shelf) || !QDir().rename(from, to)) {
        if (said)
            *said = QStringLiteral("%1 could not be moved (the folder may need an administrator). It is not opened "
                                   "either way; delete it by hand if you like: %2").arg(name, QDir::toNativeSeparators(from));
        return false;
    }
    if (said) *said = QStringLiteral("%1 moved aside to %2").arg(name, QDir::toNativeSeparators(to));
    return true;
}

// ---- This computer -----------------------------------------------------------------------------

MachineFacts read_machine_facts(const QString& models_root) {
    MachineFacts f;
    f.models_root = models_root;
    f.cpu_arch = QSysInfo::currentCpuArchitecture();
#if defined(_WIN32)
    f.windows = true;
    const QOperatingSystemVersion v = QOperatingSystemVersion::current();
    f.windows_build = v.microVersion() > 0 ? v.microVersion() : 0;
    f.os = v.majorVersion() >= 10 ? (f.windows_build >= 22000 ? QStringLiteral("Windows 11") : QStringLiteral("Windows 10"))
                                  : QSysInfo::prettyProductName();
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) f.ram_gb = double(ms.ullTotalPhys) / kGiB;
#elif defined(__APPLE__)
    f.mac = true;
    f.os = QSysInfo::prettyProductName();
    std::uint64_t mem = 0;
    std::size_t len = sizeof mem;
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) f.ram_gb = double(mem) / kGiB;
#else
    f.os = QSysInfo::prettyProductName();
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) f.ram_gb = double(pages) * double(page) / kGiB;
#endif
#ifdef RUDRA_APP_VIEWER
    if (const auto found = gpu_adapters()) {
        f.gpus.emplace();
        for (const auto& a : *found) f.gpus->push_back({QString::fromStdString(a.name), a.vram_gb});
    }
#endif
    // The disk the models go to: the folder, or the nearest one above it that exists.
    QString dir = models_root;
    while (!dir.isEmpty() && !QFileInfo::exists(dir)) {
        const QString up = QFileInfo(dir).absolutePath();
        if (up == dir) break;
        dir = up;
    }
    if (!dir.isEmpty()) {
        const QStorageInfo si(dir);
        if (si.isValid() && si.isReady()) f.free_gb = double(si.bytesAvailable()) / kGiB;
    }
    // The package's own ffmpeg is first on the PATH from the start (platform/tools.hpp).
    f.ffmpeg = !QStandardPaths::findExecutable(QStringLiteral("ffmpeg")).isEmpty();
    return f;
}

QString aifill_machine_problem(const MachineFacts& f) {
    if (f.mac) {
        if (!f.cpu_arch.startsWith(QStringLiteral("arm"))) return QStringLiteral("AI fill needs Apple silicon.");
        if (f.ram_gb > 0 && f.ram_gb < 15.5)
            return QStringLiteral("AI fill needs a Mac with 16 GB of memory or more; this one has %1 GB.").arg(std::lround(f.ram_gb));
        return {};
    }
    if (!f.gpus) return {};   // not read here: the setup checks the GPU itself
    QString name;
    double best = 0.0;
    for (const auto& g : *f.gpus)
        if (g.name.contains(QStringLiteral("NVIDIA"), Qt::CaseInsensitive) && g.vram_gb >= best) {
            best = g.vram_gb;
            name = g.name;
        }
    if (name.isEmpty()) return QStringLiteral("AI fill needs an NVIDIA GPU with 16 GB of memory (CUDA), and none was found.");
    // A 16 GB card reports a little under 16 (15.6 to 15.9): 15 is the line.
    if (best < 15.0) return QStringLiteral("AI fill needs 16 GB of GPU memory; the %1 has %2 GB.").arg(name).arg(std::lround(best));
    return {};
}

std::vector<CheckLine> judge_machine(const MachineFacts& f) {
    std::vector<CheckLine> out;
    const bool x64 = f.cpu_arch == QStringLiteral("x86_64");
    const bool arm = f.cpu_arch.startsWith(QStringLiteral("arm"));

    CheckLine os{QStringLiteral("os"), QStringLiteral("System"), f.os, {}, CheckLevel::Ok};
    if (f.windows) {
        os.value += f.windows_build > 0 ? QStringLiteral(" (build %1)").arg(f.windows_build) : QString();
        os.value += x64 ? QStringLiteral(", x64") : QStringLiteral(", ") + f.cpu_arch;
        if (f.windows_build > 0 && f.windows_build < 10240) {
            os.level = CheckLevel::Fail;
            os.note = QStringLiteral("RUDRA needs Windows 10 or 11.");
        } else if (arm) {
            os.level = CheckLevel::Warn;
            os.note = QStringLiteral("Windows on Arm runs RUDRA's x64 build through emulation: it works, more slowly, and the "
                                     "learned model may run on the CPU.");
        } else if (!x64) {
            os.level = CheckLevel::Fail;
            os.note = QStringLiteral("RUDRA for Windows is a 64-bit (x64) program.");
        } else if (f.windows_build > 0 && f.windows_build < 19045) {
            os.level = CheckLevel::Warn;
            os.note = QStringLiteral("This Windows 10 is older than 22H2, the last version Microsoft updates. RUDRA is "
                                     "tested on Windows 10 22H2 and Windows 11.");
        }
    } else if (f.mac) {
        os.value += arm ? QStringLiteral(", Apple silicon") : QStringLiteral(", Intel");
        if (!arm) {
            os.level = CheckLevel::Fail;
            os.note = QStringLiteral("RUDRA for macOS needs Apple silicon (M1 or later).");
        }
    } else {
        os.level = CheckLevel::Info;
        os.note = QStringLiteral("RUDRA is released for Windows and macOS; here it runs as built from source.");
    }
    out.push_back(os);

    CheckLine gpu{QStringLiteral("gpu"), QStringLiteral("GPU"), {}, {}, CheckLevel::Ok};
    if (!f.gpus) {
        gpu.level = CheckLevel::Info;
        gpu.value = QStringLiteral("Not read on this system");
        gpu.note = QStringLiteral("The status bar says which device the learned model runs on once it has loaded.");
    } else if (f.gpus->empty()) {
        gpu.level = CheckLevel::Warn;
        gpu.value = QStringLiteral("None found");
        gpu.note = QStringLiteral("The learned model runs on the CPU, more slowly. Updating the graphics driver can make "
                                  "a card visible.");
    } else {
        QStringList names;
        for (const auto& g : *f.gpus)
            names << (g.vram_gb >= 1.0 ? QStringLiteral("%1 · %2 GB").arg(g.name).arg(std::lround(g.vram_gb)) : g.name);
        gpu.value = names.join(QStringLiteral("\n"));
        gpu.note = f.mac ? QStringLiteral("The learned model runs on it through Core ML, and the viewer draws on it.")
                         : QStringLiteral("The learned model runs on it, and the HDR viewer draws on it.");
    }
    out.push_back(gpu);

    CheckLine ai{QStringLiteral("aifill"), QStringLiteral("AI fill (optional)"), {}, {}, CheckLevel::Ok};
    const QString problem = aifill_machine_problem(f);
    if (problem.isEmpty()) {
        ai.value = f.gpus || f.mac ? QStringLiteral("This computer can run it") : QStringLiteral("Its setup checks the GPU");
        if (f.gpus)
            for (const auto& g : *f.gpus)
                if (g.name.contains(QStringLiteral("RTX 50"), Qt::CaseInsensitive))
                    ai.note = QStringLiteral("RTX 50 series: AI fill's setup installs the CUDA 12.8 build of PyTorch these "
                                             "cards need.");
    } else {
        ai.level = CheckLevel::Warn;
        ai.value = QStringLiteral("Not on this computer");
        ai.note = problem + QStringLiteral(" Everything else in RUDRA runs.");
    }
    out.push_back(ai);

    if (f.ram_gb > 0) {
        CheckLine mem{QStringLiteral("memory"), QStringLiteral("Memory"), QStringLiteral("%1 GB").arg(std::lround(f.ram_gb)), {},
                      CheckLevel::Ok};
        if (f.ram_gb < 7.5) {
            mem.level = CheckLevel::Warn;
            mem.note = QStringLiteral("8 GB or more is recommended; long 4K movies need more.");
        }
        out.push_back(mem);
    }

    if (f.free_gb >= 0) {
        CheckLine disk{QStringLiteral("disk"), QStringLiteral("Disk"),
                       QStringLiteral("%1 GB free where models are kept").arg(std::lround(std::floor(f.free_gb))), {},
                       CheckLevel::Ok};
        if (f.free_gb < 1.0) {
            disk.level = CheckLevel::Warn;
            disk.note = QStringLiteral("A model download will not fit. Free some space, or choose another folder in "
                                       "File › Model packages (%1).").arg(QDir::toNativeSeparators(f.models_root));
        } else if (f.free_gb < 30.0) {
            disk.level = CheckLevel::Info;
            disk.note = QStringLiteral("Enough for the learned model. AI fill's setup needs about 30 GB.");
        }
        out.push_back(disk);
    }

    CheckLine ff{QStringLiteral("ffmpeg"), QStringLiteral("Movies (ffmpeg)"), f.ffmpeg ? QStringLiteral("Found") : QStringLiteral("Not found"),
                 {}, f.ffmpeg ? CheckLevel::Ok : CheckLevel::Warn};
    if (!f.ffmpeg)
        ff.note = QStringLiteral("Stills open without it; movies need it. The package carries one in its ffmpeg folder: "
                                 "install RUDRA again, or put ffmpeg on the PATH.");
    out.push_back(ff);
    return out;
}

bool machine_blocked(const std::vector<CheckLine>& lines) {
    for (const auto& l : lines)
        if (l.level == CheckLevel::Fail) return true;
    return false;
}

QString genfill_missing_text(const QString& why) {
    QString part;
    if (why.contains(QStringLiteral("Wan2.1"))) part = QStringLiteral("its base model, Wan2.1-VACE-1.3B, is not installed");
    else if (why.contains(QStringLiteral("no Python"))) part = QStringLiteral("its Python is not installed");
    else if (why.contains(QStringLiteral("no LoRA"))) part = QStringLiteral("the AI fill model is not downloaded");
    else if (why.contains(QStringLiteral("repository"))) part = QStringLiteral("its scripts were not found in this installation");
    else return why + QStringLiteral(" Help › Check this computer… sets it up.");
    return QStringLiteral("AI fill is not set up yet: %1. Help › Check this computer… sets it up in one place.").arg(part);
}

namespace {
std::optional<bool> g_aifill_here_for_tests;
}  // namespace

void set_aifill_here_for_tests(std::optional<bool> runs) { g_aifill_here_for_tests = runs; }

AiFillHere aifill_here() {
    if (g_aifill_here_for_tests)
        return {*g_aifill_here_for_tests,
                *g_aifill_here_for_tests ? QString() : QStringLiteral("This computer cannot run the AI fill.")};
    const MachineFacts f = read_machine_facts(QString());
    // A GPU that could not be read is not taken for one that can: the AI fill starts off.
    if (!f.mac && !f.gpus) return {false, QStringLiteral("RUDRA could not read this computer's GPU.")};
    if (const QString problem = aifill_machine_problem(f); !problem.isEmpty()) return {false, problem};
    QString why;
    if (!genfill_setup(&why)) return {false, genfill_missing_text(why)};
    return {true, {}};
}

// ---- The window --------------------------------------------------------------------------------

SystemCheckDialog::SystemCheckDialog(MainWindow* w, bool first_run)
    : QDialog(w), w_(w), md_(w->model_downloads()), first_run_(first_run) {
    setObjectName(QStringLiteral("systemCheck"));
    setWindowTitle(QStringLiteral("Check this computer"));
    setModal(false);
    resize(780, 680);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(24, 20, 24, 16);
    v->setSpacing(10);
    v->addWidget(text(QStringLiteral("Check this computer"), QStringLiteral("check-heading"), this));
    summary_ = text({}, QStringLiteral("check-summary"), this, QStringLiteral("checkSummary"));
    v->addWidget(summary_);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* inner = new QWidget(scroll);
    auto* iv = new QVBoxLayout(inner);
    iv->setContentsMargins(0, 0, 8, 0);
    iv->setSpacing(6);
    iv->addWidget(text(QStringLiteral("THIS COMPUTER"), QStringLiteral("check-section"), inner));
    machine_box_ = new QVBoxLayout;
    machine_box_->setSpacing(4);
    iv->addLayout(machine_box_);
    iv->addSpacing(10);
    iv->addWidget(text(QStringLiteral("WHAT RUDRA CAN INSTALL"), QStringLiteral("check-section"), inner));
    iv->addWidget(text(QStringLiteral("Nothing is downloaded until you ask. Each model is checked file by file before RUDRA "
                                      "uses it."),
                       QStringLiteral("check-note"), inner));
    parts_box_ = new QVBoxLayout;
    parts_box_->setSpacing(6);
    iv->addLayout(parts_box_);
    iv->addStretch(1);
    scroll->setWidget(inner);
    v->addWidget(scroll, 1);

    progress_text_ = text({}, QStringLiteral("check-note"), this, QStringLiteral("checkProgressText"));
    progress_ = new QProgressBar(this);
    progress_->setObjectName(QStringLiteral("checkProgress"));
    progress_->setTextVisible(false);
    progress_->setFixedHeight(5);
    progress_->setRange(0, 1000);
    progress_text_->hide();
    progress_->hide();
    v->addWidget(progress_text_);
    v->addWidget(progress_);

    auto* row = new QHBoxLayout;
    row->setSpacing(8);
    auto* copy = new QPushButton(QStringLiteral("Copy report"), this);
    copy->setObjectName(QStringLiteral("checkCopy"));
    copy->setToolTip(QStringLiteral("What this window says, as text, for a support request"));
    auto* packages = new QPushButton(QStringLiteral("Model packages…"), this);
    packages->setObjectName(QStringLiteral("checkPackages"));
    install_ = new QPushButton(QStringLiteral("Install selected"), this);
    install_->setObjectName(QStringLiteral("checkInstall"));
    install_->setProperty("role", "primary");
    auto* close = new QPushButton(first_run ? QStringLiteral("Continue") : QStringLiteral("Close"), this);
    close->setObjectName(QStringLiteral("checkClose"));
    row->addWidget(copy);
    row->addWidget(packages);
    row->addStretch(1);
    row->addWidget(install_);
    row->addWidget(close);
    v->addLayout(row);

    connect(copy, &QPushButton::clicked, this, [this] { QGuiApplication::clipboard()->setText(report()); });
    connect(packages, &QPushButton::clicked, this, [this] { w_->open_model_manager(); });
    connect(install_, &QPushButton::clicked, this, [this] { install_selected(); });
    connect(close, &QPushButton::clicked, this, [this] { accept(); });

    md_->listen(this, [this] { on_downloads_changed(); });
    // The list of models, read while this window is open (as File > Model packages does).
    if (md_->models().empty() && !md_->fetching()) md_->fetch_index();
    refresh();
}

void SystemCheckDialog::set_facts(const MachineFacts& f) {
    facts_override_ = f;
    refresh();
}

void SystemCheckDialog::refresh() {
    facts_ = facts_override_ ? *facts_override_ : read_machine_facts(md_->root());
    machine_ = judge_machine(facts_);
    // The model list: whether www.fxtdstudios.com answered (the downloads need it).
    CheckLine net{QStringLiteral("network"), QStringLiteral("Model list"), {}, {}, CheckLevel::Ok};
    if (md_->fetching()) {
        net.level = CheckLevel::Info;
        net.value = QStringLiteral("Reading www.fxtdstudios.com…");
    } else if (!md_->index_status().isEmpty()) {
        net.level = CheckLevel::Warn;
        net.value = QStringLiteral("Could not be read");
        net.note = md_->index_status() + QStringLiteral(". Without the internet, models copied from another computer are "
                                                        "added in File › Model packages › Add folder….");
    } else if (!md_->models().empty()) {
        net.value = QStringLiteral("Read from www.fxtdstudios.com");
    } else {
        net.level = CheckLevel::Info;
        net.value = QStringLiteral("Not read yet");
    }
    machine_.push_back(net);

    clear_layout(machine_box_);
    for (const CheckLine& l : machine_) {
        auto* line = new QWidget(this);
        line->setObjectName(QStringLiteral("checkLine_") + l.id);
        line->setProperty("level", level_key(l.level));
        auto* h = new QHBoxLayout(line);
        h->setContentsMargins(0, 2, 0, 2);
        h->setSpacing(10);
        auto* dot = text({}, QStringLiteral("check-dot"), line, QStringLiteral("checkDot_") + l.id);
        dot->setFixedSize(10, 10);
        dot->setProperty("state", level_key(l.level));
        dot->setToolTip(level_word(l.level));
        h->addWidget(dot, 0, Qt::AlignTop);
        auto* title = text(l.title, QStringLiteral("check-title"), line);
        title->setFixedWidth(150);
        h->addWidget(title, 0, Qt::AlignTop);
        auto* col = new QVBoxLayout;
        col->setSpacing(1);
        col->addWidget(text(l.value, QStringLiteral("check-value"), line, QStringLiteral("checkValue_") + l.id));
        if (!l.note.isEmpty()) col->addWidget(text(l.note, QStringLiteral("check-note"), line, QStringLiteral("checkNote_") + l.id));
        h->addLayout(col, 1);
        machine_box_->addWidget(line);
    }
    parts_shape_.clear();   // every row again
    build_parts();
    update_progress();
}

void SystemCheckDialog::build_parts() {
    // The rows' shape: rebuilt when a state changes, not on every progress tick.
    QString shape = md_->active() + QStringLiteral("|") + QString::number(md_->models().size()) +
                    (md_->fetching() ? QStringLiteral("|fetching") : QString());
    for (const auto& m : md_->models())
        shape += QStringLiteral("|") + m.id + QStringLiteral(":") + QString::number(int(md_->state(m.id))) + md_->error(m.id);
    for (const auto& p : w_->retired_packages()) shape += QStringLiteral("|retired:") + from_path(p);
    if (shape == parts_shape_) return;
    parts_shape_ = shape;

    std::map<QString, bool> was;   // the user's ticks survive a rebuild
    // Only boxes the user could tick: one that was off because this computer could not run it is no
    // choice, and must not stay off once it can (release CI on Windows, 19 Nov 2026).
    for (const auto& [id, box] : ticks_)
        if (box->isEnabled()) was[id] = box->isChecked();
    ticks_.clear();
    clear_layout(parts_box_);
    const QString ai_problem = aifill_machine_problem(facts_);
    bool missing_recommended = false;

    auto card = [this](const QString& id) {
        auto* c = new QWidget(this);
        c->setObjectName(QStringLiteral("checkPart_") + id);
        c->setProperty("role", "card");
        c->setAttribute(Qt::WA_StyledBackground, true);
        auto* h = new QHBoxLayout(c);
        h->setContentsMargins(12, 10, 12, 10);
        h->setSpacing(12);
        parts_box_->addWidget(c);
        return std::pair{c, h};
    };
    auto words = [this](QWidget* c, QHBoxLayout* h, const QString& title, const QString& detail, const QString& state,
                        const QString& state_key, const QString& id) {
        auto* col = new QVBoxLayout;
        col->setSpacing(2);
        col->addWidget(text(title, QStringLiteral("check-title"), c));
        if (!detail.isEmpty()) col->addWidget(text(detail, QStringLiteral("check-note"), c));
        h->addLayout(col, 1);
        auto* st = text(state, QStringLiteral("check-state"), c, QStringLiteral("checkState_") + id);
        st->setProperty("state", state_key);
        st->setMaximumWidth(200);
        h->addWidget(st, 0, Qt::AlignVCenter);
    };

    // 1. The downloads. Without the list only the built-in Florence-2 is there, so the reason still shows.
    const bool only_builtins = std::all_of(md_->models().begin(), md_->models().end(), [](const RemoteModel& m) {
        return m.kind == QStringLiteral("florence");
    });
    if (only_builtins) {
        const QString status = md_->fetching() ? QStringLiteral("Reading the list of models…")
                               : md_->index_status().isEmpty() ? QStringLiteral("The list of models has not been read yet.")
                                                               : QStringLiteral("The list of models could not be read (%1). Models copied "
                                                                                "from another computer are added in File › Model "
                                                                                "packages › Add folder….").arg(md_->index_status());
        parts_box_->addWidget(text(status, QStringLiteral("check-note"), this, QStringLiteral("checkPartsStatus")));
    }
    // 1.2.6: Florence-2 counts as there when the AI fill already finds one (beside the Wan folder, say).
    std::filesystem::path florence_found;
    {
        QString w;
        const auto gs = genfill_setup(&w);
        GenFillSetup probe = gs ? *gs : GenFillSetup{};
        if (!gs) probe.base = to_path(QSettings().value("genfill/base").toString());
        florence_found = genfill_florence_dir(probe);
    }
    for (const RemoteModel& m : md_->models()) {
        if (!listed_here(m)) continue;
        const InstallState st = md_->state(m.id);
        const bool aces = m.kind == QStringLiteral("aces_v2");
        const bool florence = m.kind == QStringLiteral("florence");
        const bool installed = st == InstallState::Installed || (aces && w_->have_aces_package()) ||
                               (florence && !florence_found.empty());
        const bool active = md_->active() == m.id;
        const bool ai = m.kind == QStringLiteral("genfill") || florence;
        const bool blocked_here = ai && !ai_problem.isEmpty();
        if (aces && !installed) missing_recommended = true;
        auto [c, h] = card(m.id);
        auto* tick = new QCheckBox(c);
        tick->setObjectName(QStringLiteral("checkTick_") + m.id);
        tick->setEnabled(!installed && !active && !blocked_here);
        // Ticked the first time: the learned model, and Florence-2 (0.5 GB) where AI fill can run.
        tick->setChecked(installed || active ? false : was.count(m.id) ? was[m.id] : (aces || florence) && !blocked_here);
        if (blocked_here) tick->setToolTip(ai_problem);
        connect(tick, &QCheckBox::toggled, this, [this] { update_progress(); });
        h->addWidget(tick, 0, Qt::AlignTop);
        ticks_[m.id] = tick;
        QString detail = (aces ? QStringLiteral("Recommended. ") : QStringLiteral("Optional. ")) + m.summary;
        detail += QStringLiteral(" · ") + human_size(m.size());
        if (!m.licence.isEmpty()) detail += QStringLiteral(" · Licence: ") + m.licence;
        if (ai && !florence) detail += QStringLiteral(". Needs the base model and Python below.");
        if (florence) detail += QStringLiteral(". It runs in AI fill's Python below.");   // what it does is in its own note
        if (blocked_here) detail += QStringLiteral("\n") + ai_problem;
        const QString err = md_->error(m.id);
        const QString state = installed ? QStringLiteral("Installed and checked")
                              : active ? QStringLiteral("Downloading…")
                              : !err.isEmpty() ? err
                              : st == InstallState::Partial ? QStringLiteral("Part downloaded: Install resumes it")
                                                            : QStringLiteral("Not installed");
        words(c, h, m.title + (m.version.isEmpty() ? QString() : QStringLiteral(" ") + m.version), detail, state,
              installed ? QStringLiteral("ok") : !err.isEmpty() ? QStringLiteral("warn") : QString(), m.id);
    }

    // 2. AI fill's base model and its Python: the setup that already exists (Reconstruct > AI generation > Setup…).
    {
        QString why;
        const auto s = genfill_setup(&why);
        const bool base_ok = s.has_value() || why.contains(QStringLiteral("no LoRA"));
        auto [c, h] = card(QStringLiteral("aifillSetup"));
        h->addSpacing(22);
        QString detail = QStringLiteral("Optional. The base model AI fill draws with (Wan2.1-VACE-1.3B, from Wan-AI, Apache 2.0, "
                                        "about 25 GB) and a Python with PyTorch for RUDRA only (about 5 GB). Set up installs "
                                        "both, and resumes where it stopped.");
        if (!ai_problem.isEmpty()) detail += QStringLiteral("\n") + ai_problem;
        words(c, h, QStringLiteral("AI fill: base model and Python"), detail,
              base_ok ? QStringLiteral("Installed") : QStringLiteral("Not set up"), base_ok ? QStringLiteral("ok") : QString(),
              QStringLiteral("aifillSetup"));
        auto* set_up = new QPushButton(base_ok ? QStringLiteral("Check…") : QStringLiteral("Set up…"), c);
        set_up->setObjectName(QStringLiteral("checkAifillSetup"));
        set_up->setEnabled(ai_problem.isEmpty());
        set_up->setToolTip(base_ok ? why : genfill_missing_text(why));
        connect(set_up, &QPushButton::clicked, this, [this] {
            GenFillSetupDialog(this).exec();
            parts_shape_.clear();
            build_parts();
            w_->refresh_check_attention();
        });
        h->addWidget(set_up, 0, Qt::AlignVCenter);
    }

    // 3. What RUDRA put right by itself.
    for (const auto& p : w_->retired_packages()) {
        const QString name = from_path(p.filename());
        auto [c, h] = card(QStringLiteral("retired_") + name);
        h->addSpacing(22);
        words(c, h, QStringLiteral("Old Beta 2 model: ") + name,
              QStringLiteral("Not opened any more. The Beta 2 models are non-commercial and were retired in 1.2.0; the "
                             "learned ACES 2.0 model replaces them. Move aside puts it in a .retired-beta2 folder beside it.\n") +
                  QDir::toNativeSeparators(from_path(p)),
              QStringLiteral("Not used"), QStringLiteral("warn"), QStringLiteral("retired_") + name);
        auto* aside = new QPushButton(QStringLiteral("Move aside"), c);
        aside->setObjectName(QStringLiteral("checkRetire_") + name);
        connect(aside, &QPushButton::clicked, this, [this, p] {
            QString said;
            archive_beta2(p, &said);
            results_ << said;
            w_->log(said);
            w_->rescan_models();
            parts_shape_.clear();
            build_parts();
            update_progress();
            w_->refresh_check_attention();
        });
        h->addWidget(aside, 0, Qt::AlignVCenter);
    }
    if (w_->use_model_released()) {
        auto [c, h] = card(QStringLiteral("useModel"));
        h->addSpacing(22);
        words(c, h, QStringLiteral("Reconstruct › Use the model"),
              QStringLiteral("It was on, with no reconstruction model installed, which held every shot back. RUDRA "
                             "turned it off; nothing to do."),
              QStringLiteral("Fixed"), QStringLiteral("ok"), QStringLiteral("useModel"));
    }

    // The line at the top.
    const bool blocked = machine_blocked(machine_);
    QString first_fail;
    for (const auto& l : machine_)
        if (l.level == CheckLevel::Fail) {
            first_fail = l.note;
            break;
        }
    summary_->setProperty("state", blocked ? QStringLiteral("fail") : missing_recommended ? QStringLiteral("warn") : QStringLiteral("ok"));
    summary_->setText(blocked ? QStringLiteral("This computer cannot run RUDRA: ") + first_fail
                      : missing_recommended
                          ? QStringLiteral("RUDRA runs on this computer. Install the learned ACES 2.0 model below for more of "
                                           "the clipped light: it is ticked, so Install selected is all it takes.")
                          : QStringLiteral("RUDRA is ready on this computer."));
    summary_->style()->unpolish(summary_);
    summary_->style()->polish(summary_);
}

void SystemCheckDialog::update_progress() {
    int ticked = 0;
    for (const auto& [id, box] : ticks_)
        if (box->isChecked() && box->isEnabled()) ++ticked;
    const bool running = !current_.isEmpty() || !queue_.isEmpty();
    install_->setEnabled(ticked > 0 && !running);
    install_->setText(ticked > 0 ? QStringLiteral("Install selected (%1)").arg(ticked) : QStringLiteral("Install selected"));
    const QString active = md_->active();
    if (!active.isEmpty()) {
        const RemoteModel* m = md_->model(active);
        const qint64 total = md_->total_bytes(), done = md_->done_bytes();
        progress_->setVisible(true);
        progress_text_->setVisible(true);
        progress_->setValue(total > 0 ? int(1000.0 * double(done) / double(total)) : 0);
        progress_text_->setText(QStringLiteral("Downloading %1: %2 of %3%4")
                                    .arg(m ? m->title : active, human_size(done), human_size(total),
                                         queue_.isEmpty() ? QString() : QStringLiteral(" · %1 more after it").arg(queue_.size())));
    } else {
        progress_->setVisible(false);
        progress_text_->setVisible(!results_.isEmpty());
        progress_text_->setText(results_.join(QStringLiteral("\n")));
    }
}

void SystemCheckDialog::install_selected() {
    queue_.clear();
    for (const auto& m : md_->models())   // the list's order: the learned model first
        if (auto it = ticks_.find(m.id); it != ticks_.end() && it->second->isChecked() && it->second->isEnabled())
            queue_ << m.id;
    results_.clear();
    download_next();
}

void SystemCheckDialog::download_next() {
    if (!md_->active().isEmpty()) {   // one at a time: the listener comes back here when it ends
        update_progress();
        return;
    }
    while (!queue_.isEmpty()) {
        const QString id = queue_.takeFirst();
        if (md_->state(id) == InstallState::Installed) continue;
        QString why;
        if (md_->download(id, &why)) {
            current_ = id;
            break;
        }
        const RemoteModel* m = md_->model(id);
        results_ << (m ? m->title : id) + QStringLiteral(": ") + why;
    }
    parts_shape_.clear();
    build_parts();
    update_progress();
}

void SystemCheckDialog::on_downloads_changed() {
    if (!current_.isEmpty() && md_->active() != current_) {   // the one this window started has ended
        const RemoteModel* m = md_->model(current_);
        const QString title = m ? m->title : current_;
        results_ << (md_->state(current_) == InstallState::Installed ? title + QStringLiteral(": installed and checked")
                                                                      : title + QStringLiteral(": ") +
                                                                            (md_->error(current_).isEmpty() ? QStringLiteral("stopped")
                                                                                                            : md_->error(current_)));
        current_.clear();
        download_next();
        w_->refresh_check_attention();
        return;
    }
    // The model list arrived (or failed): the network line and the rows.
    if (machine_.empty() || machine_.back().id != QStringLiteral("network") ||
        (md_->fetching() != (machine_.back().value == QStringLiteral("Reading www.fxtdstudios.com…")))) {
        refresh();
        return;
    }
    build_parts();
    update_progress();
}

QString SystemCheckDialog::report() const {
    QStringList out;
    out << QStringLiteral("RUDRA %1 · Check this computer").arg(QCoreApplication::applicationVersion());
    out << summary_->text() << QString();
    for (const auto& l : machine_) {
        out << QStringLiteral("[%1] %2: %3").arg(level_word(l.level), l.title, QString(l.value).replace('\n', QStringLiteral("; ")));
        if (!l.note.isEmpty()) out << QStringLiteral("    ") + l.note;
    }
    out << QString();
    for (const auto& m : md_->models()) {
        if (!listed_here(m)) continue;
        const bool installed = md_->state(m.id) == InstallState::Installed ||
                               (m.kind == QStringLiteral("aces_v2") && w_->have_aces_package());
        out << QStringLiteral("%1 %2: %3").arg(m.title, m.version, installed ? QStringLiteral("installed") : QStringLiteral("not installed"));
    }
    QString why;
    out << QStringLiteral("AI fill setup: %1").arg(genfill_setup(&why) ? QStringLiteral("ready") : why);
    for (const auto& p : w_->retired_packages()) out << QStringLiteral("Retired Beta 2 model found: ") + QDir::toNativeSeparators(from_path(p));
    if (!results_.isEmpty()) out << QString() << results_;
    return out.join(QStringLiteral("\n"));
}

// ---- MainWindow's side ---------------------------------------------------------------------------

void MainWindow::open_system_check(bool first_run) {
    if (!system_check_) system_check_ = new SystemCheckDialog(this, first_run);
    else static_cast<SystemCheckDialog*>(system_check_.data())->refresh();
    system_check_->show();
    system_check_->raise();
    system_check_->activateWindow();
}

void MainWindow::refresh_check_attention() {
    if (!machine_blocked_) machine_blocked_ = machine_blocked(judge_machine(read_machine_facts(model_downloads()->root())));
    const bool seen = QSettings().value(QStringLiteral("check/seenVersion")).toString() == QCoreApplication::applicationVersion();
    const bool want = *machine_blocked_ || !retired_.empty() || (!seen && !have_aces_package());
    // IconButton has no Q_OBJECT, which Qt 6.8's findChild<T> requires: found as a QWidget, then cast.
    if (auto* b = dynamic_cast<IconButton*>(findChild<QWidget*>(QStringLiteral("iCheck")))) {
        b->set_attention(want);
        b->setToolTip(want ? QStringLiteral("Check this computer: something wants a look") : QStringLiteral("Check this computer"));
    }
}

void MainWindow::maybe_open_system_check() {
    refresh_check_attention();
    const bool updated = QSettings().value(QStringLiteral("check/seenVersion")).toString() != QCoreApplication::applicationVersion();
    if (updated || machine_blocked_.value_or(false) || !retired_.empty()) open_system_check(false);
}

void SystemCheckDialog::done(int r) {
    QSettings().setValue(QStringLiteral("check/seenVersion"), QCoreApplication::applicationVersion());
    QDialog::done(r);
    w_->refresh_check_attention();
    if (first_run_) {
        first_run_ = false;
        w_->show_getting_started_once();   // the tips, after the check (as the welcome did before 1.2.5)
    }
}

}  // namespace rudra::app
