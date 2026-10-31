#include "aces_master.hpp"
#include "scope_widgets.hpp"
#include "paths.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QJsonArray>
#include <QJsonObject>
#include <QPixmap>
#include <QSettings>
#include <QSlider>
#include <QTemporaryDir>
#include <QUrl>
#include <QVariant>
#include <QVBoxLayout>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "aces_engine.hpp"
#include "main_window.hpp"
#include "widgets.hpp"
#include <QStatusBar>
#include "rudra/core/aces_v2_manifest.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/color.hpp"
#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif
#include "rudra/media/aces_frames.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QString q(const std::string& s) { return QString::fromStdString(s); }
QString qp(const fs::path& p) { return from_path(p); }
fs::path pq(const QString& s) { return to_path(s); }

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

QLabel* label(const QString& text, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    l->setWordWrap(true);
    return l;
}

QPushButton* button(const QString& text, const QString& id, QWidget* parent) {
    auto* b = new QPushButton(text, parent);
    b->setObjectName(id);
    return b;
}

// A new folder beside the frames: "<name>_aces_master", then _2, _3 ...
fs::path suggest_out(const fs::path& input) {
    if (input.empty()) return {};
    std::error_code ec;
    const fs::path base = fs::is_directory(input, ec) ? input : input.parent_path();
    const fs::path parent = base.parent_path().empty() ? base : base.parent_path();
    const std::string name = (fs::is_directory(input, ec) ? base.filename().string() : input.stem().string()) + "_aces_master";
    fs::path out = parent / name;
    for (int i = 2; fs::exists(out, ec) && i < 1000; ++i) out = parent / (name + "_" + std::to_string(i));
    return out;
}

}  // namespace

// ---- helpers ---------------------------------------------------------------

std::vector<AcesPackageInfo> find_aces_packages(const std::vector<fs::path>& roots) {
    std::vector<AcesPackageInfo> out;
    std::set<std::string> seen;
    auto consider = [&](const fs::path& dir) {
        std::error_code ec;
        if (!fs::is_regular_file(dir / "manifest.json", ec)) return;
        const auto key = fs::weakly_canonical(dir, ec).string();
        if (!seen.insert(key.empty() ? dir.string() : key).second) return;
        auto m = aces_v2::read_manifest(dir);
        if (!m) return;   // a v1 package, another contract, or a failed export: not offered
        out.push_back({dir, m->g1_sha256, m->checkpoint_sha256, !m->g1_onnx.empty(), m->forbid_new_negative_y,
                       m->release_authorized});
    };
    for (const auto& r : roots) {
        std::error_code ec;
        if (r.empty() || !fs::is_directory(r, ec)) continue;
        consider(r);
        std::vector<fs::path> subs;
        for (const auto& e : fs::directory_iterator(r, ec))
            if (e.is_directory(ec)) subs.push_back(e.path());
        std::sort(subs.begin(), subs.end());
        for (const auto& s : subs) consider(s);
    }
    return out;
}

bool is_aces_still(const fs::path& p) { return rudra::aces_is_still(p); }
bool is_aces_video(const fs::path& p) { return rudra::aces_is_video(p); }

std::vector<fs::path> collect_aces_stills(const fs::path& input) {
    std::vector<fs::path> frames;
    std::error_code ec;
    if (fs::is_directory(input, ec)) {
        for (const auto& e : fs::directory_iterator(input, ec))
            if (e.is_regular_file(ec) && is_aces_still(e.path())) frames.push_back(e.path());
        std::sort(frames.begin(), frames.end());
    } else if (fs::is_regular_file(input, ec) && (is_aces_still(input) || is_aces_video(input))) {
        frames.push_back(input);
    }
    return frames;
}

std::optional<fs::path> aces_name_collision(const std::vector<fs::path>& stills) {
    std::set<std::string> names;
    for (const auto& f : stills)
        if (!names.insert(lower(f.stem().string() + ".exr")).second) return f;
    return std::nullopt;
}

const std::vector<AcesRunTarget>& aces_run_targets() {
    static const std::vector<AcesRunTarget> t = {
        {"cuda", "GPU: CUDA (LibTorch, matches the reference exactly)", "libtorch", "cuda"},
        {"directml", "GPU: DirectML (ONNX Runtime, within 0.25 code)", "onnxruntime", "directml"},
        {"cpu", "CPU (LibTorch, matches the reference exactly)", "libtorch", "cpu"},
        {"cpu-onnx", "CPU (ONNX Runtime)", "onnxruntime", "cpu"},
    };
    return t;
}

const std::vector<AcesLogChoice>& aces_log_choices() {
    static const std::vector<AcesLogChoice> c = {
        {"", "Nothing else (ACEScg EXR only)"},
        {"tiff16", "ARRI LogC4: 16-bit TIFF per frame (lossless)"},
        {"dpx10", "ARRI LogC4: 10-bit DPX per frame"},
        {"prores4444xq", "ARRI LogC4: ProRes 4444 XQ movie (12-bit 4:4:4)"},
        {"prores4444", "ARRI LogC4: ProRes 4444 movie (12-bit 4:4:4)"},
        {"prores422hq", "ARRI LogC4: ProRes 422 HQ movie (10-bit 4:2:2)"},
        {"mov16", "ARRI LogC4: uncompressed 16-bit RGB movie (lossless)"},
        {"tiff16,prores4444xq", "ARRI LogC4: 16-bit TIFF + ProRes 4444 XQ"},
        {"dpx10,prores422hq", "ARRI LogC4: 10-bit DPX + ProRes 422 HQ"},
    };
    return c;
}

QStringList aces_cli_arguments(const AcesRunSettings& s) {
    QStringList a = {"aces-v2", qp(s.package), qp(s.input), "--out", qp(s.out), "--runtime", q(s.runtime),
                     "--device", q(s.device), "--ocio-threads", QString::number(s.ocio_threads)};
    a << (s.luminance_guard ? "--luminance-guard" : "--no-luminance-guard");
    if (!s.log_curve.empty() && !s.log_formats.empty()) a << "--log" << q(s.log_curve) << "--log-format" << q(s.log_formats);
    return a;
}

std::optional<AcesProgressLine> parse_aces_progress(const std::string& line) {
    static const std::regex re(R"(^(.+) -> (.+?)\s+([0-9]+(?:\.[0-9]+)?) ms\s+remaining_display_errors=([0-9]+)\s*$)");
    std::smatch m;
    if (!std::regex_match(line, m, re)) return std::nullopt;
    AcesProgressLine p;
    p.source = m[1];
    p.exr = m[2];
    p.milliseconds = std::stod(m[3]);
    p.remaining_display_errors = static_cast<std::size_t>(std::stoull(m[4]));
    return p;
}

Result<AcesRunSummary> read_aces_provenance(const fs::path& out_dir) {
    // A folder's provenance.json, or the file itself (<name>_provenance.json in a shared folder).
    const fs::path file = out_dir.extension() == ".json" ? out_dir : out_dir / "provenance.json";
    std::ifstream in(file);
    if (!in) return make_error(ErrorCode::NotFound, "The run wrote no provenance.json.", file.string());
    nlohmann::json j;
    try {
        in >> j;
        AcesRunSummary s;
        s.device = j.value("device", std::string());
        s.runtime = j.value("runtime", std::string());
        s.appearance_pass = j.value("appearance_constraint_pass", false);
        s.luminance_guard = j.value("luminance_guard", false);
        double ms = 0, rec = 0;
        for (const auto& f : j.at("frames")) {
            ++s.frames;
            ms += f.value("ms", 0.0);
            rec += f.value("reconstructed_fraction", 0.0);
            s.remaining_display_errors += f.value("remaining_display_errors", std::size_t(0));
            s.new_negative_luminance += f.value("new_negative_luminance", std::size_t(0));
        }
        if (j.contains("log_delivery") && j["log_delivery"].is_object()) {
            s.log_curve = j["log_delivery"].value("curve", std::string());
            for (const auto& f : j["log_delivery"].value("files", nlohmann::json::array()))
                s.log_files.push_back(f.value("kind", std::string()));
        }
        if (s.frames) {
            s.mean_ms = ms / double(s.frames);
            s.mean_reconstructed_fraction = rec / double(s.frames);
        }
        return s;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "provenance.json could not be read.", e.what());
    }
}

std::string aces_summary_text(const AcesRunSummary& s) {
    std::ostringstream t;
    t.setf(std::ios::fixed);
    t.precision(0);
    t << s.frames << (s.frames == 1 ? " master" : " masters") << " written on " << s.device << " (" << s.runtime << "), "
      << s.mean_ms << " ms a frame. ";
    t.precision(1);
    t << 100.0 * s.mean_reconstructed_fraction << " % of each frame reconstructed on average. ";
    if (s.appearance_pass && s.remaining_display_errors == 0)
        t << "Every frame shows within 1 code of the source through the ACES 2.0 SDR view";
    else
        t << s.remaining_display_errors << " pixels show more than 1 code from the source: check those frames";
    t << (s.luminance_guard ? "; no new negative luminance (" : "; luminance guard off (") << s.new_negative_luminance
      << " new negative pixels).";
    if (!s.log_curve.empty()) {
        t << " " << s.log_curve << " copies: ";
        for (std::size_t i = 0; i < s.log_files.size(); ++i) t << (i ? ", " : "") << s.log_files[i];
        t << ".";
    }
    return t.str();
}

QProcessEnvironment aces_child_environment() {
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    const QStringList extra = QSettings().value("aces/path_prepend").toStringList();
    if (!extra.isEmpty()) {
#ifdef _WIN32
        const QChar sep = ';';
#else
        const QChar sep = ':';
#endif
        const QString key = env.contains("Path") && !env.contains("PATH") ? QStringLiteral("Path") : QStringLiteral("PATH");
        env.insert(key, extra.join(sep) + sep + env.value(key));
    }
    return env;
}

fs::path default_aces_cli() {
#ifdef _WIN32
    const char* exe = "rudra-native.exe";
#else
    const char* exe = "rudra-native";
#endif
    const fs::path p = pq(QCoreApplication::applicationDirPath()) / exe;
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p : fs::path();
}

AcesCliProbe probe_aces_cli(const fs::path& cli, int timeout_ms) {
    std::error_code ec;
    if (cli.empty()) return {false, "No engine chosen."};
    if (!fs::is_regular_file(cli, ec)) return {false, "That file does not exist."};
    QProcess p;
    p.setProcessEnvironment(aces_child_environment());
    p.setProcessChannelMode(QProcess::MergedChannels);
    p.start(qp(cli), {"aces-v2", "--help"});   // "aces-v2" alone gets the general usage (main.cpp)
    if (!p.waitForStarted(timeout_ms)) return {false, "It did not start: " + p.errorString()};
    if (!p.waitForFinished(timeout_ms)) {
        p.kill();
        p.waitForFinished(1000);
        return {false, QStringLiteral("It did not answer within %1 s.").arg(timeout_ms / 1000)};
    }
    const QString out = QString::fromUtf8(p.readAll()).trimmed();
    if (out.contains("usage: rudra-native aces-v2 <package>")) return {true, {}};
    const int code = p.exitCode();
    // 0xC0000135 / 0xC000007B: Windows could not load one of its DLLs.
    if (p.exitStatus() == QProcess::CrashExit || code == -1073741515 || code == -1073741701 ||
        unsigned(code) == 0xC0000135u || unsigned(code) == 0xC000007Bu)
        return {false, QStringLiteral("It stopped before printing anything (exit 0x%1): a DLL it needs was not found. "
                                      "LibTorch's lib folder must be on PATH (setting aces/path_prepend).")
                           .arg(unsigned(code), 8, 16, QLatin1Char('0'))};
    return {false, QStringLiteral("It has no ACES v2 master (exit %1): %2").arg(code).arg(out.left(200))};
}

bool cli_has_aces_v2(const fs::path& cli, int timeout_ms) { return probe_aces_cli(cli, timeout_ms).ok; }

// ---- the dialog ------------------------------------------------------------

AcesMasterDialog::AcesMasterDialog(QWidget* parent, std::vector<fs::path> roots, fs::path input, AcesEngine* engine)
    : QDialog(parent), roots_(std::move(roots)), engine_(engine) {
    setObjectName("acesMaster");
    setWindowTitle("Learned ACES 2.0 master");
    resize(1360, 760);
    QSettings st;
    if (!engine_) {
        own_engine_ = std::make_unique<AcesEngine>(this);
        engine_ = own_engine_.get();
    }
    tmp_ = std::make_unique<QTemporaryDir>();
    auto* top = new QHBoxLayout(this);
    auto* v = new QVBoxLayout;
    top->addLayout(v, 5);
    auto* pv = new QVBoxLayout;
    top->addLayout(pv, 6);

    // ---- the preview through the model (right) ----
    pv->addWidget(label("Preview through the model", "acesPreviewHead", "plabel", this));
    auto* er = new QHBoxLayout;
    engine_note_ = label("", "acesEngineNote", "note", this);
    er->addWidget(engine_note_, 1);
    reload_ = button("Load model", "acesReload", this);
    er->addWidget(reload_);
    pv->addLayout(er);
    auto* pr = new QHBoxLayout;
    preview_frame_ = new QComboBox(this);
    preview_frame_->setObjectName("acesPreviewFrame");
    pr->addWidget(preview_frame_, 2);
    mode_ = new QComboBox(this);
    mode_->setObjectName("acesPreviewMode");
    mode_->addItem("ACES model", "model");
    mode_->addItem("Split: plain inverse | model", "split");
    mode_->addItem("Plain ACES inverse", "plain");
    mode_->addItem("Reconstructed areas", "mask");
    mode_->addItem("AI fill regions (Deliver thresholds)", "fill");
    mode_->addItem("SDR source", "source");
    pr->addWidget(mode_, 1);
    pv->addLayout(pr);
    auto* evr = new QHBoxLayout;
    evr->addWidget(label("Exposure", "acesEvLabelHead", "value", this));
    ev_ = new QSlider(Qt::Horizontal, this);
    ev_->setObjectName("acesPreviewEv");
    ev_->setRange(-16, 6);   // half stops: -8 .. +3 EV
    ev_->setValue(-4);       // -2 EV: the highlights the master keeps above display white
    evr->addWidget(ev_, 1);
    ev_label_ = label("", "acesEvValue", "value", this);
    ev_label_->setMinimumWidth(64);
    evr->addWidget(ev_label_);
    pv->addLayout(evr);
    image_ = new QLabel(this);
    image_->setObjectName("acesPreviewImage");
    image_->setAlignment(Qt::AlignCenter);
    image_->setMinimumSize(560, 340);
    image_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    image_->setText("No preview yet.");
    pv->addWidget(image_, 1);
    preview_stats_ = label("", "acesPreviewStats", "note", this);
    pv->addWidget(preview_stats_);
    ev_label_->setText(QStringLiteral("%1 EV").arg(ev_->value() / 2.0, 0, 'f', 1));

    v->addWidget(label("Learned ACES 2.0 master", "acesHead", "plabel", this));
    v->addWidget(label("One 32-bit float ACEScg EXR per still or video frame, with the rudra.reconstructed channel and a "
                       "provenance.json. Shadows stay the exact ACES 2.0 inverse; clipped highlights are rebuilt by the "
                       "model and are estimates, not recovered camera data. Every frame still shows within 1 code of "
                       "the source through the ACES 2.0 SDR view.",
                       "acesNote", "note", this));

    auto* g = new QGridLayout;
    int row = 0;
    g->addWidget(label("Model package", "acesPackageLabel", "value", this), row, 0);
    package_ = new QComboBox(this);
    package_->setObjectName("acesPackage");
    g->addWidget(package_, row, 1);
    auto* add_pkg = button("Add…", "acesPackageAdd", this);
    g->addWidget(add_pkg, row++, 2);
    package_note_ = label("", "acesPackageNote", "note", this);
    g->addWidget(package_note_, row++, 1, 1, 2);

    g->addWidget(label("Frames", "acesInputLabel", "value", this), row, 0);
    input_ = new QLineEdit(this);
    input_->setObjectName("acesInput");
    input_->setPlaceholderText("A still, a folder of stills (PNG, TIFF, JPEG, BMP, WebP) or a video (MP4, MOV, MKV, MXF…)");
    g->addWidget(input_, row, 1);
    auto* pick = new QHBoxLayout;
    auto* pick_file = button("Still or video…", "acesInputFile", this);
    auto* pick_dir = button("Folder…", "acesInputFolder", this);
    pick->addWidget(pick_file);
    pick->addWidget(pick_dir);
    g->addLayout(pick, row++, 2);
    frames_note_ = label("", "acesFramesNote", "note", this);
    g->addWidget(frames_note_, row++, 1, 1, 2);

    g->addWidget(label("Output folder", "acesOutLabel", "value", this), row, 0);
    out_ = new QLineEdit(this);
    out_->setObjectName("acesOut");
    out_->setPlaceholderText("A new folder (it must not exist yet)");
    g->addWidget(out_, row, 1);
    auto* pick_out = button("Choose…", "acesOutPick", this);
    g->addWidget(pick_out, row++, 2);

    g->addWidget(label("Run on", "acesTargetLabel", "value", this), row, 0);
    target_ = new QComboBox(this);
    target_->setObjectName("acesTarget");
    for (const auto& t : aces_run_targets()) target_->addItem(q(t.label), q(t.key));
    if (const int i = target_->findData(st.value("aces/target", "cuda").toString()); i >= 0) target_->setCurrentIndex(i);
    g->addWidget(target_, row++, 1, 1, 2);

    guard_ = new QCheckBox("No new negative luminance (recommended)", this);
    guard_->setObjectName("acesGuard");
    guard_->setToolTip("Highlight pixels that would come out with negative luminance where the plain ACES inverse "
                       "has none are moved back toward the plain inverse.");
    guard_->setChecked(true);
    g->addWidget(guard_, row++, 1, 1, 2);

    g->addWidget(label("Also write", "acesLogLabel", "value", this), row, 0);
    log_format_ = new QComboBox(this);
    log_format_->setObjectName("acesLog");
    for (const auto& c : aces_log_choices()) log_format_->addItem(q(c.label), q(c.key));
    log_format_->setToolTip("A camera-log copy of the same master in ARRI LogC4 / ARRI Wide Gamut 4 (no peak, knee or exposure "
                     "change; above 469.8 x diffuse white is clamped). Set the clip to LogC4 / AWG4 in Resolve or Nuke. "
                     "The rudra.reconstructed mask stays in the EXR. ProRes needs every frame the same size.");
    if (const int i = log_format_->findData(st.value("aces/log_formats", "").toString()); i >= 0) log_format_->setCurrentIndex(i);
    connect(log_format_, &QComboBox::currentIndexChanged, this, [this] { QSettings().setValue("aces/log_formats", log_format_->currentData()); });
    g->addWidget(log_format_, row++, 1, 1, 2);

    g->addWidget(label("Engine", "acesCliLabel", "value", this), row, 0);
    cli_ = new QLineEdit(this);
    cli_->setObjectName("acesCli");
    cli_->setPlaceholderText("rudra-native built with ACES v2 (OpenColorIO 2.5.2)");
    g->addWidget(cli_, row, 1);
    auto* pick_cli = button("Choose…", "acesCliPick", this);
    g->addWidget(pick_cli, row++, 2);
    cli_note_ = label("", "acesCliNote", "note", this);
    g->addWidget(cli_note_, row++, 1, 1, 2);
    g->setColumnStretch(1, 1);
    v->addLayout(g);

    progress_ = new QProgressBar(this);
    progress_->setObjectName("acesProgress");
    progress_->setRange(0, 1);
    progress_->setValue(0);
    v->addWidget(progress_);
    status_ = label("", "acesStatus", "value", this);
    v->addWidget(status_);
    log_ = new QPlainTextEdit(this);
    log_->setObjectName("acesLog");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(5000);
    v->addWidget(log_, 1);

    auto* buttons = new QHBoxLayout;
    open_out_ = button("Open output folder", "acesOpenOut", this);
    buttons->addWidget(open_out_);
    buttons->addStretch(1);
    start_ = button("Start", "acesStart", this);
    start_->setDefault(true);
    cancel_ = button("Cancel", "acesCancel", this);
    auto* close_b = button("Close", "acesClose", this);
    buttons->addWidget(start_);
    buttons->addWidget(cancel_);
    buttons->addWidget(close_b);
    v->addLayout(buttons);

    connect(package_, &QComboBox::currentIndexChanged, this, [this] { show_package(); });
    connect(add_pkg, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "ACES v2 model package (the folder with manifest.json)");
        if (d.isEmpty()) return;
        roots_.push_back(pq(d));
        QSettings s;
        QStringList r = s.value("aces/roots").toStringList();
        if (!r.contains(d)) r << d;
        s.setValue("aces/roots", r);
        s.setValue("aces/package", d);
        refresh_packages();
    });
    connect(input_, &QLineEdit::editingFinished, this, [this] { set_input(pq(input_->text())); });
    connect(pick_file, &QPushButton::clicked, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, "SDR still or video", input_->text(),
                                                       "Stills and videos (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp "
                                                       "*.mp4 *.mov *.mkv *.avi *.m4v *.mxf *.webm *.mpg *.mpeg *.ts *.mts *.m2ts *.wmv);;"
                                                       "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;"
                                                       "Videos (*.mp4 *.mov *.mkv *.avi *.m4v *.mxf *.webm *.mpg *.mpeg *.ts *.mts *.m2ts *.wmv)");
        if (!f.isEmpty()) set_input(pq(f));
    });
    connect(pick_dir, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Folder of SDR stills", input_->text());
        if (!d.isEmpty()) set_input(pq(d));
    });
    connect(out_, &QLineEdit::textChanged, this, [this] {
        hold_ = false;
        update_state();
    });
    connect(pick_out, &QPushButton::clicked, this, [this] {
        // A new folder is made by the run: pick where it goes, keep the suggested name.
        const QString where = QFileDialog::getExistingDirectory(this, "Where the new output folder goes",
                                                                 QFileInfo(out_->text()).absolutePath());
        if (where.isEmpty()) return;
        const QString name = QFileInfo(out_->text()).fileName().isEmpty() ? QStringLiteral("aces_master")
                                                                          : QFileInfo(out_->text()).fileName();
        out_->setText(QDir(where).filePath(name));
    });
    connect(target_, &QComboBox::currentIndexChanged, this, [this] {
        QSettings().setValue("aces/target", target_->currentData());
        hold_ = false;
        update_state();
        ensure_engine();
    });
    connect(cli_, &QLineEdit::editingFinished, this, [this] { set_cli(pq(cli_->text())); });
    connect(pick_cli, &QPushButton::clicked, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, "rudra-native with ACES v2", cli_->text());
        if (!f.isEmpty()) set_cli(pq(f));
    });
    connect(start_, &QPushButton::clicked, this, [this] { start(); });
    connect(cancel_, &QPushButton::clicked, this, [this] { cancel(); });
    connect(close_b, &QPushButton::clicked, this, [this] { close(); });
    connect(open_out_, &QPushButton::clicked, this, [this] {
        QDesktopServices::openUrl(QUrl::fromLocalFile(out_->text()));
    });

    connect(ev_, &QSlider::valueChanged, this, [this](int value) {
        ev_label_->setText(QStringLiteral("%1 EV").arg(value / 2.0, 0, 'f', 1));
        request_view();
    });
    connect(mode_, &QComboBox::currentIndexChanged, this, [this] { request_view(); });
    connect(preview_frame_, &QComboBox::currentIndexChanged, this, [this] { request_preview(); });
    connect(guard_, &QCheckBox::toggled, this, [this] { request_preview(); });
    connect(reload_, &QPushButton::clicked, this, [this] {
        if (engine_->busy()) {   // the Deliver tab or the batch is rendering on it
            hold_ = true;
            status_->setText(AcesEngine::busy_text());
            return;
        }
        if (engine_->state() == AcesEngine::State::Ready) engine_->stop();
        ensure_engine();
    });
    engine_->listen(this, [this] { on_engine_change(); });

    for (const auto& r : st.value("aces/roots").toStringList()) roots_.push_back(pq(r));
    refresh_packages();
    // The engine shipped beside the app comes first (an installed or copied
    // bundle); a development engine chosen earlier is the fallback.
    const fs::path beside = default_aces_cli();
    const QString saved_cli = st.value("aces/cli").toString();
    if (engine_->state() == AcesEngine::State::Ready) set_cli(engine_->config().cli);   // proven by the loaded model
    else if (!beside.empty() && probe_aces_cli(beside).ok) set_cli(beside);
    else set_cli(saved_cli.isEmpty() ? beside : pq(saved_cli));
    set_input(input);
    on_engine_change();
}

AcesMasterDialog::~AcesMasterDialog() {
    if (proc_ && proc_->state() != QProcess::NotRunning) {
        proc_->kill();
        proc_->waitForFinished(2000);
    }
}

void AcesMasterDialog::refresh_packages() {
    packages_ = find_aces_packages(roots_);
    const QString keep = package_->currentData().toString().isEmpty() ? QSettings().value("aces/package").toString()
                                                                       : package_->currentData().toString();
    package_->blockSignals(true);
    package_->clear();
    for (const auto& p : packages_)
        package_->addItem(QStringLiteral("%1  ·  %2").arg(qp(p.root.filename()), q(p.checkpoint_sha256.substr(0, 12))),
                          qp(p.root));
    if (const int i = package_->findData(keep); i >= 0) package_->setCurrentIndex(i);
    package_->blockSignals(false);
    show_package();
}

void AcesMasterDialog::show_package() {
    const int i = package_->currentIndex();
    if (i < 0 || std::size_t(i) >= packages_.size()) {
        package_note_->setText("No ACES v2 model package found. Add the folder that holds its manifest.json "
                               "(tools/export_native_v2.py writes it).");
        update_state();
        return;
    }
    const auto& p = packages_[std::size_t(i)];
    hold_ = false;
    QSettings().setValue("aces/package", qp(p.root));
    guard_->blockSignals(true);
    guard_->setChecked(p.luminance_guard);
    guard_->blockSignals(false);
    package_note_->setText(QStringLiteral("%1\nModel %2 · %3 · luminance guard %4 by default%5")
                               .arg(qp(p.root), q(p.g1_sha256.substr(0, 12)),
                                    p.has_onnx ? QStringLiteral("LibTorch and ONNX") : QStringLiteral("LibTorch only"),
                                    p.luminance_guard ? QStringLiteral("on") : QStringLiteral("off"),
                                    p.release_authorized ? QString() : QStringLiteral(" · research package, not release-authorised")));
    update_state();
    ensure_engine();
}

void AcesMasterDialog::set_input(const fs::path& input) {
    hold_ = false;
    input_->setText(qp(input));
    stills_ = input.empty() ? std::vector<fs::path>{} : collect_aces_stills(input);
    total_frames_ = stills_.size();
    video_ = false;
    QString note;
    if (stills_.size() == 1 && is_aces_video(stills_.front())) {
        // A video: every frame is mastered; the probe gives the count.
        video_ = true;
        auto vs = rudra::aces_open_video(stills_.front());
        if (!vs) {
            note = QStringLiteral("This video cannot be read: %1 %2").arg(q(vs.error().message), q(vs.error().detail));
            stills_.clear();
            total_frames_ = 0;
        } else {
            total_frames_ = std::size_t(std::max(0, vs->clock.frames));
            note = QStringLiteral("Video: %1 frames, %2 × %3, %4 fps, %5-bit → %6_000000.exr …")
                       .arg(total_frames_).arg(vs->stream.width).arg(vs->stream.height).arg(q(vs->clock.fps))
                       .arg(rudra::aces_video_bits(vs->stream.pix_fmt)).arg(qp(stills_.front().stem()));
        }
    }
    if (input.empty()) frames_note_->setText({});
    else if (!note.isEmpty()) frames_note_->setText(note);
    else if (stills_.empty()) frames_note_->setText("No supported stills or video there.");
    else if (const auto c = aces_name_collision(stills_))
        frames_note_->setText(QStringLiteral("%1 would overwrite another frame's EXR: rename it first.").arg(qp(c->filename())));
    else
        frames_note_->setText(QStringLiteral("%1 %2").arg(stills_.size()).arg(stills_.size() == 1 ? "still" : "stills"));
    if (!stills_.empty() && (out_->text().isEmpty() || !running())) out_->setText(qp(suggest_out(input)));
    preview_frame_->blockSignals(true);
    preview_frame_->clear();
    if (video_ && total_frames_ > 0) {
        // Up to 12 frames spread over the clip, first and last included.
        const std::size_t n = std::min<std::size_t>(12, total_frames_);
        std::vector<std::size_t> picks;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t k = n == 1 ? 0 : (i * (total_frames_ - 1) + (n - 1) / 2) / (n - 1);
            if (picks.empty() || picks.back() != k) picks.push_back(k);
        }
        for (std::size_t k : picks) {
            preview_frame_->addItem(QStringLiteral("%1 · frame %2").arg(qp(stills_.front().filename())).arg(k), qp(stills_.front()));
            preview_frame_->setItemData(preview_frame_->count() - 1, int(k), Qt::UserRole + 1);
        }
    } else {
        for (const auto& f : stills_) preview_frame_->addItem(qp(f.filename()), qp(f));
    }
    preview_frame_->blockSignals(false);
    update_state();
    request_preview();
}

QString AcesMasterDialog::frames_text(std::size_t n) const {
    if (video_) return QStringLiteral("%1 video %2").arg(n).arg(n == 1 ? "frame" : "frames");
    return QStringLiteral("%1 %2").arg(n).arg(n == 1 ? "still" : "stills");
}

void AcesMasterDialog::set_cli(const fs::path& cli) {
    hold_ = false;
    cli_->setText(qp(cli));
    const bool proven = engine_ && engine_->state() == AcesEngine::State::Ready && engine_->config().cli == cli;
    const AcesCliProbe probe = proven ? AcesCliProbe{true, {}} : probe_aces_cli(cli);
    cli_ok_ = probe.ok;
    if (cli.empty())
        cli_note_->setText("No rudra-native beside the app. Choose one built with ACES v2 "
                           "(scripts/NATIVE_ACES_V2.ps1 builds it in build/native_aces_v2/cli/Release).");
    else if (!cli_ok_)
        cli_note_->setText(probe.detail);
    else {
        cli_note_->setText("Ready.");
        QSettings().setValue("aces/cli", qp(cli));
    }
    update_state();
    ensure_engine();
}

AcesRunSettings AcesMasterDialog::settings() const {
    AcesRunSettings s;
    s.cli = pq(cli_->text());
    s.package = pq(package_->currentData().toString());
    s.input = pq(input_->text());
    s.out = pq(out_->text());
    const auto& t = aces_run_targets()[std::size_t(std::max(0, target_->currentIndex()))];
    s.runtime = t.runtime;
    s.device = t.device;
    s.luminance_guard = guard_->isChecked();
    s.log_formats = log_format_->currentData().toString().toStdString();
    if (!s.log_formats.empty()) s.log_curve = "logc4";
    return s;
}

bool AcesMasterDialog::ready(QString* why) const {
    auto no = [why](const QString& w) {
        if (why) *why = w;
        return false;
    };
    if (package_->currentIndex() < 0) return no("Add an ACES v2 model package.");
    if (!cli_ok_) return no("Choose a rudra-native built with ACES v2.");
    if (stills_.empty() || total_frames_ == 0) return no("Choose a still, a folder of stills or a video.");
    if (aces_name_collision(stills_)) return no("Two frames map to the same EXR name.");
    const QString out = out_->text().trimmed();
    if (out.isEmpty()) return no("Choose an output folder.");
    if (QFileInfo::exists(out)) return no("The output folder exists: choose a new one.");
    const auto& t = aces_run_targets()[std::size_t(std::max(0, target_->currentIndex()))];
    if (t.runtime == "onnxruntime") {
        const int i = package_->currentIndex();
        if (!packages_[std::size_t(i)].has_onnx) return no("This package has no ONNX model: run it on LibTorch.");
    }
    return true;
}

bool AcesMasterDialog::running() const { return engine_master_ || (proc_ && proc_->state() != QProcess::NotRunning); }

void AcesMasterDialog::update_state() {
    const bool busy = running();
    QString why;
    const bool ok = ready(&why);
    start_->setEnabled(!busy && ok);
    cancel_->setEnabled(busy);
    open_out_->setEnabled(QFileInfo(out_->text()).isDir());
    for (QWidget* w : std::initializer_list<QWidget*>{package_, target_, guard_, log_format_, input_, out_, cli_}) w->setEnabled(!busy);
    // A finished run's summary stays until something is changed.
    if (!busy && !hold_)
        status_->setText(ok ? QStringLiteral("Ready: %1.").arg(frames_text(total_frames_)) : why);
}

void AcesMasterDialog::start() {
    QString why;
    if (running() || !ready(&why)) {
        if (!why.isEmpty()) status_->setText(why);
        return;
    }
    const AcesRunSettings s = settings();
    if (engine_->state() == AcesEngine::State::Ready && engine_->config().package == s.package &&
        engine_->config().runtime == s.runtime && engine_->config().device == s.device) {
        start_with_engine(s);
        return;
    }
    log_->clear();
    pending_.clear();
    done_ = 0;
    cancelled_ = false;
    progress_->setRange(0, int(total_frames_));
    progress_->setValue(0);
    status_->setText(QStringLiteral("Loading the model on %1…").arg(target_->currentText()));
    const QStringList args = aces_cli_arguments(s);
    log_->appendPlainText(qp(s.cli) + " " + args.join(' '));
    if (!proc_) {
        proc_ = new QProcess(this);
        proc_->setProcessChannelMode(QProcess::SeparateChannels);
        connect(proc_, &QProcess::readyReadStandardOutput, this, [this] { on_output(); });
        connect(proc_, &QProcess::finished, this,
                [this](int code, QProcess::ExitStatus st) { on_finished(code, st == QProcess::CrashExit); });
        connect(proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
            if (e == QProcess::FailedToStart) {
                hold_ = true;
                status_->setText("rudra-native did not start.");
                update_state();
            }
        });
    }
    proc_->setProcessEnvironment(aces_child_environment());
    proc_->start(qp(s.cli), args);
    update_state();
}

void AcesMasterDialog::cancel() {
    if (!running()) return;
    cancelled_ = true;
    if (engine_master_) engine_->cancel_master(engine_master_id_);   // stops after the frame in hand; the model stays loaded
    else proc_->kill();
}

void AcesMasterDialog::on_output() {
    pending_ += proc_->readAllStandardOutput().toStdString();
    std::size_t nl;
    while ((nl = pending_.find('\n')) != std::string::npos) {
        std::string line = pending_.substr(0, nl);
        pending_.erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        log_->appendPlainText(q(line));
        if (const auto p = parse_aces_progress(line)) {
            ++done_;
            progress_->setValue(int(std::min(done_, total_frames_)));
            status_->setText(QStringLiteral("%1 of %2 · %3 · %4 ms").arg(done_).arg(total_frames_)
                                 .arg(q(p->exr)).arg(p->milliseconds, 0, 'f', 0));
        }
    }
}

void AcesMasterDialog::on_finished(int code, bool crashed) {
    on_output();
    const QString err = QString::fromUtf8(proc_->readAllStandardError()).trimmed();
    if (!err.isEmpty()) log_->appendPlainText(err);
    const fs::path out = pq(out_->text());
    hold_ = true;
    if (cancelled_) {
        status_->setText(QStringLiteral("Cancelled after %1 of %2 frames. The finished EXRs are in the output folder; "
                                        "it has no provenance.json.").arg(done_).arg(total_frames_));
    } else if (!crashed && (code == 0 || code == 3)) {
        auto s = read_aces_provenance(out);
        if (s) status_->setText(q(aces_summary_text(*s)) + (code == 3 ? " The run reported frames to check." : ""));
        else status_->setText(q(s.error().message));
        progress_->setValue(progress_->maximum());
    } else {
        const QString first = err.section('\n', 0, 0);
        status_->setText(QStringLiteral("The master stopped (exit %1). %2").arg(code).arg(first));
    }
    update_state();
}

// ---- the model kept loaded, and the preview through it ----------------------

void AcesMasterDialog::ensure_engine() {
    if (!engine_ || !cli_ok_ || package_->currentIndex() < 0 || running()) return;
    AcesEngine::Config c;
    c.cli = pq(cli_->text());
    c.package = pq(package_->currentData().toString());
    const auto& t = aces_run_targets()[std::size_t(std::max(0, target_->currentIndex()))];
    c.runtime = t.runtime;
    c.device = t.device;
    if (c == engine_->config() && engine_->state() == AcesEngine::State::Failed) return;   // "Load model" retries
    if (!engine_->start(c)) {   // a render elsewhere holds it: the model is not swapped under it
        hold_ = true;
        status_->setText(AcesEngine::busy_text() + " The new target or package loads afterwards (Load model).");
    }
}

void AcesMasterDialog::on_engine_change() {
    const auto st = engine_->state();
    engine_note_->setText(engine_->status_text());
    reload_->setText(st == AcesEngine::State::Ready ? "Reload model" : "Load model");
    reload_->setEnabled(!running() && st != AcesEngine::State::Loading && cli_ok_ && package_->currentIndex() >= 0);
    if (st == AcesEngine::State::Ready) request_preview();
    else if (st == AcesEngine::State::Loading) image_->setText("Loading the model…");
    else if (st == AcesEngine::State::Failed) image_->setText("No preview: " + engine_->status_text());
    update_state();
}

void AcesMasterDialog::request_preview() {
    if (!engine_ || preview_frame_->currentIndex() < 0) return;
    if (engine_->state() != AcesEngine::State::Ready) return;
    if (preview_busy_) {
        preview_again_ = true;
        return;
    }
    preview_busy_ = true;
    const QString path = preview_frame_->currentData().toString();
    const QVariant frame = preview_frame_->currentData(Qt::UserRole + 1);
    preview_stats_->setText(QStringLiteral("Running the model on %1…").arg(preview_frame_->currentText()));
    const QPointer<QObject> self(this);
    engine_->request(QJsonObject{{"op", "preview"}, {"source", path}, {"frame", frame.isValid() ? frame.toInt() : 0},
                                 {"guard", guard_->isChecked()}},
                     [this, self](const QJsonObject& o) {
                         if (!self) return;
                         preview_busy_ = false;
                         if (preview_again_) {
                             preview_again_ = false;
                             request_preview();
                             return;
                         }
                         if (!o.value("ok").toBool()) {
                             preview_stats_->setText("Preview failed: " + o.value("error").toString() + " " +
                                                     o.value("detail").toString());
                             return;
                         }
                         preview_stats_->setText(
                             QStringLiteral("%1 × %2, %3-bit · model %4 ms · %5 % reconstructed · peak %6× diffuse white "
                                            "(plain inverse %7×) · %8 display errors · %9 new negative pixels")
                                 .arg(o.value("width").toInt())
                                 .arg(o.value("height").toInt())
                                 .arg(o.value("bits").toInt())
                                 .arg(o.value("ms").toDouble(), 0, 'f', 0)
                                 .arg(100.0 * o.value("reconstructed_fraction").toDouble(), 0, 'f', 2)
                                 .arg(o.value("peak").toDouble(), 0, 'f', 1)
                                 .arg(o.value("peak_plain").toDouble(), 0, 'f', 1)
                                 .arg(o.value("remaining_display_errors").toInt())
                                 .arg(o.value("new_negative_luminance").toInt()));
                         request_view();
                     });
}

void AcesMasterDialog::request_view() {
    if (!engine_ || engine_->state() != AcesEngine::State::Ready || preview_busy_ || !tmp_ || !tmp_->isValid()) return;
    if (preview_frame_->currentIndex() < 0) return;
    if (view_busy_) {
        view_again_ = true;
        return;
    }
    view_busy_ = true;
    const QString out = tmp_->filePath(QStringLiteral("view_%1.ppm").arg(views_++ % 2));
    const QPointer<QObject> self(this);
    QJsonObject view_req{{"op", "view"}, {"ev", ev_->value() / 2.0}, {"mode", mode_->currentData().toString()},
                         {"max_side", 1600}, {"out", out}};
    if (mode_->currentData().toString() == "fill") {   // the Deliver tab's AI fill thresholds (genfill_job.hpp)
        QSettings st;
        view_req["tau_high"] = st.value("deliver/aiTauHigh", 250).toInt();
        view_req["tau_low"] = st.value("deliver/aiTauLow", -1).toInt();
    }
    engine_->request(view_req,
                     [this, self, out](const QJsonObject& o) {
                         if (!self) return;
                         view_busy_ = false;
                         if (view_again_) {
                             view_again_ = false;
                             request_view();
                             return;
                         }
                         if (!o.value("ok").toBool()) {
                             image_->setText("Preview failed: " + o.value("error").toString());
                             return;
                         }
                         QPixmap pm(out);
                         if (pm.isNull()) {
                             image_->setText("The preview image could not be read.");
                             return;
                         }
                         image_->setPixmap(pm.scaled(image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
                     });
}

void AcesMasterDialog::preview_frame(int index) { preview_frame_->setCurrentIndex(index); }
void AcesMasterDialog::set_preview_ev(double ev) { ev_->setValue(int(std::lround(ev * 2.0))); }
void AcesMasterDialog::set_preview_mode(const QString& key) {
    if (const int i = mode_->findData(key); i >= 0) mode_->setCurrentIndex(i);
}

void AcesMasterDialog::start_with_engine(const AcesRunSettings& s) {
    log_->clear();
    done_ = 0;
    cancelled_ = false;
    engine_master_ = true;
    progress_->setRange(0, int(total_frames_));
    progress_->setValue(0);
    status_->setText(QStringLiteral("Mastering %1 on the loaded model…").arg(frames_text(total_frames_)));
    log_->appendPlainText(QStringLiteral("loaded model (%1): master %2 -> %3").arg(engine_->status_text(), qp(s.input), qp(s.out)));
    QJsonArray sources;
    for (const auto& f : stills_) sources.append(qp(f));
    update_state();
    const QPointer<QObject> self(this);
    const fs::path out = s.out;
    engine_master_id_ = engine_->request(
        QJsonObject{{"op", "master"}, {"sources", sources}, {"out", qp(s.out)}, {"guard", s.luminance_guard},
                    {"log", q(s.log_curve)}, {"log_formats", q(s.log_formats)}},
        [this, self, out](const QJsonObject& o) {
            if (!self) return;
            engine_master_ = false;
            hold_ = true;
            if (!o.value("ok").toBool()) {
                status_->setText("The master stopped: " + o.value("error").toString() + " " + o.value("detail").toString());
            } else if (o.value("cancelled").toBool()) {
                status_->setText(QStringLiteral("Cancelled after %1 of %2 frames. The finished EXRs and their provenance.json "
                                                "are in the output folder.").arg(done_).arg(total_frames_));
            } else {
                auto sum = read_aces_provenance(out);
                status_->setText(sum ? q(aces_summary_text(*sum)) : q(sum.error().message));
                progress_->setValue(progress_->maximum());
            }
            update_state();
            on_engine_change();
        },
        [this, self](const QJsonObject& p) {
            if (!self) return;
            done_ = std::size_t(p.value("index").toInt());
            progress_->setValue(int(std::min(done_, total_frames_)));
            log_->appendPlainText(QStringLiteral("%1 -> %2  %3 ms  remaining_display_errors=%4")
                                      .arg(p.value("source").toString(), p.value("exr").toString())
                                      .arg(p.value("ms").toDouble(), 0, 'f', 0)
                                      .arg(p.value("remaining_display_errors").toInt()));
            status_->setText(QStringLiteral("%1 of %2 · %3 · %4 ms").arg(done_).arg(total_frames_)
                                 .arg(p.value("exr").toString()).arg(p.value("ms").toDouble(), 0, 'f', 0));
        });
}

// ---- the window's entry ------------------------------------------------------

void MainWindow::open_aces_master() {
    fs::path input;
    if (!shot_sources_.empty()) {
        std::error_code ec;
        input = shot_sources_.size() == 1 ? shot_sources_.front() : shot_sources_.front().parent_path();
        if (fs::is_directory(shot_sources_.front(), ec)) input = shot_sources_.front();
    }
    if (!aces_master_) aces_master_ = new AcesMasterDialog(this, model_roots(), input, aces_engine());
    else if (!static_cast<AcesMasterDialog*>(aces_master_.data())->running()) {
        auto* d = static_cast<AcesMasterDialog*>(aces_master_.data());
        d->refresh_packages();
        if (!input.empty()) d->set_input(input);
    }
    aces_master_->show();
    aces_master_->raise();
}

AcesEngine* MainWindow::aces_engine() {
    if (!aces_engine_) {
        aces_engine_ = std::make_shared<AcesEngine>(this);
        aces_engine_->listen(this, [this] {
            const AcesEngine::State st = aces_engine_->state();
            if (st == AcesEngine::State::Stopped) return;
            const QString text = aces_engine_->status_text();
            log(text);
            if (statusBar()) statusBar()->showMessage(text, st == AcesEngine::State::Loading ? 0 : 10000);
            if (aces_view_ && ckpt_) {
                if (st == AcesEngine::State::Ready)
                    ckpt_->setText(QStringLiteral("learned ACES 2.0 (%1)").arg(q(aces_engine_->config().device)));
                else if (st == AcesEngine::State::Loading)
                    ckpt_->setText(QStringLiteral("loading learned ACES 2.0…"));
            }
            // Ready: the frame on screen is shown again through the model; lost: the legacy path again.
            if (!frames_.empty()) step_to(current_);
            refresh_enabled();
            sync_deliver();   // the Deliver tab's source follows the engine
            batch_changed();  // the batch window says whether it can start
            show_mode();
        });
    }
    return aces_engine_.get();
}

// ---- the main viewer through the learned ACES 2.0 model ----------------------

bool MainWindow::aces_routed() const {
    return aces_view_ && aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready && !frames_.empty() &&
           !headroom_routed();
}

void MainWindow::set_aces_viewer(bool on) {
    aces_view_ = on;
    QSettings().setValue("aces/viewer", on);
    if (!on) aces_playing_ = false;
    if (!frames_.empty()) step_to(current_);
}

namespace {
// Raw planar float32 (R, G, B planes) as aces-v2-serve writes it.
std::optional<PlanarBuffer> read_planes(const QString& path, int w, int h) {
    PlanarBuffer b(3, h, w);
    std::ifstream in(pq(path), std::ios::binary);
    for (int c = 0; c < 3 && in; ++c)
        in.read(reinterpret_cast<char*>(b.plane(c)), std::streamsize(b.plane_size() * sizeof(float)));
    if (!in) return std::nullopt;
    return b;
}
// ACEScg (1.0 = diffuse white = 203 nits, as the delivery reads it) to the
// viewer's precomposited convention: linear Rec.709, 1.0 = 10 000 nits. No clamp.
NetworkLinearImage aces_view_image(const PlanarBuffer& acescg) {
    PlanarBuffer out(3, acescg.height(), acescg.width());
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Rec709);
    constexpr double k = 203.0 / 10000.0;
    float f[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) f[r][c] = float(m[std::size_t(r)][std::size_t(c)] * k);
    const float *ir = acescg.plane(0), *ig = acescg.plane(1), *ib = acescg.plane(2);
    float *orr = out.plane(0), *og = out.plane(1), *ob = out.plane(2);
    for (std::size_t i = 0; i < acescg.plane_size(); ++i) {
        const float r = ir[i], g = ig[i], bl = ib[i];
        orr[i] = f[0][0] * r + f[0][1] * g + f[0][2] * bl;
        og[i] = f[1][0] * r + f[1][1] * g + f[1][2] * bl;
        ob[i] = f[2][0] * r + f[2][1] * g + f[2][2] * bl;
    }
    return NetworkLinearImage(std::move(out));
}
}  // namespace

namespace {
// A small picture of the frame as the viewer shows it: the viewer's linear
// (1.0 = 10 000 nits) at diffuse white = 1.0, a soft shoulder, sRGB codes.
QImage aces_thumb(const NetworkLinearImage& img, int height) {
    const auto& b = img.buffer();
    if (b.width() <= 0 || b.height() <= 0) return {};
    const int h = std::max(1, std::min(height, b.height()));
    const int w = std::max(1, int(std::lround(double(b.width()) * h / b.height())));
    QImage out(w, h, QImage::Format_RGB888);
    for (int y = 0; y < h; ++y) {
        uchar* row = out.scanLine(y);
        const int sy = std::min(b.height() - 1, int((y + 0.5) * b.height() / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(b.width() - 1, int((x + 0.5) * b.width() / w));
            for (int c = 0; c < 3; ++c) {
                double v = std::max(0.0, double(b.at(c, sy, sx)) * 10000.0 / 203.0);
                v = v / (1.0 + 0.25 * v) * 1.25;   // 1.0 stays 1.0; highlights roll off
                const double s = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(std::min(v, 1.0), 1.0 / 2.4) - 0.055;
                row[x * 3 + c] = uchar(std::clamp(int(std::lround(s * 255.0)), 0, 255));
            }
        }
    }
    return out;
}
}  // namespace

void MainWindow::aces_request(int index) {
    if (frames_.empty() || !aces_engine_) return;
    const int n = int(frames_.size());
    index = ((index % n) + n) % n;
    if (aces_busy_) {
        aces_next_ = index;   // the newest ask wins; it runs when this one is back
        return;
    }
    if (!aces_tmp_) aces_tmp_ = std::make_unique<QTemporaryDir>();
    if (!aces_tmp_->isValid()) {
        log("learned ACES view: no temporary folder");
        return;
    }
    // A movie: the file and the frame number (the engine decodes it); stills: the file.
    const bool movie = shot_video_.has_value() && !shot_sources_.empty();
    const QString source = movie ? qp(shot_sources_.front()) : qp(frames_[std::size_t(index)]);
    const QString model_path = aces_tmp_->filePath(QStringLiteral("model_%1.f32").arg(index % 2));
    const QString base_path = aces_tmp_->filePath(QStringLiteral("baseline_%1.f32").arg(index % 2));
    aces_busy_ = true;
    const QPointer<MainWindow> self(this);
    QJsonObject req{{"op", "preview"}, {"source", source}, {"frame", movie ? index : 0}, {"keep", false},
                    {"model_out", model_path}, {"baseline_out", base_path}};
    // The monitoring transform, the Invented layer's codes, and a proxy while playing (28 Oct 2026).
    if (aces_monitor_ != "scene") req["display"] = aces_monitor_;
    if (aces_source_display_ != "bt1886") req["source_display"] = aces_source_display_;   // an sRGB / 2.2 master
    req["codes_out"] = aces_tmp_->filePath(QStringLiteral("codes_%1.f32").arg(index % 2));
    if (aces_playing_ && play_state_->side.load() > 0) req["max_side"] = play_state_->side.load();
    // The colorist's trims: the viewer shows what a render writes.
    if (const QJsonObject t = aces_trims_fields(aces_trims_.strength, aces_trims_.saturation, aces_trims_.peak_nits,
                                                 aces_trims_.exposure_ev, aces_trims_.contrast); !t.isEmpty())
        req["trims"] = t;
    aces_engine_->request(req,
                          [self, index, model_path, base_path](const QJsonObject& o) {
                              if (self) self->aces_after_frame(o, index, model_path, base_path);
                          });
}

void MainWindow::aces_after_frame(const QJsonObject& o, int index, const QString& model_path, const QString& base_path) {
    aces_busy_ = false;
    const bool routed = aces_routed();
    // Asked again while this ran (another frame, or the same one with new trims or monitor): show that.
    if (routed && aces_next_ >= 0) {
        const int next = aces_next_;
        aces_next_ = -1;
        aces_request(next);   // the user has moved on: show where they are now
        return;
    }
    aces_next_ = -1;
    if (!routed) return;
    waiting_ = false;
    if (!o.value("ok").toBool()) {
        aces_playing_ = false;
        if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
        problem_ = QStringLiteral("Learned ACES view: ") + o.value("error").toString() + ' ' + o.value("detail").toString();
        log(problem_);
        sync_ui();
        return;
    }
    const int w = o.value("width").toInt(), h = o.value("height").toInt();
    auto model = read_planes(model_path, w, h);
    auto base = read_planes(base_path, w, h);
    if (!model || !base) {
        log("learned ACES view: the frame could not be read back");
        return;
    }
    problem_.clear();
    aces_shown_ = index;
    current_ = index;
    aces_model_img_ = std::make_shared<const NetworkLinearImage>(aces_view_image(*model));
    aces_base_img_ = std::make_shared<const NetworkLinearImage>(aces_view_image(*base));
    // The SDR's largest code per pixel: the Invented layer tells invented from reinterpreted pixels.
    std::vector<float> codes(std::size_t(w) * std::size_t(h));
    {
        QString cp = model_path;
        cp.replace(QStringLiteral("model_"), QStringLiteral("codes_"));
        std::ifstream in(pq(cp), std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(codes.data()), std::streamsize(codes.size() * sizeof(float)))) codes.clear();
    }
    aces_codes_ = codes;
    // A reference still compared: it takes the baseline's place (Baseline, Wipe, B), when it is the same size.
    const auto* ref = ref_compare_ ? reference_still() : nullptr;
    aces_ref_shown_ = ref && ref->width() == aces_model_img_->width() && ref->height() == aces_model_img_->height();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_composited(*aces_model_img_, aces_ref_shown_ ? *ref : *aces_base_img_, codes);
#endif
    sync_reference_ui();
    show_mode();   // the status bar says the learned model, not the analytic reconstruction
    // The filmstrip's picture of this frame, from what the viewer shows (29 Oct 2026).
    if (scrub_ && index < scrub_->count()) {
        const QImage t = aces_thumb(*aces_model_img_, 88);
        scrub_->set_thumb(index, t);
        if (index == 0 || first_thumb_.isNull()) {
            first_thumb_ = t;
            refresh_library();
        }
    }
    // The playback cache: a frame rendered at the playback size while playing goes in, made with
    // what the screen shows now (anything changed empties it).
    if (aces_playing_ && play_state_->side.load() > 0) {
        if (const QString key = aces_cache_key(); key != aces_cache_key_ || aces_cache_.size() != frames_.size()) {
            aces_cache_.assign(frames_.size(), nullptr);
            aces_cache_bytes_ = 0;
            aces_cache_key_ = key;
        }
        const std::size_t bytes = std::size_t(w) * std::size_t(h) * 3 * sizeof(float);
        const std::size_t budget = std::size_t(QSettings().value("aces/playCacheMB", 2048).toInt()) * 1024 * 1024;
        if (!aces_cache_[std::size_t(index)] && aces_cache_bytes_ + bytes <= budget) {
            aces_cache_[std::size_t(index)] = aces_model_img_;
            aces_cache_bytes_ += bytes;
        }
    }
    run_stats();   // the readouts and scopes of this frame, in nits (28 Oct 2026)
    if (parade_) parade_->set_scope(qc_scope(aces_model_img_->buffer()));   // parade and gamut (28 Oct 2026 review)
    sync_check_summary();   // Simple's Check reads the same
    frame_info_ = QStringLiteral("%1  %2/%3  \u00b7  learned ACES 2.0, %4x%5, %6 ms, %7 % rebuilt, peak %8x (plain %9x)")
                      .arg(QString::fromStdString(frame_name(std::size_t(index))))
                      .arg(index + 1)
                      .arg(frames_.size())
                      .arg(w)
                      .arg(h)
                      .arg(o.value("ms").toDouble(), 0, 'f', 0)
                      .arg(100.0 * o.value("reconstructed_fraction").toDouble(), 0, 'f', 2)
                      .arg(o.value("peak").toDouble(), 0, 'f', 1)
                      .arg(o.value("peak_plain").toDouble(), 0, 'f', 1);
    sync_ui();
    show_status();
    if (aces_playing_) {
        // Plays as fast as the model runs, filling the cache; the next play is real time.
        if (index + 1 < int(frames_.size())) aces_request(index + 1);
        else {
            aces_playing_ = false;
            if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
            const int n = aces_cached_frames();
            log(n == int(frames_.size()) ? QStringLiteral("playback cache: all %1 frames in; Play again for real time").arg(n)
                                         : QStringLiteral("playback cache: %1 of %2 frames (the memory budget, aces/playCacheMB)")
                                               .arg(n)
                                               .arg(frames_.size()));
        }
    }
}

QString MainWindow::aces_cache_key() const {
    return shot_trim_key() + "|" + QString::fromStdString(aces_v2::trims_text(aces_trims_)) + "|" + aces_monitor_ + "|" +
           aces_source_display_ + "|" + QString::number(play_state_->side.load());
}

int MainWindow::aces_cached_frames() const {
    if (aces_cache_key_ != aces_cache_key() || aces_cache_.size() != frames_.size()) return 0;
    return int(std::count_if(aces_cache_.begin(), aces_cache_.end(), [](const auto& p) { return p != nullptr; }));
}

void MainWindow::aces_rt_stop() {
    if (aces_rt_timer_) aces_rt_timer_->stop();
    if (!aces_rt_playing_) return;
    aces_rt_playing_ = false;
    if (btn_play_) btn_play_->set_glyph(IconButton::Glyph::Play);
    // The full-size frame where it stopped.
    if (aces_routed() && !frames_.empty()) aces_request(current_);
}

void MainWindow::preload_aces_engine() {
    if (!QSettings().value("aces/preload", true).toBool()) return;
    const auto c = aces_engine_config(model_roots());
    if (!c) return;   // no ACES engine or package here: nothing to load, nothing to say
    aces_engine()->start(*c);
}

}  // namespace rudra::app
