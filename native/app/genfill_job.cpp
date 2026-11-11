#include "genfill_job.hpp"
#include "model_downloads.hpp"
#include "paths.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QCryptographicHash>
#include <QDateTime>
#include <QSettings>
#include <QStandardPaths>
#include <QTimer>

#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QString qp(const fs::path& p) { return from_path(p); }
fs::path pq(const QString& s) { return to_path(s); }

// The repository: the saved one, else the Python a package carries beside the app (7 Nov
// 2026: <app>/python on Windows, RUDRA.app/Contents/Resources/python on macOS), else the first
// folder upwards of the app (or the working directory) with rudra/genfill/run.py.
fs::path find_repo() {
    QSettings st;
    const QString saved = st.value("genfill/repo").toString();
    if (!saved.isEmpty()) return pq(saved);
    const QDir app(QCoreApplication::applicationDirPath());
    for (const QString& packaged : {app.filePath("python"), app.filePath("../Resources/python")})
        if (QFileInfo::exists(packaged + "/rudra/genfill/run.py")) return pq(QDir::cleanPath(packaged));
    for (const QString& start : {QCoreApplication::applicationDirPath(), QDir::currentPath()}) {
        QDir d(start);
        for (int i = 0; i < 8; ++i) {
            if (QFileInfo::exists(d.filePath("rudra/genfill/run.py"))) return pq(d.absolutePath());
            if (!d.cdUp()) break;
        }
    }
    return {};
}

// The model's parts, where a setup puts them (28 Oct 2026 review): beside the app,
// in the repository, under RUDRA_GENFILL_DIR, then the training workstation's
// folders last. The first that exists; else the first place (what Check reports).
fs::path first_existing(const std::vector<fs::path>& places, const char* probe) {
    std::error_code ec;
    for (const auto& p : places)
        if (!p.empty() && fs::exists(probe ? p / probe : p, ec)) return p;
    return places.empty() ? fs::path() : places.front();
}

// The training workstation's own folders on G: are looked in only on a developer's machine
// (security review, 11 Nov 2026). They were in every build: on anyone else's computer a drive that
// happens to be G: could then supply the base model, which is not checked by hash as the LoRA is.
// A developer's machine: RUDRA_DEV_PATHS is set, or the app runs from a source checkout (no
// python\ beside it as a package has, and a folder above it holds both rudra/genfill and .git).
bool training_workstation() {
    if (qEnvironmentVariableIsSet("RUDRA_DEV_PATHS")) return true;
    static const bool checkout = [] {
        const QDir app(QCoreApplication::applicationDirPath());
        for (const QString& packaged : {app.filePath("python"), app.filePath("../Resources/python")})
            if (QFileInfo::exists(packaged + "/rudra/genfill/run.py")) return false;
        QDir d(app);
        for (int i = 0; i < 8; ++i) {
            if (QFileInfo::exists(d.filePath("rudra/genfill/run.py"))) return QFileInfo::exists(d.filePath(".git"));
            if (!d.cdUp()) break;
        }
        return false;
    }();
    return checkout;
}

std::vector<fs::path> genfill_roots(const fs::path& repo) {
    std::vector<fs::path> r = {pq(QCoreApplication::applicationDirPath()) / "models" / "genfill"};
    if (!repo.empty()) r.push_back(repo / "models" / "genfill");
    if (const QString env = qEnvironmentVariable("RUDRA_GENFILL_DIR"); !env.isEmpty()) r.push_back(pq(env));
    return r;
}

// Wan2.1-VACE-1.3B in a Hugging Face cache (4 Nov 2026): a setup that ran
// snapshot_download without local_dir, or any other tool that fetched it,
// leaves hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/<rev>/.
// The snapshot refs/main names first, then any other.
void hf_cache_snapshots(std::vector<fs::path>& out) {
    std::vector<fs::path> hubs;
    for (const char* v : {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"})
        if (const QString e = qEnvironmentVariable(v); !e.isEmpty()) hubs.push_back(pq(e));
    if (const QString e = qEnvironmentVariable("HF_HOME"); !e.isEmpty()) hubs.push_back(pq(e) / "hub");
    hubs.push_back(pq(QDir::homePath()) / ".cache" / "huggingface" / "hub");
    if (training_workstation()) hubs.push_back("G:/datasets/models/_hf_home/hub");   // GENFILL.ps1's HF_HOME
    std::error_code ec;
    for (const auto& hub : hubs) {
        const fs::path repo = hub / "models--Wan-AI--Wan2.1-VACE-1.3B-diffusers";
        if (!fs::is_directory(repo / "snapshots", ec)) continue;
        QFile ref(qp(repo / "refs" / "main"));
        if (ref.open(QIODevice::ReadOnly)) {
            const QString rev = QString::fromUtf8(ref.readAll()).trimmed();
            if (!rev.isEmpty()) out.push_back(repo / "snapshots" / pq(rev));
        }
        for (const auto& e : fs::directory_iterator(repo / "snapshots", ec))
            if (e.is_directory(ec)) out.push_back(e.path());
    }
}

// The LoRA wherever File > Model packages put it: any folder of the download
// root that holds rudra_genfill.safetensors, newest version name last-sorted first.
void downloaded_loras(std::vector<fs::path>& out) {
    QDir root(default_download_root());
    QStringList dirs = root.entryList({"rudra_genfill*"}, QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    std::sort(dirs.begin(), dirs.end(), [](const QString& a, const QString& b) { return a > b; });   // v3 before v2
    for (const QString& d : dirs) out.push_back(pq(root.filePath(d)) / "rudra_genfill.safetensors");
}

// A saved path wins while it is there; gone (moved, deleted), the search again.
// The search runs only then: Deliver asks for the setup on every refresh.
// genfill/search false (the tests; a site that wants only its own paths): a saved
// path is kept even when it is gone, and nothing is searched past it.
template <class Search>
fs::path saved_or(const QString& key, const char* probe, Search&& search) {
    QSettings st;
    const QString saved = st.value(key).toString();
    std::error_code ec;
    if (!saved.isEmpty() && fs::exists(probe ? pq(saved) / probe : pq(saved), ec)) return pq(saved);
    if (!saved.isEmpty() && !st.value("genfill/search", true).toBool()) return pq(saved);
    return search();
}

}  // namespace

std::vector<fs::path> genfill_base_candidates(const fs::path& repo) {
    std::vector<fs::path> bases;
    for (const auto& r : genfill_roots(repo)) bases.push_back(r / "Wan2.1-VACE-1.3B-diffusers");
    bases.push_back(pq(default_download_root()) / "Wan2.1-VACE-1.3B-diffusers");
    hf_cache_snapshots(bases);
    if (training_workstation()) bases.push_back("G:/datasets/models/Wan2.1-VACE-1.3B-diffusers");
    return bases;
}

std::vector<fs::path> genfill_lora_candidates(const fs::path& repo) {
    std::vector<fs::path> loras;
    downloaded_loras(loras);   // File > Model packages (31 Oct 2026)
    for (const auto& r : genfill_roots(repo)) loras.push_back(r / "rudra_genfill.safetensors");
    if (training_workstation())   // GENFILL.ps1's v3 run
        loras.push_back("G:/rudra_training/research_genfill_v3_20261030/rudra_genfill.safetensors");
    return loras;
}

namespace {

// Where the Python and the base model come from (1 Nov 2026: macOS too, on Metal).
#ifdef _WIN32
const QString kSetupScript = QStringLiteral("scripts\\GENFILL.ps1 -Setup");
#else
const QString kSetupScript = QStringLiteral("scripts/genfill_setup.sh");
#endif

// 7 Nov 2026: the Python "Set up automatically" made (genfill_bootstrap), else the repository's
// .venv. Not a Python on the PATH: the setup would install the fill's packages into it, and one
// without PyTorch for the GPU is worse than the bootstrap's own.
fs::path default_python(const fs::path& repo) {
    std::error_code ec;
    if (const fs::path own = genfill_bootstrap_python(); fs::exists(own, ec)) return own;
#ifdef _WIN32
    return repo / ".venv" / "Scripts" / "python.exe";   // missing: the bootstrap's case
#else
    const fs::path venv = repo / ".venv" / "bin" / "python";
    return fs::exists(venv, ec) ? venv : fs::path("python3");
#endif
}

}  // namespace

fs::path genfill_bootstrap_root() {
    return pq(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation)) / "genfill";
}

fs::path genfill_bootstrap_python() {
#ifdef _WIN32
    return genfill_bootstrap_root() / "venv" / "Scripts" / "python.exe";
#else
    return genfill_bootstrap_root() / "venv" / "bin" / "python";
#endif
}

fs::path genfill_bootstrap_script(const fs::path& repo) {
#ifdef _WIN32
    const char* name = "genfill_bootstrap.ps1";
#else
    const char* name = "genfill_bootstrap.sh";
#endif
    std::error_code ec;
    for (const fs::path& p : {repo / name, repo / "scripts" / name})   // a package's python/, the repository
        if (!repo.empty() && fs::exists(p, ec)) return p;
    return {};
}

QStringList genfill_arguments(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out) {
    QStringList a{"-m", "rudra.genfill.run", "--master", master, "--out", out, "--base", qp(s.base), "--lora", qp(s.lora),
                  "--tau-high", QString::number(o.tau_high), "--tau-low", QString::number(o.tau_low),
                  "--seed", QString::number(o.seed), "--steps", QString::number(o.steps), "--json"};
    if (!o.prompt_highlight.trimmed().isEmpty()) a << "--prompt-highlight" << o.prompt_highlight.trimmed();
    if (!o.prompt_shadow.trimmed().isEmpty()) a << "--prompt-shadow" << o.prompt_shadow.trimmed();
    if (o.brightness != 0.0) a << "--brightness-offset" << QString::number(o.brightness, 'g', 6);
    if (o.prompt_strength != 1.0) a << "--prompt-strength" << QString::number(o.prompt_strength, 'g', 6);
    if (o.shape_gain != 1.0) a << "--shape-gain" << QString::number(o.shape_gain, 'g', 6);
    if (!o.region_json.isEmpty()) a << "--region" << o.region_json;
    if (o.fallback) a << "--fallback" << "spikes";
    if (o.focus_region) a << "--focus" << "areas";   // 8 Nov 2026: a box per group of areas (one area: as region)
    if (!o.glob.isEmpty()) a << "--glob" << o.glob;
    return a;
}

QStringList genfill_suggest_arguments(const GenFillSetup& s, const GenFillOptions& o, const QStringList& frames) {
    QStringList a{"-m", "rudra.genfill.prompts", "suggest"};
    a << frames;
    a << "--json" << "--cache" << qp(s.lora.parent_path() / "prompt_cache") << "--tau-high" << QString::number(o.tau_high)
      << "--tau-low" << QString::number(o.tau_low);
    return a;
}

std::optional<PromptSuggestions> parse_prompt_suggestions(const QByteArray& json, QString* why) {
    // The last JSON line: anything printed before it (a warning) is not ours.
    QByteArray line = json.trimmed();
    if (const qsizetype nl = line.lastIndexOf('\n'); nl >= 0) line = line.mid(nl + 1);
    const QJsonDocument d = QJsonDocument::fromJson(line);
    if (!d.isObject()) {
        if (why) *why = QStringLiteral("The prompt suggestion gave no answer.");
        return std::nullopt;
    }
    const QJsonObject o = d.object();
    if (!o.value("ok").toBool()) {
        if (why) *why = o.value("error").toString(QStringLiteral("The prompt suggestion failed."));
        return std::nullopt;
    }
    auto read = [](const QJsonArray& arr) {
        std::vector<PromptSuggestion> v;
        for (const auto& x : arr) {
            const QJsonObject e = x.toObject();
            v.push_back({e.value("id").toString(), e.value("label").toString(), e.value("text").toString(),
                         e.value("region").toString(), e.value("reason").toString(), e.value("cached").toBool()});
        }
        return v;
    };
    PromptSuggestions r;
    r.suggested = read(o.value("suggestions").toArray());
    r.presets = read(o.value("presets").toArray());
    r.clipped_fraction = o.value("clipped_fraction").toDouble();
    return r;
}

QString genfill_out_dir(const QString& master) {
    QString m = QDir::cleanPath(master);
    QFileInfo fi(m);
    if (fi.isFile()) m = fi.absolutePath() + "/" + fi.completeBaseName();
    QString out = m + "_aifill";
    for (int k = 2; QFileInfo::exists(out) && k < 1000; ++k) out = m + "_aifill_" + QString::number(k);
    return out;
}

QString genfill_file_sha256(const fs::path& file) {
    std::error_code ec;
    const auto size = fs::file_size(file, ec);
    if (ec) return {};
    const QFileInfo fi(qp(file));
    const QString key = QStringLiteral("genfill/sha256_cache/") +
                        QString::fromLatin1(QCryptographicHash::hash(qp(file).toUtf8(), QCryptographicHash::Sha1).toHex());
    const QString stamp = QString::number(qulonglong(size)) + QLatin1Char(':') + QString::number(fi.lastModified().toMSecsSinceEpoch());
    QSettings st;
    const QStringList cached = st.value(key).toStringList();
    if (cached.size() == 2 && cached[0] == stamp) return cached[1];
    QFile f(qp(file));
    if (!f.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash h(QCryptographicHash::Sha256);
    if (!h.addData(&f)) return {};
    const QString hex = QString::fromLatin1(h.result().toHex());
    st.setValue(key, QStringList{stamp, hex});
    return hex;
}

bool genfill_lora_is_v3(const fs::path& file, const QString& expected) {
    return !expected.isEmpty() && genfill_file_sha256(file) == expected;
}

QString genfill_ready_text(const GenFillSetup& s) {
    const QString where = QStringLiteral("Wan2.1-VACE-1.3B (%1) + %2").arg(from_path(s.base), from_path(s.lora));
    return s.lora_is_v3 ? QStringLiteral("Ready: %1\nAI fill v3 (SHA-256 verified)").arg(where)
                        : QStringLiteral("Ready, but this LoRA is NOT the published AI fill v3 (SHA-256 %1…); it may be an "
                                         "older version, and only v3 passed RUDRA's gates:\n%2")
                              .arg(QString::fromLatin1(kGenFillV3Sha256).left(12), where);
}

std::optional<GenFillSetup> genfill_setup(QString* why) {
    QSettings st;
    GenFillSetup s;
    s.repo = find_repo();
    const QString py = st.value("genfill/python").toString();
    s.python = !py.isEmpty() ? pq(py) : default_python(s.repo);
    // Found, not asked for (4 Nov 2026): the settings while their paths exist,
    // else every place a setup, a download or a Hugging Face cache puts them;
    // what is found is saved, so the next start looks once.
    s.base = saved_or("genfill/base", "transformer/config.json",
                      [&] { return first_existing(genfill_base_candidates(s.repo), "transformer/config.json"); });
    // The LoRA: with nothing saved, the first candidate that VERIFIES as v3, else the
    // first that exists (4 Nov 2026: the first that exists could be an old v1).
    const auto loras = genfill_lora_candidates(s.repo);
    for (const auto& l : loras) {
        std::error_code e;
        if (fs::exists(l, e) && genfill_lora_is_v3(l)) {
            s.v3_found = l;
            break;
        }
    }
    s.lora = saved_or("genfill/lora", nullptr, [&] { return !s.v3_found.empty() ? s.v3_found : first_existing(loras, nullptr); });
    auto refuse = [why](const QString& w) -> std::optional<GenFillSetup> {
        if (why) *why = w;
        return std::nullopt;
    };
    std::error_code ec;
    if (s.repo.empty() || !fs::exists(s.repo / "rudra" / "genfill" / "run.py", ec))
        return refuse(QStringLiteral("AI fill: the RUDRA repository (rudra/genfill) was not found; set it in Deliver > AI fill setup."));
    if (s.python.has_parent_path() && !fs::exists(s.python, ec))
        return refuse(QStringLiteral("AI fill: no Python at %1 (run %2).").arg(qp(s.python), kSetupScript));
    if (!fs::exists(s.base / "transformer" / "config.json", ec))
        return refuse(QStringLiteral("AI fill: no Wan2.1-VACE-1.3B model at %1 (run %2).").arg(qp(s.base), kSetupScript));
    if (!fs::exists(s.lora, ec))
        return refuse(QStringLiteral("AI fill: no LoRA at %1 (File > Model packages > AI fill v3 > Download).").arg(qp(s.lora)));
    if (why) why->clear();
    s.lora_is_v3 = genfill_lora_is_v3(s.lora);
    if (st.value("genfill/base").toString() != qp(s.base)) st.setValue("genfill/base", qp(s.base));
    if (st.value("genfill/lora").toString() != qp(s.lora)) st.setValue("genfill/lora", qp(s.lora));
    return s;
}

GenFillInstall genfill_install_plan() {
    QSettings st;
    GenFillInstall p;
    p.repo = find_repo();
    const QString py = st.value("genfill/python").toString();
    p.python = !py.isEmpty() ? pq(py) : default_python(p.repo);
    // A model already somewhere: the setup checks it and installs only the packages.
    std::error_code ec;
    const fs::path found = first_existing(genfill_base_candidates(p.repo), "transformer/config.json");
    p.base = fs::exists(found / "transformer" / "config.json", ec)
                 ? found
                 : pq(default_download_root()) / "Wan2.1-VACE-1.3B-diffusers";
    return p;
}

GenFillJob::GenFillJob(QObject* parent) : parent_(parent) {}

GenFillJob::~GenFillJob() {
    if (proc_) {
        proc_->disconnect();
        if (proc_->state() != QProcess::NotRunning) {
            // Asked first: it stops at the next step with every written frame whole.
            proc_->write("cancel\n");
            if (!proc_->waitForFinished(15000)) {
                proc_->kill();
                proc_->waitForFinished(3000);
            }
        }
        delete proc_;
    }
}

bool GenFillJob::running() const { return proc_ && proc_->state() != QProcess::NotRunning; }

bool GenFillJob::start(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out,
                       Event event_cb, Finished finished_cb, QString* why) {
    if (running()) {
        if (why) *why = QStringLiteral("An AI fill is already running.");
        return false;
    }
    delete proc_;
    proc_ = new QProcess(parent_);
    lines_.clear();
    stderr_tail_.clear();
    last_ = {};
    cancelled_ = false;
    out_ = out;
    on_event_ = std::move(event_cb);
    on_finished_ = std::move(finished_cb);
    auto env = QProcessEnvironment::systemEnvironment();
    const QString sep = QDir::listSeparator();
    const QString old = env.value("PYTHONPATH");
    env.insert("PYTHONPATH", qp(s.repo) + (old.isEmpty() ? QString() : sep + old));
    env.insert("HF_HUB_OFFLINE", "1");        // the model is local; never a download mid-render
    env.insert("PYTHONUNBUFFERED", "1");
    proc_->setProcessEnvironment(env);
    proc_->setWorkingDirectory(qp(s.repo));
    QObject::connect(proc_, &QProcess::readyReadStandardOutput, proc_, [this] { on_stdout(); });
    QObject::connect(proc_, &QProcess::readyReadStandardError, proc_, [this] {
        stderr_tail_ += QString::fromUtf8(proc_->readAllStandardError());
        if (stderr_tail_.size() > 4000) stderr_tail_ = stderr_tail_.right(4000);
    });
    QObject::connect(proc_, &QProcess::finished, proc_, [this](int code, QProcess::ExitStatus) { on_finished(code); });
    QObject::connect(proc_, &QProcess::errorOccurred, proc_, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) {
            last_ = {{"event", "error"}, {"ok", false}, {"error", "The AI fill could not start"},
                     {"detail", proc_->program() + ": " + proc_->errorString()}};
            if (on_finished_) {
                auto f = std::move(on_finished_);
                on_finished_ = nullptr;
                f(false, last_);
            }
        }
    });
    proc_->start(qp(s.python), genfill_arguments(s, o, master, out));
    return true;
}

void GenFillJob::cancel() {
    if (!running()) return;
    if (cancelled_) {   // asked twice: now
        proc_->kill();
        return;
    }
    cancelled_ = true;
    // A "cancel" line: the fill stops at its next step (frames are written whole,
    // as .part then renamed); killed if it has not stopped in 30 s.
    proc_->write("cancel\n");
    QProcess* p = proc_;
    QTimer::singleShot(30000, p, [p] {
        if (p->state() != QProcess::NotRunning) p->kill();
    });
}

void GenFillJob::on_stdout() {
    lines_.feed(proc_->readAllStandardOutput(), [this](const QJsonObject& o) {
        const QString ev = o.value("event").toString();
        if (ev == "done" || ev == "error") last_ = o;
        if (on_event_) on_event_(o);
    });
}

void GenFillJob::on_finished(int code) {
    on_stdout();
    if (!on_finished_) return;
    QJsonObject last = last_;
    bool ok = code == 0 && last.value("event").toString() == "done";
    if (cancelled_) {
        last = {{"event", "error"}, {"ok", false}, {"cancelled", true}, {"error", "Stopped"}, {"detail", ""}};
        ok = false;
    } else if (!ok && last.isEmpty()) {
        last = {{"event", "error"}, {"ok", false}, {"error", QStringLiteral("The AI fill ended (exit %1)").arg(code)},
                {"detail", stderr_tail_.trimmed().right(600)}};
    }
    auto f = std::move(on_finished_);
    on_finished_ = nullptr;
    f(ok, last);
}

}  // namespace rudra::app
