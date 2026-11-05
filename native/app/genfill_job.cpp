#include "genfill_job.hpp"
#include "model_downloads.hpp"
#include "paths.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QProcess>
#include <QProcessEnvironment>
#include <QSettings>
#include <QTimer>

#include <vector>

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QString qp(const fs::path& p) { return from_path(p); }
fs::path pq(const QString& s) { return to_path(s); }

// The repository: the saved one, else the first folder upwards of the app (or the
// working directory) with rudra/genfill/run.py.
fs::path find_repo() {
    QSettings st;
    const QString saved = st.value("genfill/repo").toString();
    if (!saved.isEmpty()) return pq(saved);
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

std::vector<fs::path> genfill_roots(const fs::path& repo) {
    std::vector<fs::path> r = {pq(QCoreApplication::applicationDirPath()) / "models" / "genfill"};
    if (!repo.empty()) r.push_back(repo / "models" / "genfill");
    if (const QString env = qEnvironmentVariable("RUDRA_GENFILL_DIR"); !env.isEmpty()) r.push_back(pq(env));
    return r;
}

// Where the Python and the base model come from (1 Nov 2026: macOS too, on Metal).
#ifdef _WIN32
const QString kSetupScript = QStringLiteral("scripts\\GENFILL.ps1 -Setup");
#else
const QString kSetupScript = QStringLiteral("scripts/genfill_setup.sh");
#endif

fs::path default_python(const fs::path& repo) {
#ifdef _WIN32
    return repo / ".venv" / "Scripts" / "python.exe";
#else
    const fs::path venv = repo / ".venv" / "bin" / "python";
    return fs::exists(venv) ? venv : fs::path("python3");
#endif
}

}  // namespace

QStringList genfill_arguments(const GenFillSetup& s, const GenFillOptions& o, const QString& master, const QString& out) {
    QStringList a{"-m", "rudra.genfill.run", "--master", master, "--out", out, "--base", qp(s.base), "--lora", qp(s.lora),
                  "--tau-high", QString::number(o.tau_high), "--tau-low", QString::number(o.tau_low),
                  "--seed", QString::number(o.seed), "--steps", QString::number(o.steps), "--json"};
    if (!o.prompt_highlight.trimmed().isEmpty()) a << "--prompt-highlight" << o.prompt_highlight.trimmed();
    if (!o.prompt_shadow.trimmed().isEmpty()) a << "--prompt-shadow" << o.prompt_shadow.trimmed();
    if (o.brightness != 0.0) a << "--brightness-offset" << QString::number(o.brightness, 'g', 6);
    if (!o.glob.isEmpty()) a << "--glob" << o.glob;
    return a;
}

QString genfill_out_dir(const QString& master) {
    QString m = QDir::cleanPath(master);
    QFileInfo fi(m);
    if (fi.isFile()) m = fi.absolutePath() + "/" + fi.completeBaseName();
    QString out = m + "_aifill";
    for (int k = 2; QFileInfo::exists(out) && k < 1000; ++k) out = m + "_aifill_" + QString::number(k);
    return out;
}

std::optional<GenFillSetup> genfill_setup(QString* why) {
    QSettings st;
    GenFillSetup s;
    s.repo = find_repo();
    const QString py = st.value("genfill/python").toString();
    s.python = !py.isEmpty() ? pq(py) : default_python(s.repo);
    std::vector<fs::path> bases, loras;
    for (const auto& r : genfill_roots(s.repo)) {
        bases.push_back(r / "Wan2.1-VACE-1.3B-diffusers");
        loras.push_back(r / "rudra_genfill.safetensors");
    }
    bases.push_back("G:/datasets/models/Wan2.1-VACE-1.3B-diffusers");   // the training workstation
    // Downloaded in the app (File > Model packages, 31 Oct 2026). Settings genfill/lora overrides it.
    loras.push_back(fs::path(default_download_root().toStdU16String()) / "rudra_genfill_v3" / "rudra_genfill.safetensors");
    const QString base = st.value("genfill/base").toString(), lora = st.value("genfill/lora").toString();
    s.base = !base.isEmpty() ? pq(base) : first_existing(bases, "transformer/config.json");
    s.lora = !lora.isEmpty() ? pq(lora) : first_existing(loras, nullptr);
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
    return s;
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
