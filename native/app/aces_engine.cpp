#include "aces_engine.hpp"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QObject>
#include <QProcess>
#include <QSettings>
#include <QStringList>

#include <chrono>
#include <system_error>

#include "aces_master.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {
QString qp(const fs::path& p) { return QString::fromStdString(p.string()); }
fs::path pq(const QString& s) { return fs::path(s.toStdString()); }
}  // namespace

AcesEngine::AcesEngine(QObject* parent) : parent_(parent), alive_(new QObject) {}

AcesEngine::~AcesEngine() {
    // Nothing reaches a listener or a caller from here on: they may be half
    // destroyed already (a dialog that owns this engine).
    listeners_.clear();
    waiting_.clear();
    delete alive_;   // drops the owners' destroyed() connections
    if (proc_) {
        stopping_ = true;
        QObject::disconnect(proc_, nullptr, nullptr, nullptr);
        if (proc_->state() != QProcess::NotRunning) {
            proc_->write("{\"op\":\"quit\"}\n");
            proc_->closeWriteChannel();
            if (!proc_->waitForFinished(2000)) {
                proc_->kill();
                proc_->waitForFinished(2000);
            }
        }
        delete proc_;
    }
}

QString AcesEngine::status_text() const {
    switch (state_) {
        case State::Stopped: return QStringLiteral("ACES model not loaded");
        case State::Loading: return QStringLiteral("Loading the ACES model (%1, %2)…").arg(QString::fromStdString(config_.device),
                                                                                      QString::fromStdString(config_.runtime));
        case State::Ready:
            return QStringLiteral("ACES model ready on %1 (%2), loaded in %3 s")
                .arg(ready_device_, ready_runtime_)
                .arg(load_ms_ / 1000.0, 0, 'f', 1);
        case State::Failed: return QStringLiteral("ACES model failed: %1").arg(why_);
    }
    return {};
}

void AcesEngine::set_state(State s, const QString& why) {
    state_ = s;
    why_ = why;
    // Copy: a listener may add or remove listeners.
    const auto ls = listeners_;
    for (const auto& [id, l] : ls) l.second();
}

void AcesEngine::listen(QObject* owner, std::function<void()> on_change) {
    const int id = next_listener_++;
    listeners_[id] = {owner, std::move(on_change)};
    QObject::connect(owner, &QObject::destroyed, alive_, [this, id] { listeners_.erase(id); });
}

void AcesEngine::start(const Config& c) {
    if (proc_ && proc_->state() != QProcess::NotRunning && c == config_ && state_ != State::Failed) return;
    stop();
    config_ = c;
    std::error_code ec;
    if (c.cli.empty() || !fs::is_regular_file(c.cli, ec)) {
        set_state(State::Failed, "no ACES v2 rudra-native");
        return;
    }
    if (c.package.empty() || !fs::is_regular_file(c.package / "manifest.json", ec)) {
        set_state(State::Failed, "no ACES v2 model package");
        return;
    }
    proc_ = new QProcess(parent_);
    proc_->setProcessEnvironment(aces_child_environment());
    proc_->setProcessChannelMode(QProcess::SeparateChannels);
    QProcess* p = proc_;
    QObject::connect(p, &QProcess::readyReadStandardOutput, p, [this, p] {
        if (p == proc_) on_stdout();
    });
    QObject::connect(p, &QProcess::finished, p, [this, p](int, QProcess::ExitStatus) {
        if (p == proc_) on_finished();
    });
    QObject::connect(p, &QProcess::errorOccurred, p, [this, p](QProcess::ProcessError e) {
        if (p == proc_ && e == QProcess::FailedToStart) set_state(State::Failed, "the engine did not start: " + p->errorString());
    });
    stopping_ = false;
    pending_.clear();
    set_state(State::Loading);
    proc_->start(qp(c.cli), {"aces-v2-serve", qp(c.package), "--runtime", QString::fromStdString(c.runtime), "--device",
                             QString::fromStdString(c.device), "--ocio-threads", QString::number(c.ocio_threads)});
}

void AcesEngine::stop() {
    if (!proc_) return;
    stopping_ = true;
    QProcess* p = proc_;
    proc_ = nullptr;
    QObject::disconnect(p, nullptr, nullptr, nullptr);   // nothing of it reaches this engine again
    if (p->state() != QProcess::NotRunning) {
        p->write("{\"op\":\"quit\"}\n");
        p->closeWriteChannel();
        if (!p->waitForFinished(1500)) p->kill();
    }
    p->deleteLater();
    const auto waiting = std::move(waiting_);
    waiting_.clear();
    for (const auto& [id, cb] : waiting)
        if (cb.first) cb.first(QJsonObject{{"id", id}, {"ok", false}, {"error", "The ACES engine was stopped."}});
    set_state(State::Stopped);
}

int AcesEngine::request(QJsonObject req, Reply on_reply, Reply on_progress) {
    const int id = next_id_++;
    if (state_ != State::Ready || !proc_) {
        if (on_reply) on_reply(QJsonObject{{"id", id}, {"ok", false}, {"error", status_text()}});
        return id;
    }
    req["id"] = id;
    waiting_[id] = {std::move(on_reply), std::move(on_progress)};
    proc_->write(QJsonDocument(req).toJson(QJsonDocument::Compact) + "\n");
    return id;
}

void AcesEngine::cancel_master() {
    if (proc_ && state_ == State::Ready) proc_->write("{\"op\":\"cancel\"}\n");
}

void AcesEngine::on_stdout() {
    pending_ += proc_->readAllStandardOutput().toStdString();
    std::size_t nl;
    while ((nl = pending_.find('\n')) != std::string::npos) {
        std::string line = pending_.substr(0, nl);
        pending_.erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const QJsonDocument d = QJsonDocument::fromJson(QByteArray::fromStdString(line));
        if (!d.isObject()) continue;   // anything else the engine prints is not ours
        const QJsonObject o = d.object();
        const QString event = o.value("event").toString();
        if (event == "ready") {
            load_ms_ = o.value("load_ms").toDouble();
            ready_device_ = o.value("device").toString();
            ready_runtime_ = o.value("runtime").toString();
            set_state(State::Ready);
            continue;
        }
        if (event == "error") {
            set_state(State::Failed, o.value("error").toString() + (o.value("detail").toString().isEmpty() ? QString()
                                                                    : " (" + o.value("detail").toString() + ")"));
            continue;
        }
        const int id = o.value("id").toInt(-1);
        const auto it = waiting_.find(id);
        if (it == waiting_.end()) continue;
        if (o.contains("progress")) {
            if (it->second.second) it->second.second(o.value("progress").toObject());
            continue;
        }
        const Reply cb = it->second.first;
        waiting_.erase(it);
        if (cb) cb(o);
    }
}

void AcesEngine::on_finished() {
    if (stopping_) return;
    QString err = QString::fromUtf8(proc_->readAllStandardError()).trimmed().section('\n', -1);
    const int code = proc_->exitCode();
    const auto waiting = std::move(waiting_);
    waiting_.clear();
    for (const auto& [id, cb] : waiting)
        if (cb.first) cb.first(QJsonObject{{"id", id}, {"ok", false}, {"error", "The ACES engine stopped."}});
    if (state_ != State::Failed)
        set_state(State::Failed, QStringLiteral("the engine stopped (exit %1)%2").arg(code).arg(err.isEmpty() ? QString() : ": " + err));
}

std::optional<AcesEngine::Config> aces_engine_config(const std::vector<fs::path>& roots) {
    QSettings st;
    AcesEngine::Config c;
    std::error_code ec;
    c.cli = default_aces_cli();
    if (c.cli.empty()) {
        const fs::path saved = pq(st.value("aces/cli").toString());
        if (!saved.empty() && fs::is_regular_file(saved, ec)) c.cli = saved;
    }
    std::vector<fs::path> all = roots;
    for (const auto& r : st.value("aces/roots").toStringList()) all.push_back(pq(r));
    const auto packages = find_aces_packages(all);
    const fs::path saved_pkg = pq(st.value("aces/package").toString());
    for (const auto& p : packages)
        if (!saved_pkg.empty() && fs::equivalent(p.root, saved_pkg, ec)) c.package = p.root;
    if (c.package.empty() && !saved_pkg.empty() && fs::is_regular_file(saved_pkg / "manifest.json", ec) &&
        !find_aces_packages({saved_pkg}).empty())
        c.package = saved_pkg;
    if (c.package.empty() && !packages.empty()) c.package = packages.front().root;
    const QString key = st.value("aces/target", "cuda").toString();
    for (const auto& t : aces_run_targets())
        if (QString::fromStdString(t.key) == key) {
            c.runtime = t.runtime;
            c.device = t.device;
        }
    if (c.cli.empty() || c.package.empty()) return std::nullopt;
    return c;
}

}  // namespace rudra::app
