#include "genfill_preview.hpp"
#include "genfill_swap.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QStatusBar>
#include <QTemporaryDir>
#include <QTimer>

#include <algorithm>
#include <functional>
#include <chrono>
#include <cmath>

#include "aces_engine.hpp"
#include "main_window.hpp"
#include "paths.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/deliver/exr.hpp"
#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {

// ---------------------------------------------------------------------------- the server

GenFillServer::GenFillServer(QObject* parent) : parent_(parent) {}

GenFillServer::~GenFillServer() {
    if (!proc_) return;
    proc_->disconnect();
    if (proc_->state() != QProcess::NotRunning) {
        if (busy_) proc_->write("cancel\n");
        proc_->write("quit\n");   // the request in hand stops at its next step, then the server ends
        if (!proc_->waitForFinished(15000)) {
            proc_->kill();
            proc_->waitForFinished(3000);
        }
    }
    delete proc_;
}

bool GenFillServer::ensure(const GenFillSetup& s, QString* why) {
    const bool same = s.python == setup_.python && s.repo == setup_.repo && s.base == setup_.base && s.lora == setup_.lora;
    if (proc_ && proc_->state() != QProcess::NotRunning && same) return true;
    if (proc_) {   // another setup (a LoRA downloaded, a path changed): a new server
        proc_->disconnect();
        if (proc_->state() != QProcess::NotRunning) {
            proc_->write("quit\n");
            if (!proc_->waitForFinished(5000)) proc_->kill();
        }
        proc_->deleteLater();
        proc_ = nullptr;
    }
    setup_ = s;
    loaded_ = false;
    load_seconds_ = 0;
    lines_.clear();
    stderr_tail_.clear();
    proc_ = new QProcess(parent_);
    auto env = QProcessEnvironment::systemEnvironment();
    const QString old = env.value("PYTHONPATH");
    env.insert("PYTHONPATH", from_path(s.repo) + (old.isEmpty() ? QString() : QDir::listSeparator() + old));
    env.insert("HF_HUB_OFFLINE", "1");   // the model is local, as for a render
    env.insert("PYTHONUNBUFFERED", "1");
    proc_->setProcessEnvironment(env);
    proc_->setWorkingDirectory(from_path(s.repo));
    QProcess* p = proc_;
    QObject::connect(p, &QProcess::readyReadStandardOutput, p, [this] { on_stdout(); });
    QObject::connect(p, &QProcess::readyReadStandardError, p, [this, p] {
        stderr_tail_ += QString::fromUtf8(p->readAllStandardError());
        if (stderr_tail_.size() > 4000) stderr_tail_ = stderr_tail_.right(4000);
    });
    QObject::connect(p, &QProcess::finished, p, [this](int code, QProcess::ExitStatus) {
        loaded_ = false;
        if (busy_)
            finish(false, {{"event", "error"}, {"ok", false}, {"error", "The AI fill preview stopped"},
                           {"detail", QStringLiteral("exit code %1. %2").arg(code).arg(stderr_tail_.right(600).trimmed())}});
    });
    QObject::connect(p, &QProcess::errorOccurred, p, [this, p](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart && busy_)
            finish(false, {{"event", "error"}, {"ok", false}, {"error", "The AI fill preview could not start"},
                           {"detail", p->program() + ": " + p->errorString()}});
    });
    p->start(from_path(s.python), {"-m", "rudra.genfill.run", "--serve", "--json", "--base", from_path(s.base), "--lora",
                                    from_path(s.lora)});
    if (!p->waitForStarted(10000)) {
        if (why) *why = QStringLiteral("AI fill preview: %1 did not start (%2).").arg(from_path(s.python), p->errorString());
        return false;
    }
    return true;
}

bool GenFillServer::request(const GenFillSetup& s, QJsonObject req, Event on_event, Finished on_done, QString* why) {
    if (busy_) {
        if (why) *why = QStringLiteral("An AI fill preview is running.");
        return false;
    }
    if (!ensure(s, why)) return false;
    current_id_ = next_id_++;
    req["id"] = current_id_;
    on_event_ = std::move(on_event);
    on_done_ = std::move(on_done);
    busy_ = true;
    // Queued by the server while the model still loads: it runs once it is ready.
    proc_->write(QJsonDocument(req).toJson(QJsonDocument::Compact) + '\n');
    return true;
}

void GenFillServer::cancel() {
    if (busy_ && proc_ && proc_->state() != QProcess::NotRunning) proc_->write("cancel\n");
}

void GenFillServer::finish(bool ok, const QJsonObject& last) {
    busy_ = false;
    current_id_ = 0;
    on_event_ = nullptr;
    if (on_done_) {
        auto f = std::move(on_done_);
        on_done_ = nullptr;
        f(ok, last);
    }
}

void GenFillServer::on_stdout() {
    lines_.feed(proc_->readAllStandardOutput(), [this](const QJsonObject& o) {
        const QString ev = o.value("event").toString();
        if (ev == "ready") {   // the model is loaded (once per server)
            loaded_ = true;
            load_seconds_ = o.value("load_seconds").toDouble();
            if (on_event_) on_event_(o);
            return;
        }
        if (!o.contains("id")) {   // the server's own: a failure before any request ran
            if (ev == "error" && busy_) finish(false, o);
            return;
        }
        if (o.value("id").toInt() != current_id_ || !busy_) return;   // an older request's late line
        if (ev == "done") finish(o.value("ok").toBool(), o);
        else if (ev == "error") finish(false, o);
        else if (on_event_) on_event_(o);
    });
}

// ---------------------------------------------------------------------------- the window

namespace {

// ACEScg (1.0 = 203 nits) to the viewer's precomposited convention, linear
// Rec.709 with 1.0 = 10 000 nits: as aces_master.cpp's aces_view_image.
NetworkLinearImage preview_view_image(const PlanarBuffer& acescg) {
    PlanarBuffer out(3, acescg.height(), acescg.width());
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Rec709);
    constexpr double k = 203.0 / 10000.0;
    float f[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) f[r][c] = float(m[std::size_t(r)][std::size_t(c)] * k);
    const float *ir = acescg.plane(0), *ig = acescg.plane(1), *ib = acescg.plane(2);
    float *orr = out.plane(0), *og = out.plane(1), *ob = out.plane(2);
    for (std::size_t i = 0; i < acescg.plane_size(); ++i) {
        const float r = ir[i], g = ig[i], b = ib[i];
        orr[i] = f[0][0] * r + f[0][1] * g + f[0][2] * b;
        og[i] = f[1][0] * r + f[1][1] * g + f[1][2] * b;
        ob[i] = f[2][0] * r + f[2][1] * g + f[2][2] * b;
    }
    return NetworkLinearImage(std::move(out));
}

// An ACEScg EXR's RGB (the preview's masters are written ACEScg, float).
std::optional<PlanarBuffer> read_acescg_rgb(const QString& path, QString* why) {
    auto e = read_exr(to_path(path));
    if (!e) {
        if (why) *why = QFileInfo(path).fileName() + ": " + QString::fromStdString(e.error().message);
        return std::nullopt;
    }
    const PlanarBuffer& px = e->pixels;
    if (px.channels() < 3) {
        if (why) *why = QFileInfo(path).fileName() + ": no RGB";
        return std::nullopt;
    }
    PlanarBuffer rgb(3, px.height(), px.width());
    for (int c = 0; c < 3; ++c) std::copy_n(px.plane(c), px.plane_size(), rgb.plane(c));
    return rgb;
}

QString preview_name(int index) { return QStringLiteral("preview.%1.exr").arg(index, 6, 10, QChar('0')); }

}  // namespace

// Why a finished preview may look like nothing (6 Nov 2026): the fill invents only where
// the master's SDR view is fully clipped (code 255) or crushed (0), and only above SDR
// white, which a display peak near 203 nits shows as plain white either way.
QString genfill_preview_hint(double changed, bool region_on, double display_nits) {
    if (changed <= 0.0)
        return QStringLiteral(". Nothing was invented on this frame: the AI fill acts only where the master's SDR "
                              "view is fully clipped (code 255) or crushed (0)%1, and this frame has none there. "
                              "Preview a frame with blown highlights.")
            .arg(region_on ? QStringLiteral(" inside the reconstruction region") : QString());
    if (display_nits < 600.0)
        return QStringLiteral(". The invented detail is above SDR white: at a %1-nit display peak it shows as white. "
                              "Raise Display peak (1,000 nits) or click Invented to see where it acts.")
            .arg(display_nits, 0, 'f', 0);
    return {};
}

namespace {

}  // namespace

bool MainWindow::genfill_previewing() const { return genfill_preview_busy_; }

void MainWindow::preview_genfill() {
    auto* st = findChild<QLabel*>("deliverAiPreviewStatus");
    auto say = [this, st](const QString& t) {
        if (st) {
            st->setText(t);
            st->show();
        }
        log(t);
    };
    if (genfill_preview_busy_) {   // a second click stops it
        if (genfill_preview_master_id_ > 0 && aces_engine_) aces_engine_->cancel_master(genfill_preview_master_id_);
        if (genfill_preview_job_) genfill_preview_job_->cancel();   // the analytic route's master
        if (genfill_server_) genfill_server_->cancel();
        say(QStringLiteral("AI fill preview: stopping…"));
        return;
    }
    // 7 Nov 2026: either route. The learned one renders its master in the engine; the analytic one
    // with the same MasterJob as Master EXR (ACEScg float), as its AI fill render does.
    const bool learned = deliver_aces() && aces_engine_;
    if (frames_.empty()) return say(QStringLiteral("AI fill preview: open a shot first."));
    if (!learned && needs_model()) return say(QStringLiteral("AI fill preview: the frames wait for the model package; load it first."));
    if (aces_delivering() || genfill_running() || batch_running() || mastering())
        return say(QStringLiteral("AI fill preview: a render or an AI fill is running; preview after it."));
    QString why;
    const auto setup = genfill_setup(&why);
    if (!setup) return say(why + QStringLiteral(" Setup… > Set up automatically does it for you."));
    halt_playback();
    // The suggested prompts for this frame, over the viewer while it fills (6 Nov 2026).
    if (prompt_suggest_frame_ != current_ || !prompt_suggestions_) run_prompt_suggestions(false, /*quiet=*/true);

    const int n = int(frames_.size());
    const bool movie = shot_video_.has_value() && !shot_sources_.empty();
    // The frames around the playhead: enough for the fill to show how it moves (9 by default).
    const int half = std::clamp(QSettings().value("genfill/previewHalf", 4).toInt(), 0, 16);
    const int at = std::clamp(current_, 0, n - 1);
    const int a = std::max(0, at - half), b = std::min(n - 1, at + half);
    genfill_preview_tmp_ = std::make_unique<QTemporaryDir>(QDir::temp().filePath("rudra-aifill-preview-XXXXXX"));
    if (!genfill_preview_tmp_->isValid()) return say(QStringLiteral("AI fill preview: no temporary folder."));
    const QString mdir = genfill_preview_tmp_->filePath("master"), fdir = genfill_preview_tmp_->filePath("fill");

    // The master as Deliver renders it (the trims, the source display), ACEScg float.
    const DeliverChoice c = deliver_effective();
    QJsonObject req{{"op", "master"},   {"exr", "float"},       {"exr_gamut", "acescg"}, {"out", mdir},
                    {"name", "preview"}, {"start", a},           {"into_existing", true}, {"create_dirs", true},
                    {"padding", 6},      {"separator", "."}};
    QJsonArray sources;
    if (movie) {
        sources.append(from_path(shot_sources_.front()));
        QJsonArray fr;
        for (int i = a; i <= b; ++i) fr.append(i);
        req["frames"] = fr;
    } else {
        for (int i = a; i <= b; ++i) sources.append(from_path(frames_[std::size_t(i)]));
        req["fps"] = c.fps;
    }
    req["sources"] = sources;
    if (const QJsonObject t = aces_trims_fields(aces_trims_.strength, aces_trims_.saturation, aces_trims_.peak_nits,
                                                 aces_trims_.exposure_ev, aces_trims_.contrast); !t.isEmpty())
        req["trims"] = t;
    if (aces_source_display_ != "bt1886") req["source_display"] = aces_source_display_;
    if (const QJsonObject r = aces_region_field(); !r.isEmpty()) req["reconstruction_region"] = r;

    genfill_preview_busy_ = true;
    genfill_preview_.reset();
    if (auto* btn = findChild<QPushButton*>("deliverAiPreview")) btn->setText(QStringLiteral("Stop preview"));
    const int frames = b - a + 1;
    say(QStringLiteral("AI fill preview: rendering the master of frames %1–%2…").arg(a + 1).arg(b + 1));
    const auto t0 = std::chrono::steady_clock::now();
    const QPointer<MainWindow> self(this);
    genfill_preview_master_id_ = -1;
    auto after_master = [self, setup = *setup, options = c.ai, mdir, fdir, a, b, at, frames, n, t0](const QJsonObject& o) {
            if (!self) return;
            self->genfill_preview_master_id_ = 0;
            auto* st2 = self->findChild<QLabel*>("deliverAiPreviewStatus");
            auto end = [self, st2](const QString& t) {
                self->genfill_preview_busy_ = false;
                if (auto* btn = self->findChild<QPushButton*>("deliverAiPreview")) btn->setText(QStringLiteral("Preview AI fill"));
                if (st2) st2->setText(t);
                self->log(t);
                self->refresh_enabled();
                self->sync_ai_prompt_bar();
            };
            if (!o.value("ok").toBool() || o.value("cancelled").toBool())
                return end(o.value("cancelled").toBool()
                               ? QStringLiteral("AI fill preview stopped.")
                               : QStringLiteral("AI fill preview: the master did not render: ") +
                                     (o.value("error").toString() + ' ' + o.value("detail").toString()).trimmed());
            const double master_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!self->genfill_server_) self->genfill_server_ = std::make_shared<GenFillServer>(self.data());
            QJsonObject fill{{"master", mdir},
                             {"out", fdir},
                             {"glob", "preview.*.exr"},
                             {"tau_high", options.tau_high},
                             {"tau_low", options.tau_low},
                             {"seed", options.seed},
                             {"steps", options.steps},
                             {"brightness_offset", options.brightness},
                             {"prompt_highlight", options.prompt_highlight.trimmed()},
                             {"prompt_shadow", options.prompt_shadow.trimmed()},
                             {"prompt_strength", options.prompt_strength},
                             {"shape_gain", options.shape_gain},
                             {"fallback", options.fallback ? "spikes" : "none"},
                             {"focus", options.focus_region ? "areas" : "frame"}};   // as genfill_arguments
            // RUDRA 1.1: the region, or null so the warm server drops an earlier request's.
            fill.insert("region", options.region_json.isEmpty()
                                      ? QJsonValue(QJsonValue::Null)
                                      : QJsonValue(QJsonDocument::fromJson(options.region_json.toUtf8()).object()));
            const bool cold = !self->genfill_server_->loaded();
            if (st2)
                st2->setText(cold ? QStringLiteral("AI fill preview: loading the fill model (once; a minute or two)…")
                                  : QStringLiteral("AI fill preview: filling %1 frame(s)…").arg(frames));
            const auto t1 = std::chrono::steady_clock::now();
            QString why2;
            const bool started = self->genfill_server_->request(
                setup, fill,
                [self, st2, frames](const QJsonObject& e) {
                    if (!self || !st2) return;
                    const QString ev = e.value("event").toString();
                    if (ev == "ready")
                        st2->setText(QStringLiteral("AI fill preview: model loaded on %1 in %2 s; filling %3 frame(s)…")
                                         .arg(e.value("device").toString())
                                         .arg(e.value("load_seconds").toDouble(), 0, 'f', 0)
                                         .arg(frames));
                    else if (ev == "embedding")   // a typed prompt not in the cache: umT5 loads once (6 Nov 2026)
                        st2->setText(QStringLiteral("AI fill preview: embedding %1 new prompt(s), once (a minute or more); "
                                                    "the presets are instant")
                                         .arg(e.value("prompts").toArray().size()));
                    else if (ev == "progress")
                        st2->setText(QStringLiteral("AI fill preview: step %1 / %2").arg(e.value("step").toInt()).arg(e.value("steps").toInt()));
                },
                [self, end, options, fdir, mdir, a, b, at, frames, n, master_s, t1, cold](bool ok, const QJsonObject& last) {
                    if (!self) return;
                    if (!ok || last.value("cancelled").toBool())
                        return end(last.value("cancelled").toBool()
                                       ? QStringLiteral("AI fill preview stopped.")
                                       : QStringLiteral("AI fill preview failed: ") +
                                             (last.value("error").toString() + ' ' + last.value("detail").toString()).trimmed());
                    const double fill_s = std::max(0.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() -
                                                            (cold ? self->genfill_server_->load_seconds() : 0.0));
                    QString w;
                    auto m = read_acescg_rgb(QDir(mdir).filePath(preview_name(at)), &w);
                    auto f = m ? read_acescg_rgb(QDir(fdir).filePath(preview_name(at)), &w) : std::nullopt;
                    if (!m || !f || m->width() != f->width() || m->height() != f->height())
                        return end(QStringLiteral("AI fill preview: the frames could not be read back. ") + w);
                    GenFillPreviewFrame p;
                    p.index = at;
                    p.options = options;
                    p.master = std::make_shared<const NetworkLinearImage>(preview_view_image(*m));
                    p.fill = std::make_shared<const NetworkLinearImage>(preview_view_image(*f));
                    // The invented pixels: those the fill changed (it copies every other one bit for
                    // bit), at SDR white over the master 3 stops down.
                    {
                        PlanarBuffer mk(3, m->height(), m->width());
                        const std::size_t np = m->plane_size();
                        std::size_t changed = 0;
                        for (std::size_t i = 0; i < np; ++i) {
                            const bool gen = m->plane(0)[i] != f->plane(0)[i] || m->plane(1)[i] != f->plane(1)[i] ||
                                             m->plane(2)[i] != f->plane(2)[i];
                            changed += gen ? 1 : 0;
                            for (int c = 0; c < 3; ++c) mk.plane(c)[i] = gen ? 1.0f : m->plane(c)[i] * 0.125f;
                        }
                        p.mask = std::make_shared<const NetworkLinearImage>(preview_view_image(mk));
                        p.changed = np ? double(changed) / double(np) : 0.0;
                    }
                    p.first = a;
                    p.last = b;
                    p.fill_dir = fdir;
                    self->genfill_preview_ = std::move(p);
                    self->fill_view_cache_.clear();   // a new preview: the viewer's AI fill reads it afresh
                    // The whole shot at this rate: its master and its fill, per frame (the model's load is once).
                    const double per_frame = (master_s + fill_s) / std::max(1, frames);
                    const double whole_min = per_frame * n / 60.0;
                    end(QStringLiteral("AI fill preview of frames %1–%2: master %3 s, fill %4 s%5 · %6 % of this frame "
                                       "invented · the whole shot (%7 frames) about %8 min")
                            .arg(a + 1)
                            .arg(b + 1)
                            .arg(master_s, 0, 'f', 0)
                            .arg(fill_s, 0, 'f', 0)
                            .arg(cold ? QStringLiteral(" (model load %1 s, once)").arg(self->genfill_server_->load_seconds(), 0, 'f', 0)
                                      : QString())
                            .arg(100.0 * self->genfill_preview_->changed, 0, 'f', 2)
                            .arg(n)
                            .arg(whole_min, 0, 'f', whole_min < 10 ? 1 : 0) +
                        genfill_preview_hint(self->genfill_preview_->changed, !options.region_json.isEmpty(),
                                             self->session_.display_nits()));
                    self->show_genfill_preview(QStringLiteral("fill"));
                },
                &why2);
            if (!started) end(why2);
        };
    if (learned) {
        const int id = aces_engine_->request(req, after_master, [self, frames](const QJsonObject& o) {
            if (!self) return;
            const QJsonObject p = o.value("progress").toObject();
            if (auto* st2 = self->findChild<QLabel*>("deliverAiPreviewStatus"))
                st2->setText(QStringLiteral("AI fill preview: master %1 / %2").arg(p.value("index").toInt()).arg(frames));
        });
        if (genfill_preview_master_id_ != 0) genfill_preview_master_id_ = id;   // still running (a refusal answers at once)
    } else {
        start_analytic_preview_master(mdir, a, b, std::move(after_master));
    }
    refresh_enabled();
    sync_ai_prompt_bar();
}

// The analytic route's preview master (7 Nov 2026): frames a..b as Master EXR renders them (the
// grade, the region, the container ACEScg, float), into the preview's master folder under the
// names the fill and the viewer read (preview.<index>.exr); `done` gets {"ok", "cancelled", "error"}
// on the UI thread, as the engine's master request answers.
void MainWindow::start_analytic_preview_master(const QString& mdir, int a, int b, std::function<void(const QJsonObject&)> done) {
    QDir().mkpath(mdir);
    std::vector<std::filesystem::path> targets, sources;
    for (int i = a; i <= b; ++i) {
        targets.push_back(to_path(QDir(mdir).filePath(preview_name(i))));
        sources.push_back(frames_[std::size_t(i)]);
    }
    MasterRequest q = master_request();
    q.container = "acescg";
    q.exr_half = false;
    q.mattes = false;
    q.brackets = false;
    PrepareMasterFrame prepare = master_prepare(sources, q);
    if (!prepare) {
        done(QJsonObject{{"ok", false}, {"error", QStringLiteral("this build has no still decode")}});
        return;
    }
    const QPointer<MainWindow> self(this);
    const int frames = b - a + 1;
    genfill_preview_job_ = std::make_unique<MasterJob>(frame_constants(), q, std::move(targets), std::move(prepare));
    genfill_preview_job_->start(
        [self, frames](std::size_t i) {
            QMetaObject::invokeMethod(qApp, [self, i, frames] {
                if (!self) return;
                if (auto* st = self->findChild<QLabel*>("deliverAiPreviewStatus"))
                    st->setText(QStringLiteral("AI fill preview: master %1 / %2").arg(int(i) + 1).arg(frames));
            });
        },
        [](const MasterProgress&) {},
        [self, done](const MasterOutcome& o) {
            QMetaObject::invokeMethod(qApp, [self, done, o] {
                if (!self) return;
                QJsonObject r{{"ok", !o.error && !o.cancelled && o.completed == o.total}, {"cancelled", o.cancelled}};
                if (o.error) r["error"] = QString::fromStdString(o.error->message);
                done(r);
            });
        });
}

void MainWindow::show_genfill_preview(const QString& which) {
    auto* st = findChild<QLabel*>("deliverAiPreviewStatus");
    if (!genfill_preview_) {
        if (st) st->setText(QStringLiteral("No AI fill preview yet: Preview AI fill fills the frames around the playhead."));
        return;
    }
    // The viewer moved on: back to the previewed frame; the picture asked for goes up when it
    // lands (apply_fill_view runs on every frame shown), with no second click (6 Nov 2026).
    const bool away = current_ != genfill_preview_->index;
    if (away) step_to(genfill_preview_->index);
    genfill_preview_show_ = which;
    const auto& p = *genfill_preview_;
    if (which == "fill") {
        // The AI fill: the viewer's AI fill switch on, so it stays on this frame and the
        // previewed ones around it whatever the route renders next (6 Nov 2026).
        genfill_preview_pinned_ = false;
        if (auto* t = findChild<QPushButton*>("vtAiFill"); t && !t->isChecked()) t->setChecked(true);   // -> set_fill_view
        else set_fill_view(true);
    } else {
        genfill_preview_pinned_ = true;
        if (!away) apply_fill_view();
    }
    for (const char* id : {"deliverAiShowFill", "deliverAiShowMaster", "deliverAiShowMask"})
        if (auto* b = findChild<QPushButton*>(id)) b->setChecked(QString::fromLatin1(id).endsWith(
                                                         which == "mask" ? "Mask" : which == "master" ? "Master" : "Fill"));
    statusBar()->showMessage(which == "master" ? QStringLiteral("Viewer: the master (no AI fill), frame %1").arg(p.index + 1)
                             : which == "mask" ? QStringLiteral("Viewer: the pixels the AI fill invented (white), frame %1").arg(p.index + 1)
                                               : QStringLiteral("Viewer: the AI fill preview, frame %1").arg(p.index + 1),
                             8000);
}

// ---- the viewer's AI fill (4 Nov 2026) ----
namespace {
// Box-averaged down to `side` (the playback size), so a shot's fills fit in memory and play in real time.
NetworkLinearImage fit_side(const NetworkLinearImage& img, int side) {
    const int w = img.width(), h = img.height();
    const int k = side > 0 ? std::max(1, (std::max(w, h) + side - 1) / side) : 1;
    if (k == 1) return img;
    const int ow = w / k, oh = h / k;
    PlanarBuffer out(3, oh, ow);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                double s = 0;
                for (int dy = 0; dy < k; ++dy)
                    for (int dx = 0; dx < k; ++dx) s += img.buffer().at(c, y * k + dy, x * k + dx);
                out.at(c, y, x) = float(s / (k * k));
            }
    return NetworkLinearImage(std::move(out));
}
}  // namespace

std::shared_ptr<const NetworkLinearImage> MainWindow::fill_view_frame(int index) {
    // The source: the preview's frames, else the last finished fill when it covers the shot.
    const QString key = (genfill_preview_ ? genfill_preview_->fill_dir : QString()) + "|" + genfill_last_out_ + "|" + genfill_last_master_ + "|" +
                        QString::number(play_state_->side.load()) + "|" + QString::number(aces_shown_ >= 0 && aces_model_img_ ? aces_model_img_->width() : 0);
    if (key != fill_view_key_) {
        fill_view_cache_.clear();
        fill_view_files_.clear();
        fill_view_key_ = key;
        // The fill folder's frames, or, once moved into the master in place, the master's.
        fill_view_files_ = genfill_filled_frames(genfill_last_out_, genfill_last_master_);
    }
    if (auto it = fill_view_cache_.find(index); it != fill_view_cache_.end()) return it->second;
    QString path;
    if (genfill_preview_ && index >= genfill_preview_->first && index <= genfill_preview_->last && !genfill_preview_->fill_dir.isEmpty())
        path = QDir(genfill_preview_->fill_dir).filePath(preview_name(index));
    else if (fill_view_files_.size() == int(frames_.size()) && index >= 0 && index < fill_view_files_.size())
        path = fill_view_files_[index];
    if (path.isEmpty() || !QFileInfo::exists(path)) return nullptr;
    QString why;
    auto rgb = read_acescg_rgb(path, &why);
    if (!rgb) return nullptr;
    // The size of what the viewer shows for this frame: the playback size while playing, else full.
    const int side = aces_rt_playing_ ? play_state_->side.load() : 0;
    auto img = std::make_shared<const NetworkLinearImage>(fit_side(preview_view_image(*rgb), side));
    fill_view_cache_[index] = img;
    return img;
}

bool MainWindow::apply_fill_view(bool quiet) {
#ifdef RUDRA_APP_VIEWER
    // The base the Changes layer compares with: the plain ACES inverse when it is this size.
    auto show = [this](const NetworkLinearImage& img, const NetworkLinearImage& fallback_base) {
        if (!viewer_) return;
        const bool base_ok = aces_base_img_ && aces_base_img_->width() == img.width() && aces_base_img_->height() == img.height();
        const bool codes_ok = aces_codes_.size() == std::size_t(img.width()) * std::size_t(img.height());
        viewer_->set_composited(img, base_ok ? *aces_base_img_ : fallback_base, codes_ok ? aces_codes_ : std::vector<float>{});
        if (aces_sdr_img_ && aces_sdr_img_->width() == img.width() && aces_sdr_img_->height() == img.height())
            viewer_->set_source_sdr(*aces_sdr_img_);
    };
#endif
    if (frames_.empty()) return false;
    // 6 Nov 2026: the preview's Master / Invented picture, pinned to the previewed frame. It was
    // shown once and the next frame the route rendered (any slider, the learned frame landing)
    // replaced it, so the preview looked as if it had done nothing.
    if (genfill_preview_pinned_ && genfill_preview_) {
        if (current_ != genfill_preview_->index) {
            genfill_preview_pinned_ = false;
        } else if (genfill_preview_show_ != "fill") {
            const auto& p = *genfill_preview_;
            const auto img = genfill_preview_show_ == "master" ? p.master : p.mask;
#ifdef RUDRA_APP_VIEWER
            if (img) show(*img, *p.master);
#endif
            return img != nullptr;
        }
    }
    if (!fill_view_on_) return false;
    // Any route (6 Nov 2026): the fill is of the learned ACES master, but the analytic route's
    // viewer showed nothing at all with the AI fill on.
    const auto img = fill_view_frame(current_);
    if (!img) {
        if (!quiet)
            statusBar()->showMessage(QStringLiteral("AI fill: no fill for frame %1 (Preview AI fill, or render with AI fill)").arg(current_ + 1), 3000);
        return false;
    }
#ifdef RUDRA_APP_VIEWER
    show(*img, *img);
#endif
    return true;
}

void MainWindow::set_fill_view(bool on) {
    if (fill_view_on_ == on) {
        if (on) apply_fill_view();
        return;
    }
    fill_view_on_ = on;
    sync_ai_prompt_bar();
    if (on) {
        genfill_preview_pinned_ = false;
        if (!apply_fill_view()) return;
        statusBar()->showMessage(QStringLiteral("Viewer: the AI fill (Changes shows what it invented)"), 4000);
    } else {
        genfill_preview_pinned_ = false;
        if (!frames_.empty()) step_to(current_);   // back to the frame without the fill, on whichever route
    }
}

}  // namespace rudra::app

