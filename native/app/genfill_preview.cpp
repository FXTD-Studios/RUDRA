#include "genfill_preview.hpp"
#include "child_process.hpp"
#include "genfill_swap.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QRegularExpression>
#include <QStatusBar>
#include <QStyle>
#include <QTemporaryDir>
#include <QTimer>

#include <algorithm>
#include <functional>
#include <chrono>
#include <cmath>
#include <fstream>
#include <optional>

#include "aces_engine.hpp"
#include "main_window.hpp"
#include "paths.hpp"
#include "system_check.hpp"   // genfill_missing_text (1.2.5)
#include "rudra/core/baseline.hpp"
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
    const bool same = s.python == setup_.python && s.repo == setup_.repo && s.base == setup_.base && s.lora == setup_.lora &&
                      s.rerender == setup_.rerender;
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
    QStringList args{"-m", "rudra.genfill.run", "--serve", "--json", "--base", from_path(s.base), "--lora", from_path(s.lora)};
    // v4, region re-render: the settings a request cannot carry, as it was tested (12 Nov 2026).
    if (s.rerender) args << "--feather" << "8" << "--brightness-model" << "none";
    end_with_rudra(p);   // 1.3: the preview's model ends with RUDRA, crash or not
    p->start(from_path(s.python), args);
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

// Raw planar float32 (R, G, B) as aces-v2-serve's monitor op writes it.
std::optional<PlanarBuffer> read_planes(const QString& path, int w, int h) {
    PlanarBuffer b(3, h, w);
    std::ifstream in(to_path(path), std::ios::binary);
    for (int c = 0; c < 3 && in; ++c)
        in.read(reinterpret_cast<char*>(b.plane(c)), std::streamsize(b.plane_size() * sizeof(float)));
    if (!in) return std::nullopt;
    return b;
}

}  // namespace

// 9 Nov 2026: on the learned route the viewer shows the live frame through the monitor (ACES 2.0
// SDR 100 or HDR 1000, display light); the preview's master and fill, ACEScg, must go the same way
// or the fill looks like a different picture (an open door as a cyan wall). The engine applies it.
void MainWindow::monitor_preview_planes(const QString& exr, int w, int h,
                                        std::function<void(std::optional<PlanarBuffer>, QString)> done) {
    if (!aces_engine_ || aces_monitor_ == "scene") {
        QString why;
        auto rgb = read_acescg_rgb(exr, &why);
        return done(std::move(rgb), why);
    }
    const QString out = exr + ".monitor.f32";
    const QPointer<MainWindow> self(this);
    aces_engine_->request(QJsonObject{{"op", "monitor"}, {"exr", exr}, {"display", aces_monitor_}, {"out", out}},
                          [self, out, w, h, done = std::move(done)](const QJsonObject& o) {
                              if (!self) return;
                              if (!o.value("ok").toBool())
                                  return done(std::nullopt, (o.value("error").toString() + ' ' + o.value("detail").toString()).trimmed());
                              auto p = read_planes(out, w, h);
                              QFile::remove(out);
                              done(std::move(p), p ? QString() : QStringLiteral("the monitored frame could not be read back"));
                          });
}

// Why a finished preview may look like nothing (6 Nov 2026): the fill invents only where
// the master's SDR view is fully clipped (code 255) or crushed (0), and only above SDR
// white, which a display peak near 203 nits shows as plain white either way.
QString genfill_preview_hint(double changed, bool region_on, double display_nits) {
    if (changed <= 0.0)
        return QStringLiteral(". Nothing was invented on this frame: the AI fill acts only on areas of the master's "
                              "SDR view that are fully clipped (code 255) or crushed (0)%1, and this frame has none "
                              "big enough. \"Where it fills\" shows them.")
            .arg(region_on ? QStringLiteral(" inside the reconstruction region") : QString());
    if (display_nits < 600.0)
        return QStringLiteral(". The invented detail is above SDR white: at a %1-nit display peak it shows as white. "
                              "Raise Display peak (1,000 nits) or click Invented to see where it acts.")
            .arg(display_nits, 0, 'f', 0);
    return {};
}

double genfill_progress_share(double lo, double hi, int window, int windows, int step, int steps) {
    windows = std::max(1, windows);
    steps = std::max(1, steps);
    const double done = (std::clamp(window, 1, windows) - 1 + std::clamp(double(step) / steps, 0.0, 1.0)) / windows;
    return lo + (hi - lo) * done;
}

QString ai_fill_frame_text(const AiFillRegion& r, int tau_high, bool v4) {
    if (v4) {   // 1.2.6: v4.3 redraws its region, the codes inside it not kept
        // 1.2.7: what it really redraws; codes >= 240 it leaves to the master (saturated, or kept detail) said too
        const double kept = std::max(0.0, r.seen_pct - r.filled_pct);
        const QString name = genfill_region_name();   // 1.3: v5, or v4.3 where only it is installed
        if (r.empty())
            return r.seen_pct < 0.05 ? QStringLiteral("This frame: nothing for %1 to redraw (nothing at code %2 or above).").arg(name).arg(tau_high)
                                     : QStringLiteral("This frame: nothing for %1 to redraw. Its bright area (%2 %) is saturated "
                                                      "colour, which stays the master.")
                                           .arg(name)
                                           .arg(r.seen_pct, 0, 'f', 1);
        QString t = QStringLiteral("This frame: %1 redraws %2 % (codes %3 and up)")
                        .arg(name)
                        .arg(r.filled_pct, 0, 'f', r.filled_pct < 1.0 ? 2 : 1)
                        .arg(tau_high);
        if (kept >= 0.05) t += QStringLiteral(" \u00b7 %1 % stays the master").arg(kept, 0, 'f', 1);
        return t;
    }
    if (r.empty() && r.seen_pct < 0.05) return QStringLiteral("This frame: nothing to fill (no clipped highlights).");
    if (r.empty())
        return QStringLiteral("This frame: nothing to fill. Its bright area is near white (codes %1–254), "
                              "not clipped, so it keeps its picture.")
            .arg(std::min(tau_high, 254));
    return QStringLiteral("This frame: fills %1 % · the model sees %2 %")
        .arg(r.filled_pct, 0, 'f', r.filled_pct < 1.0 ? 2 : 1)
        .arg(r.seen_pct, 0, 'f', 1);
}

QString ai_fill_frame_tip(const AiFillRegion& r, int tau_high, bool v4) {
    if (v4)
        return QStringLiteral("AI fill region (experimental; v5, or v4.3 where only it is installed) draws the near-white pixels at code %1 or above again "
                              "from the source: its detail from the model, its colour and level led by the source. Unlike v3 "
                              "the SDR codes there are not kept. A saturated highlight (a sunset, a flame) stays the master, "
                              "and so, with Keep detail, do thin and small bright parts. \"Where it fills\" shows what is "
                              "redrawn in solid orange, what stays the master tinted.")
            .arg(tau_high);
    QString t = QStringLiteral(
                    "AI fill v3 changes only fully clipped areas (code 255) of about 100 pixels or more at 1080p: "
                    "solid orange under \"Where it fills\". Its model also sees everything at code %1 or above "
                    "(tinted), as context, and leaves it as it is: the source still holds a picture there.")
                    .arg(std::min(tau_high, 256));
    if (r.specks_pct > 0.0)
        t += QStringLiteral("\n\n%1 % of the frame is clipped in scattered specks too small to fill; "
                            "they keep the master.")
                 .arg(r.specks_pct, 0, 'f', 2);
    if (r.empty() && r.seen_pct >= 0.05)
        t += QStringLiteral("\n\nTo replace a near-white area (a hazy sky in a door), AI fill v3 is the wrong tool: "
                            "it never changes what the source shows. Grade it, or replace it in compositing.");
    return t;
}

namespace {

}  // namespace

bool MainWindow::genfill_previewing() const { return genfill_preview_busy_; }

std::optional<AiFillRegion> MainWindow::ai_fill_region_now(std::shared_ptr<const SdrImage>* sdr_out) {
    std::shared_ptr<const SdrImage> sdr;
    // The learned route's frame on screen, else the analytic route's.
    if (aces_shown_ == current_ && aces_sdr_img_ && aces_model_img_ && aces_sdr_img_->width() == aces_model_img_->width())
        sdr = aces_sdr_img_;
    else if (current_frame_ && current_frame_->index == current_)
        sdr = std::shared_ptr<const SdrImage>(current_frame_, &current_frame_->sdr);
    if (!sdr || sdr->width() <= 0) return std::nullopt;
    if (sdr_out) *sdr_out = sdr;
    const DeliverChoice c = deliver_effective();
    // v4.3 region re-render: codes >= 240, as the fill weighs them (1.2.7: the near-white guard, and
    // with Keep detail the thin-and-small rule; 1.2.6 showed the codes alone), no shadows.
    const bool v4 = c.ai.rerender;
    const int hi = v4 ? kAiFillV4TauHigh : c.ai.tau_high, lo = v4 ? -1 : c.ai.tau_low;
    const AiFrameInfoKey key{sdr.get(), hi + (v4 ? 1000 + (c.ai.rerender_keep_detail ? 1000 : 0) : 0), lo, current_};
    if (!ai_frame_info_ || !(key == ai_frame_info_key_)) {
        ai_frame_info_ = v4 ? ai_fill_region_v4(*sdr, c.ai.rerender_keep_detail) : ai_fill_region(*sdr, hi, lo);
        ai_frame_info_key_ = key;
    }
    return ai_frame_info_;
}

void MainWindow::sync_ai_frame_info() {
    auto* lab = findChild<QLabel*>("deliverAiFrameInfo");
    auto* btn = findChild<QPushButton*>("deliverAiShowRegion");
    const DeliverChoice c = deliver_effective();
    const bool on = ai_generation_;   // 1.2.6: v3 and v4.3 region, each with its own rule (21 Nov 2026: Reconstruct's switch)
    const bool v4 = c.ai.rerender;
    const int tau = v4 ? kAiFillV4TauHigh : c.ai.tau_high;
    if (btn) {
        btn->setVisible(on);
        if (!on && ai_region_view_) set_ai_region_view(false);
    }
    if (!lab) return;
    if (!on) {
        lab->hide();
        return;
    }
    if (playing()) return;   // measured when it stops
    const auto r = ai_fill_region_now();
    lab->setText(r ? ai_fill_frame_text(*r, tau, v4) : QStringLiteral("This frame: not shown yet."));
    lab->setToolTip(r ? ai_fill_frame_tip(*r, tau, v4) : QString());
    lab->setProperty("state", r && r->empty() ? "warn" : "");
    lab->style()->unpolish(lab);
    lab->style()->polish(lab);
    lab->show();
    if (ai_region_view_) apply_fill_view(/*quiet=*/true);
}

void MainWindow::set_ai_progress(double share, const QString& what) {
    ai_progress_ = share < 0 ? -1.0 : std::clamp(share, 0.0, 1.0);
    for (const char* id : {"deliverAiProgress"}) {   // 21 Nov 2026: on the AI generation card only
        auto* bar = findChild<QProgressBar*>(id);
        if (!bar) continue;
        if (ai_progress_ < 0) {
            bar->setRange(0, 0);   // busy: a model loading, a prompt embedding
        } else {
            bar->setRange(0, 1000);
            bar->setValue(int(std::lround(ai_progress_ * 1000.0)));
        }
        bar->setFormat(ai_progress_ < 0 ? what : QStringLiteral("%p % \u00b7 ") + what);
        bar->setToolTip(what);
        bar->show();
    }
}

void MainWindow::ai_progress_done() {
    ai_progress_ = -2.0;
    for (const char* id : {"deliverAiProgress"})
        if (auto* bar = findChild<QProgressBar*>(id)) bar->hide();
}

void MainWindow::set_ai_region_view(bool on) {
    if (auto* b = findChild<QPushButton*>("deliverAiShowRegion"); b && b->isChecked() != on) b->setChecked(on);
    if (ai_region_view_ == on) return;
    ai_region_view_ = on;
    if (on) {
        genfill_preview_pinned_ = false;
        if (apply_fill_view())
            statusBar()->showMessage(QStringLiteral("Viewer: where the AI fill acts on this frame: solid orange it fills, "
                                                    "tinted its model only sees"),
                                     6000);
    } else if (!frames_.empty()) {
        step_to(current_);   // back to the picture
    }
}

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
    const bool rerender = deliver_effective().ai.rerender;   // v4: its own LoRA, its own server
    const auto setup = genfill_setup_for(deliver_effective().ai, &why);
    if (!setup) {
        if (!rerender) open_system_check(false);   // 1.2.5: where it is set up
        return say(rerender ? why : genfill_missing_text(why));
    }
    halt_playback();
    // The suggested prompts for this frame, on the AI generation card while it fills (6 Nov 2026).
    if (prompt_suggest_frame_ != current_ || !prompt_suggestions_) run_prompt_suggestions(false, /*quiet=*/true);

    const int n = int(frames_.size());
    const bool movie = shot_video_.has_value() && !shot_sources_.empty();
    // 8 Nov 2026: the frame at the playhead only, by default: a quick look at the fill in the viewer;
    // the render fills the whole shot (in windows of 33, which also keeps it steady). genfill/previewHalf
    // > 0 previews that many frames each side, to judge how the fill moves.
    // 12 Nov 2026: v4 previews a whole window of 33 frames round the playhead, as it was gated. It is a
    // video model: of 1 or 5 frames it drew a woven grain where the 97-frame reference has clouds
    // (measured on the kept preview: same level, 2.49 against 2.50 x white, no picture).
    const int half = std::clamp(QSettings().value("genfill/previewHalf", 0).toInt(), 0, 16);
    const int at = std::clamp(current_, 0, n - 1);
    int a = std::max(0, at - half), b = std::min(n - 1, at + half);
    if (rerender) {
        // 17 frames by default (a trial, 12 Nov 2026: half the wait of the gated 33); genfill/previewV4Frames
        // sets another count of 4k+1 from 5 to 33.
        int count = std::clamp(QSettings().value("genfill/previewV4Frames", 17).toInt(), 5, 33);
        count -= (count - 1) % 4;
        a = std::clamp(at - count / 2, 0, std::max(0, n - count));
        b = std::min(n - 1, a + count - 1);
    }
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
    if (auto* hint = findChild<QLabel*>("deliverAiViewHint")) hint->hide();   // 1.2.7: the last preview's note
    if (auto* btn = findChild<QPushButton*>("deliverAiPreview")) btn->setText(QStringLiteral("Stop preview"));
    const int frames = b - a + 1;
    // Which fill this is, in the log and the status (12 Nov 2026: v3 or v4 could not be told apart).
    log(c.ai.rerender ? QStringLiteral("AI fill preview: %1, region re-render (LoRA %2, condition %3)")
                            .arg(setup->v5 ? QStringLiteral("v5") : QStringLiteral("v4.3"), from_path(setup->lora),
                                 learned ? QStringLiteral("aces") : QStringLiteral("bt1886"))
                      : QStringLiteral("AI fill preview: v3, clipped pixels only (LoRA %1)").arg(from_path(setup->lora)));
    // 12 Nov 2026: the prompt in the log too (an empty box looked filled: its grey example text)
    log(c.ai.prompt_highlight.trimmed().isEmpty()
            ? QStringLiteral("AI fill preview: no highlight prompt%1")
                  .arg(c.ai.rerender ? QStringLiteral(" (v4 then draws texture with no subject: type what is in the bright region)") : QString())
            : QStringLiteral("AI fill preview: highlight prompt \"%1\"").arg(c.ai.prompt_highlight.trimmed()));
    say(QStringLiteral("AI fill preview: rendering the master of frames %1–%2…").arg(a + 1).arg(b + 1));
    set_ai_progress(0.0, QStringLiteral("master"));
    const auto t0 = std::chrono::steady_clock::now();
    // 1.2.6 (user feedback: "preview stuck"): the seconds so far on the status line, every second,
    // so a model loading for a minute does not look hung. It stops itself when the preview ends.
    genfill_preview_t0_ = t0;
    if (!genfill_preview_ticker_) {
        genfill_preview_ticker_ = new QTimer(this);
        genfill_preview_ticker_->setInterval(1000);
        connect(genfill_preview_ticker_, &QTimer::timeout, this, [this] {
            if (!genfill_preview_busy_) return genfill_preview_ticker_->stop();
            auto* l = findChild<QLabel*>("deliverAiPreviewStatus");
            if (!l) return;
            static const QRegularExpression tail(QStringLiteral(" \u00b7 \\d+ s$"));
            QString t = l->text();
            t.remove(tail);
            const auto s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - genfill_preview_t0_).count();
            l->setText(t + QStringLiteral(" \u00b7 %1 s").arg(s));
        });
    }
    genfill_preview_ticker_->start();
    const QPointer<MainWindow> self(this);
    genfill_preview_master_id_ = -1;
    // v4, region re-render: the source the master was made from, and the route's inverse.
    const QString rr_source = from_path(movie ? shot_sources_.front() : frames_.front().parent_path());
    const QString rr_condition = learned ? QStringLiteral("aces") : QStringLiteral("bt1886");
    auto after_master = [self, setup = *setup, options = c.ai, mdir, fdir, a, b, at, frames, n, t0, rr_source,
                         rr_condition](const QJsonObject& o) {
            if (!self) return;
            self->genfill_preview_master_id_ = 0;
            auto* st2 = self->findChild<QLabel*>("deliverAiPreviewStatus");
            auto end = [self, st2](const QString& t) {
                self->genfill_preview_busy_ = false;
                self->ai_progress_done();
                if (auto* btn = self->findChild<QPushButton*>("deliverAiPreview")) btn->setText(QStringLiteral("Preview AI fill"));
                if (st2) st2->setText(t);
                self->log(t);
                self->refresh_enabled();
                self->sync_ai_generation();
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
            if (options.rerender) {   // AI fill v4: the amended inference of 11 Nov 2026, as genfill_arguments
                fill.insert("region_rerender", true);
                fill.insert("prompt_shadow", QString());   // the highlight prompt is sent (as genfill_arguments)
                fill.insert("source", rr_source);
                fill.insert("source_start", a);
                fill.insert("condition", rr_condition);
                fill.insert("tau_high", 240);
                fill.insert("tau_low", -1);
                fill.insert("rerender_hue", "source");
                fill.insert("rerender_colour", 0.3);
                fill.insert("rerender_level", 3.3);
                fill.insert("rerender_smooth", 0.8);
                fill.insert("rerender_dip", 0.3);   // as genfill_arguments
                fill.insert("rerender_independent", true);
                fill.insert("rerender_calm", 0.9);
                fill.insert("rerender_full_last", true);   // 1.2.9, as genfill_arguments
                // 1.3: v5's finish, as genfill_arguments; "" sent with v4.3 so the warm server drops an earlier v5 request's
                fill.insert("rerender_hold_clipped", setup.v5 ? QString::fromLatin1(kGenFillV5Hold) : QString());
                // 1.3 (24 Nov 2026): the v5e finish, as genfill_arguments; 0 / false with v4.3 so the warm server
                // drops an earlier v5 request's
                fill.insert("rerender_blend_still", setup.v5 ? QString::fromLatin1(kGenFillV5BlendStill).toDouble() : 0.0);
                fill.insert("rerender_hold_calm", setup.v5 ? QString::fromLatin1(kGenFillV5HoldCalm).toDouble() : 0.0);
                fill.insert("rerender_cap_master", setup.v5 && rr_condition == QLatin1String("aces"));
                fill.insert("mask_first_frame", setup.v5 && rr_condition == QLatin1String("aces"));
                if (options.rerender_keep_detail) fill.insert("rerender_keep_detail", true);   // 1.2.7
                fill.insert("brightness_offset", 0.0);
                fill.insert("shape_gain", 1.0);
                fill.insert("fallback", "none");
                fill.insert("focus", "frame");
            }
            const bool cold = !self->genfill_server_->loaded();
            if (st2)
                st2->setText(cold ? QStringLiteral("AI fill preview: loading the fill model (once; a minute or two)…")
                                  : QStringLiteral("AI fill preview: filling %1 frame(s)…").arg(frames));
            if (cold) self->set_ai_progress(-1.0, QStringLiteral("loading the fill model (once)"));
            else self->set_ai_progress(0.1, QStringLiteral("filling"));
            const auto t1 = std::chrono::steady_clock::now();
            QString why2;
            const bool started = self->genfill_server_->request(
                setup, fill,
                [self, st2, frames](const QJsonObject& e) {
                    if (!self || !st2) return;
                    const QString ev = e.value("event").toString();
                    if (ev == "ready") {
                        st2->setText(QStringLiteral("AI fill preview: model loaded on %1 in %2 s; filling %3 frame(s)…")
                                         .arg(e.value("device").toString())
                                         .arg(e.value("load_seconds").toDouble(), 0, 'f', 0)
                                         .arg(frames));
                        self->set_ai_progress(0.1, QStringLiteral("filling"));
                    } else if (ev == "embedding") {   // a typed prompt not in the cache: umT5 loads once (6 Nov 2026)
                        st2->setText(QStringLiteral("AI fill preview: embedding %1 new prompt(s), once (a minute or more); "
                                                    "the presets are instant")
                                         .arg(e.value("prompts").toArray().size()));
                        self->set_ai_progress(-1.0, QStringLiteral("embedding the prompt (once)"));
                    } else if (ev == "progress") {
                        const int step = e.value("step").toInt(), steps = std::max(1, e.value("steps").toInt());
                        st2->setText(QStringLiteral("AI fill preview: step %1 / %2").arg(step).arg(steps));
                        self->set_ai_progress(genfill_progress_share(0.1, 0.95, e.value("window").toInt(1),
                                                                     e.value("windows").toInt(1), step, steps),
                                              QStringLiteral("step %1 / %2").arg(step).arg(steps));
                    }
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
                    self->set_ai_progress(0.97, QStringLiteral("showing it"));
                    auto p = std::make_shared<GenFillPreviewFrame>();
                    p->index = at;
                    p->options = options;
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
                        p->mask = std::make_shared<const NetworkLinearImage>(preview_view_image(mk));
                        p->changed = np ? double(changed) / double(np) : 0.0;
                    }
                    p->first = a;
                    p->last = b;
                    p->fill_dir = fdir;
                    const int pw = m->width(), ph = m->height();
                    const QString mexr = QDir(mdir).filePath(preview_name(at)), fexr = QDir(fdir).filePath(preview_name(at));
                    // The master and the fill as the viewer shows the live frame: through the monitor.
                    auto finish = [self, end, options, p, a, b, n, frames, master_s, fill_s, cold] {
                        self->genfill_preview_ = std::move(*p);
                        self->fill_view_cache_.clear();   // a new preview: the viewer's AI fill reads it afresh
                        // The whole shot at this rate: its master and its fill, per frame (the model's load is once).
                        const double per_frame = (master_s + fill_s) / std::max(1, frames);
                        const double whole_min = per_frame * n / 60.0;
                        const QString full = (QStringLiteral("AI fill preview of frames %1–%2: master %3 s, fill %4 s%5 · %6 % of this frame "
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
                        end(full);   // the log has all of it
                        // 12 Nov 2026: the card says it in one line; the timings on hover
                        if (auto* st3 = self->findChild<QLabel*>("deliverAiPreviewStatus")) {
                            st3->setToolTip(full);
                            st3->setText(QStringLiteral("Preview ready \u00b7 %1 % invented \u00b7 whole shot about %2 min")
                                             .arg(100.0 * self->genfill_preview_->changed, 0, 'f', 1)
                                             .arg(whole_min, 0, 'f', whole_min < 10 ? 1 : 0) +
                                         genfill_preview_hint(self->genfill_preview_->changed, !options.region_json.isEmpty(),
                                                              self->session_.display_nits()));
                        }
                        self->show_genfill_preview(QStringLiteral("fill"));
                        self->frame_genfill_preview();   // 1.2.7
                    };
                    self->monitor_preview_planes(mexr, pw, ph, [self, end, p, fexr, pw, ph, finish](std::optional<PlanarBuffer> mm, QString why) {
                        if (!self) return;
                        if (!mm) return end(QStringLiteral("AI fill preview: the master could not be shown. ") + why);
                        p->master = std::make_shared<const NetworkLinearImage>(preview_view_image(*mm));
                        self->monitor_preview_planes(fexr, pw, ph, [self, end, p, finish](std::optional<PlanarBuffer> ff, QString why2) {
                            if (!self) return;
                            if (!ff) return end(QStringLiteral("AI fill preview: the fill could not be shown. ") + why2);
                            p->fill = std::make_shared<const NetworkLinearImage>(preview_view_image(*ff));
                            finish();
                        });
                    });
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
            self->set_ai_progress(0.1 * p.value("index").toInt() / std::max(1, frames),
                                  QStringLiteral("master %1 / %2").arg(p.value("index").toInt()).arg(frames));
        });
        if (genfill_preview_master_id_ != 0) genfill_preview_master_id_ = id;   // still running (a refusal answers at once)
    } else {
        start_analytic_preview_master(mdir, a, b, std::move(after_master));
    }
    refresh_enabled();
    sync_ai_generation();
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

void MainWindow::frame_genfill_preview() {
    if (!genfill_preview_ || !QSettings().value("genfill/previewFrames", true).toBool()) return;
    QStringList did;
    // 1.3 (24 Nov 2026): the wipe stays; the fill is on its RUDRA side, the SDR side shows the source
    if (session_.wipe) did << QStringLiteral("the fill is on the RUDRA side of the wipe (drag it to see more)");
    // Only from 0 EV: a view exposure the user set stays theirs. The f/stop button (Alt+E) goes back to 0.
    if (session_.view_ev == 0.0 && genfill_preview_->changed > 0.0) {
        session_.set_view_ev(-2.0);
        did << QStringLiteral("view at \u22122 EV (Alt+E for 0)");
    }
    const QString why = genfill_preview_->changed > 0.0
                            ? QStringLiteral("AI fill preview: what it drew is above SDR white, so at 0 EV it looks white")
                            : QStringLiteral("AI fill preview: nothing was invented on this frame");
    const QString text = did.isEmpty() ? why + QStringLiteral(". Invented shows where it acted.")
                                       : why + QStringLiteral(": ") + did.join(QStringLiteral(", ")) + QStringLiteral(". Invented shows where it acted.");
    if (auto* hint = findChild<QLabel*>("deliverAiViewHint")) {   // 21 Nov 2026: on the AI generation card
        hint->setText(text);
        hint->setToolTip(text);
        hint->show();
    }
    statusBar()->showMessage(text, 15000);
    log(text);
    sync_ui();
}

void MainWindow::show_genfill_preview(const QString& which) {
    if (ai_region_view_) set_ai_region_view(false);   // 1.2.6: the preview's pictures replace the region
    auto* st = findChild<QLabel*>("deliverAiPreviewStatus");
    if (!genfill_preview_) {
        if (st) st->setText(QStringLiteral("No AI fill preview yet: Preview AI fill fills the frame at the playhead."));
        return;
    }
    // The viewer moved on: back to the previewed frame; the picture asked for goes up when it
    // lands (apply_fill_view runs on every frame shown), with no second click (6 Nov 2026).
    const bool away = current_ != genfill_preview_->index;
    if (away) step_to(genfill_preview_->index);
    genfill_preview_show_ = which;
    const auto& p = *genfill_preview_;
    if (which == "fill") {
        // The AI fill: on (24 Nov 2026: the viewer's AI fill button is AI generation itself), so it stays on this
        // frame and the previewed ones around it whatever the route renders next (6 Nov 2026).
        genfill_preview_pinned_ = false;
        if (!ai_generation_) set_deliver("ai_generation", "1");   // the button turns blue and shows the fill
        set_fill_view(true);
    } else {
        genfill_preview_pinned_ = true;
        if (!away) apply_fill_view();
    }
    if (auto* b = findChild<QPushButton*>("deliverAiShowMask")) b->setChecked(which == "mask");
    sync_ai_generation();   // 1.3: the viewer's What AI drew follows
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
    // The previewed frame's fill went through the monitor when the preview came back (9 Nov 2026):
    // show that one. The frames beside it, and a finished render's, are read here as scene light.
    if (genfill_preview_ && index == genfill_preview_->index && genfill_preview_->fill && !aces_rt_playing_) {
        fill_view_cache_[index] = genfill_preview_->fill;
        return genfill_preview_->fill;
    }
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
    // 1.3 (24 Nov 2026, Ahmed: "AI fill only on the RUDRA side"): the SDR side of the viewer (the wipe's left, Hold B,
    // the Original SDR view) is always the frame as it came in, never the fill: the learned route's own SDR and
    // plain inverse, or the analytic frame's SDR and baseline. The analytic route had neither here, so its
    // baseline was the fill itself and its SDR side showed the fill.
    auto show = [this](const NetworkLinearImage& img, const NetworkLinearImage& fallback_base) {
        if (!viewer_) return;
        const auto fits = [&img](int w, int h) { return w == img.width() && h == img.height(); };
        const bool base_ok = aces_base_img_ && fits(aces_base_img_->width(), aces_base_img_->height());
        const bool sdr_ok = aces_sdr_img_ && fits(aces_sdr_img_->width(), aces_sdr_img_->height());
        const auto* cur = (current_frame_ && current_frame_->index == current_ &&
                           fits(current_frame_->sdr.width(), current_frame_->sdr.height()))
                              ? current_frame_.get()
                              : nullptr;
        std::shared_ptr<const NetworkLinearImage> own_base;
        if (!base_ok && cur)
            own_base = (cur->baseline && fits(cur->baseline->width(), cur->baseline->height()))
                           ? cur->baseline
                           : std::make_shared<const NetworkLinearImage>(analytic_baseline(cur->sdr, cur->model.corpus_ev));
        const bool codes_ok = aces_codes_.size() == std::size_t(img.width()) * std::size_t(img.height());
        viewer_->set_composited(img, base_ok ? *aces_base_img_ : own_base ? *own_base : fallback_base,
                                codes_ok ? aces_codes_ : std::vector<float>{});
        if (sdr_ok) viewer_->set_source_sdr(*aces_sdr_img_);
        else if (cur) viewer_->set_source_sdr(cur->sdr);
    };
#endif
    if (frames_.empty()) return false;
    // 1.2.6 (18 Nov 2026): "Where it fills": the AI fill's region on the frame on screen, any route.
    if (ai_region_view_ && !playing()) {
        std::shared_ptr<const SdrImage> sdr;
        if (const auto r = ai_fill_region_now(&sdr)) {
#ifdef RUDRA_APP_VIEWER
            const NetworkLinearImage pic(ai_fill_region_picture(*sdr, *r));
            show(pic, pic);
#endif
            return true;
        }
    }
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
            statusBar()->showMessage(QStringLiteral("AI fill: no fill for frame %1 (Reconstruct \u203a AI generation \u203a Preview "
                                                    "AI fill, or render with Deliver's AI Fill Generation)").arg(current_ + 1), 3000);
        return false;
    }
#ifdef RUDRA_APP_VIEWER
    show(*img, *img);
#endif
    return true;
}

void MainWindow::set_fill_view(bool on) {
    if (on && ai_region_view_) set_ai_region_view(false);   // 1.2.6
    if (fill_view_on_ == on) {
        if (on) apply_fill_view();
        return;
    }
    fill_view_on_ = on;
    sync_ai_generation();
    if (auto* m = findChild<QPushButton*>("deliverAiShowMask")) m->setChecked(false);   // the button's picture replaces it
    if (on) {
        // The fill is drawn on RUDRA's side only. 21 Nov 2026 (testwindow.mp4: "we cannot see it in the
        // viewport") the button took the start's SDR | RUDRA wipe off, since a fill on the left of the frame
        // was under the SDR. 1.3 (24 Nov 2026, Ahmed): the wipe stays where it is, its SDR side shows the
        // source, and one side is RUDRA; without a wipe RUDRA is shown, as before.
        QStringList did;
        if (session_.wipe) {
            if (session_.wipe_left != "model" && session_.wipe_right != "model") {
                const std::string left = session_.wipe_left;   // a copy: set_wipe_sources writes wipe_left
                session_.set_wipe_sources(left, "model");
                did << QStringLiteral("RUDRA on the wipe's right");
            }
            did << QStringLiteral("the fill is on the RUDRA side of the wipe (drag it to see more)");
        } else if (session_.show != "model") {
            session_.set_show("model");
            did << QStringLiteral("RUDRA shown");
        }
        did << tone_map_for_fill();   // 1.3: on an SDR screen, a view that shows a fill above white
        genfill_preview_pinned_ = false;
        if (!apply_fill_view()) return;
        statusBar()->showMessage(QStringLiteral("Viewer: the AI fill (Changes shows what it invented)") +
                                     (did.isEmpty() ? QString() : QStringLiteral("; ") + did.join(QStringLiteral(", "))),
                                 6000);
    } else {
        genfill_preview_pinned_ = false;
        untone_map_for_fill();
        if (!frames_.empty()) step_to(current_);   // back to the frame without the fill, on whichever route
    }
}

}  // namespace rudra::app

