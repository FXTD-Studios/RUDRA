#include "workflow_check.hpp"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QSaveFile>
#include <QTimer>
#include <QSpinBox>
#include <QThread>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "main_window.hpp"
#include "rudra/deliver/master.hpp"

#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/still.hpp"
#endif
#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {

QString q(const std::string& s) { return QString::fromStdString(s); }

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, std::size_t(p * double(v.size() - 1) + 0.5))];
}

}  // namespace

int run_workflow_check(MainWindow& w, const WorkflowArgs& a) {
    if (QFileInfo::exists(a.report)) {
        std::fprintf(stderr, "Workflow report already exists; choose a fresh report path.\n");
        return 1;
    }
    QJsonObject report;
    report["mode"] = a.analytic ? "analytic" : "model";
    report["timeout_ms"] = a.timeout_ms;
    QJsonArray dialogs;
    // Active only during this scripted check. Cancel unexpected message boxes
    // rather than entering an unbounded nested event loop or approving an action.
    QTimer dialog_watch;
    QObject::connect(&dialog_watch, &QTimer::timeout, [&] {
        for (QWidget* widget : QApplication::topLevelWidgets())
            if (auto* box = qobject_cast<QMessageBox*>(widget); box && box->isVisible()) {
                dialogs.append(QJsonObject{{"title", box->windowTitle()}, {"text", box->text()}});
                box->reject();
            }
    });
    dialog_watch.start(10);
    auto wait_for = [&](const std::function<bool()>& done, int ms) {
        QElapsedTimer t; t.start();
        while (!done()) {
            if (!dialogs.isEmpty() || t.elapsed() >= std::min(ms, a.timeout_ms)) return false;
            QApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(1);
        }
        return dialogs.isEmpty();
    };
    bool all = true;
    auto step = [&](const char* name, QJsonObject o, bool ok) {
        ok = ok && dialogs.isEmpty();
        o["ok"] = ok;
        report[name] = o;
        all = all && ok;
        std::printf("  %-8s %s\n", name, ok ? "PASS" : "FAIL");
        std::fflush(stdout);
        return ok;
    };
    auto finish = [&]() {
        all = all && dialogs.isEmpty();
        report["unexpected_dialogs"] = dialogs;
        report["verdict"] = all ? "PASS" : "FAIL";
        if (w.mastering()) w.master(); // a second click requests cancellation
        if (w.video_queues().running()) w.video_queues().stop();
        QSaveFile f(a.report);
        const auto bytes = QJsonDocument(report).toJson();
        if (!f.open(QIODevice::WriteOnly) || f.write(bytes) != bytes.size() || !f.commit()) {
            std::fprintf(stderr, "Could not save workflow report: %s\n", f.errorString().toLocal8Bit().constData());
            return 1;
        }
        std::printf("  => %s (%s)\n", all ? "PASS" : "FAIL", a.report.toLocal8Bit().constData());
        return all ? 0 : 1;
    };
    std::printf("RUDRA workflow check: %s (%s)\n", a.frames.toLocal8Bit().constData(),
                a.analytic ? "analytic, no model" : a.package.toLocal8Bit().constData());
    if (!step("arguments", {{"fresh_output", !QFileInfo::exists(a.out)}},
        !a.frames.isEmpty() && !a.out.isEmpty() && !QFileInfo::exists(a.out) && a.timeout_ms > 0 && a.timeout_ms <= 3600000 &&
        (a.analytic ? a.package.isEmpty() && a.backend.isEmpty() : !a.package.isEmpty()))) return finish();

    // 1. The model, on the backend asked for, its goldens checked on first use.
    if (a.analytic) {
        w.set_use_model(false, false);
        if (!step("model", {{"skipped", true}, {"reason", "Analytic reconstruction; no model loaded"}},
             !w.using_model() && w.model_package().empty())) return finish();
    } else {
        QJsonObject o;
        struct Outcome { std::optional<bool> done; QString why; };
        auto outcome = std::make_shared<Outcome>(); // callback can outlive a timed-out wait
        QElapsedTimer t;
        t.start();
        w.use_model(a.package.toStdString(), BackendChoice::from_key(a.backend.toStdString()),
                    [outcome](bool ok, const QString& y) {
                        outcome->done = ok;
                        outcome->why = y;
                    });
        const bool completed = wait_for([&] { return outcome->done.has_value(); }, 600000);
        o["package"] = a.package;
        o["backend"] = w.model_backend() ? q(w.model_backend()->label()) : QString();
        o["device"] = w.findChild<QLabel*>("device")->text();
        o["load_s"] = double(t.elapsed()) / 1000.0;
        if (!outcome->why.isEmpty()) o["why"] = outcome->why;
        if (!step("model", o, completed && outcome->done.value_or(false))) return finish();
        // This check is of the model's workflow: the model runs (the app's
        // default is the analytic reconstruction).
        w.set_use_model(true, false);
    }

    // 2. The shot.
    QElapsedTimer total;
    total.start();
    w.open_source(a.frames, true);
    const int n = int(w.frame_count());
    {
        QJsonObject o;
        o["folder"] = a.frames;
        o["frames"] = n;
        o["preview_max_side"] = w.preview_max_side();
        if (!step("open", o, n > 0)) return finish();
    }

    // 3. Scrub every frame, as the transport steps: each delivered as itself.
    {
        std::vector<double> ms;
        int checked = 0, wrong = 0, missing = 0;
        for (int i = 0; i < n; ++i) {
            QElapsedTimer t;
            t.start();
            if (i == 0) w.run("first");
            else w.run("next");
            const bool got = wait_for([&] { return w.current_index() == i && w.presented_index() == i && w.frame_fields(); },
                                      120000);
            if (!got) {
                ++missing;
                break;
            }
            ms.push_back(double(t.nsecsElapsed()) / 1e6);
#ifdef RUDRA_HAVE_STILL_DECODE
            if (a.check_every > 0 && (i % a.check_every == 0 || i == n - 1)) {
                auto d = decode_sdr_file(w.frames()[std::size_t(i)]);
                const auto* sdr = w.frame_sdr();
                ++checked;
                // The frame on screen is its own decode at the preview size.
                const SdrImage want = d ? fit_max_side(d->rgb, w.preview_max_side()) : SdrImage();
                if (!d || !sdr || want.buffer().span().size() != sdr->buffer().span().size() ||
                    !std::equal(want.buffer().span().begin(), want.buffer().span().end(), sdr->buffer().span().begin()))
                    ++wrong;
            }
#endif
        }
        QJsonObject o;
        o["frames"] = n;
        o["delivered"] = int(ms.size());
        o["missing"] = missing;
        o["checked_against_decode"] = checked;
        o["not_itself"] = wrong;
        o["median_ms"] = percentile(ms, 0.5);
        o["p95_ms"] = percentile(ms, 0.95);
        o["max_ms"] = ms.empty() ? 0.0 : *std::max_element(ms.begin(), ms.end());
        o["first_ms"] = ms.empty() ? 0.0 : ms.front();
        if (!step("scrub", o, missing == 0 && wrong == 0 && int(ms.size()) == n)) return finish();
    }

    // 4. Grade, then undo it all and redo it all: params() at each end.
    {
        Session& s = w.session();
        const std::string before = s.params_json();
        const std::size_t depth0 = s.undo_depth();
        w.run("strength-up");
        w.run("strength-up");
        w.run("mode-highlights");
        w.run("preserve");
        s.region_press(0, 100.0);
        s.region_move(160.0, false);
        s.region_release();
        const std::string graded = s.params_json();
        const std::size_t moves = s.undo_depth() - depth0;
        for (std::size_t k = 0; k < moves; ++k) w.run("undo");
        const bool undone = s.params_json() == before;
        for (std::size_t k = 0; k < moves; ++k) w.run("redo");
        const bool redone = s.params_json() == graded;
        QJsonObject o;
        o["moves"] = int(moves);
        o["params"] = q(graded);
        o["undo_restores"] = undone;
        o["redo_restores"] = redone;
        step("grade", o, moves == 5 && undone && redone && graded != before);
    }

    // 5. Compare: the wipe, the flip to the baseline, the layers.
    {
        QJsonObject o;
        Session& s = w.session();
        w.run("wipe");
        bool ok = s.wipe.has_value();
#ifdef RUDRA_APP_VIEWER
        if (auto* v = w.viewer()) {
            QApplication::processEvents();
            ok = ok && v->view().wipe >= 0.0;
            o["viewer_wipe"] = v->view().wipe;
        }
#endif
        w.run("wipe");
        ok = ok && !s.wipe.has_value();
        s.key_down("b", false);
        ok = ok && s.flip_held;
        s.key_up("b");
        w.run("view-false-colour");
        ok = ok && s.view_layer == 1;
#ifdef RUDRA_APP_VIEWER
        if (auto* v = w.viewer()) ok = ok && v->view().mode == ViewMode::FalseColour;
#endif
        w.run("view-difference");
        ok = ok && s.view_layer == 2;
        w.run("view-image");
        ok = ok && s.view_layer == 0;
        o["wipe_flip_layers"] = ok;
        step("compare", o, ok);
    }

    // 6. Probe and measure the frame on screen (the graded one).
    {
        QJsonObject o;
        w.measure_now();
        const auto* m = w.measurement();
        bool ok = m != nullptr;
        if (m) {
            const auto& mm = m->measured.metrics;
            o["maxcll"] = mm.maxcll;
            o["maxfall"] = mm.maxfall;
            o["peak_nits"] = mm.peak_nits;
            o["p99_nits"] = mm.p99_nits;
            o["median_nits"] = mm.median_nits;
            o["above_diffuse_white_pct"] = mm.above_diffuse_white_pct;
            o["highlight_mask_pct"] = mm.highlight_mask_pct;
            o["compose_ms"] = m->compose_ms;
            QJsonArray probes;
            for (auto [fx, fy] : {std::pair{0.5, 0.5}, std::pair{0.0, 0.0}, std::pair{0.999, 0.999}}) {
                const double x = fx * m->width, y = fy * m->height;
                const auto p = m->probe_at(x, y);
                ok = ok && p.has_value();
                w.probe_pixel(std::pair{x, y});
                ok = ok && w.probe_box() && w.probe_box()->isVisible();
                if (p) probes.append(QJsonObject{{"x", p->x}, {"y", p->y}, {"model_nits", p->model_nits},
                                                 {"baseline_nits", p->baseline_nits}});
            }
            w.probe_pixel(std::nullopt);
            o["probes"] = probes;
            ok = ok && std::isfinite(mm.peak_nits) && mm.maxcll >= 1.0;
        }
        step("measure", o, ok);
    }

    // 7. Master three frames, the first, the middle and the last, each an
    // image master of the frame on screen with the grade above.
    {
        QJsonObject o;
        QJsonArray masters;
        bool ok = true;
        QDir().mkpath(a.out);
        w.findChild<QLineEdit*>("renderDir")->setText(a.out);
        auto* mode = w.findChild<QComboBox*>("renderMode");
        mode->setCurrentIndex(mode->findData("image"));
        std::vector<int> which{0, n / 2, n - 1};
        which.erase(std::unique(which.begin(), which.end()), which.end());
        for (int idx : which) {
            w.run("first");
            for (int k = 0; k < idx; ++k) w.run("next");
            if (!wait_for([&] { return w.current_index() == idx && w.presented_index() == idx && w.frame_fields(); }, 120000)) {
                ok = false; break;
            }
            const QString name = QStringLiteral("workflow_%1").arg(idx, 6, 10, QChar('0'));
            w.findChild<QLineEdit*>("renderName")->setText(name);
            QElapsedTimer t;
            t.start();
            auto* status_label = w.findChild<QLabel*>("renderStatus");
            status_label->clear();
            w.master();
            // Finished is the page's status line, set when the job's end reaches this thread.
            const bool done = wait_for([&] {
                const QString line = status_label->text();
                return !w.mastering() && !line.isEmpty();
            }, 600000);
            const QString status = status_label->text();
            const QString exr = QDir(a.out).filePath(name + ".exr"), side = QDir(a.out).filePath(name + ".json");
            const bool written = QFile::exists(exr) && QFile::exists(side);
            bool analytic_tag = !a.analytic;
            if (a.analytic) {
                QFile sidecar(side);
                if (sidecar.open(QIODevice::ReadOnly))
                    analytic_tag = QJsonDocument::fromJson(sidecar.readAll()).object()["reconstruction"].toString() == "analytic (no model)";
            }
            ok = ok && done && written && analytic_tag && status.startsWith("Rendered 1 frame");
            masters.append(QJsonObject{{"frame", q(w.frames()[std::size_t(idx)].string())},
                                       {"exr", exr},
                                       {"sidecar", side},
                                       {"params", q(master_request_json(w.master_request()))},
                                       {"seconds", double(t.elapsed()) / 1000.0},
                                       {"status", status}});
            if (!ok) break;
        }
        o["masters"] = masters;
        o["folder"] = a.out;
        if (!step("master", o, ok)) return finish();
    }
    if (!a.movie.isEmpty()) {
        // Phase 4: the movie as a shot, every frame delivered, then HDR10 through the queue.
        QJsonObject o;
        QElapsedTimer t;
        t.start();
        w.open_source(a.movie);
        const int count = int(w.frame_count());
        bool ok = w.shot_is_video() && count > 0;
        int delivered = 0;
        for (int i = 0; ok && i < count; ++i) {
            if (i) w.run("next");
            if (wait_for([&] { return w.current_index() == i && w.presented_index() == i && w.frame_fields(); }, 120000)) ++delivered;
            else ok = false;
        }
        o["frames"] = count;
        o["delivered"] = delivered;
        o["scrub_s"] = double(t.elapsed()) / 1000.0;
        ok = ok && delivered == count;
        const QString name = "workflow_movie_hdr10";
        w.findChild<QLineEdit*>("renderDir")->setText(a.out);
        w.findChild<QLineEdit*>("renderName")->setText(name);
        QElapsedTimer e;
        e.start();
        const bool queued = ok && w.queue_video_export("hdr10");
        const QString master = QDir(a.out).filePath(name + ".mp4");
        bool passed = false;
        if (queued) {
            const bool completed = wait_for([&] { return !w.video_queues().running(); }, 3600000);
            o["completed"] = completed;
            QFile side(master + ".json");
            if (completed && side.open(QIODevice::ReadOnly)) {
                const auto doc = QJsonDocument::fromJson(side.readAll()).object();
                passed = doc["qc"].toObject()["passed"].toBool();
                o["max_cll"] = doc["max_cll"];
                o["max_fall"] = doc["max_fall"];
            }
        }
        o["queued"] = queued;
        o["master"] = master;
        o["qc_passed"] = passed;
        o["export_s"] = double(e.elapsed()) / 1000.0;
        step("movie", o, ok && queued && passed && QFile::exists(master));
    }
    report["total_s"] = double(total.elapsed()) / 1000.0;
    return finish();
}

}  // namespace rudra::app
