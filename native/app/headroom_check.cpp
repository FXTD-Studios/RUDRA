#include "headroom_check.hpp"

#include <algorithm>
#include <memory>

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QLineEdit>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QSaveFile>
#include <QThread>
#include <QTimer>

#include <cstdio>
#include <functional>

#include "main_window.hpp"
#include "rudra/platform/hash.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {
QString hex(std::uint64_t v) { return QString::number(qulonglong(v), 16); }
std::uint64_t image_hash(const NetworkLinearImage& img) { return xxh64(std::as_bytes(img.buffer().span())); }
}  // namespace

int run_headroom_check(MainWindow& w, const HeadroomCheckArgs& a) {
    if (QFileInfo::exists(a.report)) {
        std::fprintf(stderr, "Headroom report already exists; choose a fresh report path.\n");
        return 1;
    }
    QJsonObject report;
    report["schema"] = "rudra-headroom-check-v1";
    report["movie"] = a.movie;
    report["timeout_ms"] = a.timeout_ms;
    QJsonArray dialogs;
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
        QElapsedTimer t;
        t.start();
        while (!done()) {
            if (!dialogs.isEmpty() || t.elapsed() >= ms) return false;
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
        std::printf("  %-9s %s\n", name, ok ? "PASS" : "FAIL");
        std::fflush(stdout);
        return ok;
    };
    auto stats_json = [&] {
        const auto s = w.headroom_stats();
        return QJsonObject{{"rendered", qint64(s.rendered)},
                           {"presented", qint64(s.presented)},
                           {"discarded", qint64(s.discarded)},
                           {"rejected", qint64(s.rejected)},
                           {"pending_rendered_bytes", qint64(s.pending_rendered_bytes)},
                           {"shown_rendered_bytes", qint64(s.shown_rendered_bytes)}};
    };
    auto finish = [&] {
        all = all && dialogs.isEmpty();
        report["unexpected_dialogs"] = dialogs;
        report["problem"] = w.headroom_problem();
        report["verdict"] = all ? "PASS" : "FAIL";
        QSaveFile f(a.report);
        const auto bytes = QJsonDocument(report).toJson();
        if (!f.open(QIODevice::WriteOnly) || f.write(bytes) != bytes.size() || !f.commit()) {
            std::fprintf(stderr, "Could not save headroom report: %s\n", f.errorString().toLocal8Bit().constData());
            return 1;
        }
        std::printf("  => %s (%s)\n", all ? "PASS" : "FAIL", a.report.toLocal8Bit().constData());
        return all ? 0 : 1;
    };
    std::printf("RUDRA headroom check: %s\n", a.movie.toLocal8Bit().constData());

    if (!step("arguments", {{"capable", w.headroom_capable()}, {"movie_exists", QFileInfo(a.movie).isFile()}},
              w.headroom_capable() && QFileInfo(a.movie).isFile()))
        return finish();

    w.raise();
    w.activateWindow();
    QElapsedTimer t;
    t.start();
    w.open_source(a.movie);
    const bool opened = wait_for([&] { return w.frame_count() > 0 && w.shot_is_video(); }, a.timeout_ms);
    const int n = int(w.frame_count());
    if (!step("open", {{"frames", n}, {"ms", double(t.elapsed())}}, opened && n >= 3)) return finish();

    // The controls v1 refuses, turned off as an artist would; nothing else changes.
    w.set_steady_shots(false);
    if (w.session().anchor) w.session().toggle_anchor();
    if (w.session().carry_chroma) w.session().toggle_carry_chroma();
    if (w.session().grade.clip_fill) w.session().toggle_clip_fill();
    t.restart();
    const auto enabled = w.set_headroom_preview(true);
    bool shown = enabled && wait_for([&] { return w.headroom_frame() != nullptr || !w.headroom_problem().isEmpty(); },
                                     a.timeout_ms) && w.headroom_frame();
    QJsonObject first{{"ms", double(t.elapsed())}, {"routed", w.headroom_routed()},
                      {"legacy_frame_cleared", w.frame_fields() == nullptr}};
    std::uint64_t first_hash = 0;
    if (shown) {
        const auto& f = *w.headroom_frame();
        first_hash = image_hash(*f.model);
        first["index"] = f.index;
        first["width"] = f.width;
        first["height"] = f.height;
        first["model_xxh64"] = hex(first_hash);
        first["baseline_xxh64"] = hex(image_hash(*f.baseline));
        first["stats"] = stats_json();
    }
    if (!step("paused", first, shown && w.headroom_frame()->index == 0 && w.frame_fields() == nullptr)) return finish();

    // Seeks: forward one, to the last frame, back to the first.
    QJsonArray seeks;
    bool seek_ok = true;
    for (int target : {1, n - 1, 0}) {
        t.restart();
        for (int guard = 0; w.current_index() != target && guard < n; ++guard)
            w.run(target > w.current_index() ? "next" : "prev");
        const bool ok = wait_for([&] { return w.presented_index() == target; }, a.timeout_ms);
        seeks.append(QJsonObject{{"to", target}, {"ok", ok}, {"ms", double(t.elapsed())}});
        seek_ok = seek_ok && ok;
    }
    const bool same_pixels = w.headroom_frame() && image_hash(*w.headroom_frame()->model) == first_hash;
    if (!step("seek", {{"seeks", seeks}, {"frame0_repeatable", same_pixels}}, seek_ok && same_pixels)) return finish();

    // A grade change re-renders the paused frame; its undo returns the same pixels.
    t.restart();
    auto before = w.headroom_stats().presented;
    w.session().nudge_exposure(1);
    const bool graded = wait_for([&] { return w.headroom_stats().presented > before && w.headroom_frame(); }, a.timeout_ms);
    const std::uint64_t graded_hash = graded ? image_hash(*w.headroom_frame()->model) : 0;
    const double grade_ms = double(t.elapsed());
    before = w.headroom_stats().presented;
    w.run("undo");
    const bool undone = wait_for([&] { return w.headroom_stats().presented > before && w.headroom_frame(); }, a.timeout_ms);
    const bool restored = undone && image_hash(*w.headroom_frame()->model) == first_hash;
    if (!step("grade", {{"grade_ms", grade_ms}, {"graded_xxh64", hex(graded_hash)}, {"changed", graded_hash != first_hash},
                        {"undo_restores_pixels", restored}},
              graded && graded_hash != first_hash && restored))
        return finish();

    // Playback to the end: in order, timed (an inspection mode, not real time).
    t.restart();
    w.run("play");
    int last = w.presented_index(), order_errors = 0, frames_seen = 0;
    const bool played = wait_for([&] {
        const int now = w.presented_index();
        if (now != last) {
            if (now < last) ++order_errors;
            last = now;
            ++frames_seen;
        }
        return !w.playing();
    }, std::max(a.timeout_ms, 1000 * n));
    const double play_s = double(t.elapsed()) / 1000.0;
    if (!step("play", {{"frames_seen", frames_seen}, {"order_errors", order_errors}, {"seconds", play_s},
                       {"fps", play_s > 0 ? frames_seen / play_s : 0.0}, {"ended_at", w.presented_index()},
                       {"stats", stats_json()}},
              played && order_errors == 0 && w.presented_index() == n - 1))
        return finish();

    // The transactional export: decode, render, encode, QC, then publish.
    {
        const QString out = QDir::temp().filePath(QStringLiteral("rudra-headroom-check-%1").arg(QDateTime::currentMSecsSinceEpoch()));
        QDir().mkpath(out);
        w.findChild<QLineEdit*>("renderDir")->setText(out);
        w.findChild<QLineEdit*>("renderName")->setText("headroom_check");
        t.restart();
        const bool started = w.queue_video_export("hdr10");
        const bool done = started && wait_for([&] { return !w.exporting_headroom(); }, std::max(a.timeout_ms, 4000 * n));
        const QString file = QDir(out).filePath("headroom_check.mp4");
        const bool published = QFileInfo::exists(file) && QFileInfo::exists(file + ".json");
        QJsonObject side;
        if (published) {
            QFile f(file + ".json");
            if (f.open(QIODevice::ReadOnly)) side = QJsonDocument::fromJson(f.readAll()).object();
        }
        if (!step("export", {{"seconds", double(t.elapsed()) / 1000.0}, {"file", file}, {"result", w.headroom_export_result()},
                             {"sidecar_frames", side["frames"]}, {"maxcll", side["maxcll"]}, {"policy", side["policy"]}},
                  done && published && side["frames"].toInt() == n && side["policy"].toString() == "bt1886-headroom-v1"))
            return finish();
    }

#ifdef RUDRA_APP_VIEWER
    if (auto* v = w.viewer()) {
        // The viewer holds the uploaded headroom picture and presents it. A
        // window that is never exposed (locked session, display off, minimized)
        // renders nothing at all; that is reported as such, not as a headroom fault.
        w.showNormal();
        w.raise();
        w.activateWindow();
        wait_for([&] { return v->isExposed(); }, 10000);
        const bool exposed = v->isExposed();
        const auto s0 = v->status();
        // grab() is asynchronous: the callback can fire after a timed-out wait,
        // so it writes into shared state, never into this stack frame.
        struct GrabState { bool grabbed = false, nonzero = false; QString format; };
        auto gs = std::make_shared<GrabState>();
        v->grab([gs](const ViewerWindow::Grab& g) {
            gs->grabbed = true;
            gs->format = QString::fromStdString(g.format);
            gs->nonzero = std::any_of(g.bytes.begin(), g.bytes.end(), [](unsigned char b) { return b != 0; });
        });
        wait_for([&] { return gs->grabbed; }, std::min(a.timeout_ms, 30000));
        const bool grabbed = gs->grabbed, nonzero = gs->nonzero;
        const QString format = gs->format;
        const auto s1 = v->status();
        step("viewer", {{"exposed", exposed}, {"window_visible", w.isVisible()}, {"minimized", w.isMinimized()},
                        {"reason", exposed ? QString() : QStringLiteral("the viewer window was never exposed: unlock the screen and leave the window visible")},
                        {"has_frame", s1.has_frame}, {"frames_presented", qint64(s1.frames)},
                        {"swapchain", QString::fromStdString(s1.swapchain)}, {"grab_format", format},
                        {"grab_nonzero", nonzero}, {"begin_failures", qint64(s1.begin_failures - s0.begin_failures)}},
             s1.has_frame && grabbed && nonzero);
    } else {
        step("viewer", {{"skipped", "window built without the viewer"}}, true);
    }
#else
    step("viewer", {{"skipped", "build without the viewer"}}, true);
#endif

    // Off again: the legacy engine shows the frame on screen.
    t.restart();
    (void)w.set_headroom_preview(false);
    const bool legacy = wait_for([&] { return w.frame_fields() != nullptr; }, a.timeout_ms);
    step("legacy", {{"ms", double(t.elapsed())}, {"routed", w.headroom_routed()}}, legacy && !w.headroom_routed());
    return finish();
}

}  // namespace rudra::app
