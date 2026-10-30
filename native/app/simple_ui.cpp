// Simple (29 Oct 2026, the approved "Look, Check, Deliver" redesign): the
// workspace a new user starts in. Three tabs, the controls most shots need, and
// "Advanced" to everything else (Full, which keeps the original layout).
//
//   Look     the monitor (SDR 100 / HDR 1000), Strength, Peak, Exposure,
//            Contrast; Copy look, Paste, Reset.
//   Check    the frame in four numbers, Grab still / Compare; the scopes show
//            beside the picture while it is on.
//   Deliver  Grading master, HDR10 movie, Broadcast HDR or Custom… (the full
//            What to write card), the file and Render.
//
// The viewer bar keeps a hint (hold B for the original SDR), one View menu,
// Fit and Scopes. The first-run tips (coach_marks.hpp) point at each part once.

#include <QApplication>
#include <QFileInfo>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include <cmath>

#include "aces_engine.hpp"
#include "coach_marks.hpp"
#include "main_window.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"

namespace rudra::app {

QString MainWindow::shown_tab() const {
    if (workspace_ == "simple") return tab_ == "grade" ? QStringLiteral("rec") : tab_;
    return tab_ == "check" ? QStringLiteral("rec") : tab_;
}

void MainWindow::build_simple_parts() {
    // The View menu (Simple's viewer bar): what Full's compare controls do, in one place.
    auto* menu = new QMenu(view_menu_btn_);
    menu->setObjectName("viewMenu");
    struct Item { const char* id; const char* text; std::function<void()> act; };
    const std::vector<Item> items = {
        {"viewHdr", "HDR", [this] {
             if (session_.wipe) session_.toggle_wipe();
             if (session_.show_changes) session_.toggle_changes();
             session_.set_view_layer(0);
             session_.set_show("model");
         }},
        {"viewSdr", "Original SDR", [this] {
             if (session_.wipe) session_.toggle_wipe();
             session_.set_show("baseline");
         }},
        {"viewWipe", "Wipe: SDR left, HDR right", [this] { session_.toggle_wipe(); }},
        {nullptr, nullptr, {}},
        {"viewChanged", "What changed", [this] { session_.toggle_changes(); }},
        {"viewInvented", "Invented pixels", [this] { session_.set_view_layer(session_.view_layer == 3 ? 0 : 3); }},
        {"viewFalseColour", "False colour (nits)", [this] { session_.set_view_layer(session_.view_layer == 1 ? 0 : 1); }},
    };
    for (const auto& it : items) {
        if (!it.id) {
            menu->addSeparator();
            continue;
        }
        QAction* a = menu->addAction(QString::fromUtf8(it.text));
        a->setObjectName(it.id);
        a->setCheckable(true);
        auto act = it.act;
        connect(a, &QAction::triggered, this, [this, act] {
            act();
            sync_checks();
            sync_ui();
        });
    }
    connect(menu, &QMenu::aboutToShow, this, [this, menu] {
        auto check = [menu](const char* id, bool on) {
            if (auto* a = menu->findChild<QAction*>(id)) a->setChecked(on);
        };
        const bool wipe = session_.wipe.has_value(), sdr = session_.show == "baseline";
        check("viewHdr", !wipe && !sdr && session_.view_layer == 0 && !session_.show_changes);
        check("viewSdr", !wipe && sdr);
        check("viewWipe", wipe);
        check("viewChanged", session_.show_changes);
        check("viewInvented", session_.view_layer == 3);
        check("viewFalseColour", session_.view_layer == 1);
    });
    view_menu_btn_->setMenu(menu);

    // Help > Show tips, beside Getting started.
    for (QAction* m : menuBar()->actions()) {
        if (!m->menu() || m->text().remove('&') != "Help") continue;
        auto* tips = new QAction("Show tips", m->menu());
        tips->setObjectName("showTips");
        tips->setStatusTip("The four tips of the first run, on the Simple workspace");
        connect(tips, &QAction::triggered, this, [this] { start_tips(); });
        const auto acts = m->menu()->actions();
        m->menu()->insertAction(acts.size() > 1 ? acts[1] : nullptr, tips);
        break;
    }
    coach_ = new CoachMarks(this);
}

void MainWindow::sync_simple() {
    const bool simple = workspace_ == "simple";
    const QString tab = shown_tab();
    auto vis = [this](const char* id, bool on) {
        if (auto* w = findChild<QWidget*>(id)) w->setVisible(on);
    };
    // The tabs: Look, Check, Deliver, numbered as the steps are.
    tabs_->button("rec")->setText(simple ? QStringLiteral("1  Look") : QStringLiteral("Reconstruct"));
    tabs_->button("check")->setText(QStringLiteral("2  Check"));
    tabs_->button("deliver")->setText(simple ? QStringLiteral("3  Deliver") : QStringLiteral("Deliver"));
    tabs_->button("check")->setVisible(simple);
    tabs_->button("grade")->setVisible(!simple);
    if (probe_btn_) probe_btn_->setVisible(!simple);
    // The sidebar: the shots only.
    vis("collections", !simple);
    vis("probePanelWrap", !simple);
    vis("modelCard", !simple);
    // The viewer bar: the hint, View, Fit, Scopes.
    hold_hint_->setVisible(simple);
    view_menu_btn_->setVisible(simple);
    {
        const QString shown = session_.wipe ? QStringLiteral("Wipe")
                              : session_.show == "baseline" ? QStringLiteral("Original SDR")
                              : session_.view_layer == 1 ? QStringLiteral("False colour")
                              : session_.view_layer == 3 ? QStringLiteral("Invented")
                              : session_.show_changes ? QStringLiteral("What changed")
                                                      : QStringLiteral("HDR");
        view_menu_btn_->setText(QStringLiteral("View: %1  ▾").arg(shown));
    }
    for (QWidget* w : std::initializer_list<QWidget*>{view_mode_, changes_btn_, invented_btn_, view_layer_, zoom_val_, guide_btn_})
        if (w) w->setVisible(!simple);
    if (auto* hud = findChild<QWidget*>("hud"))
        for (auto* c : hud->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly))
            if (c->property("role").toString() == "hud-sep") c->setVisible(!simple);
    compare_bar_->setVisible(!simple);
    vis("pipe", !simple);
    // The empty viewer: one target and the recent shots.
    vis("emptyFull", !simple);
    vis("emptySimple", simple && frames_.empty());
    if (simple && frames_.empty()) fill_empty_recent();
    // Look: the four trims and the monitor on the learned route; without the learned model (1.0's
    // guaranteed path, 29 Oct 2026) the analytic reconstruction's Exposure, Strength and Peak.
    const bool routed_now = aces_routed();
    vis("recAnalytic", !simple || !routed_now);
    for (const char* id : {"recSourceTitle", "recSourceCard"}) vis(id, !simple);
    if (mode_seg_) mode_seg_->setVisible(!simple);
    if (preserve_) preserve_->setVisible(!simple);
    if (auto* t = findChild<QLabel*>("recReconTitle")) t->setText(simple ? QStringLiteral("Look") : QStringLiteral("Reconstruction"));
    for (const char* id : {"acesSourceDisplayRow", "acesSaturationRow", "refRow", "acesTrimsNote"}) vis(id, !simple);
    if (auto* mon = dynamic_cast<Seg*>(findChild<QWidget*>("acesMonitor"))) mon->button("scene")->setVisible(!simple);
    if (auto* h = findChild<QLabel*>("acesTrimsHead")) h->setText(simple ? QStringLiteral("Look") : QStringLiteral("Learned ACES 2.0"));
    if (auto* c = findChild<QPushButton*>("acesTrimsCopy")) c->setText(simple ? QStringLiteral("Copy look") : QStringLiteral("Copy trims"));
    vis("lookAdvanced", simple);
    {   // No learned model on this computer: the analytic look works; the model is an optional download.
        const bool need = simple && !routed_now && !have_aces_package();
        vis("lookDownload", need);
    }
    if (auto* wait = findChild<QLabel*>("lookWaiting")) {
        const bool routed = aces_routed();
        wait->setVisible(simple && !routed);
        if (simple && !routed)
            wait->setText(!have_aces_package() ? QStringLiteral("Analytic reconstruction. The learned ACES 2.0 model (preview) "
                                                                "rebuilds more of the clipped highlights: an optional download, "
                                                                "checked before it is used.")
                          : frames_.empty() ? QStringLiteral("Open a shot: its look is shaped here.")
                          : aces_engine_  ? QStringLiteral("The look appears when the learned ACES 2.0 model is running: ") +
                                               aces_engine_->status_text()
                                          : QStringLiteral("The learned ACES 2.0 model is not loaded. Advanced has the "
                                                           "analytic reconstruction; Deliver > Learned ACES 2.0 master… "
                                                           "sets the model up."));
    }
    // Check.
    if (auto* g = findChild<QPushButton*>("checkGrab")) g->setEnabled(aces_routed() && !frames_.empty());
    if (auto* c = findChild<QPushButton*>("checkCompare")) {
        const QSignalBlocker b(c);
        c->setChecked(ref_compare_);
        c->setEnabled(!ref_stills_.empty());
    }
    sync_check_summary();
    // Deliver: four choices; Custom… is the full card.
    const QString preset = deliver_preset();
    vis("deliverPresets", simple);
    if (auto* p = dynamic_cast<Seg*>(findChild<QWidget*>("deliverPreset"))) {
        p->set_on(preset);
        const bool movie_ok = shot_is_video() || frames_.size() > 1;
        const bool learned = aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready;
        for (const char* k : {"hdr10", "hlg"}) {
            p->button(k)->setEnabled(movie_ok && learned);
            p->button(k)->setToolTip(!learned ? QStringLiteral("Needs the learned ACES 2.0 model (File > Model packages). "
                                                               "Custom\u2026 has the analytic PQ movie")
                                     : movie_ok ? QString()
                                                : QStringLiteral("For a movie or a folder of frames; a single still goes out as a Grading master"));
        }
    }
    vis("deliverFullCards", !simple || preset == "custom");
    for (const char* id : {"renderStart", "renderStartLabel", "renderTimecode", "renderTimecodeLabel", "renderReel",
                           "renderReelLabel", "btnReprocess"})
        vis(id, !simple);
    // The inspector: the tab's page only.
    vis("frameStats", !simple);
    vis("logCard", !simple);
    // The tips wait for a shot on screen.
    if (tips_pending_ && !frames_.empty()) {
        tips_pending_ = false;
        QPointer<MainWindow> self(this);
        QTimer::singleShot(900, this, [self] {
            if (self) self->start_tips();
        });
    }
}

void MainWindow::sync_check_summary() {
    auto set = [this](const char* id, const QString& v) {
        if (auto* l = findChild<QLabel*>(id)) l->setText(v);
    };
    auto nits = [](double v) { return std::isfinite(v) ? QStringLiteral("%1 nits").arg(std::lround(v)) : QStringLiteral("—"); };
    auto pct = [](double v) {
        if (!std::isfinite(v)) return QStringLiteral("—");
        return v > 0 && v < 0.05 ? QStringLiteral("under 0.1 %") : QStringLiteral("%1 %").arg(v, 0, 'f', 1);
    };
    const QString dash = QStringLiteral("—");
    if (frames_.empty()) {
        for (const char* id : {"checkPeak", "checkAverage", "checkAbove", "checkGamut"}) set(id, dash);
        set("checkSource", QStringLiteral("no frame"));
        return;
    }
    if (aces_routed() && parade_ && !parade_->scope().empty()) {   // the learned HDR on screen
        const QcScope& q = parade_->scope();
        set("checkPeak", nits(q.peak_nits));
        set("checkAverage", nits(q.avg_nits));
        set("checkAbove", pct(q.above_white * 100.0));
        set("checkGamut", pct(q.out_p3 * 100.0));
        set("checkSource", QStringLiteral("the HDR on screen"));
    } else if (measure_) {                                          // the analytic frame's measure
        const FrameMetrics m = measure_->frame_metrics();
        set("checkPeak", nits(m.peak_nits));
        set("checkAverage", nits(m.maxfall));
        set("checkAbove", pct(m.above_diffuse_white_pct));
        set("checkGamut", dash);
        set("checkSource", QStringLiteral("measured"));
    } else {
        for (const char* id : {"checkPeak", "checkAverage", "checkAbove", "checkGamut"}) set(id, dash);
        set("checkSource", QStringLiteral("measuring…"));
    }
}

void MainWindow::fill_empty_recent() {
    auto* box = findChild<QWidget*>("emptyRecent");
    if (!box) return;
    QStringList r = recent_sources();
    if (r.size() > 5) r = r.mid(0, 5);
    if (r == empty_recent_ && box->property("filled").toBool()) return;
    empty_recent_ = r;
    box->setProperty("filled", true);
    auto* v = box->layout();
    while (QLayoutItem* it = v->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    int i = 0;
    for (const auto& p : r) {
        const QFileInfo fi(p);
        auto* b = new QPushButton(QStringLiteral("%1   ·   %2").arg(fi.fileName(), fi.absolutePath()), box);
        b->setObjectName(QStringLiteral("emptyRecent%1").arg(i++));
        b->setProperty("role", "recent");
        b->setFlat(true);
        b->setToolTip(p);
        b->setEnabled(fi.exists());
        connect(b, &QPushButton::clicked, this, [this, p] { open_paths({p}); });
        v->addWidget(b);
    }
    box->setVisible(!r.isEmpty());
    if (auto* none = findChild<QWidget*>("emptyRecentNone")) none->setVisible(r.isEmpty());
}

QString MainWindow::deliver_preset() const {
    if (deliver_custom_) return QStringLiteral("custom");
    const DeliverChoice& c = deliver_;
    if (c.ai_fill) return QStringLiteral("custom");
    // The grading master is the learned one when the model runs, the analytic EXR (32-bit, ACES 2065-1) otherwise.
    if ((c.output == "still" || c.output == "sequence") && (c.image == "exr16" || c.image == "exr32") && c.gamut == "ap0" &&
        (c.source == "aces" || c.source == "analytic"))
        return QStringLiteral("master");
    if (c.source != "aces") return QStringLiteral("custom");
    if (c.output == "movie" && c.bits == 10 && c.codec == "hevc10" && c.encoding == "pq" && c.pq_peak == 1000)
        return QStringLiteral("hdr10");
    if (c.output == "movie" && c.bits == 10 && c.codec == "prores422hq" && c.encoding == "hlg") return QStringLiteral("hlg");
    return QStringLiteral("custom");
}

void MainWindow::apply_deliver_preset(const QString& key) {
    if (key == "custom") {
        deliver_custom_ = true;
        sync_ui();
        return;
    }
    const bool many = shot_is_video() || frames_.size() > 1;
    const bool learned = aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready;
    if ((key == "hdr10" || key == "hlg") && (!many || !learned)) return;   // a single still, or no learned model: no movie
    deliver_custom_ = false;
    set_deliver("source", learned ? "aces" : "analytic");
    if (key == "master") {
        set_deliver("output", many ? "sequence" : "still");
        if (deliver_.image == "dpx10") set_deliver("image", "exr16");
        set_deliver("gamut", "ap0");
        log("Deliver: Grading master — ACES 2065-1 EXR");
    } else if (key == "hdr10") {
        set_deliver("output", "movie");
        set_deliver("bits", "10");
        set_deliver("codec", "hevc10");
        set_deliver("encoding", "pq");
        set_deliver("pq_peak", "1000");
        log("Deliver: HDR10 movie — HEVC 10-bit, PQ 1000 nits");
    } else if (key == "hlg") {
        set_deliver("output", "movie");
        set_deliver("bits", "10");
        set_deliver("codec", "prores422hq");
        set_deliver("encoding", "hlg");
        log("Deliver: Broadcast HDR — HLG, ProRes 422 HQ");
    }
    if (deliver_.ai_fill) set_deliver("ai_fill", "0");
    sync_ui();
}

// ---------------------------------------------------------------------------
// The tips

void MainWindow::show_tips_once() {
    QSettings st;
    if (st.value("help/tipsDone", false).toBool()) return;
    if (st.value("help/tipsWelcomed", false).toBool()) {   // welcomed before: the tips with the next shot
        tips_pending_ = true;
        sync_ui();
        return;
    }
    QPointer<MainWindow> self(this);
    QTimer::singleShot(600, this, [self] {
        if (self) self->show_welcome_tips();
    });
}

void MainWindow::show_welcome_tips() {
    if (!coach_) return;
    coach_->welcome_go = [this] {
        QSettings().setValue("help/tipsWelcomed", true);
        if (workspace_ != "simple") set_workspace("simple");
        if (!frames_.empty()) start_tips();
        else {
            tips_pending_ = true;
            sync_ui();
        }
    };
    coach_->welcome_skip = [this] {
        QSettings().setValue("help/tipsWelcomed", true);
        QSettings().setValue("help/tipsDone", true);
        tips_pending_ = false;
    };
    coach_->welcome(QStringLiteral("SDR in, HDR out — in three steps"),
                    QStringLiteral("<b>1 Look</b> — open a shot and shape its HDR.<br>"
                                   "<b>2 Check</b> — read it on the scopes.<br>"
                                   "<b>3 Deliver</b> — pick what you need and render.<br><br>"
                                   "The first time a shot is open, four tips point at each part. "
                                   "Advanced shows every control; Help › Show tips brings the tips back."),
                    QStringLiteral("I know RUDRA — no tips"), QStringLiteral("Got it"));
}

void MainWindow::start_tips() {
    if (!coach_) return;
    tips_pending_ = false;
    if (workspace_ != "simple") set_workspace("simple");
    if (!session_.rail_right) run("rail-right");
    show_tab("rec");
    QApplication::sendPostedEvents(nullptr, QEvent::LayoutRequest);
    QWidget* trims = findChild<QWidget*>("acesTrimsCard");
    if (!trims || !trims->isVisible()) trims = pages_.empty() ? nullptr : pages_[0];
    using Step = CoachMarks::Step;
    std::vector<Step> steps = {
        {static_cast<QWidget*>(viewer_stack_), QStringLiteral("This is your HDR"),
         QStringLiteral("RUDRA rebuilds the light the SDR clipped. Hold <b>B</b> to see the original SDR; "
                        "<b>View ▾</b> has a wipe and what changed.")},
        {trims, QStringLiteral("Shape it"),
         QStringLiteral("Strength, Peak, Exposure and Contrast trim the look. <b>Copy look</b> puts it on your next shot.")},
        {scopes_dock_, QStringLiteral("Check it"),
         QStringLiteral("The <b>Check</b> tab measures the frame and puts the scopes beside the picture. "
                        "Grab a still to compare shots.")},
        {findChild<QWidget*>("deliverPresets"), QStringLiteral("Deliver"),
         QStringLiteral("Pick a grading master, an HDR10 movie or broadcast HLG, then <b>Render</b>. "
                        "Custom… has every format.")},
    };
    coach_->step_entered = [this](int i) {
        show_tab(i == 2 ? QStringLiteral("check") : i == 3 ? QStringLiteral("deliver") : QStringLiteral("rec"));
        QApplication::sendPostedEvents(nullptr, QEvent::LayoutRequest);
    };
    auto finish = [this] {
        QSettings().setValue("help/tipsWelcomed", true);
        QSettings().setValue("help/tipsDone", true);
        show_tab("rec");
    };
    coach_->done = finish;
    coach_->skipped = finish;
    coach_->tips(std::move(steps));
}

}  // namespace rudra::app
