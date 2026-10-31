// The main window's layout: the "Pro direction" boards (RUDRA Studio Desktop
// canvas, "Pro direction: main window").
//
// A unified toolbar (the sidebar toggle, the shot's name, the Reconstruct /
// Grade / Deliver tabs in the middle, the machine pill, Probe, Scopes, the
// inspector and Export on the right); the sidebar library (open by path, the
// shots, collections from the clipping lane, the probe card and the model
// card); the viewer on its neutral surround with its compare bar and badges,
// the colour pipeline under it and the timeline (timecode, round transport,
// the filmstrip with its clipping lane); and the inspector's grouped cards.
// Every control the Studio page names keeps that page's id as its object
// name, so every action, handler and read-out behaves as before.

#include <QCheckBox>
#include <QSettings>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <initializer_list>
#include <tuple>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>

#include "main_window.hpp"
#include "genfill_review.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "source_panel.hpp"
#include "widgets.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {

QLabel* label(const QString& text, const QString& id = {}, const QString& role = {}, QWidget* parent = nullptr) {
    auto* l = new QLabel(text, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

QVBoxLayout* column(QWidget* w, int margin = 0, int spacing = 0) {
    auto* v = new QVBoxLayout(w);
    v->setContentsMargins(margin, margin, margin, margin);
    v->setSpacing(spacing);
    return v;
}

QWidget* styled(const QString& id = {}, const QString& role = {}) {
    auto* w = new QWidget;
    if (!id.isEmpty()) w->setObjectName(id);
    if (!role.isEmpty()) w->setProperty("role", role);
    w->setAttribute(Qt::WA_StyledBackground, true);
    return w;
}

// A .ctl block: padded, spaced.
QWidget* ctl(QWidget* parent = nullptr) {
    auto* w = new QWidget(parent);
    w->setProperty("role", "ctl");
    return w;
}

// A .prow / .field / .ro: a key on the left, a value on the right.
QWidget* row(const QString& key, QLabel* value, const QString& role = "prow") {
    auto* w = new QWidget;
    w->setProperty("role", role);
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 3, 0, 3);
    h->setSpacing(6);
    h->addWidget(label(key, {}, "key"));
    h->addStretch(1);
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    h->addWidget(value);
    return w;
}

}  // namespace

void MainWindow::build_ui(bool with_viewer) {
    auto* app = styled("app");
    auto* v = column(app);
    v->addWidget(build_toolbar());
    auto* body = styled("body");
    auto* h = new QHBoxLayout(body);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);
    rail_left_ = build_left_rail();
    rail_right_ = build_right_rail();
    h->addWidget(rail_left_);
    h->addWidget(build_centre(with_viewer), 1);
    // The scopes beside the viewer (29 Oct 2026): their own column, shown or hidden
    // with the Scopes button on the viewer bar, the toolbar's scope icon or S.
    {
        scopes_dock_ = styled("scopesDock", "rail-right");
        scopes_dock_->setFixedWidth(300);
        auto* dv = column(scopes_dock_);
        auto* scroll = new QScrollArea(scopes_dock_);
        scroll->setObjectName("scopesScroll");
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* inner = new QWidget(scroll);
        inner->setObjectName("scopesInner");
        auto* iv = column(inner, 12, 12);
        scopes_->setParent(inner);
        iv->addWidget(scopes_);
        iv->addStretch(1);
        scroll->setWidget(inner);
        dv->addWidget(scroll, 1);
        h->addWidget(scopes_dock_);
    }
    h->addWidget(rail_right_);
    v->addWidget(body, 1);
    setCentralWidget(app);
    menuBar()->setFixedHeight(28);
}

QWidget* MainWindow::build_toolbar() {
    auto* bar = styled("toolbar");
    bar->setFixedHeight(52);
    auto* h = new QHBoxLayout(bar);
    h->setContentsMargins(12, 0, 14, 0);
    h->setSpacing(10);
    i_media_ = new IconButton("iMedia", IconButton::Glyph::Sidebar, "Show or hide the sidebar", bar);
    i_media_->setFixedSize(32, 28);
    connect(i_media_, &QToolButton::clicked, this, [this] { run("rail-left"); });
    h->addWidget(i_media_);
    auto* title = new QWidget(bar);
    auto* tv = column(title, 0, 0);
    tv->addStretch(1);
    tv->addWidget(label("No shot open", "shotTitle", {}, title));
    tv->addWidget(label("Drop frames, or open a folder", "shotSub", {}, title));
    tv->addStretch(1);
    title->setMinimumWidth(200);
    h->addWidget(title);
    h->addStretch(1);
    // Check is Simple's (29 Oct 2026): Look, Check, Deliver; Full keeps Reconstruct, Grade, Deliver.
    tabs_ = new Seg("itabs", {{"rec", "Reconstruct"}, {"grade", "Grade"}, {"check", "Check"}, {"deliver", "Deliver"}}, bar);
    tabs_->button("rec")->setObjectName("tabRec");
    tabs_->button("grade")->setObjectName("tabGrade");
    tabs_->button("check")->setObjectName("tabCheck");
    tabs_->button("check")->hide();
    tabs_->button("deliver")->setObjectName("tabDeliver");
    tabs_->clicked = [this](const QString& k) {
        if (!session_.rail_right) run("rail-right");   // a tab brings its inspector back
        show_tab(k);
    };
    tabs_->setFixedHeight(30);
    h->addWidget(tabs_, 0, Qt::AlignVCenter);
    h->addStretch(1);

    auto* pill = styled("devicePill");
    pill->setFixedHeight(24);
    auto* ph = new QHBoxLayout(pill);
    ph->setContentsMargins(10, 0, 10, 0);
    ph->setSpacing(6);
    lamp_ = label("", "lamp", "lamp", pill);
    lamp_->setFixedSize(6, 6);
    device_ = label("\u2014", "device", "pill", pill);
    ph->addWidget(lamp_, 0, Qt::AlignVCenter);
    ph->addWidget(device_, 0, Qt::AlignVCenter);
    h->addWidget(pill, 0, Qt::AlignVCenter);

    ws_ = new Seg("wsswitch", {{"simple", "Simple"}, {"full", "Full"}}, bar);
    ws_->button("simple")->setObjectName("wsSimple");
    ws_->button("full")->setObjectName("wsFull");
    ws_->button("simple")->setToolTip("Look, Check, Deliver: the controls most shots need");
    ws_->button("full")->setToolTip("Everything");
    ws_->clicked = [this](const QString& k) { set_workspace(k); };
    ws_->setFixedHeight(26);
    h->addWidget(ws_, 0, Qt::AlignVCenter);

    probe_btn_ = new QPushButton("Probe", bar);
    probe_btn_->setObjectName("probeBtn");
    probe_btn_->setCheckable(true);
    probe_btn_->setToolTip("Read one pixel: baseline, RUDRA, and whether the SDR clipped there");
    connect(probe_btn_, &QPushButton::clicked, this, [this](bool on) {
        probe_on_ = on;
        if (!on) probe_pixel(std::nullopt);
    });
    probe_btn_->setFixedHeight(28);
    h->addWidget(probe_btn_, 0, Qt::AlignVCenter);
    i_scopes_ = new IconButton("iScopes", IconButton::Glyph::Scopes, "Scopes", bar);
    i_inspector_ = new IconButton("iInspector", IconButton::Glyph::Inspector, "Inspector", bar);
    auto* i_export = new IconButton("iExport", IconButton::Glyph::Export, "Export\u2026", bar);
    auto* i_help = new IconButton("iHelp", IconButton::Glyph::Help, "Keyboard shortcuts", bar);
    for (auto* b : {i_scopes_, i_inspector_, i_export, i_help}) {
        b->setFixedSize(32, 28);
        h->addWidget(b);
    }
    connect(i_inspector_, &QToolButton::clicked, this, [this] { run("rail-right"); });
    connect(i_scopes_, &QToolButton::clicked, this, [this] { run("scopes"); });
    connect(i_export, &QToolButton::clicked, this, [this] { open_export_sheet(); });
    connect(i_help, &QToolButton::clicked, this, [this] { run("shortcuts"); });
    return bar;
}

QWidget* MainWindow::build_left_rail() {
    auto* rail = styled("railLeft", "rail-left");
    rail->setFixedWidth(240);
    auto* v = column(rail, 0, 4);
    v->setContentsMargins(10, 12, 10, 12);

    // Open by path: the board's search field.
    auto* seq = new QWidget(rail);
    auto* sh = new QHBoxLayout(seq);
    sh->setContentsMargins(0, 0, 0, 0);
    sh->setSpacing(6);
    seq_path_ = new QLineEdit(seq);
    seq_path_->setObjectName("seqPath");
    seq_path_->setPlaceholderText("Open a folder of frames\u2026");
    seq_path_->setToolTip("A folder of frames, read where it sits: nothing is copied.");
    auto* seq_open = new QPushButton("Open", seq);
    seq_open->setObjectName("seqOpen");
    seq_open->setToolTip("Open the shot at that path");
    auto open_path = [this] {
        const QString p = seq_path_->text().trimmed();
        if (!p.isEmpty()) open_source(p);
    };
    connect(seq_open, &QPushButton::clicked, this, open_path);
    connect(seq_path_, &QLineEdit::returnPressed, this, open_path);
    sh->addWidget(seq_path_, 1);
    sh->addWidget(seq_open);
    v->addWidget(seq);
    auto* note = label("", "seqNote", "seq-note", rail);
    note->hide();
    v->addWidget(note);

    auto* shots_head = new QWidget(rail);
    auto* shh = new QHBoxLayout(shots_head);
    shh->setContentsMargins(0, 0, 8, 0);
    shh->addWidget(label("Shots", {}, "section"));
    shh->addStretch(1);
    shot_count_ = label("0", "shotCount", {}, shots_head);
    shh->addWidget(shot_count_, 0, Qt::AlignBottom);
    v->addWidget(shots_head);

    auto* dz = new QFrame(rail);
    dz->setObjectName("dropzone");
    auto* dv = column(dz, 12, 4);
    dv->addWidget(label("Drop footage here", {}, "drop-b"), 0, Qt::AlignHCenter);
    dv->addWidget(label("a frame or a folder of frames", {}, "drop-t"), 0, Qt::AlignHCenter);
    dv->addWidget(label("or", {}, "drop-or"), 0, Qt::AlignHCenter);
    auto* browse = new QPushButton("Open\u200a\u2026", dz);
    browse->setObjectName("browseBtn");
    browse->setProperty("role", "primary");
    connect(browse, &QPushButton::clicked, this, [this] { run("open"); });
    dv->addWidget(browse, 0, Qt::AlignHCenter);
    v->addWidget(dz);

    auto* shots = styled("shots", "shots");
    column(shots, 0, 2);
    v->addWidget(shots);
    frames_empty_ = label("Nothing open yet", "framesEmpty", "empty-rail", rail);
    v->addWidget(frames_empty_);

    // Collections: the frames the clipping lane has marked.
    auto* collections = new QWidget(rail);
    collections->setObjectName("collections");
    auto* colv = column(collections, 0, 4);
    colv->addWidget(label("Collections", {}, "section"));
    const struct { const char* id; const char* dot; const char* text; } cols[] = {
        {"colHighlights", "dot-gold", "Clipped highlights"},
        {"colShadows", "dot-violet", "Crushed shadows"},
        {"colDelivered", "dot-ok", "Delivered"}};
    for (const auto& c : cols) {
        auto* r = styled({}, "collection");
        auto* rh = new QHBoxLayout(r);
        rh->setContentsMargins(8, 5, 8, 5);
        rh->setSpacing(10);
        auto* dot = label({}, {}, c.dot, r);
        dot->setFixedSize(8, 8);
        rh->addWidget(dot, 0, Qt::AlignVCenter);
        rh->addWidget(label(c.text, {}, {}, r));
        rh->addStretch(1);
        rh->addWidget(label("0", c.id, "count", r));
        colv->addWidget(r);
    }
    v->addWidget(collections);
    v->addStretch(1);

    // The probe, as a card.
    auto* probe = styled("probePanelWrap", "probepanel");
    auto* pv = column(probe, 0, 0);
    pv->setContentsMargins(12, 10, 12, 10);
    auto* phead = new QWidget(probe);
    auto* phh = new QHBoxLayout(phead);
    phh->setContentsMargins(0, 0, 0, 4);
    phh->addWidget(label("Probe", {}, "card-head"));
    phh->addStretch(1);
    phh->addWidget(label("\u2014", "probeXY", {}, phead));
    pv->addWidget(phead);
    auto* body = styled("probePanel", "pbody");
    auto* bv = column(body, 0, 0);
    auto* nits = new QWidget(body);
    auto* nh = new QHBoxLayout(nits);
    nh->setContentsMargins(0, 0, 0, 0);
    nh->setSpacing(5);
    auto* nit = label("\u2014", "probeNits", "pnit", nits);
    nit->setProperty("state", "idle");
    nh->addWidget(nit, 0, Qt::AlignBottom);
    auto* unit = label("nits", "probeNitsUnit", "pnit-unit", nits);
    unit->hide();
    nh->addWidget(unit, 0, Qt::AlignBottom);
    nh->addStretch(1);
    bv->addWidget(nits);
    auto* delta = label("pick a pixel with Probe, or hold Alt", "probeDelta", "pdelta", body);
    delta->setProperty("state", "idle");
    delta->setWordWrap(true);
    bv->addWidget(delta);
    bv->addSpacing(4);
    for (const auto& [k, id] : std::vector<std::pair<QString, QString>>{
             {"Source", "probeSrc"}, {"Baseline", "probeBase"}, {"RUDRA", "probeModel"}, {"Masks", "probeMask"}})
        bv->addWidget(row(k, label("\u2014", id, "value")));
    pv->addWidget(body);
    v->addWidget(probe);

    // The model card.
    auto* model = styled("modelCard");
    auto* mv = column(model, 0, 2);
    mv->setContentsMargins(12, 10, 12, 10);
    mv->addWidget(label("Model", "modelCardHead", {}, model));
    ckpt_ = label("no model", "ckpt", "pill", model);
    mv->addWidget(ckpt_);
    mv->addWidget(label("File > Model packages\u2026", "modelCardState", {}, model));
    v->addWidget(model);
    return rail;
}

QWidget* MainWindow::build_centre(bool with_viewer) {
    auto* centre = styled("centre", "centre");
    auto* v = column(centre);

    // ── over the picture: the HDR badge, the compare bar, the MaxCLL badge ──
    // (a strip on the surround: widgets cannot float over the viewer's own
    // swapchain window on every platform)
    auto* tools = styled("viewerTools", "vtools");
    tools->setFixedHeight(52);
    auto* th = new QHBoxLayout(tools);
    th->setContentsMargins(16, 8, 16, 8);
    th->setSpacing(10);
    auto* hdr = styled("hdrBadge", "badge");
    hdr->setFixedHeight(26);
    auto* hh = new QHBoxLayout(hdr);
    hh->setContentsMargins(12, 0, 12, 0);
    hh->setSpacing(7);
    auto* dot = label({}, "hdrDot", "dot", hdr);
    dot->setFixedSize(7, 7);
    dot->setProperty("state", "sdr");
    hh->addWidget(dot, 0, Qt::AlignVCenter);
    hh->addWidget(label("SDR", "hdrBadgeText", {}, hdr));
    hh->addWidget(label("display not measured", "hdrBadgeSub", {}, hdr));
    // Badges in two equal sides, so the compare bar sits in the middle.
    auto* left = new QWidget(tools);
    auto* lh = new QHBoxLayout(left);
    lh->setContentsMargins(0, 0, 0, 0);
    lh->addWidget(hdr);
    lh->addStretch(1);
    th->addWidget(left, 1);

    auto* hud = styled("hud");
    hud->setFixedHeight(38);
    auto* hudh = new QHBoxLayout(hud);
    hudh->setContentsMargins(5, 4, 5, 4);
    hudh->setSpacing(8);
    // Simple's viewer bar (29 Oct 2026): the hint and one View menu in place of the compare controls.
    hold_hint_ = label("Hold B for the original SDR", "holdHint", "vstatic", hud);
    hold_hint_->setToolTip("Hold B to see the SDR you opened; let go for the HDR");
    hold_hint_->hide();
    hudh->addWidget(hold_hint_);
    view_menu_btn_ = new QPushButton("View \u25be", hud);
    view_menu_btn_->setObjectName("viewMenuBtn");
    view_menu_btn_->setToolTip("What the viewer shows: the HDR, the original SDR, a wipe, what changed, false colour");
    view_menu_btn_->hide();
    hudh->addWidget(view_menu_btn_);
    view_mode_ = new Seg("viewMode", {{"model", "RUDRA"}, {"baseline", "Baseline"}, {"#wipeBtn", "Wipe"}}, hud, false);
    view_mode_->button("model")->setToolTip("The reconstruction (1)");
    view_mode_->button("baseline")->setToolTip("The analytic baseline (2)");
    view_mode_->button("#wipeBtn")->setToolTip("Wipe: baseline left, RUDRA right (W)");
    view_mode_->clicked = [this](const QString& k) {
        if (k == "#wipeBtn") run("wipe");
        else session_.set_show(k.toStdString());
    };
    hudh->addWidget(view_mode_);
    changes_btn_ = new QPushButton("Changes", hud);
    changes_btn_->setObjectName("changesBtn");
    changes_btn_->setCheckable(true);
    changes_btn_->setToolTip("Tint what RUDRA changed: amber brighter, blue darker than the baseline");
    connect(changes_btn_, &QPushButton::clicked, this, [this] { session_.toggle_changes(); });
    hudh->addWidget(changes_btn_);
    invented_btn_ = new QPushButton("Invented", hud);
    invented_btn_->setObjectName("inventedBtn");
    invented_btn_->setCheckable(true);
    invented_btn_->setToolTip("Invented-pixel map: magenta where RUDRA made up values the SDR clipped or crushed, "
                              "cyan where it reinterpreted SDR detail");
    connect(invented_btn_, &QPushButton::clicked, this,
            [this] { session_.set_view_layer(session_.view_layer == 3 ? 0 : 3); });
    hudh->addWidget(invented_btn_);
    hudh->addWidget(styled({}, "hud-sep"));
    view_layer_ = new Seg("viewLayer", {{"0", "Image"}, {"1", "False colour"}, {"2", "Difference"}}, hud);
    view_layer_->button("0")->setToolTip("The composed picture");
    view_layer_->button("1")->setToolTip("Luminance zones, in nits");
    view_layer_->button("2")->setToolTip("What the network changed, and where it did not");
    view_layer_->clicked = [this](const QString& k) { session_.set_view_layer(k.toInt()); };
    hudh->addWidget(view_layer_);
    hudh->addWidget(styled({}, "hud-sep"));
    zoom_seg_ = new Seg("zoomSeg", {{"fit", "Fit"}, {"actual", "100%"}}, hud);
    zoom_seg_->clicked = [this](const QString& k) { run(k == "actual" ? "zoom-actual" : "zoom-fit"); };
    hudh->addWidget(zoom_seg_);
    zoom_val_ = label("100%", "zoomVal", "vstatic", hud);
    zoom_val_->setToolTip("Scroll to zoom, middle-drag to pan, double-click to fit");
    hudh->addWidget(zoom_val_);
    scopes_btn_ = new QPushButton("Scopes", hud);
    scopes_btn_->setObjectName("scopesBtn");
    scopes_btn_->setCheckable(true);
    scopes_btn_->setToolTip("Show or hide the scopes beside the viewer: waveform, histogram, vectorscope, RGB parade");
    connect(scopes_btn_, &QPushButton::clicked, this, [this] {
        if (workspace_ == "simple") {   // Simple: shown on Check, hidden elsewhere, unless the button says otherwise
            simple_scopes_ = !scopes_dock_->isVisible();
            sync_ui();
        } else {
            run("scopes");
        }
    });
    hudh->addWidget(scopes_btn_);
    guide_btn_ = new QPushButton("Guides", hud);
    guide_btn_->setObjectName("guideBtn");
    guide_btn_->setCheckable(true);
    guide_btn_->setToolTip("Framing guides, 90% and 80%");
    connect(guide_btn_, &QPushButton::clicked, this, [this](bool on) {
        guides_on_ = on;
#ifdef RUDRA_APP_VIEWER
        if (viewer_) {
            auto g = viewer_->guides();
            g.action_safe = g.title_safe = on;
            viewer_->set_guides(g);
        }
#endif
        sync_checks();
    });
    hudh->addWidget(guide_btn_);
    th->addWidget(hud);
    auto* right = new QWidget(tools);
    auto* rh = new QHBoxLayout(right);
    rh->setContentsMargins(0, 0, 0, 0);
    rh->addStretch(1);
    th->addWidget(right, 1);
    auto* cll = styled("cllBadgeWrap", "badge");
    cll->setFixedHeight(26);
    auto* ch = new QHBoxLayout(cll);
    ch->setContentsMargins(12, 0, 12, 0);
    ch->addWidget(label("MaxCLL \u2014", "cllBadge", {}, cll));
    rh->addWidget(cll);
    v->addWidget(tools);

    // ── the compare line: what RUDRA changed, and how much of it this view shows ──
    compare_bar_ = styled("compareBar");
    compare_bar_->setFixedHeight(24);
    auto* cb = new QHBoxLayout(compare_bar_);
    cb->setContentsMargins(16, 0, 16, 4);
    cb->setSpacing(10);
    cb->addStretch(1);
    compare_line_ = label({}, "compareLine", {}, compare_bar_);
    cb->addWidget(compare_line_);
    compare_fit_ = new QPushButton(compare_bar_);
    compare_fit_->setObjectName("compareFit");
    compare_fit_->setToolTip("Raise the view peak so the change is not clipped away (the master is unchanged)");
    compare_fit_->hide();
    connect(compare_fit_, &QPushButton::clicked, this, [this] { session_.peak_input(compare_fit_ev_); });
    cb->addWidget(compare_fit_);
    cb->addStretch(1);
    compare_line_->setText(" ");
    v->addWidget(compare_bar_);

    // ── the viewer ──
    viewer_stack_ = new QStackedWidget(centre);
    viewer_stack_->setObjectName("viewer");
    auto* empty = new QWidget(viewer_stack_);
    empty->setObjectName("empty");
    auto* ev = column(empty, 0, 6);
    ev->addStretch(1);
    auto* empty_full = new QWidget(empty);
    empty_full->setObjectName("emptyFull");
    auto* efv = column(empty_full, 0, 6);
    efv->addWidget(label("Drop an SDR frame", {}, "empty-b"), 0, Qt::AlignHCenter);
    efv->addWidget(label("PNG, JPEG or TIFF \u2014 one, or a whole sequence", {}, "empty-t"), 0, Qt::AlignHCenter);
    ev->addWidget(empty_full, 0, Qt::AlignHCenter);
    {   // Simple's empty viewer (29 Oct 2026): one target, a button, the recent shots.
        auto* es = styled("emptySimple", "empty-target");
        es->setFixedWidth(440);
        auto* esv = column(es, 28, 10);
        esv->addWidget(label("Open a shot", "emptyTitle", "empty-title", es), 0, Qt::AlignHCenter);
        auto* sub = label("Drop a frame, a folder of frames or a movie here", "emptySub", "empty-t", es);
        sub->setAlignment(Qt::AlignHCenter);
        sub->setWordWrap(true);
        esv->addWidget(sub);
        auto* choose = new QPushButton("Choose\u2026", es);
        choose->setObjectName("emptyChoose");
        choose->setProperty("role", "primary");
        choose->setMinimumWidth(140);
        connect(choose, &QPushButton::clicked, this, [this] { run("open"); });
        esv->addWidget(choose, 0, Qt::AlignHCenter);
        esv->addSpacing(8);
        auto* recent = new QWidget(es);
        recent->setObjectName("emptyRecent");
        column(recent, 0, 2);
        esv->addWidget(recent);
        auto* none = label("Your recent shots will appear here.", "emptyRecentNone", "empty-t", es);
        none->setAlignment(Qt::AlignHCenter);
        esv->addWidget(none);
        es->hide();
        ev->addWidget(es, 0, Qt::AlignHCenter);
    }
    ev->addStretch(1);
    viewer_stack_->addWidget(empty);
#ifdef RUDRA_APP_VIEWER
    if (with_viewer) {
        viewer_ = new ViewerWindow();
        auto* container = QWidget::createWindowContainer(viewer_, viewer_stack_);
        container->setObjectName("viewerHost");
        container->setFocusPolicy(Qt::StrongFocus);
        viewer_stack_->addWidget(container);
    }
#else
    (void)with_viewer;
#endif
    v->addWidget(viewer_stack_, 1);
    v->addWidget(build_pipe());

    // ── the timeline ──
    auto* transport = styled("transport", "transport");
    transport->setFixedHeight(128);
    auto* tv = column(transport, 0, 8);
    tv->setContentsMargins(16, 10, 16, 10);
    auto* top = new QWidget(transport);
    auto* trh = new QHBoxLayout(top);
    trh->setContentsMargins(0, 0, 0, 0);
    trh->setSpacing(14);
    tc_ = label("01:00:00:00", "tc", "tc", top);
    trh->addWidget(tc_);
    auto* seg = new QWidget(top);
    auto* sgh = new QHBoxLayout(seg);
    sgh->setContentsMargins(8, 0, 0, 0);
    sgh->setSpacing(6);
    btn_prev_ = new IconButton("btnPrev", IconButton::Glyph::Prev, "Previous frame", seg);
    btn_play_ = new IconButton("btnPlay", IconButton::Glyph::Play, "Play", seg);
    btn_next_ = new IconButton("btnNext", IconButton::Glyph::Next, "Next frame", seg);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) {
        b->setFixedSize(30, 30);
        b->setProperty("role", "tbtn");
        sgh->addWidget(b);
    }
    connect(btn_prev_, &QToolButton::clicked, this, [this] {
        play_.stop();
        run("prev");
    });
    connect(btn_next_, &QToolButton::clicked, this, [this] {
        play_.stop();
        run("next");
    });
    connect(btn_play_, &QToolButton::clicked, this, [this] { run("play"); });
    trh->addWidget(seg);
    trh->addStretch(1);
    src_info_ = label("\u2014", "srcInfo", "srcinfo", top);
    trh->addWidget(src_info_);
    tv->addWidget(top);
    scrub_ = new ScrubBar(transport);
    scrub_->seek = [this](double f) {
        if (frames_.size() < 2) return;
        play_.stop();
        step_to(int(std::lround(f * double(frames_.size() - 1))));
    };
    tv->addWidget(scrub_, 1);
    v->addWidget(transport);
    return centre;
}

QWidget* MainWindow::build_right_rail() {
    auto* rail = styled("railRight", "rail-right");
    rail->setFixedWidth(320);
    auto* rv = column(rail);
    auto* scroll = new QScrollArea(rail);
    scroll->setObjectName("railscroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* col = styled("railColumn");
    auto* v = column(col, 0, 12);
    v->setContentsMargins(14, 14, 14, 14);
    auto card = [](QWidget* parent, int pad = 12, int spacing = 12) {
        auto* c = styled({}, "card");
        c->setParent(parent);
        auto* cv = column(c, pad, spacing);
        return std::pair{c, cv};
    };

    // The tab's page: the three live in one column and only the tab's shows,
    // so the cards below follow its height.
    panels_ = styled("ipanels");
    panels_->setParent(col);
    auto* pages = column(panels_);

    // Reconstruct
    auto* rec = ctl();
    auto* recv = column(rec, 0, 12);
    // The learned ACES 2.0 master's trims (28 Oct 2026, core/aces_v2_trims.hpp):
    // shown while the viewer runs the model, the same in the viewer and the renders.
    {
        auto [tc, tcv] = card(rec, 12, 6);
        tc->setObjectName("acesTrimsCard");
        auto* head = new QWidget(tc);
        auto* hh = new QHBoxLayout(head);
        hh->setContentsMargins(0, 0, 0, 0);
        hh->addWidget(label("Learned ACES 2.0", "acesTrimsHead", "card-head"));
        hh->addStretch(1);
        auto* reset = new QPushButton("Reset", head);
        reset->setObjectName("acesTrimsReset");
        reset->setToolTip("Back to the model's own master: strength 100 %, saturation 100 %, no peak limit");
        connect(reset, &QPushButton::clicked, this, [this] { set_aces_trims({}); });
        hh->addWidget(reset);
        tcv->addWidget(head);
        // What the monitor is shown: the master's scene light, or an ACES 2.0 output in display light.
        tcv->addWidget(label("Monitor", {}, "note", tc));
        auto* mon = new Seg("acesMonitor", {{"scene", "Scene"}, {"sdr100", "SDR 100"}, {"hdr1000", "HDR 1000"}}, tc);
        mon->setToolTip("What the viewer shows: the master's scene light (RUDRA's view, to the view peak), or the ACES 2.0 "
                        "output a deliverable gets, SDR 100 nits Rec.709 or HDR 1000 nits P3 D65 on Rec.2100 PQ, in "
                        "display light (an HDR swapchain shows the nits as they are)");
        mon->clicked = [this](const QString& k) { set_aces_monitor(k); };
        tcv->addWidget(mon);
        // What the SDR was made for: the model reads Rec.709 on BT.1886 (2.4).
        auto* sdrow = new QWidget(tc);
        sdrow->setObjectName("acesSourceDisplayRow");
        auto* sdv = column(sdrow, 0, 6);
        sdv->addWidget(label("Source display", {}, "note", sdrow));
        auto* sd = new Seg("acesSourceDisplay", {{"bt1886", "BT.1886 2.4"}, {"srgb", "sRGB"}, {"gamma22", "Gamma 2.2"}}, sdrow);
        sd->setToolTip("The display the SDR was mastered for. The model reads every frame as the ACES 2.0 SDR 100 nits "
                       "(Rec.709) output on a BT.1886 2.4 display; an sRGB or gamma 2.2 master is re-encoded to BT.1886 "
                       "code values first, so the same display light reaches the model. Broadcast Rec.709: BT.1886");
        sd->clicked = [this](const QString& k) { set_aces_source_display(k); };
        sdv->addWidget(sd);
        tcv->addWidget(sdrow);
        // Reference stills: grab the frame, then compare it in the baseline's place.
        {
            auto* row = new QWidget(tc);
            row->setObjectName("refRow");
            auto* rh = new QHBoxLayout(row);
            rh->setContentsMargins(0, 4, 0, 0);
            rh->setSpacing(6);
            auto* grab = new QPushButton("Grab still", row);
            grab->setObjectName("refGrab");
            grab->setToolTip("Keep the frame on screen as a reference still (Ctrl+Alt+G)");
            connect(grab, &QPushButton::clicked, this, [this] { run("still-grab"); });
            auto* box = new QComboBox(row);
            box->setObjectName("refStills");
            box->setToolTip("The reference still to compare with");
            box->setEnabled(false);
            connect(box, &QComboBox::activated, this, [this](int i) { set_reference_compare(ref_compare_, i); });
            auto* cmp = new QPushButton("Compare", row);
            cmp->setObjectName("refCompare");
            cmp->setCheckable(true);
            cmp->setEnabled(false);
            cmp->setToolTip("Show the still in the baseline's place: Baseline, Wipe and B compare it with this frame (Ctrl+Alt+W)");
            connect(cmp, &QPushButton::toggled, this, [this](bool on) { set_reference_compare(on); });
            rh->addWidget(grab);
            rh->addWidget(box, 1);
            rh->addWidget(cmp);
            tcv->addWidget(row);
            auto* rn = label("", "refNote", "small", tc);
            rn->hide();
            tcv->addWidget(rn);
        }
        auto slider = [&](const char* id, const char* name, const char* unit, int lo, int hi, const char* tip) {
            auto* w = new QWidget(tc);
            w->setObjectName(QString(id) + "Row");
            w->setProperty("role", "slider");
            auto* sv = column(w, 0, 6);
            auto* top = new QWidget(w);
            auto* th = new QHBoxLayout(top);
            th->setContentsMargins(0, 0, 0, 0);
            th->setSpacing(4);
            th->addWidget(label(name, {}, "slider-lab"));
            th->addStretch(1);
            th->addWidget(label("", QString(id) + "Val", "slider-val"));
            th->addWidget(label(unit, {}, "unit"));
            sv->addWidget(top);
            auto* s = new QSlider(Qt::Horizontal, w);
            s->setObjectName(id);
            s->setRange(lo, hi);
            s->setToolTip(QString::fromUtf8(tip));
            sv->addWidget(s);
            tcv->addWidget(w);
            return s;
        };
        auto* strength = slider("acesStrength", "Strength", "%", 0, 150,
                                "How far the master moves from the plain ACES inverse (0 %) to the model (100 %) or past it");
        auto* sat = slider("acesSaturation", "Highlight saturation", "%", 0, 200,
                           "The colour of the rebuilt pixels about their own luminance; the rest of the frame is untouched");
        auto* peak = slider("acesPeak", "Peak limit", "nits", 3, 41,
                            "A soft limit on the rebuilt pixels' brightness (rolls off from 75 % of it); far right is no limit");
        auto* expo = slider("acesExposure", "Exposure", "EV", -30, 30,
                            "The whole master up or down, in tenths of a stop (after the strength)");
        auto* con = slider("acesContrast", "Contrast", "%", 50, 150,
                           "About 18 % grey, on luminance: the hue stays (100 % = as the model made it)");
        auto changed = [this, strength, sat, peak, expo, con] {
            aces_v2::AcesTrims t;
            t.strength = strength->value() / 100.0;
            t.saturation = sat->value() / 100.0;
            t.peak_nits = peak->value() >= peak->maximum() ? 0.0 : peak->value() * 250.0;   // 750 .. 10000 by 250
            t.exposure_ev = expo->value() / 10.0;
            t.contrast = con->value() / 100.0;
            if (!(t == aces_trims_)) set_aces_trims(t);
        };
        for (auto* s : {strength, sat, peak, expo, con}) connect(s, &QSlider::valueChanged, this, changed);
        {   // Per shot: the trims are remembered with each shot; copy one shot's onto another.
            auto* row = new QWidget(tc);
            auto* rh = new QHBoxLayout(row);
            rh->setContentsMargins(0, 4, 0, 0);
            rh->setSpacing(6);
            auto* copy = new QPushButton("Copy trims", row);
            copy->setObjectName("acesTrimsCopy");
            copy->setToolTip("Keep this shot's trims to paste onto another shot (Batch: Apply puts them on the selected jobs)");
            connect(copy, &QPushButton::clicked, this, [this] { copy_aces_trims(); });
            auto* paste = new QPushButton("Paste", row);
            paste->setObjectName("acesTrimsPaste");
            paste->setEnabled(false);
            paste->setToolTip("The copied trims onto this shot");
            connect(paste, &QPushButton::clicked, this, [this] { paste_aces_trims(); });
            rh->addWidget(copy);
            rh->addWidget(paste);
            rh->addStretch(1);
            tcv->addWidget(row);
        }
        auto* note = label("Applied after the model, the same in the viewer, Deliver and the batch; written into each "
                           "render's provenance. Remembered per shot: a shot opened again gets its own back.",
                           "acesTrimsNote", "small", tc);
        note->setWordWrap(true);
        tcv->addWidget(note);
        tc->hide();   // until the viewer runs the learned model
        recv->addWidget(tc);
    }
    {   // Simple: what the Look page says before the learned model runs, and the way to everything else.
        auto* wait = label("", "lookWaiting", "small", rec);
        wait->setWordWrap(true);
        wait->hide();
        recv->addWidget(wait);
        auto* get = new QPushButton("Download the model\u2026", rec);
        get->setObjectName("lookDownload");
        get->setProperty("role", "primary");
        get->setToolTip("File > Model packages: download the learned ACES 2.0 model, checked before it is used");
        connect(get, &QPushButton::clicked, this, [this] { open_model_manager(); });
        get->hide();
        recv->addWidget(get, 0, Qt::AlignLeft);
        auto* adv = new QPushButton("Advanced \u2192", rec);
        adv->setObjectName("lookAdvanced");
        adv->setFlat(true);
        adv->setToolTip("Full: the analytic route, Source curves, calibration, region exposure and every option");
        connect(adv, &QPushButton::clicked, this, [this] { set_workspace("full"); });
        adv->hide();
        recv->addWidget(adv, 0, Qt::AlignLeft);
    }
    // Full's analytic reconstruction, under one parent so Simple hides it at once.
    auto* analytic = new QWidget(rec);
    analytic->setObjectName("recAnalytic");
    auto* anv = column(analytic, 0, 12);
    // Source (roadmap 3.1): the curve that made the SDR, before anything is
    // reconstructed from it. The baseline is its inverse; the masks and the
    // residual do not depend on it, so the picker runs no inference.
    anv->addWidget(label("Source", "recSourceTitle", "title", rec));
    {
        auto scard = card(rec, 12, 12);
        source_panel_ = new SourcePanel(scard.first);
        source_panel_->picked = [this](SourceCurve c) { session_.set_source(source_curve_id(c)); };
        source_panel_->arm = [this](int slot) { arm_calibration(slot); };
        source_panel_->anchor_changed = [this](int slot, int code, double nits) {
            session_.set_calibration(slot, code, nits);
        };
        source_panel_->cleared = [this] { session_.clear_calibration(); };
        source_panel_->load_reference = [this] { load_reference({}); };
        source_panel_->reference_cleared = [this] { session_.clear_reference(); };
        scard.second->addWidget(source_panel_);
        scard.first->setObjectName("recSourceCard");
        anv->addWidget(scard.first);
        auto* n0 = label("The baseline is the analytic inverse of this curve. Unknown is the ACES inverse every "
                         "master so far used. The masks and the residual are unchanged by the choice.",
                         {}, "note", rec);   // "note", not "note-p": the page's three note-p are compared in order
        n0->setWordWrap(true);
        notes_.push_back(n0);
        anv->addWidget(n0);
    }
    anv->addWidget(label("Reconstruction", "recReconTitle", "title", rec));
    // Not a structured binding: the lambdas below use both, and Apple clang 15
    // cannot capture a structured binding (C++20 allows it, from clang 16).
    auto rcard_rcv = card(rec, 12, 14);
    auto* rcard = rcard_rcv.first;
    auto* rcv = rcard_rcv.second;
    mode_seg_ = new Seg("mode", {{"all", "All"}, {"highlights", "Highlights"}, {"shadows", "Shadows"}, {"off", "Off"}},
                        rcard);
    mode_seg_->clicked = [this](const QString& k) { session_.set_mode(k.toStdString()); };
    rcv->addWidget(mode_seg_);
    auto slider = [&](const QString& lab, const QString& val_id, const QString& unit, const QString& id, int lo,
                      int hi, QLabel*& val_out, QSlider*& out) {
        auto* w = new QWidget(rcard);
        w->setProperty("role", "slider");
        auto* sv2 = column(w, 0, 6);
        auto* top = new QWidget(w);
        auto* tph = new QHBoxLayout(top);
        tph->setContentsMargins(0, 0, 0, 0);
        tph->setSpacing(4);
        tph->addWidget(label(lab, {}, "slider-lab"));
        tph->addStretch(1);
        val_out = label("", val_id, "slider-val");
        tph->addWidget(val_out);
        tph->addWidget(label(unit, {}, "unit"));
        sv2->addWidget(top);
        out = new QSlider(Qt::Horizontal, w);
        out->setObjectName(id);
        out->setRange(lo, hi);
        sv2->addWidget(out);
        rcv->addWidget(w);
    };
    // The page's range inputs, in their own steps: strength 0 to 2 by 0.05,
    // peak -1 to 5 by 0.5. A step n reads back as n / 20 (n / 2): the double
    // the page's parseFloat of the same value gives.
    // 13 Oct 2026: the master exposure, -3 to +3 stops by 0.05 (n / 20), live
    // like strength: one undo entry per drag, double-click resets.
    slider("Exposure", "exposureVal", "EV", "exposure", -60, 60, exposure_val_, exposure_);
    connect(exposure_, &QSlider::sliderPressed, this, [this] { session_.exposure_press(); });
    connect(exposure_, &QSlider::valueChanged, this, [this](int n) {
        if (std::lround(session_.grade.exposure_ev * 20.0) == n) return;   // sync_ui, not a move
        if (!exposure_->isSliderDown()) session_.exposure_press();          // a key or a click on the track
        session_.exposure_input(n / 20.0);
    });
    exposure_->installEventFilter(this);   // double-click: back to 0 (MainWindow::eventFilter)
    slider("Residual strength", "strengthVal", "\u00d7", "strength", 0, 40, strength_val_, strength_);
    slider("Display peak", "peakVal", "nits", "peak", -2, 10, peak_val_, peak_);
    connect(strength_, &QSlider::sliderPressed, this, [this] { session_.strength_press(); });
    connect(strength_, &QSlider::valueChanged, this, [this](int n) {
        if (std::lround(session_.grade.strength * 20.0) == n) return;   // sync_ui, not a move
        if (!strength_->isSliderDown()) session_.strength_press();      // a key or a click on the track
        session_.strength_input(n / 20.0);
    });
    connect(peak_, &QSlider::valueChanged, this, [this](int n) {
        if (std::lround(session_.peak_ev * 2.0) == n) return;
        session_.peak_input(n / 2.0);
    });
    preserve_ = new CheckRow("preserve", "Preserve outside masks", "preserveHint", "do-no-harm", rcard);
    preserve_->clicked = [this] { session_.toggle_preserve(); };
    rcv->addWidget(preserve_);
    rcard->setObjectName("recReconCard");
    anv->addWidget(rcard);
    {   // Shown on the learned route: these grade the analytic reconstruction only.
        auto* rn = label("Learned ACES 2.0 is rendering: these controls are the analytic reconstruction's and do not "
                         "reach its frames or renders. Its trims are in the Learned ACES 2.0 card above.",
                         "routeNoteRec", "small", rec);
        rn->setWordWrap(true);
        rn->hide();
        anv->addWidget(rn);
    }
    auto* n1 = label("The reconstruction is applied where clipping or crushed shadows made the SDR mapping "
                     "non-invertible. Everywhere else the analytic baseline is preserved unchanged.",
                     {}, "note-p", rec);
    n1->setWordWrap(true);
    notes_.push_back(n1);
    anv->addWidget(n1);
    recv->addWidget(analytic);
    pages->addWidget(rec);
    pages_.push_back(rec);

    // Grade
    auto* grade = new QWidget;
    auto* gv = column(grade, 0, 12);
    gv->addWidget(label("Region exposure", {}, "title", grade));
    auto [gcard, gcv] = card(grade, 4, 0);
    gcv->addWidget(panel_label("Region EV", "3", "regionCount", gcard));
    region_count_ = gcard->findChild<QLabel*>("regionCount");
    regions_ = new RegionEditor(session_, gcard);
    regions_->mask_pressed = [this](int band) { arm_mask(session_.painted_band() == band ? -1 : band); };
    gcv->addWidget(regions_);
    mask_panel_ = new MaskPanel(session_, gcard);
    mask_panel_->done = [this] { arm_mask(-1); };
    gcv->addWidget(mask_panel_);
    gv->addWidget(gcard);
    {
        auto* rn = label("Learned ACES 2.0 is rendering: region exposure is the analytic master's and does not reach "
                         "the learned frames or renders.", "routeNoteGrade", "small", grade);
        rn->setWordWrap(true);
        rn->hide();
        gv->addWidget(rn);
    }
    auto* n2 = label("A region is a soft luminance qualifier, not a mask, so it follows the picture. Drag a value to "
                     "scrub it, double-click to zero it. Master applies the identical qualifier and gain to the file.",
                     {}, "note-p", grade);
    n2->setWordWrap(true);
    notes_.push_back(n2);
    gv->addWidget(n2);
    pages->addWidget(grade);
    pages_.push_back(grade);

    // Deliver
    auto* del = ctl();
    auto* dv = column(del, 0, 12);
    dv->addWidget(label("Delivery", {}, "title", del));
    // Simple's Deliver (29 Oct 2026): four choices; Custom… opens the full What to write card.
    {
        auto [pc, pcv] = card(del, 12, 8);
        pc->setObjectName("deliverPresets");
        pcv->addWidget(label("What do you need?", {}, "card-head", pc));
        auto* presets = new Seg("deliverPreset",
                                {{"master", "Grading master\nACES 2065-1 EXR, for the colourist"},
                                 {"hdr10", "HDR10 movie\nHEVC 10-bit, PQ 1000 nits, with metadata"},
                                 {"hlg", "Broadcast HDR\nHLG, ProRes 422 HQ"},
                                 {"custom", "Custom\u2026\nEvery format, colour and codec"}},
                                pc, true, 1);
        presets->clicked = [this](const QString& k) { apply_deliver_preset(k); };
        pcv->addWidget(presets);
        pc->hide();
        dv->addWidget(pc);
    }
    // Full's cards (and Simple's Custom…), under one parent so Simple hides them at once.
    auto* full_cards = new QWidget(del);
    full_cards->setObjectName("deliverFullCards");
    auto* fcv_outer = column(full_cards, 0, 12);
    // What renders the files and what they are (27 Oct 2026, deliver_choice.hpp):
    // each control under its name, as the Render card's fields are.
    {
        auto [kc, kcv] = card(del, 12, 4);
        kc->setObjectName("deliverCard");
        kcv->addWidget(label("What to write", {}, "card-head", kc));
        auto field = [&](const QString& key, QWidget* control, const char* id, const char* tip) {
            auto* w = new QWidget(kc);
            w->setObjectName(id);
            auto* fv = column(w, 0, 4);
            fv->setContentsMargins(0, 4, 0, 2);
            fv->addWidget(label(key, {}, "note", w));
            control->setParent(w);
            control->setToolTip(QString::fromUtf8(tip));
            fv->addWidget(control);
            kcv->addWidget(w);
        };
        auto* src = new Seg("deliverSource", {{"aces", "Learned ACES 2.0"}, {"analytic", "Analytic"}});
        src->clicked = [this](const QString& k) { set_deliver("source", k); };
        field("Source", src, "deliverSourceRow",
              "What renders the files: the learned ACES 2.0 model (what the viewer shows when it is loaded) "
              "or the analytic reconstruction");
        auto* out = new Seg("deliverOutput", {{"still", "Still"}, {"sequence", "Sequence"}, {"movie", "MOV"}});
        out->clicked = [this](const QString& k) { set_deliver("output", k); };
        field("Output", out, "deliverOutputRow",
              "Still: the frame on screen. Sequence: every loaded frame, one file each. MOV: one QuickTime movie");
        auto* image = new QComboBox;
        image->setObjectName("deliverImage");
        image->addItem("EXR · 32-bit float", "exr32");
        image->addItem("EXR · 16-bit half", "exr16");
        image->addItem("DPX · 10-bit", "dpx10");
        connect(image, &QComboBox::activated, this, [this, image](int i) { set_deliver("image", image->itemData(i).toString()); });
        field("Format", image, "deliverImageRow", "The file type of a still or a sequence");
        auto* gamut = new Seg("deliverGamut", {{"ap0", "ACES 2065-1"}, {"acescg", "ACEScg"}, {"linear", "Rec.2020 linear"}});
        gamut->clicked = [this](const QString& k) { set_deliver("gamut", k); };
        field("Colour", gamut, "deliverGamutRow",
              "The EXR's primaries: ACES 2065-1 (AP0, the archive and interchange master) or ACEScg (AP1, for compositing)");
        auto* bits = new Seg("deliverBits", {{"10", "10-bit"}, {"16", "16-bit"}});
        bits->clicked = [this](const QString& k) { set_deliver("bits", k); };
        field("Bit depth", bits, "deliverBitsRow",
              "10-bit: ProRes. 16-bit: uncompressed RGB QuickTime, 'b48r' (lossless, 6 bytes a pixel a frame)");
        auto* codec = new QComboBox;
        codec->setObjectName("deliverCodec");
        codec->addItem("ProRes 422 HQ \u00b7 10-bit 4:2:2", "prores422hq");
        codec->addItem("ProRes 4444 \u00b7 4:4:4 (12-bit)", "prores4444");
        codec->addItem("ProRes 4444 XQ \u00b7 4:4:4 (12-bit, highest rate)", "prores4444xq");
        codec->addItem("HEVC HDR10 \u00b7 Main 10 MP4 (PQ, metadata in the file)", "hevc10");
        connect(codec, &QComboBox::activated, this, [this, codec](int i) { set_deliver("codec", codec->itemData(i).toString()); });
        field("Codec", codec, "deliverCodecRow",
              "ProRes 422 HQ is 10-bit 4:2:2 (review, editorial); ProRes 4444 / 4444 XQ keep full chroma and store up "
              "to 12 bits (XQ: the HDR master rate). HEVC HDR10: a Main 10 MP4 with the measured MaxCLL / MaxFALL and "
              "the P3 D65 1000-nit mastering display in the file (the learned model, PQ)");
        auto* enc = new Seg("deliverEncoding",
                            {{"logc4", "LogC4"}, {"acescct", "ACEScct"}, {"pq", "PQ"}, {"hlg", "HLG"}, {"sdr", "SDR"}});
        enc->clicked = [this](const QString& k) { set_deliver("encoding", k); };
        enc->button("logc4")->setToolTip("ARRI LogC4 in ARRI Wide Gamut 4: the scene-linear master, no peak or knee");
        enc->button("acescct")->setToolTip("ACEScct in ACES AP1: the scene-linear master, no peak or knee");
        field("Encoding", enc, "deliverEncodingRow",
              "How a DPX or a movie holds the master: a camera log (scene-referred), or an ACES 2.0 output "
              "(PQ, HLG, or the SDR Rec.709 trim pass of the same master)");
        auto* pk = new Seg("deliverPqPeak", {{"1000", "1000"}, {"2000", "2000"}, {"4000", "4000"}});
        pk->clicked = [this](const QString& k) { set_deliver("pq_peak", k); };
        field("PQ peak", pk, "deliverPqPeakRow",
              "The ACES 2.0 HDR output's peak, in nits (P3 D65): 1000 for HDR10 and most streaming, 2000 / 4000 "
              "for Dolby Vision mastering and theatrical HDR");
        auto* fps = new QComboBox;
        fps->setObjectName("deliverFps");
        for (const auto& [text, value] : std::initializer_list<std::pair<const char*, double>>{
                 {"23.976", 24000.0 / 1001.0}, {"24", 24.0}, {"25", 25.0}, {"29.97", 30000.0 / 1001.0},
                 {"30", 30.0}, {"48", 48.0}, {"50", 50.0}, {"59.94", 60000.0 / 1001.0}, {"60", 60.0}})
            fps->addItem(text, value);
        fps->setCurrentIndex(1);
        connect(fps, &QComboBox::activated, this,
                [this, fps](int i) { set_deliver("fps", QString::number(fps->itemData(i).toDouble(), 'g', 12)); });
        field("Frame rate", fps, "deliverFpsRow", "A movie made from stills has no clock of its own: its frame rate");
        auto* keep = new CheckRow("deliverKeepExr", "Keep the EXR masters", "deliverKeepExrHint", "beside the movie", kc);
        keep->setToolTip("Also write the EXR master of every frame: ACEScg float32 with the mask channel, or ACES 2065-1 "
                         "half (ST 2065-4) with the mask in mask/");
        keep->clicked = [this] { set_deliver("keep_exr", deliver_.keep_exr ? "0" : "1"); };
        kcv->addWidget(keep);
        // The AI fill (layer 3, genfill_job.hpp): an AI-filled copy of a learned EXR render.
        auto* ai = new CheckRow("deliverAiFill", "AI fill (experimental)", "deliverAiFillHint",
                                "clipped highlights, crushed shadows", kc);
        ai->setToolTip("After the render, a Wan2.1-VACE model trained on RUDRA's own footage fills only what the "
                       "8-bit source cannot know: fully clipped highlights and fully crushed shadows. It writes a "
                       "copy beside the master (<name>_aifill); no 8-bit code of the SDR view changes, and "
                       "rudra.generated marks every pixel it invented");
        ai->clicked = [this] { set_deliver("ai_fill", deliver_.ai_fill ? "0" : "1"); };
        kcv->addWidget(ai);
        {
            auto* box = new QWidget(kc);
            box->setObjectName("deliverAiFillOptions");
            auto* bv = column(box, 0, 2);
            bv->setContentsMargins(14, 0, 0, 4);
            auto spin = [&](const char* id, const QString& key, int lo, int hi, const QString& special, const char* tip) {
                auto* sb = new QSpinBox(box);
                sb->setObjectName(id);
                sb->setRange(lo, hi);
                if (!special.isEmpty()) sb->setSpecialValueText(special);
                connect(sb, &QSpinBox::valueChanged, this, [this, key](int value) { set_deliver(key, QString::number(value)); });
                auto* r = new QWidget(box);
                auto* rcol = column(r, 0, 2);
                rcol->addWidget(label(QString::fromUtf8(tip).section('|', 0, 0), {}, "note", r));
                sb->setToolTip(QString::fromUtf8(tip).section('|', 1));
                sb->setParent(r);
                rcol->addWidget(sb);
                bv->addWidget(r);
            };
            spin("deliverAiTauHigh", "ai_tau_high", 128, 256, {},
                 "Highlight threshold (8-bit code)|Codes at or above it are filled; 256 fills no highlights. "
                 "The ACES window's \"AI fill regions\" view shows them");
            spin("deliverAiTauLow", "ai_tau_low", -1, 64, "off",
                 "Shadow threshold (8-bit code)|Codes at or below it are filled; off fills no shadows");
            auto line = [&](const char* id, const QString& key, const QString& head, const QString& placeholder) {
                auto* le = new QLineEdit(box);
                le->setObjectName(id);
                le->setPlaceholderText(placeholder);
                connect(le, &QLineEdit::editingFinished, this, [this, le, key] { set_deliver(key, le->text()); });
                bv->addWidget(label(head, {}, "note", box));
                bv->addWidget(le);
            };
            line("deliverAiPromptHighlight", "ai_prompt_highlight", "Highlight prompt (optional)",
                 "e.g. warm sunset sun with soft halo");
            line("deliverAiPromptShadow", "ai_prompt_shadow", "Shadow prompt (optional)", "e.g. dark wood grain");
            spin("deliverAiSeed", "ai_seed", 0, 999999, {}, "Seed|Another seed gives another plausible fill");
            spin("deliverAiSteps", "ai_steps", 1, 100, {}, "Steps|Denoising steps: more is slower, often cleaner (30)");
            {   // v3 (30 Oct 2026): the AI draws the shape; how bright it is comes from RUDRA's estimate plus this
                auto* r = new QWidget(box);
                auto* rcol = column(r, 0, 2);
                rcol->addWidget(label("Highlight brightness (stops)", {}, "note", r));
                auto* bs = new QDoubleSpinBox(r);
                bs->setObjectName("deliverAiBrightness");
                bs->setRange(-3.0, 3.0);
                bs->setSingleStep(0.25);
                bs->setDecimals(2);
                bs->setSuffix(" stops");
                bs->setToolTip("How much brighter (or darker) than RUDRA's estimate the filled highlights are. "
                               "0 = the estimate. The AI draws the shape inside a clipped area; its brightness is yours to set");
                connect(bs, &QDoubleSpinBox::valueChanged, this, [this](double v) { set_deliver("ai_brightness", QString::number(v, 'g', 6)); });
                rcol->addWidget(bs);
                bv->addWidget(r);
            }
            auto* setup = label("", "deliverAiSetup", "small", box);
            setup->setWordWrap(true);
            bv->addWidget(setup);
            {   // Where the model is, and the last fill's review before it goes to a client (28 Oct 2026 review).
                auto* row = new QWidget(box);
                auto* rh = new QHBoxLayout(row);
                rh->setContentsMargins(0, 4, 0, 0);
                auto* su = new QPushButton("Setup…", row);
                su->setObjectName("deliverAiSetupButton");
                su->setToolTip("Where the Python, the repository, Wan2.1-VACE and the RUDRA LoRA are; checks them");
                connect(su, &QPushButton::clicked, this, [this] { GenFillSetupDialog(this).exec(); sync_deliver(); });
                auto* review = new QPushButton("Review fill…", row);
                review->setObjectName("deliverAiReview");
                review->setToolTip("Page through the last AI fill (or one you pick): A master, B fill, the invented pixels, "
                               "and the frames where the fill flickers");
                connect(review, &QPushButton::clicked, this, [this] { review_genfill(); });
                rh->addWidget(su);
                rh->addWidget(review);
                rh->addStretch(1);
                bv->addWidget(row);
            }
            kcv->addWidget(box);
        }
        auto* note = label("", "deliverNote", "small", kc);
        note->setWordWrap(true);
        kcv->addWidget(note);
        fcv_outer->addWidget(kc);
    }
    auto [fcard, fcv] = card(del, 12, 0);
    container_field_ = label("OpenEXR \u2014 ACES 2065-1", "containerField", "value");
    primaries_field_ = label("AP0 (ST 2065-4)", "primariesField", "value");
    for (auto* r : {row("Container", container_field_, "field"),
                    row("Transfer", label("linear (scene-referred)", "transferField", "value"), "field"),
                    row("Primaries", primaries_field_, "field"),
                    row("White", label("diffuse 1.0 = 203 nits", "whiteField", "value"), "field")}) {
        r->layout()->setContentsMargins(0, 9, 0, 9);
        fcv->addWidget(r);
    }
    if (auto* wf = fcard->findChild<QLabel*>("whiteField")) wf->parentWidget()->setObjectName("whiteRow");
    fcv_outer->addWidget(fcard);
    auto [ocard, ocv] = card(del, 12, 10);
    ocard->setObjectName("deliverAnalyticOptions");   // the analytic master's own; hidden for the learned source
    anchor_ = new CheckRow("anchor", "Anchor to source exposure", "anchorHint", "conform", ocard);
    anchor_->clicked = [this] {
        session_.toggle_anchor();
        log(session_.anchor ? "master will anchor to the source exposure"
                            : "master will keep the inverse tone map's own level");
    };
    carry_chroma_ = new CheckRow("carryChroma", "Carry source chroma", "chromaHint", "below the clip", ocard);
    carry_chroma_->clicked = [this] {
        session_.toggle_carry_chroma();
        log(session_.carry_chroma ? "master will take hue from the source below the clip"
                                  : "master will keep the per-channel expansion's own hue");
    };
    ocv->addWidget(anchor_);
    // The anchor's knee (roadmap 3.3): where on the SDR's code range the
    // per-pixel level match hands over to the frame's hold gain. Live on the
    // viewer while Anchor is on, so what is seen is what the master does.
    {
        auto* w = new QWidget(ocard);
        w->setProperty("role", "slider");
        auto* sv2 = column(w, 0, 6);
        auto* top = new QWidget(w);
        auto* tph = new QHBoxLayout(top);
        tph->setContentsMargins(0, 0, 0, 0);
        tph->setSpacing(4);
        tph->addWidget(label("Anchor knee", {}, "slider-lab"));
        tph->addStretch(1);
        anchor_knee_val_ = label("", "anchorKneeVal", "slider-val");
        tph->addWidget(anchor_knee_val_);
        tph->addWidget(label("code", {}, "unit"));
        sv2->addWidget(top);
        anchor_knee_ = new QSlider(Qt::Horizontal, w);
        anchor_knee_->setObjectName("anchorKnee");
        anchor_knee_->setRange(50, 99);   // 0.50 to 0.99 by 0.01
        sv2->addWidget(anchor_knee_);
        connect(anchor_knee_, &QSlider::valueChanged, this, [this](int n) {
            if (std::lround(session_.anchor_knee * 100.0) == n) return;   // sync_ui, not a move
            session_.set_anchor_knee(n / 100.0);
        });
        ocv->addWidget(w);
    }
    ocv->addWidget(carry_chroma_);
    fcv_outer->addWidget(ocard);

    dv->addWidget(full_cards);

    auto [rendc, form] = card(del, 12, 4);
    form->addWidget(label("Render", {}, "card-head", rendc));
    auto* dir = new QLineEdit(rendc);
    dir->setObjectName("renderDir");
    dir->setPlaceholderText("C:\\Renders\\Shot01");
    auto* mode = new QComboBox(rendc);
    mode->setObjectName("renderMode");
    mode->addItem("Current image", "image");
    mode->addItem("All loaded frames \u2014 sequence", "sequence");
    auto* name = new QLineEdit("master", rendc);
    name->setObjectName("renderName");
    auto* start = new QSpinBox(rendc);
    start->setObjectName("renderStart");
    start->setRange(0, 9999999);
    start->setValue(1);
    // The timecode and reel a render carries (28 Oct 2026): a movie's own by
    // default; for stills and sequences, a start timecode typed here.
    auto* tcode = new QLineEdit(rendc);
    tcode->setObjectName("renderTimecode");
    tcode->setPlaceholderText("from the movie, else none (e.g. 01:00:00:00)");
    tcode->setToolTip("The first frame's timecode (HH:MM:SS:FF, ;FF for drop frame). Empty: a movie keeps its own; "
                      "stills and sequences get none. Written into the EXR header and the MOV");
    auto* reel = new QLineEdit(rendc);
    reel->setObjectName("renderReel");
    reel->setPlaceholderText("from the movie, else its file name");
    reel->setToolTip("The reel name written into the EXR header (reelName) and the MOV, for the conform");
    // The file, as Nuke's Write takes it (29 Oct 2026): one path with the frame
    // number as ####, a Browse… beside it, and Create directories. The folder and
    // name fields stay (hidden) as what Master EXR, the settings and projects read.
    auto* file_row = new QWidget(rendc);
    auto* frh = new QHBoxLayout(file_row);
    frh->setContentsMargins(0, 0, 0, 0);
    frh->setSpacing(6);
    auto* file = new QLineEdit(file_row);
    file->setObjectName("renderFile");
    file->setPlaceholderText("D:\\Renders\\SH010\\SH010_hdr.####.exr");
    file->setToolTip("Where the render goes, as in Nuke's Write: folder, name, #### for the frame number (or %04d), "
                     "extension. The files are written into that folder; nothing there is ever replaced");
    auto* browse = new QPushButton("Browse\u2026", file_row);
    browse->setObjectName("renderBrowse");
    browse->setToolTip("Choose the folder and the name");
    frh->addWidget(file, 1);
    frh->addWidget(browse);
    auto* mkdirs = new QCheckBox("Create directories", rendc);
    mkdirs->setObjectName("renderCreateDirs");
    mkdirs->setChecked(QSettings().value("deliver/createDirs", true).toBool());
    mkdirs->setToolTip("Make the folder in the path when it is not there yet (Nuke's create directories); off: a "
                       "missing folder stops the render");
    connect(mkdirs, &QCheckBox::toggled, this, [](bool on) { QSettings().setValue("deliver/createDirs", on); });
    for (const auto& [text, field] : std::vector<std::pair<QString, QWidget*>>{
             {"File", file_row}, {"Render folder", dir}, {"Output", mode}, {"Render name", name}, {"Start frame", start},
             {"Start timecode", tcode}, {"Reel", reel}}) {
        auto* l = label(text, {}, "note", rendc);
        l->setObjectName((field == file_row ? QStringLiteral("renderFile") : field->objectName()) + "Label");
        form->addWidget(l);
        form->addWidget(field);
        form->addSpacing(2);
        if (field == mode || field == dir || field == name) {
            l->hide();
            field->hide();
        }
        if (field == file_row) {
            form->addWidget(mkdirs);
            form->addSpacing(2);
        }
    }
    // The path and the hidden fields follow each other.
    connect(file, &QLineEdit::editingFinished, this, [this] { render_file_edited(); });
    connect(browse, &QPushButton::clicked, this, [this] { browse_render_file(); });
    // The folder or name changed (a project, a test, the settings): the path and the note follow.
    for (auto* e : {dir, name}) connect(e, &QLineEdit::textChanged, this, [this] { sync_deliver(); });
    connect(start, &QSpinBox::valueChanged, this, [this] { sync_deliver(); });
    connect(mode, &QComboBox::currentIndexChanged, this, [this, mode](int) {
        const QString m = mode->currentData().toString();
        if (m == "image" && deliver_.output != "still") set_deliver("output", "still");
        else if (m == "sequence" && deliver_.output == "still") set_deliver("output", "sequence");
    });
    auto* small = label("Sequence uses loaded frame order and the current grade for every frame. Nothing is ever "
                        "replaced.",
                        {}, "small", rendc);
    small->setWordWrap(true);
    small->setObjectName("renderHint");
    form->addWidget(small);
    auto* status = label("", "renderStatus", "render-status", rendc);
    status->setWordWrap(true);
    status->hide();
    form->addWidget(status);
    auto* acts = new QWidget(rendc);
    auto* ah = new QHBoxLayout(acts);
    ah->setContentsMargins(0, 6, 0, 0);
    ah->setSpacing(8);
    btn_reprocess_ = new QPushButton("Reprocess", acts);
    btn_reprocess_->setObjectName("btnReprocess");
    btn_master_ = new QPushButton("Master EXR", acts);
    btn_master_->setObjectName("btnMaster");
    btn_master_->setProperty("role", "primary");
    connect(btn_reprocess_, &QPushButton::clicked, this, [this] {
        if (!frames_.empty()) step_to(current_);
    });
    connect(btn_master_, &QPushButton::clicked, this, [this] { run("master"); });
    ah->addWidget(btn_reprocess_, 1);
    ah->addWidget(btn_master_, 1);
    form->addWidget(acts);
    auto* batch_add = new QPushButton("Add to batch queue", rendc);
    batch_add->setObjectName("btnBatchAdd");
    batch_add->setToolTip("Queue this shot with these settings, to render with others through the learned ACES 2.0 "
                          "model (Ctrl+B; Deliver > Batch queue… shows the queue)");
    connect(batch_add, &QPushButton::clicked, this, [this] { run("batch-add"); });
    form->addWidget(batch_add);
    dv->addWidget(rendc);
    auto* n3 = label("Stills, sequences and MOVs render here. HDR10 and HLG movies are on the toolbar's Export sheet; "
                     "the learned ACES 2.0 master window (Ctrl+Shift+E) adds 16-bit TIFFs and a preview.",
                     {}, "note-p", del);
    n3->setTextFormat(Qt::RichText);
    n3->setWordWrap(true);
    notes_.push_back(n3);
    dv->addWidget(n3);
    dv->addStretch(1);   // the Output card's rows come and go: spare height goes below, not between the cards
    pages->addWidget(del);
    pages_.push_back(del);

    // Check (Simple, 29 Oct 2026): the frame in four numbers, the reference stills;
    // the scopes show in their column beside the picture while this tab is on.
    {
        auto* chk = ctl();
        chk->setObjectName("checkPage");
        auto* cv = column(chk, 0, 12);
        cv->addWidget(label("Check", {}, "title", chk));
        auto [sc, scv] = card(chk, 12, 4);
        sc->setObjectName("checkSummary");
        auto* head = new QWidget(sc);
        auto* hh = new QHBoxLayout(head);
        hh->setContentsMargins(0, 0, 0, 4);
        hh->addWidget(label("This frame", {}, "card-head"));
        hh->addStretch(1);
        hh->addWidget(label("measured", "checkSource", "note"));
        scv->addWidget(head);
        for (const auto& [k, id, tip] : std::vector<std::tuple<QString, QString, QString>>{
                 {"Peak", "checkPeak", "The brightest pixel of the HDR, in nits"},
                 {"Average", "checkAverage", "The frame's average light (what MaxFALL counts), in nits"},
                 {"Brighter than SDR white", "checkAbove", "The share of the frame above diffuse white (203 nits): the light the SDR could not hold"},
                 {"Outside P3-D65", "checkGamut", "The share of pixels outside the P3-D65 gamut an HDR10 or Dolby Vision master is graded in"}}) {
            auto* val = label("\u2014", id, "value");
            auto* r = row(k, val);
            r->setToolTip(tip);
            scv->addWidget(r);
        }
        cv->addWidget(sc);
        auto* rrow = new QWidget(chk);
        auto* rh = new QHBoxLayout(rrow);
        rh->setContentsMargins(0, 0, 0, 0);
        rh->setSpacing(8);
        auto* grab = new QPushButton("Grab still", rrow);
        grab->setObjectName("checkGrab");
        grab->setToolTip("Keep this frame as a reference, to compare other frames and shots with (Ctrl+Alt+G)");
        connect(grab, &QPushButton::clicked, this, [this] { run("still-grab"); });
        auto* cmp = new QPushButton("Compare", rrow);
        cmp->setObjectName("checkCompare");
        cmp->setCheckable(true);
        cmp->setToolTip("Hold B or use View \u25be Wipe to see the last still beside this frame");
        connect(cmp, &QPushButton::clicked, this, [this](bool on) { set_reference_compare(on); });
        rh->addWidget(grab, 1);
        rh->addWidget(cmp, 1);
        cv->addWidget(rrow);
        auto* note = label("The scopes are beside the picture. Hold B for the original SDR; View \u25be has the wipe, "
                           "what changed and false colour.", "checkNote", "small", chk);
        note->setWordWrap(true);
        cv->addWidget(note);
        cv->addStretch(1);
        pages->addWidget(chk);
        pages_.push_back(chk);
    }
    v->addWidget(panels_);

    // ── this frame ──
    auto* stats = styled("frameStats", "card");
    auto* fv = column(stats, 12, 6);
    auto* shead = new QWidget(stats);
    auto* shh = new QHBoxLayout(shead);
    shh->setContentsMargins(0, 0, 0, 0);
    shh->addWidget(label("This frame", {}, "card-head"));
    shh->addStretch(1);
    shh->addWidget(label("measured", {}, "note"));
    fv->addWidget(shead);
    for (const char* id : {"measA", "measB"}) {
        auto* ms = styled(id, "ms");
        column(ms);
        fv->addWidget(ms);
    }
    clip_bar_ = new ClipBar(stats);
    clip_bar_->hide();   // until a frame is measured
    fv->addWidget(clip_bar_);
    v->addWidget(stats);

    // ── scopes, each a card ──
    scopes_ = styled("scopes", "scopes");
    auto* sv = column(scopes_, 0, 12);
    const struct { const char* title; const char* note; const char* id; int h; } scopes[] = {
        {"Waveform", "luma percentile \u00b7 nits/log", "wave", 112},
        {"Histogram", "absolute nits \u00b7 log2 stops \u00b7 DW", "hist", 84},
        {"Vectorscope", "Rec.2020", "vector", 0}};
    for (const auto& s : scopes) {
        auto [sc, scv] = card(scopes_, 12, 8);
        auto* head = new QWidget(sc);
        auto* hh = new QHBoxLayout(head);
        hh->setContentsMargins(0, 0, 0, 0);
        hh->addWidget(label(s.title, {}, "card-head"));
        hh->addStretch(1);
        hh->addWidget(label(s.note, {}, "note"));
        scv->addWidget(head);
        auto* plot = styled({}, "plot");
        auto* pl = column(plot);
        if (s.h > 0) {
            auto* area = new ScopePlot(s.id, QString(s.id) == "wave" ? ScopePlot::Kind::Waveform
                                                                      : ScopePlot::Kind::Histogram,
                                       s.h, plot);
            (QString(s.id) == "wave" ? wave_ : hist_) = area;
            pl->addWidget(area);
        } else {
            vector_ = new VectorscopeView(plot);
            pl->addWidget(vector_);
        }
        scv->addWidget(plot);
        sv->addWidget(sc);
    }
    {   // The colorist's QC of the learned frame: RGB parade on the PQ scale, and gamut (28 Oct 2026 review).
        auto [pc, pcv] = card(scopes_, 12, 8);
        pc->setObjectName("paradeCard");
        auto* head = new QWidget(pc);
        auto* hh = new QHBoxLayout(head);
        hh->setContentsMargins(0, 0, 0, 0);
        hh->addWidget(label("RGB parade", {}, "card-head"));
        hh->addStretch(1);
        hh->addWidget(label("Rec.2020 \u00b7 PQ \u00b7 gamut", {}, "note"));
        pcv->addWidget(head);
        parade_ = new ParadeView(pc);
        pcv->addWidget(parade_);
        sv->addWidget(pc);
    }
    // (the scopes go in their own column beside the viewer: build_ui)

    // ── log ──
    auto [lcard, lcv] = card(col, 12, 6);
    lcard->setObjectName("logCard");
    lcv->addWidget(label("Log", {}, "card-head", lcard));
    log_ = new QPlainTextEdit(lcard);
    log_->setObjectName("log");
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(400);
    log_->setFixedHeight(88);
    lcv->addWidget(log_);
    v->addWidget(lcard);
    v->addStretch(1);
    scroll->setWidget(col);
    rv->addWidget(scroll, 1);
    return rail;
}

QWidget* MainWindow::build_pipe() {
    auto* pipe = styled("pipe", "pipe");
    pipe->setFixedHeight(30);
    auto* h = new QHBoxLayout(pipe);
    h->setContentsMargins(12, 0, 12, 0);
    h->setSpacing(8);
    h->addWidget(label("\u2014", "statusMask", "pipe-s", pipe));
    h->addStretch(1);
    auto seg = [&](const QString& id, const QString& text) {
        auto* b = label(text, id, "pipe-b", pipe);
        h->addWidget(b);
        return b;
    };
    auto arrow = [&] { h->addWidget(label("\u203a", {}, "pipe-arrow", pipe)); };
    seg("pipeIn", "sRGB \u00b7 Rec.709");
    arrow();
    seg("pipeWorking", "scene-linear \u00b7 203 nits = 1.0");
    arrow();
    view_transform_ = seg("viewTransform", "exposure + clip \u00b7 203 nits");
    arrow();
    seg("pipeMaster", "ACES 2065-1 EXR, float32");
    auto* warn = label("", "pipeWarn", {}, pipe);
    warn->hide();
    h->addWidget(warn);
    h->addStretch(1);
    h->addWidget(label("\u2014", "statusTime", "pipe-s", pipe));
    return pipe;
}

}  // namespace rudra::app
