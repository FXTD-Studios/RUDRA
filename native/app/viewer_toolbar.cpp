// The viewer toolbar (docs/view.spec.md revision 6, 1 Nov 2026): Nuke's
// viewer f/stop and gamma, A | B sources with a wipe, the source's clamp and
// the screen's clip as zebras, the Range layer, exposure brackets, and the
// range readout under them. Everything here is view state: none of it reaches
// a master, a render, a measurement or a saved project.
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QScreen>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QAction>
#include <QSignalBlocker>
#include <QSlider>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "flow_layout.hpp"
#include "main_window.hpp"
#include "widgets.hpp"

namespace rudra::app {
namespace {

// The bracket the Alt+B key and the buttons step through.
constexpr double kBrackets[] = {-6.0, -4.0, -2.0, 0.0, 2.0};

QPushButton* tool_button(const QString& text, const QString& id, const QString& tip, QWidget* parent,
                         bool checkable = false) {
    auto* b = new QPushButton(text, parent);
    b->setObjectName(id);
    b->setToolTip(tip);
    b->setCheckable(checkable);
    b->setFocusPolicy(Qt::NoFocus);   // the viewer keeps the keyboard
    return b;
}

QComboBox* source_box(const QString& id, const QString& tip, QWidget* parent) {
    auto* c = new QComboBox(parent);
    c->setObjectName(id);
    c->setToolTip(tip);
    c->setFocusPolicy(Qt::NoFocus);
    c->addItem("SDR (original)", "sdr");
    c->addItem("Baseline", "baseline");
    c->addItem("RUDRA", "model");
    return c;
}

QString signed_ev(double ev) {
    if (ev == 0.0) return QStringLiteral("0.0");
    return (ev > 0.0 ? QStringLiteral("+") : QStringLiteral("\u2212")) + QString::number(std::abs(ev), 'f', 1);
}

QString pct(double v) { return QString::number(v, 'f', v < 10.0 ? 1 : 0) + QStringLiteral(" %"); }
QString stops(double v) { return (v >= 0.0 ? QStringLiteral("+") : QStringLiteral("\u2212")) + QString::number(std::abs(v), 'f', 1); }

// 1.2.6 (user feedback: a cleaner viewer, no function lost): a button that opens a panel
// under it; the panel closes on a click outside it (Qt::Popup). The controls moved into it
// keep their names, shortcuts and behaviour.
struct Popover {
    QPushButton* button = nullptr;
    QFrame* panel = nullptr;
    QVBoxLayout* v = nullptr;
};

Popover popover(const QString& text, const QString& id, const QString& tip, QWidget* parent) {
    Popover p;
    p.button = tool_button(text, id, tip, parent);
    p.panel = new QFrame(parent, Qt::Popup);
    p.panel->setObjectName(id + QStringLiteral("Panel"));
    p.panel->setProperty("role", "popover");
    p.panel->setAttribute(Qt::WA_StyledBackground, true);
    p.v = new QVBoxLayout(p.panel);
    p.v->setContentsMargins(12, 10, 12, 12);
    p.v->setSpacing(4);
    QObject::connect(p.button, &QPushButton::clicked, p.panel, [b = p.button, panel = p.panel] {
        panel->adjustSize();
        QPoint at = b->mapToGlobal(QPoint(0, b->height() + 4));
        if (const QScreen* s = b->screen()) {   // on the screen, whole
            const QRect r = s->availableGeometry();
            at.setX(std::clamp(at.x(), r.left(), std::max(r.left(), r.right() - panel->width())));
            at.setY(std::clamp(at.y(), r.top(), std::max(r.top(), r.bottom() - panel->height())));
        }
        panel->move(at);
        panel->show();
    });
    return p;
}

// One titled row of a panel; each widget keeps whether it was shown (sync code hides some).
void section(const Popover& p, const QString& title, std::initializer_list<QWidget*> parts) {
    if (!title.isEmpty()) {
        auto* cap = new QLabel(title, p.panel);
        cap->setProperty("role", "pop-head");
        p.v->addWidget(cap);
    }
    auto* r = new QWidget(p.panel);
    auto* h = new QHBoxLayout(r);
    h->setContentsMargins(0, 0, 0, 6);
    h->setSpacing(6);
    for (QWidget* w : parts) {
        if (!w) continue;
        const bool hidden = w->isHidden();
        if (QWidget* old = w->parentWidget(); old && old->layout()) old->layout()->removeWidget(w);
        w->setParent(r);
        h->addWidget(w);
        w->setVisible(!hidden);
    }
    h->addStretch(1);
    p.v->addWidget(r);
}

}  // namespace

void MainWindow::build_viewer_popovers(FlowLayout* row) {
    // The window is still being put together: the centre column is not its child yet, so the
    // controls are looked up from the top of the toolbar's own tree.
    QWidget* root = vtool_;
    while (root->parentWidget()) root = root->parentWidget();
    auto find = [root](const char* id) { return root->findChild<QWidget*>(QString::fromLatin1(id)); };
    // Overlays: every layer and check drawn over the picture, in one panel, under one button
    // that counts the ones on ("Overlays · 2"). Was ten buttons in two rows.
    {
        Popover p = popover(QStringLiteral("Overlays \u25be"), QStringLiteral("vtOverlays"),
                            QStringLiteral("What is drawn over the picture: what changed, the invented pixels, false "
                                           "colour, the source's clamp, the screen's clip, the range, the AI fill, and "
                                           "the gamut and invalid-value checks. View only: never rendered"),
                            vtool_);
        vt_overlays_ = p.button;
        auto* hud = find("hud");
        auto* hud_flow = hud ? dynamic_cast<FlowLayout*>(hud->layout()) : nullptr;
        if (hud_flow) {   // the first separator (between Invented and the layers) goes with them
            for (auto* c : hud->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly))
                if (c->property("role").toString() == "hud-sep") {
                    hud_flow->removeWidget(c);
                    c->deleteLater();
                    break;
                }
        }
        section(p, QStringLiteral("Compare"), {changes_btn_, invented_btn_});
        section(p, QStringLiteral("Picture"), {view_layer_});
        section(p, QStringLiteral("Range"), {vt_clamp_, vt_clip_, vt_range_});
        if (auto* checks = find("vtChecksGroup")) {
            if (auto* cap = checks->findChild<QLabel*>(QStringLiteral("vtChecksGroupLabel"))) cap->hide();   // the head says it
            section(p, QStringLiteral("Colour checks"), {checks});
        }
        if (hud_flow) {
            vt_overlays_->setParent(hud);
            hud_flow->insert_widget(hud_flow->indexOf(view_mode_) + 1, vt_overlays_);
            // 21 Nov 2026 (user feedback): the AI fill is ONE button on the viewer's bar, on and off in one
            // click to compare, not an entry in Overlays. Its settings are Reconstruct's AI generation card.
            if (auto* fill = find("vtAiFill")) {
                if (QWidget* old = fill->parentWidget(); old && old->layout()) old->layout()->removeWidget(fill);
                fill->setParent(hud);
                hud_flow->insert_widget(hud_flow->indexOf(vt_overlays_) + 1, fill);
                fill->hide();   // sync_ai_generation shows it once there is a fill, or AI generation is on
                if (auto* drew = find("vtAiInvented")) {   // 1.3: right after it
                    if (QWidget* old = drew->parentWidget(); old && old->layout()) old->layout()->removeWidget(drew);
                    drew->setParent(hud);
                    hud_flow->insert_widget(hud_flow->indexOf(fill) + 1, drew);
                    drew->hide();
                }
            }
        }
        for (auto* b : p.panel->findChildren<QPushButton*>())
            connect(b, &QPushButton::toggled, this, [this] { sync_viewer_popover_buttons(); });
    }
    // Colour: the view, the working space and how the SDR is shown, behind one button that names
    // the view in use.
    {
        Popover p = popover(QString(), QStringLiteral("vtColourBtn"),
                            QStringLiteral("How the picture is shown on this screen: the view (RUDRA's or an "
                                           "OpenColorIO one), the working space, and how the SDR is brought in. "
                                           "View only: never rendered"),
                            vtool_);
        vt_colour_btn_ = p.button;
        for (const char* id : {"vtViewGroup", "vtWorkingGroup", "vtOriginalGroup"})
            section(p, QString(), {find(id)});
        row->insert_widget(row->indexOf(vt_flip_) + 1, vt_colour_btn_);
        connect(vt_ocio_, &QComboBox::currentIndexChanged, this, [this] { sync_viewer_popover_buttons(); });
    }
    // Exposure: the slider and its value stay; gamma, the brackets and Reset go one click away.
    {
        Popover p = popover(QStringLiteral("\u03b3 \u00b7 brackets \u25be"), QStringLiteral("vtMore"),
                            QStringLiteral("View gamma, the exposure brackets (Alt+B) and Reset (Alt+0)"), vtool_);
        vt_more_ = p.button;
        section(p, QStringLiteral("Gamma"), {vt_gamma_});
        std::vector<QWidget*> br(vt_brackets_.begin(), vt_brackets_.end());
        {
            auto* cap = new QLabel(QStringLiteral("Brackets"), p.panel);
            cap->setProperty("role", "pop-head");
            p.v->addWidget(cap);
            auto* r = new QWidget(p.panel);
            auto* h = new QHBoxLayout(r);
            h->setContentsMargins(0, 0, 0, 6);
            h->setSpacing(6);
            for (QWidget* w : br) {
                row->removeWidget(w);
                w->setParent(r);
                h->addWidget(w);
                w->show();
            }
            h->addStretch(1);
            p.v->addWidget(r);
        }
        section(p, QString(), {vt_reset_});
        row->insert_widget(row->indexOf(vt_ev_) + 1, vt_more_);
    }
    sync_viewer_popover_buttons();
}

void MainWindow::sync_viewer_popover_buttons() {
    if (vt_overlays_) {
        int on = 0;
        for (const char* id : {"changesBtn", "inventedBtn", "vtClamp", "vtClip", "vtRange", "vtGamut", "vtInvalid"})
            if (auto* b = findChild<QPushButton*>(QString::fromLatin1(id)); b && b->isChecked()) ++on;
        if (view_layer_ && (view_layer_->on() == "1" || view_layer_->on() == "2")) ++on;
        vt_overlays_->setText(on ? QStringLiteral("Overlays \u00b7 %1 \u25be").arg(on) : QStringLiteral("Overlays \u25be"));
        if (vt_overlays_->property("on").toBool() != (on > 0)) {
            vt_overlays_->setProperty("on", on > 0);
            vt_overlays_->style()->unpolish(vt_overlays_);
            vt_overlays_->style()->polish(vt_overlays_);
        }
    }
    if (vt_colour_btn_ && vt_ocio_) {
        QString name = vt_ocio_->currentText();
        if (name.size() > 30) name = name.left(29) + QStringLiteral("\u2026");
        vt_colour_btn_->setText(QStringLiteral("Colour: %1 \u25be").arg(name.isEmpty() ? QStringLiteral("RUDRA view") : name));
    }
    if (vt_more_) {
        const bool gamma = vt_gamma_ && std::abs(vt_gamma_->value() - 1.0) > 1e-9;
        vt_more_->setText(gamma ? QStringLiteral("\u03b3 %1 \u00b7 brackets \u25be").arg(vt_gamma_->value(), 0, 'f', 2)
                                : QStringLiteral("\u03b3 \u00b7 brackets \u25be"));
    }
}

void MainWindow::build_viewer_toolbar(QVBoxLayout* v, QWidget* parent) {
    vtool_ = new QWidget(parent);
    vtool_->setObjectName("viewerToolbar");
    auto* col = new QVBoxLayout(vtool_);
    col->setContentsMargins(12, 2, 12, 2);
    col->setSpacing(2);
    // 8 Nov 2026: a flow, not a fixed row: on a narrow window the buttons wrap onto a second
    // line instead of widening the viewer past the screen (the right rail was pushed off it).
    auto* row_host = new QWidget(vtool_);
    auto* row = new FlowLayout(row_host, 6, 4);
    col->addWidget(row_host);

    // ── exposure and gamma, as Nuke's viewer ──
    vt_stop_ = tool_button("f/stop", "vtStop",
                           "View exposure, only what you see, like Nuke's f/stop: never in the master. "
                           "Click: 0 \u2194 the last value (Alt+E). Alt+[ and Alt+] step half a stop.",
                           vtool_, true);
    connect(vt_stop_, &QPushButton::clicked, this, [this] { session_.toggle_view_ev(); });
    row->addWidget(vt_stop_);
    vt_ev_slider_ = new QSlider(Qt::Horizontal, vtool_);
    vt_ev_slider_->setObjectName("vtEvSlider");
    vt_ev_slider_->setRange(-100, 100);   // tenths of a stop
    vt_ev_slider_->setFixedWidth(140);
    vt_ev_slider_->setFocusPolicy(Qt::NoFocus);
    vt_ev_slider_->setToolTip("View exposure in stops. Double-click the field to reset.");
    connect(vt_ev_slider_, &QSlider::valueChanged, this, [this](int n) { session_.set_view_ev(n / 10.0); });
    row->addWidget(vt_ev_slider_);
    vt_ev_ = new QDoubleSpinBox(vtool_);
    vt_ev_->setObjectName("vtEv");
    vt_ev_->setRange(-10.0, 10.0);
    vt_ev_->setSingleStep(0.1);
    vt_ev_->setDecimals(1);
    vt_ev_->setSuffix(" EV");
    vt_ev_->setKeyboardTracking(false);
    vt_ev_->setToolTip("View exposure in stops (scroll or type; Shift for 1.0 steps)");
    connect(vt_ev_, &QDoubleSpinBox::valueChanged, this, [this](double ev) { session_.set_view_ev(ev); });
    row->addWidget(vt_ev_);
    vt_gamma_ = new QDoubleSpinBox(vtool_);
    vt_gamma_->setObjectName("vtGamma");
    vt_gamma_->setRange(0.2, 5.0);
    vt_gamma_->setSingleStep(0.1);
    vt_gamma_->setDecimals(2);
    vt_gamma_->setPrefix(QStringLiteral("\u03b3 "));
    vt_gamma_->setKeyboardTracking(false);
    vt_gamma_->setToolTip("View gamma, like Nuke's viewer gamma: pivots on the view peak. Never in the master.");
    connect(vt_gamma_, &QDoubleSpinBox::valueChanged, this, [this](double g) { session_.set_view_gamma(g); });
    row->addWidget(vt_gamma_);
    vt_reset_ = tool_button("Reset", "vtReset", "View exposure 0 and gamma 1 (Alt+0)", vtool_);
    connect(vt_reset_, &QPushButton::clicked, this, [this] { session_.reset_viewer(); });
    row->addWidget(vt_reset_);

    // ── brackets ──
    for (double ev : kBrackets) {
        auto* b = tool_button(signed_ev(ev), QStringLiteral("vtBracket%1").arg(int(ev)),
                              QStringLiteral("View at %1 EV (Alt+B steps through)").arg(signed_ev(ev)), vtool_, true);
        connect(b, &QPushButton::clicked, this, [this, ev] { session_.set_view_ev(ev); });
        row->addWidget(b);
        vt_brackets_.push_back(b);
    }

    // ── A | B ──
    row->add_gap(8);
    vt_a_ = source_box("vtA", "Left side of the wipe (A)", vtool_);
    vt_b_ = source_box("vtB", "Right side of the wipe (B)", vtool_);
    auto apply_ab = [this] {
        session_.set_wipe_sources(vt_a_->currentData().toString().toStdString(), vt_b_->currentData().toString().toStdString());
    };
    connect(vt_a_, &QComboBox::activated, this, apply_ab);
    connect(vt_b_, &QComboBox::activated, this, apply_ab);
    // 1.2.6: each side's letter and its box are one item, so a wrap never parts them
    for (auto [letter, box, id] : {std::tuple{"A", vt_a_, "vtAGroup"}, std::tuple{"B", vt_b_, "vtBGroup"}}) {
        auto* g = new QWidget(vtool_);
        g->setObjectName(QString::fromLatin1(id));
        auto* h = new QHBoxLayout(g);
        h->setContentsMargins(0, 0, 0, 0);
        h->setSpacing(5);
        auto* l = new QLabel(QString::fromLatin1(letter), g);
        l->setProperty("role", "vt-cap");
        l->setToolTip(box->toolTip());
        h->addWidget(l);
        box->setParent(g);
        h->addWidget(box);
        row->addWidget(g);
    }
    vt_flip_ = tool_button(QStringLiteral("A \u21c4 B"), "vtFlip", "Swap the wipe's sides; without a wipe, flip A and B (Alt+X)",
                           vtool_);
    connect(vt_flip_, &QPushButton::clicked, this, [this] { session_.flip_ab(); });
    row->addWidget(vt_flip_);

    // ── colour management (revision 7, colour_view.cpp): the OCIO view, the working space,
    // the original's way in, and the gamut and invalid-value checks ──
    build_colour_controls(row);

    // ── range inspection ──
    row->add_gap(8);
    vt_clamp_ = tool_button("Clamped", "vtClamp",
                            "Where the ORIGINAL is clamped: red stripes all channels at code 254+, thin orange some "
                            "channels, blue crushed at code 1 or below. The same pixels on both sides, at any exposure (Alt+K)",
                            vtool_, true);
    connect(vt_clamp_, &QPushButton::clicked, this, [this] { session_.toggle_clamp_zebra(); });
    row->addWidget(vt_clamp_);
    vt_clip_ = tool_button("Clipping", "vtClip",
                           "Yellow stripes where the picture clips on screen at this exposure and peak (Alt+L)", vtool_, true);
    connect(vt_clip_, &QPushButton::clicked, this, [this] { session_.toggle_clip_zebra(); });
    row->addWidget(vt_clip_);
    vt_range_ = tool_button("Range", "vtRange",
                            "Stops above the SDR white (203 nits): grey ramp the SDR's own range, blue +0\u20131, green +1\u20132, "
                            "yellow +2\u20133, orange +3\u20134, red +4 and up. White hatching: clamped in the original (Alt+R)",
                            vtool_, true);
    connect(vt_range_, &QPushButton::clicked, this, [this] { session_.set_view_layer(session_.view_layer == 4 ? 0 : 4); });
    row->addWidget(vt_range_);
    // 4 Nov 2026: the AI fill in the viewer, frame by frame and playing (the preview's
    // frames, or the last finished fill); with Changes on, what it invented.
    auto* fill = tool_button("AI fill", "vtAiFill",
                             "The AI fill, on or off: on, the viewer shows the fill on every frame that has one (the "
                             "preview's frames, Reconstruct \u203a AI generation \u203a Preview AI fill, or the last finished "
                             "fill); off, the master. What AI drew shows what it invented.",
                             vtool_, true);
    connect(fill, &QPushButton::toggled, this, [this](bool on) {
        // 24 Nov 2026 (Ahmed, the approved mockup): the button is the AI fill, on or off for this session, and
        // Reconstruct > AI generation is the same switch: grey off, blue on. On, the viewer shows the fill wherever
        // there is one (set_deliver); with none yet, it says how to make one. Off, the master.
        if (on != ai_generation_) set_deliver("ai_generation", on ? "1" : "0");
        if (on && !aifill_runs_here_ && !aifill_why_.isEmpty())
            statusBar()->showMessage(QStringLiteral("AI fill is on. ") + aifill_why_, 12000);
        else if (on && !genfill_preview_.has_value() && genfill_last_out_.isEmpty())
            statusBar()->showMessage(QStringLiteral("AI fill is on: Reconstruct \u203a AI generation \u203a Preview AI fill makes "
                                                    "one on this frame"),
                                     10000);
        sync_ai_generation();
    });
    row->addWidget(fill);
    // 1.3 (23 Nov 2026): the check every fill needs, beside the fill itself (it was only at the bottom of
    // Reconstruct > AI generation, as "Show what it invented"). Shown while there is a preview.
    auto* drew = tool_button("What AI drew", "vtAiInvented",
                             "White where the AI fill invented the pixels on the previewed frame; click again for the "
                             "picture. The same as Reconstruct \u203a AI generation \u203a Show what it invented",
                             vtool_, true);
    connect(drew, &QPushButton::clicked, this, [this](bool on) { show_genfill_preview(on ? "mask" : "fill"); });
    row->addWidget(drew);
    vt_badge_ = new QLabel(vtool_);
    vt_badge_->setObjectName("vtBadge");
    vt_badge_->setToolTip("The viewer is not neutral: what you see is gained, the master is not");
    vt_badge_->hide();
    row->addWidget(vt_badge_);

    range_line_ = new QLabel(vtool_);
    range_line_->setObjectName("rangeLine");
    range_line_->setTextFormat(Qt::PlainText);
    range_line_->setWordWrap(true);   // a long readout wraps; it never widens the viewer (8 Nov 2026)
    range_line_->setMinimumWidth(0);
    range_line_->setToolTip("Range of this frame in absolute nits; the view exposure does not change these numbers");
    col->addWidget(range_line_);
    v->addWidget(vtool_);

    // ── keys (Alt+…: the page's catalog uses none of them) ──
    // QActions on the window, as the catalog's keys are (not in the catalog:
    // AppActions checks the page's keys, and these are native only).
    build_viewer_popovers(row);   // 1.2.6

    auto key = [this](QKeyCombination k, std::function<void()> f) {
        auto* a = new QAction(this);
        a->setShortcut(QKeySequence(k));
        a->setShortcutContext(Qt::WindowShortcut);
        addAction(a);
        connect(a, &QAction::triggered, this, std::move(f));
    };
    key(Qt::ALT | Qt::Key_BracketLeft, [this] { session_.nudge_view_ev(-0.5); });
    key(Qt::ALT | Qt::Key_BracketRight, [this] { session_.nudge_view_ev(0.5); });
    key(Qt::ALT | Qt::Key_E, [this] { session_.toggle_view_ev(); });
    key(Qt::ALT | Qt::Key_0, [this] { session_.reset_viewer(); });
    key(Qt::ALT | Qt::Key_X, [this] { session_.flip_ab(); });
    key(Qt::ALT | Qt::Key_K, [this] { session_.toggle_clamp_zebra(); });
    key(Qt::ALT | Qt::Key_L, [this] { session_.toggle_clip_zebra(); });
    key(Qt::ALT | Qt::Key_R, [this] { session_.set_view_layer(session_.view_layer == 4 ? 0 : 4); });
    key(Qt::ALT | Qt::Key_B, [this] {
        // The next bracket above the current exposure, wrapping to the first.
        double next = kBrackets[0];
        for (double ev : kBrackets)
            if (ev > session_.view_ev + 1e-9) {
                next = ev;
                break;
            }
        session_.set_view_ev(next);
    });
}

void MainWindow::sync_viewer_toolbar() {
    if (!vtool_) return;
    vtool_->setVisible(workspace_ != "simple");   // Simple keeps its one View menu
    {
        const QSignalBlocker b1(vt_ev_slider_), b2(vt_ev_), b3(vt_gamma_);
        vt_ev_slider_->setValue(int(std::lround(session_.view_ev * 10.0)));
        vt_ev_->setValue(session_.view_ev);
        vt_gamma_->setValue(session_.view_gamma);
    }
    vt_stop_->setChecked(session_.view_ev != 0.0);
    for (std::size_t i = 0; i < vt_brackets_.size(); ++i) vt_brackets_[i]->setChecked(session_.view_ev == kBrackets[i]);
    auto pick = [](QComboBox* c, const std::string& s) {
        const QSignalBlocker b(c);
        c->setCurrentIndex(std::max(0, c->findData(QString::fromStdString(s))));
    };
    pick(vt_a_, session_.wipe_left);
    pick(vt_b_, session_.wipe_right);
    vt_clamp_->setChecked(session_.clamp_zebra);
    vt_clip_->setChecked(session_.clip_zebra);
    vt_range_->setChecked(session_.view_layer == 4);
    // The SDR sources, the clamp zebra and Range need the frame's SDR.
    const bool have_sdr = viewer_has_sdr();
    for (QComboBox* c : {vt_a_, vt_b_}) {
        if (auto* m = qobject_cast<QStandardItemModel*>(c->model()))
            if (auto* it = m->item(0)) it->setEnabled(have_sdr);
    }
    vt_clamp_->setEnabled(have_sdr);
    vt_clamp_->setVisible(have_sdr);   // 1.2.6: offered when the frame's SDR is there to read
    vt_range_->setEnabled(true);
    // Always visible when the view is gained, so no screenshot passes for the grade.
    if (session_.viewer_neutral()) {
        vt_badge_->hide();
    } else {
        vt_badge_->setText(QStringLiteral("VIEW %1 EV \u00b7 \u03b3 %2").arg(signed_ev(session_.view_ev)).arg(session_.view_gamma, 0, 'f', 2));
        vt_badge_->show();
    }
    sync_colour_controls();
    sync_viewer_popover_buttons();   // 1.2.6: the names and counts on the panel buttons
    range_line_->setText(range_readout_text());
}

bool MainWindow::viewer_has_sdr() const {
    if (aces_routed()) return aces_sdr_img_ != nullptr;
    return current_frame_ != nullptr;
}

QString MainWindow::range_readout_text() const {
    // Revision 7: the frame's gamut and invalid-value checks, when on, under the range.
    const QString checks = colour_check_text();
    if (!measure_ || !measure_->range) return checks.isEmpty() ? QStringLiteral(" ") : checks;
    const RangeStats& r = *measure_->range;
    // 14 Nov 2026: one readout per line (SDR, RUDRA, the clamped area); one long line wrapped mid-readout.
    QString t = QStringLiteral("SDR  clamped %1 (R %2 \u00b7 G %3 \u00b7 B %4) \u00b7 crushed %5 \u00b7 range %6 st\n")
                    .arg(pct(r.clamped_pct), pct(r.clamped_channel_pct[0]), pct(r.clamped_channel_pct[1]),
                         pct(r.clamped_channel_pct[2]), pct(r.crushed_pct))
                    .arg(r.sdr_range_stops, 0, 'f', 1);
    t += QStringLiteral("RUDRA  peak %1 nits (%2 st over SDR white) \u00b7 above white %3 \u00b7 above 1,000 nits %4 \u00b7 range %5 st")
             .arg(QString::number(r.peak_nits, 'f', 0), stops(r.peak_stops_over_white), pct(r.above_white_pct),
                  pct(r.above_1000_pct))
             .arg(r.hdr_range_stops, 0, 'f', 1);
    if (r.clamped_pixels == 0) {
        t += QStringLiteral("\nIn the clamped area: none");
    } else {
        t += QStringLiteral("\nIn the clamped area  median %1 st \u00b7 99th %2 st")
                 .arg(stops(r.clamped_median_stops), stops(r.clamped_p99_stops));
        if (r.clamped_unchanged_pct)
            t += QStringLiteral(" \u00b7 not rebuilt %1").arg(pct(*r.clamped_unchanged_pct));
    }
    if (!checks.isEmpty()) t += QStringLiteral("\n") + checks;
    return t;
}

}  // namespace rudra::app
