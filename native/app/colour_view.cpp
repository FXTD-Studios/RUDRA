// The colour-managed viewer's controls (17 Nov 2026, docs/view.spec.md revision 7): the
// picture through an OpenColorIO display and view as Nuke's viewer shows it (the built-in
// ACES 2.0 studio config, $OCIO or a config.ocio), in ACEScg or ACES2065-1, the original
// SDR through an input transform, and the range checks: out of gamut, invalid values and
// clipped by the view, with the probe's working-space values and display codes. Like the
// rest of the viewer toolbar, all of it is view state: none of it reaches a master.
#include <QAction>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "flow_layout.hpp"
#include "main_window.hpp"
#include "rudra/core/view.hpp"
#include "rudra/infer/ocio_viewer.hpp"
#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {

constexpr Primaries kGamuts[] = {Primaries::Ap1, Primaries::Rec2020, Primaries::P3D65, Primaries::Rec709};
constexpr const char* kGamutNames[] = {"AP1", "Rec.2020", "P3-D65", "Rec.709"};
const QString kRudraReading = QStringLiteral("rudra");   // the original as the Source picker reads it

Primaries gamut_of(int i) { return kGamuts[std::clamp(i, 0, 3)]; }

QString key(const QString& display, const QString& view) { return display + QLatin1Char('\t') + view; }

void restyle(QWidget* w, const char* state) {
    if (w->property("state").toString() == state) return;
    w->setProperty("state", state);
    w->style()->unpolish(w);
    w->style()->polish(w);
}

}  // namespace

void MainWindow::load_colour_settings() {
    QSettings st;
    ocio_display_ = st.value("viewer/ocioDisplay").toString();
    ocio_view_ = st.value("viewer/ocioView").toString();
    ocio_working_ = std::clamp(st.value("viewer/working", 0).toInt(), 0, 1);
    ocio_input_ = st.value("viewer/original").toString();
    gamut_of_ = std::clamp(st.value("viewer/gamutOf", 0).toInt(), 0, 3);
}

void MainWindow::open_colour_config() {
    ocio_cfg_.reset();
    ocio_cfg_error_.clear();
    if (!ocio_viewer::available()) {
        ocio_cfg_error_ = QStringLiteral("This build has no OpenColorIO: the viewer shows RUDRA's own view.");
        return;
    }
    QSettings st;
    const QString kind = st.value("viewer/ocioSource", "builtin").toString();
    ocio_viewer::ConfigSource src;
    src.kind = kind == "env" ? ocio_viewer::ConfigKind::Environment
               : kind == "file" ? ocio_viewer::ConfigKind::File
                                : ocio_viewer::ConfigKind::Builtin;
    src.path = st.value("viewer/ocioFile").toString().toStdString();
    auto cfg = ocio_viewer::Config::open(src);
    if (!cfg && src.kind != ocio_viewer::ConfigKind::Builtin) {
        // A config that went away (a moved file, $OCIO unset): the built-in one, and say so.
        ocio_cfg_error_ = QString::fromStdString(cfg.error().message + " " + cfg.error().detail).trimmed() +
                          QStringLiteral(" The built-in ACES 2.0 config is used.");
        cfg = ocio_viewer::Config::open({});
    }
    if (!cfg) {
        ocio_cfg_error_ = QString::fromStdString(cfg.error().message + " " + cfg.error().detail).trimmed();
        return;
    }
    ocio_cfg_ = cfg.value();
    // A view this config does not have: RUDRA's own view until one is chosen.
    if (!ocio_display_.isEmpty()) {
        const auto views = ocio_cfg_->views(ocio_display_.toStdString());
        if (std::find(views.begin(), views.end(), ocio_view_.toStdString()) == views.end()) {
            ocio_display_.clear();
            ocio_view_.clear();
        }
    }
    const auto inputs = ocio_cfg_->inputs();
    const bool known = ocio_input_ == kRudraReading ||
                       std::any_of(inputs.begin(), inputs.end(), [&](const auto& i) { return QString::fromStdString(i.id) == ocio_input_; });
    // 1.3.0 (24 Nov 2026, Ahmed): by default the original is read as an sRGB texture, as Nuke reads an SDR plate in
    // an ACES 2.0 config (Input Transform: sRGB Encoded Rec.709 (sRGB)), so RUDRA's A | B wipe is the one done in Nuke.
    // The inverse of the SDR view (the original as it was graded) is the next choice down; without an sRGB texture
    // space in the config, the first input. A choice the user made is kept (viewer/original).
    if (!known) {
        const QString srgb = QStringLiteral("space:sRGB Encoded Rec.709 (sRGB)");
        const bool has_srgb = std::any_of(inputs.begin(), inputs.end(), [&](const auto& i) { return QString::fromStdString(i.id) == srgb; });
        ocio_input_ = has_srgb ? srgb : inputs.empty() ? kRudraReading : QString::fromStdString(inputs.front().id);
    }
}

void MainWindow::make_colour_transforms() {
    ocio_view_tf_.reset();
    ocio_input_tf_.reset();
    ocio_transform_error_.clear();
    if (!ocio_cfg_ || ocio_display_.isEmpty()) return;
    const WorkingSpace w = WorkingSpace(ocio_working_);
    auto v = ocio_cfg_->view(ocio_display_.toStdString(), ocio_view_.toStdString(), w);
    if (!v) {
        ocio_transform_error_ = QString::fromStdString(v.error().message + " " + v.error().detail).trimmed();
        return;
    }
    ocio_view_tf_ = v.value();
    if (ocio_input_ != kRudraReading && !ocio_input_.isEmpty()) {
        auto in = ocio_cfg_->input(ocio_input_.toStdString(), w);
        if (in) ocio_input_tf_ = in.value();
        else ocio_transform_error_ = QString::fromStdString(in.error().message + " " + in.error().detail).trimmed();
    }
}

void MainWindow::apply_colour(ViewParams& v) const {
    v.working = WorkingSpace(ocio_working_);
    // The learned route's ACES monitor already puts out display light: the OCIO view would
    // be applied twice. It shows the picture as it is until the monitor is Scene light.
    const bool monitor_applies = aces_routed() && aces_monitor_ != "scene";
    v.ocio = monitor_applies ? nullptr : ocio_view_tf_;
    v.original = v.ocio ? ocio_input_tf_ : nullptr;
    v.gamut_zebra = gamut_check_;
    v.gamut = gamut_of(gamut_of_);
    v.invalid_zebra = invalid_check_;
}

void MainWindow::refresh_colour() {
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        ViewParams v = viewer_->view();
        apply_colour(v);
        viewer_->set_view(v);
    }
#endif
    check_cache_key_ = nullptr;
    sync_viewer_toolbar();
}

void MainWindow::set_colour_view(const QString& display, const QString& view) {
    ocio_display_ = view.isEmpty() ? QString() : display;
    ocio_view_ = display.isEmpty() ? QString() : view;
    QSettings st;
    st.setValue("viewer/ocioDisplay", ocio_display_);
    st.setValue("viewer/ocioView", ocio_view_);
    if (!ocio_display_.isEmpty() && aces_routed() && aces_monitor_ != "scene") {
        set_aces_monitor("scene");
        statusBar()->showMessage(QStringLiteral("The learned route's monitor is now Scene light: the OCIO view makes the "
                                                "display picture."),
                                 8000);
    }
    make_colour_transforms();
    refresh_colour();
}

QStringList MainWindow::tone_map_for_fill() {
    QStringList did;
    if (!QSettings().value("viewer/fillToneMapped", true).toBool()) return did;
#ifdef RUDRA_APP_VIEWER
    // An HDR screen shows the nits as they are: nothing to do.
    if (viewer_ && viewer_->view().target.path != OutputPath::SdrPqSimulation) return did;
#endif
    if (ocio_cfg_ && ocio_display_.isEmpty()) {   // RUDRA's own view: the first ACES 2.0 SDR preset (sRGB first)
        for (const auto& p : ocio_cfg_->presets()) {
            const QString d = QString::fromStdString(p.display), v = QString::fromStdString(p.view);
            if (!v.startsWith(QStringLiteral("ACES 2.0 - SDR 100 nits"))) continue;
            set_colour_view(d, v);
            if (ocio_display_ == d && ocio_view_ == v) {
                fill_view_set_colour_ = std::make_pair(d, v);
                did << QStringLiteral("tone-mapped view (%1)").arg(QString::fromStdString(p.label));
            }
            break;
        }
    }
    if (session_.view_ev == 0.0) {   // only from 0 EV: an exposure the user set stays theirs
        session_.set_view_ev(-2.0);
        fill_view_set_ev_ = -2.0;
        did << QStringLiteral("view at \u22122 EV");
    }
    return did;
}

void MainWindow::untone_map_for_fill() {
    if (fill_view_set_colour_ && ocio_display_ == fill_view_set_colour_->first && ocio_view_ == fill_view_set_colour_->second)
        set_colour_view(QString(), QString());   // back to RUDRA's own view, unless the user picked another
    fill_view_set_colour_.reset();
    if (fill_view_set_ev_ && session_.view_ev == *fill_view_set_ev_) session_.set_view_ev(0.0);
    fill_view_set_ev_.reset();
}

void MainWindow::build_colour_controls(FlowLayout* row) {
    load_colour_settings();
    open_colour_config();
    make_colour_transforms();
    row->add_gap(8);
    vt_ocio_ = new QComboBox(vtool_);
    vt_ocio_->setObjectName("vtOcio");
    vt_ocio_->setFocusPolicy(Qt::NoFocus);
    vt_ocio_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    vt_ocio_->setMinimumContentsLength(22);
    vt_ocio_->setToolTip("What the picture goes through to the screen, like Nuke's viewer process: RUDRA's own view "
                         "(absolute nits, no tone mapping) or an OpenColorIO display and view. On an SDR screen the "
                         "view's code values are shown as Nuke shows them; on an HDR screen their light, so PQ 1000 "
                         "shows at its real level. View > Colour management… picks the config.");
    connect(vt_ocio_, &QComboBox::activated, this, [this](int i) {
        const QStringList dv = vt_ocio_->itemData(i).toString().split(QLatin1Char('\t'));
        set_colour_view(dv.value(0), dv.value(1));
    });
    // 18 Nov 2026 (1.2.6, user feedback: "which box is what?"): every box has its name in front
    // of it, and the name and the box are one item, so a narrow row never parts them.
    // The checks are one group of their own, named as what they mark.
    auto group = [this, row](const char* id, const QString& caption, std::initializer_list<QWidget*> parts) {
        auto* g = new QWidget(vtool_);
        g->setObjectName(QString::fromLatin1(id));
        auto* h = new QHBoxLayout(g);
        h->setContentsMargins(0, 0, 0, 0);
        h->setSpacing(5);
        auto* cap = new QLabel(caption, g);
        cap->setObjectName(QString::fromLatin1(id) + QStringLiteral("Label"));
        cap->setProperty("role", "vt-cap");
        h->addWidget(cap);
        for (QWidget* p : parts) {
            p->setParent(g);
            h->addWidget(p);
        }
        row->addWidget(g);
        return cap;
    };
    group("vtViewGroup", QStringLiteral("View"), {vt_ocio_})
        ->setToolTip(QStringLiteral("How the picture is shown on this screen. Changes the viewer only, never a render"));
    vt_working_ = new QComboBox(vtool_);
    vt_working_->setObjectName("vtWorking");
    vt_working_->setFocusPolicy(Qt::NoFocus);
    vt_working_->addItem("ACEScg", 0);
    vt_working_->addItem("ACES2065-1 (AP0)", 1);
    vt_working_->setToolTip("The working space the view is fed and the probe reads, as a render to that space holds "
                            "it (1.0 = diffuse white, 203 nits): what Nuke shows for the EXR read as this space.");
    connect(vt_working_, &QComboBox::activated, this, [this](int i) {
        ocio_working_ = vt_working_->itemData(i).toInt();
        QSettings().setValue("viewer/working", ocio_working_);
        make_colour_transforms();
        refresh_colour();
    });
    group("vtWorkingGroup", QStringLiteral("Working space"), {vt_working_})
        ->setToolTip(QStringLiteral("The space the view is fed and the probe reads. Changes the viewer only, never a render"));
    vt_original_ = new QComboBox(vtool_);
    vt_original_->setObjectName("vtOriginal");
    vt_original_->setFocusPolicy(Qt::NoFocus);
    vt_original_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    vt_original_->setMinimumContentsLength(24);   // 1.2.6: "Inverse ACES 2.0 SDR view" was cut
    vt_original_->setToolTip("How the original SDR enters the working space when it is shown through an OCIO view (the "
                             "SDR source and the wipe's SDR side). The inverse of the SDR view makes it look as it was "
                             "graded through that view; a texture space reads its codes as linear light.");
    connect(vt_original_, &QComboBox::activated, this, [this](int i) {
        ocio_input_ = vt_original_->itemData(i).toString();
        QSettings().setValue("viewer/original", ocio_input_);
        make_colour_transforms();
        refresh_colour();
    });
    group("vtOriginalGroup", QStringLiteral("SDR shown as"), {vt_original_})
        ->setToolTip(QStringLiteral("Only for an OCIO view: how the original SDR (the A side of the wipe) is brought "
                                    "into the working space. Hidden while the view is RUDRA's own"));

    row->add_gap(8);
    vt_gamut_ = new QPushButton("Out of gamut", vtool_);
    vt_gamut_->setObjectName("vtGamut");
    vt_gamut_->setCheckable(true);
    vt_gamut_->setFocusPolicy(Qt::NoFocus);
    vt_gamut_->setToolTip("Magenta bands where the picture is outside the gamut on the right: AP1 is what an ACEScg "
                          "render holds without negative values (Alt+G)");
    connect(vt_gamut_, &QPushButton::clicked, this, [this] {
        gamut_check_ = !gamut_check_;
        refresh_colour();
    });
    vt_gamut_of_ = new QComboBox(vtool_);
    vt_gamut_of_->setObjectName("vtGamutOf");
    vt_gamut_of_->setFocusPolicy(Qt::NoFocus);
    for (int i = 0; i < 4; ++i) vt_gamut_of_->addItem(kGamutNames[i], i);
    vt_gamut_of_->setToolTip("The gamut \"Out of gamut\" marks against; picking one turns the check on");
    connect(vt_gamut_of_, &QComboBox::activated, this, [this](int i) {
        gamut_of_ = vt_gamut_of_->itemData(i).toInt();
        QSettings().setValue("viewer/gamutOf", gamut_of_);
        gamut_check_ = true;
        refresh_colour();
    });
    vt_invalid_ = new QPushButton("Invalid values", vtool_);
    vt_invalid_->setObjectName("vtInvalid");
    vt_invalid_->setCheckable(true);
    vt_invalid_->setFocusPolicy(Qt::NoFocus);
    vt_invalid_->setToolTip("Solid green where a value is not a colour a render can hold: NaN or infinite, outside AP0 "
                            "(no colour the eye sees), or above 65,504 in the working space (a half-float EXR's "
                            "largest value) (Alt+I)");
    connect(vt_invalid_, &QPushButton::clicked, this, [this] {
        invalid_check_ = !invalid_check_;
        refresh_colour();
    });
    group("vtChecksGroup", QStringLiteral("Mark"), {vt_gamut_, vt_gamut_of_, vt_invalid_})
        ->setToolTip(QStringLiteral("Overlays on the viewer that mark problem pixels; they are never rendered"));
    vt_colour_ = new QLabel(vtool_);
    vt_colour_->setObjectName("vtColour");
    vt_colour_->setTextFormat(Qt::PlainText);
    vt_colour_->setToolTip("What the picture goes through to the screen");
    row->addWidget(vt_colour_);

    auto key_action = [this](QKeyCombination k, std::function<void()> f) {
        auto* a = new QAction(this);
        a->setShortcut(QKeySequence(k));
        a->setShortcutContext(Qt::WindowShortcut);
        addAction(a);
        connect(a, &QAction::triggered, this, std::move(f));
    };
    key_action(Qt::ALT | Qt::Key_G, [this] {
        gamut_check_ = !gamut_check_;
        refresh_colour();
    });
    key_action(Qt::ALT | Qt::Key_I, [this] {
        invalid_check_ = !invalid_check_;
        refresh_colour();
    });
}

void MainWindow::sync_colour_controls() {
    if (!vt_ocio_) return;
    // The boxes are filled again only when the config changes (never from their own signal).
    const bool refill = !colour_filled_once_ || colour_filled_ != ocio_cfg_.get();
    colour_filled_once_ = true;
    colour_filled_ = ocio_cfg_.get();
    {
        const QSignalBlocker b(vt_ocio_);
        if (refill) vt_ocio_->clear();
        if (refill) {   // 1.2.6: short enough for the box (it was cut at "no tone mappi"); the rest on hover
            vt_ocio_->addItem(QStringLiteral("RUDRA view (nits)"), QString());
            vt_ocio_->setItemData(vt_ocio_->count() - 1, QStringLiteral("RUDRA view: absolute nits, no tone mapping"),
                                  Qt::ToolTipRole);
        }
        if (refill && ocio_cfg_) {
            for (const auto& p : ocio_cfg_->presets())
                vt_ocio_->addItem(QString::fromStdString(p.label),
                                  key(QString::fromStdString(p.display), QString::fromStdString(p.view)));
            vt_ocio_->insertSeparator(vt_ocio_->count());
            for (const std::string& d : ocio_cfg_->displays())
                for (const std::string& v : ocio_cfg_->views(d))
                    vt_ocio_->addItem(QString::fromStdString(d + "  ·  " + v),
                                      key(QString::fromStdString(d), QString::fromStdString(v)));
        }
        const int at = ocio_display_.isEmpty() ? 0 : vt_ocio_->findData(key(ocio_display_, ocio_view_));
        vt_ocio_->setCurrentIndex(std::max(0, at));
        vt_ocio_->setEnabled(ocio_cfg_ != nullptr);
    }
    {
        const QSignalBlocker b(vt_working_);
        vt_working_->setCurrentIndex(std::max(0, vt_working_->findData(ocio_working_)));
    }
    {
        const QSignalBlocker b(vt_original_);
        if (refill) {
            vt_original_->clear();
            if (ocio_cfg_)
                for (const auto& i : ocio_cfg_->inputs())
                    vt_original_->addItem(QString::fromStdString(i.label), QString::fromStdString(i.id));
            vt_original_->addItem(QStringLiteral("As RUDRA reads it (Source picker)"), kRudraReading);
        }
        vt_original_->setCurrentIndex(std::max(0, vt_original_->findData(ocio_input_)));
        // The input transform is for an OCIO view; RUDRA's view reads the original its own way.
        // 1.2.6: shown only while it does something (an OCIO view); a box that does nothing confused users.
        if (auto* g = vtool_->findChild<QWidget*>("vtOriginalGroup")) g->setVisible(ocio_cfg_ != nullptr && ocio_view_tf_ != nullptr);
    }
    {
        const QSignalBlocker b(vt_gamut_of_);
        vt_gamut_of_->setCurrentIndex(gamut_of_);
    }
    vt_gamut_->setChecked(gamut_check_);
    vt_invalid_->setChecked(invalid_check_);
    vt_working_->setToolTip(QStringLiteral("The working space the view is fed and the probe reads (%1), as a render to "
                                           "that space holds it (1.0 = diffuse white, 203 nits).")
                                .arg(ocio_working_ == 1 ? "ACES2065-1" : "ACEScg"));
}

void MainWindow::show_colour_status(const ViewerStatus& s) {
#ifdef RUDRA_APP_VIEWER
    if (!vt_colour_) return;
    QString text, tip;
    const char* state = "";
    if (aces_routed() && aces_monitor_ != "scene" && ocio_view_tf_) {
        text = QStringLiteral("ACES monitor on: OCIO view off");
        tip = QStringLiteral("The learned route's monitor already puts out ACES 2.0 display light. View > Monitor > "
                             "Scene light hands the picture to the OCIO view.");
        state = "warn";
    } else if (!ocio_transform_error_.isEmpty() && !ocio_display_.isEmpty()) {
        text = QStringLiteral("OCIO view not shown");
        tip = ocio_transform_error_;
        state = "warn";
    } else if (!s.colour_error.empty()) {
        text = QStringLiteral("OCIO view not shown");
        tip = QString::fromStdString(s.colour_error);
        state = "warn";
    } else {
        // Before the first frame the viewer has shown nothing: say what it will show.
        const std::string what = s.has_frame && !s.colour.empty() ? s.colour
                                 : ocio_view_tf_ && !(aces_routed() && aces_monitor_ != "scene")
                                     ? ocio_view_text(*ocio_view_tf_)
                                     : std::string("RUDRA view");
        text = QString::fromStdString(what);
        tip = ocio_cfg_ ? QString::fromStdString(ocio_cfg_->name()) : ocio_cfg_error_;
        if (!ocio_cfg_error_.isEmpty() && ocio_cfg_) tip += QStringLiteral("\n") + ocio_cfg_error_;
    }
    vt_colour_->setText(text);
    vt_colour_->setToolTip(tip);
    restyle(vt_colour_, state);
#else
    (void)s;
#endif
}

QString MainWindow::colour_check_text() const {
    if (!gamut_check_ && !invalid_check_) return {};
    const NetworkLinearImage* shown = aces_routed() ? aces_model_img_.get() : (measure_ ? &measure_->model : nullptr);
    if (!shown || shown->width() <= 0) return {};
    if (check_cache_key_ == shown && check_cache_gamut_ == (gamut_check_ ? gamut_of_ : -1) &&
        check_cache_working_ == (invalid_check_ ? ocio_working_ : -1))
        return check_cache_text_;
    Primaries source = Primaries::Rec709;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) source = viewer_->view().source;
#endif
    const ViewMatrices m = view_matrices(source, WorkingSpace(ocio_working_), gamut_of(gamut_of_), 203.0);
    const PlanarBuffer& b = shown->buffer();
    const std::size_t n = b.plane_size();
    std::size_t outside = 0, nan = 0, ap0 = 0, half = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float h[3] = {b.plane(0)[i], b.plane(1)[i], b.plane(2)[i]};
        if (gamut_check_ && std::isfinite(h[0]) && std::isfinite(h[1]) && std::isfinite(h[2]) && out_of_gamut(h, m.chk))
            ++outside;
        if (invalid_check_ && invalid_value(h, m.work, m.ap0)) {
            if (!std::isfinite(h[0]) || !std::isfinite(h[1]) || !std::isfinite(h[2])) ++nan;
            else if (out_of_gamut(h, m.ap0)) ++ap0;
            else ++half;
        }
    }
    auto pct = [n](std::size_t k) {
        const double v = n ? 100.0 * double(k) / double(n) : 0.0;
        return QString::number(v, 'f', v < 10.0 ? 2 : 1) + QStringLiteral(" %");
    };
    QStringList parts;
    if (gamut_check_) parts << QStringLiteral("outside %1 %2").arg(kGamutNames[std::clamp(gamut_of_, 0, 3)], pct(outside));
    if (invalid_check_) {
        const std::size_t bad = nan + ap0 + half;
        parts << (bad == 0 ? QStringLiteral("invalid none")
                           : QStringLiteral("invalid %1 (NaN or infinite %2 · outside AP0 %3 · above half float %4)")
                                 .arg(pct(bad))
                                 .arg(nan)
                                 .arg(ap0)
                                 .arg(half));
    }
    check_cache_key_ = shown;
    check_cache_gamut_ = gamut_check_ ? gamut_of_ : -1;
    check_cache_working_ = invalid_check_ ? ocio_working_ : -1;
    check_cache_text_ = QStringLiteral("Checks  ") + parts.join(QStringLiteral(" · "));
    return check_cache_text_;
}

std::vector<std::array<QString, 3>> MainWindow::colour_probe_rows(const float rgb[3]) const {
    Primaries source = Primaries::Rec709;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) source = viewer_->view().source;
#endif
    const WorkingSpace w = WorkingSpace(ocio_working_);
    const bool monitor_applies = aces_routed() && aces_monitor_ != "scene";
    const PixelReading r = read_pixel(rgb, source, w, gamut_of(gamut_of_), monitor_applies ? nullptr : ocio_view_tf_.get());
    std::vector<std::array<QString, 3>> rows;
    auto num = [](double v) { return std::isfinite(v) ? QString::number(v, 'f', std::abs(v) < 10.0 ? 4 : 2) : QStringLiteral("NaN"); };
    rows.push_back({QString::fromLatin1(working_name(w)),
                    num(r.working[0]) + QLatin1Char(' ') + num(r.working[1]) + QLatin1Char(' ') + num(r.working[2]), QString()});
    if (r.has_code) {
        auto code = [](float v) { return QString::number(std::lround(std::clamp(v, 0.0f, 1.0f) * 1023.0f)); };
        const bool clipped = std::max({r.code[0], r.code[1], r.code[2]}) >= float(ocio_view_tf_->clip_code);
        rows.push_back({QStringLiteral("View code"),
                        code(r.code[0]) + QLatin1Char(' ') + code(r.code[1]) + QLatin1Char(' ') + code(r.code[2]) +
                            QStringLiteral(" /1023") + (clipped ? QStringLiteral(" clipped") : QString()),
                        clipped ? QStringLiteral("warn") : QString()});
    }
    if (gamut_check_ || r.out_of_gamut)
        rows.push_back({QStringLiteral("Gamut"),
                        (r.out_of_gamut ? QStringLiteral("outside ") : QStringLiteral("inside ")) +
                            QString::fromLatin1(kGamutNames[std::clamp(gamut_of_, 0, 3)]),
                        r.out_of_gamut ? QStringLiteral("warn") : QString()});
    if (r.invalid) rows.push_back({QStringLiteral("Invalid"), QString::fromStdString(r.why), QStringLiteral("warn")});
    return rows;
}

void MainWindow::open_colour_dialog() {
    QDialog dlg(this);
    dlg.setObjectName("colourDialog");
    dlg.setWindowTitle("Colour management");
    auto* v = new QVBoxLayout(&dlg);
    auto* intro = new QLabel("The config the viewer's OpenColorIO views come from. Pick a display and view on the viewer "
                             "toolbar, as in Nuke's viewer.", &dlg);
    intro->setWordWrap(true);
    v->addWidget(intro);
    QSettings st;
    const QString kind = st.value("viewer/ocioSource", "builtin").toString();
    auto* builtin = new QRadioButton(QStringLiteral("Built-in ACES 2.0 studio config (%1)").arg(ocio_viewer::kBuiltinConfig), &dlg);
    builtin->setObjectName("ocioBuiltin");
    const QString env = qEnvironmentVariable("OCIO");
    auto* envr = new QRadioButton(!env.isEmpty() ? QStringLiteral("$OCIO: %1").arg(env)
                                                 : QStringLiteral("$OCIO (not set on this computer)"),
                                  &dlg);
    envr->setObjectName("ocioEnv");
    envr->setEnabled(!env.isEmpty());
    auto* file = new QRadioButton("A config file:", &dlg);
    file->setObjectName("ocioFile");
    auto* path = new QLineEdit(st.value("viewer/ocioFile").toString(), &dlg);
    path->setObjectName("ocioPath");
    path->setPlaceholderText("config.ocio");
    auto* browse = new QPushButton("Browse…", &dlg);
    connect(browse, &QPushButton::clicked, &dlg, [&] {
        const QString f = QFileDialog::getOpenFileName(&dlg, "OpenColorIO config", path->text(), "OpenColorIO config (*.ocio *.ocioz)");
        if (!f.isEmpty()) {
            path->setText(f);
            file->setChecked(true);
        }
    });
    (kind == "env" && envr->isEnabled() ? envr : kind == "file" ? file : builtin)->setChecked(true);
    v->addWidget(builtin);
    v->addWidget(envr);
    auto* fr = new QHBoxLayout;
    fr->addWidget(file);
    fr->addWidget(path, 1);
    fr->addWidget(browse);
    v->addLayout(fr);
    auto* now = new QLabel(&dlg);
    now->setObjectName("ocioNow");
    now->setWordWrap(true);
    now->setText(ocio_cfg_ ? QStringLiteral("In use: %1").arg(QString::fromStdString(ocio_cfg_->name())) +
                                 (ocio_cfg_error_.isEmpty() ? QString() : QStringLiteral("\n") + ocio_cfg_error_)
                           : ocio_cfg_error_);
    v->addWidget(now);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    connect(buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    v->addWidget(buttons);
    if (dlg.exec() != QDialog::Accepted) return;
    st.setValue("viewer/ocioSource", envr->isChecked() ? "env" : file->isChecked() ? "file" : "builtin");
    st.setValue("viewer/ocioFile", path->text());
    open_colour_config();
    make_colour_transforms();
    refresh_colour();
    if (!ocio_cfg_error_.isEmpty()) statusBar()->showMessage(ocio_cfg_error_, 10000);
}

}  // namespace rudra::app
