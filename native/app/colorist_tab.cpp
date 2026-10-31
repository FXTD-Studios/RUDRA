// The colorist's side of the learned ACES 2.0 route (28 Oct 2026): the trims
// on the master (core/aces_v2_trims.hpp) and the timecode and reel a render
// carries. The scopes and readouts of the route are in run_stats().

#include <QCryptographicHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>

#include <algorithm>
#include <cmath>

#include "main_window.hpp"
#include "paths.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"

#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QPushButton>
#include <QStatusBar>

namespace rudra::app {

void MainWindow::set_aces_trims(const aces_v2::AcesTrims& t) {
    aces_v2::AcesTrims c = t;
    c.strength = std::clamp(c.strength, 0.0, 1.5);
    c.saturation = std::clamp(c.saturation, 0.0, 2.0);
    if (c.peak_nits != 0) c.peak_nits = std::clamp(c.peak_nits, 100.0, 10000.0);
    c.exposure_ev = std::clamp(c.exposure_ev, -3.0, 3.0);
    c.contrast = std::clamp(c.contrast, 0.5, 1.5);
    const bool changed = !(c == aces_trims_);
    aces_trims_ = c;
    QSettings st;
    st.setValue("aces/trimStrength", c.strength);
    st.setValue("aces/trimSaturation", c.saturation);
    st.setValue("aces/trimPeakNits", c.peak_nits);
    st.setValue("aces/trimExposure", c.exposure_ev);
    st.setValue("aces/trimContrast", c.contrast);
    // This shot's own, for when it is opened again.
    if (const QString key = shot_trim_key(); !key.isEmpty()) {
        trims_shot_ = key;
        const QByteArray h = QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Md5).toHex();
        st.setValue("aces/shotTrims/" + QString::fromLatin1(h),
                    QJsonDocument(QJsonObject{{"shot", key}, {"strength", c.strength}, {"saturation", c.saturation},
                                              {"peak_nits", c.peak_nits}, {"exposure_ev", c.exposure_ev},
                                              {"contrast", c.contrast}})
                        .toJson(QJsonDocument::Compact));
    }
    sync_aces_trims();
    if (!changed) return;
    log(QString::fromStdString("learned ACES 2.0 trims: " + aces_v2::trims_text(c)));
    // The frame on screen again with the new trims, once a drag settles.
    if (!aces_trim_timer_) {
        aces_trim_timer_ = new QTimer(this);
        aces_trim_timer_->setSingleShot(true);
        aces_trim_timer_->setInterval(150);
        connect(aces_trim_timer_, &QTimer::timeout, this, [this] {
            if (aces_routed() && !frames_.empty()) aces_request(current_);
        });
    }
    aces_trim_timer_->start();
}

void MainWindow::sync_aces_trims() {
    auto* card = findChild<QWidget*>("acesTrimsCard");
    if (!card) return;
    card->setVisible(aces_routed());
    if (auto* mon = dynamic_cast<Seg*>(findChild<QWidget*>("acesMonitor"))) mon->set_on(aces_monitor_);
    if (auto* sd = dynamic_cast<Seg*>(findChild<QWidget*>("acesSourceDisplay"))) sd->set_on(aces_source_display_);
    // On the learned route only the trims above reach a render: the analytic controls
    // are locked, and say so (28 Oct 2026 review). The view peak follows an SDR 100 /
    // HDR 1000 monitor, so the screen is what the deliverable is.
    const bool routed = aces_routed();
    for (QWidget* w : std::initializer_list<QWidget*>{mode_seg_, exposure_, strength_, preserve_, regions_, mask_panel_})
        if (w) w->setEnabled(!routed);
    if (peak_) {
        peak_->setEnabled(!routed || aces_monitor_ == "scene");
        if (!peak_->property("baseTip").isValid()) peak_->setProperty("baseTip", peak_->toolTip());
        const QString base = peak_->property("baseTip").toString();
        peak_->setToolTip(routed && aces_monitor_ != "scene"
                              ? QStringLiteral("Locked to the monitor: %1 nits, as the deliverable is").arg(aces_monitor_ == "sdr100" ? 100 : 1000)
                              : base);
    }
    if (parade_ && !routed) parade_->clear();   // the parade is the learned frame's
    for (const char* id : {"routeNoteRec", "routeNoteGrade"})
        if (auto* n = findChild<QLabel*>(id)) n->setVisible(routed);
    const auto& t = aces_trims_;
    auto set = [this](const char* id, int v, const QString& text) {
        if (auto* s = findChild<QSlider*>(id)) {
            const QSignalBlocker b(s);
            s->setValue(v);
        }
        if (auto* l = findChild<QLabel*>(QString(id) + "Val")) l->setText(text);
    };
    set("acesStrength", int(std::lround(t.strength * 100)), QString::number(std::lround(t.strength * 100)));
    set("acesSaturation", int(std::lround(t.saturation * 100)), QString::number(std::lround(t.saturation * 100)));
    set("acesPeak", t.peak_nits > 0 ? int(std::lround(t.peak_nits / 250.0)) : 41,
        t.peak_nits > 0 ? QString::number(std::lround(t.peak_nits)) : QStringLiteral("off"));
    set("acesExposure", int(std::lround(t.exposure_ev * 10)), QString::number(t.exposure_ev, 'f', 1));
    set("acesContrast", int(std::lround(t.contrast * 100)), QString::number(std::lround(t.contrast * 100)));
    if (auto* p = findChild<QPushButton*>("acesTrimsPaste")) p->setEnabled(trims_clipboard_.has_value());
}

QString MainWindow::shot_trim_key() const {
    if (frames_.empty()) return {};
    if (shot_video_ && !shot_sources_.empty()) return from_path(shot_sources_.front());
    return from_path(frames_.size() > 1 ? frames_.front().parent_path() : frames_.front());
}

std::optional<aces_v2::AcesTrims> MainWindow::remembered_trims(const QString& key) const {
    if (key.isEmpty()) return std::nullopt;
    const QByteArray h = QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Md5).toHex();
    const QJsonObject o = QJsonDocument::fromJson(QSettings().value("aces/shotTrims/" + QString::fromLatin1(h)).toByteArray()).object();
    if (o.value("shot").toString() != key) return std::nullopt;
    aces_v2::AcesTrims t;
    t.strength = o.value("strength").toDouble(1.0);
    t.saturation = o.value("saturation").toDouble(1.0);
    t.peak_nits = o.value("peak_nits").toDouble(0.0);
    t.exposure_ev = o.value("exposure_ev").toDouble(0.0);
    t.contrast = o.value("contrast").toDouble(1.0);
    return t;
}

void MainWindow::copy_aces_trims() {
    trims_clipboard_ = aces_trims_;
    log(QString::fromStdString("trims copied: " + aces_v2::trims_text(aces_trims_)));
    sync_aces_trims();
}

void MainWindow::paste_aces_trims() {
    if (!trims_clipboard_) return;
    set_aces_trims(*trims_clipboard_);
    log(QStringLiteral("trims pasted onto ") + shot_title() + ": " + QString::fromStdString(aces_v2::trims_text(*trims_clipboard_)));
}

void MainWindow::set_aces_source_display(const QString& s) {
    if (s != "bt1886" && s != "srgb" && s != "gamma22") return;
    const bool changed = s != aces_source_display_;
    aces_source_display_ = s;
    QSettings().setValue("aces/sourceDisplay", s);
    sync_aces_trims();
    update_pipe();
    if (!changed) return;
    log(QStringLiteral("source display: ") + (s == "srgb" ? QStringLiteral("sRGB (re-encoded to BT.1886 for the model)")
                                              : s == "gamma22" ? QStringLiteral("gamma 2.2 (re-encoded to BT.1886 for the model)")
                                                               : QStringLiteral("Rec.709 on BT.1886 2.4 (the model's own)")));
    if (aces_routed() && !frames_.empty()) aces_request(current_);
}

void MainWindow::set_aces_monitor(const QString& m) {
    if (m != "scene" && m != "sdr100" && m != "hdr1000") return;
    const bool changed = m != aces_monitor_;
    aces_monitor_ = m;
    QSettings().setValue("aces/monitor", m);
    // The view peak follows the display the output is made for: 100 or 1000 nits.
    if (changed && m != "scene") session_.peak_input(std::log2((m == "sdr100" ? 100.0 : 1000.0) / 203.0));
    sync_aces_trims();
    sync_checks();
    if (!changed) return;
    log(m == "scene" ? QStringLiteral("monitor: scene light (the RUDRA view)")
        : m == "sdr100" ? QStringLiteral("monitor: ACES 2.0 SDR 100 nits (Rec.709)")
                        : QStringLiteral("monitor: ACES 2.0 HDR 1000 nits (P3 D65, Rec.2100 PQ)"));
    if (aces_routed() && !frames_.empty()) aces_request(current_);
}

int MainWindow::grab_reference_still() {
    if (!aces_routed() || !aces_model_img_ || aces_shown_ < 0) return -1;
    RefStill s;
    s.label = QStringLiteral("%1 \u00b7 %2").arg(ref_stills_.size() + 1).arg(QString::fromStdString(frame_name(std::size_t(aces_shown_))));
    s.image = aces_model_img_;
    ref_stills_.push_back(std::move(s));
    ref_index_ = int(ref_stills_.size()) - 1;
    log("Reference still " + ref_stills_.back().label);
    if (statusBar()) statusBar()->showMessage("Grabbed reference still " + ref_stills_.back().label, 4000);
    sync_reference_ui();
    return ref_index_;
}

const NetworkLinearImage* MainWindow::reference_still() const {
    if (ref_index_ < 0 || ref_index_ >= int(ref_stills_.size())) return nullptr;
    return ref_stills_[std::size_t(ref_index_)].image.get();
}

void MainWindow::set_reference_compare(bool on, int index) {
    if (index >= 0 && index < int(ref_stills_.size())) ref_index_ = index;
    ref_compare_ = on && reference_still() != nullptr;
    sync_reference_ui();
    sync_checks();
    if (aces_routed() && !frames_.empty()) aces_request(current_);
}

void MainWindow::sync_reference_ui() {
    if (auto* box = findChild<QComboBox*>("refStills")) {
        const QSignalBlocker b(box);
        box->clear();
        for (const auto& s : ref_stills_) box->addItem(s.label);
        box->setCurrentIndex(ref_index_);
        box->setEnabled(!ref_stills_.empty());
    }
    if (auto* c = findChild<QPushButton*>("refCompare")) {
        const QSignalBlocker b(c);
        c->setChecked(ref_compare_);
        c->setEnabled(!ref_stills_.empty());
    }
    if (auto* n = findChild<QLabel*>("refNote")) {
        const auto* ref = reference_still();
        n->setText(!ref_compare_ ? QString()
                   : aces_ref_shown_ || !aces_model_img_ ? QStringLiteral("Baseline, Wipe and B show the still")
                                                         : QStringLiteral("Not the frame's size: the plain inverse is shown"));
        n->setVisible(ref_compare_ && ref);
    }
}

QString MainWindow::render_timecode() const {
    auto* e = findChild<QLineEdit*>("renderTimecode");
    return e ? e->text().trimmed() : QString();
}

QString MainWindow::render_reel() const {
    auto* e = findChild<QLineEdit*>("renderReel");
    return e ? e->text().trimmed() : QString();
}

}  // namespace rudra::app

// ---- the Render card's File path, as Nuke's Write takes it (29 Oct 2026) -------

namespace rudra::app {

void MainWindow::sync_render_file() {
    if (syncing_render_file_) return;
    auto* file = findChild<QLineEdit*>("renderFile");
    auto* dir = findChild<QLineEdit*>("renderDir");
    auto* name = findChild<QLineEdit*>("renderName");
    if (!file || !dir || !name) return;
    const DeliverChoice c = deliver_effective();
    RenderPath p;
    p.dir = QDir::fromNativeSeparators(dir->text().trimmed());
    p.name = name->text().trimmed();
    p.ext = render_ext(c);
    p.padding = c.output == "movie" ? 0 : render_padding_;
    p.separator = render_separator_;
    const QString text = p.dir.isEmpty() && p.name.isEmpty() ? QString() : compose_render_path(p);
    if (file->text() != text) {
        const QSignalBlocker b(file);
        file->setText(text);
    }
}

void MainWindow::render_file_edited() {
    auto* file = findChild<QLineEdit*>("renderFile");
    if (!file) return;
    const RenderPath p = parse_render_path(file->text());
    syncing_render_file_ = true;
    if (!p.dir.isEmpty()) findChild<QLineEdit*>("renderDir")->setText(QDir::toNativeSeparators(p.dir));
    if (!p.name.isEmpty()) findChild<QLineEdit*>("renderName")->setText(p.name);
    syncing_render_file_ = false;
    if (p.padding > 0) {
        render_padding_ = p.padding;
        render_separator_ = p.separator;
        QSettings st;
        st.setValue("deliver/renderPadding", render_padding_);
        st.setValue("deliver/renderSeparator", render_separator_);
    }
    // The extension picks the file type, as the Write's file type follows its name.
    const DeliverChoice c = deliver_effective();
    if (p.ext == "dpx" && c.image != "dpx10" && deliver_aces()) {
        if (c.output == "movie") set_deliver("output", "sequence");
        set_deliver("image", "dpx10");
    } else if (p.ext == "exr" && (c.image == "dpx10" || c.output == "movie")) {
        if (c.output == "movie") set_deliver("output", "sequence");
        if (c.image == "dpx10") set_deliver("image", "exr32");
    } else if (p.ext == "mov" && c.output != "movie") {
        set_deliver("output", "movie");
    }
    sync_render_file();   // the path as it will be written (folder, name, padding, the type's extension)
    sync_deliver();
}

void MainWindow::browse_render_file() {
    auto* file = findChild<QLineEdit*>("renderFile");
    if (!file) return;
    const DeliverChoice c = deliver_effective();
    const QString ext = render_ext(c);
    const RenderPath now = parse_render_path(file->text());
    QString start = now.dir.isEmpty() ? QString() : QDir(now.dir).filePath(now.name + "." + ext);
    const QString filter = ext == "mov"   ? QStringLiteral("QuickTime (*.mov)")
                           : ext == "dpx" ? QStringLiteral("DPX (*.dpx)")
                                          : QStringLiteral("OpenEXR (*.exr)");
    const QString picked = QFileDialog::getSaveFileName(this, QStringLiteral("Render to"), start, filter, nullptr,
                                                        QFileDialog::DontConfirmOverwrite);
    if (picked.isEmpty()) return;
    RenderPath p = parse_render_path(picked);
    if (p.ext.isEmpty()) p.ext = ext;
    if (p.padding == 0 && c.output != "movie") {   // a picked "shot.exr" becomes "shot.####.exr"
        p.padding = render_padding_;
        p.separator = render_separator_;
    }
    file->setText(compose_render_path(p));
    render_file_edited();
}

}  // namespace rudra::app
