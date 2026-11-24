// The Deliver tab (27 Oct 2026): the source, still / sequence / MOV, the file
// type, its colour and the movie's bit depth (deliver_choice.hpp), and Render:
// the learned ACES 2.0 masters through the engine kept loaded (aces_engine.hpp,
// cli/aces_v2_deliver.hpp), or the analytic master and movie exports as before.

#include <QCheckBox>
#include <QCoreApplication>
#include <QCursor>
#include <QFile>
#include <QMenu>
#include <QProcess>
#include <QStandardItemModel>
#include <QProcessEnvironment>
#include <QStatusBar>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStorageInfo>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "aces_engine.hpp"
#include "paths.hpp"
#include "aces_master.hpp"
#include "genfill_job.hpp"
#include "genfill_preview.hpp"
#include "genfill_swap.hpp"
#include "genfill_review.hpp"
#include "main_window.hpp"
#include "system_check.hpp"   // genfill_missing_text (1.2.5)
#include <rudra/core/resample.hpp>
#include "rudra/core/timecode.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/platform/png8.hpp"
#include "widgets.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QString qp(const fs::path& p) { return from_path(p); }

// The characters a render name may use (cli/aces_v2_deliver.cpp); others become '_'.
QString clean_name(QString n) {
    n = n.trimmed();
    for (QChar& c : n)
        if (!((c.isLetterOrNumber() && c.unicode() < 128) || c == '_' || c == '-' || c == '.' || c == ' ')) c = '_';
    while (n.endsWith('.') || n.endsWith(' ')) n.chop(1);
    return n.left(120);
}

}  // namespace

std::pair<int, int> MainWindow::shot_source_size() const {
    if (!current_frame_) return {0, 0};
    const QString r = QString::fromStdString(current_frame_->header.source_resolution.value_or(std::string()));
    const QStringList p = r.split(QLatin1Char('x'));
    if (p.size() != 2) return {0, 0};
    bool okw = false, okh = false;
    const int w = p[0].toInt(&okw), h = p[1].toInt(&okh);
    return okw && okh && w > 0 && h > 0 ? std::pair{w, h} : std::pair{0, 0};
}

bool MainWindow::deliver_aces() const {
    return deliver_.source == "aces" && aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready;
}

bool MainWindow::deliver_learned_wanted() const {
    return deliver_.source == "aces" && aces_engine_ && aces_engine_->state() != AcesEngine::State::Stopped;
}

DeliverChoice MainWindow::deliver_effective() const {
    const bool aces = deliver_aces() || deliver_learned_wanted();
    DeliverChoice c = deliver_;
    // RUDRA 1.1: the AI fill acts only inside the grade's reconstruction region, when one is on.
    c.ai.region_json = session_.grade.region.enabled
                           ? QString::fromStdString(reconstruction_region_json(session_.grade.region))
                           : QString();
    if (!aces) c.gamut = session_.container == "aces" ? QStringLiteral("ap0") : QString::fromStdString(session_.container);
    // 21 Nov 2026: Deliver's AI fill checkbox counts only while Reconstruct > AI generation is on.
    if (!ai_generation_) c.ai_fill = false;
    const bool movie = shot_is_video();
    DeliverChoice fit = deliver_fit(c, aces, movie, !movie && frames_.size() > 1);
    // 1.3: Output size. The AI fill works at the source's size (the model's, and a 2x fill would take
    // four times the disk and the time); the analytic movie of a movie is the export queue's, which
    // writes the source's size.
    if (fit.ai_fill || (!aces && fit.output == "movie" && movie)) fit.size = QStringLiteral("source");
    // 1.3 (24 Nov 2026, audit): a size past 8192 on a side for this shot is written at the source's.
    if (const auto [sw, sh] = shot_source_size(); !output_size_fits(fit.size.toStdString(), sw, sh)) fit.size = QStringLiteral("source");
    return fit;
}

DeliverChoice MainWindow::deliver_for_batch() const {
    DeliverChoice c = deliver_;
    if (!ai_generation_) c.ai_fill = false;
    if (c.ai_fill) c.size = QStringLiteral("source");   // 1.3: the fill works at the source's size
    return c;
}

void MainWindow::load_deliver_settings() {
    QSettings st;
    DeliverChoice c;
    c.source = st.value("deliver/source", c.source).toString();
    c.output = st.value("deliver/output", c.output).toString();
    c.image = st.value("deliver/image", c.image).toString();
    c.gamut = st.value("deliver/gamut", c.gamut).toString();
    c.encoding = st.value("deliver/encoding", c.encoding).toString();
    c.pq_peak = st.value("deliver/pqPeak", c.pq_peak).toInt();
    c.bits = st.value("deliver/bits", c.bits).toInt();
    c.codec = st.value("deliver/codec", c.codec).toString();
    c.keep_exr = st.value("deliver/keepExr", c.keep_exr).toBool();
    c.mattes = st.value("deliver/mattes", c.mattes).toBool();
    c.brackets = st.value("deliver/brackets", c.brackets).toBool();
    c.fps = st.value("deliver/fps", c.fps).toDouble();
    c.size = st.value("deliver/size", c.size).toString();   // 1.3: Output size
    // 1.3: EXR compression; before it, the hidden deliver/exrZip (on unless turned off) decided.
    c.exr_compression = st.value("deliver/exrCompression", st.value("deliver/exrZip", true).toBool() ? "zip" : "none").toString();
    // 24 Nov 2026 (Ahmed): Deliver's checkbox (the export includes the fill) is off at every launch; ticked, it stays
    // on for this session only. Reconstruct > AI generation starts on where this computer can run the fill (below).
    // Its settings (prompts, strengths, seed) are kept.
    c.ai_fill = false;
    c.ai.tau_high = st.value("deliver/aiTauHigh", c.ai.tau_high).toInt();
    c.ai.tau_low = st.value("deliver/aiTauLow", c.ai.tau_low).toInt();
    c.ai.seed = st.value("deliver/aiSeed", c.ai.seed).toInt();
    c.ai.steps = st.value("deliver/aiSteps", c.ai.steps).toInt();
    c.ai.brightness = st.value("deliver/aiBrightness", c.ai.brightness).toDouble();
    c.ai.prompt_strength = std::clamp(st.value("deliver/aiPromptStrength", c.ai.prompt_strength).toDouble(), 0.0, 2.0);
    c.ai.shape_gain = std::clamp(st.value("deliver/aiShapeGain", c.ai.shape_gain).toDouble(), 0.0, 2.0);
    c.ai.fallback = st.value("deliver/aiFallback", true).toBool();   // RUDRA 1.1: on unless turned off
    c.ai.focus_region = st.value("deliver/aiFocus", false).toBool();  // 7 Nov 2026: off (v3) unless turned on
    // 1.2.6 (18 Nov 2026, user request): v3 or v4.3 region is the user's choice, a switch in the AI
    // fill card; v4 runs where its LoRA is found (the setup line says when it is not).
    c.ai.rerender = st.value("deliver/aiRerender", false).toBool();
    c.ai.rerender_keep_detail = st.value("deliver/aiKeepDetail", false).toBool();   // 1.2.7
    c.ai.prompt_highlight = st.value("deliver/aiPromptHighlight").toString();
    c.ai.prompt_shadow = st.value("deliver/aiPromptShadow").toString();
    // 21 Nov 2026: Reconstruct > AI generation, which the viewer's AI fill button is. 24 Nov 2026 (Ahmed): at launch
    // it is on where this computer can run the fill (a 16 GB NVIDIA GPU, or Apple silicon with 16 GB, and the AI
    // fill installed) and off anywhere else, whatever an older RUDRA saved; a click changes it for this session.
    const AiFillHere here = aifill_here();
    aifill_runs_here_ = here.runs;
    aifill_why_ = here.why;
    ai_generation_ = here.runs;
    if (c.source != "aces" && c.source != "analytic") c.source = "aces";
    deliver_ = deliver_fit(c, true, true, true);   // only the values; the shot decides the rest
}

void MainWindow::set_deliver(const QString& key, const QString& value) {
    QSettings st;
    if (key == "source" && (value == "aces" || value == "analytic")) {
        deliver_.source = value;
        st.setValue("deliver/source", value);
    } else if (key == "output" && (value == "still" || value == "sequence" || value == "movie")) {
        deliver_.output = value;
        st.setValue("deliver/output", value);
        // Master EXR, the settings and projects read the hidden Render > Output.
        if (auto* mode = findChild<QComboBox*>("renderMode")) {
            const QSignalBlocker b(mode);
            mode->setCurrentIndex(value == "still" ? 0 : 1);
        }
    } else if (key == "image" && (value == "exr32" || value == "exr16" || value == "dpx10")) {
        deliver_.image = value;
        st.setValue("deliver/image", value);
    } else if (key == "gamut" && (value == "ap0" || value == "acescg" || value == "linear" || value == "rec709")) {
        // The analytic master's gamut is the session's container (Deliver menu); RUDRA 1.1: all four.
        if (!deliver_aces()) session_.set_container(value == "ap0" ? "aces" : value.toStdString());
        if (deliver_aces() && value == "rec709") return;
        if (deliver_aces() || value == "ap0") {
            deliver_.gamut = value == "linear" ? QStringLiteral("acescg") : value;
            st.setValue("deliver/gamut", deliver_.gamut);
        }
    } else if (key == "encoding" && (value == "logc4" || value == "acescct" || value == "pq" || value == "hlg" || value == "sdr")) {
        deliver_.encoding = value;
        st.setValue("deliver/encoding", value);
    } else if (key == "pq_peak" && (value == "1000" || value == "2000" || value == "4000")) {
        deliver_.pq_peak = value.toInt();
        st.setValue("deliver/pqPeak", deliver_.pq_peak);
    } else if (key == "bits" && (value == "10" || value == "16")) {
        deliver_.bits = value.toInt();
        st.setValue("deliver/bits", deliver_.bits);
    } else if (key == "codec" && (value == "prores422hq" || value == "prores4444" || value == "prores4444xq" || value == "hevc10")) {
        deliver_.codec = value;
        st.setValue("deliver/codec", value);
    } else if (key == "brackets") {   // RUDRA 1.1 priority 4
        deliver_.brackets = value == "1" || value == "true";
        st.setValue("deliver/brackets", deliver_.brackets);
    } else if (key == "mattes") {   // RUDRA 1.1 priority 3
        deliver_.mattes = value == "1" || value == "true";
        st.setValue("deliver/mattes", deliver_.mattes);
    } else if (key == "keep_exr") {
        deliver_.keep_exr = value == "1" || value == "true";
        st.setValue("deliver/keepExr", deliver_.keep_exr);
    } else if (key == "size" && (value == "source" || value == "2x" || value == "uhd" || value == "hd")) {   // 1.3
        deliver_.size = value;
        st.setValue("deliver/size", value);
    } else if (key == "exr_compression" && (value == "zip" || value == "zips" || value == "none")) {   // 1.3
        deliver_.exr_compression = value;
        st.setValue("deliver/exrCompression", value);
    } else if (key == "fps") {
        bool ok = false;
        const double f = value.toDouble(&ok);
        if (ok && f > 0 && f <= 240) {
            deliver_.fps = f;
            st.setValue("deliver/fps", f);
        }
    } else if (key == "ai_generation") {   // 21 Nov 2026: Reconstruct > AI generation; 24 Nov: the viewer's AI fill button
        const bool was = ai_generation_;
        ai_generation_ = value == "1" || value == "true";   // for this session (the next launch decides again)
        if (!ai_generation_ && ai_region_view_) set_ai_region_view(false);
        log(ai_generation_ ? QStringLiteral("AI generation on (Reconstruct)") : QStringLiteral("AI generation off (Reconstruct)"));
        // 24 Nov 2026 (Ahmed): one AI fill, on or off. On, the viewer shows the fill wherever there is one (the
        // preview's frames or the last finished fill); off, the master.
        if (ai_generation_ != was) {
            if (ai_generation_ && !frames_.empty() && (genfill_preview_.has_value() || !genfill_last_out_.isEmpty()))
                set_fill_view(true);
            else if (!ai_generation_ && fill_view_on_)
                set_fill_view(false);
        }
        sync_ai_generation();
    } else if (key == "ai_fill") {   // Deliver's one checkbox: the export includes the fill
        deliver_.ai_fill = value == "1" || value == "true";   // for this session (off at the next launch)
    } else if (key == "ai_tau_high" || key == "ai_tau_low" || key == "ai_seed" || key == "ai_steps") {
        bool ok = false;
        const int v = value.toInt(&ok);
        if (!ok) return;
        if (key == "ai_tau_high" && v >= 128 && v <= 256 && v > deliver_.ai.tau_low) deliver_.ai.tau_high = v;
        else if (key == "ai_tau_low" && v >= -1 && v <= 64 && v < deliver_.ai.tau_high) deliver_.ai.tau_low = v;
        else if (key == "ai_seed" && v >= 0) deliver_.ai.seed = v;
        else if (key == "ai_steps" && v >= 1 && v <= 100) deliver_.ai.steps = v;
        else return;
        st.setValue("deliver/aiTauHigh", deliver_.ai.tau_high);
        st.setValue("deliver/aiTauLow", deliver_.ai.tau_low);
        st.setValue("deliver/aiSeed", deliver_.ai.seed);
        st.setValue("deliver/aiSteps", deliver_.ai.steps);
    } else if (key == "ai_brightness") {
        bool ok = false;
        const double v = value.toDouble(&ok);
        if (!ok || v < -3.0 || v > 3.0) return;
        deliver_.ai.brightness = v;
        st.setValue("deliver/aiBrightness", v);
    } else if (key == "ai_focus") {   // 7 Nov 2026: draw the clipped area at more detail
        deliver_.ai.focus_region = value == "1" || value == "true";
        st.setValue("deliver/aiFocus", deliver_.ai.focus_region);
    } else if (key == "ai_rerender") {   // 12 Nov 2026: AI fill v4, region re-render (experimental)
        deliver_.ai.rerender = value == "1" || value == "true" || value == "v4";   // 1.2.6: the switch
        st.setValue("deliver/aiRerender", deliver_.ai.rerender);
        refresh_prompt_chips();   // v4 shows the presets only
    } else if (key == "ai_keep_detail") {   // 1.2.7: v4.3 leaves thin and small bright parts to the master
        deliver_.ai.rerender_keep_detail = value == "1" || value == "true";
        st.setValue("deliver/aiKeepDetail", deliver_.ai.rerender_keep_detail);
    } else if (key == "ai_fallback") {   // RUDRA 1.1 priority 5
        deliver_.ai.fallback = value == "1" || value == "true";
        st.setValue("deliver/aiFallback", deliver_.ai.fallback);
    } else if (key == "ai_prompt_strength" || key == "ai_shape_gain") {   // RUDRA 1.1
        bool ok = false;
        const double v = value.toDouble(&ok);
        if (!ok || !std::isfinite(v) || v < 0.0 || v > 2.0) return;
        if (key == "ai_prompt_strength") {
            deliver_.ai.prompt_strength = v;
            st.setValue("deliver/aiPromptStrength", v);
        } else {
            deliver_.ai.shape_gain = v;
            st.setValue("deliver/aiShapeGain", v);
        }
    } else if (key == "ai_prompt_highlight") {
        deliver_.ai.prompt_highlight = value.left(400);
        st.setValue("deliver/aiPromptHighlight", deliver_.ai.prompt_highlight);
    } else if (key == "ai_prompt_shadow") {
        deliver_.ai.prompt_shadow = value.left(400);
        st.setValue("deliver/aiPromptShadow", deliver_.ai.prompt_shadow);
    } else {
        return;
    }
    sync_deliver();
}

void MainWindow::sync_deliver() {
    auto* card = findChild<QWidget*>("deliverCard");
    if (!card || !btn_master_) return;
    const bool aces = deliver_aces();
    const DeliverChoice c = deliver_effective();
    const bool movie_shot = shot_is_video(), folder = !movie_shot && frames_.size() > 1;
    auto show = [this](const char* id, bool on) {
        if (auto* w = findChild<QWidget*>(id)) w->setVisible(on);
    };
    auto* src = dynamic_cast<Seg*>(findChild<QWidget*>("deliverSource"));
    const bool wanted = deliver_learned_wanted();
    src->set_on(aces || wanted ? "aces" : "analytic");
    const bool engine_ready = aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready;
    src->button("aces")->setEnabled(engine_ready || wanted);
    src->button("aces")->setToolTip(engine_ready ? QStringLiteral("The learned ACES 2.0 model renders the files, as the viewer shows them")
                                    : aces_engine_ ? aces_engine_->status_text()
                                                   : QStringLiteral("The learned ACES 2.0 model is not loaded (Deliver > Learned ACES 2.0 master…)"));
    if (auto* n = dynamic_cast<NoteLabel*>(findChild<QLabel*>("deliverCurveNote"))) {
        const bool steer = (aces || wanted) && aces_curve_mismatch();
        // 12 Nov 2026: one line here, the whole note in the tooltip (it was eight lines in a narrow column)
        const QString brief = QStringLiteral("This source is not an ACES curve, which the learned model expects. "
                                             "Analytic is the route for it (hover for more).");
        if (steer && n->text() != brief) n->setText(brief);
        if (steer) n->setToolTip(aces_curve_note());
        n->setVisible(steer);
        if (auto* use = findChild<QPushButton*>("deliverUseAnalytic")) use->setVisible(steer && aces);   // 1.2.6: item 4
    }
    auto* out = dynamic_cast<Seg*>(findChild<QWidget*>("deliverOutput"));
    out->set_on(c.output);
    out->button("movie")->setEnabled(movie_shot || folder);
    out->button("movie")->setVisible(movie_shot || folder);   // 1.2.6: a choice this shot cannot take is not offered
    out->button("movie")->setToolTip(movie_shot || folder ? QStringLiteral("One QuickTime movie of every frame")
                                                          : QStringLiteral("For a movie or a folder of frames; a single still goes out as a still"));
    out->button("sequence")->setEnabled(!frames_.empty());
    out->button("sequence")->setVisible(!frames_.empty());
    const bool movie = c.output == "movie";
    // Still and sequence: the file type (the analytic master is EXR 32-bit only).
    show("deliverImageRow", !movie && aces);
    if (auto* image = findChild<QComboBox*>("deliverImage")) {
        const QSignalBlocker b(image);
        image->setCurrentIndex(image->findData(c.image));
    }
    const bool exr = !movie && c.image != "dpx10";
    show("deliverGamutRow", exr);
    if (auto* gamut = dynamic_cast<Seg*>(findChild<QWidget*>("deliverGamut"))) {
        gamut->button("linear")->setVisible(!aces);
        gamut->button("rec709")->setVisible(!aces);   // RUDRA 1.1: the analytic master writes it too
        gamut->set_on(c.gamut);
    }
    // A movie: 10-bit ProRes or 16-bit uncompressed RGB.
    show("deliverBitsRow", movie);
    if (auto* bits = dynamic_cast<Seg*>(findChild<QWidget*>("deliverBits"))) bits->set_on(QString::number(c.bits));
    show("deliverCodecRow", movie && c.bits == 10);
    if (auto* codec = findChild<QComboBox*>("deliverCodec")) {
        const QSignalBlocker b(codec);
        // ProRes 4444 XQ and HDR10 HEVC: the learned engine's only.
        if (auto* v = qobject_cast<QListView*>(codec->view()))
            for (const char* k : {"prores4444xq", "hevc10"})
                if (const int i = codec->findData(QString::fromLatin1(k)); i >= 0) v->setRowHidden(i, !aces);
        codec->setCurrentIndex(codec->findData(c.codec));
    }
    const bool encoded = movie || (aces && c.image == "dpx10");
    show("deliverEncodingRow", encoded);
    if (auto* enc = dynamic_cast<Seg*>(findChild<QWidget*>("deliverEncoding"))) {
        enc->set_on(c.encoding);
        const bool logs = aces || (movie_shot && c.bits == 10);   // the analytic 16-bit and folder movies are PQ
        enc->button("logc4")->setEnabled(logs);
        enc->button("acescct")->setEnabled(logs);
        enc->button("logc4")->setVisible(logs);   // 1.2.6: hidden, not greyed
        enc->button("acescct")->setVisible(logs);
        for (const char* k : {"hlg", "sdr"})   // the learned engine's ACES 2.0 outputs
            if (auto* b = enc->button(k)) b->setVisible(aces);
        enc->button("pq")->setToolTip(aces ? QStringLiteral("The ACES 2.0 output transform, HDR %1 nits (P3 D65) on Rec.2100 PQ: "
                                                            "display-referred, ready to view").arg(c.pq_peak)
                                           : QStringLiteral("Rec.2020 PQ, graded to the peak and knee on this tab"));
    }
    // The PQ target's peak: 1000 (HDR10, most streaming), 2000 or 4000 nits (Dolby Vision mastering, theatrical HDR).
    show("deliverPqPeakRow", encoded && aces && c.encoding == "pq");
    if (auto* pk = dynamic_cast<Seg*>(findChild<QWidget*>("deliverPqPeak"))) pk->set_on(QString::number(c.pq_peak));
    show("deliverFpsRow", movie && !movie_shot);
    if (auto* fps = findChild<QComboBox*>("deliverFps")) {
        const QSignalBlocker b(fps);
        for (int i = 0; i < fps->count(); ++i)
            if (std::abs(fps->itemData(i).toDouble() - c.fps) < 1e-6) fps->setCurrentIndex(i);
    }
    // 1.3: Output size, each choice with the size it writes for this shot.
    const bool queue_movie = !aces && movie && movie_shot;
    show("deliverSizeRow", !queue_movie);
    if (auto* sz = findChild<QComboBox*>("deliverSize")) {
        const QSignalBlocker b(sz);
        const auto [sw, sh] = shot_source_size();
        auto* model = qobject_cast<QStandardItemModel*>(sz->model());
        for (int i = 0; i < sz->count(); ++i) {
            const QString k = sz->itemData(i).toString();
            const OutputSize o = output_size(k.toStdString(), sw, sh);
            // 1.3 (24 Nov 2026, audit): a size with a side past 8192 is greyed out (2x of a 6K or 8K source).
            const bool fits = output_size_fits(k.toStdString(), sw, sh);
            QString text = k == "source" ? QStringLiteral("Source") : k == "2x" ? QStringLiteral("2\u00d7")
                           : k == "uhd" ? QStringLiteral("UHD") : QStringLiteral("HD");
            if (sw > 0) text += fits ? QStringLiteral(" \u00b7 %1 \u00d7 %2").arg(o.width).arg(o.height)
                                     : QStringLiteral(" \u00b7 over %1 pixels").arg(kMaxOutputSide);
            sz->setItemText(i, text);
            if (model)
                if (QStandardItem* item = model->item(i)) item->setEnabled(fits);
            sz->setItemData(i, fits ? QVariant() : QVariant(QStringLiteral("More than %1 pixels on a side for this shot: too large to "
                                                                           "write (memory, and HEVC stops at 8192 x 4320)")
                                                                .arg(kMaxOutputSide)),
                            Qt::ToolTipRole);
        }
        sz->setCurrentIndex(std::max(0, sz->findData(output_size_fits(c.size.toStdString(), sw, sh) ? c.size : QStringLiteral("source"))));
        sz->setEnabled(!c.ai_fill);
        sz->setToolTip(c.ai_fill ? QStringLiteral("The AI fill works at the source size; turn it off to write another size.")
                                 : QStringLiteral("A clean resize of the master in linear light: Lanczos 3 held between "
                                                  "neighbouring pixels, so no halo or dark ring round a highlight, and "
                                                  "nothing invented. The model runs at the source size."));
    }
    {   // 1.3 (24 Nov 2026): EXR compression, where an EXR is written; ACES 2065-1 is uncompressed by its standard
        const bool exr = c.output == "movie" ? (aces && c.keep_exr) : c.image != "dpx10";
        show("deliverExrCompressionRow", exr);
        if (auto* zc = findChild<QComboBox*>("deliverExrCompression")) {
            const QSignalBlocker b(zc);
            const bool ap0 = c.gamut == "ap0";
            const bool zlib = rudra::exr_zip_available();   // a build without zlib writes every EXR uncompressed
            zc->setCurrentIndex(std::max(0, zc->findData(ap0 || !zlib ? QStringLiteral("none") : c.exr_compression)));
            zc->setEnabled(!ap0 && zlib);
            zc->setToolTip(!zlib ? QStringLiteral("This build of RUDRA has no zlib: EXRs are written uncompressed.")
                           : ap0 ? QStringLiteral("ACES 2065-1 files are uncompressed: the SMPTE ST 2065-4 container asks for it. "
                                                "ACEScg or linear can be ZIP or ZIPS.")
                               : QStringLiteral("How the EXR masters are stored. ZIP and ZIPS are lossless (every value kept "
                                                "exactly) and make smaller files; ZIPS packs one line at a time instead of 16, a "
                                                "little faster to read a line at a time. Every OpenEXR reader takes all three."));
        }
    }
    show("deliverKeepExr", aces && movie);
    if (auto* keep = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverKeepExr"))) keep->set_on(c.keep_exr);
    show("deliverMattes", !aces);   // a movie's mattes go to <name>_mattes, a frame each
    if (auto* mt = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverMattes"))) mt->set_on(c.mattes);
    show("deliverBrackets", !aces && c.output != "movie");
    if (auto* br = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverBrackets"))) br->set_on(c.brackets);
    // The AI fill (layer 3): the learned EXR still or sequence only.
    const bool ai_possible = ai_fill_possible(c, aces);
    show("deliverAiFill", ai_possible);
    // 1.2.6: the cards of their own come and go with what they hold, so no empty card is left
    show("deliverAiCard", ai_possible);
    show("deliverExtrasCard", !aces);   // mattes and brackets: the analytic master's
    // 21 Nov 2026 (user feedback): the settings are Reconstruct's AI generation card, open while its
    // switch is on; Deliver's checkbox is greyed out until then.
    show("deliverAiFillOptions", ai_generation_);
    if (auto* g = dynamic_cast<CheckRow*>(findChild<QWidget*>("reconAiGeneration"))) g->set_on(ai_generation_);
    if (auto* ai = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverAiFill"))) {
        ai->set_on(c.ai_fill);
        ai->setEnabled(ai_generation_);
        ai->set_hint(ai_generation_ ? QStringLiteral("the export includes the fill")
                                    : QStringLiteral("turn on AI generation in Reconstruct first"));
        // 1.2.8 (seen running the app): Deliver's switch lines are tooltips (1.2.6), so a greyed switch
        // gave no reason. The reason shows while it is greyed out.
        ai->hint()->setVisible(!ai_generation_);
    }
    auto set_spin = [this](const char* id, int v) {
        if (auto* sb = findChild<QSpinBox*>(id)) {
            const QSignalBlocker b(sb);
            sb->setValue(v);
        }
    };
    set_spin("deliverAiTauHigh", c.ai.tau_high);
    set_spin("deliverAiTauLow", c.ai.tau_low);
    set_spin("deliverAiSeed", c.ai.seed);
    set_spin("deliverAiSteps", c.ai.steps);
    if (auto* bs = findChild<QDoubleSpinBox*>("deliverAiBrightness")) {
        const QSignalBlocker b(bs);
        bs->setValue(c.ai.brightness);
    }
    for (const auto& [id, v] : {std::pair{"deliverAiPromptStrength", c.ai.prompt_strength},
                                std::pair{"deliverAiShapeGain", c.ai.shape_gain}})
        if (auto* ds = findChild<QDoubleSpinBox*>(id)) {
            const QSignalBlocker b(ds);
            ds->setValue(v);
        }
    if (auto* fb = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverAiFallback"))) fb->set_on(c.ai.fallback);
    if (auto* fo = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverAiFocus"))) fo->set_on(c.ai.focus_region);
    if (auto* v = dynamic_cast<Seg*>(findChild<QWidget*>("deliverAiVersion"))) {
        v->set_on(c.ai.rerender ? "v4" : "v3");
        if (auto* b = v->button("v4")) b->setText(genfill_region_name() + QStringLiteral(" region"));   // 1.3: v5 where installed
    }
    if (auto* kd = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverAiKeepDetail"))) {   // 1.2.7: v4.3 only
        kd->set_on(c.ai.rerender_keep_detail);
        kd->setVisible(c.ai.rerender);
    }
    if (auto* rn = findChild<QLabel*>("deliverAiRegion"))   // the grade's region reaches the fill
        rn->setVisible(ai_generation_ && !c.ai.region_json.isEmpty());
    auto set_line = [this](const char* id, const QString& v) {
        if (auto* le = findChild<QLineEdit*>(id); le && le->text() != v) {
            const QSignalBlocker b(le);
            le->setText(v);
        }
    };
    set_line("deliverAiPromptHighlight", c.ai.prompt_highlight);
    set_line("deliverAiPromptShadow", c.ai.prompt_shadow);
    // AI fill v4 runs with its own fixed settings: v3's controls rest while it is on (12 Nov 2026),
    // so the card does not show thresholds and strengths that are not in use.
    for (const char* id : {"deliverAiTauHigh", "deliverAiTauLow", "deliverAiBrightness", "deliverAiPromptStrength",
                           "deliverAiShapeGain", "deliverAiFocus", "deliverAiFallback",
                           "deliverAiPromptShadow"})   // the highlight prompt and Suggest stay: v4 takes it (1.2.6)
        if (auto* wdg = findChild<QWidget*>(id)) {
            wdg->setEnabled(!c.ai.rerender);
            // 12 Nov 2026: and they are not shown (a number box goes with its heading)
            (qobject_cast<QAbstractSpinBox*>(wdg) ? wdg->parentWidget() : wdg)->setVisible(!c.ai.rerender);
        }
    if (auto* l = findChild<QLabel*>("deliverAiPromptShadowLabel")) l->setVisible(!c.ai.rerender);
    if (auto* st = findChild<QLabel*>("deliverAiSetup"); st && ai_generation_) {
        QString why;
        // Where each part was found (4 Nov 2026); missing: Setup… > Set up automatically.
        const auto s = genfill_setup(&why);
        if (genfill_ready_ != bool(s)) {   // the viewer's AI fill button and Preview follow (7 Nov 2026)
            genfill_ready_ = bool(s);
            if (genfill_ready_ && !aifill_runs_here_) {   // 24 Nov 2026: set up since launch; the button's note follows
                const AiFillHere now = aifill_here();
                aifill_runs_here_ = now.runs;
                aifill_why_ = now.why;
            }
            sync_ai_generation();
        }
        if (s && c.ai.rerender) {   // 12 Nov 2026: v4 is on, so the line names v4's LoRA, not v3's
            // 1.3: v5 first, v4.3 where only it is installed (genfill_setup_for)
            const auto v5 = genfill_v5_lora();
            const auto region = v5.empty() ? genfill_v4_lora() : v5;
            // One short line (12 Nov 2026); where the parts are is in the tooltip and in Setup…
            st->setText(region.empty() ? genfill_v4_missing_text()
                        : v5.empty()   ? QStringLiteral("Ready · AI fill v4.3 region (experimental; v5 in File \u203a Model packages)")
                                       : QStringLiteral("Ready · AI fill v5 region (experimental)"));
            st->setToolTip(QStringLiteral("Wan2.1-VACE-1.3B: %1\nLoRA: %2").arg(from_path(s->base), from_path(region)));
        } else if (s) {
            st->setText(s->lora_is_v3 ? QStringLiteral("Ready · AI fill v3")
                                      : QStringLiteral("Ready, but the LoRA is NOT AI fill v3: Setup… to switch"));
            st->setToolTip(QStringLiteral("Wan2.1-VACE-1.3B: %1\nLoRA: %2").arg(from_path(s->base), from_path(s->lora)));
        } else {
            // 1.2.5: what is missing in plain words, and where it is set up; the paths in the tooltip.
            // 1.2.6: one short line in the card; the whole sentence and the paths on hover.
            st->setText(QStringLiteral("Not set up yet: Setup\u2026 installs it."));
            st->setToolTip(genfill_missing_text(why) + QStringLiteral("\n\n") + why);
        }
    }
    // The preview's buttons: the views once there is one; a setting changed since, it says so.
    if (auto* b = findChild<QPushButton*>("deliverAiShowMask")) {   // fill or master: the viewer's AI fill button
        b->setEnabled(genfill_preview_.has_value() && !genfill_preview_busy_);
        b->parentWidget()->setVisible(genfill_preview_.has_value());   // 1.2.6: the view once there is a preview
    }
    if (auto* ps = findChild<QLabel*>("deliverAiPreviewStatus");
        ps && genfill_preview_ && !genfill_preview_busy_ && !(genfill_preview_->options == c.ai)) {
        const QString stale = QStringLiteral(" · settings changed since: Preview AI fill again");
        if (!ps->text().endsWith(stale)) ps->setText(ps->text() + stale);
    }
    sync_ai_frame_info();   // 1.2.6: on this frame, what the fill will change (Reconstruct's card, any format)
    show("deliverAnalyticOptions", !aces);
    // The Signal card says what the file is.
    const DeliverSignal sig = deliver_signal(c, aces);
    container_field_->setText(sig.container);
    if (auto* t = findChild<QLabel*>("transferField")) t->setText(sig.transfer);
    // RUDRA's white convention is a scene-linear one: said where the file is scene-linear,
    // not on a display-referred PQ, HLG or SDR file (28 Oct 2026 review).
    {
        const bool scene = sig.transfer.contains("scene-referred");
        if (auto* wr = findChild<QWidget*>("whiteRow")) wr->setVisible(scene);
        if (auto* wf = findChild<QLabel*>("whiteField")) {
            // 1.2.6: short values; the long form on hover
            wf->setText(sig.transfer.startsWith("linear") ? QStringLiteral("1.0 = 203 nits")
                                                          : QStringLiteral("the curve's 1.0"));
            wf->setToolTip(sig.transfer.startsWith("linear") ? QStringLiteral("1.0 is diffuse white, RUDRA's 203 nits")
                                                             : QStringLiteral("Diffuse white at the curve's 1.0 scene value"));
        }
    }
    primaries_field_->setText(sig.primaries);
    if (const QString pipe = deliver_pipe_label(c, aces); !pipe.isEmpty())
        if (auto* p = findChild<QLabel*>("pipeMaster")) p->setText(pipe);
    // Where it goes.
    QString note;
    const QString name = clean_name(findChild<QLineEdit*>("renderName")->text());
    const QString shown_name = name.isEmpty() ? QStringLiteral("<name>") : name;
    if (aces) {
        const int start = findChild<QSpinBox*>("renderStart")->value();
        const QString ext = movie ? QStringLiteral(".mov") : c.image == "dpx10" ? QStringLiteral(".dpx") : QStringLiteral(".exr");
        note = movie ? QStringLiteral("Writes %1.mov into the File's folder").arg(shown_name)
                     : QStringLiteral("Writes %1%2%3%4%5 into the File's folder")
                           .arg(shown_name, render_separator_)
                           .arg(start + (c.output == "still" ? std::max(current_, 0) : 0), render_padding_, 10, QChar('0'))
                           .arg(ext)
                           .arg(c.output == "sequence" ? QStringLiteral(" …") : QString());
        if (movie && c.bits == 16) note += QStringLiteral(". Uncompressed: large files, lossless");
        if (c.ai_fill)
            note += QSettings().value("genfill/replaceInMaster", true).toBool()
                        ? QStringLiteral(", then the AI fill fixes the clipped areas in place (the originals kept in %1_before_aifill)").arg(shown_name)
                        : QStringLiteral(", then the AI fill writes a copy in %1_aifill").arg(shown_name);
        note += QStringLiteral(". Nothing is ever replaced.");
    } else if (movie) {
        note = movie_shot ? QStringLiteral("Queued as an export with the movie's audio; see Queue.")
                          : QStringLiteral("The frames are mastered as EXR, then encoded (no audio).");
    }
    if (auto* n = findChild<QLabel*>("deliverNote")) {
        // 12 Nov 2026: what is written, in one line; the rest (the AI fill's folders, nothing replaced) on hover
        QString brief = note.contains(QStringLiteral(" into the File's folder"))
                            ? note.section(QStringLiteral(" into the File's folder"), 0, 0) +
                                  (c.ai_fill && aces ? QStringLiteral(", then AI fill") : QString())
                            : note;
        if (const auto [sw, sh] = shot_source_size(); c.size != "source" && sw > 0 && !brief.isEmpty()) {   // 1.3
            const OutputSize o = output_size(c.size.toStdString(), sw, sh);
            if (brief.endsWith(QLatin1Char('.'))) brief.chop(1);
            brief += QStringLiteral(", at %1 \u00d7 %2.").arg(o.width).arg(o.height);
            if (note != brief && !note.contains(QStringLiteral(" into the File's folder"))) note = brief;
        }
        n->setText(brief);
        n->setToolTip(brief == note ? QString() : note);
        n->setVisible(!note.isEmpty());
    }
    if (auto* hint = findChild<QLabel*>("renderHint")) hint->setVisible(false);   // 1.2.6: the Render card's tooltip
    sync_render_file();   // the File path's extension follows the type
    // The button: what Render does, or a click stops what runs.
    if (aces_delivering()) {
        btn_master_->setText("Rendering…");
        btn_master_->setEnabled(true);
    } else if (genfill_running()) {
        btn_master_->setText("AI filling…");
        btn_master_->setEnabled(true);
    } else if (!mastering() && !encoding_movie_) {
        btn_master_->setText(deliver_button_text(c, aces || wanted));
        if (aces) btn_master_->setEnabled(!frames_.empty());
        // The learned model chosen but not ready: the button says so; it never renders the analytic look instead.
        if (!aces && wanted) {
            btn_master_->setEnabled(!frames_.empty());
            btn_master_->setToolTip(QStringLiteral("The learned model is not ready (%1). Wait for it, load it, or choose "
                                                   "Analytic to render the analytic reconstruction (another look).")
                                        .arg(aces_engine_->status_text()));
        }
    }
}

QJsonObject MainWindow::aces_deliver_request(QString* why) const {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return QJsonObject{};
    };
    if (!deliver_aces()) return refuse(QStringLiteral("The learned ACES 2.0 model is not loaded."));
    if (frames_.empty()) return refuse(QStringLiteral("Open a shot first."));
    QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) return refuse(QStringLiteral("Choose a render folder first."));
    // "D:" alone is the drive's current folder to Windows, not its root: a render typed to "D:"
    // went looking in the app's own folder (12 Nov 2026). A bare drive means its root.
    if (dir.size() == 2 && dir[0].isLetter() && dir[1] == QLatin1Char(':')) dir += QLatin1Char('/');
    QString name = clean_name(findChild<QLineEdit*>("renderName")->text());
    if (name.isEmpty()) name = clean_name(QFileInfo(shot_title()).completeBaseName());
    if (name.isEmpty()) name = QStringLiteral("master");
    const DeliverChoice c = deliver_effective();
    // Into the folder of the File path, as Nuke's Write writes (29 Oct 2026): made when
    // Create directories is on, and nothing in it replaced (the engine checks every file).
    const fs::path out = to_path(dir);
    std::error_code ec;
    const auto* mk = findChild<QCheckBox*>("renderCreateDirs");
    if (!fs::is_directory(out, ec) && mk && !mk->isChecked())
        return refuse(QStringLiteral("%1 does not exist; turn on Create directories.").arg(QDir::toNativeSeparators(dir)));
    const bool movie = shot_is_video() && !shot_sources_.empty();
    const int at = std::max(current_, 0);
    const int first = findChild<QSpinBox*>("renderStart")->value() + (c.output == "still" ? at : 0);
    {
        const QString file = c.output == "movie" ? name + "." + render_ext(c)
                                                 : name + render_separator_ +
                                                       QStringLiteral("%1").arg(first, render_padding_, 10, QChar('0')) + "." +
                                                       render_ext(c);
        if (fs::exists(out / to_path(file), ec))
            // With the folder (12 Nov 2026): the Render folder is remembered from the last session, and
            // a refusal that named only the file read as "the render does not work".
            return refuse(QStringLiteral("%1 is already there; nothing is replaced. Change the name or version up. (Render folder: %2)")
                              .arg(file, QDir::toNativeSeparators(dir)));
    }
    QJsonArray sources;
    QJsonObject req = aces_deliver_fields(c);
    if (movie) {
        sources.append(qp(shot_sources_.front()));
        if (c.output == "still") req["frames"] = QJsonArray{at};
    } else if (c.output == "still") {
        sources.append(qp(frames_[std::size_t(at)]));
    } else {
        for (const auto& f : frames_) sources.append(qp(f));
    }
    // Frames have no clock of their own: their rate goes with them, for the movie and the
    // timecode on every frame (a typed :24 at 25 fps was refused or stamped at 24; 28 Oct 2026 review).
    if (!movie) req["fps"] = c.fps;
    req["op"] = "master";
    req["sources"] = sources;
    req["out"] = qp(out);
    req["name"] = name;
    req["start"] = first;
    req["into_existing"] = true;
    req["create_dirs"] = !mk || mk->isChecked();
    req["padding"] = std::clamp(render_padding_, 1, 10);
    req["separator"] = render_separator_;
    // The colorist's trims, and the timecode and reel for the conform (28 Oct 2026).
    if (const QJsonObject t = aces_trims_fields(aces_trims_.strength, aces_trims_.saturation, aces_trims_.peak_nits,
                                                 aces_trims_.exposure_ev, aces_trims_.contrast); !t.isEmpty())
        req["trims"] = t;
    if (const QString tc = render_timecode(); !tc.isEmpty()) {
        // A still of a sequence is frame `at` of it; a movie's own timecode wins unless this is set.
        // The shot's own rate: a movie's (29.97, 25, 59.94 ...), else the frames' chosen one.
        const double rate = movie && shot_video_ && shot_video_->fps && *shot_video_->fps > 0 ? *shot_video_->fps : c.fps;
        const std::string at_tc = timecode_at(tc.toStdString(), c.output == "still" && !movie ? at : 0, rate);
        if (at_tc.empty()) return refuse(QStringLiteral("The start timecode is HH:MM:SS:FF (or ;FF for drop frame)."));
        req["timecode"] = QString::fromStdString(c.output == "still" && !movie ? at_tc : tc.toStdString());
    }
    if (const QString reel = render_reel(); !reel.isEmpty()) req["reel"] = reel;
    if (aces_source_display_ != "bt1886") req["source_display"] = aces_source_display_;
    if (const QJsonObject r = aces_region_field(); !r.isEmpty()) req["reconstruction_region"] = r;   // as the viewer shows it
    return req;
}

void MainWindow::deliver_render() {
    auto* status = findChild<QLabel*>("renderStatus");
    auto say = [this, status](const QString& t) {
        status->setText(t);
        status->setVisible(!t.isEmpty());
        log(t);
    };
    if (aces_delivering()) {   // a click while it renders stops after the frame in hand
        if (aces_engine_ && aces_deliver_id_ > 0) aces_engine_->cancel_master(aces_deliver_id_);
        status->setText(QStringLiteral("Stopping after this frame…"));
        return;
    }
    if (genfill_running()) {   // a click while the AI fill runs stops it (the frames written stay)
        genfill_->cancel();
        status->setText(QStringLiteral("Stopping the AI fill…"));
        return;
    }
    if (!deliver_aces() && deliver_learned_wanted()) {   // never the analytic look in its place (28 Oct 2026 review)
        say(QStringLiteral("Not rendered: the learned ACES 2.0 model is not ready (%1). Wait for it or load it, or choose "
                           "Analytic on the Deliver tab for the analytic reconstruction.")
                .arg(aces_engine_->status_text()));
        return;
    }
    if (deliver_aces() && batch_running()) {   // one master at a time through the engine
        say(QStringLiteral("The batch queue is rendering through the model; stop it first (the Batch tab)."));
        return;
    }
    if (deliver_aces()) {
        aces_deliver();
        return;
    }
    const DeliverChoice c = deliver_effective();
    if (c.output == "movie") {
        const QString format = analytic_movie_format(c);
        if (shot_is_video()) {
            if (queue_video_export(format)) open_queue_window();
        } else if (shot_is_frame_folder()) {
            export_frames_to_movie(format, c.fps);
        } else {
            say(QStringLiteral("A MOV is made from a movie or a folder of frames; a single still goes out as a still."));
        }
        return;
    }
    master();
}

void MainWindow::aces_deliver() {
    auto* status = findChild<QLabel*>("renderStatus");
    QString why;
    QJsonObject req = aces_deliver_request(&why);
    if (req.isEmpty()) {
        status->setText(why);
        status->show();
        if (why.startsWith("Choose a render folder")) findChild<QLineEdit*>("renderDir")->setFocus();
        log("Render stopped: " + why);
        return;
    }
    halt_playback();
    // A movie with the AI fill (6 Nov 2026): the masters as EXR first (the fill reads EXR), the
    // movie made from the filled frames after it, with the same name, format, rate and timecode.
    aifill_movie_req_ = {};
    render_with_fill_ = deliver_effective().ai_fill;   // 1.2.8: the fill is decided at the start
    if (const DeliverChoice c = deliver_effective(); c.output == "movie" && c.ai_fill) {
        QJsonObject enc{{"op", "encode"}, {"out", req.value("out")}, {"name", req.value("name")}, {"log", req.value("log")},
                        {"log_formats", req.value("log_formats")}};
        const double rate = shot_video_ && shot_video_->fps && *shot_video_->fps > 0 ? *shot_video_->fps : c.fps;
        enc["fps"] = req.contains("fps") ? req.value("fps").toDouble() : rate;
        for (const char* k : {"timecode", "reel"})
            if (req.contains(k)) enc[k] = req.value(k);
        aifill_movie_req_ = enc;
        req["exr"] = c.gamut == "ap0" ? "half" : "float";
        req["exr_gamut"] = c.gamut == "ap0" ? "ap0" : "acescg";
        req.remove("log");
        req.remove("log_formats");
    }
    const QString out = req.value("out").toString();
    const int total = req.value("frames").isArray() ? int(req.value("frames").toArray().size()) : -1;
    log("Render destination: " + QDir::toNativeSeparators(out) + " (learned ACES 2.0, " + container_field_->text() + ")");
    status->setText(QStringLiteral("Rendering with the learned ACES 2.0 model…"));
    status->show();
    btn_master_->setToolTip("Click to stop after the frame being written");
    const QPointer<MainWindow> self(this);
    aces_deliver_id_ = -1;   // running; the engine's id once it is sent
    // The AI fill after it reads only this render's frames from the shared folder.
    render_name_for_fill_ = req.value("into_existing").toBool() ? req.value("name").toString() : QString();
    // The QC report's context, as this render starts (code review, 19 Nov 2026).
    render_qc_context_ = qc_context_now(QStringLiteral("Learned ACES 2.0"),
                                        QJsonDocument::fromJson(QByteArray::fromStdString(master_request_json(master_request()))).object());
    const int id = aces_engine_->request(
        req,
        [self, out](const QJsonObject& o) {
            if (self) self->aces_deliver_finished(o, out);
        },
        [self, total](const QJsonObject& o) {
            if (!self) return;
            const QJsonObject p = o.value("progress").toObject();
            const int n = total > 0 ? total : p.value("total").toInt();
            if (auto* st = self->findChild<QLabel*>("renderStatus"))
                st->setText(QStringLiteral("Rendering %1 / %2 · %3 ms a frame")
                                .arg(p.value("index").toInt())
                                .arg(n)
                                .arg(p.value("ms").toDouble(), 0, 'f', 0));
            const QString saved = p.value("exr").toString();   // the engine does not always name the file
            self->log(saved.isEmpty() ? QStringLiteral("Saved frame %1 of %2").arg(p.value("index").toInt()).arg(n) : "Saved " + saved);
        });
    if (aces_deliver_id_ != 0) aces_deliver_id_ = id;   // still running (a refusal answers at once)
    refresh_enabled();
    sync_ui();
}

void MainWindow::aces_deliver_finished(const QJsonObject& o, const QString& out) {
    aces_deliver_id_ = 0;
    auto* status = findChild<QLabel*>("renderStatus");
    btn_master_->setToolTip({});
    if (!o.value("ok").toBool()) {
        const QString why = (o.value("error").toString() + ' ' + o.value("detail").toString()).trimmed();
        status->setText("Stopped: " + why);
        log("Render stopped: " + why);
    } else {
        const int n = o.value("frames").toInt();
        const bool cancelled = o.value("cancelled").toBool();
        QString text = cancelled ? QStringLiteral("Stopped after %1 frame(s); written to %2").arg(n).arg(QDir::toNativeSeparators(out))
                                 : QStringLiteral("Rendered %1 frame(s) to %2").arg(n).arg(QDir::toNativeSeparators(out));
        const QString prov = o.value("provenance").toString();
        if (auto s = read_aces_provenance(to_path((prov.isEmpty() ? out : prov)))) {
            log(QString::fromStdString(aces_summary_text(*s)));
            if (!s->appearance_pass) text += QStringLiteral(" · the appearance check did not pass: see provenance.json");
        }
        if (const QString h = o.value("hdr10").toString(); !h.isEmpty())   // the HDR10 numbers of the PQ files
            text += QStringLiteral(" · HDR10 MaxCLL %1 / MaxFALL %2 nits (%3)")
                        .arg(std::lround(std::ceil(o.value("maxcll").toDouble())))
                        .arg(std::lround(std::ceil(std::min(o.value("maxfall").toDouble(), o.value("maxcll").toDouble()))))
                        .arg(QFileInfo(h).fileName()) +
                    (o.value("dolby_vision").toString().isEmpty()
                         ? QString()
                         : QStringLiteral(" · Dolby Vision XML (%1)").arg(QFileInfo(o.value("dolby_vision").toString()).fileName()));
        if (n > 0) {   // 1.2.6: the QC report, from the masters' sidecars and the run's provenance
            QJsonObject learned = o;
            if (auto s = read_aces_provenance(to_path((prov.isEmpty() ? out : prov)))) {
                learned["remaining_display_errors"] = qint64(s->remaining_display_errors);
                learned["appearance_pass"] = s->appearance_pass;
                learned["mean_reconstructed_fraction"] = s->mean_reconstructed_fraction;
            }
            std::vector<std::filesystem::path> masters;
            std::error_code ec;
            for (const auto& e : std::filesystem::directory_iterator(to_path(out), ec))
                if (e.path().extension() == ".exr" || e.path().extension() == ".dpx") masters.push_back(e.path());
            std::sort(masters.begin(), masters.end());
            if (const QString qc = write_render_qc(out, masters, QStringLiteral("Learned ACES 2.0"), learned); !qc.isEmpty())
                text += QStringLiteral(" \u00b7 QC: ") + QFileInfo(qc).fileName();
        }
        status->setText(text);
        log(text);
        delivered_ += n;
        if (auto* d = findChild<QLabel*>("colDelivered")) d->setText(QString::number(delivered_));
        if (n > 0) note_media_delivered();   // 1.2.6: the Media panel's Delivered
        if (!cancelled && n > 0 && render_with_fill_) {   // 1.2.8: as decided when the render started
            QString why;
            // A movie with the AI fill: once the fill is in, its frames make the movie.
            std::function<void(bool, QString)> then;
            if (!aifill_movie_req_.isEmpty()) {
                const QPointer<MainWindow> self(this);
                then = [self, out](bool done, const QString&) {
                    if (!self) return;
                    if (done) self->encode_aifill_movie(out, self->genfill_last_out_);
                    else self->aifill_movie_req_ = {};
                };
            }
            if (!start_genfill(out, &why, then)) {
                aifill_movie_req_ = {};
                status->setText(text + QStringLiteral(" · AI fill not run: ") + why);
                log("AI fill not run: " + why);
            }
        } else {
            aifill_movie_req_ = {};
        }
    }
    status->show();
    refresh_enabled();
    sync_ui();
}

bool MainWindow::genfill_running() const { return genfill_ && genfill_->running(); }

bool MainWindow::review_genfill(const QString& dir_in, QString* why) {
    QString dir = dir_in.isEmpty() ? genfill_last_out_ : dir_in;
    if (dir.isEmpty() && isVisible())
        dir = QFileDialog::getExistingDirectory(this, "An AI fill folder (<name>_aifill)", render_dir());
    if (dir.isEmpty()) {
        if (why) *why = QStringLiteral("No AI fill yet.");
        return false;
    }
    QString w;
    auto r = read_genfill_review(dir, &w);
    if (!r) {
        if (why) *why = w;
        if (auto* st = findChild<QLabel*>("renderStatus")) {
            st->setText(w);
            st->show();
        }
        return false;
    }
    log(QStringLiteral("AI fill review: %1 frames, %2 flagged for flicker").arg(r->frames.size()).arg(r->flagged()));
    if (!isVisible()) return true;   // tests: the review read, no window
    GenFillReviewDialog(std::move(*r), this).exec();
    return true;
}

bool MainWindow::start_genfill(const QString& master_dir, QString* why, std::function<void(bool, QString)> finished) {
    // The Deliver render's: its frames only when it wrote into a folder that holds others.
    GenFillOptions o = deliver_effective().ai;
    QString base = master_dir;
    if (!render_name_for_fill_.isEmpty()) {
        o.glob = render_name_for_fill_ + render_separator_ + QString("[0-9]").repeated(render_padding_) + "*.exr";
        base = QDir(master_dir).filePath(render_name_for_fill_);
    }
    return start_genfill_with(master_dir, o, base, std::move(finished), why);
}

void MainWindow::encode_aifill_movie(const QString& master_dir, const QString& fill_dir) {
    auto* status = findChild<QLabel*>("renderStatus");
    QJsonObject req = aifill_movie_req_;
    aifill_movie_req_ = {};
    if (req.isEmpty()) return;
    // This render's filled frames, in order: the fill folder's, or the master's once fixed in place.
    const QString prefix = req.value("name").toString() + render_separator_;
    QJsonArray exrs;
    for (const QString& f : genfill_filled_frames(fill_dir, master_dir))
        if (QFileInfo(f).fileName().startsWith(prefix)) exrs.append(qp(to_path(f)));
    if (exrs.isEmpty()) {
        const QString t = QStringLiteral("AI fill: no filled frames were found for the movie in %1").arg(QDir::toNativeSeparators(master_dir));
        status->setText(t);
        status->show();
        log(t);
        return;
    }
    req["exrs"] = exrs;
    if (!aces_engine_) return;
    const int total = int(exrs.size());
    status->setText(QStringLiteral("Making the movie from the AI-filled frames…"));
    status->show();
    btn_master_->setToolTip("Click to stop the movie (the filled EXR frames stay)");
    const QPointer<MainWindow> self(this);
    aces_deliver_id_ = -1;
    const int id = aces_engine_->request(
        req,
        [self, master_dir](const QJsonObject& o) {
            if (!self) return;
            self->aces_deliver_id_ = 0;
            self->btn_master_->setToolTip({});
            auto* st = self->findChild<QLabel*>("renderStatus");
            QString t;
            if (!o.value("ok").toBool())
                t = QStringLiteral("The AI-filled movie was not made: ") + (o.value("error").toString() + ' ' + o.value("detail").toString()).trimmed() +
                    QStringLiteral(" (the filled EXR frames are in %1)").arg(QDir::toNativeSeparators(master_dir));
            else if (o.value("cancelled").toBool())
                t = QStringLiteral("Movie stopped; the filled EXR frames are in %1").arg(QDir::toNativeSeparators(master_dir));
            else
                t = QStringLiteral("AI-filled movie: %1 (%2 frames; the filled EXR frames are beside it)")
                        .arg(QDir::toNativeSeparators(o.value("movie").toString()))
                        .arg(o.value("frames").toInt()) +
                    (o.value("hdr10").toString().isEmpty()
                         ? QString()
                         : QStringLiteral(" · HDR10 MaxCLL %1 / MaxFALL %2 nits")
                               .arg(std::lround(std::ceil(o.value("maxcll").toDouble())))
                               .arg(std::lround(std::ceil(std::min(o.value("maxfall").toDouble(), o.value("maxcll").toDouble())))));
            st->setText(t);
            st->show();
            self->log(t);
            self->refresh_enabled();
            self->sync_ui();
        },
        [self, total](const QJsonObject& p) {
            if (!self) return;
            if (auto* st = self->findChild<QLabel*>("renderStatus"))
                st->setText(QStringLiteral("Making the AI-filled movie %1 / %2").arg(p.value("index").toInt()).arg(total));
        });
    if (aces_deliver_id_ != 0) aces_deliver_id_ = id;
    refresh_enabled();
}

bool MainWindow::start_genfill_with(const QString& master_dir, const GenFillOptions& options, const QString& out_base,
                                    std::function<void(bool, QString)> finished, QString* why) {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return false;
    };
    if (genfill_running()) return refuse(QStringLiteral("An AI fill is already running."));
    QString w;
    // AI fill v4 (region re-render): the fill needs the shot the master was made from, the frame the
    // master starts at (Current image renders one frame) and the route's inverse; and it runs with
    // the v4 LoRA. The master is taken to be of the shot that is open.
    GenFillOptions fill_options = options;
    if (fill_options.rerender) {
        if (frames_.empty()) return refuse(QStringLiteral("AI fill region re-render: open the shot the master was made from."));
        const bool movie = shot_is_video() && !shot_sources_.empty();
        fill_options.source = qp(movie ? shot_sources_.front() : frames_.front().parent_path());
        fill_options.condition = deliver_aces() ? QStringLiteral("aces") : QStringLiteral("bt1886");
        fill_options.source_start = deliver_effective().output == "still" ? std::max(current_, 0) : 0;
    }
    const auto setup = genfill_setup_for(fill_options, &w);
    if (!setup) {
        if (!fill_options.rerender) open_system_check(false);   // 1.2.5: where it is set up
        return refuse(fill_options.rerender ? w : genfill_missing_text(w));
    }
    if (!genfill_) genfill_ = std::make_shared<GenFillJob>(this);
    log(fill_options.rerender ? QStringLiteral("AI fill: v4, region re-render (LoRA %1, condition %2, source %3)")
                                    .arg(qp(setup->lora), fill_options.condition, fill_options.source)
                              : QStringLiteral("AI fill: v3, clipped pixels only (LoRA %1)").arg(qp(setup->lora)));
    const QString out = genfill_out_dir(out_base.isEmpty() ? master_dir : out_base);
    auto* status = findChild<QLabel*>("renderStatus");
    // 1.3 (23 Nov 2026): disk space, time and a busy GPU, before anything starts (genfill_preflight).
    {
        const QFileInfo mi(master_dir);
        QStringList files;
        if (mi.isFile()) files << mi.absoluteFilePath();
        else
            for (const QString& f : QDir(master_dir).entryList({fill_options.glob.isEmpty() ? QStringLiteral("*.exr") : fill_options.glob},
                                                               QDir::Files, QDir::Name))
                files << QDir(master_dir).filePath(f);
        int fw = 0, fh = 0;
        if (!files.isEmpty())
            if (auto e = read_exr(to_path(files.front()))) fw = e->pixels.width(), fh = e->pixels.height();
        QString disk = QFileInfo(out).absolutePath();
        while (!disk.isEmpty() && !QFileInfo::exists(disk)) {
            const QString up = QFileInfo(disk).absolutePath();
            if (up == disk) break;
            disk = up;
        }
        const QStorageInfo si(disk);
        const qint64 free = si.isValid() && si.isReady() ? si.bytesAvailable() : -1;
        auto load = QSettings().value("genfill/checkGpu", true).toBool() ? genfill_gpu_load() : std::pair<int, int>{-1, -1};
        if (genfill_server_ && genfill_server_->loaded()) load.second = -1;   // the memory is our own warm preview model
        const GenFillPreflight pf = genfill_preflight(int(files.size()), fw, fh, free,
                                                      QSettings().value("genfill/secPerFrame", kGenFillSecPerFrame).toDouble(),
                                                      load.first, load.second);
        log(pf.text);
        if (!pf.ok) return refuse(pf.text);
        if (status) {
            status->setText(pf.text);
            status->show();
        }
        statusBar()->showMessage(pf.text, 15000);
    }
    const QPointer<MainWindow> self(this);
    const bool ok = genfill_->start(
        *setup, fill_options, master_dir, out,
        [self](const QJsonObject& o) {
            if (!self) return;
            auto* st = self->findChild<QLabel*>("renderStatus");
            const QString ev = o.value("event").toString();
            // 1.2.6: and the AI fill's progress bars (the windows' steps, then the frames written)
            const double was = std::max(0.0, self->ai_progress_share());
            if (ev == "embedding") {   // a prompt not in the cache: umT5 loads once to embed it (6 Nov 2026)
                st->setText(QStringLiteral("AI fill: embedding %1 new prompt(s), once (a minute or more)…")
                                .arg(o.value("prompts").toArray().size()));
                self->set_ai_progress(-1.0, QStringLiteral("embedding the prompt (once)"));
            } else if (ev == "ready") {
                st->setText(QStringLiteral("AI fill: model loaded on %1…").arg(o.value("device").toString()));
                self->set_ai_progress(0.0, QStringLiteral("model loaded"));
            } else if (ev == "progress") {
                st->setText(QStringLiteral("AI fill: window %1 / %2 · step %3 / %4")
                                .arg(o.value("window").toInt()).arg(o.value("windows").toInt())
                                .arg(o.value("step").toInt()).arg(o.value("steps").toInt()));
                self->set_ai_progress(std::max(was, genfill_progress_share(0.0, 0.98, o.value("window").toInt(1), o.value("windows").toInt(1),
                                                                           o.value("step").toInt(), o.value("steps").toInt(1))),
                                      QStringLiteral("window %1 / %2").arg(o.value("window").toInt()).arg(o.value("windows").toInt()));
            } else if (ev == "frame") {
                st->setText(QStringLiteral("AI fill: frame %1 / %2 written").arg(o.value("index").toInt()).arg(o.value("of").toInt()));
                self->set_ai_progress(std::max(was, 0.98 * o.value("index").toInt() / std::max(1, o.value("of").toInt())),
                                      QStringLiteral("frame %1 / %2").arg(o.value("index").toInt()).arg(o.value("of").toInt()));
                self->log("AI fill saved " + o.value("file").toString());
            }
            st->show();
        },
        [self, out, master_dir, finished](bool done, const QJsonObject& last) {
            if (!self) return;
            self->ai_progress_done();
            auto* st = self->findChild<QLabel*>("renderStatus");
            QString text;
            if (done) {
                // 1.3: this machine's speed, for the next fill's time estimate (genfill_preflight)
                const int nf = last.value("frames").toInt();
                if (nf > 0 && last.value("seconds").toDouble() > 0)
                    QSettings().setValue("genfill/secPerFrame", last.value("seconds").toDouble() / nf);
                self->genfill_last_out_ = out;
                self->genfill_last_master_ = master_dir;
                self->fill_view_cache_.clear();
                // 4 Nov 2026 (Ahmed: option a): the fill fixes the master in place; the
                // originals go to <master>_before_aifill. Only after a fill that finished.
                QString where = QStringLiteral("to %1").arg(QDir::toNativeSeparators(out));
                if (QSettings().value("genfill/replaceInMaster", true).toBool()) {
                    const GenFillSwap sw = genfill_replace_in_master(master_dir, out);
                    if (!sw.error.isEmpty())
                        where = QStringLiteral("to %1 (the master was NOT updated: %2)").arg(QDir::toNativeSeparators(out), sw.error);
                    else if (sw.replaced > 0)
                        where = QStringLiteral("fixed in place in %1; the originals are in %2")
                                    .arg(QDir::toNativeSeparators(master_dir), QDir::toNativeSeparators(sw.backup));
                }
                // 1.2.6 (checklist item 6): the disclosure record beside genfill.json.
                const QString route = self->deliver_choice().source == QLatin1String("aces") ? QStringLiteral("Learned ACES 2.0")
                                                                                           : QStringLiteral("Analytic");
                const QString disclosure = write_genfill_disclosure(out, route);
                if (!disclosure.isEmpty()) where += QStringLiteral(" · disclosure: %1").arg(QDir::toNativeSeparators(disclosure));
                if (const QString qc = self->rewrite_qc_with_fill(master_dir, disclosure); !qc.isEmpty())   // 1.3
                    where += QStringLiteral(" · QC: %1").arg(QFileInfo(qc).fileName());
                text = QStringLiteral("AI fill: %1 frame(s) %2 · %3 % of pixels generated")
                           .arg(last.value("frames").toInt())
                           .arg(where)
                           .arg(100.0 * last.value("generated_fraction").toDouble(), 0, 'f', 2) +
                       (last.value("flagged_frames").toInt() > 0
                            ? QStringLiteral(" · %1 frame(s) flicker: Review fill… before it goes out").arg(last.value("flagged_frames").toInt())
                            : QStringLiteral(" · Review fill… to check it"));
            } else if (last.value("cancelled").toBool()) {
                text = QStringLiteral("AI fill stopped; the frames written are in %1").arg(QDir::toNativeSeparators(out));
            } else {
                text = QStringLiteral("AI fill failed: ") + (last.value("error").toString() + ' ' + last.value("detail").toString()).trimmed();
            }
            st->setText(text);
            st->show();
            self->log(text);
            self->btn_master_->setToolTip({});
            self->refresh_enabled();
            self->sync_ui();
            if (finished) finished(done, text);
        },
        &w);
    if (!ok) return refuse(w);
    log("AI fill: " + QDir::toNativeSeparators(master_dir) + " -> " + QDir::toNativeSeparators(out));
    status->setText(QStringLiteral("AI fill: loading the model…"));
    status->show();
    set_ai_progress(-1.0, QStringLiteral("loading the fill model"));
    btn_master_->setToolTip("Click to stop the AI fill (the frames written stay)");
    refresh_enabled();
    sync_ui();
    return true;
}

}  // namespace rudra::app
