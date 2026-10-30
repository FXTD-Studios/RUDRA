// The Deliver tab (27 Oct 2026): the source, still / sequence / MOV, the file
// type, its colour and the movie's bit depth (deliver_choice.hpp), and Render:
// the learned ACES 2.0 masters through the engine kept loaded (aces_engine.hpp,
// cli/aces_v2_deliver.hpp), or the analytic master and movie exports as before.

#include <QCheckBox>
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

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "aces_engine.hpp"
#include "paths.hpp"
#include "aces_master.hpp"
#include "genfill_job.hpp"
#include "genfill_review.hpp"
#include "main_window.hpp"
#include "rudra/core/timecode.hpp"
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

bool MainWindow::deliver_aces() const {
    return deliver_.source == "aces" && aces_engine_ && aces_engine_->state() == AcesEngine::State::Ready;
}

bool MainWindow::deliver_learned_wanted() const {
    return deliver_.source == "aces" && aces_engine_ && aces_engine_->state() != AcesEngine::State::Stopped;
}

DeliverChoice MainWindow::deliver_effective() const {
    const bool aces = deliver_aces() || deliver_learned_wanted();
    DeliverChoice c = deliver_;
    if (!aces) c.gamut = session_.container == "linear" ? QStringLiteral("linear") : QStringLiteral("ap0");
    const bool movie = shot_is_video();
    return deliver_fit(c, aces, movie, !movie && frames_.size() > 1);
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
    c.fps = st.value("deliver/fps", c.fps).toDouble();
    c.ai_fill = st.value("deliver/aiFill", c.ai_fill).toBool();
    c.ai.tau_high = st.value("deliver/aiTauHigh", c.ai.tau_high).toInt();
    c.ai.tau_low = st.value("deliver/aiTauLow", c.ai.tau_low).toInt();
    c.ai.seed = st.value("deliver/aiSeed", c.ai.seed).toInt();
    c.ai.steps = st.value("deliver/aiSteps", c.ai.steps).toInt();
    c.ai.brightness = st.value("deliver/aiBrightness", c.ai.brightness).toDouble();
    c.ai.prompt_highlight = st.value("deliver/aiPromptHighlight").toString();
    c.ai.prompt_shadow = st.value("deliver/aiPromptShadow").toString();
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
    } else if (key == "gamut" && (value == "ap0" || value == "acescg" || value == "linear")) {
        // The analytic master's gamut is the session's container (Deliver menu).
        if (!deliver_aces() && value != "acescg") run(value == "ap0" ? "container-aces" : "container-linear");
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
    } else if (key == "keep_exr") {
        deliver_.keep_exr = value == "1" || value == "true";
        st.setValue("deliver/keepExr", deliver_.keep_exr);
    } else if (key == "fps") {
        bool ok = false;
        const double f = value.toDouble(&ok);
        if (ok && f > 0 && f <= 240) {
            deliver_.fps = f;
            st.setValue("deliver/fps", f);
        }
    } else if (key == "ai_fill") {
        deliver_.ai_fill = value == "1" || value == "true";
        st.setValue("deliver/aiFill", deliver_.ai_fill);
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
    auto* out = dynamic_cast<Seg*>(findChild<QWidget*>("deliverOutput"));
    out->set_on(c.output);
    out->button("movie")->setEnabled(movie_shot || folder);
    out->button("movie")->setToolTip(movie_shot || folder ? QStringLiteral("One QuickTime movie of every frame")
                                                          : QStringLiteral("For a movie or a folder of frames; a single still goes out as a still"));
    out->button("sequence")->setEnabled(!frames_.empty());
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
        gamut->button("acescg")->setVisible(aces);
        gamut->button("linear")->setVisible(!aces);
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
    show("deliverKeepExr", aces && movie);
    if (auto* keep = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverKeepExr"))) keep->set_on(c.keep_exr);
    // The AI fill (layer 3): the learned EXR still or sequence only.
    const bool ai_possible = ai_fill_possible(c, aces);
    show("deliverAiFill", ai_possible);
    show("deliverAiFillOptions", ai_possible && c.ai_fill);
    if (auto* ai = dynamic_cast<CheckRow*>(findChild<QWidget*>("deliverAiFill"))) ai->set_on(c.ai_fill);
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
    auto set_line = [this](const char* id, const QString& v) {
        if (auto* le = findChild<QLineEdit*>(id); le && le->text() != v) {
            const QSignalBlocker b(le);
            le->setText(v);
        }
    };
    set_line("deliverAiPromptHighlight", c.ai.prompt_highlight);
    set_line("deliverAiPromptShadow", c.ai.prompt_shadow);
    if (auto* st = findChild<QLabel*>("deliverAiSetup"); st && ai_possible && c.ai_fill) {
        QString why;
        st->setText(genfill_setup(&why) ? QStringLiteral("Ready: Wan2.1-VACE-1.3B + the RUDRA fill LoRA") : why);
    }
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
        if (auto* wf = findChild<QLabel*>("whiteField"))
            wf->setText(sig.transfer.startsWith("linear") ? QStringLiteral("1.0 = diffuse white (RUDRA's 203 nits)")
                                                          : QStringLiteral("diffuse white at the curve's 1.0 scene value"));
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
        if (c.ai_fill) note += QStringLiteral(", then the AI fill writes a copy in %1_aifill").arg(shown_name);
        note += QStringLiteral(". Nothing is ever replaced.");
    } else if (movie) {
        note = movie_shot ? QStringLiteral("Queued as an export with the movie's audio; see Queue.")
                          : QStringLiteral("The frames are mastered as EXR, then encoded (no audio).");
    }
    if (auto* n = findChild<QLabel*>("deliverNote")) {
        n->setText(note);
        n->setVisible(!note.isEmpty());
    }
    if (auto* hint = findChild<QLabel*>("renderHint")) hint->setVisible(!aces);
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
    const QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) return refuse(QStringLiteral("Choose a render folder first."));
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
            return refuse(QStringLiteral("%1 is already there; nothing is replaced. Change the name or version up.").arg(file));
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
        say(QStringLiteral("The batch queue is rendering through the model; stop it first (Deliver > Batch queue…)."));
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
    const QJsonObject req = aces_deliver_request(&why);
    if (req.isEmpty()) {
        status->setText(why);
        status->show();
        if (why.startsWith("Choose a render folder")) findChild<QLineEdit*>("renderDir")->setFocus();
        log("Render stopped: " + why);
        return;
    }
    halt_playback();
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
            self->log("Saved " + p.value("exr").toString());
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
        status->setText(text);
        log(text);
        delivered_ += n;
        if (auto* d = findChild<QLabel*>("colDelivered")) d->setText(QString::number(delivered_));
        if (!cancelled && n > 0 && deliver_effective().ai_fill) {
            QString why;
            if (!start_genfill(out, &why)) {
                status->setText(text + QStringLiteral(" · AI fill not run: ") + why);
                log("AI fill not run: " + why);
            }
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

bool MainWindow::start_genfill(const QString& master_dir, QString* why) {
    // The Deliver render's: its frames only when it wrote into a folder that holds others.
    GenFillOptions o = deliver_effective().ai;
    QString base = master_dir;
    if (!render_name_for_fill_.isEmpty()) {
        o.glob = render_name_for_fill_ + render_separator_ + QString("[0-9]").repeated(render_padding_) + "*.exr";
        base = QDir(master_dir).filePath(render_name_for_fill_);
    }
    return start_genfill_with(master_dir, o, base, {}, why);
}

bool MainWindow::start_genfill_with(const QString& master_dir, const GenFillOptions& options, const QString& out_base,
                                    std::function<void(bool, QString)> finished, QString* why) {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return false;
    };
    if (genfill_running()) return refuse(QStringLiteral("An AI fill is already running."));
    QString w;
    const auto setup = genfill_setup(&w);
    if (!setup) return refuse(w);
    if (!genfill_) genfill_ = std::make_shared<GenFillJob>(this);
    const QString out = genfill_out_dir(out_base.isEmpty() ? master_dir : out_base);
    auto* status = findChild<QLabel*>("renderStatus");
    const QPointer<MainWindow> self(this);
    const bool ok = genfill_->start(
        *setup, options, master_dir, out,
        [self](const QJsonObject& o) {
            if (!self) return;
            auto* st = self->findChild<QLabel*>("renderStatus");
            const QString ev = o.value("event").toString();
            if (ev == "ready") {
                st->setText(QStringLiteral("AI fill: model loaded on %1…").arg(o.value("device").toString()));
            } else if (ev == "progress") {
                st->setText(QStringLiteral("AI fill: window %1 / %2 · step %3 / %4")
                                .arg(o.value("window").toInt()).arg(o.value("windows").toInt())
                                .arg(o.value("step").toInt()).arg(o.value("steps").toInt()));
            } else if (ev == "frame") {
                st->setText(QStringLiteral("AI fill: frame %1 / %2 written").arg(o.value("index").toInt()).arg(o.value("of").toInt()));
                self->log("AI fill saved " + o.value("file").toString());
            }
            st->show();
        },
        [self, out, finished](bool done, const QJsonObject& last) {
            if (!self) return;
            auto* st = self->findChild<QLabel*>("renderStatus");
            QString text;
            if (done) {
                self->genfill_last_out_ = out;
                text = QStringLiteral("AI fill: %1 frame(s) to %2 · %3 % of pixels generated")
                           .arg(last.value("frames").toInt())
                           .arg(QDir::toNativeSeparators(out))
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
    btn_master_->setToolTip("Click to stop the AI fill (the frames written stay)");
    refresh_enabled();
    sync_ui();
    return true;
}

}  // namespace rudra::app
