#include "deliver_choice.hpp"

#include <QDir>
#include <QRegularExpression>

#include <algorithm>

namespace rudra::app {

DeliverChoice deliver_fit(DeliverChoice c, bool aces_source, bool shot_movie, bool shot_folder) {
    if (c.output != "still" && c.output != "sequence" && c.output != "movie") c.output = "still";
    if (c.image != "exr32" && c.image != "exr16" && c.image != "dpx10") c.image = "exr32";
    if (c.encoding != "logc4" && c.encoding != "acescct" && c.encoding != "pq" && c.encoding != "hlg" && c.encoding != "sdr")
        c.encoding = "logc4";
    if (c.pq_peak != 1000 && c.pq_peak != 2000 && c.pq_peak != 4000) c.pq_peak = 1000;
    // HLG, the SDR pass and the 2000 / 4000-nit targets are the learned engine's; the analytic PQ is graded on its tab.
    if (!aces_source && (c.encoding == "hlg" || c.encoding == "sdr")) c.encoding = "pq";
    if (!aces_source) c.pq_peak = 1000;
    if (c.bits != 10 && c.bits != 16) c.bits = 10;
    if (c.codec != "prores422hq" && c.codec != "prores4444" && c.codec != "prores4444xq" && c.codec != "hevc10")
        c.codec = "prores422hq";
    // ProRes 4444 XQ and HDR10 HEVC are the learned engine's (28 Oct 2026); HEVC is PQ, 10-bit.
    if (!aces_source && c.codec == "prores4444xq") c.codec = "prores4444";
    if (!aces_source && c.codec == "hevc10") c.codec = "prores422hq";
    if (c.codec == "hevc10") {   // HDR10 is PQ (at its peak), 10-bit
        if (c.encoding != "pq") c.encoding = "pq";
        c.bits = 10;
    }
    if (!(c.fps > 0.0 && c.fps <= 240.0)) c.fps = 24.0;
    // A movie is made from a movie or a folder of frames, not from one still.
    if (c.output == "movie" && !shot_movie && !shot_folder) c.output = "still";
    c.ai.tau_high = std::clamp(c.ai.tau_high, 128, 256);
    c.ai.tau_low = std::clamp(c.ai.tau_low, -1, 64);
    if (c.ai.tau_low >= c.ai.tau_high) c.ai.tau_low = -1;
    c.ai.steps = std::clamp(c.ai.steps, 1, 100);
    c.ai.seed = std::max(c.ai.seed, 0);
    c.ai.brightness = std::clamp(c.ai.brightness, -3.0, 3.0);
    if (!ai_fill_possible(c, aces_source)) c.ai_fill = false;
    if (aces_source) {
        if (c.gamut != "ap0" && c.gamut != "acescg") c.gamut = c.gamut == "linear" ? "acescg" : "ap0";
        // ACES 2065-1 is the SMPTE ST 2065-4 container: half float RGB (28 Oct 2026 review).
        if (c.gamut == "ap0" && c.image == "exr32") c.image = "exr16";
        return c;
    }
    // The analytic master is a 32-bit EXR in ACES 2065-1 or linear Rec.2020.
    c.image = "exr32";
    if (c.gamut != "ap0" && c.gamut != "linear") c.gamut = c.gamut == "acescg" ? "linear" : "ap0";
    c.keep_exr = false;
    // Its 16-bit movie and a folder's movie are the graded PQ only.
    if (c.output == "movie" && (c.bits == 16 || !shot_movie)) c.encoding = "pq";
    return c;
}

bool ai_fill_possible(const DeliverChoice& c, bool aces_source) {
    return aces_source && c.output != "movie" && (c.image == "exr32" || c.image == "exr16");
}

QJsonObject aces_deliver_fields(const DeliverChoice& c) {
    const QString log = c.encoding == "pq"    ? QStringLiteral("pq%1").arg(c.pq_peak)
                        : c.encoding == "hlg" ? QStringLiteral("hlg1000")
                        : c.encoding == "sdr" ? QStringLiteral("sdr709")
                                              : c.encoding;
    if (c.output == "movie") {
        const QString fmt = c.bits == 16 ? QStringLiteral("mov16") : c.codec;
        return {{"exr", !c.keep_exr ? "none" : c.gamut == "ap0" ? "half" : "float"}, {"exr_gamut", c.gamut}, {"log", log},
                {"log_formats", fmt}};
    }
    if (c.image == "dpx10") return {{"exr", "none"}, {"log", log}, {"log_formats", "dpx10"}};
    return {{"exr", c.image == "exr16" ? "half" : "float"}, {"exr_gamut", c.gamut}};
}

QJsonObject aces_trims_fields(double strength, double saturation, double peak_nits, double exposure_ev, double contrast) {
    if (strength == 1.0 && saturation == 1.0 && peak_nits <= 0 && exposure_ev == 0.0 && contrast == 1.0) return {};
    QJsonObject o{{"strength", strength}, {"saturation", saturation}, {"peak_nits", peak_nits}};
    if (exposure_ev != 0.0) o["exposure_ev"] = exposure_ev;
    if (contrast != 1.0) o["contrast"] = contrast;
    return o;
}

QString analytic_movie_format(const DeliverChoice& c) {
    if (c.bits == 16) return QStringLiteral("rgb16mov");
    const QString base = c.codec == "prores4444" ? QStringLiteral("prores4444") : QStringLiteral("prores422hq");
    return c.encoding == "pq" ? base : base + "_" + c.encoding;
}

namespace {
QString encoding_transfer(const QString& e, bool aces_source, int pq_peak = 1000) {
    if (e == "logc4") return QStringLiteral("ARRI LogC4 (scene-referred)");
    if (e == "acescct") return QStringLiteral("ACEScct (scene-referred)");
    if (e == "hlg") return QStringLiteral("HLG — ACES 2.0 HDR 1000 nits");
    if (e == "sdr") return QStringLiteral("BT.1886 — ACES 2.0 SDR 100 nits (trim pass)");
    return aces_source ? QStringLiteral("PQ — ACES 2.0 HDR %1 nits").arg(pq_peak) : QStringLiteral("PQ — graded to the peak and knee");
}
QString encoding_primaries(const QString& e, bool aces_source) {
    if (e == "logc4") return QStringLiteral("ARRI Wide Gamut 4");
    if (e == "acescct") return QStringLiteral("AP1 (ACEScct)");
    if (e == "sdr") return QStringLiteral("Rec.709");
    // The ACES 2.0 HDR 1000 output is mastered to P3 D65 inside the Rec.2020 container.
    return aces_source ? QStringLiteral("Rec.2020 container, P3 D65 limited") : QStringLiteral("Rec.2020");
}
QString codec_words(const DeliverChoice& c) {
    if (c.bits == 16) return QStringLiteral("RGB 16-bit");
    if (c.codec == "hevc10") return QStringLiteral("HEVC HDR10 (Main 10)");
    if (c.codec == "prores4444xq") return QStringLiteral("ProRes 4444 XQ");
    return c.codec == "prores4444" ? QStringLiteral("ProRes 4444") : QStringLiteral("ProRes 422 HQ");
}
QString movie_container(const DeliverChoice& c) {
    return (c.codec == "hevc10" && c.bits != 16 ? QStringLiteral("MP4 — ") : QStringLiteral("MOV — ")) + codec_words(c);
}
}  // namespace

QString deliver_encoding_words(const DeliverChoice& c) {
    if (c.encoding == "pq") return c.pq_peak != 1000 ? QStringLiteral("PQ %1").arg(c.pq_peak) : QStringLiteral("PQ");
    if (c.encoding == "hlg") return QStringLiteral("HLG");
    if (c.encoding == "sdr") return QStringLiteral("SDR Rec.709");
    return c.encoding == "logc4" ? QStringLiteral("LogC4") : QStringLiteral("ACEScct");
}

DeliverSignal deliver_signal(const DeliverChoice& c, bool aces_source) {
    if (c.output == "movie")
        return {movie_container(c), encoding_transfer(c.encoding, aces_source, c.pq_peak), encoding_primaries(c.encoding, aces_source)};
    if (!aces_source) {   // the analytic master, in the words it always had
        const bool ap0 = c.gamut != "linear";
        return {ap0 ? QStringLiteral("OpenEXR — ACES 2065-1") : QStringLiteral("OpenEXR — linear Rec.2020"),
                QStringLiteral("linear (scene-referred)"), ap0 ? QStringLiteral("AP0 (ST 2065-4)") : QStringLiteral("Rec.2020")};
    }
    if (c.image == "dpx10")
        return {QStringLiteral("DPX — 10-bit RGB"), encoding_transfer(c.encoding, true, c.pq_peak), encoding_primaries(c.encoding, true)};
    const QString depth = c.image == "exr16" ? QStringLiteral("half") : QStringLiteral("float");
    const bool ap0 = c.gamut == "ap0";
    return {QStringLiteral("EXR %1 — %2").arg(depth, ap0 ? QStringLiteral("ACES 2065-1 (ST 2065-4)") : QStringLiteral("ACEScg")),
            QStringLiteral("linear (scene-referred)"),
            ap0 ? QStringLiteral("AP0 · mask in mask/") : QStringLiteral("AP1 (ACEScg) · mask channel")};
}

QString deliver_pipe_label(const DeliverChoice& c, bool aces_source) {
    QString what;
    if (c.output == "movie") {
        what = codec_words(c) + (c.codec == "hevc10" && c.bits != 16 ? QStringLiteral(" MP4") : QStringLiteral(" MOV"));
        what += QStringLiteral(", ") + deliver_encoding_words(c);
    } else if (!aces_source) {
        return {};   // the analytic EXR: the pipe's own words
    } else if (c.image == "dpx10") {
        what = QStringLiteral("DPX 10-bit, ") + deliver_encoding_words(c);
    } else {
        what = (c.gamut == "ap0" ? QStringLiteral("ACES 2065-1 EXR, ") : QStringLiteral("ACEScg EXR, ")) +
               (c.image == "exr16" ? QStringLiteral("half") : QStringLiteral("float32"));
    }
    return aces_source ? QStringLiteral("learned ACES 2.0 → ") + what : what;
}

RenderPath parse_render_path(const QString& raw) {
    RenderPath p;
    QString path = QDir::fromNativeSeparators(raw.trimmed());
    const int slash = int(path.lastIndexOf('/'));
    QString file = slash >= 0 ? path.mid(slash + 1) : path;
    p.dir = slash >= 0 ? path.left(slash) : QString();
    if (p.dir.isEmpty() && slash == 0) p.dir = "/";
    static const QRegularExpression numbered(R"(^(.*?)([._]?)(#+|%0?(\d+)d)\.([A-Za-z0-9]+)$)");
    static const QRegularExpression plain(R"(^(.+)\.([A-Za-z0-9]+)$)");
    if (const auto m = numbered.match(file); m.hasMatch()) {
        p.name = m.captured(1);
        p.separator = m.captured(2);
        const QString hashes = m.captured(3);
        p.padding = hashes.startsWith('#') ? int(hashes.size()) : std::max(1, m.captured(4).toInt());
        p.ext = m.captured(5).toLower();
    } else if (const auto m2 = plain.match(file); m2.hasMatch()) {
        p.name = m2.captured(1);
        p.ext = m2.captured(2).toLower();
    } else {
        p.name = file;
    }
    p.padding = std::clamp(p.padding, 0, 10);
    return p;
}

QString compose_render_path(const RenderPath& p) {
    QString file = p.name;
    if (p.padding > 0) file += p.separator + QString(p.padding, '#');
    if (!p.ext.isEmpty()) file += "." + p.ext;
    if (p.dir.isEmpty()) return file;
    return QDir::toNativeSeparators(p.dir + (p.dir.endsWith('/') ? QString() : QStringLiteral("/")) + file);
}

QString render_ext(const DeliverChoice& c) {
    if (c.output == "movie") return c.codec == "hevc10" && c.bits != 16 ? QStringLiteral("mp4") : QStringLiteral("mov");
    return c.image == "dpx10" ? QStringLiteral("dpx") : QStringLiteral("exr");
}

QString deliver_button_text(const DeliverChoice& c, bool aces_source) {
    if (c.output == "movie") return aces_source ? QStringLiteral("Render MOV") : QStringLiteral("Export MOV");
    if (!aces_source) return QStringLiteral("Master EXR");
    return c.image == "dpx10" ? QStringLiteral("Render DPX") : QStringLiteral("Render EXR");
}

}  // namespace rudra::app
