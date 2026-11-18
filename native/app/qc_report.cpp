#include "qc_report.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocale>
#include <QSaveFile>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "rudra/core/gamut.hpp"
#include "rudra/deliver/exr.hpp"

namespace rudra::app {
namespace {

// The page's style (theme/qc_report.css, embedded): a colour belongs in a style file, not here.
QString qc_css() {
    QFile f(QStringLiteral(":/chrome/qc_report.css"));
    return f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll()) : QString();
}

QString esc(const QString& s) { return s.toHtmlEscaped(); }

// The settings as an artist reads them; a key not listed shows as it is written.
QString setting_label(const QString& k) {
    static const std::pair<const char*, const char*> names[] = {
        {"analytic", "Analytic reconstruction"}, {"anchor", "Anchor to source exposure"}, {"anchor_knee", "Anchor knee"},
        {"carry_chroma", "Carry source chroma"}, {"chroma_knee", "Chroma knee"}, {"container", "Master container"},
        {"deliver_output", "Output"}, {"deliver_format", "Format"}, {"deliver_colour", "Colour"},
        {"deliver_encoding", "Encoding"}, {"pq_peak_nits", "PQ peak (nits)"}, {"exr_half", "Half-float EXR"},
        {"lossy_source", "Compressed source: noise smoothed near white"}, {"preserve_outside", "Preserve outside masks"},
        {"recovery_mode", "Recovery mode"}, {"region_softness_stops", "Region EV softness (stops)"},
        {"settle_grain", "Settle grain"}, {"source_curve", "Source curve"}, {"source_space", "Source primaries"},
        {"strength", "Residual strength"}, {"exposure_ev", "Exposure (EV)"}, {"highlight_detail", "Highlight detail"},
        {"clip_fill", "Fill clipped highlights"}, {"checkpoint", "Model"}, {"regions", "Region EV bands"},
        {"reconstruction_region", "Luminance region"}, {"calibration", "Calibration anchors"}, {"reference", "Reference"}};
    for (const auto& [key, label] : names)
        if (k == QLatin1String(key)) return QString::fromUtf8(label);
    return k;
}

QString num(double v, int decimals = 0) {
    return v < 0 ? QStringLiteral("—") : QLocale(QLocale::English).toString(v, 'f', decimals);
}

// MaxCLL and MaxFALL per frame, a small line chart (inline SVG, log nits).
QString chart(const std::vector<QcFrame>& frames) {
    if (frames.size() < 2) return {};
    const int w = 720, h = 180, pad = 34;
    double hi = 1;
    for (const auto& f : frames) hi = std::max(hi, double(std::max(f.maxcll, 1)));
    const double top = std::log10(std::max(hi * 1.2, 10.0)), bottom = 0.0;   // 1 nit .. above the peak
    auto x = [&](std::size_t i) { return pad + (w - 2 * pad) * double(i) / double(frames.size() - 1); };
    auto y = [&](double nits) { return h - pad + 10 - (h - pad) * (std::log10(std::max(nits, 1.0)) - bottom) / (top - bottom); };
    auto line = [&](auto get, const char* cls) {
        QString d;
        for (std::size_t i = 0; i < frames.size(); ++i)
            d += QStringLiteral("%1%2,%3 ").arg(i ? "L" : "M").arg(x(i), 0, 'f', 1).arg(y(get(frames[i])), 0, 'f', 1);
        return QStringLiteral("<path d=\"%1\" class=\"%2\" fill=\"none\" stroke-width=\"1.6\"/>").arg(d, QString::fromLatin1(cls));
    };
    QString svg = QStringLiteral("<svg viewBox=\"0 0 %1 %2\" width=\"100%\" role=\"img\" aria-label=\"MaxCLL and MaxFALL per frame\">")
                      .arg(w)
                      .arg(h);
    for (double n : {1.0, 10.0, 100.0, 1000.0, 10000.0}) {
        if (std::log10(n) > top) break;
        svg += QStringLiteral("<line x1=\"%1\" x2=\"%2\" y1=\"%3\" y2=\"%3\" class=\"grid\"/><text x=\"4\" y=\"%4\" font-size=\"10\" "
                              "class=\"gl\">%5</text>")
                   .arg(pad)
                   .arg(w - pad)
                   .arg(y(n), 0, 'f', 1)
                   .arg(y(n) + 3, 0, 'f', 1)
                   .arg(QLocale(QLocale::English).toString(n, 'f', 0));
    }
    svg += line([](const QcFrame& f) { return double(f.maxcll); }, "cll");
    svg += line([](const QcFrame& f) { return double(f.maxfall); }, "fall");
    svg += QStringLiteral("<text x=\"%1\" y=\"14\" font-size=\"11\" class=\"cllt\">MaxCLL</text>"
                          "<text x=\"%2\" y=\"14\" font-size=\"11\" class=\"fallt\">MaxFALL</text></svg>")
               .arg(w - 170)
               .arg(w - 100);
    return svg;
}

}  // namespace

std::vector<QcFrame> qc_frames_from_sidecars(const std::vector<std::filesystem::path>& sidecars, QString* container,
                                             QString* transfer) {
    std::vector<QcFrame> out;
    for (const auto& p : sidecars) {
        QFile f(QString::fromStdU16String(p.u16string()));
        if (!f.open(QIODevice::ReadOnly)) continue;
        const QJsonObject j = QJsonDocument::fromJson(f.readAll()).object();
        if (!j.contains("maxcll_nits")) continue;   // not a master's sidecar
        QcFrame q;
        q.file = QFileInfo(f.fileName()).completeBaseName();
        q.maxcll = j.value("maxcll_nits").toInt();
        q.maxfall = j.value("maxfall_nits").toInt();
        q.peak_nits = j.value("peak_nits").toDouble();
        const QJsonArray res = j.value("resolution").toArray();
        if (res.size() == 2) q.resolution = QStringLiteral("%1×%2").arg(res[0].toInt()).arg(res[1].toInt());
        const QJsonObject inv = j.value("invented_pixels").toObject();
        if (inv.contains("invented_pct")) q.invented_pct = inv.value("invented_pct").toDouble();
        const QJsonObject cls = j.value("pixel_classes").toObject();
        if (cls.contains("reconstructed_pct")) q.reconstructed_pct = cls.value("reconstructed_pct").toDouble();
        if (cls.contains("generated_pct")) q.generated_pct = cls.value("generated_pct").toDouble();
        const QJsonObject br = j.value("brackets").toObject().value("preview_vs_export").toObject();
        if (br.contains("max_code_delta")) q.bracket_code_delta = br.value("max_code_delta").toInt();
        if (container && container->isEmpty()) *container = j.value("container").toString();
        if (transfer && transfer->isEmpty()) *transfer = j.value("transfer").toString();
        out.push_back(q);
    }
    return out;
}

QcVerdict qc_verdict(const QcReport& r, int pq_peak_nits) {
    QcVerdict v;
    auto review = [&v](const QString& why) {
        v.pass = false;
        v.reasons << why;
    };
    const bool pq = r.transfer.contains(QStringLiteral("PQ"), Qt::CaseInsensitive);
    int over_peak = 0, over_10k = 0, invented = 0, export_off = 0, worst_export = 0;
    double worst_invented = 0;
    for (const auto& f : r.frames) {
        over_peak += pq && f.maxcll > pq_peak_nits;
        over_10k += f.maxcll > 10000;
        if (f.invented_pct > 5.0) {
            ++invented;
            worst_invented = std::max(worst_invented, f.invented_pct);
        }
        if (f.bracket_code_delta > 2) {
            ++export_off;
            worst_export = std::max(worst_export, f.bracket_code_delta);
        }
    }
    if (over_10k) review(QStringLiteral("%1 frame(s) go past 10,000 nits, the top of PQ: they will clip on any display").arg(over_10k));
    if (over_peak)
        review(QStringLiteral("%1 frame(s) have a MaxCLL over the %2-nit peak this PQ file is mastered for").arg(over_peak).arg(pq_peak_nits));
    if (invented)
        review(QStringLiteral("%1 frame(s) have more than 5 % invented pixels (worst %2 %): check them, they are RUDRA's guess")
                   .arg(invented)
                   .arg(worst_invented, 0, 'f', 1));
    if (export_off)
        review(QStringLiteral("%1 frame(s) differ from what the viewer showed by more than 2 codes (worst %2)").arg(export_off).arg(worst_export));
    // Flicker: MaxFALL (the frame's average light) jumping more than half a stop into a frame and out of it.
    int flicker = 0;
    for (std::size_t i = 1; i + 1 < r.frames.size(); ++i) {
        auto stops = [](int a, int b) { return std::abs(std::log2(std::max(a, 1) / double(std::max(b, 1)))); };
        const auto &a = r.frames[i - 1], &b = r.frames[i], &c = r.frames[i + 1];
        if (stops(b.maxfall, a.maxfall) > 0.5 && stops(b.maxfall, c.maxfall) > 0.5 && stops(a.maxfall, c.maxfall) < 0.25) ++flicker;
    }
    if (flicker) review(QStringLiteral("%1 frame(s) jump in average light against both neighbours (more than half a stop): possible flicker").arg(flicker));
    if (r.remaining_display_errors > 0)
        review(QStringLiteral("%1 display error(s) remain after the learned model (see provenance.json)").arg(r.remaining_display_errors));
    if (!r.appearance_pass) review(QStringLiteral("The learned model's appearance check did not pass (see provenance.json)"));
    if (!r.curve_mismatch.isEmpty())
        review(QStringLiteral("The learned ACES 2.0 route rendered a source whose curve is %1, not ACES: bright areas parked "
                              "just under white can come back far too bright; the analytic route is the one for this grade")
                   .arg(r.curve_mismatch));
    if (r.frames.empty() && r.remaining_display_errors < 0) review(QStringLiteral("No frame's measurements were found beside the files"));
    // 1.2.6 (checklist item 5): pixels outside P3-D65, the gamut HDR10 and Dolby Vision are graded in.
    int out_p3 = 0, out_p3_review = 0;
    double worst_p3 = 0;
    for (const auto& f : r.frames) {
        if (f.out_of_p3_pct <= 0) continue;
        ++out_p3;
        out_p3_review += f.out_of_p3_pct > kQcOutOfP3ReviewPct;
        worst_p3 = std::max(worst_p3, f.out_of_p3_pct);
    }
    if (out_p3_review)
        review(QStringLiteral("%1 frame(s) have more than %2 % of their pixels outside P3-D65 (worst %3 %): a P3-D65 "
                              "master will clip them")
                   .arg(out_p3_review)
                   .arg(kQcOutOfP3ReviewPct, 0, 'f', 0)
                   .arg(worst_p3, 0, 'f', 2));
    else if (out_p3)
        v.notes << QStringLiteral("%1 measured frame(s) have a few pixels outside P3-D65 (worst %2 %).").arg(out_p3).arg(worst_p3, 0, 'f', 2);
    if (r.lossy_source)
        v.notes << QStringLiteral("The source was compressed (JPEG or an 8-bit movie codec): the highlight fill smoothed its "
                                  "compression noise near white.");
    return v;
}

QString qc_html(const QcReport& r, const QcVerdict& v) {
    int maxcll = 0, maxfall = 0;
    double invented = 0;
    bool any_invented = false;
    for (const auto& f : r.frames) {
        maxcll = std::max(maxcll, f.maxcll);
        maxfall = std::max(maxfall, f.maxfall);
        if (f.invented_pct >= 0) {
            invented = std::max(invented, f.invented_pct);
            any_invented = true;
        }
    }
    QString h;
    h += QStringLiteral(
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>QC · %1</title><style>"
        "%2</style></head><body>")
             .arg(esc(r.title), qc_css());
    h += QStringLiteral("<h1>QC report · %1</h1><div class=\"sub\">%2 · RUDRA %3%4</div>")
             .arg(esc(r.title), esc(r.when), esc(r.app_version), r.build.isEmpty() ? QString() : QStringLiteral(" (") + esc(r.build) + ")");
    h += QStringLiteral("<div class=\"verdict %1\"><b>%2</b>").arg(v.pass ? "pass" : "review", v.pass ? "Pass" : "Review");
    if (v.pass) h += QStringLiteral("<div>No check found anything to look at.</div>");
    else {
        h += QStringLiteral("<ul>");
        for (const auto& s : v.reasons) h += QStringLiteral("<li>") + esc(s) + QStringLiteral("</li>");
        h += QStringLiteral("</ul>");
    }
    for (const auto& s : v.notes) h += QStringLiteral("<div>") + esc(s) + QStringLiteral("</div>");
    h += QStringLiteral("</div>");
    h += QStringLiteral("<div class=\"kpi\"><div><span>Frames</span><b>%1</b></div><div><span>MaxCLL</span><b>%2</b> nits</div>"
                        "<div><span>MaxFALL</span><b>%3</b> nits</div><div><span>Invented, worst frame</span><b>%4</b></div></div>")
             .arg(r.frames.empty() ? QStringLiteral("—") : QString::number(r.frames.size()))
             .arg(r.frames.empty() ? QStringLiteral("—") : num(maxcll))
             .arg(r.frames.empty() ? QStringLiteral("—") : num(maxfall))
             .arg(any_invented ? QStringLiteral("%1 %").arg(invented, 0, 'f', 2) : QStringLiteral("—"));
    h += chart(r.frames);
    h += QStringLiteral("<h2>Source and output</h2><table>");
    auto row = [&h](const QString& k, const QString& val) {
        if (!val.isEmpty()) h += QStringLiteral("<tr><th>%1</th><td>%2</td></tr>").arg(esc(k), esc(val));
    };
    row(QStringLiteral("Source"), QDir::toNativeSeparators(r.source));
    row(QStringLiteral("Kind"), r.source_kind + (r.lossy_source ? QStringLiteral(" (compressed, 8-bit)") : QString()));
    row(QStringLiteral("Reconstruction"), r.route);
    row(QStringLiteral("Container"), r.container);
    row(QStringLiteral("Transfer"), r.transfer);
    row(QStringLiteral("Written to"), QDir::toNativeSeparators(r.output_folder));
    if (r.mean_reconstructed_fraction >= 0)
        row(QStringLiteral("Reconstructed (learned model, mean)"), QStringLiteral("%1 %").arg(100.0 * r.mean_reconstructed_fraction, 0, 'f', 1));
    h += QStringLiteral("</table><h2>Settings</h2><table>");
    for (auto it = r.settings.begin(); it != r.settings.end(); ++it) {
        const QJsonValue val = it.value();
        if ((val.isObject() && val.toObject().isEmpty()) || (val.isArray() && val.toArray().isEmpty()) ||
            (val.isString() && val.toString().isEmpty()) || val.isNull())
            continue;   // nothing set
        if (it.key() == QLatin1String("regions") && val.isArray()) {   // the Region EV bands that move something
            QStringList moved;
            for (const auto& b : val.toArray()) {
                const QJsonObject o = b.toObject();
                if (o.value("ev").toDouble() == 0.0) continue;
                moved << QStringLiteral("%1 %2%3 EV (%4\u2013%5 nits)")
                             .arg(o.value("label").toString())
                             .arg(o.value("ev").toDouble() > 0 ? "+" : "")
                             .arg(o.value("ev").toDouble(), 0, 'f', 2)
                             .arg(o.value("low_nits").toDouble())
                             .arg(o.value("high_nits").toDouble());
            }
            if (!moved.isEmpty()) row(setting_label(it.key()), moved.join(QStringLiteral("; ")));
            continue;
        }
        row(setting_label(it.key()), val.isString() ? val.toString() : val.isBool() ? (val.toBool() ? "on" : "off")
                                          : val.isDouble() ? QString::number(val.toDouble(), 'g', 6)
                                          : val.isArray() ? QString::fromUtf8(QJsonDocument(val.toArray()).toJson(QJsonDocument::Compact))
                                                           : QString::fromUtf8(QJsonDocument(val.toObject()).toJson(QJsonDocument::Compact)));
    }
    h += QStringLiteral("</table>");
    if (!r.frames.empty()) {
        h += QStringLiteral("<h2>Frames</h2><table><tr><th>File</th><th>Size</th><th>MaxCLL</th><th>MaxFALL</th><th>Peak</th>"
                            "<th>Invented %</th><th>Reconstructed %</th><th>Outside P3-D65 %</th><th>Export check</th></tr>");
        const std::size_t shown = std::min<std::size_t>(r.frames.size(), 500);
        for (std::size_t i = 0; i < shown; ++i) {
            const auto& f = r.frames[i];
            h += QStringLiteral("<tr><td class=\"mono\">%1</td><td>%2</td><td>%3</td><td>%4</td><td>%5</td><td>%6</td><td>%7</td><td>%8</td><td>%9</td></tr>")
                     .arg(esc(f.file), esc(f.resolution), num(f.maxcll), num(f.maxfall), num(f.peak_nits, 1), num(f.invented_pct, 2),
                          num(f.reconstructed_pct, 2), num(f.out_of_p3_pct, 2),
                          f.bracket_code_delta < 0 ? QStringLiteral("—") : QStringLiteral("%1 code(s)").arg(f.bracket_code_delta));
        }
        h += QStringLiteral("</table>");
        if (shown < r.frames.size()) h += QStringLiteral("<p>… and %1 more (every frame is in its sidecar JSON)</p>").arg(r.frames.size() - shown);
    }
    if (!r.files.isEmpty()) {
        h += QStringLiteral("<h2>Files written</h2><div class=\"mono\">");
        for (const auto& f : r.files) h += esc(f) + QStringLiteral("<br>");
        h += QStringLiteral("</div>");
    }
    h += QStringLiteral("<p class=\"foot\">Measured by RUDRA from the files as written. MaxCLL and "
                        "MaxFALL are scene-referred nits (diffuse white 203). “Invented” pixels are where the SDR was clipped "
                        "or crushed and RUDRA estimated what was there.</p></body></html>");
    return h;
}

QJsonObject qc_json(const QcReport& r, const QcVerdict& v) {
    auto opt = [](double x) { return x < 0 ? QJsonValue(QJsonValue::Null) : QJsonValue(x); };
    QJsonArray frames;
    for (const auto& f : r.frames)
        frames.append(QJsonObject{{"file", f.file},
                                  {"resolution", f.resolution},
                                  {"maxcll_nits", f.maxcll},
                                  {"maxfall_nits", f.maxfall},
                                  {"peak_nits", f.peak_nits},
                                  {"invented_pct", opt(f.invented_pct)},
                                  {"reconstructed_pct", opt(f.reconstructed_pct)},
                                  {"generated_pct", opt(f.generated_pct)},
                                  {"out_of_p3_pct", opt(f.out_of_p3_pct)},
                                  {"export_code_delta", f.bracket_code_delta < 0 ? QJsonValue(QJsonValue::Null) : QJsonValue(f.bracket_code_delta)}});
    int maxcll = 0, maxfall = 0;
    for (const auto& f : r.frames) {
        maxcll = std::max(maxcll, f.maxcll);
        maxfall = std::max(maxfall, f.maxfall);
    }
    return QJsonObject{{"schema", "rudra_qc_v1"},
                       {"title", r.title},
                       {"when", r.when},
                       {"app_version", r.app_version},
                       {"build", r.build},
                       {"verdict", v.pass ? "pass" : "review"},
                       {"reasons", QJsonArray::fromStringList(v.reasons)},
                       {"notes", QJsonArray::fromStringList(v.notes)},
                       {"source", r.source},
                       {"source_kind", r.source_kind},
                       {"lossy_source", r.lossy_source},
                       {"route", r.route},
                       {"curve_mismatch", r.curve_mismatch.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(r.curve_mismatch)},
                       {"container", r.container},
                       {"transfer", r.transfer},
                       {"output_folder", r.output_folder},
                       {"maxcll_nits", maxcll},
                       {"maxfall_nits", maxfall},
                       {"settings", r.settings},
                       {"frames", frames},
                       {"files", QJsonArray::fromStringList(r.files)}};
}

double qc_out_of_p3_pct(const std::filesystem::path& exr, int step) {
    auto img = read_exr(exr);
    if (!img || img->pixels.channels() < 3) return -1;
    // The primaries from the chromaticities' red x (AP0 0.7347, AP1 0.713, Rec.2020 0.708, Rec.709 0.64).
    Primaries src = Primaries::Rec2020;
    for (const auto& [name, bytes] : img->attributes)
        if (name == "chromaticities" && bytes.size() >= 4) {
            float rx = 0;
            std::memcpy(&rx, bytes.data(), 4);
            src = std::abs(rx - 0.7347f) < 2e-3f ? Primaries::Ap0
                  : std::abs(rx - 0.713f) < 2e-3f ? Primaries::Ap1
                  : std::abs(rx - 0.708f) < 2e-3f ? Primaries::Rec2020
                  : std::abs(rx - 0.64f) < 2e-3f  ? Primaries::Rec709
                                                   : Primaries::Rec2020;
        }
    const Mat3 m = rgb_to_rgb_matrix(src, Primaries::P3D65);
    const PlanarBuffer& p = img->pixels;
    step = std::max(1, step);
    std::size_t seen = 0, out = 0;
    for (int y = 0; y < p.height(); y += step)
        for (int x = 0; x < p.width(); x += step) {
            const double c[3] = {p.at(0, y, x), p.at(1, y, x), p.at(2, y, x)};
            if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2])) continue;
            double q[3];
            for (int i = 0; i < 3; ++i) q[i] = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2];
            const double hi = std::max({q[0], q[1], q[2]}), lo = std::min({q[0], q[1], q[2]});
            ++seen;
            out += hi > 1e-6 && lo < -1e-3 * hi;
        }
    return seen ? 100.0 * double(out) / double(seen) : -1;
}

QString write_qc_report(const QcReport& r, int pq_peak_nits) {
    const QcVerdict v = qc_verdict(r, pq_peak_nits);
    const QString base = QDir(r.output_folder).filePath((r.title.isEmpty() ? QStringLiteral("render") : r.title) + QStringLiteral(".qc"));
    QString path = base + QStringLiteral(".html"), json = base + QStringLiteral(".json");
    for (int i = 2; QFileInfo::exists(path) || QFileInfo::exists(json); ++i) {   // nothing is ever replaced
        path = base + QStringLiteral("_%1.html").arg(i);
        json = base + QStringLiteral("_%1.json").arg(i);
    }
    QSaveFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return {};
    f.write(qc_html(r, v).toUtf8());
    if (!f.commit()) return {};
    QSaveFile j(json);   // 1.2.6 (checklist item 5): the JSON twin
    if (j.open(QIODevice::WriteOnly)) {
        j.write(QJsonDocument(qc_json(r, v)).toJson(QJsonDocument::Indented));
        j.commit();
    }
    return path;
}

}  // namespace rudra::app
