#pragma once
// The QC report (1.2.6, 18 Nov 2026): one self-contained HTML page beside every render, so a
// studio can see what was delivered without opening RUDRA. Built from what the render wrote
// (each master's sidecar JSON: MaxCLL, MaxFALL, peak, the invented / reconstructed shares,
// the preview-vs-export check) and the settings of the moment. A few plain checks turn the
// numbers into a verdict: Pass, or Review with the reasons.

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <vector>

namespace rudra::app {

struct QcFrame {
    QString file;                  // the master's file name
    int maxcll = 0, maxfall = 0;   // nits
    double peak_nits = 0;
    double invented_pct = -1;      // the SDR's clipped or crushed pixels RUDRA made up; -1 unknown
    double reconstructed_pct = -1; // pixel classes, when written
    double generated_pct = -1;
    int bracket_code_delta = -1;   // preview vs export, worst 8-bit code; -1 not checked
    QString resolution;
    double out_of_p3_pct = -1;     // 1.2.6: pixels outside P3-D65 (qc_out_of_p3_pct); -1 not measured
    double p9998_nits = -1;        // 1.2.7: the sidecar's light_p99_98_nits; -1 not written (before 1.2.7)
};

struct QcReport {
    QString title;                 // the render's name
    QString when;                  // ISO date and time
    QString app_version, build;
    QString source, source_kind;   // the shot's path; still / sequence / movie
    bool lossy_source = false;     // JPEG / 8-bit H.264…: the clip fill smoothed it
    QString route;                 // "Analytic reconstruction" or "Learned ACES 2.0 (<model>)"
    QString output_folder;
    QString container, transfer;
    QJsonObject settings;          // what the grade and the Deliver tab were (key: value)
    std::vector<QcFrame> frames;
    QStringList files;             // every file the render wrote, by name
    // Learned route: what its provenance says.
    int remaining_display_errors = -1;
    bool appearance_pass = true;
    double mean_reconstructed_fraction = -1;
    // 1.2.6 (readiness report, checklist item 4): the learned route rendered a shot whose Source
    // curve is not ACES; that curve's name, "" otherwise. A Review reason.
    QString curve_mismatch;
    // 1.3 (23 Nov 2026): the AI fill made over this render, as its disclosure record says (genfill_disclosure);
    // empty when none. The report was written before the fill and said nothing of it; it is written again,
    // as <title>_aifill.qc.html, once the fill is in.
    QJsonObject ai_fill;
};

struct QcVerdict {
    bool pass = true;
    QStringList reasons;   // why it wants a look ("Review"), each one line
    QStringList notes;     // facts worth knowing that are not problems
};

// Reads masters' sidecars (<name>.json beside each EXR) into frames; unreadable ones are skipped.
std::vector<QcFrame> qc_frames_from_sidecars(const std::vector<std::filesystem::path>& sidecars, QString* container = nullptr,
                                             QString* transfer = nullptr);

// 1.3: with an AI fill, frames it flagged for flicker or dropped are Review reasons; that the render holds
// generated pixels is always said (a note), with the share and the worst frame.
// The checks: MaxCLL over the PQ peak or 10 000 nits, a frame-to-frame jump in MaxFALL of more
// than half a stop (flicker), invented pixels over 5 % of a frame, the preview-vs-export check
// over 2 codes, display errors on the learned route, its appearance check, the learned route on
// a source curve that is not ACES.
QcVerdict qc_verdict(const QcReport& r, int pq_peak_nits = 1000);

QString qc_html(const QcReport& r, const QcVerdict& v);

// 1.2.6 (readiness report, checklist item 5): the same report as JSON, for a QC system to ingest:
// the verdict ("pass" / "review"), reasons, notes, the source, route and output, the settings
// and every frame's values. Written beside the HTML as <title>.qc.json.
QJsonObject qc_json(const QcReport& r, const QcVerdict& v);

// The share of a scene-linear EXR master's pixels outside P3-D65, in per cent (every `step`-th
// pixel each way; the primaries from its chromaticities: AP0, AP1, Rec.2020 or Rec.709); -1 when
// it cannot be read. A pixel is outside when a P3 component is below -0.1 % of its largest one.
double qc_out_of_p3_pct(const std::filesystem::path& exr, int step = 4);

// 1.2.7 (multi-role review of 1.2.6): a frame whose MaxCLL is over this many times the light its
// brightest 0.02 % leave out (and over kQcSpeckMinNits) has its MaxCLL set by a few pixels: a Review
// reason. A TV tone-maps the whole programme against MaxCLL (07V: 4,386 nits from 24 pixels, 182 at p99.99).
inline constexpr double kQcSpeckRatio = 4.0;
inline constexpr int kQcSpeckMinNits = 1000;

// A P3-D65 share over this is a Review reason (an HDR10 or Dolby Vision master is graded in P3-D65).
inline constexpr double kQcOutOfP3ReviewPct = 1.0;

// Writes <folder>/<title>.qc.html (a number added rather than replacing one) and its JSON twin
// <title>.qc.json (same number); the HTML's path, or "".
QString write_qc_report(const QcReport& r, int pq_peak_nits = 1000);

}  // namespace rudra::app
