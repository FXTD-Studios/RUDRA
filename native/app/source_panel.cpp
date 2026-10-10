#include "source_panel.hpp"

#include <QDoubleValidator>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "rudra/core/baseline.hpp"
#include "theme.hpp"
#include "widgets.hpp"

namespace rudra::app {
namespace {

QString nits_text(double n) {
    if (n >= 100.0) return QString::number(std::llround(n)) + " nits";
    if (n >= 10.0) return QString::number(n, 'f', 1) + " nits";
    return QString::number(n, 'f', 2) + " nits";
}

QString stops_text(double a, double b) {
    const double s = std::log2(a / b);
    return (s >= 0 ? "+" : "") + QString::number(s, 'f', 2);
}

QLabel* readout_row(QVBoxLayout* v, const QString& key, QLabel*& value, QWidget* parent) {
    auto* w = new QWidget(parent);
    w->setProperty("role", "prow");
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 3, 0, 3);
    h->setSpacing(6);
    auto* k = new QLabel(key, w);
    k->setProperty("role", "key");
    h->addWidget(k);
    h->addStretch(1);
    value = new QLabel(w);
    value->setProperty("role", "value");
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    h->addWidget(value);
    v->addWidget(w);
    return value;
}

}  // namespace

SourceCurvePlot::SourceCurvePlot(QWidget* parent) : QWidget(parent) {
    setObjectName("sourcePlot");
    setMinimumHeight(140);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void SourceCurvePlot::set_source(SourceCurve source, float corpus_ev) {
    source_ = source;
    corpus_ev_ = corpus_ev;
    update();
}

void SourceCurvePlot::set_calibration(std::vector<CalibrationPoint> points) {
    calibration_ = std::move(points);
    update();
}

void SourceCurvePlot::set_reference(ReferenceFit fit) {
    reference_ = std::move(fit);
    update();
}

void SourceCurvePlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QRectF r = rect();
    p.setPen(QPen(theme_colour("line-2"), 1.0));
    p.setBrush(theme_colour("scope-bg"));
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6.0, 6.0);

    const double L = 46.0, R = 10.0, T = 10.0, B = 24.0;
    const double ymin = std::log2(0.05), ymax = std::log2(4000.0);
    auto X = [&](double code) { return L + (r.width() - L - R) * code / 255.0; };
    auto Y = [&](double nits) {
        const double t = (std::log2(std::max(nits, 0.05)) - ymin) / (ymax - ymin);
        return T + (r.height() - T - B) * (1.0 - std::clamp(t, 0.0, 1.0));
    };
    QFont f = font();
    f.setPointSizeF(std::max(7.0, f.pointSizeF() - 2.0));
    p.setFont(f);
    p.setPen(QPen(theme_colour("scope-line"), 1.0));
    const QColor ink4 = theme_colour("ink-4");
    for (double n : {0.1, 1.0, 10.0, 100.0, 1000.0}) {
        const double y = Y(n);
        p.setPen(QPen(theme_colour("scope-line"), 1.0));
        p.drawLine(QPointF(L, y), QPointF(r.width() - R, y));
        p.setPen(ink4);
        p.drawText(QRectF(0, y - 7, L - 5, 14), Qt::AlignRight | Qt::AlignVCenter,
                   n >= 1.0 ? QString::number(int(n)) : QString::number(n, 'f', 1));
    }
    for (int c : {0, 64, 128, 192, 255}) {
        const double x = X(c);
        p.setPen(QPen(theme_colour("scope-line"), 1.0));
        p.drawLine(QPointF(x, T), QPointF(x, r.height() - B));
        p.setPen(ink4);
        p.drawText(QRectF(x - 16, r.height() - B + 2, 32, 14), Qt::AlignHCenter | Qt::AlignTop, QString::number(c));
    }
    QColor gold = theme_colour("gold");
    gold.setAlphaF(0.55f);
    p.setPen(QPen(gold, 1.0, Qt::DashLine));
    for (int c : {235, 255}) p.drawLine(QPointF(X(c), T), QPointF(X(c), r.height() - B));

    auto curve = [&](SourceCurve s, const QColor& colour, double width) {
        QPainterPath path;
        for (int c = 1; c <= 255; ++c) {
            const QPointF pt(X(c), Y(source_code_nits(s, c, corpus_ev_)));
            c == 1 ? path.moveTo(pt) : path.lineTo(pt);
        }
        p.setPen(QPen(colour, width));
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);
    };
    curve(SourceCurve::Aces, theme_colour("ink-4"), 1.25);
    curve(source_, theme_colour("accent"), 2.0);
    // The reference's target (3.4) over it, a dot per code the fit saw.
    if (!reference_.empty()) {
        const QColor violet = theme_colour("violet");
        QColor dot = violet;
        dot.setAlphaF(0.45f);
        p.setPen(Qt::NoPen);
        p.setBrush(dot);
        for (int c = 0; c < kSourceCurveKnots; ++c)
            if (c < int(reference_.samples_per_code.size()) && reference_.samples_per_code[std::size_t(c)] > 0)
                p.drawEllipse(QPointF(X(c), Y(std::exp2(double(reference_.target_log2_nits[std::size_t(c)])))), 1.6, 1.6);
        QPainterPath path;
        for (int c = 1; c <= 255; ++c) {
            const QPointF pt(X(c), Y(std::exp2(double(reference_.target_log2_nits[std::size_t(c)]))));
            c == 1 ? path.moveTo(pt) : path.lineTo(pt);
        }
        p.setPen(QPen(violet, 2.0));
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);
    }
    // The calibrated curve (3.2) over it, when the anchors describe one.
    else if (!calibration_.empty() && calibration_is_monotone(calibration_, source_, corpus_ev_)) {
        const auto params = calibration_params(calibration_, source_, corpus_ev_);
        if (!params.empty()) {
            QPainterPath path;
            for (int c = 1; c <= 255; ++c) {
                const double n = source_code_nits(source_, c, corpus_ev_) * std::exp2(double(params[1 + c]));
                const QPointF pt(X(c), Y(n));
                c == 1 ? path.moveTo(pt) : path.lineTo(pt);
            }
            const QColor ok = theme_colour("ok");
            p.setPen(QPen(ok, 2.0));
            p.setBrush(Qt::NoBrush);
            p.drawPath(path);
            p.setPen(Qt::NoPen);
            p.setBrush(ok);
            for (const auto& a : calibration_)
                if (a.nits > 0.0) p.drawEllipse(QPointF(X(a.code), Y(a.nits)), 3.5, 3.5);
        }
    }
}

SourcePanel::SourcePanel(QWidget* parent) : QWidget(parent) {
    setObjectName("sourcePanel");
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(12);
    std::vector<std::pair<QString, QString>> buttons;
    for (SourceCurve c : all_source_curves())
        buttons.push_back({QString::fromUtf8(source_curve_id(c).data(), int(source_curve_id(c).size())),
                           QString::fromUtf8(source_curve_label(c).data(), int(source_curve_label(c).size()))});
    seg_ = new Seg("source", buttons, this, true, 3);
    for (SourceCurve c : all_source_curves()) {
        if (auto* b = seg_->button(QString::fromUtf8(source_curve_id(c).data(), int(source_curve_id(c).size()))))
            b->setToolTip(QString::fromUtf8(source_curve_detail(c).data(), int(source_curve_detail(c).size())));
    }
    seg_->clicked = [this](const QString& k) {
        if (auto c = parse_source_curve(k.toStdString()); c && picked) picked(*c);
    };
    v->addWidget(seg_);
    plot_ = new SourceCurvePlot(this);
    v->addWidget(plot_);
    auto* rows = new QWidget(this);
    auto* rv = new QVBoxLayout(rows);
    rv->setContentsMargins(0, 0, 0, 0);
    rv->setSpacing(0);
    readout_row(rv, "Middle grey (code 118)", grey_, rows);
    readout_row(rv, "Diffuse white (code 235)", white_, rows);
    readout_row(rv, "Clip (code 255)", clip_, rows);
    readout_row(rv, "vs ACES at grey \u00b7 white", delta_, rows);
    v->addWidget(rows);

    // Calibrate (3.2): three anchors, each a pick button, the code it read,
    // the nits it should be, and its delta against the picker's curve.
    auto* cal = new QWidget(this);
    calibrate_ = cal;
    cal->setObjectName("calibrate");
    auto* cv = new QVBoxLayout(cal);
    cv->setContentsMargins(0, 8, 0, 0);
    cv->setSpacing(6);
    auto* head = new QWidget(cal);
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(0, 0, 0, 0);
    auto* title = new QLabel("Calibrate", head);
    title->setProperty("role", "card-head");
    hh->addWidget(title);
    hh->addStretch(1);
    auto* hint = new QLabel("click the frame, type the nits it should be", head);
    hint->setProperty("role", "note");
    hh->addWidget(hint);
    cv->addWidget(head);
    auto* grid = new QGridLayout;
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(6);
    grid->setVerticalSpacing(4);
    static const char* kLabels[kMaxCalibrationPoints] = {"Black", "18% grey", "Highlight"};
    for (int i = 0; i < kMaxCalibrationPoints; ++i) {
        Row& r = rows_[i];
        r.pick = new QPushButton(QString::number(i + 1), cal);
        r.pick->setObjectName(QString("calPick%1").arg(i));
        r.pick->setProperty("role", "pick");
        r.pick->setCheckable(true);
        r.pick->setToolTip("Pick on the frame");
        r.pick->setFixedSize(22, 22);
        connect(r.pick, &QPushButton::clicked, this, [this, i] {
            if (arm) arm(armed_ == i ? -1 : i);
        });
        auto* lab = new QLabel(kLabels[i], cal);
        lab->setProperty("role", "key");
        r.code = new QLabel("\u2014", cal);
        r.code->setObjectName(QString("calCode%1").arg(i));
        r.code->setProperty("role", "value");
        r.code->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        r.nits = new QLineEdit(cal);
        r.nits->setObjectName(QString("calNits%1").arg(i));
        r.nits->setPlaceholderText("nits");
        r.nits->setAlignment(Qt::AlignRight);
        auto* valid = new QDoubleValidator(0.0, 100000.0, 3, r.nits);
        valid->setNotation(QDoubleValidator::StandardNotation);
        r.nits->setValidator(valid);
        connect(r.nits, &QLineEdit::editingFinished, this, [this, i] {
            if (!syncing_) emit_anchor(i, -1);
        });
        r.delta = new QLabel(cal);
        r.delta->setObjectName(QString("calDelta%1").arg(i));
        r.delta->setProperty("role", "note");
        r.delta->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        grid->addWidget(r.pick, i, 0);
        grid->addWidget(lab, i, 1);
        grid->addWidget(r.code, i, 2);
        grid->addWidget(r.nits, i, 3);
        grid->addWidget(r.delta, i, 4);
    }
    grid->setColumnStretch(3, 1);
    cv->addLayout(grid);
    fit_line_ = new QLabel(cal);
    fit_line_->setObjectName("calFit");
    fit_line_->setProperty("role", "note");
    fit_line_->setWordWrap(true);
    cv->addWidget(fit_line_);
    auto* btns = new QWidget(cal);
    auto* bh = new QHBoxLayout(btns);
    bh->setContentsMargins(0, 0, 0, 0);
    bh->addStretch(1);
    clear_ = new QPushButton("Clear", btns);
    clear_->setObjectName("calClear");
    connect(clear_, &QPushButton::clicked, this, [this] {
        if (cleared) cleared();
    });
    bh->addWidget(clear_);
    cv->addWidget(btns);
    v->addWidget(cal);

    // Reference (3.4): the graded HDR of this frame, loaded and fitted; the
    // file, the exposure it asks for, the residual, the codes it covered.
    auto* ref = new QWidget(this);
    ref->setObjectName("reference");
    auto* rv2 = new QVBoxLayout(ref);
    rv2->setContentsMargins(0, 8, 0, 0);
    rv2->setSpacing(6);
    auto* rhead = new QWidget(ref);
    auto* rhh = new QHBoxLayout(rhead);
    rhh->setContentsMargins(0, 0, 0, 0);
    auto* rtitle = new QLabel("Reference", rhead);
    rtitle->setProperty("role", "card-head");
    rhh->addWidget(rtitle);
    rhh->addStretch(1);
    auto* rhint = new QLabel("the graded HDR of this frame", rhead);
    rhint->setProperty("role", "note");
    rhh->addWidget(rhint);
    rv2->addWidget(rhead);
    auto* rrow = new QWidget(ref);
    auto* rrh = new QHBoxLayout(rrow);
    rrh->setContentsMargins(0, 0, 0, 0);
    rrh->setSpacing(8);
    ref_load_ = new QPushButton("Load…", rrow);
    ref_load_->setObjectName("refLoad");
    connect(ref_load_, &QPushButton::clicked, this, [this] {
        if (load_reference) load_reference();
    });
    rrh->addWidget(ref_load_);
    ref_file_ = new QLabel(rrow);
    ref_file_->setObjectName("refFile");
    ref_file_->setProperty("role", "file");
    ref_file_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    rrh->addWidget(ref_file_, 1);
    rv2->addWidget(rrow);
    ref_stats_ = new QWidget(ref);
    auto* sg = new QGridLayout(ref_stats_);
    sg->setContentsMargins(0, 0, 0, 0);
    sg->setHorizontalSpacing(6);
    sg->setVerticalSpacing(1);
    auto stat = [&](int col, const QString& key, QLabel*& value, const char* name) {
        auto* k = new QLabel(key, ref_stats_);
        k->setProperty("role", "note");
        value = new QLabel(ref_stats_);
        value->setObjectName(name);
        value->setProperty("role", "value");
        sg->addWidget(k, 0, col);
        sg->addWidget(value, 1, col);
    };
    stat(0, "Exposure", ref_exposure_, "refExposure");
    stat(1, "Residual µ · p95", ref_residual_, "refResidual");
    stat(2, "Codes seen", ref_codes_, "refCodes");
    rv2->addWidget(ref_stats_);
    ref_line_ = new QLabel(ref);
    ref_line_->setObjectName("refFit");
    ref_line_->setProperty("role", "note");
    ref_line_->setWordWrap(true);
    rv2->addWidget(ref_line_);
    auto* rbtns = new QWidget(ref);
    auto* rbh = new QHBoxLayout(rbtns);
    rbh->setContentsMargins(0, 0, 0, 0);
    rbh->addStretch(1);
    ref_clear_ = new QPushButton("Clear", rbtns);
    ref_clear_->setObjectName("refClear");
    connect(ref_clear_, &QPushButton::clicked, this, [this] {
        if (reference_cleared) reference_cleared();
    });
    rbh->addWidget(ref_clear_);
    rv2->addWidget(rbtns);
    v->addWidget(ref);
    sync(SourceCurve::Unknown, {}, -1.0f);
}

void SourcePanel::emit_anchor(int slot, int code) {
    if (!anchor_changed) return;
    bool ok = false;
    const double nits = rows_[slot].nits->text().toDouble(&ok);
    anchor_changed(slot, code, ok ? nits : 0.0);
}

void SourcePanel::set_armed(int slot) {
    armed_ = slot;
    for (int i = 0; i < kMaxCalibrationPoints; ++i) rows_[i].pick->setChecked(i == slot);
}

void SourcePanel::set_picked(int slot, int code) {
    if (slot < 0 || slot >= kMaxCalibrationPoints) return;
    rows_[slot].code->setText(QString::number(code));
    emit_anchor(slot, code);
}

void SourcePanel::sync(SourceCurve source, const std::vector<CalibrationPoint>& calibration, float corpus_ev,
                       const ReferenceFit& reference) {
    source_ = source;
    corpus_ev_ = corpus_ev;
    calibration_ = calibration;
    reference_ = reference;
    plot_->set_reference(reference);
    const bool has_ref = !reference.empty();
    calibrate_->setEnabled(!has_ref);
    ref_stats_->setVisible(has_ref);
    ref_clear_->setEnabled(has_ref);
    if (has_ref) {
        ref_file_->setText(QString::fromStdString(reference.file));
        ref_file_->setProperty("state", "");
        const double ev = reference_exposure(reference, source, corpus_ev);
        ref_exposure_->setText((ev >= 0 ? "+" : "") + QString::number(ev, 'f', 2) + " st");
        ref_residual_->setText(QString::number(reference.residual_mean, 'f', 2) + " · " +
                               QString::number(reference.residual_p95, 'f', 2) + " st");
        ref_residual_->setProperty("state", reference.residual_p95 > 0.5 ? "warn" : "");
        ref_codes_->setText(QString::number(reference.codes_seen) + " / 256");
        ref_line_->setText(QString::fromStdString(reference_summary(reference, source, corpus_ev)));
    } else {
        ref_file_->setText("EXR, PQ PNG or TIFF, same frame");
        ref_file_->setProperty("state", "empty");
        ref_residual_->setProperty("state", "");
        ref_line_->setText("No reference. Anchors above drive the fit.");
    }
    for (QLabel* l : {ref_file_, ref_residual_}) {
        l->style()->unpolish(l);
        l->style()->polish(l);
    }
    seg_->set_on(QString::fromUtf8(source_curve_id(source).data(), int(source_curve_id(source).size())));
    plot_->set_source(source, corpus_ev);
    std::vector<CalibrationPoint> usable;
    for (const auto& c : calibration)
        if (c.nits > 0.0) usable.push_back(c);
    plot_->set_calibration(usable);
    syncing_ = true;
    for (int i = 0; i < kMaxCalibrationPoints; ++i) {
        Row& r = rows_[i];
        const CalibrationPoint c = i < int(calibration.size()) ? calibration[std::size_t(i)] : CalibrationPoint{};
        const bool has_code = i < int(calibration.size()) && c.code >= 0;
        r.code->setText(has_code ? QString::number(c.code) : QStringLiteral("\u2014"));
        if (!r.nits->hasFocus()) r.nits->setText(c.nits > 0.0 ? QString::number(c.nits, 'g', 6) : QString());
        if (c.nits > 0.0 && has_code) {
            const double st = std::log2(c.nits / std::max(source_code_nits(source, c.code, corpus_ev), 1e-9));
            r.delta->setText((st >= 0 ? "+" : "") + QString::number(st, 'f', 2) + " st");
        } else r.delta->clear();
        r.pick->setChecked(i == armed_);
    }
    syncing_ = false;
    const std::string summary = calibration_summary(usable, source, corpus_ev);
    fit_line_->setText(has_ref ? QStringLiteral("Replaced by the reference.")
                       : summary.empty() ? QStringLiteral("No anchors. The curve above is the picker's.")
                                         : QString::fromStdString(summary));
    fit_line_->setProperty("state", summary.rfind("These anchors", 0) == 0 ? "bad" : "");
    fit_line_->style()->unpolish(fit_line_);
    fit_line_->style()->polish(fit_line_);
    clear_->setEnabled(!usable.empty() || !calibration.empty());
    // The readouts show the effective curve: the picker's, with the anchors
    // or the reference over it.
    std::vector<float> corr;
    if (has_ref) corr = reference_params(reference, source, corpus_ev);
    else if (!usable.empty() && calibration_is_monotone(usable, source, corpus_ev))
        corr = calibration_params(usable, source, corpus_ev);
    auto eff = [&](int code) {
        const double n = source_code_nits(source, code, corpus_ev);
        return corr.empty() ? n : n * std::exp2(double(corr[1 + std::size_t(code)]));
    };
    const double g = eff(118), w = eff(235), c = eff(255);
    grey_->setText(nits_text(g));
    white_->setText(nits_text(w));
    clip_->setText(nits_text(c));
    if ((source == SourceCurve::Unknown || source == SourceCurve::Aces) && corr.empty()) delta_->setText("identical");
    else
        delta_->setText(stops_text(g, source_code_nits(SourceCurve::Aces, 118, corpus_ev)) + " · " +
                        stops_text(w, source_code_nits(SourceCurve::Aces, 235, corpus_ev)) + " stops");
}

}  // namespace rudra::app
