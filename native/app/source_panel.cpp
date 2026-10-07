#include "source_panel.hpp"

#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>

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

double source_code_nits(SourceCurve source, int code, float corpus_ev) {
    const float display = srgb_to_linear(float(code) / 255.0f);
    const double scene = source_curve_inverse(source, display);
    return scene * std::exp2(-double(corpus_ev)) * 203.0;
}

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
    gold.setAlphaF(0.55);
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
    sync(SourceCurve::Unknown, -1.0f);
}

void SourcePanel::sync(SourceCurve source, float corpus_ev) {
    seg_->set_on(QString::fromUtf8(source_curve_id(source).data(), int(source_curve_id(source).size())));
    plot_->set_source(source, corpus_ev);
    const double g = source_code_nits(source, 118, corpus_ev), w = source_code_nits(source, 235, corpus_ev),
                 c = source_code_nits(source, 255, corpus_ev);
    grey_->setText(nits_text(g));
    white_->setText(nits_text(w));
    clip_->setText(nits_text(c));
    if (source == SourceCurve::Unknown || source == SourceCurve::Aces) delta_->setText("identical");
    else
        delta_->setText(stops_text(g, source_code_nits(SourceCurve::Aces, 118, corpus_ev)) + " · " +
                        stops_text(w, source_code_nits(SourceCurve::Aces, 235, corpus_ev)) + " stops");
}

}  // namespace rudra::app
