#include "scope_widgets.hpp"

#include <QFontDatabase>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyleOption>

#include <algorithm>
#include <cmath>
#include <sstream>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "theme.hpp"

namespace rudra::app {
namespace {

double number(const SvgElement& e, const char* name, double fallback = 0.0) {
    const std::string* v = e.attr(name);
    return v ? std::strtod(v->c_str(), nullptr) : fallback;
}

std::vector<QPointF> points(const std::string& s) {
    std::vector<QPointF> out;
    std::istringstream in(s);
    std::string pair;
    while (in >> pair) {
        const auto comma = pair.find(',');
        if (comma == std::string::npos) continue;
        out.emplace_back(std::strtod(pair.c_str(), nullptr), std::strtod(pair.c_str() + comma + 1, nullptr));
    }
    return out;
}

QColor colour(const std::string* s, double opacity) {
    if (!s) return {};
    const auto c = parse_colour(*s);
    if (!c) return {};
    QColor q(c->r, c->g, c->b);
    q.setAlphaF(float(std::clamp(opacity, 0.0, 1.0)));
    return q;
}

// One element, with the opacity of the groups around it and the fill a group
// hands down (SVG's inheritance, for the attributes the page uses).
void paint(QPainter& p, const SvgElement& e, double group_opacity, const std::string* inherited_fill, bool text) {
    const double opacity = number(e, "opacity", 1.0) * group_opacity;
    const std::string* fill = e.attr("fill") ? e.attr("fill") : inherited_fill;
    if (e.tag == "g") {
        for (const auto& c : e.children) paint(p, c, opacity, fill, text);
        return;
    }
    QPen pen(Qt::NoPen);
    if (const std::string* stroke = e.attr("stroke"); stroke && *stroke != "none") {
        const double w = number(e, "stroke-width", 1.0);
        // A stroke under one device pixel is drawn as the browser draws it
        // (Skia's thin-stroke rule): one pixel wide, its alpha scaled by the
        // width it should have had.
        const double scale = std::sqrt(std::abs(p.transform().determinant()));
        const double device = w * scale;
        QColor c = colour(stroke, opacity);
        double pen_w = w;
        if (device < 1.0 && device > 0.0) {
            c.setAlphaF(float(c.alphaF() * device));
            pen_w = 1.0 / scale;
        }
        pen = QPen(c, pen_w);
        pen.setCapStyle(Qt::FlatCap);
        if (const std::string* dash = e.attr("stroke-dasharray")) {
            // SVG dashes are in user units; Qt's are in pen widths.
            QList<qreal> pattern;
            std::istringstream in(*dash);
            double d;
            while (in >> d) pattern << d / std::max(pen_w, 1e-6);
            if (pattern.size() % 2) pattern << pattern;
            pen.setDashPattern(pattern);
        }
    }
    const QBrush brush = fill && *fill != "none" ? QBrush(colour(fill, opacity)) : QBrush(Qt::NoBrush);
    if (e.tag == "line") {
        p.setPen(pen);
        p.drawLine(QPointF(number(e, "x1"), number(e, "y1")), QPointF(number(e, "x2"), number(e, "y2")));
    } else if (e.tag == "polyline" || e.tag == "polygon") {
        const auto pts = points(*e.attr("points"));
        QPolygonF poly(QList<QPointF>(pts.begin(), pts.end()));
        p.setPen(pen);
        p.setBrush(brush);
        if (e.tag == "polygon") p.drawPolygon(poly, Qt::OddEvenFill);
        else p.drawPolyline(poly);
    } else if (e.tag == "rect") {
        p.setPen(pen);
        p.setBrush(brush);
        p.drawRect(QRectF(number(e, "x"), number(e, "y"), number(e, "width"), number(e, "height")));
    } else if (e.tag == "text" && text) {
        // SVG's default white space: runs of spaces collapse to one.
        const QString t = QString::fromStdString(e.text).simplified();
        // At the size it lands on the screen, fractional and unhinted, as
        // the browser sets small text (a 1 px font scaled up has the wrong
        // advances).
        const double scale = std::sqrt(std::abs(p.transform().determinant()));
        const QPointF at = p.transform().map(QPointF(number(e, "x"), number(e, "y")));
        QFont f("IBM Plex Mono");
        f.setHintingPreference(QFont::PreferNoHinting);
        f.setPointSizeF(number(e, "font-size", 8.0) * scale * 72.0 / 96.0);   // points at 96 dpi
        p.save();
        p.resetTransform();
        p.setFont(f);
        p.setPen(colour(fill, opacity));
        const QFontMetricsF fm(f, p.device());
        double x = at.x();
        if (const std::string* a = e.attr("text-anchor"); a && *a == "middle") x -= fm.horizontalAdvance(t) / 2.0;
        p.drawText(QPointF(x, at.y()), t);
        p.restore();
    }
}

}  // namespace

void paint_svg(QPainter& p, const SvgDrawing& d, const QRectF& target, bool text) {
    std::istringstream in(d.view_box);
    double vx = 0, vy = 0, vw = 1, vh = 1;
    in >> vx >> vy >> vw >> vh;
    // preserveAspectRatio xMidYMid meet: the whole box, centred.
    const double s = std::min(target.width() / vw, target.height() / vh);
    const double ox = target.x() + (target.width() - vw * s) / 2.0 - vx * s;
    const double oy = target.y() + (target.height() - vh * s) / 2.0 - vy * s;
    p.save();
    p.setClipRect(target);
    p.translate(ox, oy);
    p.scale(s, s);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);
    for (const auto& e : d.elements) paint(p, e, 1.0, nullptr, text);
    p.restore();
}

ScopePlot::ScopePlot(const QString& id, Kind kind, int height, QWidget* parent) : QWidget(parent), kind_(kind) {
    setObjectName(id);
    setProperty("role", "plot-area");
    setAttribute(Qt::WA_StyledBackground, true);
    setFixedHeight(height);
}

void ScopePlot::set_data(const ScopeData& s, std::optional<double> maxcll) {
    drawing_ = kind_ == Kind::Waveform ? waveform_svg(s, maxcll) : histogram_svg(s);
    update();
}

void ScopePlot::clear() {
    drawing_ = {};
    update();
}

void ScopePlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    QStyleOption opt;
    opt.initFrom(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);   // the sheet's background and border
    if (drawing_.elements.empty()) return;
    paint_svg(p, drawing_, QRectF(rect()).adjusted(1, 1, -1, -1), text_);
}

VectorscopeView::VectorscopeView(QWidget* parent) : QWidget(parent) {
    setObjectName("vector");
    setProperty("role", "plot");   // it is .plot.vs: the plot's own background
    setAttribute(Qt::WA_StyledBackground, true);
    // .plot.vs: 8 px above the ring and 6 below, the labels placed in this box.
    setFixedHeight(8 + vector_frame().display + 6);
}

void VectorscopeView::set_image(const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() != std::size_t(kVectorSize) * kVectorSize * 4) return;
    image_ = QImage(rgba.data(), kVectorSize, kVectorSize, QImage::Format_RGBA8888).copy();
    update();
}

void VectorscopeView::clear() {
    image_ = {};
    update();
}

void VectorscopeView::paintEvent(QPaintEvent*) {
    const auto& f = vector_frame();
    QPainter p(this);
    QStyleOption opt;
    opt.initFrom(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    const double d = f.display;
    // Centred and snapped to whole pixels, as the browser lays the canvas out
    // (the page's 47.5 px lands on 48).
    const QRectF box(std::floor((width() - d) / 2.0 + 0.5), 8.0, d, d);
    auto rgb = [](const Rgb8& c) { return QColor(c.r, c.g, c.b); };
    // The canvas: a disc of the background, the picture over it.
    QPainterPath disc;
    disc.addEllipse(box);
    p.fillPath(disc, rgb(f.background));
    if (!image_.isNull()) {
        p.save();
        p.setClipPath(disc);
        p.drawImage(box, image_);
        p.restore();
    }
    // The rings: 1 px borders, the inner one inset 27 % of the inside.
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(rgb(f.ring), 1));
    p.drawEllipse(box.adjusted(0.5, 0.5, -0.5, -0.5));
    const double inset = f.inner_inset * (d - 2);
    p.setPen(QPen(rgb(f.inner_ring), 1));
    p.drawEllipse(box.adjusted(1 + inset + 0.5, 1 + inset + 0.5, -1 - inset - 0.5, -1 - inset - 0.5));
    // The hue labels, centred on their places.
    QFont font("IBM Plex Mono");
    font.setPixelSize(int(std::lround(f.label_px)));
    p.setFont(font);
    p.setPen(rgb(f.label));
    const QFontMetricsF fm(font);
    for (const auto& l : f.labels) {
        const QString t = QString::fromLatin1(l.text);
        const QPointF c(l.left * width(), l.top * height());   // left and top of .plot.vs
        p.drawText(QPointF(c.x() - fm.horizontalAdvance(t) / 2.0, c.y() + (fm.ascent() - fm.descent()) / 2.0), t);
    }
}

QcScope qc_scope(const PlanarBuffer& rgb, int width, int height) {
    QcScope q;
    const int W = rgb.width(), H = rgb.height();
    if (W <= 0 || H <= 0 || rgb.channels() < 3 || width < 3 || height < 8) return q;
    const Mat3 to2020 = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::Rec2020);
    const Mat3 toP3 = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::P3D65);
    const int panel = width / 3;
    std::vector<std::uint32_t> count(std::size_t(panel) * 3 * std::size_t(height), 0);
    // A sample of about 300 000 pixels: every step-th row and column.
    const int step = std::max(1, int(std::sqrt(double(W) * H / 300000.0)));
    std::size_t n = 0, out_p3 = 0, out_2020 = 0, above = 0;
    double sum_y = 0;
    const float* p[3] = {rgb.plane(0), rgb.plane(1), rgb.plane(2)};
    for (int y = 0; y < H; y += step)
        for (int x = 0; x < W; x += step) {
            const std::size_t i = std::size_t(y) * std::size_t(W) + std::size_t(x);
            const double v[3] = {p[0][i], p[1][i], p[2][i]};
            if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) continue;
            ++n;
            const double y_nits = std::max(0.0, 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]) * 10000.0;
            sum_y += y_nits;
            if (y_nits > 203.0) ++above;
            double c2020[3], cp3[3];
            for (int r = 0; r < 3; ++r) {
                c2020[r] = to2020[std::size_t(r)][0] * v[0] + to2020[std::size_t(r)][1] * v[1] + to2020[std::size_t(r)][2] * v[2];
                cp3[r] = toP3[std::size_t(r)][0] * v[0] + toP3[std::size_t(r)][1] * v[1] + toP3[std::size_t(r)][2] * v[2];
            }
            // Outside a gamut: a component below zero by more than a hair (1e-4 of the brightest one).
            const double tol = 1e-4 * std::max({c2020[0], c2020[1], c2020[2], 1e-6});
            if (std::min({cp3[0], cp3[1], cp3[2]}) < -tol) ++out_p3;
            if (std::min({c2020[0], c2020[1], c2020[2]}) < -tol) ++out_2020;
            const int col = int(std::lround(double(x) / std::max(1, W - 1) * (panel - 1)));
            for (int c = 0; c < 3; ++c) {
                const double nits = std::max(0.0, c2020[c]) * 10000.0;
                q.peak_nits = std::max(q.peak_nits, nits);
                const double code = pq_oetf(float(std::min(nits, 10000.0)));
                const int row = std::clamp(int(std::lround((1.0 - code) * (height - 1))), 0, height - 1);
                ++count[(std::size_t(row) * 3 + std::size_t(c)) * std::size_t(panel) + std::size_t(col)];
            }
        }
    if (n == 0) return q;
    q.out_p3 = double(out_p3) / double(n);
    q.out_2020 = double(out_2020) / double(n);
    q.avg_nits = sum_y / double(n);
    q.above_white = double(above) / double(n);
    QImage img(panel * 3, height, QImage::Format_ARGB32);
    img.fill(Qt::transparent);
    // Density: a log of the count against the busiest cell, so faint traces still show.
    std::uint32_t top = 1;
    for (auto c : count) top = std::max(top, c);
    const double lt = std::log1p(double(top));
    const QColor tint[3] = {QColor::fromRgbF(1.0f, 0.35f, 0.3f), QColor::fromRgbF(0.35f, 1.0f, 0.4f),
                            QColor::fromRgbF(0.4f, 0.55f, 1.0f)};
    for (int row = 0; row < height; ++row)
        for (int c = 0; c < 3; ++c)
            for (int col = 0; col < panel; ++col) {
                const std::uint32_t k = count[(std::size_t(row) * 3 + std::size_t(c)) * std::size_t(panel) + std::size_t(col)];
                if (!k) continue;
                QColor t = tint[c];
                t.setAlphaF(float(std::clamp(0.25 + 0.75 * std::log1p(double(k)) / lt, 0.0, 1.0)));
                img.setPixelColor(c * panel + col, row, t);
            }
    q.parade = img;
    return q;
}

ParadeView::ParadeView(QWidget* parent) : QWidget(parent) {
    setObjectName("parade");
    setMinimumHeight(132);
    setToolTip("RGB parade of the learned frame: the Rec.2020 components on the PQ scale (0-100 % code), "
               "marked at 100, 203 (diffuse white), 1000 and 4000 nits; below, the share outside P3-D65 and Rec.2020");
}

void ParadeView::set_scope(QcScope s) {
    s_ = std::move(s);
    update();
}

void ParadeView::clear() {
    s_ = {};
    update();
}

void ParadeView::paintEvent(QPaintEvent*) {
    QPainter p(this);
    const QRectF r = QRectF(rect()).adjusted(28, 2, -2, -16);
    p.fillRect(r, theme_colour("scope-bg"));
    p.setPen(theme_colour("scope-line"));
    QFont f = font();
    f.setPointSizeF(7.0);
    p.setFont(f);
    for (double nits : {100.0, 203.0, 1000.0, 4000.0}) {
        const double y = r.top() + (1.0 - pq_oetf(float(nits))) * r.height();
        p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y));
        p.drawText(QRectF(0, y - 6, 26, 12), Qt::AlignRight | Qt::AlignVCenter, nits >= 1000 ? QString("%1k").arg(nits / 1000) : QString::number(nits));
    }
    if (!s_.empty()) {
        p.setRenderHint(QPainter::SmoothPixmapTransform, true);
        p.drawImage(r, s_.parade);
        const QString gam = QStringLiteral("outside P3-D65 %1 % · Rec.2020 %2 % · peak %3 nits")
                                .arg(100.0 * s_.out_p3, 0, 'f', 2)
                                .arg(100.0 * s_.out_2020, 0, 'f', 2)
                                .arg(std::lround(s_.peak_nits));
        p.setPen(s_.out_p3 > 0.001 ? theme_colour("gold") : theme_colour("ink-4"));
        p.drawText(QRectF(r.left(), r.bottom() + 2, r.width(), 14), Qt::AlignLeft | Qt::AlignVCenter, gam);
    } else {
        p.setPen(theme_colour("ink-4"));
        p.drawText(r, Qt::AlignCenter, "learned frames only");
    }
}

}  // namespace rudra::app
