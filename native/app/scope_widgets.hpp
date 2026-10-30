#pragma once
// The scope widgets (Phase 3 step 5): the waveform and the histogram paint
// the SVG core/scope_draw builds (the page's drawScopes, element for element),
// scaled as the browser scales an <svg> with its viewBox and xMidYMid meet;
// the vectorscope shows core/scopes' 256 x 256 picture 176 px across in the
// page's ring.

#include <QImage>
#include <QWidget>

#include <optional>

#include "rudra/core/scope_draw.hpp"

namespace rudra::app {

class ScopePlot : public QWidget {
public:
    enum class Kind { Waveform, Histogram };
    ScopePlot(const QString& id, Kind kind, int height, QWidget* parent = nullptr);
    void set_data(const ScopeData& s, std::optional<double> maxcll);
    void clear();
    const SvgDrawing& drawing() const { return drawing_; }
    bool has_data() const { return !drawing_.elements.empty(); }
    void set_text_shown(bool on) { text_ = on; update(); }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    Kind kind_;
    SvgDrawing drawing_;
    bool text_ = true;
};

class VectorscopeView : public QWidget {
public:
    explicit VectorscopeView(QWidget* parent = nullptr);
    // core/scopes vectorscope(): 256 x 256 RGBA8, straight alpha.
    void set_image(const std::vector<std::uint8_t>& rgba);
    void clear();
    QSize sizeHint() const override { return {256, 190}; }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    QImage image_;
};

// The colorist's QC scope of the learned frame (28 Oct 2026 review): an RGB parade
// of the Rec.2020 components on the PQ scale (0-100 % code, nits marked at 100,
// 203, 1000, 4000), and how much of the frame lies outside P3-D65 and Rec.2020.
struct QcScope {
    QImage parade;                 // R | G | B panels, density, transparent background
    double out_p3 = 0, out_2020 = 0;   // fraction of pixels outside each gamut
    double peak_nits = 0;          // brightest component
    double avg_nits = 0;           // mean luminance (the frame average light, as MaxFALL counts it)
    double above_white = 0;        // fraction of pixels brighter than diffuse white (203 nits)
    bool empty() const { return parade.isNull(); }
};
// `rgb709_linear`: planar Rec.709 linear, 1.0 = 10 000 nits (the viewer's convention).
QcScope qc_scope(const PlanarBuffer& rgb709_linear, int width = 300, int height = 120);

class ParadeView : public QWidget {
public:
    explicit ParadeView(QWidget* parent = nullptr);
    void set_scope(QcScope s);
    void clear();
    const QcScope& scope() const { return s_; }
    QSize sizeHint() const override { return {300, 132}; }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    QcScope s_;
};

// Paints one drawing into `target` as an <svg> of that box would show it;
// without its <text> elements when `text` is false (the tests).
void paint_svg(class QPainter& p, const SvgDrawing& d, const QRectF& target, bool text = true);

}  // namespace rudra::app
