#pragma once
// The Source card (roadmap 3.1): the curve the artist says made the SDR, the
// inverse it implies drawn against the ACES inverse, and three readouts a
// colourist can check by eye (code 118, 235 and 255 in nits). Everything
// here is drawn from core/source_curve.hpp; the choice itself is the
// session's (Session::set_source) and comes back through sync().

#include <QLabel>
#include <QWidget>

#include <functional>

#include "rudra/core/source_curve.hpp"

namespace rudra::app {

class Seg;

// The plot: SDR code on x, nits on a log y; the ACES inverse in grey, the
// chosen curve in the accent, dashed marks at code 235 and 255.
class SourceCurvePlot : public QWidget {
public:
    explicit SourceCurvePlot(QWidget* parent = nullptr);
    void set_source(SourceCurve source, float corpus_ev);
    QSize sizeHint() const override { return {296, 150}; }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    SourceCurve source_ = SourceCurve::Unknown;
    float corpus_ev_ = -1.0f;
};

class SourcePanel : public QWidget {
public:
    explicit SourcePanel(QWidget* parent = nullptr);
    // The session's choice and the loaded model's exposure convention.
    void sync(SourceCurve source, float corpus_ev);
    std::function<void(SourceCurve)> picked;

private:
    Seg* seg_ = nullptr;
    SourceCurvePlot* plot_ = nullptr;
    QLabel *grey_ = nullptr, *white_ = nullptr, *clip_ = nullptr, *delta_ = nullptr;
};

// nits of an SDR code under a source curve at the model's exposure.
double source_code_nits(SourceCurve source, int code, float corpus_ev);

}  // namespace rudra::app
