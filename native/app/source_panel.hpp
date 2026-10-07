#pragma once
// The Source card (roadmap 3.1): the curve the artist says made the SDR, the
// inverse it implies drawn against the ACES inverse, and three readouts a
// colourist can check by eye (code 118, 235 and 255 in nits). Everything
// here is drawn from core/source_curve.hpp; the choice itself is the
// session's (Session::set_source) and comes back through sync().

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QWidget>

#include <functional>

#include "rudra/core/calibration.hpp"
#include "rudra/core/reference_fit.hpp"
#include "rudra/core/source_curve.hpp"

namespace rudra::app {

class Seg;

// The plot: SDR code on x, nits on a log y; the ACES inverse in grey, the
// chosen curve in the accent, dashed marks at code 235 and 255.
class SourceCurvePlot : public QWidget {
public:
    explicit SourceCurvePlot(QWidget* parent = nullptr);
    void set_source(SourceCurve source, float corpus_ev);
    // The anchors (3.2): the calibrated curve is drawn over the source's when
    // they describe a monotone one, with a dot per anchor.
    void set_calibration(std::vector<CalibrationPoint> points);
    // The reference (3.4): its target curve drawn in violet with a dot per
    // code the fit saw; replaces the calibrated curve while set.
    void set_reference(ReferenceFit fit);
    QSize sizeHint() const override { return {296, 150}; }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    SourceCurve source_ = SourceCurve::Unknown;
    float corpus_ev_ = -1.0f;
    std::vector<CalibrationPoint> calibration_;
    ReferenceFit reference_;
};

class SourcePanel : public QWidget {
public:
    explicit SourcePanel(QWidget* parent = nullptr);
    // The session's choice, its anchors by slot, and the loaded model's
    // exposure convention.
    void sync(SourceCurve source, const std::vector<CalibrationPoint>& calibration, float corpus_ev,
              const ReferenceFit& reference = {});
    std::function<void(SourceCurve)> picked;
    // Calibrate (3.2): a slot's pick button was pressed (the app arms the
    // viewer; set_picked() when the click lands), its nits were typed (code -1:
    // keep), or Clear.
    std::function<void(int slot)> arm;
    std::function<void(int slot, int code, double nits)> anchor_changed;
    std::function<void()> cleared;
    // Reference (3.4): Load… was pressed (the app asks for the file and fits
    // it), or its Clear.
    std::function<void()> load_reference;
    std::function<void()> reference_cleared;
    void set_armed(int slot);          // -1 for none; the button shows it
    int armed() const { return armed_; }
    // A click landed for the armed slot: the code it read goes into the row;
    // nits typed before the click are kept.
    void set_picked(int slot, int code);

private:
    struct Row {
        QPushButton* pick = nullptr;
        QLabel* code = nullptr;
        QLineEdit* nits = nullptr;
        QLabel* delta = nullptr;
    };
    void emit_anchor(int slot, int code);
    Seg* seg_ = nullptr;
    SourceCurvePlot* plot_ = nullptr;
    QLabel *grey_ = nullptr, *white_ = nullptr, *clip_ = nullptr, *delta_ = nullptr;
    Row rows_[kMaxCalibrationPoints];
    QLabel* fit_line_ = nullptr;
    QPushButton* clear_ = nullptr;
    QWidget* calibrate_ = nullptr;
    QPushButton *ref_load_ = nullptr, *ref_clear_ = nullptr;
    QLabel *ref_file_ = nullptr, *ref_exposure_ = nullptr, *ref_residual_ = nullptr, *ref_codes_ = nullptr,
           *ref_line_ = nullptr;
    QWidget* ref_stats_ = nullptr;
    int armed_ = -1;
    SourceCurve source_ = SourceCurve::Unknown;
    float corpus_ev_ = -1.0f;
    std::vector<CalibrationPoint> calibration_;
    ReferenceFit reference_;
    bool syncing_ = false;
};


}  // namespace rudra::app
