#pragma once
// The Region EV editor (Phase 3 step 6): the page's #regions. One row a band:
// its swatch, its range ("400 – 2k nits"), its value ("+0.37", green when not
// zero) and "EV". A press on a row selects it; a drag on the value scrubs it
// (0.01 EV a pixel, 0.002 with Shift, into engine/session's undo); a double
// click zeroes it. drawRegions and bindRegions in ui/app.js.

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QWidget>

#include <functional>

#include <vector>

#include "rudra/engine/session.hpp"

namespace rudra::app {

class CheckRow;

class RegionEditor : public QWidget {
public:
    RegionEditor(Session& session, QWidget* parent = nullptr);
    // Redraw from the session (drawRegions).
    void sync();
    // The value label of a row (the page's .ev), for gestures and tests.
    QLabel* value(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].ev : nullptr; }
    QWidget* row(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].row : nullptr; }
    QLabel* range(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].q : nullptr; }
    int rows() const { return int(rows_.size()); }
    // The mask button of a row (3.3): "∅" for none, the coverage when painted,
    // checked while the band is armed. Pressing it tells the app.
    QPushButton* mask_button(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].mask : nullptr; }
    std::function<void(int band)> mask_pressed;

protected:
    bool eventFilter(QObject* o, QEvent* e) override;

private:
    struct Row {
        QWidget* row = nullptr;
        QLabel *q = nullptr, *ev = nullptr;
        QPushButton* mask = nullptr;
    };
    void build(int n);
    Session& session_;
    std::vector<Row> rows_;
    int dragging_ = -1;
};

// The brush card (3.3), shown under the rows while a band is armed: Add /
// Erase, Size, Softness, Flow, Show, Invert, Clear, Done. Reads the session's
// brush and masks; every change goes back through the session.
class MaskPanel : public QWidget {
public:
    MaskPanel(Session& session, QWidget* parent = nullptr);
    void sync();   // from the session: visible while a band is armed
    std::function<void()> done;   // the Done button: the app disarms

private:
    Session& session_;
    QLabel *title_ = nullptr, *coverage_ = nullptr;
    QPushButton *add_ = nullptr, *erase_ = nullptr, *invert_ = nullptr, *clear_ = nullptr, *done_ = nullptr;
    CheckRow* show_ = nullptr;
    QSlider *size_ = nullptr, *soft_ = nullptr, *flow_ = nullptr;
    QLabel *size_v_ = nullptr, *soft_v_ = nullptr, *flow_v_ = nullptr;
    bool syncing_ = false;
};

}  // namespace rudra::app
