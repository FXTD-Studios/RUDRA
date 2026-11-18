#pragma once
// The page's small controls as Qt widgets (Phase 3 step 4). Each carries the
// page's id as its object name and the page's class as its "role", so
// app/theme/studio.qss styles it the way ui/theme.css styles the element.

#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

#include <cstdint>
#include <functional>
#include <vector>

class QButtonGroup;

namespace rudra::app {

// .seg: a row of buttons, one of which may be "on".
class Seg : public QWidget {
public:
    // columns > 0 lays the buttons out as a grid that many wide (the Source
    // picker's six); 0 is the page's single row.
    Seg(const QString& id, const std::vector<std::pair<QString, QString>>& buttons, QWidget* parent = nullptr,
        bool exclusive = true, int columns = 0);
    QPushButton* button(const QString& key) const;   // by its data-* value
    void set_on(const QString& key);                  // the one that is on ("" for none)
    QString on() const;
    std::function<void(const QString&)> clicked;     // with the button's key
    std::vector<QPushButton*> buttons() const { return buttons_; }

private:
    std::vector<QPushButton*> buttons_;
    std::vector<QString> keys_;
};

// .check, as the Pro boards draw it: the label, its hint under it, and a
// switch on the right; clicking anywhere toggles.
class CheckRow : public QWidget {
public:
    CheckRow(const QString& id, const QString& label, const QString& hint_id, const QString& hint,
             QWidget* parent = nullptr);
    void set_on(bool on);
    bool on() const { return on_; }
    void set_hint(const QString& h);
    QLabel* label() const { return label_; }
    QLabel* hint() const { return hint_; }
    std::function<void()> clicked;

protected:
    void mousePressEvent(class QMouseEvent* e) override;
    void paintEvent(class QPaintEvent* e) override;

private:
    QLabel* label_ = nullptr;
    QLabel* hint_ = nullptr;
    bool on_ = false;
};

// A wrapped note whose height is its text's at the width it has (9 Nov 2026). QLabel's own
// heightForWidth can come up a line short in a pinned column, and the note was drawn under
// its neighbours.
class NoteLabel : public QLabel {
public:
    NoteLabel(const QString& text, const QString& id, const QString& role, QWidget* parent = nullptr);
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int w) const override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;
    void setText(const QString& t);   // not virtual in QLabel: call it on a NoteLabel*

protected:
    void resizeEvent(QResizeEvent* e) override;
};

// A one-line label that elides when its row is too narrow (14 Nov 2026): the pipeline strip's
// segments were cut mid-word ("ACES 2.0 SDR 1C", "— clippe"). text() stays whole; the
// full text shows on hover while it is elided.
class ElideLabel : public QLabel {
public:
    ElideLabel(const QString& text, const QString& id, const QString& role,
               Qt::TextElideMode mode = Qt::ElideRight, QWidget* parent = nullptr);
    QSize minimumSizeHint() const override;
    bool elided() const;

protected:
    void paintEvent(QPaintEvent* e) override;
    bool event(QEvent* e) override;

private:
    QRect text_rect() const;
    Qt::TextElideMode mode_;
};

// A path field that elides in the middle while it is not being edited (14 Nov 2026): the
// shots rail showed "ned\Downloads\09V.mp4", the start cut mid-word. text() stays whole.
class PathEdit : public QLineEdit {
public:
    using QLineEdit::QLineEdit;

protected:
    void paintEvent(QPaintEvent* e) override;
};

// .plabel: a panel label with an optional note on the right.
QWidget* panel_label(const QString& text, const QString& note = {}, const QString& note_id = {},
                     QWidget* parent = nullptr);

// The timeline's filmstrip (the Pro boards): thumbnails of the shot across
// its width as frames arrive, the playhead at 0..1, and under it the clipping
// lane, a mark per measured frame (gold where the SDR clipped highlights,
// violet where it crushed shadows). A press or a drag seeks.
class ScrubBar : public QWidget {
public:
    explicit ScrubBar(QWidget* parent = nullptr);
    void set_position(double f);
    double position() const { return pos_; }
    void set_count(int n);                          // frames in the shot; clears thumbnails and marks
    int count() const { return count_; }
    void set_thumb(int frame, const QImage& small);
    enum class Lane : std::uint8_t { None, Highlights, Shadows };
    void set_lane(int frame, Lane l);
    Lane lane(int frame) const;
    std::function<void(double)> seek;
    QSize sizeHint() const override { return {400, 58}; }

protected:
    void paintEvent(class QPaintEvent* e) override;
    void mousePressEvent(class QMouseEvent* e) override;
    void mouseMoveEvent(class QMouseEvent* e) override;

private:
    double pos_ = 0.0;
    int count_ = 0;
    std::vector<QImage> thumbs_;
    std::vector<Lane> lanes_;
};

// The clip bar under the frame stats (paintClipBar): red for what the SDR
// lost, amber after it for where the network acted, as widths of the bar.
class ClipBar : public QWidget {
public:
    explicit ClipBar(QWidget* parent = nullptr);
    void set(double clipped_pct, double mask_pct);
    QWidget* lost() const { return i_; }
    QWidget* acted() const { return u_; }

protected:
    void resizeEvent(class QResizeEvent* e) override;

private:
    void place();
    QWidget *i_ = nullptr, *u_ = nullptr;
    double i_w_ = 0, u_l_ = 0, u_w_ = 0;
};

// The icon rail's buttons and the transport's, drawn as the page's SVGs are.
class IconButton : public QToolButton {
public:
    // 16 Nov 2026 (1.2.5): the window's caption buttons (Menu, Minimise, Maximise, Restore, Close)
    // are drawn here too, not taken from a font (title_bar.cpp), and Check opens Check this computer.
    enum class Glyph { Media, Scopes, Inspector, Help, Prev, Play, Pause, Next, Sidebar, Probe, Export,
                       Menu, Minimise, Maximise, Restore, Close, Check };
    IconButton(const QString& id, Glyph g, const QString& title, QWidget* parent = nullptr);
    void set_glyph(Glyph g);
    Glyph glyph() const { return glyph_; }
    void set_on(bool on);
    bool on() const { return on_; }
    // A small dot at the top right, in the theme's warning colour: something wants a look.
    void set_attention(bool on);
    bool attention() const { return attention_; }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    Glyph glyph_;
    bool on_ = false;
    bool attention_ = false;
};

// 1.2.6 (user feedback: a cleaner UI, no function lost): the settings most shots never touch,
// behind one "Title \u25b8" button in their card. The widgets keep their names and state; they
// move into a box that the button shows and hides. Whether it is open is remembered under
// `settings_key`. Placed where the first of `moved` was in `into`.
QPushButton* add_fold(QVBoxLayout* into, QWidget* parent, const QString& id, const QString& title,
                      const std::vector<QWidget*>& moved, const QString& settings_key, bool at_end = false);

}  // namespace rudra::app
