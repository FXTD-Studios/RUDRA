#pragma once
// The first-run tips (29 Oct 2026, "Simple" redesign): a welcome card, then
// coach marks. The window dims, the control a tip is about stays lit with a
// ring round it, and a card beside it says what it is in a sentence: Tip n of
// 4, Skip tips, Next / Done. Esc skips, Enter or Right goes on.
// No Q_OBJECT (as elsewhere in the app): the callbacks are std::function.

#include <QPointer>
#include <QString>
#include <QWidget>

#include <functional>
#include <vector>

class QLabel;
class QPushButton;

namespace rudra::app {

class CoachMarks : public QWidget {
public:
    struct Step {
        QPointer<QWidget> target;   // the control the tip is about (none or hidden: the card in the middle)
        QString title, body;
    };
    // Covers `host` and follows its size.
    explicit CoachMarks(QWidget* host);

    // The welcome card, in the middle, over the dimmed window.
    void welcome(const QString& title, const QString& body, const QString& skip, const QString& go);
    // The tips, one after another.
    void tips(std::vector<Step> steps);

    bool welcoming() const { return welcoming_; }
    int step() const { return index_; }   // the tip shown, from 0
    int count() const { return int(steps_.size()); }
    QRect hole() const { return hole_; }  // the lit control, in the host's coordinates (empty: none)
    QWidget* card() const { return card_; }

    void next();   // the next tip, or done after the last
    void skip();   // no more tips

    std::function<void()> welcome_go, welcome_skip;   // the welcome card's two buttons
    std::function<void()> skipped, done;              // the tips: Skip tips, or Done on the last
    std::function<void(int)> step_entered;            // before a tip is placed (to bring its tab forward)

protected:
    void paintEvent(class QPaintEvent* e) override;
    bool eventFilter(QObject* o, class QEvent* e) override;
    void keyPressEvent(class QKeyEvent* e) override;
    void mousePressEvent(class QMouseEvent* e) override;

private:
    void show_step();
    void place();
    void finish(const std::function<void()>& then);

    QPointer<QWidget> host_;
    std::vector<Step> steps_;
    int index_ = -1;
    bool welcoming_ = false;
    QRect hole_;
    QWidget* card_ = nullptr;
    QLabel *count_ = nullptr, *title_ = nullptr, *body_ = nullptr, *dots_ = nullptr;
    QPushButton *skip_ = nullptr, *next_ = nullptr;
};

}  // namespace rudra::app
