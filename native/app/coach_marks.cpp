#include "coach_marks.hpp"

#include <QEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QVBoxLayout>

#include "theme.hpp"

namespace rudra::app {

// A window of its own over the host, not a child widget: the viewer is a native
// swapchain window that a child widget cannot paint over on every platform.
CoachMarks::CoachMarks(QWidget* host) : QWidget(host, Qt::Tool | Qt::FramelessWindowHint), host_(host) {
    setObjectName("coachMarks");
    setAttribute(Qt::WA_TranslucentBackground, true);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setFocusPolicy(Qt::StrongFocus);
    host->installEventFilter(this);

    card_ = new QWidget(this);
    card_->setObjectName("coachCard");
    card_->setAttribute(Qt::WA_StyledBackground, true);
    card_->setFixedWidth(340);
    auto* v = new QVBoxLayout(card_);
    v->setContentsMargins(20, 18, 20, 16);
    v->setSpacing(8);
    count_ = new QLabel(card_);
    count_->setObjectName("coachCount");
    title_ = new QLabel(card_);
    title_->setObjectName("coachTitle");
    title_->setWordWrap(true);
    body_ = new QLabel(card_);
    body_->setObjectName("coachBody");
    body_->setWordWrap(true);
    v->addWidget(count_);
    v->addWidget(title_);
    v->addWidget(body_);
    v->addSpacing(6);
    auto* foot = new QWidget(card_);
    auto* h = new QHBoxLayout(foot);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(10);
    skip_ = new QPushButton(foot);
    skip_->setObjectName("coachSkip");
    skip_->setFlat(true);
    dots_ = new QLabel(foot);
    dots_->setObjectName("coachDots");
    next_ = new QPushButton(foot);
    next_->setObjectName("coachNext");
    next_->setProperty("role", "primary");
    next_->setDefault(true);
    h->addWidget(skip_);
    h->addStretch(1);
    h->addWidget(dots_);
    h->addStretch(1);
    h->addWidget(next_);
    v->addWidget(foot);
    connect(skip_, &QPushButton::clicked, this, [this] {
        if (welcoming_) finish(welcome_skip);
        else skip();
    });
    connect(next_, &QPushButton::clicked, this, [this] {
        if (welcoming_) finish(welcome_go);
        else next();
    });
    hide();
}

void CoachMarks::welcome(const QString& title, const QString& body, const QString& skip, const QString& go) {
    welcoming_ = true;
    steps_.clear();
    index_ = -1;
    count_->setText("Welcome to RUDRA");
    title_->setText(title);
    body_->setText(body);
    skip_->setText(skip);
    next_->setText(go);
    dots_->clear();
    place();
    show();
    raise();
    activateWindow();
    setFocus();
}

void CoachMarks::tips(std::vector<Step> steps) {
    welcoming_ = false;
    steps_ = std::move(steps);
    index_ = 0;
    if (steps_.empty()) return;
    show_step();
    show();
    raise();
    activateWindow();
    setFocus();
}

void CoachMarks::show_step() {
    if (step_entered) step_entered(index_);
    const auto& s = steps_[std::size_t(index_)];
    const int n = count();
    count_->setText(QStringLiteral("Tip %1 of %2").arg(index_ + 1).arg(n));
    title_->setText(s.title);
    body_->setText(s.body);
    skip_->setText("Skip tips");
    skip_->setVisible(index_ + 1 < n);
    next_->setText(index_ + 1 < n ? QStringLiteral("Next") : QStringLiteral("Done"));
    QString d;
    for (int i = 0; i < n; ++i) d += i == index_ ? QChar(0x25cf) : QChar(0x25cb);
    dots_->setText(d);
    place();
}

void CoachMarks::next() {
    if (welcoming_ || index_ < 0) return;
    if (index_ + 1 >= count()) {
        finish(done);
        return;
    }
    ++index_;
    show_step();
}

void CoachMarks::skip() { finish(skipped); }

void CoachMarks::finish(const std::function<void()>& then) {
    hide();
    welcoming_ = false;
    index_ = -1;
    steps_.clear();
    hole_ = {};
    if (then) then();
}

void CoachMarks::place() {
    if (host_) setGeometry(QRect(host_->mapToGlobal(QPoint(0, 0)), host_->size()));
    hole_ = {};
    if (!welcoming_ && index_ >= 0 && index_ < count()) {
        QWidget* t = steps_[std::size_t(index_)].target;
        if (t && host_ && t->isVisible() && t->width() > 0 && host_->isAncestorOf(t)) {
            const QPoint at = t->mapTo(host_, QPoint(0, 0));
            hole_ = QRect(at, t->size()).adjusted(-6, -6, 6, 6).intersected(rect().adjusted(2, 2, -2, -2));
        }
    }
    card_->adjustSize();
    const QSize c = card_->size();
    const int gap = 18;
    QPoint pos((width() - c.width()) / 2, (height() - c.height()) / 2);
    if (!hole_.isEmpty()) {
        const int cy = std::clamp(hole_.center().y() - c.height() / 2, 12, std::max(12, height() - c.height() - 12));
        const int cx = std::clamp(hole_.center().x() - c.width() / 2, 12, std::max(12, width() - c.width() - 12));
        if (hole_.right() + gap + c.width() + 12 <= width()) pos = {hole_.right() + gap, cy};          // beside it, right
        else if (hole_.left() - gap - c.width() >= 12) pos = {hole_.left() - gap - c.width(), cy};     // left
        else if (hole_.bottom() + gap + c.height() + 12 <= height()) pos = {cx, hole_.bottom() + gap}; // below
        else if (hole_.top() - gap - c.height() >= 12) pos = {cx, hole_.top() - gap - c.height()};     // above
        else pos = {cx, std::max(12, height() - c.height() - 24)};   // over it, near the bottom
    }
    card_->move(pos);
    update();
}

void CoachMarks::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    QPainterPath dim;
    dim.addRect(rect());
    if (!hole_.isEmpty()) {
        QPainterPath h;
        h.addRoundedRect(hole_, 10, 10);
        dim = dim.subtracted(h);
    }
    p.fillPath(dim, QColor(0, 0, 0, 170));
    if (!hole_.isEmpty()) {
        QPen ring(theme_colour("gold"), 2.0);
        p.setPen(ring);
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(hole_).adjusted(1, 1, -1, -1), 10, 10);
    }
}

bool CoachMarks::eventFilter(QObject* o, QEvent* e) {
    if (o == host_ && isVisible() &&
        (e->type() == QEvent::Resize || e->type() == QEvent::Move || e->type() == QEvent::LayoutRequest))
        place();
    if (o == host_ && e->type() == QEvent::WindowStateChange && isVisible()) place();
    return QWidget::eventFilter(o, e);
}

void CoachMarks::keyPressEvent(QKeyEvent* e) {
    switch (e->key()) {
        case Qt::Key_Escape:
            if (welcoming_) finish(welcome_skip);
            else skip();
            return;
        case Qt::Key_Return:
        case Qt::Key_Enter:
        case Qt::Key_Right:
            if (welcoming_) finish(welcome_go);
            else next();
            return;
        default: QWidget::keyPressEvent(e);
    }
}

void CoachMarks::mousePressEvent(QMouseEvent* e) { e->accept(); }   // the window under the dim waits

}  // namespace rudra::app
