#include "splash.hpp"

#include <QApplication>
#include <QDateTime>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QScreen>
#include <QTimer>
#include <QVariantAnimation>

#include <algorithm>
#include <cmath>

namespace rudra::app {
namespace {

// The FXTD Pro palette (ui/theme.css; the same tokens Radiance Studio paints with).
const QColor kGround(0x1c, 0x1c, 0x1c);    // --panel: the app ground
const QColor kInk(0xf5, 0xf5, 0xf5);       // --ink
const QColor kInk3(0xa1, 0xa1, 0xa1);      // --ink-3: secondary text
const QColor kInk4(0x8e, 0x8e, 0x8e);      // --ink-4: the version
const QColor kRule(0x3a, 0x3a, 0x3a);      // --line-2: the rule's track
const QColor kAccent(0x0a, 0x84, 0xff);    // --accent

QFont tracked(int px, int weight, double spacing) {
    QFont f;                               // the app font (Geist, theme.cpp)
    f.setPixelSize(px);
    f.setWeight(QFont::Weight(weight));
    f.setLetterSpacing(QFont::AbsoluteSpacing, spacing);
    return f;
}

double ease_out(double t) { return 1.0 - std::pow(1.0 - std::clamp(t, 0.0, 1.0), 3.0); }

}  // namespace

Splash::Splash(QWidget* parent) : QWidget(parent, Qt::SplashScreen | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint) {
    setAttribute(Qt::WA_DeleteOnClose, false);
    setFixedSize(kWidth, kHeight);
    setWindowTitle(QStringLiteral("RUDRA"));
    mark_ = QPixmap(QStringLiteral(":/assets/rudra-mark-large.png"));
    if (mark_.isNull()) mark_ = QPixmap(QStringLiteral(":/assets/rudra-mark.png"));
    version_ = QApplication::applicationVersion();
    if (auto* screen = QGuiApplication::primaryScreen()) {
        const QRect g = screen->availableGeometry();
        move(g.center() - QPoint(kWidth / 2, kHeight / 2));
    }
    shown_at_ = QDateTime::currentMSecsSinceEpoch();
    intro_anim_ = new QVariantAnimation(this);
    intro_anim_->setStartValue(0.0);
    intro_anim_->setEndValue(1.0);
    intro_anim_->setDuration(500);
    QObject::connect(intro_anim_, &QVariantAnimation::valueChanged, this, [this](const QVariant& v) {
        intro_ = v.toDouble();
        update();
    });
    intro_anim_->start();
    bar_anim_ = new QVariantAnimation(this);
    bar_anim_->setDuration(450);
    bar_anim_->setEasingCurve(QEasingCurve::OutCubic);
    QObject::connect(bar_anim_, &QVariantAnimation::valueChanged, this, [this](const QVariant& v) {
        shown_progress_ = v.toDouble();
        update();
    });
}

void Splash::set_status(const QString& text) {
    status_ = text;
    update();
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void Splash::set_progress(double fraction) {
    progress_ = std::clamp(fraction, 0.0, 1.0);
    bar_anim_->stop();
    bar_anim_->setStartValue(shown_progress_);
    bar_anim_->setEndValue(progress_);
    bar_anim_->start();
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void Splash::settle() {
    intro_anim_->stop();
    bar_anim_->stop();
    intro_ = 1.0;
    shown_progress_ = progress_;
    fade_ = 1.0;
    update();
}

void Splash::finish(QWidget* window, int min_ms) {
    set_progress(1.0);
    const qint64 waited = QDateTime::currentMSecsSinceEpoch() - shown_at_;
    const int delay = int(std::max<qint64>(0, min_ms - waited));
    const QPointer<Splash> self(this);
    const QPointer<QWidget> win(window);
    QTimer::singleShot(delay, this, [self, win] {
        if (!self) return;
        if (win) {
            win->raise();
            win->activateWindow();
        }
        auto* fade = new QVariantAnimation(self);
        fade->setStartValue(1.0);
        fade->setEndValue(0.0);
        fade->setDuration(320);
        fade->setEasingCurve(QEasingCurve::InCubic);
        QObject::connect(fade, &QVariantAnimation::valueChanged, self, [self](const QVariant& v) {
            if (!self) return;
            self->fade_ = v.toDouble();
            self->setWindowOpacity(self->fade_);
            self->update();
        });
        QObject::connect(fade, &QVariantAnimation::finished, self, [self] {
            if (self) self->close();
        });
        fade->start(QAbstractAnimation::DeleteWhenStopped);
    });
}

void Splash::paintEvent(QPaintEvent*) {
    // The FXTD family layout (Radiance Studio's splash, RadianceSplash.cpp): the
    // app ground, a small lockup top-left, the mark at 40 % height, the two-tone
    // tracked wordmark, the studio credit, then status, a 2 px rule and the version.
    QPainter p(this);
    p.setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform | QPainter::TextAntialiasing);
    const int w = width(), h = height();
    p.fillRect(rect(), kGround);

    // Top-left lockup: the accent dot and the name.
    p.setPen(Qt::NoPen);
    p.setBrush(kAccent);
    p.drawEllipse(QPointF(28, 30), 4.5, 4.5);
    p.setFont(tracked(11, QFont::Normal, 3.0));
    p.setPen(kInk3);
    p.drawText(QRect(44, 18, 300, 24), Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("RUDRA HDR"));

    // The mark, centred at 40 % of the height (RUDRA's eclipse, at the size of
    // Radiance's ring), easing in once.
    const double cx = w / 2.0, cy = h * 0.40;
    const double in = ease_out(intro_);
    if (!mark_.isNull()) {
        const double mh = 84.0;
        const double mw = mh * mark_.width() / std::max(1, mark_.height());
        p.save();
        p.setOpacity(in);
        p.drawPixmap(QRectF(cx - mw / 2, cy - mh / 2, mw, mh), mark_, QRectF(mark_.rect()));
        p.restore();
    }

    // Wordmark, two-tone: RUDRA in ink, HDR in the accent, measured as one unit.
    const QFont word = tracked(28, QFont::Medium, 7.0);
    p.setFont(word);
    const QFontMetrics fm(word);
    const QString a = QStringLiteral("RUDRA "), b = QStringLiteral("HDR");
    const int aw = fm.horizontalAdvance(a), bw = fm.horizontalAdvance(b);
    const int x0 = int((w - (aw + bw)) / 2.0), wy = int(cy) + 44;
    p.setPen(kInk);
    p.drawText(QRect(x0, wy, aw, 42), Qt::AlignVCenter | Qt::AlignLeft, a);
    p.setPen(kAccent);
    p.drawText(QRect(x0 + aw, wy, bw, 42), Qt::AlignVCenter | Qt::AlignLeft, b);

    // The studio credit.
    p.setFont(tracked(11, QFont::Normal, 4.0));
    p.setPen(kInk3);
    p.drawText(QRect(0, int(cy) + 92, w, 20), Qt::AlignHCenter, QStringLiteral("BY FXTD STUDIOS"));

    // Status, the hairline rule, the version.
    const int bar_w = 200, bar_x = int((w - bar_w) / 2.0), rule_y = int(cy) + 150;
    p.setFont(tracked(10, QFont::Normal, 1.0));
    p.setPen(kInk3);
    p.drawText(QRect(0, rule_y - 26, w, 18), Qt::AlignHCenter, status_);
    p.setPen(Qt::NoPen);
    p.setBrush(kRule);
    p.drawRect(QRectF(bar_x, rule_y, bar_w, 2));
    p.setBrush(kAccent);
    p.drawRect(QRectF(bar_x, rule_y, bar_w * shown_progress_, 2));
    p.setFont(tracked(10, QFont::Normal, 2.0));
    p.setPen(kInk4);
    if (!version_.isEmpty())
        p.drawText(QRect(0, rule_y + 16, w, 18), Qt::AlignHCenter, QStringLiteral("v%1").arg(version_));
}

}  // namespace rudra::app
