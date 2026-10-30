#pragma once
// The start screen (29 Oct 2026), in the FXTD family layout of Radiance Studio's
// splash (RadianceSplash.cpp): the Pro ground, a lockup top-left, RUDRA's mark,
// the two-tone tracked wordmark "RUDRA HDR", "BY FXTD STUDIOS", then the status,
// a 2 px accent rule and the version. Painted (no style sheet).
//
//   Splash s;            s.show();               // at once, before MainWindow is built
//   s.set_status("Loading the learned ACES 2.0 engine");   s.set_progress(0.4);
//   s.finish(&window);   // fades out once the window is up (and the mark had its moment)
//
// The mark eases in once; the rule eases to each step; it fades out on finish.
// Off with the setting ui/splash = false, and for --grab, the checks and tests.
// RUDRA --splash-grab out.png renders it once (the review, the docs).
// No Q_OBJECT (as elsewhere in the app).

#include <QPixmap>
#include <QString>
#include <QWidget>

class QVariantAnimation;

namespace rudra::app {

class Splash : public QWidget {
public:
    explicit Splash(QWidget* parent = nullptr);
    void set_status(const QString& text);
    void set_progress(double fraction);          // 0..1; the hairline eases towards it
    // Fades out and closes once `window` is shown and at least min_ms have passed
    // since the splash appeared (a flash of a logo is worse than none).
    void finish(QWidget* window, int min_ms = 1100);
    // The finished look at once (no animation), e.g. for a grab.
    void settle();
    QString status() const { return status_; }
    double progress() const { return progress_; }

    static constexpr int kWidth = 760, kHeight = 440;

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    QPixmap mark_;
    QString status_;
    QString version_;
    double progress_ = 0.0, shown_progress_ = 0.0;
    double intro_ = 0.0;     // 0 -> 1: the mark and words appear
    double fade_ = 1.0;      // 1 -> 0 on finish
    qint64 shown_at_ = 0;
    QVariantAnimation* intro_anim_ = nullptr;
    QVariantAnimation* bar_anim_ = nullptr;
};

}  // namespace rudra::app
