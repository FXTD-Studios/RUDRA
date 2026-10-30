#pragma once
// Reviewing an AI fill before it reaches a client (28 Oct 2026 review), and
// where its model lives.
//
// The fill (rudra/genfill/run.py) writes, beside its EXRs, review/<frame>.a.png
// (the master), .b.png (the fill) and .mask.png (rudra.generated), 3 stops down
// so the invented highlights show, and genfill.json lists every frame's
// generated share and the flicker the fill adds (stops of frame-to-frame change
// beyond the master's own). The review window pages through the frames with A
// (master), B (fill) and the mask, and flags the frames that jump.
// No Q_OBJECT (as elsewhere in the app): connections are lambdas.

#include <QDialog>
#include <QString>

#include <optional>
#include <vector>

class QLabel;
class QTreeWidget;

namespace rudra::app {

struct GenFillReview {
    struct Frame {
        QString name;
        double generated = 0;              // share of pixels the fill invented
        std::optional<double> temporal;    // stops of flicker the fill adds (none for the first frame)
        bool flagged = false;
        QString a, b, mask;                // the review pictures ("" when missing)
    };
    QString dir;
    std::vector<Frame> frames;
    double flag_stops = 0.25;
    double review_stops = -3;
    QString model;
    int flagged() const;
};

// genfill.json of a fill folder and its review pictures; nullopt and `why` when it is not one.
std::optional<GenFillReview> read_genfill_review(const QString& fill_dir, QString* why = nullptr);

class GenFillReviewDialog : public QDialog {
public:
    GenFillReviewDialog(GenFillReview review, QWidget* parent = nullptr);
    // "a", "b" or "mask"; the frame shown.
    void show_frame(int index, const QString& which);
    QString showing() const { return which_; }
    int current() const { return index_; }

protected:
    void keyPressEvent(class QKeyEvent* e) override;
    void resizeEvent(class QResizeEvent* e) override;

private:
    void refresh();
    GenFillReview r_;
    QTreeWidget* list_ = nullptr;
    QLabel* image_ = nullptr;
    QLabel* caption_ = nullptr;
    int index_ = 0;
    QString which_ = "b";
};

// Where the AI fill's parts are: the Python, the repository, Wan2.1-VACE and the LoRA.
// Paths saved in QSettings (genfill/python, repo, base, lora); Check says what is missing.
class GenFillSetupDialog : public QDialog {
public:
    explicit GenFillSetupDialog(QWidget* parent = nullptr);
};

}  // namespace rudra::app
