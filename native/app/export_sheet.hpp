#pragma once
// The export sheet (the "Pro direction: export sheet" board): the shot at the
// top, the formats as tiles, what the file will carry, what is checked, and
// Export. It drives the Deliver tab's own fields (the container, Output,
// the folder and the name) and Master EXR, so a render from the sheet is the
// same render as one from the tab. EXR (ACES or linear) is written here; for a
// movie, HDR10, HLG and ProRes 422 HQ go to the queue (Phase 4, step 11) and
// the queue window shows them to the end.

#include <QDialog>

#include <map>

class QLabel;

namespace rudra::app {

class Seg;

class MainWindow;

class ExportSheet : public QDialog {
public:
    explicit ExportSheet(MainWindow* w);
    void refresh();              // from the window's state now
    void pick(const QString& id);  // "exr-aces", "exr-linear", or for a movie "hdr10", "hlg", "prores", "prores4444"
    QString picked() const { return picked_; }
    // The ProRes tiles' encoding: "pq" (graded, display-referred), "acescct" or "logc4" (scene-referred).
    void set_encoding(const QString& e);
    QString encoding() const { return encoding_; }
    QString video_format() const;  // the delivery format the queue gets, e.g. "prores4444_logc4"
    void export_now();           // Master EXR, or the movie's export queued; then close

private:
    MainWindow* w_;
    QString picked_ = "exr-aces";
    QString encoding_ = "pq";
    QLabel *title_ = nullptr, *sub_ = nullptr, *signal_ = nullptr, *frames_ = nullptr, *dest_ = nullptr, *hint_ = nullptr;
    QWidget* enc_row_ = nullptr;
    Seg* enc_seg_ = nullptr;   // no Q_OBJECT: held here, not found with findChild
    std::map<QString, QWidget*> tiles_;
};

}  // namespace rudra::app
