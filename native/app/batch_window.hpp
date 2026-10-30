#pragma once
// Deliver > Batch queue… (28 Oct 2026, batch_queue.hpp): the shots waiting to
// render through the learned ACES 2.0 model, each with its own output, name
// and folder; add files or folders (or drop them), reorder, apply the Deliver
// tab's settings, start, stop after a shot or at once, retry.

#include <QDialog>

#include <vector>

class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QTreeWidget;
class QDragEnterEvent;
class QDropEvent;

namespace rudra::app {

class MainWindow;

class BatchWindow : public QDialog {
public:
    explicit BatchWindow(MainWindow* w);
    void refresh();                               // from the queue and the runner
    std::vector<QString> selected_ids() const;
    void add_paths(const QStringList& paths);     // as a drop or Add does

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;

private:
    MainWindow* w_;
    QTreeWidget* list_ = nullptr;
    QLineEdit* dir_ = nullptr;
    QLabel *status_ = nullptr, *sub_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPushButton *start_ = nullptr, *stop_now_ = nullptr;
    bool filling_ = false, was_running_ = false;
};

}  // namespace rudra::app
