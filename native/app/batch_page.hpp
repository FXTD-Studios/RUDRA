#pragma once
// The Batch tab (RUDRA 1.3, 24 Nov 2026; docs/PLAN_1_4_BATCH.md, the approved mockup): the batch queue
// (batch_queue.hpp: shots through the learned ACES 2.0 model, with their AI fill) and the video exports
// (video_queues.hpp: what the Queue window shows) in one list; one panel for the selected shot (its status,
// one action, output, AI fill, and Details: frames, name, folder, timecode, trims, MaxCLL, the EDL event);
// one button. It replaces the Batch queue dialog (batch_window.cpp, no longer built): every control the
// dialog had is here, in the + menu, the right-click menu or Details (AppBatch tests).

#include <QHash>
#include <QImage>
#include <QPixmap>
#include <QSet>
#include <QTimer>
#include <QWidget>

#include <vector>

class QCheckBox;
class QDragEnterEvent;
class QDropEvent;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QToolButton;
class QTreeWidget;
class QTreeWidgetItem;

namespace rudra::app {

class MainWindow;
class Seg;
struct BatchJob;

class BatchPage : public QWidget {
public:
    explicit BatchPage(MainWindow* w, QWidget* parent = nullptr);
    void refresh();                               // from the queues and the runner
    std::vector<QString> selected_ids() const;    // the selected batch jobs (not exports)
    void add_paths(const QStringList& paths);     // as a drop or Add does
    void set_filter(const QString& key);          // "all", "attention" or "done"
    QString filter() const { return filter_; }
    // Whether a job wants a look (a failure, a stop, an EDL warning, a MaxCLL far above the others).
    static bool needs_attention(const BatchJob& j, double median_cll, int n_cll);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;

private:
    QWidget* build_panel();
    void show_selected();
    void context_menu(const QPoint& at);
    QPixmap thumb_for(const BatchJob& j);   // the cached one, or a placeholder while a worker reads it
    void thumb_ready(const QString& src, const QImage& img);   // a worker's thumbnail, back on the UI thread
    MainWindow* w_;
    QTreeWidget* list_ = nullptr;
    Seg* filters_ = nullptr;
    QLabel *sub_ = nullptr, *status_ = nullptr;
    QLineEdit* dir_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPushButton *start_ = nullptr, *stop_now_ = nullptr;
    QToolButton* add_ = nullptr;
    // the panel
    QWidget* panel_ = nullptr;
    QLabel *p_thumb_ = nullptr, *p_title_ = nullptr, *p_meta_ = nullptr, *p_status_ = nullptr, *p_note_ = nullptr,
           *p_output_ = nullptr, *p_info_ = nullptr, *p_ai_note_ = nullptr;
    QPushButton *p_action_ = nullptr, *p_apply_ = nullptr, *p_details_ = nullptr;
    QCheckBox* p_ai_ = nullptr;
    QWidget *p_settings_ = nullptr, *p_details_box_ = nullptr;
    QLineEdit *p_range_ = nullptr, *p_name_ = nullptr;
    QString filter_ = QStringLiteral("all");
    QString shown_;                               // the panel's row: a job id or an export's queue file
    QHash<QString, QPixmap> thumbs_;
    QSet<QString> loading_;                       // stills being read off the UI thread (audit, 24 Nov 2026)
    bool filling_ = false, was_running_ = false;
    QTimer tick_;                                 // the exports' state files, while any is listed
};

}  // namespace rudra::app
