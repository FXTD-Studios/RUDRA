#include "batch_page.hpp"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QDesktopServices>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMimeData>
#include <QPainter>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QShortcut>
#include <QToolButton>
#include <QThreadPool>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "aces_engine.hpp"
#include "batch_queue.hpp"
#include "main_window.hpp"
#include "media_library.hpp"
#include "paths.hpp"
#include "queue_window.hpp"
#include "theme.hpp"
#include "video_queues.hpp"
#include "widgets.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

constexpr int kId = Qt::UserRole, kKind = Qt::UserRole + 1;   // the row's job id or queue file; "job" or "export"
constexpr int kSrc = Qt::UserRole + 2;                         // the first source file: its thumbnail's key
const QSize kThumb(80, 45);

QLabel* label(const QString& text, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

QPushButton* button(const QString& text, const QString& id, const QString& tip, QWidget* parent) {
    auto* b = new QPushButton(text, parent);
    b->setObjectName(id);
    b->setToolTip(tip);
    return b;
}

// The words a row's status column shows, and its colour token.
std::pair<QString, QString> short_status(const BatchJob& j) {
    if (j.status == "running")
        return {j.total > 0 ? QStringLiteral("Rendering %1 / %2").arg(j.done).arg(j.total) : QStringLiteral("Rendering"), "ink"};
    if (j.status == "filling") return {QStringLiteral("AI fill\u2026"), "ink"};
    if (j.status == "done") return {QStringLiteral("Done"), "ok"};
    if (j.status == "failed") return {QStringLiteral("Failed"), "bad"};
    if (j.status == "stopped") return {QStringLiteral("Stopped"), "warn"};
    return {QStringLiteral("Waiting"), "ink-3"};
}

// The panel's sentence for a job: what happened, in full.
QString long_status(const BatchJob& j) {
    if (j.status == "running")
        return j.total > 0 ? QStringLiteral("Rendering frame %1 of %2.").arg(j.done).arg(j.total) : QStringLiteral("Rendering.");
    if (j.status == "filling") return QStringLiteral("The master is written; the AI fill is running.");
    if (j.status == "done")
        return QStringLiteral("Done: %1 frame(s).").arg(j.done) + (j.note.isEmpty() ? QString() : " " + j.note);
    if (j.status == "failed") return QStringLiteral("Failed: ") + j.error;
    if (j.status == "stopped") return QStringLiteral("Stopped after %1 frame(s).").arg(j.done);
    return j.note.isEmpty() ? QStringLiteral("Waiting.") : QStringLiteral("Waiting (") + j.note + ").";
}

QString kind_text(const BatchJob& j) {
    if (j.shot.kind == "movie") return QStringLiteral("Movie");
    if (j.shot.kind == "sequence") return QStringLiteral("Sequence of %1 frames").arg(j.shot.sources.size());
    return QStringLiteral("Still");
}

QString range_text(const BatchJob& j) {
    return j.first >= 0 ? QStringLiteral("%1-%2").arg(j.first).arg(j.last) : QString();
}

}  // namespace

bool BatchPage::needs_attention(const BatchJob& j, double median_cll, int n_cll) {
    if (j.status == "failed" || j.status == "stopped") return true;
    if (!j.edl.value("warnings").toArray().isEmpty()) return true;
    return j.maxcll > 0 && n_cll >= 3 && j.maxcll >= 2.0 * median_cll;
}

BatchPage::BatchPage(MainWindow* w, QWidget* parent) : QWidget(parent), w_(w) {
    setObjectName("batchPage");
    setAttribute(Qt::WA_StyledBackground, true);
    setAcceptDrops(true);
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);
    auto* body = new QWidget(this);
    auto* bh = new QHBoxLayout(body);
    bh->setContentsMargins(0, 0, 0, 0);
    bh->setSpacing(0);

    // ── the list ──
    auto* main = new QWidget(body);
    main->setObjectName("batchMain");
    auto* v = new QVBoxLayout(main);
    v->setContentsMargins(40, 28, 40, 16);
    v->setSpacing(12);
    auto* head = new QWidget(main);
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(0, 0, 0, 0);
    hh->setSpacing(10);
    auto* titles = new QWidget(head);
    auto* tv = new QVBoxLayout(titles);
    tv->setContentsMargins(0, 0, 0, 0);
    tv->setSpacing(2);
    tv->addWidget(label("Batch", "batchTitle", "batch-title", titles));
    sub_ = label("", "batchSub", "batch-sub", titles);
    tv->addWidget(sub_);
    hh->addWidget(titles);
    hh->addStretch(1);
    filters_ = new Seg("batchFilter", {{"all", "All"}, {"attention", "Needs attention"}, {"done", "Done"}}, head);
    filters_->set_on(filter_);
    filters_->clicked = [this](const QString& k) { set_filter(k); };
    hh->addWidget(filters_, 0, Qt::AlignBottom);
    add_ = new QToolButton(head);
    add_->setObjectName("batchAdd");
    add_->setText("+");
    add_->setToolTip("Add shots");
    add_->setPopupMode(QToolButton::InstantPopup);
    auto* menu = new QMenu(add_);
    auto* a_files = menu->addAction("Add files\u2026");
    a_files->setObjectName("batchAddFiles");
    a_files->setToolTip("Movies, stills or the frames of sequences (numbered names are grouped)");
    auto* a_folder = menu->addAction("Add folder\u2026");
    a_folder->setObjectName("batchAddFolder");
    auto* a_current = menu->addAction("Add the shot on screen\tCtrl+B");
    a_current->setObjectName("batchAddCurrent");
    menu->addSeparator();
    auto* a_edl = menu->addAction("Import EDL\u2026");
    a_edl->setObjectName("batchImportEdl");
    menu->addSeparator();
    auto* a_queue = menu->addAction("Video exports (Queue window)\u2026");
    a_queue->setObjectName("batchQueueWindow");
    add_->setMenu(menu);
    connect(a_files, &QAction::triggered, this, [this] {
        add_paths(QFileDialog::getOpenFileNames(
            this, "Add to the batch", {},
            "Movies and stills (*.mov *.mp4 *.mkv *.mxf *.avi *.m4v *.webm *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.dpx);;All files (*)"));
    });
    connect(a_folder, &QAction::triggered, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Add a folder to the batch");
        if (!d.isEmpty()) add_paths({d});
    });
    connect(a_current, &QAction::triggered, this, [this] {
        QString why;
        if (!w_->batch_add_current(&why, dir_->text().trimmed())) status_->setText(why);
    });
    connect(a_edl, &QAction::triggered, this, [this] {
        const QString file = QFileDialog::getOpenFileName(this, "Import an EDL", {}, "EDL (*.edl);;All files (*)");
        if (file.isEmpty()) return;
        const QString media = QFileDialog::getExistingDirectory(this, "The source movies (searched one folder deep)",
                                                                QFileInfo(file).absolutePath());
        if (media.isEmpty()) return;
        bool ok = false;
        const int handles = QInputDialog::getInt(this, "Handles", "Frames before and after each edit:", 12, 0, 1000, 1, &ok);
        if (!ok) return;
        if (dir_->text().trimmed().isEmpty()) {
            status_->setText("Choose where renders go first.");
            return;
        }
        QString report;
        w_->batch_import_edl(file, media, handles, dir_->text().trimmed(), &report);
        status_->setText(report.section('\n', 0, 0));
        status_->setToolTip(report.contains('\n') ? report : QString());
    });
    connect(a_queue, &QAction::triggered, this, [this] { w_->open_queue_window(); });
    hh->addWidget(add_, 0, Qt::AlignBottom);
    v->addWidget(head);

    auto* where = new QWidget(main);
    auto* wh = new QHBoxLayout(where);
    wh->setContentsMargins(0, 0, 0, 0);
    wh->setSpacing(8);
    wh->addWidget(label("New shots render to", {}, "note", where));
    dir_ = new QLineEdit(where);
    dir_->setObjectName("batchDir");
    dir_->setPlaceholderText("C:\\Renders");
    dir_->setToolTip("Where shots added from now on render: a new folder per shot, named after it");
    wh->addWidget(dir_, 1);
    auto* browse = button("Change\u2026", "batchBrowse", "Choose the render folder", where);
    connect(browse, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Render folder", dir_->text());
        if (!d.isEmpty()) dir_->setText(QDir::toNativeSeparators(d));
    });
    wh->addWidget(browse);
    v->addWidget(where);

    list_ = new QTreeWidget(main);
    list_->setObjectName("batchList");
    list_->setColumnCount(2);
    list_->setHeaderHidden(true);
    list_->setRootIsDecorated(false);
    list_->setIndentation(0);
    list_->setUniformRowHeights(false);
    list_->setIconSize(kThumb);
    list_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    list_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    list_->setFrameShape(QFrame::NoFrame);
    list_->setContextMenuPolicy(Qt::CustomContextMenu);
    list_->header()->setStretchLastSection(false);
    list_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    list_->header()->setSectionResizeMode(1, QHeaderView::Fixed);
    list_->setColumnWidth(1, 190);
    connect(list_, &QTreeWidget::itemSelectionChanged, this, [this] { show_selected(); });
    connect(list_, &QTreeWidget::customContextMenuRequested, this, [this](const QPoint& at) { context_menu(at); });
    // The dialog's Up, Down and Remove as keys (the right-click menu has them too).
    auto* earlier = new QShortcut(QKeySequence("Ctrl+Up"), list_, nullptr, nullptr, Qt::WidgetShortcut);
    connect(earlier, &QShortcut::activated, this, [this] {
        for (const auto& id : selected_ids()) w_->batch_queue().move(id, -1);
    });
    auto* later = new QShortcut(QKeySequence("Ctrl+Down"), list_, nullptr, nullptr, Qt::WidgetShortcut);
    connect(later, &QShortcut::activated, this, [this] {
        auto ids = selected_ids();
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) w_->batch_queue().move(*it, +1);
    });
    auto* del = new QShortcut(QKeySequence(QKeySequence::Delete), list_, nullptr, nullptr, Qt::WidgetShortcut);
    connect(del, &QShortcut::activated, this, [this] { w_->batch_queue().remove(selected_ids()); });
    v->addWidget(list_, 1);
    auto* drop = label("Drop movies, sequences or folders here", "batchDrop", "batch-drop", main);
    drop->setAlignment(Qt::AlignCenter);
    drop->setFixedHeight(52);
    v->addWidget(drop);
    bh->addWidget(main, 1);
    panel_ = build_panel();
    bh->addWidget(panel_);
    outer->addWidget(body, 1);

    // ── the footer: one line, one button ──
    auto* foot = new QWidget(this);
    foot->setObjectName("batchFoot");
    foot->setAttribute(Qt::WA_StyledBackground, true);
    auto* fv = new QVBoxLayout(foot);
    fv->setContentsMargins(0, 0, 0, 0);
    fv->setSpacing(0);
    progress_ = new QProgressBar(foot);
    progress_->setObjectName("batchProgress");
    progress_->setTextVisible(false);
    progress_->setFixedHeight(2);
    fv->addWidget(progress_);
    auto* row = new QWidget(foot);
    auto* fh = new QHBoxLayout(row);
    fh->setContentsMargins(24, 10, 24, 10);
    fh->setSpacing(12);
    status_ = label("", "batchStatus", "batch-status", row);
    fh->addWidget(status_, 1);
    stop_now_ = button("Stop now", "batchStopNow", "Stop the shot in hand after its current frame, and the batch", row);
    connect(stop_now_, &QPushButton::clicked, this, [this] { w_->batch_runner().stop_now(); });
    fh->addWidget(stop_now_);
    start_ = button("Start", "batchStart", "Render every waiting shot, one after another", row);
    start_->setProperty("role", "primary");
    connect(start_, &QPushButton::clicked, this, [this] {
        auto& r = w_->batch_runner();
        if (r.running()) {
            r.stop_after_current();
            return;
        }
        QString why;
        if (!w_->batch_start(&why)) status_->setText(why);
    });
    fh->addWidget(start_);
    fv->addWidget(row);
    outer->addWidget(foot);

    tick_.setInterval(500);
    connect(&tick_, &QTimer::timeout, this, [this] {
        if (isVisible() && !w_->video_queues().entries().empty()) refresh();
    });
    tick_.start();
    refresh();
    if (const QString warn = w_->batch_queue().load_warning(); !warn.isEmpty()) status_->setText(warn);
}

QWidget* BatchPage::build_panel() {
    auto* p = new QWidget(this);
    p->setObjectName("batchPanel");
    p->setAttribute(Qt::WA_StyledBackground, true);
    p->setFixedWidth(340);
    auto* pv = new QVBoxLayout(p);
    pv->setContentsMargins(0, 0, 0, 0);
    auto* scroll = new QScrollArea(p);
    scroll->setObjectName("batchPanelScroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* inner = new QWidget(scroll);
    inner->setObjectName("batchPanelInner");
    auto* v = new QVBoxLayout(inner);
    v->setContentsMargins(24, 24, 24, 24);
    v->setSpacing(12);
    p_thumb_ = new QLabel(inner);
    p_thumb_->setObjectName("batchPanelThumb");
    p_thumb_->setFixedSize(292, 164);
    p_thumb_->setAlignment(Qt::AlignCenter);
    v->addWidget(p_thumb_);
    p_title_ = label("", "batchPanelTitle", "batch-panel-title", inner);
    p_title_->setWordWrap(true);
    v->addWidget(p_title_);
    p_meta_ = label("", "batchPanelMeta", "note", inner);
    p_meta_->setWordWrap(true);
    v->addWidget(p_meta_);
    p_status_ = label("", "batchPanelStatus", "batch-panel-status", inner);
    p_status_->setWordWrap(true);
    v->addWidget(p_status_);
    p_note_ = label("", "batchPanelNote", {}, inner);
    p_note_->setWordWrap(true);
    v->addWidget(p_note_);
    p_action_ = button("", "batchPanelAction", {}, inner);
    v->addWidget(p_action_, 0, Qt::AlignLeft);
    connect(p_action_, &QPushButton::clicked, this, [this] {
        const QString what = p_action_->property("do").toString();
        if (what == "retry") w_->batch_queue().retry({shown_});
        else if (what == "open") {
            if (const BatchJob* j = w_->batch_queue().find(shown_); j && !j->out.isEmpty())
                QDesktopServices::openUrl(QUrl::fromLocalFile(j->out));
        } else if (what == "resume") w_->resume_queue(to_path(shown_));
        else if (what == "stop") w_->video_queues().stop();
    });

    // The settings most shots need: output, AI fill.
    p_settings_ = new QWidget(inner);
    p_settings_->setObjectName("batchSettings");
    p_settings_->setAttribute(Qt::WA_StyledBackground, true);
    auto* sv = new QVBoxLayout(p_settings_);
    sv->setContentsMargins(14, 12, 14, 12);
    sv->setSpacing(10);
    sv->addWidget(label("Output", {}, "note", p_settings_));
    p_output_ = label("", "batchPanelOutput", {}, p_settings_);
    p_output_->setWordWrap(true);
    sv->addWidget(p_output_);
    p_apply_ = button("Use the Deliver tab's settings", "batchApply",
                      "The Deliver tab's output, format, colour, encoding and bit depth and the learned trims, on the "
                      "selected shots (every waiting shot when none is selected)", p_settings_);
    connect(p_apply_, &QPushButton::clicked, this, [this] {
        const auto& t = w_->aces_trims();
        w_->batch_queue().apply(selected_ids(), w_->deliver_choice(), dir_->text().trimmed(), w_->render_start(),
                                BatchTrims{t.strength, t.saturation, t.peak_nits, t.exposure_ev, t.contrast});
    });
    sv->addWidget(p_apply_, 0, Qt::AlignLeft);
    p_ai_ = new QCheckBox("AI fill", p_settings_);
    p_ai_->setObjectName("batchAi");
    p_ai_->setToolTip("Run the AI fill after the selected shots' masters; EXR stills and sequences only, written to <name>_aifill");
    connect(p_ai_, &QCheckBox::toggled, this, [this](bool on) {
        if (filling_) return;
        auto ids = selected_ids();
        if (ids.empty() && !shown_.isEmpty()) ids = {shown_};
        w_->batch_queue().set_ai_fill(ids, on);
    });
    sv->addWidget(p_ai_);
    p_ai_note_ = label("", "batchAiNote", "note", p_settings_);
    p_ai_note_->setWordWrap(true);
    sv->addWidget(p_ai_note_);
    v->addWidget(p_settings_);

    // Details: what the dialog's columns showed, and what a conform needs.
    p_details_ = button("Details", "batchDetails", "Frames, name, folder, timecode, trims, MaxCLL and the EDL event", inner);
    p_details_->setCheckable(true);
    p_details_->setFlat(true);
    v->addWidget(p_details_, 0, Qt::AlignLeft);
    p_details_box_ = new QWidget(inner);
    p_details_box_->setObjectName("batchDetailsBox");
    auto* dv = new QVBoxLayout(p_details_box_);
    dv->setContentsMargins(0, 0, 0, 0);
    dv->setSpacing(6);
    dv->addWidget(label("Frames (first-last, empty for all)", {}, "note", p_details_box_));
    p_range_ = new QLineEdit(p_details_box_);
    p_range_->setObjectName("batchRange");
    p_range_->setPlaceholderText("all");
    connect(p_range_, &QLineEdit::editingFinished, this, [this] {
        if (filling_ || shown_.isEmpty()) return;
        const QString t = p_range_->text().trimmed();
        const QStringList parts = t.split(QRegularExpression("\\s*[-:\u2013]\\s*"));
        bool a = false, b = false;
        const int first = parts.size() == 2 ? parts[0].toInt(&a) : -1, last = parts.size() == 2 ? parts[1].toInt(&b) : -1;
        if (t.isEmpty() || t.compare("all", Qt::CaseInsensitive) == 0) w_->batch_queue().set_range(shown_, -1, -1);
        else if (!(a && b && w_->batch_queue().set_range(shown_, first, last)))
            status_->setText("A range is first-last, for a movie (e.g. 120-480)");
    });
    dv->addWidget(p_range_);
    dv->addWidget(label("Name (its folder and files)", {}, "note", p_details_box_));
    p_name_ = new QLineEdit(p_details_box_);
    p_name_->setObjectName("batchName");
    connect(p_name_, &QLineEdit::editingFinished, this, [this] {
        if (filling_ || shown_.isEmpty() || p_name_->text().trimmed().isEmpty()) return;
        w_->batch_queue().rename(shown_, p_name_->text().trimmed());
    });
    dv->addWidget(p_name_);
    p_info_ = label("", "batchInfo", "note", p_details_box_);
    p_info_->setWordWrap(true);
    p_info_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    dv->addWidget(p_info_);
    p_details_box_->hide();
    connect(p_details_, &QPushButton::toggled, p_details_box_, &QWidget::setVisible);
    v->addWidget(p_details_box_);
    v->addStretch(1);
    scroll->setWidget(inner);
    pv->addWidget(scroll);
    return p;
}

std::vector<QString> BatchPage::selected_ids() const {
    std::vector<QString> ids;
    for (int i = 0; i < list_->topLevelItemCount(); ++i)
        if (auto* it = list_->topLevelItem(i); it->isSelected() && it->data(0, kKind).toString() == "job")
            ids.push_back(it->data(0, kId).toString());
    return ids;
}

void BatchPage::add_paths(const QStringList& paths) {
    if (paths.isEmpty()) return;
    std::vector<fs::path> p;
    for (const auto& s : paths) p.push_back(to_path(s));
    const auto shots = batch_shots_from(p);
    if (shots.empty()) {
        status_->setText("No movies, sequences or stills there.");
        return;
    }
    const QString dir = dir_->text().trimmed();
    if (dir.isEmpty()) {
        status_->setText("Choose where renders go first.");
        dir_->setFocus();
        return;
    }
    const auto& t = w_->aces_trims();
    w_->batch_queue().add(shots, w_->deliver_choice(), dir, w_->render_start(),
                          BatchTrims{t.strength, t.saturation, t.peak_nits, t.exposure_ev, t.contrast});
    status_->setText(QStringLiteral("Added %1 shot(s)").arg(shots.size()));
}

void BatchPage::set_filter(const QString& key) {
    filter_ = key == "attention" || key == "done" ? key : QStringLiteral("all");
    filters_->set_on(filter_);
    refresh();
}

namespace {
// A thumbnail: the image cropped to kThumb's shape on the panel's dark ground (the ground alone for none).
QPixmap thumb_pixmap(const QImage& img) {
    QPixmap px(kThumb * 2);
    px.setDevicePixelRatio(2.0);
    px.fill(theme_colour("scope-bg"));
    if (!img.isNull()) {
        QPainter g(&px);
        const QImage fit = img.scaled(kThumb * 2, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        g.drawImage(QPoint(0, 0), fit, QRect((fit.width() - kThumb.width() * 2) / 2, (fit.height() - kThumb.height() * 2) / 2,
                                             kThumb.width() * 2, kThumb.height() * 2));
    }
    return px;
}
}  // namespace

QPixmap BatchPage::thumb_for(const BatchJob& j) {
    const QString src = j.shot.sources.empty() ? QString() : from_path(j.shot.sources.front());
    if (auto it = thumbs_.constFind(src); it != thumbs_.constEnd()) return it.value();
    if (j.shot.kind == "movie" || src.isEmpty()) {   // a movie seen in Media has one in its cache (small, read here)
        const QPixmap px = thumb_pixmap(j.shot.kind == "movie" && w_->media_panel() ? w_->media_panel()->thumb(src) : QImage());
        thumbs_.insert(src, px);
        return px;
    }
    // 1.3 (24 Nov 2026, audit): a still is read off the UI thread (a long queue of large stills froze the
    // window when the tab opened); the rows show the dark ground until it is back.
    const QPixmap blank = thumb_pixmap({});
    if (!loading_.contains(src)) {
        loading_.insert(src);
        const QPointer<QWidget> self(this);
        QThreadPool::globalInstance()->start([self, src] {
            QImageReader r(src);
            r.setAutoTransform(true);
            const QSize s = r.size();
            if (s.isValid()) r.setScaledSize(s.scaled(kThumb * 2, Qt::KeepAspectRatioByExpanding));
            const QImage img = r.read();
            QMetaObject::invokeMethod(qApp, [self, src, img] {
                if (self) static_cast<BatchPage*>(self.data())->thumb_ready(src, img);
            }, Qt::QueuedConnection);
        });
    }
    return blank;
}

void BatchPage::thumb_ready(const QString& src, const QImage& img) {
    loading_.remove(src);
    const QPixmap px = thumb_pixmap(img);
    thumbs_.insert(src, px);
    for (int i = 0; i < list_->topLevelItemCount(); ++i)
        if (QTreeWidgetItem* it = list_->topLevelItem(i); it->data(0, kSrc).toString() == src) it->setIcon(0, QIcon(px));
    if (const BatchJob* j = w_->batch_queue().find(shown_); j && !j->shot.sources.empty() && from_path(j->shot.sources.front()) == src)
        p_thumb_->setPixmap(px.scaled(p_thumb_->size() * 2, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
}

void BatchPage::refresh() {
    auto& q = w_->batch_queue();
    auto& r = w_->batch_runner();
    auto& vq = w_->video_queues();
    if (dir_->text().isEmpty()) dir_->setText(w_->render_dir());
    std::vector<double> clls;
    for (const auto& j : q.jobs())
        if (j.maxcll > 0) clls.push_back(j.maxcll);
    std::sort(clls.begin(), clls.end());
    const int n_cll = int(clls.size());
    const double median_cll = n_cll ? clls[std::size_t(n_cll / 2)] : 0.0;
    const auto exports = vq.entries();
    const auto current_export = vq.current();

    // The rows: the batch's jobs, then the video exports. Kept in place while the same rows are there.
    QStringList keys;
    for (const auto& j : q.jobs()) keys << j.id;
    for (const auto& e : exports) keys << from_path(e.queue);
    bool same = list_->topLevelItemCount() == keys.size();
    for (int i = 0; same && i < keys.size(); ++i) same = list_->topLevelItem(i)->data(0, kId).toString() == keys[i];
    filling_ = true;
    QStringList keep;
    for (auto* it : list_->selectedItems()) keep << it->data(0, kId).toString();
    if (!same) list_->clear();
    int n = 0, done = 0, waiting = 0, failed = 0, attention = 0, row = 0;
    auto item_at = [&](int i) { return same ? list_->topLevelItem(i) : new QTreeWidgetItem(list_); };
    for (const auto& j : q.jobs()) {
        ++n;
        done += j.status == "done";
        waiting += j.status == "pending";
        failed += j.status == "failed" || j.status == "stopped";
        const bool look = needs_attention(j, median_cll, n_cll);
        attention += look;
        const DeliverChoice eff = batch_effective(j);
        QString line = batch_output_text(j) + (eff.ai_fill ? QStringLiteral(" \u00b7 AI fill") : QString());
        if (const QString rt = range_text(j); !rt.isEmpty()) line += QStringLiteral(" \u00b7 frames ") + rt;
        auto [st, colour] = short_status(j);
        if (look && j.status == "done") {
            st = QStringLiteral("Done \u00b7 look");
            colour = "gold";
        }
        QTreeWidgetItem* it = item_at(row++);
        it->setData(0, kId, j.id);
        it->setData(0, kKind, QStringLiteral("job"));
        it->setText(0, (j.name.isEmpty() ? j.shot.title : j.name) + "\n" + line);
        it->setToolTip(0, j.shot.sources.empty() ? QString() : from_path(j.shot.sources.front()));
        it->setData(0, kSrc, j.shot.sources.empty() ? QString() : from_path(j.shot.sources.front()));
        it->setText(1, st);
        it->setForeground(1, theme_colour(colour));
        it->setTextAlignment(1, Qt::AlignRight | Qt::AlignVCenter);
        if (!same || it->icon(0).isNull()) it->setIcon(0, QIcon(thumb_for(j)));   // a still's arrives later (thumb_ready)
        it->setSizeHint(0, QSize(0, 60));
        const bool show = filter_ == "all" || (filter_ == "done" && j.status == "done") || (filter_ == "attention" && look);
        it->setHidden(!show);
    }
    for (const auto& e : exports) {
        ++n;
        const QueueJobState s = read_queue_state(e.queue);
        const bool is_current = current_export && *current_export == e.queue;
        const bool is_waiting = vq.waiting(e.queue);
        const bool ok = s.status == "complete", bad = s.status == "failed" || s.status == "interrupted";
        done += ok;
        failed += bad;
        attention += bad;
        QTreeWidgetItem* it = item_at(row++);
        it->setData(0, kId, from_path(e.queue));
        it->setData(0, kKind, QStringLiteral("export"));
        it->setText(0, QString::fromStdString(e.title) + QStringLiteral("\nVideo export \u00b7 analytic route"));
        it->setToolTip(0, from_path(e.queue));
        const QString st = is_current && s.frames_total > 0 ? QStringLiteral("Rendering %1 / %2").arg(s.frames_done).arg(s.frames_total)
                           : ok                             ? QStringLiteral("Done")
                           : bad                            ? QStringLiteral("Failed")
                           : is_waiting                     ? QStringLiteral("Waiting")
                                                            : QStringLiteral("Stopped");
        it->setText(1, st);
        it->setForeground(1, theme_colour(ok ? "ok" : bad ? "bad" : is_current ? "ink" : "ink-3"));
        it->setTextAlignment(1, Qt::AlignRight | Qt::AlignVCenter);
        it->setSizeHint(0, QSize(0, 60));
        const bool show = filter_ == "all" || (filter_ == "done" && ok) || (filter_ == "attention" && bad);
        it->setHidden(!show);
    }
    if (!same)
        for (int i = 0; i < list_->topLevelItemCount(); ++i)
            if (keep.contains(list_->topLevelItem(i)->data(0, kId).toString())) list_->topLevelItem(i)->setSelected(true);
    filling_ = false;

    AcesEngine* e = w_->aces_engine();
    const bool ready = e && e->state() == AcesEngine::State::Ready;
    sub_->setText(n == 0 ? QStringLiteral("Nothing in the queue yet")
                         : QStringLiteral("%1 shot(s) \u00b7 %2 waiting \u00b7 %3 done%4")
                               .arg(n)
                               .arg(waiting)
                               .arg(done)
                               .arg(attention ? QStringLiteral(" \u00b7 %1 to look at").arg(attention) : QString()));
    filters_->button("attention")->setText(attention ? QStringLiteral("Needs attention (%1)").arg(attention)
                                                     : QStringLiteral("Needs attention"));
    progress_->setRange(0, std::max(1, n));
    progress_->setValue(done);
    if (r.running()) {
        start_->setText(r.stopping() ? "Stopping\u2026" : "Pause");
        start_->setToolTip("Stop after the shot in hand (Stop now stops it after its current frame)");
        start_->setEnabled(!r.stopping());
        if (const BatchJob* j = q.find(r.current())) status_->setText(j->shot.title + ": " + long_status(*j));
        was_running_ = true;
    } else {
        if (was_running_)
            status_->setText(QStringLiteral("Batch ended: %1 done, %2 waiting%3")
                                 .arg(done)
                                 .arg(waiting)
                                 .arg(failed ? QStringLiteral(", %1 failed or stopped").arg(failed) : QString()));
        else if (status_->text().isEmpty() || status_->text().startsWith("Ready"))
            status_->setText(ready ? QStringLiteral("Ready \u00b7 %1").arg(e->status_text())
                                   : QStringLiteral("Ready when the learned ACES 2.0 model is loaded"));
        was_running_ = false;
        start_->setText("Start");
        start_->setEnabled(ready && waiting > 0);
        start_->setToolTip(ready ? QStringLiteral("Render every waiting shot, one after another")
                                 : QStringLiteral("Load the learned ACES 2.0 model first"));
    }
    stop_now_->setVisible(r.running());
    show_selected();
}

void BatchPage::show_selected() {
    if (filling_) return;
    QTreeWidgetItem* it = list_->currentItem();
    if (!it || !it->isSelected()) {
        const auto sel = list_->selectedItems();
        it = sel.isEmpty() ? nullptr : sel.front();
    }
    filling_ = true;
    shown_ = it ? it->data(0, kId).toString() : QString();
    const QString kind = it ? it->data(0, kKind).toString() : QString();
    p_action_->hide();
    p_note_->clear();
    if (kind == "job") {
        const BatchJob* j = w_->batch_queue().find(shown_);
        if (!j) {
            filling_ = false;
            return;
        }
        p_thumb_->setPixmap(thumb_for(*j).scaled(p_thumb_->size() * 2, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
        p_title_->setText(j->name.isEmpty() ? j->shot.title : j->name);
        p_meta_->setText(j->shot.title + QStringLiteral(" \u00b7 ") + kind_text(*j) + QStringLiteral(" \u00b7 learned ACES 2.0"));
        p_status_->setText(long_status(*j));
        const QJsonArray ew = j->edl.value("warnings").toArray();
        if (!ew.isEmpty()) {
            QStringList ws;
            for (const auto& x : ew) ws << x.toString();
            p_note_->setText(QStringLiteral("From the EDL: ") + ws.join("; "));
        }
        if (j->status == "failed" || j->status == "stopped") {
            p_action_->setText("Try again");
            p_action_->setProperty("do", "retry");
            p_action_->show();
        } else if (j->status == "done" && !j->out.isEmpty()) {
            p_action_->setText("Show in folder");
            p_action_->setProperty("do", "open");
            p_action_->show();
        }
        p_settings_->show();
        p_output_->setText(batch_output_text(*j));
        const DeliverChoice eff = batch_effective(*j);
        p_ai_->setChecked(j->choice.ai_fill);
        p_ai_->setEnabled(j->status != "running" && j->status != "filling");
        p_ai_note_->setText(j->choice.ai_fill && !eff.ai_fill ? QStringLiteral("This output does not take the AI fill (EXR stills and sequences only).")
                            : j->choice.ai_fill               ? QStringLiteral("After the master, on the GPU; written to %1_aifill.").arg(j->name)
                                                              : QStringLiteral("Off"));
        p_details_->show();
        if (!p_range_->hasFocus()) p_range_->setText(range_text(*j));   // a range being typed survives a refresh
        p_range_->setEnabled(j->shot.kind == "movie" && j->status != "running");
        if (!p_name_->hasFocus()) p_name_->setText(j->name);
        p_name_->setEnabled(j->status != "running");
        QStringList info;
        info << QStringLiteral("Folder: ") + QDir::toNativeSeparators(j->render_dir);
        info << QStringLiteral("Start frame: %1").arg(j->start);
        info << QStringLiteral("Timecode: ") + (j->timecode.isEmpty() ? QStringLiteral("the movie's own") : j->timecode) +
                    QStringLiteral(" \u00b7 reel: ") + (j->reel.isEmpty() ? QStringLiteral("the file's name") : j->reel);
        info << QStringLiteral("The SDR was made for: ") + j->source_display;
        info << QStringLiteral("Trims: strength %1 %, highlight saturation %2 %, peak %3, exposure %4 EV, contrast %5 %")
                    .arg(std::lround(j->strength * 100))
                    .arg(std::lround(j->saturation * 100))
                    .arg(j->peak_nits > 0 ? QStringLiteral("%1 nits").arg(std::lround(j->peak_nits)) : QStringLiteral("off"))
                    .arg(j->exposure_ev, 0, 'f', 1)
                    .arg(std::lround(j->contrast * 100));
        if (j->maxcll > 0)
            info << QStringLiteral("MaxCLL / MaxFALL: %1 / %2 nits").arg(std::lround(std::ceil(j->maxcll))).arg(std::lround(std::ceil(j->maxfall)));
        if (!j->edl.isEmpty())
            info << QStringLiteral("EDL %1, event %2, record %3-%4")
                        .arg(j->edl.value("edl").toString())
                        .arg(j->edl.value("event").toInt())
                        .arg(j->edl.value("rec_in").toString(), j->edl.value("rec_out").toString());
        if (!j->out.isEmpty()) info << QStringLiteral("Written to: ") + QDir::toNativeSeparators(j->out);
        p_info_->setText(info.join("\n"));
    } else if (kind == "export") {
        const fs::path queue = to_path(shown_);
        const QueueJobState s = read_queue_state(queue);
        auto& vq = w_->video_queues();
        const bool is_current = vq.current() && *vq.current() == queue;
        const bool is_waiting = vq.waiting(queue);
        p_thumb_->clear();
        p_title_->setText(it->text(0).section('\n', 0, 0));
        p_meta_->setText(QStringLiteral("Video export \u00b7 analytic route \u00b7 ") + QDir::toNativeSeparators(shown_));
        p_status_->setText(QueueWindow::describe(s, is_waiting, is_current));
        if (is_current) {
            p_action_->setText("Stop");
            p_action_->setProperty("do", "stop");
            p_action_->show();
        } else if (!is_waiting && s.status != "complete") {
            p_action_->setText("Resume");
            p_action_->setProperty("do", "resume");
            p_action_->show();
        }
        p_settings_->hide();
        p_details_->hide();
        p_details_box_->hide();
    } else {
        p_thumb_->clear();
        p_title_->setText(list_->topLevelItemCount() ? QStringLiteral("No shot selected") : QStringLiteral("An empty queue"));
        p_meta_->setText(list_->topLevelItemCount() ? QStringLiteral("Pick a shot on the left.")
                                                    : QStringLiteral("Add shots with + or drop them on the list."));
        p_status_->clear();
        p_settings_->hide();
        p_details_->hide();
        p_details_box_->hide();
    }
    if (kind == "job") p_details_box_->setVisible(p_details_->isChecked());
    filling_ = false;
}

void BatchPage::context_menu(const QPoint& at) {
    QTreeWidgetItem* it = list_->itemAt(at);
    if (!it) return;
    QMenu m(this);
    if (it->data(0, kKind).toString() == "export") {
        const fs::path queue = to_path(it->data(0, kId).toString());
        auto& vq = w_->video_queues();
        const bool is_current = vq.current() && *vq.current() == queue;
        if (is_current) connect(m.addAction("Stop"), &QAction::triggered, this, [this] { w_->video_queues().stop(); });
        else connect(m.addAction("Resume"), &QAction::triggered, this, [this, queue] { w_->resume_queue(queue); });
        auto* forget = m.addAction("Remove from the list");
        forget->setEnabled(!is_current);
        connect(forget, &QAction::triggered, this, [this, queue] {
            w_->forget_queue(queue);
            refresh();
        });
        m.exec(list_->viewport()->mapToGlobal(at));
        return;
    }
    auto* earlier = m.addAction("Render earlier\tCtrl+Up");
    auto* later = m.addAction("Render later\tCtrl+Down");
    m.addSeparator();
    auto* ai_on = m.addAction("AI fill on");
    auto* ai_off = m.addAction("AI fill off");
    m.addSeparator();
    auto* retry = m.addAction("Try again");
    auto* remove = m.addAction("Remove\tDelete");
    m.addSeparator();
    auto* clear = m.addAction("Clear the finished shots");
    connect(earlier, &QAction::triggered, this, [this] {
        for (const auto& id : selected_ids()) w_->batch_queue().move(id, -1);
    });
    connect(later, &QAction::triggered, this, [this] {
        auto ids = selected_ids();
        for (auto i = ids.rbegin(); i != ids.rend(); ++i) w_->batch_queue().move(*i, +1);
    });
    connect(ai_on, &QAction::triggered, this, [this] { w_->batch_queue().set_ai_fill(selected_ids(), true); });
    connect(ai_off, &QAction::triggered, this, [this] { w_->batch_queue().set_ai_fill(selected_ids(), false); });
    connect(retry, &QAction::triggered, this, [this] { w_->batch_queue().retry(selected_ids()); });
    connect(remove, &QAction::triggered, this, [this] { w_->batch_queue().remove(selected_ids()); });
    connect(clear, &QAction::triggered, this, [this] { w_->batch_queue().clear_finished(); });
    m.exec(list_->viewport()->mapToGlobal(at));
}

void BatchPage::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void BatchPage::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const QUrl& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    add_paths(paths);
    e->acceptProposedAction();
}

}  // namespace rudra::app
