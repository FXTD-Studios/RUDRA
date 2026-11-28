#include "genfill_review.hpp"
#include "child_process.hpp"

#include <QClipboard>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPalette>
#include <QPixmap>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSysInfo>
#include <QTextBlock>
#include <QTextCursor>
#include <QTextDocument>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include "genfill_job.hpp"
#include "main_window.hpp"
#include "paths.hpp"
#include "theme.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>

namespace rudra::app {

namespace {

// 1.3.2 (end-user report, 27 Nov 2026, "Wan2.1-VACE does not set up"): what one "Set up automatically" run
// printed, in the dialog and in setup.log (genfill_setup_log), which the artist can send. The dialog used to keep
// only the last line of each piece the installers wrote, nothing went to a file, and a run that stopped before it
// could say why (a Python error, Windows refusing the script) left the step before it on the status line.
struct SetupRun {
    QFile file;
    QPointer<QPlainTextEdit> view;
    bool progress = false;   // the view's last line is a progress bar, redrawn in place
    QString progress_text;   // that bar's latest state: written to the file once it is left behind
    QStringList tail;        // the installers' last lines (stderr): the reason of a run that stopped
    QByteArray out, err;     // what is not a whole line yet
    bool done = false, ok = false;
    bool blocked = false;    // Windows refused to run the PowerShell script (its execution policy)

    void open(const QString& command) {
        const QString path = from_path(genfill_setup_log());
        QDir().mkpath(QFileInfo(path).absolutePath());
        const QString before = QFileInfo(path).dir().filePath(QStringLiteral("setup.previous.log"));
        QFile::remove(before);
        QFile::rename(path, before);
        file.setFileName(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) return;
        write(QStringLiteral("RUDRA %1, AI fill setup, %2, %3")
                  .arg(QCoreApplication::applicationVersion(), QDateTime::currentDateTime().toString(Qt::ISODate),
                       QSysInfo::prettyProductName()));
        write(QStringLiteral("== ") + command);
    }
    void write(const QString& text) {
        if (!file.isOpen()) return;
        file.write(text.toUtf8());
        file.write("\n");
        file.flush();
    }
    void show(const QString& text, bool in_place) {
        if (!view) return;
        if (in_place && progress) {
            QTextCursor c(view->document()->lastBlock());
            c.movePosition(QTextCursor::StartOfBlock);
            c.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
            c.insertText(text);
        } else {
            view->appendPlainText(text);
        }
        view->verticalScrollBar()->setValue(view->verticalScrollBar()->maximum());
    }
    // A line of the setup's own (stdout, JSON): the file gets it as it came, the view what it says.
    void event_line(const QString& raw, const QString& shown) {
        if (progress) write(progress_text);
        progress = false;
        write(raw);
        if (!shown.isEmpty()) show(shown, false);
    }
    // A line of the installers (stderr): whole, or the latest state of a progress bar being redrawn.
    void err_line(const QString& text, bool unfinished) {
        show(text, true);
        progress = unfinished;
        progress_text = unfinished ? text : QString();
        if (unfinished) return;
        write(text);
        // PowerShell names a refused script in words of the system's language, and this way in any language.
        static const QRegularExpression policy(QStringLiteral("PSSecurityException|UnauthorizedAccess|running scripts is disabled|"
                                                              "is not digitally signed"));
        blocked = blocked || text.contains(policy);
        if (text.startsWith(QLatin1String("+ ")) || text.startsWith(QLatin1String("At line")) || text.startsWith(QLatin1Char('~')))
            return;   // PowerShell's own decoration around an error
        tail << text;
        if (tail.size() > 40) tail.removeFirst();
    }
    // The installers' last line that reads as an error; with any_line, else their last line.
    QString last_error(bool any_line) const {
        static const QRegularExpression words(QStringLiteral("(?i)\\b(error|exception|failed|denied|cannot|could not|not found|"
                                                             "refused|timed out|no space|unable)\\b"));
        for (auto it = tail.crbegin(); it != tail.crend(); ++it)
            if (it->contains(words)) return *it;
        return any_line && !tail.isEmpty() ? tail.back() : QString();
    }
    QString stopped_text(int code, QProcess::ExitStatus how) const {
        if (blocked)
            return QStringLiteral("AI fill setup: Windows did not let RUDRA run its setup script, as PowerShell scripts are "
                                  "blocked on this computer%1. Whoever looks after this computer can allow it: trust the "
                                  "publisher the script is signed by (RUDRA's releases sign it), or change PowerShell's "
                                  "execution policy; then press Set up automatically again.")
                .arg(tail.isEmpty() ? QString() : QStringLiteral(" (\"%1\")").arg(tail.front()));
        const QString last = last_error(true);
        return QStringLiteral("AI fill setup stopped before it finished (%1)%2")
            .arg(how == QProcess::NormalExit ? QStringLiteral("exit code %1").arg(code) : QStringLiteral("stopped or crashed"),
                 last.isEmpty() ? QStringLiteral(".") : QStringLiteral(": ") + last);
    }
    void finish(int code, QProcess::ExitStatus how) {
        if (!err.isEmpty() && !progress) err_line(QString::fromUtf8(err).trimmed(), false);   // a last line with no newline
        if (progress) write(progress_text);
        progress = false;
        write(QStringLiteral("== the setup ended: %1").arg(how == QProcess::NormalExit ? QStringLiteral("exit code %1").arg(code)
                                                                                        : QStringLiteral("stopped or crashed")));
        file.close();
    }
};

}  // namespace

int GenFillReview::flagged() const {
    int n = 0;
    for (const auto& f : frames) n += f.flagged ? 1 : 0;
    return n;
}

std::optional<GenFillReview> read_genfill_review(const QString& fill_dir, QString* why) {
    auto no = [why](const QString& w) -> std::optional<GenFillReview> {
        if (why) *why = w;
        return std::nullopt;
    };
    QFile f(QDir(fill_dir).filePath("genfill.json"));
    if (!f.open(QIODevice::ReadOnly)) return no(QStringLiteral("No genfill.json in %1: not an AI fill folder.").arg(fill_dir));
    const QJsonDocument d = QJsonDocument::fromJson(f.readAll());
    if (!d.isObject()) return no(QStringLiteral("genfill.json cannot be read."));
    const QJsonObject o = d.object();
    GenFillReview r;
    r.dir = fill_dir;
    const QJsonObject rv = o.value("review").toObject();
    r.flag_stops = rv.value("temporal_flag_stops").toDouble(0.25);
    r.review_stops = rv.value("exposure_stops").toDouble(-3);
    r.model = o.value("model").toVariant().toString();
    const QDir pics(QDir(fill_dir).filePath(rv.value("folder").toString("review")));
    const QJsonArray frames = o.value("frames").toArray();
    if (frames.isEmpty()) return no(QStringLiteral("This fill has no review (made before 28 Oct 2026): run it again."));
    for (const auto& v : frames) {
        const QJsonObject fo = v.toObject();
        GenFillReview::Frame fr;
        fr.name = fo.value("name").toString();
        fr.generated = fo.value("generated_fraction").toDouble();
        if (fo.value("temporal_excess_stops").isDouble()) fr.temporal = fo.value("temporal_excess_stops").toDouble();
        fr.flagged = fo.value("flagged").toBool();
        const QString stem = QFileInfo(fr.name).completeBaseName();
        for (auto [suffix, out] : {std::pair{"a", &fr.a}, std::pair{"b", &fr.b}, std::pair{"mask", &fr.mask}}) {
            const QString p = pics.filePath(stem + "." + suffix + ".png");
            if (QFileInfo::exists(p)) *out = p;
        }
        r.frames.push_back(fr);
    }
    if (why) why->clear();
    return r;
}

GenFillReviewDialog::GenFillReviewDialog(GenFillReview review, QWidget* parent) : QDialog(parent), r_(std::move(review)) {
    setWindowTitle(QStringLiteral("AI fill review — %1").arg(QDir::toNativeSeparators(r_.dir)));
    resize(1180, 640);
    auto* root = new QHBoxLayout(this);
    list_ = new QTreeWidget(this);
    list_->setObjectName("fillReviewList");
    list_->setRootIsDecorated(false);
    list_->setHeaderLabels({"Frame", "Invented", "Flicker"});
    list_->headerItem()->setToolTip(1, "The share of the frame's pixels the fill invented (rudra.generated > 0)");
    list_->headerItem()->setToolTip(2, QStringLiteral("Stops of frame-to-frame change the fill adds beyond the master's own, on "
                                                      "the pixels it touched; over %1 is flagged")
                                           .arg(r_.flag_stops, 0, 'f', 2));
    for (const auto& f : r_.frames) {
        auto* it = new QTreeWidgetItem(list_, {f.name, QStringLiteral("%1 %").arg(100.0 * f.generated, 0, 'f', 2),
                                               f.temporal ? QStringLiteral("%1").arg(*f.temporal, 0, 'f', 2) : QStringLiteral("—")});
        if (f.flagged) {
            for (int c = 0; c < 3; ++c) it->setForeground(c, theme_colour("gold"));
            it->setToolTip(2, "Flickers: the fill changes more from the last frame than the master does");
        }
    }
    list_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    list_->setMinimumWidth(320);
    connect(list_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem* it) {
        if (it) show_frame(list_->indexOfTopLevelItem(it), which_);
    });
    auto* left = new QVBoxLayout;
    auto* summary = new QLabel(QStringLiteral("%1 frames · %2 flagged for flicker · pictures %3 stops down so invented "
                                              "highlights show. The master is untouched; the fill is a copy.")
                                   .arg(r_.frames.size())
                                   .arg(r_.flagged())
                                   .arg(r_.review_stops, 0, 'f', 0),
                               this);
    summary->setObjectName("fillReviewSummary");
    summary->setWordWrap(true);
    left->addWidget(summary);
    left->addWidget(list_, 1);
    root->addLayout(left);
    auto* right = new QVBoxLayout;
    auto* bar = new QHBoxLayout;
    for (auto [key, text, tip] : {std::tuple{"a", "A · master", "The learned master (A)"},
                                  std::tuple{"b", "B · fill", "The AI fill (B)"},
                                  std::tuple{"mask", "Invented", "rudra.generated: white = invented by the fill"}}) {
        auto* b = new QPushButton(QString::fromUtf8(text), this);
        b->setObjectName(QStringLiteral("fillReview_") + key);
        b->setCheckable(true);
        b->setAutoExclusive(true);
        b->setToolTip(QString::fromUtf8(tip) + " · Space flips A and B, M the mask, arrows the frames");
        const QString k = QString::fromLatin1(key);
        connect(b, &QPushButton::clicked, this, [this, k] { show_frame(index_, k); });
        bar->addWidget(b);
    }
    bar->addStretch(1);
    auto* open = new QPushButton("Open folder", this);
    connect(open, &QPushButton::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(r_.dir)); });
    bar->addWidget(open);
    right->addLayout(bar);
    image_ = new QLabel(this);
    image_->setObjectName("fillReviewImage");
    image_->setAlignment(Qt::AlignCenter);
    image_->setMinimumSize(480, 270);
    image_->setAutoFillBackground(true);
    {
        QPalette pal = image_->palette();
        pal.setColor(QPalette::Window, theme_colour("scope-bg"));
        image_->setPalette(pal);
    }
    right->addWidget(image_, 1);
    caption_ = new QLabel(this);
    caption_->setObjectName("fillReviewCaption");
    right->addWidget(caption_);
    root->addLayout(right, 1);
    if (!r_.frames.empty()) {
        // The first flagged frame first: what needs a look.
        int start = 0;
        for (std::size_t i = 0; i < r_.frames.size(); ++i)
            if (r_.frames[i].flagged) {
                start = int(i);
                break;
            }
        show_frame(start, "b");
    }
}

void GenFillReviewDialog::show_frame(int index, const QString& which) {
    if (r_.frames.empty()) return;
    index_ = std::clamp(index, 0, int(r_.frames.size()) - 1);
    which_ = which == "a" || which == "mask" ? which : QStringLiteral("b");
    if (list_->currentItem() != list_->topLevelItem(index_)) {
        const QSignalBlocker b(list_);
        list_->setCurrentItem(list_->topLevelItem(index_));
    }
    for (const char* k : {"a", "b", "mask"})
        if (auto* b = findChild<QPushButton*>(QStringLiteral("fillReview_") + k)) b->setChecked(which_ == k);
    refresh();
}

void GenFillReviewDialog::refresh() {
    const auto& f = r_.frames[std::size_t(index_)];
    const QString path = which_ == "a" ? f.a : which_ == "mask" ? f.mask : f.b;
    const QPixmap pm(path);
    if (pm.isNull()) image_->setText(QStringLiteral("No review picture for %1").arg(f.name));
    else image_->setPixmap(pm.scaled(image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    caption_->setText(QStringLiteral("%1 · %2 · %3 % invented%4")
                          .arg(f.name, which_ == "a" ? QStringLiteral("A: master") : which_ == "mask" ? QStringLiteral("invented pixels")
                                                                                                     : QStringLiteral("B: fill"))
                          .arg(100.0 * f.generated, 0, 'f', 2)
                          .arg(f.flagged ? QStringLiteral(" · flickers (%1 stops)").arg(f.temporal.value_or(0), 0, 'f', 2) : QString()));
}

void GenFillReviewDialog::keyPressEvent(QKeyEvent* e) {
    switch (e->key()) {
        case Qt::Key_Space: show_frame(index_, which_ == "a" ? "b" : "a"); return;
        case Qt::Key_M: show_frame(index_, which_ == "mask" ? "b" : "mask"); return;
        case Qt::Key_A: show_frame(index_, "a"); return;
        case Qt::Key_B: show_frame(index_, "b"); return;
        case Qt::Key_Right:
        case Qt::Key_Down: show_frame(index_ + 1, which_); return;
        case Qt::Key_Left:
        case Qt::Key_Up: show_frame(index_ - 1, which_); return;
        default: QDialog::keyPressEvent(e);
    }
}

void GenFillReviewDialog::resizeEvent(QResizeEvent* e) {
    QDialog::resizeEvent(e);
    if (!r_.frames.empty()) refresh();
}

GenFillSetupDialog::GenFillSetupDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle("AI fill setup");
    resize(760, 260);
    auto* v = new QVBoxLayout(this);
    auto* grid = new QGridLayout;
    QString why;
    const auto found = genfill_setup(&why);
    struct Row {
        const char* key;
        const char* label;
        bool folder;
        QString now;
    };
    QSettings st;
    const std::vector<Row> rows = {
        {"python", "Python (torch, diffusers, OpenEXR)", false, st.value("genfill/python").toString()},
        {"repo", "RUDRA repository (rudra/genfill)", true, st.value("genfill/repo").toString()},
        {"base", "Wan2.1-VACE-1.3B-diffusers folder", true, st.value("genfill/base").toString()},
        {"lora", "RUDRA LoRA (rudra_genfill.safetensors)", false, st.value("genfill/lora").toString()},
        // 1.2.6 (checklist item 7): Suggest a prompt's Florence-2, from a folder (optional).
        {"florence", "Florence-2 folder (Suggest a prompt, optional)", true, st.value("genfill/florence").toString()},
    };
    int r = 0;
    for (const Row& row : rows) {
        auto* le = new QLineEdit(this);
        le->setObjectName(QStringLiteral("genfillSetup_") + row.key);
        le->setText(row.now);
        auto* mark = new QLabel(this);
        mark->setObjectName(QStringLiteral("genfillSetupMark_") + row.key);
        auto* b = new QPushButton("Browse…", this);
        const bool folder = row.folder;
        connect(b, &QPushButton::clicked, this, [this, le, folder] {
            const QString p = folder ? QFileDialog::getExistingDirectory(this, {}, le->text())
                                     : QFileDialog::getOpenFileName(this, {}, le->text());
            if (!p.isEmpty()) le->setText(p);
        });
        grid->addWidget(new QLabel(QString::fromUtf8(row.label), this), r, 0);
        grid->addWidget(le, r, 1);
        grid->addWidget(b, r, 2);
        grid->addWidget(mark, r, 3);
        ++r;
    }
    v->addLayout(grid);
    // 1.2.6: Florence-2 from the internet, here where its folder is: the app's own model download
    // (Hugging Face at the pinned revision, every file checked), into the models folder; the
    // AI fill uses it from then on (genfill/florence). Shown while no Florence-2 is found.
    if (auto* mw = dynamic_cast<MainWindow*>(parent)) {
        auto* get_florence = new QPushButton(QStringLiteral("Download Florence-2 (MIT, 0.5 GB)"), this);
        get_florence->setObjectName("genfillGetFlorence");
        get_florence->setToolTip(QStringLiteral("Suggest a prompt describes the shot with Florence-2. Downloads it from "
                                                "Hugging Face (florence-community/Florence-2-base, pinned) and checks "
                                                "every file's SHA-256"));
        QString w;
        const auto s = genfill_setup(&w);
        GenFillSetup probe = s ? *s : GenFillSetup{};
        if (!s) probe.base = to_path(QSettings().value("genfill/base").toString());
        get_florence->setVisible(genfill_florence_dir(probe).empty());
        v->addWidget(get_florence, 0, Qt::AlignLeft);
        QPointer<QPushButton> btn(get_florence);
        connect(get_florence, &QPushButton::clicked, this, [this, mw, btn] {
            ModelDownloads* md = mw->model_downloads();
            md->ensure_builtin();
            // The list's Florence-2 by kind: a published entry may have another id (code review, 19 Nov).
            QString id = QString::fromLatin1(kFlorenceModelId), folder;
            for (const RemoteModel& m : md->models())
                if (m.kind == QLatin1String("florence")) {
                    id = m.id;
                    folder = QDir(md->root()).filePath(m.folder);
                    break;
                }
            QPointer<QLabel> st(findChild<QLabel*>("genfillSetupStatus"));
            // Listening and disabled before the download starts: a failure (or a model already in
            // place) is reported inside download() itself.
            if (btn) btn->setEnabled(false);
            // This click's download only: once it is in (or has failed) the listener does nothing more,
            // so a folder saved later is never written over by later notifications (review, 19 Nov).
            auto settled = std::make_shared<bool>(false);
            md->listen(this, [this, md, st, btn, id, folder, settled] {
                if (*settled) return;
                if (md->active() == id) {
                    if (st) st->setText(QStringLiteral("Downloading Florence-2… %1 of %2")
                                            .arg(human_size(md->done_bytes()), human_size(md->total_bytes())));
                } else if (md->state(id) == InstallState::Installed) {
                    *settled = true;
                    if (st) st->setText(QStringLiteral("Florence-2 is downloaded and checked: Suggest a prompt uses it."));
                    if (btn) btn->setVisible(false);
                    // The download's folder itself: the setting may be saved only after this call.
                    if (auto* le = findChild<QLineEdit*>(QStringLiteral("genfillSetup_florence")); le && le->text().trimmed().isEmpty())
                        le->setText(QDir::toNativeSeparators(folder));
                    QSettings().setValue("genfill/florence", folder);
                    refresh_found();
                } else if (!md->error(id).isEmpty()) {
                    *settled = true;   // a new click listens again
                    if (st) st->setText(QStringLiteral("Florence-2 was not downloaded: ") + md->error(id));
                    if (btn) btn->setEnabled(true);
                }
            });
            QString why;
            if (!md->download(id, &why)) {
                *settled = true;
                if (st) st->setText(QStringLiteral("Florence-2: ") + why);
                if (btn) btn->setEnabled(true);
            }
        });
    }
    auto* status = new QLabel(found ? genfill_ready_text(*found) : why, this);
    status->setObjectName("genfillSetupStatus");
    status->setWordWrap(true);
    v->addWidget(status);
    // 4 Nov 2026: a saved LoRA that is not v3 stays (the artist chose it); one click
    // switches to the verified v3 the search found (File > Model packages, the repo ...).
    auto* use_v3 = new QPushButton("Use AI fill v3", this);
    use_v3->setObjectName("genfillUseV3");
    use_v3->setToolTip("Switch the LoRA to the published AI fill v3 found on this computer (SHA-256 verified)");
    use_v3->setVisible(found && !found->lora_is_v3 && !found->v3_found.empty());
    v->addWidget(use_v3, 0, Qt::AlignLeft);
    connect(use_v3, &QPushButton::clicked, this, [this, status, use_v3] {
        QString w;
        const auto now = genfill_setup(&w);
        if (!now || now->v3_found.empty()) return;
        QSettings().setValue("genfill/lora", from_path(now->v3_found));
        if (auto* le = findChild<QLineEdit*>(QStringLiteral("genfillSetup_lora"))) le->setText(from_path(now->v3_found));
        const auto ok = genfill_setup(&w);
        status->setText(ok ? genfill_ready_text(*ok) : w);
        use_v3->setVisible(ok && !ok->lora_is_v3 && !ok->v3_found.empty());
    });
    // One click (4 Nov 2026): rudra/genfill/setup.py checks the GPU and the disk,
    // installs the packages into the Python above and downloads the base model,
    // with its log here. Closing the window stops it; run again, it resumes.
    auto* log_view = new QPlainTextEdit(this);
    log_view->setObjectName("genfillSetupLog");
    log_view->setReadOnly(true);
    log_view->setMaximumBlockCount(2000);
    log_view->setVisible(false);
    v->addWidget(log_view, 1);
    auto* busy = new QProgressBar(this);
    busy->setObjectName("genfillSetupBusy");
    busy->setRange(0, 0);   // steps, not bytes: the download's own progress is in the log
    busy->setVisible(false);
    v->addWidget(busy);

    auto* box = new QDialogButtonBox(QDialogButtonBox::Close, this);
    auto* automatic = box->addButton("Set up automatically…", QDialogButtonBox::ActionRole);
    automatic->setObjectName("genfillSetupAuto");
    // 1.3.2 (end-user report, 27 Nov 2026): the whole of the last setup's log, for whoever helps.
    // Its own button, not the button box's: Qt before 6.7 shows every button of a box again when it lays them out
    // (the Qt shell CI, 28 Nov 2026: "Copy log" showed with no log to copy).
    auto* copy_log = new QPushButton(QStringLiteral("Copy log"), this);
    copy_log->setObjectName("genfillSetupCopyLog");
    v->addWidget(copy_log, 0, Qt::AlignLeft);
    copy_log->setToolTip(QStringLiteral("Puts everything the last setup printed on the clipboard\n(%1)")
                             .arg(QDir::toNativeSeparators(from_path(genfill_setup_log()))));
    copy_log->setVisible(QFileInfo::exists(from_path(genfill_setup_log())));
    connect(copy_log, &QPushButton::clicked, this, [copy_log] {
        QFile f(from_path(genfill_setup_log()));
        if (!f.open(QIODevice::ReadOnly)) return;
        QGuiApplication::clipboard()->setText(QString::fromUtf8(f.readAll()));
        copy_log->setText(QStringLiteral("Copied"));
        QTimer::singleShot(2500, copy_log, [copy_log] { copy_log->setText(QStringLiteral("Copy log")); });
    });
    connect(automatic, &QPushButton::clicked, this, [this, status, log_view, busy, automatic, copy_log] {
        // 1.2.6 (checklist item 7): set by "Install from a folder or zip…": --from / --florence-from.
        const QStringList offline = automatic->property("offlineArgs").toStringList();
        automatic->setProperty("offlineArgs", QVariant());
        GenFillInstall plan = genfill_install_plan();
        if (const QString b = findChild<QLineEdit*>("genfillSetup_base")->text().trimmed(); !b.isEmpty()) plan.base = to_path(b);
        if (const QString p = findChild<QLineEdit*>("genfillSetup_python")->text().trimmed(); !p.isEmpty()) plan.python = to_path(p);
        if (plan.repo.empty() || !QFileInfo::exists(from_path(plan.repo / "rudra" / "genfill" / "setup.py"))) {
            status->setText("AI fill setup: the RUDRA repository (rudra/genfill/setup.py) was not found; set it above.");
            return;
        }
        // 7 Nov 2026: no Python for the fill: the bootstrap makes RUDRA its own (Python 3.12, PyTorch,
        // the packages) under the app's data folder, then runs the same setup.
        const std::filesystem::path script = genfill_bootstrap_script(plan.repo);
        const bool python_there = QFileInfo::exists(from_path(plan.python));
        // 1.3.2 (end-user report, 27 Nov 2026): RUDRA's own Python goes through the bootstrap every time, not only
        // when it is missing. A bootstrap that stopped (the 2.5 GB PyTorch download, the window closed) left a
        // Python with no PyTorch, and every click after it ended in "PyTorch is not installed". Each step of the
        // bootstrap checks itself and is passed over when it is done, so a second run reaches the model in seconds.
        const bool own_python =
            python_there && QFileInfo(from_path(plan.python)).absoluteFilePath().compare(
                                QFileInfo(from_path(genfill_bootstrap_python())).absoluteFilePath(), Qt::CaseInsensitive) == 0;
        // 1.3.2 (user feedback): installing a model from a folder or zip goes through the bootstrap too, so it works
        // on a computer with no Python for the fill yet: the bootstrap makes it, then installs the model.
        const bool bootstrap = !script.empty() && (!python_there || own_python);
        if (!python_there && !bootstrap) {
            status->setText(QStringLiteral("AI fill setup: no Python at %1. Set the Python above (one with PyTorch for your GPU).")
                                .arg(from_path(plan.python)));
            return;
        }
        // 1.3.2: a download that stopped has transformer/config.json too; the setup carries it on.
        const bool model_files = QFileInfo::exists(from_path(plan.base / "transformer" / "config.json"));
        const QString unfinished = model_files ? genfill_wan_missing(plan.base) : QString();
        const bool have_model = model_files && unfinished.isEmpty();
        const QString model_step =
            have_model ? QStringLiteral("checks the base model already at\n%1").arg(from_path(plan.base))
            : !unfinished.isEmpty()
                ? QStringLiteral("finishes the Wan2.1-VACE-1.3B download that stopped, in\n%1").arg(from_path(plan.base))
                : QStringLiteral("downloads Wan2.1-VACE-1.3B (Apache 2.0, about 18 GB) into\n%1").arg(from_path(plan.base));
        const QString offline_model = offline.value(0) == QLatin1String("--from") ? QStringLiteral("Wan2.1-VACE-1.3B")
                                                                                  : QStringLiteral("Florence-2");
        const QString what = !offline.isEmpty() && !python_there
            ? QStringLiteral("This computer has no Python for the AI fill yet. This first installs one for RUDRA only, in\n%1\n\n"
                             "(Python 3.12, PyTorch and the fill's packages: about 5 GB from the internet), then installs %2 "
                             "from\n%3\n\nwith no download. Every file is checked against the model manifest (its SHA-256) "
                             "before it is moved into place.")
                  .arg(from_path(genfill_bootstrap_root()), offline_model, QDir::toNativeSeparators(offline.value(1)))
            : !offline.isEmpty()
            ? QStringLiteral("This installs %1 from\n%2\n\nwith no download. Every file is checked against the model "
                             "manifest (its SHA-256) before it is moved into place; a model already there is kept beside it.")
                  .arg(offline_model, QDir::toNativeSeparators(offline.value(1)))
            : bootstrap && python_there
            ? QStringLiteral("This checks RUDRA's own Python for the AI fill, in\n%1\n\nand installs whatever it is missing "
                             "(PyTorch, the fill's packages), then %2\n\nand Florence-2 (MIT, about 0.5 GB) for Suggest a "
                             "prompt. A step that stopped carries on where it stopped.")
                  .arg(from_path(genfill_bootstrap_root()), model_step)
            : bootstrap
            ? QStringLiteral("This computer has no Python for the AI fill. This installs one for RUDRA only, in\n%1\n\n"
                             "Python 3.12, PyTorch and the fill's packages (about 5 GB), then %2\n\n"
                             "Nothing else on this computer changes. It needs a 16 GB GPU (NVIDIA, or Apple silicon). "
                             "A step that stops resumes when you run this again.")
                  .arg(from_path(genfill_bootstrap_root()),
                       model_step + QStringLiteral("\n\nand Florence-2 (MIT, about 0.5 GB) for Suggest a prompt."))
            : have_model
            ? QStringLiteral("The base model is already at\n%1\n\nThis installs the missing Python packages into\n%2\n\n"
                             "and Florence-2 (MIT, about 0.5 GB) for Suggest a prompt, if it is not there yet.")
                  .arg(from_path(plan.base), from_path(plan.python))
            : QStringLiteral("This downloads Wan2.1-VACE-1.3B (Apache 2.0, about 18 GB) and Florence-2 (MIT, about 0.5 GB, "
                             "for Suggest a prompt) into\n%1\n\n"
                             "and installs the Python packages the fill needs into\n%2\n\n"
                             "It needs a 16 GB GPU. A download that stops resumes when you run this again.")
                  .arg(from_path(plan.base), from_path(plan.python));
        if (!skip_question_ && QMessageBox::question(this, "Set up AI fill", what) != QMessageBox::Yes) return;
        auto* proc = new QProcess(this);   // the dialog's: closing it ends the setup (1.3.2: it asks first, reject())
        setup_ = proc;
        proc->setWorkingDirectory(from_path(plan.repo));
        if (bootstrap) {
#ifdef _WIN32
            // By its full path: a bare name is also looked for in the current folder (11 Nov 2026).
            proc->setProgram(QDir(qEnvironmentVariable("SystemRoot", QStringLiteral("C:/Windows")))
                                 .filePath(QStringLiteral("System32/WindowsPowerShell/v1.0/powershell.exe")));
            QStringList args{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", from_path(script), "-Root",
                             from_path(genfill_bootstrap_root()), "-Base", from_path(plan.base), "-Repo", from_path(plan.repo)};
            if (offline.size() == 2)   // 1.3.2: --from / --florence-from, as the bootstrap's -From / -FlorenceFrom
                args << (offline.first() == QLatin1String("--from") ? QStringLiteral("-From") : QStringLiteral("-FlorenceFrom"))
                     << offline.value(1);
            proc->setArguments(args);
#else
            proc->setProgram(QStringLiteral("/bin/bash"));
            proc->setArguments(QStringList{from_path(script), from_path(genfill_bootstrap_root()), from_path(plan.base),
                                           from_path(plan.repo)} + offline);   // 1.3.2: --from / --florence-from
#endif
        } else {
            proc->setProgram(from_path(plan.python));
            // 1.2.6: Florence-2 (MIT, about 0.5 GB, Suggest a prompt) comes with the setup, pinned and checked.
            proc->setArguments(QStringList{"-m", "rudra.genfill.setup", "--base", from_path(plan.base), "--json"} +
                               (offline.isEmpty() ? QStringList{"--florence"} : offline));
        }
        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONUNBUFFERED", "1");
        proc->setProcessEnvironment(env);
        log_view->setVisible(true);
        log_view->clear();
        busy->setVisible(true);
        automatic->setEnabled(false);
        copy_log->setVisible(false);
        status->setText("Setting up the AI fill…");
        auto run = std::make_shared<SetupRun>();
        run->view = log_view;
        run->open(QDir::toNativeSeparators(proc->program()) + QLatin1Char(' ') + proc->arguments().join(QLatin1Char(' ')));
        run->write(QStringLiteral("== ") + QString(what).replace(QLatin1Char('\n'), QLatin1Char(' ')));
        // 1.3.2 (28 Nov 2026, end-user report): the window stayed empty until the setup's first step, so a slow start
        // looked like a setup that hung. It says at once that it started, and where the whole log goes.
        run->show(QStringLiteral("Starting the setup. Everything it prints also goes to %1")
                      .arg(QDir::toNativeSeparators(from_path(genfill_setup_log()))), false);
        connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc, run, status] {
            run->out.append(proc->readAllStandardOutput());
            for (int nl; (nl = run->out.indexOf('\n')) >= 0;) {
                const QByteArray line = run->out.left(nl).trimmed();
                run->out.remove(0, nl + 1);
                if (line.isEmpty()) continue;
                const QJsonObject o = QJsonDocument::fromJson(line).object();
                const QString ev = o.value("event").toString();
                QString shown;
                if (ev == "step") {
                    status->setText(o.value("text").toString());
                    shown = o.value("text").toString();
                } else if (ev == "bootstrap") {
                    // The fill's own Python exists now: it is the one to use. Its path is the app's own (1.3.2), not
                    // the script's line, which a console code page garbled for a user name in another alphabet.
                    const QString py = from_path(genfill_bootstrap_python());
                    QSettings().setValue("genfill/python", py);
                    if (auto* le = findChild<QLineEdit*>("genfillSetup_python")) le->setText(py);
                    shown = QStringLiteral("Python for the AI fill: ") + QDir::toNativeSeparators(py);
                } else if (ev == "check") {
                    shown = QStringLiteral("%1 %2 GB on %3 · %4 GB free · model %5")
                                .arg(o.value("gpu").toString(o.value("device").toString()))
                                .arg(o.value("memory_gb").toDouble(), 0, 'f', 1)
                                .arg(o.value("device").toString())
                                .arg(o.value("free_gb").toDouble(), 0, 'f', 0)
                                .arg(o.value("model").toBool() ? "found" : "to download");
                } else if (ev == "done") {
                    run->done = true;
                    run->ok = o.value("ok").toBool();
                    if (!run->ok) status->setText("AI fill setup: " + o.value("error").toString());
                    shown = !run->ok ? QStringLiteral("Stopped: ") + o.value("error").toString()
                                     : o.value("florence_warning").toString();
                } else {
                    shown = o.isEmpty() ? QString::fromUtf8(line) : o.value("text").toString();
                }
                run->event_line(QString::fromUtf8(line), shown);
            }
        });
        // The installers' own lines (stderr): every whole line, and a progress bar (tqdm: redrawn after a carriage
        // return) as its latest state, in place.
        connect(proc, &QProcess::readyReadStandardError, this, [proc, run] {
            run->err.append(proc->readAllStandardError());
            for (int nl; (nl = run->err.indexOf('\n')) >= 0;) {
                QString line = QString::fromUtf8(run->err.left(nl));
                run->err.remove(0, nl + 1);
                if (line.endsWith(QLatin1Char('\r'))) line.chop(1);
                line = line.mid(line.lastIndexOf(QLatin1Char('\r')) + 1).trimmed();
                if (!line.isEmpty()) run->err_line(line, false);
            }
            // What is left has no newline yet. A progress bar redraws itself after a carriage return: its latest state
            // is shown in place, and only that is kept. A carriage return at the very end is half of a Windows line
            // end (the newline comes in the next piece): the line is kept whole until it does.
            qsizetype cr = run->err.lastIndexOf('\r');
            if (cr >= 0 && cr == run->err.size() - 1) cr = cr > 0 ? run->err.lastIndexOf('\r', cr - 1) : -1;
            if (cr >= 0) {
                const QString latest = QString::fromUtf8(run->err.mid(cr + 1)).trimmed();
                run->err.remove(0, cr);   // the bar's line is whole once its newline comes
                if (!latest.isEmpty()) run->err_line(latest, true);
            }
        });
        connect(proc, &QProcess::finished, this, [this, proc, run, status, busy, automatic, copy_log, plan](int code, QProcess::ExitStatus how) {
            busy->setVisible(false);
            automatic->setEnabled(true);
            proc->deleteLater();
            run->finish(code, how);
            copy_log->setVisible(true);
            // 1.3.2: success is the setup's own "done" with ok, not the exit code alone (a PowerShell that refused
            // the script can end with 0), and every other end says why and where the whole log is.
            if (how != QProcess::NormalExit || code != 0 || !run->ok) {
                if (!run->done) {
                    status->setText(run->stopped_text(code, how));
                } else if (status->text().contains(QLatin1String("see the log"))) {   // the installer's own words for it
                    if (const QString last = run->last_error(false); !last.isEmpty()) status->setText(status->text() + QStringLiteral("\n") + last);
                }
                status->setText(status->text() + QStringLiteral("\n\nEverything the setup printed is in %1: Copy log puts it "
                                                                "on the clipboard.")
                                                     .arg(QDir::toNativeSeparators(from_path(genfill_setup_log()))));
                return;
            }
            QSettings s;
            s.setValue("genfill/base", from_path(plan.base));
            if (auto* le = findChild<QLineEdit*>("genfillSetup_base")) le->setText(from_path(plan.base));
            QString w;
            const auto ok = genfill_setup(&w);
            status->setText(ok ? genfill_ready_text(*ok)
                               : QStringLiteral("Packages and base model are set up. ") + w);
            // The last part, the LoRA, is a model package (7 Nov 2026): one click to it.
            if (auto* lora = findChild<QPushButton*>("genfillGetLora")) lora->setVisible(!ok && w.contains("no LoRA"));
            refresh_found();
        });
        connect(proc, &QProcess::errorOccurred, this, [proc, run, status, busy, automatic, copy_log](QProcess::ProcessError e) {
            if (e != QProcess::FailedToStart) return;   // a crash or a stop ends in finished(), which says so
            busy->setVisible(false);
            automatic->setEnabled(true);
            run->write(QStringLiteral("== could not start: ") + proc->errorString());
            run->finish(-1, QProcess::CrashExit);
            copy_log->setVisible(true);
            status->setText(QStringLiteral("AI fill setup: %1 could not be started (%2).")
                                .arg(QDir::toNativeSeparators(proc->program()), proc->errorString()));
            proc->deleteLater();   // no finished() after a start that failed
        });
        end_with_rudra(proc);   // 1.3: the set-up (and the installers it runs) ends with RUDRA
        proc->start();
    });
    // 1.2.6 (readiness report, checklist item 7): a facility with no internet installs the models
    // from a folder or zip it was given; rudra.genfill.setup checks every file against
    // rudra/genfill/model_manifest.json (the pinned revision's SHA-256) before it is used.
    auto* from = box->addButton("Install from a folder or zip…", QDialogButtonBox::ActionRole);
    from->setObjectName("genfillSetupFrom");
    from->setToolTip("Install Wan2.1-VACE or Florence-2 from a folder or zip, with no model download, checked by SHA-256 "
                     "(1.3.2: without AI fill's Python, it sets that up first, about 5 GB from the internet)");
    auto* from_menu = new QMenu(from);
    for (const auto& [label, flag, zip] : std::vector<std::tuple<QString, QString, bool>>{
             {"Wan2.1-VACE-1.3B from a zip…", "--from", true},
             {"Wan2.1-VACE-1.3B from a folder…", "--from", false},
             {"Florence-2 from a zip…", "--florence-from", true},
             {"Florence-2 from a folder…", "--florence-from", false}}) {
        connect(from_menu->addAction(label), &QAction::triggered, this, [this, automatic, flag, zip] {
            const QString p = zip ? QFileDialog::getOpenFileName(this, {}, {}, QStringLiteral("Zip (*.zip)"))
                                  : QFileDialog::getExistingDirectory(this);
            if (p.isEmpty()) return;
            automatic->setProperty("offlineArgs", QStringList{flag, p});
            automatic->click();
        });
    }
    from->setMenu(from_menu);
    // 7 Nov 2026: the LoRA is File > Model packages > AI fill v3; this opens it (the dialog closes first,
    // it is modal) when it is the only part missing.
    auto* get_lora = box->addButton("Download AI fill v3…", QDialogButtonBox::ActionRole);
    get_lora->setObjectName("genfillGetLora");
    get_lora->setToolTip("File > Model packages: AI fill v3 (the RUDRA LoRA, Apache 2.0), checked by its SHA-256");
    get_lora->setVisible(!found && why.contains("no LoRA"));
    connect(get_lora, &QPushButton::clicked, this, [this] {
        auto* mw = dynamic_cast<MainWindow*>(parentWidget());
        accept();
        if (mw) QTimer::singleShot(0, mw, [mw] { mw->open_model_manager(); });
    });
    auto* check = box->addButton("Save and check", QDialogButtonBox::ApplyRole);
    connect(check, &QPushButton::clicked, this, [this, status] {
        QSettings s;
        for (const char* k : {"python", "repo", "base", "lora", "florence"}) {
            const QString t = findChild<QLineEdit*>(QStringLiteral("genfillSetup_") + k)->text().trimmed();
            if (t.isEmpty()) s.remove(QStringLiteral("genfill/") + k);
            else s.setValue(QStringLiteral("genfill/") + k, t);
        }
        QString w;
        const auto ok = genfill_setup(&w);
        status->setText(ok ? genfill_ready_text(*ok) : w);
        if (auto* u = findChild<QPushButton*>(QStringLiteral("genfillUseV3"))) u->setVisible(ok && !ok->lora_is_v3 && !ok->v3_found.empty());
        if (auto* l = findChild<QPushButton*>(QStringLiteral("genfillGetLora"))) l->setVisible(!ok && w.contains("no LoRA"));
        refresh_found();
    });
    connect(box, &QDialogButtonBox::rejected, this, &QDialog::reject);
    v->addWidget(box);
    refresh_found();
}

void GenFillSetupDialog::refresh_found() {
    QString why;
    const auto found = genfill_setup(&why);
    const GenFillInstall plan = genfill_install_plan();
    for (const char* key : {"python", "repo", "base", "lora", "florence"}) {
        const std::string k = key;
        QString auto_path;
        if (k == "florence") {
            if (found) auto_path = from_path(genfill_florence_dir(GenFillSetup{found->python, found->repo, found->base, {}}));
        } else if (found)
            auto_path = from_path(k == "python" ? found->python : k == "repo" ? found->repo : k == "base" ? found->base : found->lora);
        else if (k == "python" && !plan.python.empty() && QFileInfo::exists(from_path(plan.python)))
            auto_path = from_path(plan.python);
        else if (k == "repo" && !plan.repo.empty())
            auto_path = from_path(plan.repo);
        auto* le = findChild<QLineEdit*>(QStringLiteral("genfillSetup_") + key);
        auto* mark = findChild<QLabel*>(QStringLiteral("genfillSetupMark_") + key);
        if (!le || !mark) continue;
        const bool typed = !le->text().trimmed().isEmpty();
        const bool ok = typed ? QFileInfo::exists(le->text().trimmed()) : !auto_path.isEmpty();
        le->setPlaceholderText(auto_path.isEmpty() ? QStringLiteral("not found: Browse…, or Set up automatically")
                                                   : QStringLiteral("found automatically: ") + QDir::toNativeSeparators(auto_path));
        le->setToolTip(auto_path.isEmpty() ? QStringLiteral("Not found on this computer")
                                           : QStringLiteral("In use: ") + QDir::toNativeSeparators(auto_path) +
                                                 QStringLiteral("\nLeave empty to keep finding it automatically"));
        mark->setText(ok ? QStringLiteral("\u2713") : QStringLiteral("\u2717"));
        mark->setToolTip(ok ? QStringLiteral("found") : QStringLiteral("missing"));
    }
    if (auto* a = findChild<QPushButton*>("genfillSetupAuto")) {
        a->setText(found ? QStringLiteral("Check and repair…") : QStringLiteral("Set up automatically…"));
        a->setToolTip(found ? QStringLiteral("Everything was found automatically. This checks the GPU and the disk and installs any missing package")
                            : QStringLiteral("Installs the Python packages and downloads the base model, then finds every part by itself"));
        a->setDefault(!found);
    }
}

void GenFillSetupDialog::start_automatic() {
    skip_question_ = true;
    if (auto* a = findChild<QPushButton*>(QStringLiteral("genfillSetupAuto"))) a->click();   // runs the setup at once
    skip_question_ = false;
}

void GenFillSetupDialog::reject() {
    // 1.3.2 (end-user report, 27 Nov 2026): closing this window stops the setup. It did so without a word, mid
    // download, and on Windows the installers the setup had started went on unseen beside the next run.
    if (setup_ && setup_->state() != QProcess::NotRunning) {
        if (QMessageBox::question(this, QStringLiteral("AI fill setup"),
                                  QStringLiteral("The AI fill setup is still running. Closing this window stops it; Set up "
                                                 "automatically… carries on where it stopped the next time.\n\n"
                                                 "Stop the setup and close?")) != QMessageBox::Yes)
            return;
        const QString pid = QString::number(setup_->processId());
#ifdef _WIN32
        // The whole tree: PowerShell, and the uv or Python under it (by its full path, like PowerShell's).
        QProcess::execute(QDir(qEnvironmentVariable("SystemRoot", QStringLiteral("C:/Windows"))).filePath(QStringLiteral("System32/taskkill.exe")),
                          {QStringLiteral("/T"), QStringLiteral("/F"), QStringLiteral("/PID"), pid});
#else
        QProcess::execute(QStringLiteral("pkill"), {QStringLiteral("-TERM"), QStringLiteral("-P"), pid});
#endif
        if (setup_) setup_->kill();
        if (setup_) setup_->waitForFinished(5000);
    }
    QDialog::reject();
}

}  // namespace rudra::app
