#include "genfill_review.hpp"

#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QPalette>
#include <QPixmap>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QSettings>
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
    connect(automatic, &QPushButton::clicked, this, [this, status, log_view, busy, automatic] {
        GenFillInstall plan = genfill_install_plan();
        if (const QString b = findChild<QLineEdit*>("genfillSetup_base")->text().trimmed(); !b.isEmpty()) plan.base = to_path(b);
        if (const QString p = findChild<QLineEdit*>("genfillSetup_python")->text().trimmed(); !p.isEmpty()) plan.python = to_path(p);
        if (plan.repo.empty() || !QFileInfo::exists(from_path(plan.repo / "rudra" / "genfill" / "setup.py"))) {
            status->setText("AI fill setup: the RUDRA repository (rudra/genfill/setup.py) was not found; set it above.");
            return;
        }
        // 7 Nov 2026: no Python for the fill: the bootstrap makes RUDRA its own (uv, Python 3.12,
        // PyTorch, the packages) under the app's data folder, then runs the same setup.
        const std::filesystem::path script = genfill_bootstrap_script(plan.repo);
        const bool bootstrap = !QFileInfo::exists(from_path(plan.python)) && !script.empty();
        if (!QFileInfo::exists(from_path(plan.python)) && !bootstrap) {
            status->setText(QStringLiteral("AI fill setup: no Python at %1. Set the Python above (one with PyTorch for your GPU).")
                                .arg(from_path(plan.python)));
            return;
        }
        const bool have_model = QFileInfo::exists(from_path(plan.base / "transformer" / "config.json"));
        const QString what = bootstrap
            ? QStringLiteral("This computer has no Python for the AI fill. This installs one for RUDRA only, in\n%1\n\n"
                             "Python 3.12, PyTorch and the fill's packages (about 5 GB), then %2\n\n"
                             "Nothing else on this computer changes. It needs a 16 GB GPU (NVIDIA, or Apple silicon). "
                             "A step that stops resumes when you run this again.")
                  .arg(from_path(genfill_bootstrap_root()),
                       have_model ? QStringLiteral("checks the base model already at\n%1").arg(from_path(plan.base))
                                  : QStringLiteral("downloads Wan2.1-VACE-1.3B (Apache 2.0, about 25 GB) into\n%1").arg(from_path(plan.base)))
            : have_model
            ? QStringLiteral("The base model is already at\n%1\n\nThis installs the missing Python packages into\n%2")
                  .arg(from_path(plan.base), from_path(plan.python))
            : QStringLiteral("This downloads Wan2.1-VACE-1.3B (Apache 2.0, about 25 GB) into\n%1\n\n"
                             "and installs the Python packages the fill needs into\n%2\n\n"
                             "It needs a 16 GB GPU. A download that stops resumes when you run this again.")
                  .arg(from_path(plan.base), from_path(plan.python));
        if (QMessageBox::question(this, "Set up AI fill", what) != QMessageBox::Yes) return;
        auto* proc = new QProcess(this);   // the dialog's: closing it ends the setup
        proc->setWorkingDirectory(from_path(plan.repo));
        if (bootstrap) {
#ifdef _WIN32
            proc->setProgram(QStringLiteral("powershell.exe"));
            proc->setArguments({"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", from_path(script), "-Root",
                                from_path(genfill_bootstrap_root()), "-Base", from_path(plan.base), "-Repo", from_path(plan.repo)});
#else
            proc->setProgram(QStringLiteral("/bin/bash"));
            proc->setArguments({from_path(script), from_path(genfill_bootstrap_root()), from_path(plan.base), from_path(plan.repo)});
#endif
        } else {
            proc->setProgram(from_path(plan.python));
            proc->setArguments({"-m", "rudra.genfill.setup", "--base", from_path(plan.base), "--json"});
        }
        auto env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONUNBUFFERED", "1");
        proc->setProcessEnvironment(env);
        log_view->setVisible(true);
        log_view->clear();
        busy->setVisible(true);
        automatic->setEnabled(false);
        status->setText("Setting up the AI fill…");
        auto out_buf = std::make_shared<QByteArray>();
        connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc, out_buf, status, log_view] {
            out_buf->append(proc->readAllStandardOutput());
            for (int nl; (nl = out_buf->indexOf('\n')) >= 0;) {
                const QByteArray line = out_buf->left(nl).trimmed();
                out_buf->remove(0, nl + 1);
                const QJsonObject o = QJsonDocument::fromJson(line).object();
                const QString ev = o.value("event").toString();
                if (ev == "step") status->setText(o.value("text").toString());
                else if (ev == "bootstrap") {   // the fill's own Python exists now: it is the one to use
                    QSettings().setValue("genfill/python", o.value("python").toString());
                    if (auto* le = findChild<QLineEdit*>("genfillSetup_python")) le->setText(o.value("python").toString());
                    log_view->appendPlainText(QStringLiteral("Python for the AI fill: ") + o.value("python").toString());
                }
                else if (ev == "check")
                    log_view->appendPlainText(QStringLiteral("%1 %2 GB on %3 · %4 GB free · model %5")
                                                  .arg(o.value("gpu").toString(o.value("device").toString()))
                                                  .arg(o.value("memory_gb").toDouble(), 0, 'f', 1)
                                                  .arg(o.value("device").toString())
                                                  .arg(o.value("free_gb").toDouble(), 0, 'f', 0)
                                                  .arg(o.value("model").toBool() ? "found" : "to download"));
                else if (ev == "done" && !o.value("ok").toBool()) status->setText("AI fill setup: " + o.value("error").toString());
                else if (!line.isEmpty()) log_view->appendPlainText(o.isEmpty() ? QString::fromUtf8(line) : o.value("text").toString());
            }
        });
        // The download's progress bars (tqdm, on stderr): the latest line only.
        connect(proc, &QProcess::readyReadStandardError, this, [proc, log_view] {
            const QString chunk = QString::fromUtf8(proc->readAllStandardError());
            const QStringList parts = chunk.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);
            if (!parts.isEmpty()) log_view->appendPlainText(parts.back().trimmed());
        });
        connect(proc, &QProcess::finished, this, [this, proc, status, busy, automatic, plan](int code, QProcess::ExitStatus) {
            busy->setVisible(false);
            automatic->setEnabled(true);
            proc->deleteLater();
            if (code != 0) return;   // the reason is on the status line already
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
        connect(proc, &QProcess::errorOccurred, this, [status, busy, automatic](QProcess::ProcessError) {
            busy->setVisible(false);
            automatic->setEnabled(true);
            status->setText("AI fill setup: the Python could not be started.");
        });
        proc->start();
    });
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
        for (const char* k : {"python", "repo", "base", "lora"}) {
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
    for (const char* key : {"python", "repo", "base", "lora"}) {
        const std::string k = key;
        QString auto_path;
        if (found)
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

}  // namespace rudra::app
