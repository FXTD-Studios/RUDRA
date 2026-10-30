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
#include <QPushButton>
#include <QSettings>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include "genfill_job.hpp"
#include "paths.hpp"
#include "theme.hpp"

#include <algorithm>
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
        le->setPlaceholderText("found automatically (beside the app, the repository, RUDRA_GENFILL_DIR)");
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
        ++r;
    }
    v->addLayout(grid);
    auto* status = new QLabel(found ? QStringLiteral("Ready: %1").arg(from_path(found->lora)) : why, this);
    status->setObjectName("genfillSetupStatus");
    status->setWordWrap(true);
    v->addWidget(status);
    auto* box = new QDialogButtonBox(QDialogButtonBox::Close, this);
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
        status->setText(ok ? QStringLiteral("Ready: %1").arg(from_path(ok->lora)) : w);
    });
    connect(box, &QDialogButtonBox::rejected, this, &QDialog::reject);
    v->addWidget(box);
}

}  // namespace rudra::app
