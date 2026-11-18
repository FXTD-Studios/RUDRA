#include "media_library.hpp"

#include <QApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMimeData>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QStandardPaths>
#include <QStyle>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <optional>

#include "rudra/media/sequence.hpp"
#include "theme.hpp"

namespace rudra::app {
namespace {

bool listed(const std::vector<std::string>& suffixes, const QString& path) {
    const std::string s = QFileInfo(path).suffix().toLower().prepend('.').toStdString();
    return std::find(suffixes.begin(), suffixes.end(), s) != suffixes.end();
}

// The theme's hues (ui/theme.css): gold for clipped, the accent for crushed, ok for delivered.
QColor smart_dot(const char* key) {
    return theme_colour(QString(key) == MediaLibrary::kClipped   ? QStringLiteral("gold")
                        : QString(key) == MediaLibrary::kCrushed ? QStringLiteral("accent")
                                                                 : QStringLiteral("ok"));
}

QString thumb_dir() {
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/thumbs");
}

QString thumb_file(const QString& path) {
    return thumb_dir() + QLatin1Char('/') +
           QString::fromLatin1(QCryptographicHash::hash(path.toUtf8(), QCryptographicHash::Sha1).toHex()) +
           QStringLiteral(".png");
}

}  // namespace

// ---- MediaClip / MediaLibrary ---------------------------------------------------------------

QString MediaClip::name() const { return QFileInfo(path).fileName(); }

QJsonObject MediaClip::settings() const {
    return has_settings() ? versions.at(version).toObject().value("settings").toObject() : QJsonObject();
}

void MediaClip::set_settings(const QJsonObject& s) {
    if (!has_settings()) {
        add_version(s);
        return;
    }
    QJsonObject v = versions.at(version).toObject();
    v["settings"] = s;
    versions.replace(version, v);
}

int MediaClip::add_version(const QJsonObject& s) {
    versions.append(QJsonObject{{"name", QStringLiteral("v%1").arg(versions.size() + 1)}, {"settings", s}});
    version = int(versions.size()) - 1;
    return version;
}

QString MediaClip::version_name() const {
    return has_settings() ? versions.at(version).toObject().value("name").toString() : QString();
}

QString MediaClip::badge() const {
    const QString ext = QFileInfo(path).suffix().toUpper();
    if (kind == "folder") return QStringLiteral("SEQ");
    if (ext == "JPEG") return QStringLiteral("JPG");
    if (ext == "TIFF") return QStringLiteral("TIF");
    return ext.left(4);
}

QString MediaLibrary::clean(const QString& path) { return QDir::cleanPath(QFileInfo(path).absoluteFilePath()); }

QString MediaLibrary::kind_of(const QString& path) {
    const QFileInfo fi(path);
    if (fi.isDir()) return QStringLiteral("folder");
    if (listed(video_suffixes(), path)) return QStringLiteral("movie");
    return QStringLiteral("still");
}

MediaClip* MediaLibrary::add(const QString& path) {
    if (path.isEmpty() || QFileInfo(path).suffix().compare(QStringLiteral("rudra"), Qt::CaseInsensitive) == 0) return nullptr;
    const QString p = clean(path);
    if (MediaClip* c = find(p)) return c;
    MediaClip c;
    c.path = p;
    c.kind = kind_of(p);
    clips_.insert(clips_.begin(), c);   // the newest first
    return &clips_.front();
}

MediaClip* MediaLibrary::find(const QString& path) {
    const QString p = clean(path);
    for (auto& c : clips_)
        if (c.path == p) return &c;
    return nullptr;
}

const MediaClip* MediaLibrary::find(const QString& path) const { return const_cast<MediaLibrary*>(this)->find(path); }

bool MediaLibrary::remove(const QString& path) {
    const QString p = clean(path);
    const auto it = std::find_if(clips_.begin(), clips_.end(), [&](const MediaClip& c) { return c.path == p; });
    if (it == clips_.end()) return false;
    clips_.erase(it);
    return true;
}

QString MediaLibrary::create_bin(const QString& wanted) {
    QString base = wanted.trimmed().isEmpty() ? QStringLiteral("Bin") : wanted.trimmed();
    QString name = base;
    for (int i = 2; bins_.contains(name, Qt::CaseInsensitive) || name == kAll || name.startsWith("smart:"); ++i)
        name = base + QStringLiteral(" %1").arg(i);
    bins_ << name;
    return name;
}

bool MediaLibrary::rename_bin(const QString& from, const QString& to) {
    const QString t = to.trimmed();
    if (!bins_.contains(from) || t.isEmpty() || (t != from && bins_.contains(t, Qt::CaseInsensitive))) return false;
    bins_[bins_.indexOf(from)] = t;
    for (auto& c : clips_)
        for (auto& b : c.bins)
            if (b == from) b = t;
    return true;
}

bool MediaLibrary::delete_bin(const QString& name) {
    if (!bins_.removeOne(name)) return false;
    for (auto& c : clips_) c.bins.removeAll(name);
    return true;
}

bool MediaLibrary::file(const QString& path, const QString& bin) {
    MediaClip* c = find(path);
    if (!c || !bins_.contains(bin) || c->bins.contains(bin)) return false;
    c->bins << bin;
    return true;
}

bool MediaLibrary::unfile(const QString& path, const QString& bin) {
    MediaClip* c = find(path);
    return c && c->bins.removeAll(bin) > 0;
}

std::vector<const MediaClip*> MediaLibrary::in(const QString& bin, const QString& search) const {
    std::vector<const MediaClip*> out;
    for (const auto& c : clips_) {
        const bool member = bin == kAll || (bin == kClipped && c.clipped) || (bin == kCrushed && c.crushed) ||
                            (bin == kDelivered && c.delivered) || c.bins.contains(bin);
        if (member && (search.isEmpty() || c.name().contains(search, Qt::CaseInsensitive))) out.push_back(&c);
    }
    return out;
}

QJsonObject MediaLibrary::to_json() const {
    QJsonArray clips;
    for (const auto& c : clips_) {
        QJsonObject o{{"path", c.path}, {"kind", c.kind}};
        if (!c.bins.isEmpty()) o["bins"] = QJsonArray::fromStringList(c.bins);
        if (c.clipped) o["clipped"] = true;
        if (c.crushed) o["crushed"] = true;
        if (c.delivered) o["delivered"] = true;
        if (!c.info.isEmpty()) o["info"] = c.info;
        if (c.scanned()) {
            o["clipped_pct"] = c.clipped_pct;
            o["crushed_pct"] = c.crushed_pct;
        }
        if (c.has_settings()) {
            o["versions"] = c.versions;
            o["version"] = c.version;
        }
        clips.append(o);
    }
    return QJsonObject{{"bins", QJsonArray::fromStringList(bins_)}, {"clips", clips}};
}

void MediaLibrary::from_json(const QJsonObject& j) {
    clear();
    for (const auto& b : j.value("bins").toArray())
        if (const QString n = b.toString().trimmed(); !n.isEmpty() && !bins_.contains(n)) bins_ << n;
    for (const auto& v : j.value("clips").toArray()) {
        const QJsonObject o = v.toObject();
        const QString p = o.value("path").toString();
        if (p.isEmpty() || find(p)) continue;
        MediaClip c;
        c.path = clean(p);
        c.kind = o.value("kind").toString(kind_of(p));
        for (const auto& b : o.value("bins").toArray())
            if (bins_.contains(b.toString())) c.bins << b.toString();
        c.clipped = o.value("clipped").toBool();
        c.crushed = o.value("crushed").toBool();
        c.delivered = o.value("delivered").toBool();
        c.info = o.value("info").toString();
        c.clipped_pct = o.value("clipped_pct").toDouble(-1.0);
        c.crushed_pct = o.value("crushed_pct").toDouble(-1.0);
        c.versions = o.value("versions").toArray();
        c.version = c.versions.isEmpty() ? -1 : std::clamp(o.value("version").toInt(0), 0, int(c.versions.size()) - 1);
        clips_.push_back(c);
    }
}

void MediaLibrary::clear() {
    clips_.clear();
    bins_.clear();
}

// ---- ClipList ----------------------------------------------------------------------------------

ClipList::ClipList(QWidget* parent) : QListWidget(parent) {
    setObjectName("mediaClips");
    setDragEnabled(true);
    setDragDropMode(QAbstractItemView::DragOnly);
    setDefaultDropAction(Qt::CopyAction);
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setMovement(QListView::Static);
    setResizeMode(QListView::Adjust);
    setUniformItemSizes(true);
    setWordWrap(true);
    setFrameShape(QFrame::NoFrame);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void ClipList::startDrag(Qt::DropActions actions) {
    QStringList paths;
    for (auto* it : selectedItems()) paths << it->data(Qt::UserRole).toString();
    if (dragging) dragging(paths);
    QListWidget::startDrag(actions);   // returns when the drop is done
    if (dragging) dragging({});
}

void ClipList::resizeEvent(QResizeEvent* e) {
    QListWidget::resizeEvent(e);
    fit_grid();
}

void ClipList::fit_grid() {
    if (viewMode() != QListView::IconMode) return;
    // Two to a row at the rail's width (240 px and up); one more for every 130 px past it.
    // The width with the scroll bar's room taken off always: when the bar comes, the cells still fit.
    const int w = std::max(80, width() - 2 * frameWidth() - style()->pixelMetric(QStyle::PM_ScrollBarExtent) - 2);
    const int cols = std::max(2, w / 130);
    const int cell = w / cols;
    const QSize icon(cell - 10, int((cell - 10) * 0.56));
    if (iconSize() != icon) setIconSize(icon);
    const QSize grid(cell, icon.height() + 42);
    if (gridSize() != grid) {
        setGridSize(grid);
        for (int i = 0; i < count(); ++i) item(i)->setSizeHint(grid);
    }
}

QStringList ClipList::mimeTypes() const { return {QString::fromLatin1(kMime), QStringLiteral("text/uri-list")}; }

QMimeData* ClipList::mimeData(const QList<QListWidgetItem*>& items) const {
    auto* m = new QMimeData;
    QList<QUrl> urls;
    QStringList paths;
    for (auto* it : items) {
        const QString p = it->data(Qt::UserRole).toString();
        paths << p;
        urls << QUrl::fromLocalFile(p);
    }
    m->setData(QString::fromLatin1(kMime), paths.join(QLatin1Char('\n')).toUtf8());
    m->setUrls(urls);   // and as files: another app, or the window's own drop, takes them too
    return m;
}

// ---- MediaPanel --------------------------------------------------------------------------------

MediaPanel::MediaPanel(MediaLibrary* library, QWidget* parent) : QWidget(parent), lib_(library) {
    setObjectName("mediaPanel");
    setAcceptDrops(true);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(6);

    auto* head = new QWidget(this);
    auto* hh = new QHBoxLayout(head);
    hh->setContentsMargins(4, 0, 4, 0);
    hh->setSpacing(4);
    auto* title = new QLabel(QStringLiteral("Media"), head);
    title->setProperty("role", "media-title");
    hh->addWidget(title, 1);
    auto tool = [head, hh](const char* id, const QString& text, const QString& tip, bool checkable) {
        auto* b = new QPushButton(text, head);
        b->setObjectName(QString::fromLatin1(id));
        b->setToolTip(tip);
        b->setCheckable(checkable);
        b->setProperty("role", "media-tool");
        b->setFixedSize(28, 26);
        b->setFocusPolicy(Qt::NoFocus);
        hh->addWidget(b);
        return b;
    };
    auto* add = tool("mediaImport", QStringLiteral("+"), QStringLiteral("Import files or a folder of frames"), false);
    connect(add, &QPushButton::clicked, this, [this, add] {
        QMenu m(this);
        m.addAction(QStringLiteral("Import files…"), this, [this] { if (import_files) import_files(); });
        m.addAction(QStringLiteral("Import a folder of frames…"), this, [this] { if (import_folder) import_folder(); });
        m.exec(add->mapToGlobal(QPoint(0, add->height())));
    });
    grid_btn_ = tool("mediaGrid", QStringLiteral("▦"), QStringLiteral("Thumbnails"), true);
    list_btn_ = tool("mediaList", QStringLiteral("☰"), QStringLiteral("List"), true);
    connect(grid_btn_, &QPushButton::clicked, this, [this] { set_list_mode(false); });
    connect(list_btn_, &QPushButton::clicked, this, [this] { set_list_mode(true); });
    v->addWidget(head);

    search_ = new QLineEdit(this);
    search_->setObjectName("mediaSearch");
    search_->setPlaceholderText(QStringLiteral("Search media"));
    search_->setClearButtonEnabled(true);
    connect(search_, &QLineEdit::textChanged, this, [this] { refresh(); });
    v->addWidget(search_);

    bins_box_ = new QWidget(this);
    bins_box_->setObjectName("mediaBins");
    bins_v_ = new QVBoxLayout(bins_box_);
    bins_v_->setContentsMargins(0, 0, 0, 0);
    bins_v_->setSpacing(1);
    v->addWidget(bins_box_);

    auto* ih = new QWidget(this);
    auto* ihh = new QHBoxLayout(ih);
    ihh->setContentsMargins(4, 6, 4, 0);
    bin_title_ = new QLabel(this);
    bin_title_->setObjectName("mediaBinTitle");
    bin_title_->setProperty("role", "media-bin-title");
    bin_count_ = new QLabel(this);
    bin_count_->setObjectName("mediaBinCount");
    bin_count_->setProperty("role", "note");
    ihh->addWidget(bin_title_, 1);
    ihh->addWidget(bin_count_);
    v->addWidget(ih);

    list_view_ = new ClipList(this);
    list_view_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(list_view_, &QWidget::customContextMenuRequested, this, [this](const QPoint& at) { context_menu(at); });
    connect(list_view_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* it) {
        if (open) open({it->data(Qt::UserRole).toString()});
    });
    v->addWidget(list_view_, 1);

    auto* hint = new QLabel(QStringLiteral("Drag a clip onto the viewer to open it. Drop files here to import."), this);
    hint->setObjectName("mediaHint");
    hint->setProperty("role", "note");
    hint->setWordWrap(true);
    v->addWidget(hint);

    set_list_mode(false);
    rebuild_bins();
}

void MediaPanel::set_list_mode(bool list) {
    list_ = list;
    grid_btn_->setChecked(!list);
    list_btn_->setChecked(list);
    if (list) {
        list_view_->setViewMode(QListView::ListMode);
        list_view_->setIconSize(QSize(64, 40));
        list_view_->setGridSize(QSize());
        list_view_->setSpacing(2);
    } else {
        list_view_->setViewMode(QListView::IconMode);
        list_view_->setSpacing(0);
        list_view_->fit_grid();
    }
    list_view_->setDragEnabled(true);   // IconMode resets the drag settings
    list_view_->setDragDropMode(QAbstractItemView::DragOnly);
    list_view_->setMovement(QListView::Static);
    refresh();
}

void MediaPanel::set_open_path(const QString& path) {
    const QString p = path.isEmpty() ? QString() : MediaLibrary::clean(path);
    if (p != open_path_) list_view_->clearSelection();   // another shot opened: the selection follows it
    open_path_ = p;
    refresh();
}

void MediaPanel::set_thumb(const QString& path, const QImage& thumb) {
    if (thumb.isNull()) return;
    const QString p = MediaLibrary::clean(path);
    for (auto& [k, img] : thumbs_)
        if (k == p) {
            img = thumb;
            refresh();
            return;
        }
    thumbs_.emplace_back(p, thumb);
    QDir().mkpath(thumb_dir());
    thumb.save(thumb_file(p));   // the next session has it before the shot is opened
    refresh();
}

QImage MediaPanel::thumb(const QString& path) const {
    const QString p = MediaLibrary::clean(path);
    for (const auto& [k, img] : thumbs_)
        if (k == p) return img;
    QImage img(thumb_file(p));
    QString still = QFileInfo(p).isFile() && MediaLibrary::kind_of(p) == "still" ? p : QString();
    if (img.isNull() && still.isEmpty() && QFileInfo(p).isDir()) {   // a folder of frames: its first frame
        QStringList filters;
        for (const auto& s : frame_suffixes()) filters << QStringLiteral("*") + QString::fromStdString(s);
        const QStringList first = QDir(p).entryList(filters, QDir::Files, QDir::Name);
        if (!first.isEmpty()) still = QDir(p).filePath(first.front());
    }
    if (img.isNull() && !still.isEmpty()) {
        QImageReader r(still);   // a still never opened: read small (JPEG decodes at a fraction of its size)
        r.setAutoTransform(true);
        const QSize s = r.size();
        if (s.isValid()) r.setScaledSize(s.scaled(176, 100, Qt::KeepAspectRatioByExpanding));
        img = r.read();
    }
    thumbs_.emplace_back(p, img);   // a miss is remembered too (a movie before it is opened)
    return img;
}

QIcon MediaPanel::icon_for(const MediaClip& c) const {
    const QSize sz = list_ ? QSize(64, 40) : list_view_->iconSize();
    const qreal dpr = qApp ? qApp->devicePixelRatio() : 1.0;
    QPixmap pm(sz * dpr);
    pm.setDevicePixelRatio(dpr);
    pm.fill(QColor(0x2a, 0x2a, 0x2a));
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);
    QPainterPath clip;
    clip.addRoundedRect(QRectF(0, 0, sz.width(), sz.height()), 4, 4);
    p.setClipPath(clip);
    const QImage t = thumb(c.path);
    if (!t.isNull()) {
        const QImage s = t.scaled(sz, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        p.drawImage(QPoint((sz.width() - s.width()) / 2, (sz.height() - s.height()) / 2), s);
    } else {
        p.setPen(QColor(0x8e, 0x8e, 0x8e));
        p.drawText(QRect(QPoint(0, 0), sz), Qt::AlignCenter, c.kind == "movie" ? QStringLiteral("▶") : QStringLiteral("□"));
    }
    p.setClipping(false);
    // The type, top left.
    QFont f = p.font();
    f.setPixelSize(list_ ? 8 : 9);
    f.setBold(true);
    p.setFont(f);
    const QString b = c.badge();
    const QRect br = QFontMetrics(f).boundingRect(b).adjusted(-4, -2, 4, 2);
    const QRect at(4, 4, br.width(), br.height());
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 170));
    p.drawRoundedRect(at, 3, 3);
    p.setPen(QColor(0xdd, 0xdd, 0xdd));
    p.drawText(at, Qt::AlignCenter, b);
    // The smart bins it is in, top right.
    int x = sz.width() - 10;
    for (auto [on, key] : {std::pair{c.delivered, MediaLibrary::kDelivered}, std::pair{c.crushed, MediaLibrary::kCrushed},
                           std::pair{c.clipped, MediaLibrary::kClipped}})
        if (on) {
            p.setPen(QPen(QColor(0, 0, 0, 140), 1));
            p.setBrush(smart_dot(key));
            p.drawEllipse(QPointF(x, 9), 3.5, 3.5);
            x -= 10;
        }
    // Its own settings: the version in use, bottom left.
    if (c.has_settings()) {
        const QString vn = c.version_name();
        const QRect vr = QFontMetrics(f).boundingRect(vn).adjusted(-4, -2, 4, 2);
        const QRect vat(4, sz.height() - vr.height() - 4, vr.width(), vr.height());
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0x40, 0x9c, 0xff, 220));
        p.drawRoundedRect(vat, 3, 3);
        p.setPen(Qt::white);
        p.drawText(vat, Qt::AlignCenter, vn);
    }
    // The shot on screen: a gold frame.
    if (c.path == open_path_) {
        p.setPen(QPen(QColor(0xe8, 0xb0, 0x7a), 2));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(1, 1, sz.width() - 2, sz.height() - 2), 4, 4);
    }
    QIcon icon;   // the same picture selected: Qt would tint it with the selection colour
    icon.addPixmap(pm, QIcon::Normal);
    icon.addPixmap(pm, QIcon::Selected);
    return icon;
}

void MediaPanel::refresh() {
    const QStringList keep = list_view_ ? selected() : QStringList();
    const auto shown = lib_->in(bin_, search_ ? search_->text().trimmed() : QString());
    list_view_->clear();
    for (const MediaClip* c : shown) {
        const QString sub = !c->info.isEmpty() ? c->info
                            : c->kind == "folder" ? QStringLiteral("Folder of frames")
                            : c->kind == "movie"  ? QStringLiteral("Movie")
                                                  : QStringLiteral("Still");
        auto* it = new QListWidgetItem(icon_for(*c), c->name() + QLatin1Char('\n') + sub, list_view_);
        it->setData(Qt::UserRole, c->path);
        QString tip = QDir::toNativeSeparators(c->path);
        if (c->scanned())   // 1.2.6: what the scan on import found
            tip += QStringLiteral("\nClipped highlights %1 % \u00b7 crushed shadows %2 %")
                       .arg(c->clipped_pct, 0, 'f', c->clipped_pct < 1.0 ? 2 : 1)
                       .arg(c->crushed_pct, 0, 'f', c->crushed_pct < 1.0 ? 2 : 1);
        it->setToolTip(tip + QStringLiteral("\nDrag onto the viewer, or double-click, to open"));
        if (!list_) it->setSizeHint(list_view_->gridSize());
        if (keep.isEmpty() ? c->path == open_path_ : keep.contains(c->path)) it->setSelected(true);
    }
    const QString name = bin_ == MediaLibrary::kAll       ? QStringLiteral("All media")
                         : bin_ == MediaLibrary::kClipped ? QStringLiteral("Clipped highlights")
                         : bin_ == MediaLibrary::kCrushed ? QStringLiteral("Crushed shadows")
                         : bin_ == MediaLibrary::kDelivered ? QStringLiteral("Delivered")
                                                            : bin_;
    bin_title_->setText(name);
    bin_count_->setText(QStringLiteral("%1 %2").arg(shown.size()).arg(shown.size() == 1 ? "clip" : "clips"));
    rebuild_bins();
}

void MediaPanel::show_bin(const QString& bin) {
    bin_ = bin == MediaLibrary::kAll || bin.startsWith("smart:") || lib_->bins().contains(bin) ? bin : QString(MediaLibrary::kAll);
    refresh();
}

void MediaPanel::rebuild_bins() {
    while (QLayoutItem* it = bins_v_->takeAt(0)) {
        if (it->widget()) it->widget()->deleteLater();
        delete it;
    }
    auto section = [this](const QString& text, bool add) {
        auto* w = new QWidget(bins_box_);
        auto* h = new QHBoxLayout(w);
        h->setContentsMargins(6, 6, 4, 2);
        auto* l = new QLabel(text, w);
        l->setProperty("role", "media-section");
        h->addWidget(l, 1);
        if (add) {
            auto* b = new QPushButton(QStringLiteral("+"), w);
            b->setObjectName("mediaNewBin");
            b->setProperty("role", "media-add");
            b->setToolTip(QStringLiteral("New bin"));
            b->setFixedSize(22, 20);
            b->setFocusPolicy(Qt::NoFocus);
            connect(b, &QPushButton::clicked, this, [this] {
                bool ok = false;
                const QString n = QInputDialog::getText(this, QStringLiteral("New bin"), QStringLiteral("Name"), QLineEdit::Normal,
                                                        QStringLiteral("Bin"), &ok);
                if (!ok) return;
                show_bin(lib_->create_bin(n));
                if (changed) changed();
            });
            h->addWidget(b);
        }
        bins_v_->addWidget(w);
    };
    auto row = [this](const QString& key, const QString& text, std::optional<QColor> dot) {
        auto* r = new QWidget(bins_box_);
        r->setObjectName(QStringLiteral("bin:") + key);
        r->setProperty("role", "media-bin");
        r->setProperty("sel", key == bin_);
        r->setAttribute(Qt::WA_StyledBackground, true);
        r->setCursor(Qt::PointingHandCursor);
        r->setProperty("bin", key);
        auto* h = new QHBoxLayout(r);
        h->setContentsMargins(8, 4, 8, 4);
        h->setSpacing(8);
        auto* d = new QLabel(dot ? QString() : QStringLiteral("▢"), r);
        d->setFixedWidth(10);
        if (dot) {
            d->setFixedSize(8, 8);
            d->setStyleSheet(QStringLiteral("background:%1;border-radius:4px;").arg(dot->name()));
        } else {
            d->setProperty("role", "media-folder");
        }
        h->addWidget(d, 0, Qt::AlignVCenter);
        h->addWidget(new QLabel(text, r), 1);
        auto* n = new QLabel(QString::number(lib_->count(key)), r);
        n->setProperty("role", "note");
        h->addWidget(n);
        r->installEventFilter(this);
        r->setAcceptDrops(!key.startsWith("smart:"));   // a clip can be filed in a bin (All media holds it already)
        bins_v_->addWidget(r);
    };
    section(QStringLiteral("BINS"), true);
    row(MediaLibrary::kAll, QStringLiteral("All media"), std::nullopt);
    for (const QString& b : lib_->bins()) row(b, b, std::nullopt);
    section(QStringLiteral("SMART BINS"), false);
    row(MediaLibrary::kClipped, QStringLiteral("Clipped highlights"), smart_dot(MediaLibrary::kClipped));
    row(MediaLibrary::kCrushed, QStringLiteral("Crushed shadows"), smart_dot(MediaLibrary::kCrushed));
    row(MediaLibrary::kDelivered, QStringLiteral("Delivered"), smart_dot(MediaLibrary::kDelivered));    // 1.2.6: what each smart bin holds, on hover (each clip is scanned when it is imported)
    const std::pair<const char*, const char*> tips[] = {
        {MediaLibrary::kClipped, "Clips with blown highlights (0.05 % of the picture or more at code 254+): "
                                 "the shots RUDRA and the AI fill have work in. Found when a clip is imported"},
        {MediaLibrary::kCrushed, "Clips with crushed blacks (0.5 % or more at code 0-1, black bars left out)"},
        {MediaLibrary::kDelivered, "Clips you have rendered a master or a movie of"}};
    for (QWidget* r : findChildren<QWidget*>())
        for (const auto& [key, tip] : tips)
            if (r->property("bin").toString() == QLatin1String(key)) r->setToolTip(QString::fromUtf8(tip));
}

bool MediaPanel::eventFilter(QObject* o, QEvent* e) {
    auto* r = qobject_cast<QWidget*>(o);
    const QString key = r ? r->property("bin").toString() : QString();
    if (key.isEmpty()) return QWidget::eventFilter(o, e);
    auto hot = [r](bool on) {
        r->setProperty("drop", on);
        r->style()->unpolish(r);
        r->style()->polish(r);
    };
    switch (e->type()) {
    case QEvent::MouseButtonRelease: {
        auto* me = static_cast<QMouseEvent*>(e);
        if (me->button() == Qt::LeftButton) {
            const QString k = key;
            QMetaObject::invokeMethod(this, [this, k] { show_bin(k); }, Qt::QueuedConnection);   // the row is rebuilt
        } else if (me->button() == Qt::RightButton) {
            bin_menu(key, me->globalPosition().toPoint());
        }
        return true;
    }
    case QEvent::DragEnter: {
        auto* de = static_cast<QDragEnterEvent*>(e);
        if (lib_->bins().contains(key) && de->mimeData()->hasFormat(ClipList::kMime)) {
            de->acceptProposedAction();
            hot(true);
            return true;
        }
        return false;
    }
    case QEvent::DragLeave:
        hot(false);
        return true;
    case QEvent::Drop: {
        auto* de = static_cast<QDropEvent*>(e);
        hot(false);
        if (!lib_->bins().contains(key)) return false;
        bool any = false;
        for (const QString& p : QString::fromUtf8(de->mimeData()->data(ClipList::kMime)).split('\n', Qt::SkipEmptyParts))
            any = lib_->file(p, key) || any;
        de->acceptProposedAction();
        if (any) {
            QMetaObject::invokeMethod(this, [this] { refresh(); }, Qt::QueuedConnection);
            if (changed) changed();
        }
        return true;
    }
    default:
        return QWidget::eventFilter(o, e);
    }
}

void MediaPanel::bin_menu(const QString& bin, const QPoint& global) {
    QMenu m(this);
    if (bin_settings_menu) {
        bin_settings_menu(&m, bin);
        m.addSeparator();
    }
    if (!lib_->bins().contains(bin)) {   // All media or a smart bin: only what applies to its clips
        if (!m.isEmpty()) m.exec(global);
        return;
    }
    m.addAction(QStringLiteral("Rename bin…"), this, [this, bin] {
        bool ok = false;
        const QString n = QInputDialog::getText(this, QStringLiteral("Rename bin"), QStringLiteral("Name"), QLineEdit::Normal, bin, &ok);
        if (ok && lib_->rename_bin(bin, n)) {
            if (bin_ == bin) bin_ = n.trimmed();
            refresh();
            if (changed) changed();
        }
    });
    m.addAction(QStringLiteral("Delete bin (the clips stay in All media)"), this, [this, bin] {
        if (lib_->delete_bin(bin)) {
            if (bin_ == bin) bin_ = MediaLibrary::kAll;
            refresh();
            if (changed) changed();
        }
    });
    m.exec(global);
}

QStringList MediaPanel::selected() const {
    QStringList out;
    for (int i = 0; i < list_view_->count(); ++i)
        if (list_view_->item(i)->isSelected()) out << list_view_->item(i)->data(Qt::UserRole).toString();
    return out;
}

void MediaPanel::select(const QStringList& paths) {
    QStringList clean;
    for (const QString& p : paths) clean << MediaLibrary::clean(p);
    for (int i = 0; i < list_view_->count(); ++i)
        list_view_->item(i)->setSelected(clean.contains(list_view_->item(i)->data(Qt::UserRole).toString()));
}

void MediaPanel::context_menu(const QPoint& at) {
    QListWidgetItem* it = list_view_->itemAt(at);
    if (!it) return;
    if (!it->isSelected()) {   // a right click on a clip outside the selection: that clip alone
        list_view_->clearSelection();
        it->setSelected(true);
    }
    const QStringList sel = selected();
    const QString p = it->data(Qt::UserRole).toString();
    const QString n = sel.size() > 1 ? QStringLiteral(" (%1 clips)").arg(sel.size()) : QString();
    QMenu m(this);
    m.addAction(QStringLiteral("Open in viewer") + n, this, [this, sel] { if (open) open(sel); });
    if (batch) m.addAction(QStringLiteral("Add to batch queue") + n, this, [this, sel] { batch(sel); });
    if (settings_menu) settings_menu(m.addMenu(QStringLiteral("Settings")), sel);
    if (!lib_->bins().isEmpty()) {
        QMenu* mv = m.addMenu(QStringLiteral("Add to bin"));
        for (const QString& b : lib_->bins())
            mv->addAction(b, this, [this, sel, b] {
                bool any = false;
                for (const QString& c : sel) any = lib_->file(c, b) || any;
                if (any) {
                    refresh();
                    if (changed) changed();
                }
            });
    }
    m.addSeparator();
    m.addAction(QStringLiteral("Reveal in folder"), this, [p] {
        const QFileInfo fi(p);
        QUrl u = QUrl::fromLocalFile(fi.isDir() ? fi.absoluteFilePath() : fi.absolutePath());
        QMetaObject::invokeMethod(qApp, [u] { QDesktopServices::openUrl(u); }, Qt::QueuedConnection);
    });
    if (lib_->bins().contains(bin_))
        m.addAction(QStringLiteral("Remove from this bin") + n, this, [this, sel] {
            bool any = false;
            for (const QString& c : sel) any = lib_->unfile(c, bin_) || any;
            if (any) {
                refresh();
                if (changed) changed();
            }
        });
    m.addAction(QStringLiteral("Remove from Media (the files stay on disk)") + n, this, [this, sel] {
        bool any = false;
        for (const QString& c : sel) any = lib_->remove(c) || any;
        if (any) {
            refresh();
            if (changed) changed();
        }
    });
    m.exec(list_view_->viewport()->mapToGlobal(at));
}

void MediaPanel::dragEnterEvent(QDragEnterEvent* e) {
    // Files from the computer are imported here; a clip of the panel's own is not re-imported.
    if (e->mimeData()->hasFormat(ClipList::kMime) || !e->mimeData()->hasUrls()) return;
    const auto urls = e->mimeData()->urls();
    if (std::any_of(urls.begin(), urls.end(), [](const QUrl& u) { return u.isLocalFile(); })) e->acceptProposedAction();
}

void MediaPanel::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const auto& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    if (paths.isEmpty()) return;
    e->acceptProposedAction();
    if (imported) imported(paths);
}

}  // namespace rudra::app
