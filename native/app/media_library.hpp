#pragma once
// The Media panel (1.2.6, 18 Nov 2026, user feedback: "the left panel for shots should be a bin
// and media section, and we can drag and drop from there to the viewer").
//
// MediaLibrary: every shot this project knows (stills, folders of frames, movies), the bins the
// artist made, and three smart bins that fill themselves (clipped highlights, crushed shadows,
// delivered). Saved with the project (Project::media) and, between projects, in the settings.
// MediaPanel: the left rail's Media section: search, thumbnails or a list, bins, smart bins.
// A clip drags out as its file (text/uri-list) plus its own format, so the viewer, the window
// and a bin all take the same drag: the viewer opens it, a bin files it.

#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QListWidget>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <vector>

class QLabel;
class QMenu;
class QLineEdit;
class QPushButton;
class QVBoxLayout;

namespace rudra::app {

struct MediaClip {
    QString path;                 // clean, absolute: a still, a folder of frames or a movie
    QString kind;                 // "still", "folder" or "movie"
    QStringList bins;             // the artist's bins it is filed in
    bool clipped = false;         // smart bins: what RUDRA measured when it was open
    bool crushed = false;
    bool delivered = false;
    // 1.2.6: what the scan on import found (media_scan.hpp): its worst frame's clipped and crushed
    // shares, in per cent; -1 until it is scanned.
    double clipped_pct = -1.0, crushed_pct = -1.0;
    bool scanned() const { return clipped_pct >= 0.0; }
    QString info;                 // "1088×745", "120 f · 1280×720"; empty until opened
    // 1.2.6: the clip's own settings (the grade, the view peak, the Deliver settings), in versions
    // ({"name": "v2", "settings": {...}}); `version` is the one in use, -1 for none (the session's).
    QJsonArray versions;
    int version = -1;
    bool has_settings() const { return version >= 0 && version < versions.size(); }
    QJsonObject settings() const;              // the version in use; {} for none
    void set_settings(const QJsonObject& s);   // into the version in use (v1 when there is none)
    int add_version(const QJsonObject& s);     // a new version, now the one in use; its index
    QString version_name() const;              // "v2", "" for none
    QString name() const;
    QString badge() const;        // "JPG", "PNG SEQ", "MOV", …
};

class MediaLibrary {
public:
    // The smart bins' keys; "all" is every clip.
    static constexpr const char* kAll = "all";
    static constexpr const char* kClipped = "smart:clipped";
    static constexpr const char* kCrushed = "smart:crushed";
    static constexpr const char* kDelivered = "smart:delivered";

    static QString kind_of(const QString& path);   // "still", "folder" or "movie" (by the file system)
    static QString clean(const QString& path);

    // Adds (or finds) a clip; a .rudra project is not media and is refused (nullptr).
    MediaClip* add(const QString& path);
    MediaClip* find(const QString& path);
    const MediaClip* find(const QString& path) const;
    bool remove(const QString& path);   // off the library and its bins; nothing on disk is touched
    const std::vector<MediaClip>& clips() const { return clips_; }

    const QStringList& bins() const { return bins_; }
    QString create_bin(const QString& wanted);   // a unique name ("Bin", "Bin 2", …), returned
    bool rename_bin(const QString& from, const QString& to);
    bool delete_bin(const QString& name);        // its clips stay in All media
    bool file(const QString& path, const QString& bin);     // into a bin (no duplicate)
    bool unfile(const QString& path, const QString& bin);

    // The clips a bin shows (a user bin, a smart bin or kAll), filtered by `search` (in the name, any case).
    std::vector<const MediaClip*> in(const QString& bin, const QString& search = {}) const;
    int count(const QString& bin) const { return int(in(bin).size()); }

    QJsonObject to_json() const;
    void from_json(const QJsonObject& j);   // replaces everything
    void clear();

private:
    std::vector<MediaClip> clips_;
    QStringList bins_;
};

// The list of clips: drags its items out as files.
class ClipList : public QListWidget {
public:
    explicit ClipList(QWidget* parent = nullptr);
    static constexpr const char* kMime = "application/x-rudra-clip";
    // Told when a drag of clips starts (their paths) and ends (empty): the window marks the viewer
    // as the place to drop, and its drop opens the clip as a shot of its own.
    std::function<void(const QStringList& paths)> dragging;

    void fit_grid();   // thumbnails sized to the rail's width

protected:
    void startDrag(Qt::DropActions actions) override;
    void resizeEvent(QResizeEvent* e) override;
    QMimeData* mimeData(const QList<QListWidgetItem*>& items) const override;
    QStringList mimeTypes() const override;
};

class MediaPanel : public QWidget {
public:
    MediaPanel(MediaLibrary* library, QWidget* parent = nullptr);

    // What the window does for the panel.
    std::function<void(const QStringList& paths)> open;       // into the viewer (several stills: one shot of them)
    std::function<void()> import_files, import_folder;        // the + menu
    std::function<void(const QStringList& paths)> imported;   // dropped from the computer onto the panel
    std::function<void(const QStringList& paths)> batch;      // Add to batch queue (no need to open them)
    std::function<void()> changed;                            // the library changed: save it
    // 1.2.6: the clip's settings submenu (versions, copy, paste, reset) and a bin's (paste to all).
    std::function<void(QMenu* menu, const QStringList& paths)> settings_menu;
    std::function<void(QMenu* menu, const QString& bin)> bin_settings_menu;

    void set_open_path(const QString& path);   // the shot on screen (a gold frame)
    void set_thumb(const QString& path, const QImage& thumb);
    QImage thumb(const QString& path) const;
    void refresh();
    QString bin() const { return bin_; }
    void show_bin(const QString& bin);
    void set_list_mode(bool list);
    bool list_mode() const { return list_; }
    ClipList* list() const { return list_view_; }
    QStringList selected() const;   // the selected clips' paths, in the panel's order
    void select(const QStringList& paths);

protected:
    void dragEnterEvent(class QDragEnterEvent* e) override;
    void dropEvent(class QDropEvent* e) override;
    bool eventFilter(QObject* o, QEvent* e) override;

private:
    void rebuild_bins();
    QIcon icon_for(const MediaClip& c) const;
    void context_menu(const QPoint& at);
    void bin_menu(const QString& bin, const QPoint& global);

    MediaLibrary* lib_;
    QString bin_ = MediaLibrary::kAll, open_path_;
    bool list_ = false;
    QLineEdit* search_ = nullptr;
    QWidget* bins_box_ = nullptr;
    QVBoxLayout* bins_v_ = nullptr;
    QLabel *bin_title_ = nullptr, *bin_count_ = nullptr;
    QPushButton *grid_btn_ = nullptr, *list_btn_ = nullptr;
    ClipList* list_view_ = nullptr;
    mutable std::vector<std::pair<QString, QImage>> thumbs_;
};

}  // namespace rudra::app
