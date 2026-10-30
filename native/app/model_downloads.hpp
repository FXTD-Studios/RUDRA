#pragma once
// Model downloads (29 Oct 2026): the models RUDRA can fetch, listed by an
// index we publish (models.json on www.fxtdstudios.com; the files themselves on
// Hugging Face), downloaded into the user's models folder, resumed when they
// stop, and checked file by file against their SHA-256 before anything uses
// them. A model is installed only when every file is there and verified: a
// failed or cancelled download leaves `.part` files, never a half model.
//
// The index:
//   {"version": 1, "models": [{
//      "id": "aces_v2_gp600", "title": "Learned ACES 2.0", "version": "2.0",
//      "channel": "preview" | "release" | "legacy", "kind": "aces_v2" | "package" | "genfill",
//      "summary": "...", "licence": "...", "licence_url": "https://...",
//      "folder": "aces_v2_guarded_projected600", "min_app": "1.0.0-rc.2",
//      "files": [{"path": "manifest.json", "url": "https://...", "size": 1234, "sha256": "..."}]}]}
// No Q_OBJECT (as elsewhere in the app): callbacks are std::function.

#include <QByteArray>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

class QFile;
class QNetworkAccessManager;
#include <QNetworkReply>
class QObject;

namespace rudra::app {

struct RemoteFile {
    QString path;      // inside the model's folder, '/'-separated, no ".."
    QString url;
    qint64 size = 0;
    QString sha256;    // lowercase hex
};

struct RemoteModel {
    QString id, title, version, channel, kind, summary, licence, licence_url, folder, min_app;
    std::vector<RemoteFile> files;
    qint64 size() const;
};

// The index's models; nullopt and `why` when the text is not one (a model with
// an unsafe path, no files or a bad checksum is refused with the rest).
std::optional<std::vector<RemoteModel>> parse_model_index(const QByteArray& json, QString* why = nullptr);

QString default_model_index_url();   // QSettings models/indexUrl, else https://www.fxtdstudios.com/rudra/models.json
QString default_download_root();     // QSettings models/downloadRoot, else <AppLocalData>/models

// "12.4 MB", "2.1 GB".
QString human_size(qint64 bytes);

enum class InstallState { NotInstalled, Partial, Installed };
// Installed: the folder's marker (.rudra-download.json) names this id and version
// and every file is there at its size. Partial: some of it (.part files included).
InstallState install_state(const RemoteModel& m, const QString& root);

class ModelDownloads {
public:
    ModelDownloads(QNetworkAccessManager* net, QObject* owner);
    ~ModelDownloads();

    // The index: fetched now, models() and index_status() follow.
    void fetch_index(const QString& url = default_model_index_url());
    const std::vector<RemoteModel>& models() const { return models_; }
    const RemoteModel* model(const QString& id) const;
    QString index_status() const { return index_status_; }   // "" while fetching or when fine
    bool fetching() const { return index_reply_ != nullptr; }
    void set_models(std::vector<RemoteModel> m);              // tests

    QString root() const { return root_; }
    void set_root(const QString& root);                       // remembered (models/downloadRoot)
    InstallState state(const QString& id) const;

    // One download at a time. false (and why) when another runs or the id is unknown.
    bool download(const QString& id, QString* why = nullptr);
    void pause();    // stops, keeping what came (.part): download() again resumes
    void cancel(const QString& id = {});   // stops (the active one) and deletes that model's .part files
    QString active() const { return active_; }               // the id downloading, or ""
    qint64 done_bytes() const { return done_; }
    qint64 total_bytes() const { return total_; }
    double bytes_per_second() const { return rate_; }
    QString error(const QString& id) const;                  // the last failure for that model

    // Called on every change (progress included) while `owner` lives.
    void listen(QObject* owner, std::function<void()> on_change);
    // A model verified and installed: its folder.
    std::function<void(const RemoteModel&, const QString& folder)> installed;

private:
    void changed();
    void next_file();
    void file_finished();
    void fail(const QString& why);
    void stop_reply();

    QNetworkAccessManager* net_;
    QObject* alive_;
    QString root_;
    std::vector<RemoteModel> models_;
    QString index_status_;
    QPointer<QNetworkReply> index_reply_;
    std::map<int, std::pair<QPointer<QObject>, std::function<void()>>> listeners_;
    int next_listener_ = 0;
    std::map<QString, QString> errors_;
    // The download in progress.
    QString active_;
    std::size_t file_ = 0;
    qint64 done_ = 0, total_ = 0, file_start_ = 0;
    double rate_ = 0;
    qint64 rate_mark_bytes_ = 0;
    qint64 rate_mark_ms_ = 0;
    bool ranged_ = false, checked_status_ = false;
    QPointer<QNetworkReply> reply_;
    std::unique_ptr<QFile> part_;
};

}  // namespace rudra::app
