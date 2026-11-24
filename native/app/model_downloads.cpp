#include "model_downloads.hpp"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSettings>
#include <QStandardPaths>

#include <algorithm>

namespace rudra::app {
namespace {

constexpr const char* kMarker = ".rudra-download.json";

// A path inside the model's folder: relative, '/'-separated, no "." or "..".
bool safe_relative(const QString& p) {
    if (p.isEmpty() || p.startsWith('/') || p.contains('\\') || p.contains(':')) return false;
    for (const auto& part : p.split('/'))
        if (part.isEmpty() || part == "." || part == "..") return false;
    return true;
}

QString join(const QString& a, const QString& b) { return QDir(a).filePath(b); }

// Is the file at `path` the one the list names: its size AND its SHA-256 (11 Nov 2026).
// Size alone was the test, and RUDRA ACES 2's 2.0 weights are byte for byte as long as 1.0's
// (rudra_g1.ts 1,037,772, rudra_g1.onnx 1,157,294). Updating the model in place fetched the new
// manifest and kept the old weights as "already here", so the package no longer checked against
// its own manifest ("The model file is damaged or was changed") while this list said "Installed
// and checked" and offered no way to download it again. Hashed once per file as written (path,
// size, time), so the dialog's refreshes cost nothing after the first.
bool file_checks(const QString& path, const RemoteFile& f) {
    const QFileInfo fi(path);
    if (!fi.isFile() || fi.size() != f.size) return false;
    static QHash<QString, QString> hashed;
    const QString key = fi.absoluteFilePath() + QLatin1Char('|') + QString::number(fi.size()) + QLatin1Char('|') +
                        QString::number(fi.lastModified().toMSecsSinceEpoch());
    QString sha = hashed.value(key);
    if (sha.isEmpty()) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return false;
        QCryptographicHash h(QCryptographicHash::Sha256);
        if (!h.addData(&file)) return false;
        sha = QString::fromLatin1(h.result().toHex());
        hashed.insert(key, sha);
    }
    return sha == f.sha256;
}

}  // namespace

qint64 RemoteModel::size() const {
    qint64 s = 0;
    for (const auto& f : files) s += f.size;
    return s;
}

QString human_size(qint64 b) {
    if (b < 1000) return QStringLiteral("%1 B").arg(b);
    if (b < 1000 * 1000) return QStringLiteral("%1 KB").arg(double(b) / 1e3, 0, 'f', 0);
    if (b < 1000LL * 1000 * 1000) return QStringLiteral("%1 MB").arg(double(b) / 1e6, 0, 'f', b < 10000000 ? 1 : 0);
    return QStringLiteral("%1 GB").arg(double(b) / 1e9, 0, 'f', 1);
}

std::optional<std::vector<RemoteModel>> parse_model_index(const QByteArray& json, QString* why) {
    auto say = [why](const QString& s) {
        if (why) *why = s;
        return std::nullopt;
    };
    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return say(QStringLiteral("the model list is not JSON"));
    const QJsonObject root = doc.object();
    if (root.value("version").toInt() != 1) return say(QStringLiteral("the model list is a newer kind: update RUDRA"));
    static const QRegularExpression hex(QStringLiteral("^[0-9a-f]{64}$"));
    std::vector<RemoteModel> out;
    for (const auto& v : root.value("models").toArray()) {
        const QJsonObject o = v.toObject();
        RemoteModel m;
        m.id = o.value("id").toString();
        m.title = o.value("title").toString(m.id);
        m.version = o.value("version").toString();
        m.channel = o.value("channel").toString("release");
        m.kind = o.value("kind").toString("package");
        m.summary = o.value("summary").toString();
        m.licence = o.value("licence").toString();
        m.licence_url = o.value("licence_url").toString();
        // The list is remote data and this link is handed to the system: https only (11 Nov 2026).
        if (QUrl(m.licence_url).scheme() != "https") m.licence_url.clear();
        m.folder = o.value("folder").toString(m.id);
        m.min_app = o.value("min_app").toString();
        bool ok = !m.id.isEmpty() && safe_relative(m.folder) && !m.folder.contains('/');
        for (const auto& fv : o.value("files").toArray()) {
            const QJsonObject fo = fv.toObject();
            RemoteFile f;
            f.path = fo.value("path").toString();
            f.url = fo.value("url").toString();
            f.size = qint64(fo.value("size").toDouble());
            f.sha256 = fo.value("sha256").toString().toLower();
            const QString scheme = QUrl(f.url).scheme();
            ok = ok && safe_relative(f.path) && f.size > 0 && hex.match(f.sha256).hasMatch() &&
                 (scheme == "https" || scheme == "file");
            m.files.push_back(f);
        }
        if (ok && !m.files.empty()) out.push_back(std::move(m));   // a bad entry is left out, not the list
    }
    return out;
}

namespace {

// Every file of every model RUDRA has released, by SHA-256 (see model_downloads.hpp). Taken from
// the published list and checked against the files themselves on 11 Nov 2026.
constexpr const char* kKnownModelFiles[] = {
    // RUDRA ACES 2, version 2.0 (w4 + lock), released 8 Nov 2026
    "db21ffb4a0a652f354d24fd1b6a997717ec862b009a64e5175d1f4c4f25198e8",   // manifest.json
    "beef682e39f8276147a329f57eca70817c056ef0dcc3b053d0b481dadb634c69",   // parity.json
    "880e7b698a099b8dd60549444392d51af3022efbae8f68f05c9f0f3a29f8ee49",   // rudra_g1.onnx
    "7082eb869d265d6e8c4fc209f7cac7e5dce2f40c10df5e73a889fb85efb9dbcc",   // rudra_g1.ts
    // AI fill v3
    "9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a",   // rudra_genfill.safetensors
    "34892fb840de138fb3ddd2e2c8e6e3b13c232973ed59ea23338444fcaff2a95a",   // brightness.json
    "5b72b3888e37cc1d198594c6f534c5ba1dcbdbb2fdf924f6a93f37d8dfe502f7",   // registration_v3.json
    "aa11a3cb65a249898af299d184ec7f697e965ccacc69bcaa006f9ecc76df29bf",   // report_v3.json
    "fbbb1040d9c7f522bdf9327ab04cf3ce21b85f2651846223a3111400805bfecd",   // prompt_cache/19805c5814225ef6a3c96bad.json
    "a25186b90f16ad0ae397e72bd748a8150ebbfbcedc107057605e64245d36f243",   // prompt_cache/19805c5814225ef6a3c96bad.pt
    "995d82dd6f27b0b3de6bd526d585a8b12aa6ae461fa0de9d1ccc63b7c2653150",   // prompt_cache/c9138bdd2a8e8feabc812558.json
    "6af2658b9e0c99ed3e3446eae0484fd84ec2bdeee3bb90d1f9312eaae3202fe6",   // prompt_cache/c9138bdd2a8e8feabc812558.pt
    // AI fill v4, region re-render (experimental, 1.2.4), folder rudra_genfill_v4
    "406b1dc7b7d3ada5406b396a6aa8c7e5ec9eac8d0be0d04e8038d68b5bf1b390",   // rudra_genfill.safetensors (v4.2 training)
    "8f2324358a10feb7599446443af7622a096f360f99d16a12830f5ebc6a68e90d",   // registration_v4.json (docs/models/genfill_v4)
    "d2f507bbcfdcc0203bec82050a3f5ef89188b12e8310b8b5f5a12d7816b5db8c",   // report_v4.json (docs/models/genfill_v4)
    // AI fill v5, region re-render (experimental, 1.3), folder rudra_genfill_v5: the LoRA of
    // research_genfill_v5_20261122, run with --rerender-hold-clipped thin (research_genfill_v5d_thin_20261123)
    "5c5d4ab4107088f86bdc8213ae2ec21c1a882cc6d942141f0e12fb7733fda5e6",   // rudra_genfill.safetensors (v5 training)
    "1d9d6a5a1803ca8704dff357c1a1a06065486ff0d65c272bed979f5a14df1039",   // registration_v5.json (docs/models/genfill_v5)
    "bc17f9678828e0957efbd4d581c9a1a2cfa783101fa01402873c584173ae2237",   // report_v5.json (docs/models/genfill_v5)
    // Florence-2 base (MIT; Suggest a prompt), kind "florence", folder Florence-2-base: florence-community/
    // Florence-2-base at revision 00921df66db728a9ceb750f5eca43e5c203a2051, as in rudra/genfill/model_manifest.json
    // (1.2.6). Downloaded from its own Hugging Face repo at that revision; listed in Check this computer.
    "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",   // .gitattributes
    "9bc4297ba09ae961d3b172358f4bf723d6757fe11507b62f2369cc968e063e15",   // README.md
    "1d75deda84dfa81fb6c09301f3fed00f9695059568bbb1403a6bf299cd84fc37",   // added_tokens.json
    "2a4dfe9885d183da121f239ada734bd700ee6908b650a7a25ab32eb104fe1d7a",   // config.json
    "0251459c49cc358ac033b5d4b8569e61ac22bde5d76be404b97a44cfb33fb12e",   // generation_config.json
    "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",   // merges.txt
    "62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0",   // model.safetensors
    "1396ec5a0a7adfe1c04fb777b09e8ba753be6dbb5868212ab3c3ef39d91fe031",   // preprocessor_config.json
    "cd0e3bf41a39b1276503fbd273bc03b9afc70d7a15ea681a92a1b4b77f858ee6",   // processor_config.json
    "72ff172dc769bc1551b1b4211628ce3271643bc60379e4da45d85a9be9332c39",   // special_tokens_map.json
    "3ad7001f773409abe6bba33eac92662611a73d72f459bda2f00d2a221dd31ce4",   // tokenizer.json
    "cb5f80bd9afa767bb1bb798ee5f0a79eae45239b8e72506166b4218175a16723",   // tokenizer_config.json
    "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",   // vocab.json
};

}  // namespace

QString published_model_index_url() { return QStringLiteral("https://www.fxtdstudios.com/rudra-docs/models.json"); }

QString default_model_index_url() { return QSettings().value("models/indexUrl", published_model_index_url()).toString(); }

bool is_known_model_file(const QString& sha256) {
    const QByteArray want = sha256.toLower().toLatin1();
    for (const char* known : kKnownModelFiles)
        if (want == known) return true;
    return false;
}

std::vector<RemoteModel> known_models(std::vector<RemoteModel> all, int* left_out) {
    std::vector<RemoteModel> out;
    int left = 0;
    for (auto& m : all) {
        bool ok = !m.files.empty();
        for (const auto& f : m.files) ok = ok && QUrl(f.url).scheme() == "https" && is_known_model_file(f.sha256);
        if (ok) out.push_back(std::move(m));
        else ++left;
    }
    if (left_out) *left_out = left;
    return out;
}

QString default_download_root() {
    const QString saved = QSettings().value("models/downloadRoot").toString();
    if (!saved.isEmpty()) return saved;
    return join(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation), QStringLiteral("models"));
}

InstallState install_state(const RemoteModel& m, const QString& root) {
    const QString dir = join(root, m.folder);
    if (!QFileInfo(dir).isDir()) return InstallState::NotInstalled;
    QFile marker(join(dir, kMarker));
    bool all = marker.open(QIODevice::ReadOnly);
    if (all) {
        const QJsonObject o = QJsonDocument::fromJson(marker.readAll()).object();
        all = o.value("id").toString() == m.id && o.value("version").toString() == m.version;
    }
    bool any = false;
    for (const auto& f : m.files) {
        const QFileInfo fi(join(dir, f.path));
        const bool here = file_checks(fi.filePath(), f);
        // A file that is there but is not the one the list names counts as a part: "Resume" replaces it.
        any = any || here || fi.isFile() || QFileInfo::exists(fi.filePath() + ".part");
        all = all && here;
    }
    return all ? InstallState::Installed : any ? InstallState::Partial : InstallState::NotInstalled;
}

// ---------------------------------------------------------------------------

ModelDownloads::ModelDownloads(QNetworkAccessManager* net, QObject* owner)
    : net_(net), alive_(new QObject(owner)), root_(default_download_root()) {}

ModelDownloads::~ModelDownloads() {
    listeners_.clear();
    stop_reply();
    if (part_) part_->close();
    delete alive_;   // its connections (the replies', the listeners' owners) go with it
}

const RemoteModel* ModelDownloads::model(const QString& id) const {
    for (const auto& m : models_)
        if (m.id == id) return &m;
    return nullptr;
}

void ModelDownloads::set_models(std::vector<RemoteModel> m) {
    models_ = std::move(m);
    index_status_.clear();
    changed();
}

std::vector<RemoteModel> builtin_models() {
    // Florence-2 base: florence-community/Florence-2-base at the revision rudra/genfill/model_manifest.json
    // pins; sizes and SHA-256 from there (the same hashes are in kKnownModelFiles).
    RemoteModel m;
    m.id = QString::fromLatin1(kFlorenceModelId);
    m.title = QStringLiteral("Florence-2 base");
    m.version = QStringLiteral("2026.11");
    m.channel = QStringLiteral("release");
    m.kind = QStringLiteral("florence");
    m.summary = QStringLiteral("Describes the shot for Suggest a prompt (captions only, runs on your computer)");
    m.licence = QStringLiteral("MIT");
    m.licence_url = QStringLiteral("https://huggingface.co/florence-community/Florence-2-base");
    m.folder = QStringLiteral("Florence-2-base");
    m.min_app = QStringLiteral("1.2.6");
    const QString base = QStringLiteral("https://huggingface.co/florence-community/Florence-2-base/resolve/"
                                        "00921df66db728a9ceb750f5eca43e5c203a2051/");
    struct F { const char* path; qint64 size; const char* sha; };
    static const F files[] = {
        {".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"},
        {"README.md", 13423, "9bc4297ba09ae961d3b172358f4bf723d6757fe11507b62f2369cc968e063e15"},
        {"added_tokens.json", 22430, "1d75deda84dfa81fb6c09301f3fed00f9695059568bbb1403a6bf299cd84fc37"},
        {"config.json", 2387, "2a4dfe9885d183da121f239ada734bd700ee6908b650a7a25ab32eb104fe1d7a"},
        {"generation_config.json", 292, "0251459c49cc358ac033b5d4b8569e61ac22bde5d76be404b97a44cfb33fb12e"},
        {"merges.txt", 456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"},
        {"model.safetensors", 463178864, "62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0"},
        {"preprocessor_config.json", 603, "1396ec5a0a7adfe1c04fb777b09e8ba753be6dbb5868212ab3c3ef39d91fe031"},
        {"processor_config.json", 2264, "cd0e3bf41a39b1276503fbd273bc03b9afc70d7a15ea681a92a1b4b77f858ee6"},
        {"special_tokens_map.json", 146627, "72ff172dc769bc1551b1b4211628ce3271643bc60379e4da45d85a9be9332c39"},
        {"tokenizer.json", 3748144, "3ad7001f773409abe6bba33eac92662611a73d72f459bda2f00d2a221dd31ce4"},
        {"tokenizer_config.json", 197922, "cb5f80bd9afa767bb1bb798ee5f0a79eae45239b8e72506166b4218175a16723"},
        {"vocab.json", 798293, "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e"},
    };
    for (const F& f : files)
        m.files.push_back({QString::fromLatin1(f.path), base + QString::fromLatin1(f.path), f.size, QString::fromLatin1(f.sha)});
    return {m};
}

void add_missing_builtins(std::vector<RemoteModel>& models) {
    for (RemoteModel& b : builtin_models())
        if (std::none_of(models.begin(), models.end(), [&](const RemoteModel& m) { return m.kind == b.kind || m.id == b.id; }))
            models.push_back(std::move(b));
}

void ModelDownloads::ensure_builtin() {
    const auto before = models_.size();
    add_missing_builtins(models_);
    if (models_.size() != before) changed();
}

void ModelDownloads::set_root(const QString& root) {
    if (!active_.isEmpty() || root.isEmpty()) return;
    root_ = root;
    QSettings().setValue("models/downloadRoot", root);
    changed();
}

InstallState ModelDownloads::state(const QString& id) const {
    const RemoteModel* m = model(id);
    return m ? install_state(*m, root_) : InstallState::NotInstalled;
}

QString ModelDownloads::error(const QString& id) const {
    const auto it = errors_.find(id);
    return it == errors_.end() ? QString() : it->second;
}

void ModelDownloads::listen(QObject* owner, std::function<void()> on_change) {
    const int id = next_listener_++;
    listeners_[id] = {owner, std::move(on_change)};
    QObject::connect(owner, &QObject::destroyed, alive_, [this, id] { listeners_.erase(id); });
}

void ModelDownloads::changed() {
    const auto copy = listeners_;
    for (const auto& [id, l] : copy)
        if (l.first && l.second) l.second();
}

void ModelDownloads::fetch_index(const QString& url) {
    if (index_reply_) return;
    QNetworkRequest req{QUrl(url)};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setHeader(QNetworkRequest::UserAgentHeader, "RUDRA/" + QCoreApplication::applicationVersion());
    index_status_.clear();
    index_reply_ = net_->get(req);
    QNetworkReply* r = index_reply_;
    QObject::connect(r, &QNetworkReply::finished, alive_, [this, r, url] {
        r->deleteLater();
        index_reply_ = nullptr;
        if (r->error() != QNetworkReply::NoError) {
            index_status_ = QStringLiteral("Could not read the model list (%1): %2").arg(QUrl(url).host().isEmpty() ? url : QUrl(url).host(), r->errorString());
            add_missing_builtins(models_);   // 1.2.6: Florence-2 comes from Hugging Face, so it is offered without the list
        } else {
            QString why;
            if (auto m = parse_model_index(r->readAll(), &why)) {
                int left_out = 0;
                // From the published address, only the models this RUDRA knows by their checksums.
                if (QUrl(url) == QUrl(published_model_index_url())) {
                    models_ = known_models(std::move(*m), &left_out);
                    add_missing_builtins(models_);   // 1.2.6: Florence-2 before the published list names it
                } else {
                    models_ = std::move(*m);
                }
                index_status_ = left_out > 0 ? QStringLiteral("%1 in the list %2 a newer RUDRA: update RUDRA to download %3.")
                                                   .arg(left_out == 1 ? QStringLiteral("One model") : QStringLiteral("%1 models").arg(left_out),
                                                        left_out == 1 ? QStringLiteral("needs") : QStringLiteral("need"),
                                                        left_out == 1 ? QStringLiteral("it") : QStringLiteral("them"))
                                : models_.empty() ? QStringLiteral("No models to download yet.")
                                                  : QString();
            } else {
                index_status_ = why;
                add_missing_builtins(models_);
            }
        }
        changed();
    });
    changed();
}

bool ModelDownloads::download(const QString& id, QString* why) {
    if (!active_.isEmpty()) {
        if (why) *why = QStringLiteral("Another model is downloading: wait for it, or pause it.");
        return false;
    }
    const RemoteModel* m = model(id);
    if (!m) {
        if (why) *why = QStringLiteral("That model is not in the list.");
        return false;
    }
    active_ = id;
    errors_.erase(id);
    file_ = 0;
    done_ = 0;
    total_ = m->size();
    rate_ = 0;
    rate_mark_bytes_ = 0;
    rate_mark_ms_ = QDateTime::currentMSecsSinceEpoch();
    next_file();
    return true;
}

void ModelDownloads::next_file() {
    const RemoteModel* m = model(active_);
    if (!m) return fail(QStringLiteral("The model left the list."));
    const QString dir = join(root_, m->folder);
    // What is already here and verified (a file is renamed into place only after its check).
    while (file_ < m->files.size()) {
        const auto& f = m->files[file_];
        if (!file_checks(join(dir, f.path), f)) break;   // by its hash: a same-sized other file is fetched again
        done_ += f.size;
        ++file_;
    }
    if (file_ >= m->files.size()) {   // all of it: the marker makes it installed
        QSaveFile marker(join(dir, kMarker));
        QJsonArray files;
        for (const auto& f : m->files) files.append(f.path);
        if (marker.open(QIODevice::WriteOnly)) {
            marker.write(QJsonDocument(QJsonObject{{"id", m->id}, {"version", m->version}, {"title", m->title},
                                                   {"licence", m->licence}, {"files", files},
                                                   {"installed", QDateTime::currentDateTimeUtc().toString(Qt::ISODate)}})
                             .toJson());
            marker.commit();
        }
        const RemoteModel done_model = *m;
        active_.clear();
        changed();
        if (installed) installed(done_model, dir);
        return;
    }
    const auto& f = m->files[file_];
    const QString dest = join(dir, f.path);
    if (!QDir().mkpath(QFileInfo(dest).absolutePath())) return fail(QStringLiteral("Cannot make the folder ") + QFileInfo(dest).absolutePath());
    part_ = std::make_unique<QFile>(dest + ".part");
    if (!part_->open(QIODevice::ReadWrite)) return fail(QStringLiteral("Cannot write ") + part_->fileName() + ": " + part_->errorString());
    qint64 have = part_->size();
    if (have > f.size) {
        part_->resize(0);
        have = 0;
    }
    part_->seek(have);
    file_start_ = done_;
    done_ += have;
    http_status_ = 0;
    net_error_.clear();
    leftover_ = false;
    // All of it is here already (the app closed while it was checked, or it could
    // not be put in place): check it. Asking for "the rest" gets a 416 and, before
    // 9 Nov 2026, the whole file again.
    if (have == f.size) {
        leftover_ = true;
        ranged_ = false;
        checked_status_ = true;
        changed();
        return file_finished();
    }
    QNetworkRequest req{QUrl(f.url)};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setHeader(QNetworkRequest::UserAgentHeader, "RUDRA/" + QCoreApplication::applicationVersion());
    ranged_ = have > 0;
    if (ranged_) req.setRawHeader("Range", "bytes=" + QByteArray::number(have) + "-");
    checked_status_ = false;
    reply_ = net_->get(req);
    QNetworkReply* r = reply_;
    QObject::connect(r, &QNetworkReply::readyRead, alive_, [this, r] {
        if (r != reply_ || !part_) return;
        // An error's body (a 404 page, a proxy's 503) is not the file: read and dropped,
        // so what is in the .part stays what the server sent of the file.
        const int status = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (status >= 400) {
            r->readAll();
            return;
        }
        if (!checked_status_) {
            checked_status_ = true;
            // Asked for the rest and got the whole file (no Range support): start this file again.
            if (ranged_ && status != 206) {
                part_->resize(0);
                part_->seek(0);
                done_ = file_start_;
            }
        }
        const QByteArray b = r->readAll();
        part_->write(b);
        done_ += b.size();
        const qint64 now = QDateTime::currentMSecsSinceEpoch();
        if (now - rate_mark_ms_ >= 250) {
            const double r_now = double(done_ - rate_mark_bytes_) * 1000.0 / double(now - rate_mark_ms_);
            rate_ = rate_ > 0 ? 0.7 * rate_ + 0.3 * r_now : r_now;
            rate_mark_bytes_ = done_;
            rate_mark_ms_ = now;
            changed();
        }
    });
    QObject::connect(r, &QNetworkReply::finished, alive_, [this, r] {
        r->deleteLater();
        if (r != reply_) return;   // paused or cancelled
        reply_ = nullptr;
        http_status_ = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        net_error_ = r->error() == QNetworkReply::NoError ? QString() : r->errorString();
        file_finished();
    });
    changed();
}

void ModelDownloads::file_finished() {
    const RemoteModel* m = model(active_);
    if (!m || !part_) return;
    const auto& f = m->files[file_];
    // The server refused (403, 404, 5xx): say so. The .part keeps what it had, to
    // resume from; after a 416 (it no longer matches the server's file) it is emptied.
    if (http_status_ >= 400) {
        if (http_status_ == 416) part_->resize(0);
        part_->close();
        part_.reset();
        return fail(QStringLiteral("%1 could not be downloaded (HTTP %2 from %3): Download again to retry")
                        .arg(f.path)
                        .arg(http_status_)
                        .arg(QUrl(f.url).host()));
    }
    // No data came back for a ranged request: start this file again next time.
    if (!checked_status_ && ranged_ && net_error_.isEmpty()) {
        part_->resize(0);
        done_ = file_start_;
    }
    part_->flush();
    const qint64 got = part_->size();
    if (got != f.size) {
        part_->close();
        part_.reset();
        return fail(QStringLiteral("%1 stopped at %2 of %3%4: Download again to resume")
                        .arg(f.path, human_size(got), human_size(f.size),
                             net_error_.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(net_error_)));
    }
    part_->seek(0);
    QCryptographicHash h(QCryptographicHash::Sha256);
    h.addData(part_.get());
    const QString sha = QString::fromLatin1(h.result().toHex());
    const QString part_name = part_->fileName();
    part_->close();
    part_.reset();
    if (sha != f.sha256) {
        QFile::remove(part_name);
        done_ = file_start_;
        if (leftover_) return next_file();   // a whole .part from an earlier try that does not check: fetched fresh
        return fail(QStringLiteral("%1 did not match its checksum and was deleted: Download again").arg(f.path));
    }
    const QString dest = part_name.chopped(5);
    QFile::remove(dest);
    if (!QFile::rename(part_name, dest)) return fail(QStringLiteral("Cannot put ") + dest + QStringLiteral(" in place"));
    ++file_;
    next_file();
}

void ModelDownloads::stop_reply() {
    if (QNetworkReply* r = reply_) {
        reply_ = nullptr;
        r->abort();
    }
}

void ModelDownloads::fail(const QString& why) {
    stop_reply();
    if (part_) {
        part_->close();
        part_.reset();
    }
    if (!active_.isEmpty()) errors_[active_] = why;
    active_.clear();
    rate_ = 0;
    changed();
}

void ModelDownloads::pause() {
    if (active_.isEmpty()) return;
    stop_reply();
    if (part_) {
        part_->close();
        part_.reset();
    }
    errors_[active_] = QStringLiteral("Paused: Download again to resume");
    active_.clear();
    rate_ = 0;
    changed();
}

void ModelDownloads::cancel(const QString& which) {
    const QString id = which.isEmpty() ? active_ : which;
    if (id == active_) pause();
    if (const RemoteModel* m = model(id)) {
        for (const auto& f : m->files) QFile::remove(join(join(root_, m->folder), f.path) + ".part");
        errors_.erase(id);
    }
    changed();
}

}  // namespace rudra::app
