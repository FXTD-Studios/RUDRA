#include "batch_queue.hpp"
#include "paths.hpp"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSettings>
#include <QUuid>

#include <algorithm>
#include <map>

#include "aces_engine.hpp"
#include "aces_master.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

QString qp(const fs::path& p) { return from_path(p); }
fs::path pq(const QString& s) { return to_path(s); }

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// "SH010.1001" -> ("SH010.", 1001); a stem without trailing digits -> nullopt.
std::optional<std::pair<std::string, long long>> numbered(const std::string& stem) {
    std::size_t i = stem.size();
    while (i > 0 && std::isdigit(static_cast<unsigned char>(stem[i - 1]))) --i;
    if (i == stem.size() || stem.size() - i > 12) return std::nullopt;
    return std::pair{stem.substr(0, i), std::stoll(stem.substr(i))};
}

QString title_of_prefix(const std::string& prefix, const fs::path& folder) {
    std::string t = prefix;
    while (!t.empty() && (t.back() == '.' || t.back() == '_' || t.back() == '-' || t.back() == ' ')) t.pop_back();
    return QString::fromStdString(t.empty() ? folder.filename().string() : t);
}

// Movies one each; stills grouped by (folder, prefix, extension) when numbered.
void group(const std::vector<fs::path>& files, std::vector<BatchShot>& out) {
    struct Group {
        fs::path folder;
        std::string prefix;
        std::vector<std::pair<long long, fs::path>> frames;
    };
    std::vector<Group> groups;
    std::map<std::string, std::size_t> by_key;
    std::vector<std::pair<std::size_t, BatchShot>> order;   // (position, shot) for movies and lone stills
    std::vector<std::size_t> group_pos;
    std::size_t pos = 0;
    for (const auto& f : files) {
        ++pos;
        if (is_aces_video(f)) {
            order.push_back({pos, BatchShot{qp(f.filename()), "movie", {f}}});
            continue;
        }
        if (!is_aces_still(f)) continue;
        const auto n = numbered(f.stem().string());
        if (!n) {
            order.push_back({pos, BatchShot{qp(f.filename()), "still", {f}}});
            continue;
        }
        const std::string key = lower(f.parent_path().string()) + "|" + lower(n->first) + "|" + lower(f.extension().string());
        auto it = by_key.find(key);
        if (it == by_key.end()) {
            it = by_key.emplace(key, groups.size()).first;
            groups.push_back({f.parent_path(), n->first, {}});
            group_pos.push_back(pos);
        }
        groups[it->second].frames.push_back({n->second, f});
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
        auto& fr = groups[g].frames;
        std::sort(fr.begin(), fr.end(), [](const auto& a, const auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
        if (fr.size() == 1) {
            order.push_back({group_pos[g], BatchShot{qp(fr.front().second.filename()), "still", {fr.front().second}}});
            continue;
        }
        BatchShot s{title_of_prefix(groups[g].prefix, groups[g].folder), "sequence", {}};
        for (const auto& [num, p] : fr) s.sources.push_back(p);
        order.push_back({group_pos[g], std::move(s)});
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& [p, s] : order) out.push_back(std::move(s));
}

std::vector<fs::path> files_in(const fs::path& dir) {
    std::vector<fs::path> f;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file(ec)) f.push_back(e.path());
    std::sort(f.begin(), f.end());
    return f;
}

QJsonObject choice_json(const DeliverChoice& c) {
    return {{"output", c.output}, {"image", c.image}, {"gamut", c.gamut}, {"encoding", c.encoding}, {"pq_peak", c.pq_peak}, {"bits", c.bits},
            {"codec", c.codec}, {"keep_exr", c.keep_exr}, {"fps", c.fps}, {"ai_fill", c.ai_fill},
            {"ai", QJsonObject{{"tau_high", c.ai.tau_high}, {"tau_low", c.ai.tau_low}, {"seed", c.ai.seed}, {"steps", c.ai.steps}, {"brightness", c.ai.brightness},
                               {"prompt_highlight", c.ai.prompt_highlight}, {"prompt_shadow", c.ai.prompt_shadow}}}};
}

DeliverChoice choice_from(const QJsonObject& o) {
    DeliverChoice c;
    c.output = o.value("output").toString(c.output);
    c.image = o.value("image").toString(c.image);
    c.gamut = o.value("gamut").toString(c.gamut);
    c.encoding = o.value("encoding").toString(c.encoding);
    c.pq_peak = o.value("pq_peak").toInt(c.pq_peak);
    c.bits = o.value("bits").toInt(c.bits);
    c.codec = o.value("codec").toString(c.codec);
    c.keep_exr = o.value("keep_exr").toBool(c.keep_exr);
    c.fps = o.value("fps").toDouble(c.fps);
    c.ai_fill = o.value("ai_fill").toBool(false);
    const QJsonObject ai = o.value("ai").toObject();
    c.ai.tau_high = ai.value("tau_high").toInt(c.ai.tau_high);
    c.ai.tau_low = ai.value("tau_low").toInt(c.ai.tau_low);
    c.ai.seed = ai.value("seed").toInt(c.ai.seed);
    c.ai.steps = ai.value("steps").toInt(c.ai.steps);
    c.ai.brightness = ai.value("brightness").toDouble(c.ai.brightness);
    c.ai.prompt_highlight = ai.value("prompt_highlight").toString();
    c.ai.prompt_shadow = ai.value("prompt_shadow").toString();
    return c;
}

}  // namespace

std::vector<BatchShot> batch_shots_from(const std::vector<fs::path>& paths) {
    std::vector<BatchShot> out;
    std::vector<fs::path> loose;
    auto flush = [&] {
        group(loose, out);
        loose.clear();
    };
    for (const auto& p : paths) {
        std::error_code ec;
        if (!fs::is_directory(p, ec)) {
            loose.push_back(p);
            continue;
        }
        flush();
        group(files_in(p), out);
        std::vector<fs::path> subs;
        for (const auto& e : fs::directory_iterator(p, ec))
            if (e.is_directory(ec)) subs.push_back(e.path());
        std::sort(subs.begin(), subs.end());
        for (const auto& s : subs) {
            std::vector<BatchShot> inner;
            group(files_in(s), inner);
            // A sub-folder that is one sequence is named after the folder.
            if (inner.size() == 1 && inner.front().kind == "sequence") inner.front().title = qp(s.filename());
            for (auto& x : inner) out.push_back(std::move(x));
        }
    }
    flush();
    return out;
}

QString batch_clean_name(const QString& title) {
    QString n = title.trimmed();
    // "clip.mov" / "plate.png": the name without the media extension.
    if (const fs::path p = pq(n); p.has_extension() && (is_aces_video(p) || is_aces_still(p))) n = qp(p.stem());
    for (QChar& c : n)
        if (!((c.isLetterOrNumber() && c.unicode() < 128) || c == '_' || c == '-' || c == '.' || c == ' ')) c = '_';
    n = n.trimmed();
    while (n.endsWith('.') || n.endsWith(' ')) n.chop(1);
    return n.left(120);
}

QJsonObject BatchJob::to_json() const {
    QJsonArray src;
    for (const auto& s : shot.sources) src.append(qp(s));
    return {{"id", id}, {"title", shot.title}, {"kind", shot.kind}, {"sources", src}, {"choice", choice_json(choice)},
            {"render_dir", render_dir}, {"name", name}, {"start", start},
            {"trims", QJsonObject{{"strength", strength}, {"saturation", saturation}, {"peak_nits", peak_nits},
                                  {"exposure_ev", exposure_ev}, {"contrast", contrast}}},
            {"timecode", timecode}, {"reel", reel}, {"source_display", source_display}, {"edl", edl}, {"first", first}, {"last", last}, {"maxcll", maxcll},
            {"maxfall", maxfall}, {"status", status}, {"done", done},
            {"total", total}, {"out", out}, {"error", error}, {"note", note}};
}

std::optional<BatchJob> BatchJob::from_json(const QJsonObject& o) {
    BatchJob j;
    j.id = o.value("id").toString();
    j.shot.title = o.value("title").toString();
    j.shot.kind = o.value("kind").toString("still");
    for (const auto& s : o.value("sources").toArray()) j.shot.sources.push_back(pq(s.toString()));
    if (j.id.isEmpty() || j.shot.sources.empty()) return std::nullopt;
    j.choice = choice_from(o.value("choice").toObject());
    j.render_dir = o.value("render_dir").toString();
    j.name = o.value("name").toString();
    j.start = o.value("start").toInt(1001);
    const QJsonObject t = o.value("trims").toObject();
    j.strength = t.value("strength").toDouble(1.0);
    j.saturation = t.value("saturation").toDouble(1.0);
    j.peak_nits = t.value("peak_nits").toDouble(0.0);
    j.exposure_ev = t.value("exposure_ev").toDouble(0.0);
    j.contrast = t.value("contrast").toDouble(1.0);
    j.timecode = o.value("timecode").toString();
    j.reel = o.value("reel").toString();
    j.source_display = o.value("source_display").toString("bt1886");
    j.edl = o.value("edl").toObject();
    j.first = o.value("first").toInt(-1);
    j.last = o.value("last").toInt(-1);
    j.maxcll = o.value("maxcll").toDouble();
    j.maxfall = o.value("maxfall").toDouble();
    j.status = o.value("status").toString("pending");
    j.done = o.value("done").toInt();
    j.total = o.value("total").toInt();
    j.out = o.value("out").toString();
    j.error = o.value("error").toString();
    j.note = o.value("note").toString();
    return j;
}

DeliverChoice batch_effective(const BatchJob& j) {
    DeliverChoice c = j.choice;
    c.source = "aces";
    c = deliver_fit(c, true, j.shot.kind == "movie", j.shot.kind == "sequence");
    if (!ai_fill_possible(c, true)) c.ai_fill = false;   // EXR stills and sequences only
    return c;
}

QString batch_output_text(const BatchJob& j) {
    const DeliverChoice c = batch_effective(j);
    QString t = deliver_pipe_label(c, false);   // the words without "learned ACES 2.0 →"
    if (t.isEmpty() || c.output != "movie") {
        if (c.image == "dpx10")
            t = QStringLiteral("DPX 10-bit, ") + deliver_encoding_words(c);
        else
            t = (c.gamut == "acescg" ? QStringLiteral("ACEScg EXR, ") : QStringLiteral("ACES 2065-1 EXR, ")) +
                (c.image == "exr16" ? QStringLiteral("half") : QStringLiteral("float32"));
    }
    const QString frames = c.output == "still"
                               ? (j.first >= 0 ? QStringLiteral("frame %1").arg(j.first) : QStringLiteral("first frame"))
                           : c.output == "sequence" && j.first < 0 ? QStringLiteral("every frame")
                                                                   : QString();
    if (!frames.isEmpty()) t += QStringLiteral(" \u00b7 ") + frames;
    if (j.strength != 1.0 || j.saturation != 1.0 || j.peak_nits > 0 || j.exposure_ev != 0.0 || j.contrast != 1.0)
        t += QStringLiteral(" \u00b7 trimmed");
    if (c.ai_fill) t += QStringLiteral(" + AI fill");
    return t;
}

QJsonObject batch_request(const BatchJob& j, QString* why) {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return QJsonObject{};
    };
    if (j.render_dir.trimmed().isEmpty()) return refuse(QStringLiteral("No render folder."));
    QString name = batch_clean_name(j.name);
    if (name.isEmpty()) name = batch_clean_name(j.shot.title);
    if (name.isEmpty()) name = QStringLiteral("master");
    std::error_code ec;
    for (const auto& s : j.shot.sources)
        if (!fs::exists(s, ec)) return refuse(QStringLiteral("Missing: ") + qp(s));
    const fs::path root = pq(j.render_dir.trimmed());
    fs::create_directories(root, ec);
    if (!fs::is_directory(root, ec)) return refuse(QStringLiteral("The render folder cannot be made: ") + j.render_dir);
    fs::path out = root / pq(name);
    for (int k = 2; fs::exists(out, ec) && k < 1000; ++k) out = root / pq(name + "_" + QString::number(k));
    if (fs::exists(out, ec)) return refuse(QStringLiteral("Too many renders called %1 here.").arg(name));
    const DeliverChoice c = batch_effective(j);
    QJsonObject req = aces_deliver_fields(c);
    QJsonArray sources;
    if (j.shot.kind == "movie") {
        sources.append(qp(j.shot.sources.front()));
        if (c.output == "still") req["frames"] = QJsonArray{std::max(j.first, 0)};
        else if (j.first >= 0) req["range"] = QJsonArray{j.first, j.last};
    } else if (c.output == "still") {
        sources.append(qp(j.shot.sources.front()));
    } else {
        for (const auto& s : j.shot.sources) sources.append(qp(s));
    }
    if (j.shot.kind != "movie") req["fps"] = c.fps;   // frames have no clock: their rate, for the movie and the timecode
    req["op"] = "master";
    req["sources"] = sources;
    req["out"] = qp(out);
    req["name"] = name;
    req["start"] = j.start;
    {   // the Deliver tab's frame numbering, so a batch render relinks like a Deliver one
        QSettings st;
        req["padding"] = std::clamp(st.value("deliver/renderPadding", 4).toInt(), 1, 10);
        const QString sep = st.value("deliver/renderSeparator", ".").toString();
        req["separator"] = (sep == "." || sep == "_" || sep.isEmpty()) ? sep : QStringLiteral(".");
    }
    if (const QJsonObject t = aces_trims_fields(j.strength, j.saturation, j.peak_nits, j.exposure_ev, j.contrast); !t.isEmpty())
        req["trims"] = t;
    if (!j.timecode.isEmpty()) req["timecode"] = j.timecode;
    if (!j.reel.isEmpty()) req["reel"] = j.reel;
    if (j.source_display != "bt1886" && !j.source_display.isEmpty()) req["source_display"] = j.source_display;
    if (!j.edl.isEmpty()) req["edl_event"] = j.edl;   // into provenance.json, for the conform
    return req;
}

// ---- the queue ----------------------------------------------------------------

BatchQueue::BatchQueue(fs::path file) : file_(std::move(file)) { load(); }

void BatchQueue::load() {
    QFile f(qp(file_));
    if (!f.open(QIODevice::ReadOnly)) return;
    QJsonParseError err{};
    const QJsonDocument d = QJsonDocument::fromJson(f.readAll(), &err);
    f.close();
    if (err.error != QJsonParseError::NoError || !d.isObject()) {
        // Never written over (28 Oct 2026 review): kept aside, and the window says so.
        const QString aside = qp(file_) + ".corrupt-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss");
        QFile::rename(qp(file_), aside);
        load_warning_ = QStringLiteral("The batch queue file could not be read (%1); it was kept as %2 and a new queue started.")
                            .arg(err.errorString(), QDir::toNativeSeparators(aside));
        return;
    }
    const QJsonArray a = d.object().value("jobs").toArray();
    for (const auto& v : a)
        if (auto j = BatchJob::from_json(v.toObject())) {
            if (j->status == "filling") {   // the app stopped during the AI fill: the master is there
                j->status = "done";
                j->note = QStringLiteral("AI fill interrupted");
            } else if (j->status == "running") {   // the app stopped while it ran: its partial render stays named
                j->status = "pending";
                j->note = j->out.isEmpty() ? QStringLiteral("interrupted")
                                           : QStringLiteral("interrupted: the partial render is in %1 (the next run writes a new folder)")
                                                 .arg(QDir::toNativeSeparators(j->out));
            }
            jobs_.push_back(std::move(*j));
        }
}

void BatchQueue::save() const {
    std::error_code ec;
    fs::create_directories(file_.parent_path(), ec);
    QJsonArray a;
    for (const auto& j : jobs_) a.append(j.to_json());
    QSaveFile f(qp(file_));
    if (!f.open(QIODevice::WriteOnly)) return;
    f.write(QJsonDocument(QJsonObject{{"version", 1}, {"jobs", a}}).toJson());
    f.commit();
}

void BatchQueue::touch() {
    save();
    if (changed) changed();
}

BatchJob* BatchQueue::find(const QString& id) {
    for (auto& j : jobs_)
        if (j.id == id) return &j;
    return nullptr;
}

std::vector<QString> BatchQueue::add(const std::vector<BatchShot>& shots, const DeliverChoice& choice,
                                     const QString& render_dir, int start, BatchTrims trims) {
    std::vector<QString> ids;
    for (const auto& s : shots) {
        BatchJob j;
        j.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        j.shot = s;
        j.choice = choice;
        j.choice.source = "aces";
        j.render_dir = render_dir;
        j.name = batch_clean_name(s.title);
        j.start = start;
        j.strength = trims.strength;
        j.saturation = trims.saturation;
        j.peak_nits = trims.peak_nits;
        j.exposure_ev = trims.exposure_ev;
        j.contrast = trims.contrast;
        ids.push_back(j.id);
        jobs_.push_back(std::move(j));
    }
    if (!ids.empty()) touch();
    return ids;
}

void BatchQueue::remove(const std::vector<QString>& ids) {
    const auto before = jobs_.size();
    std::erase_if(jobs_, [&](const BatchJob& j) {
        return j.status != "running" && j.status != "filling" && std::find(ids.begin(), ids.end(), j.id) != ids.end();
    });
    if (jobs_.size() != before) touch();
}

void BatchQueue::move(const QString& id, int delta) {
    for (std::size_t i = 0; i < jobs_.size(); ++i)
        if (jobs_[i].id == id) {
            const long long k = static_cast<long long>(i) + delta;
            if (k < 0 || k >= static_cast<long long>(jobs_.size())) return;
            std::swap(jobs_[i], jobs_[std::size_t(k)]);
            touch();
            return;
        }
}

void BatchQueue::apply(const std::vector<QString>& ids, const DeliverChoice& choice, const QString& render_dir, int start,
                       std::optional<BatchTrims> trims) {
    bool any = false;
    for (auto& j : jobs_) {
        if (j.status == "running" || (!ids.empty() && std::find(ids.begin(), ids.end(), j.id) == ids.end())) continue;
        j.choice = choice;
        j.choice.source = "aces";
        if (!render_dir.isEmpty()) j.render_dir = render_dir;
        if (start >= 0) j.start = start;
        if (trims)
            j.strength = trims->strength, j.saturation = trims->saturation, j.peak_nits = trims->peak_nits,
            j.exposure_ev = trims->exposure_ev, j.contrast = trims->contrast;
        any = true;
    }
    if (any) touch();
}

void BatchQueue::rename(const QString& id, const QString& name) {
    if (auto* j = find(id); j && j->status != "running") {
        j->name = batch_clean_name(name);
        touch();
    }
}

bool BatchQueue::set_range(const QString& id, int first, int last) {
    BatchJob* j = find(id);
    if (!j || (j->status == "running" || j->status == "filling")) return false;
    if (first < 0 || last < 0) {
        j->first = j->last = -1;
    } else {
        if (j->shot.kind != "movie" || last < first) return false;
        j->first = first;
        j->last = last;
    }
    touch();
    return true;
}

void BatchQueue::set_ai_fill(const std::vector<QString>& ids, bool on) {
    bool any = false;
    for (auto& j : jobs_)
        if (j.status != "running" && j.status != "filling" &&
            (ids.empty() ? j.status == "pending" : std::find(ids.begin(), ids.end(), j.id) != ids.end())) {
            j.choice.ai_fill = on;
            any = true;
        }
    if (any) touch();
}

void BatchQueue::retry(const std::vector<QString>& ids) {
    bool any = false;
    for (auto& j : jobs_)
        if (j.status != "running" && j.status != "filling" && j.status != "pending" &&
            (ids.empty() ? j.status != "done" : std::find(ids.begin(), ids.end(), j.id) != ids.end())) {
            j.status = "pending";
            j.done = j.total = 0;
            j.error.clear();
            j.note.clear();
            any = true;
        }
    if (any) touch();
}

void BatchQueue::clear_finished() {
    const auto before = jobs_.size();
    std::erase_if(jobs_, [](const BatchJob& j) { return j.status == "done"; });
    if (jobs_.size() != before) touch();
}

std::optional<QString> BatchQueue::next_pending() const {
    for (const auto& j : jobs_)
        if (j.status == "pending") return j.id;
    return std::nullopt;
}

// ---- the runner ---------------------------------------------------------------

BatchRunner::BatchRunner(BatchQueue& queue, std::function<AcesEngine*()> engine)
    : queue_(queue), engine_(std::move(engine)) {}

BatchRunner::~BatchRunner() { *alive_ = false; }

bool BatchRunner::start(QString* why) {
    auto refuse = [why](const QString& w) {
        if (why) *why = w;
        return false;
    };
    if (running_) return refuse(QStringLiteral("The batch is already rendering."));
    AcesEngine* e = engine_ ? engine_() : nullptr;
    if (!e || e->state() != AcesEngine::State::Ready)
        return refuse(QStringLiteral("The learned ACES 2.0 model is not loaded (Deliver > Learned ACES 2.0 master… loads it)."));
    if (!queue_.next_pending()) return refuse(QStringLiteral("Nothing is waiting in the queue."));
    running_ = true;
    stop_after_ = stop_now_ = false;
    if (log) log(QStringLiteral("Batch started"));
    run_next();
    return true;
}

void BatchRunner::stop_after_current() {
    if (!running_) return;
    stop_after_ = true;
    if (changed) changed();
}

void BatchRunner::stop_now() {
    if (!running_) return;
    stop_after_ = stop_now_ = true;
    if (AcesEngine* e = engine_ ? engine_() : nullptr) e->cancel_master(request_id_);
    if (changed) changed();
}

void BatchRunner::run_next() {
    for (;;) {
        const auto id = stop_after_ ? std::nullopt : queue_.next_pending();
        AcesEngine* e = engine_ ? engine_() : nullptr;
        if (!id || !e || e->state() != AcesEngine::State::Ready || e->busy()) {
            if (id && log)
                log(e && e->busy() ? QStringLiteral("Batch stopped: ") + AcesEngine::busy_text()
                                   : QStringLiteral("Batch stopped: the learned ACES 2.0 model is not loaded"));
            running_ = false;
            current_.clear();
            if (log) log(stop_after_ ? QStringLiteral("Batch stopped") : QStringLiteral("Batch finished"));
            stop_after_ = stop_now_ = false;
            queue_.touch();
            if (changed) changed();
            return;
        }
        BatchJob* j = queue_.find(*id);
        QString why;
        const QJsonObject req = batch_request(*j, &why);
        if (req.isEmpty()) {
            j->status = "failed";
            j->error = why;
            if (log) log(j->shot.title + ": " + why);
            queue_.touch();
            continue;
        }
        j->status = "running";
        j->note.clear();
        j->error.clear();
        j->done = 0;
        j->total = batch_effective(*j).output == "still" ? 1
                   : j->shot.kind == "movie"              ? (j->first >= 0 ? j->last - j->first + 1 : 0)
                                                          : int(j->shot.sources.size());
        j->out = req.value("out").toString();
        current_ = j->id;
        queue_.touch();
        if (log) log(j->shot.title + " → " + QDir::toNativeSeparators(j->out));
        const QString jid = j->id;
        const std::weak_ptr<bool> alive = alive_;
        request_id_ = e->request(
            req,
            [this, alive, jid](const QJsonObject& o) {
                if (alive.lock() && *alive.lock()) finished(jid, o);
            },
            [this, alive, jid](const QJsonObject& o) {
                if (!(alive.lock() && *alive.lock())) return;
                if (BatchJob* job = queue_.find(jid)) {
                    const QJsonObject p = o.value("progress").toObject();
                    job->done = p.value("index").toInt();
                    job->total = p.value("total").toInt(job->total);
                    if (changed) changed();   // no save per frame
                }
            });
        return;
    }
}

void BatchRunner::finished(const QString& id, const QJsonObject& o) {
    if (BatchJob* j = queue_.find(id)) {
        if (!o.value("ok").toBool()) {
            j->status = "failed";
            j->error = (o.value("error").toString() + ' ' + o.value("detail").toString()).trimmed();
        } else if (o.value("cancelled").toBool()) {
            j->status = "stopped";
            j->done = o.value("frames").toInt();
        } else {
            j->status = "done";
            j->done = j->total = o.value("frames").toInt();
            j->maxcll = o.value("maxcll").toDouble();
            j->maxfall = o.value("maxfall").toDouble();
            // Then the AI fill of this shot's master, when it is asked for (29 Oct 2026).
            if (batch_effective(*j).ai_fill && !stop_now_) {
                QString why;
                const std::weak_ptr<bool> alive = alive_;
                const QString jid = j->id;
                j->status = "filling";
                queue_.touch();
                if (changed) changed();
                if (fill && fill(*j, &why, [this, alive, jid](bool ok, const QString& text) {
                        if (!(alive.lock() && *alive.lock())) return;
                        if (BatchJob* job = queue_.find(jid)) {
                            job->status = ok ? "done" : "failed";
                            job->note = ok ? QStringLiteral("AI fill done") : QString();
                            if (!ok) job->error = text;
                            if (log) log(job->shot.title + ": " + text);
                        }
                        current_.clear();
                        queue_.touch();
                        run_next();
                    }))
                    return;
                j->status = "done";
                j->note = QStringLiteral("AI fill not run: ") + (fill ? why : QStringLiteral("no AI fill here"));
            }
        }
        if (log) log(j->shot.title + ": " + j->status + (j->error.isEmpty() ? QString() : " (" + j->error + ")") +
                     (j->note.isEmpty() ? QString() : " (" + j->note + ")"));
    }
    current_.clear();
    queue_.touch();
    run_next();
}

}  // namespace rudra::app
