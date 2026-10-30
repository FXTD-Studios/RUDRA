#include "edl.hpp"
#include "paths.hpp"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>

#include <algorithm>

#include "aces_master.hpp"
#include "rudra/core/timecode.hpp"

namespace fs = std::filesystem;

namespace rudra::app {
namespace {

bool is_tc(const QString& s) { return parse_timecode(s.toStdString()).has_value(); }

std::optional<long long> frames_of(const QString& tc, int base) {
    auto t = parse_timecode(tc.toStdString());
    if (!t) return std::nullopt;
    if (t->drop && base != 30 && base != 60) t->drop = false;   // a drop label at a rate that has none: count it plain
    return timecode_to_frames(*t, base);
}

QString stem_of(const QString& name) { return QFileInfo(name).completeBaseName(); }

}  // namespace

std::vector<EdlEvent> parse_edl(const QString& text, QString* title) {
    std::vector<EdlEvent> out;
    const QStringList lines = text.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts);
    for (const QString& raw : lines) {
        const QString line = raw.trimmed();
        if (line.startsWith("TITLE:", Qt::CaseInsensitive)) {
            if (title) *title = line.mid(6).trimmed();
            continue;
        }
        if (line.startsWith('*')) {
            if (out.empty()) continue;
            const QString c = line.mid(1).trimmed();
            if (c.startsWith("FROM CLIP NAME:", Qt::CaseInsensitive)) out.back().clip_name = c.mid(15).trimmed();
            else if (c.startsWith("SOURCE FILE:", Qt::CaseInsensitive)) out.back().source_file = c.mid(12).trimmed();
            continue;
        }
        const QStringList t = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        // "M2   A001C003   048.0   01:00:10:00": a retime of the event above, at the source's frames a second.
        if (t.size() >= 4 && t[0].compare("M2", Qt::CaseInsensitive) == 0) {
            bool ok = false;
            const double speed = t[2].toDouble(&ok);
            for (auto it = out.rbegin(); ok && it != out.rend(); ++it)
                if (it->reel.compare(t[1], Qt::CaseInsensitive) == 0) {
                    it->speed = speed;
                    break;
                }
            continue;
        }
        if (t.size() < 8) continue;
        bool ok = false;
        const int number = t[0].toInt(&ok);
        if (!ok) continue;
        const int n = int(t.size());
        if (!(is_tc(t[n - 4]) && is_tc(t[n - 3]) && is_tc(t[n - 2]) && is_tc(t[n - 1]))) continue;
        EdlEvent e;
        e.number = number;
        e.reel = t[1];
        e.track = t[2];
        e.transition = t.mid(3, n - 7).join(' ');
        e.src_in = t[n - 4];
        e.src_out = t[n - 3];
        e.rec_in = t[n - 2];
        e.rec_out = t[n - 1];
        out.push_back(e);
    }
    return out;
}

bool edl_media_info(const fs::path& movie, EdlMedia& out, QString* why) {
    const QString ffprobe = QStandardPaths::findExecutable("ffprobe");
    if (ffprobe.isEmpty()) {
        if (why) *why = "ffprobe is not on the PATH";
        return false;
    }
    QProcess p;
    p.start(ffprobe, {"-v", "error", "-show_streams", "-show_format", "-of", "json", from_path(movie)});
    if (!p.waitForFinished(60000) || p.exitCode() != 0) {
        if (why) *why = "ffprobe could not read it";
        return false;
    }
    const QJsonObject j = QJsonDocument::fromJson(p.readAllStandardOutput()).object();
    out = EdlMedia{};
    out.path = movie;
    bool video = false;
    auto tags_of = [&](const QJsonObject& o) {
        const QJsonObject tags = o.value("tags").toObject();
        for (auto it = tags.begin(); it != tags.end(); ++it) {
            const QString k = it.key().toLower(), v = it.value().toString();
            if (k == "timecode" && out.start_tc.isEmpty() && is_tc(v)) out.start_tc = v;
            if (k == "reel_name" && out.reel.isEmpty()) out.reel = v;
        }
    };
    double duration = j.value("format").toObject().value("duration").toString().toDouble();
    for (const auto& sv : j.value("streams").toArray()) {
        const QJsonObject s = sv.toObject();
        tags_of(s);
        if (!video && s.value("codec_type").toString() == "video") {
            video = true;
            const QStringList r = s.value("avg_frame_rate").toString().split('/');
            const double fps = r.size() == 2 && r[1].toDouble() > 0 ? r[0].toDouble() / r[1].toDouble() : r[0].toDouble();
            if (fps > 0) out.fps = fps;
            out.frames = s.value("nb_frames").toString().toLongLong();
            if (s.contains("duration")) duration = s.value("duration").toString().toDouble();
        }
    }
    tags_of(j.value("format").toObject());
    if (!video) {
        if (why) *why = "no video stream";
        return false;
    }
    if (out.frames <= 0 && duration > 0) out.frames = static_cast<long long>(duration * out.fps + 0.5);
    return true;
}

std::vector<EdlMedia> edl_media_in(const fs::path& folder) {
    std::vector<fs::path> files;
    std::error_code ec;
    auto scan = [&](const fs::path& d) {
        for (const auto& e : fs::directory_iterator(d, ec))
            if (e.is_regular_file(ec) && is_aces_video(e.path())) files.push_back(e.path());
    };
    scan(folder);
    for (const auto& e : fs::directory_iterator(folder, ec))
        if (e.is_directory(ec)) scan(e.path());
    std::sort(files.begin(), files.end());
    std::vector<EdlMedia> out;
    for (const auto& f : files) {
        EdlMedia m;
        if (edl_media_info(f, m)) out.push_back(m);
    }
    return out;
}

EdlMatch edl_match(const std::vector<EdlEvent>& events, const std::vector<EdlMedia>& media, int handles) {
    EdlMatch r;
    handles = std::max(handles, 0);
    for (const EdlEvent& e : events) {
        if (!e.video()) continue;
        const QString label = QStringLiteral("%1 %2").arg(e.number, 3, 10, QChar('0')).arg(e.reel);
        if (e.reel.compare("BL", Qt::CaseInsensitive) == 0) continue;   // black: nothing to render
        // The movie: by clip name, then by reel (its tag, else its file name); its timecode must hold the edit.
        auto holds = [&](const EdlMedia& m, long long& first, long long& last) {
            const int base = timecode_base(m.fps);
            const auto in = frames_of(e.src_in, base), out = frames_of(e.src_out, base);
            const auto start = frames_of(m.start_tc.isEmpty() ? QStringLiteral("00:00:00:00") : m.start_tc, base);
            if (!in || !out || !start || *out <= *in) return false;
            first = *in - *start;
            last = *out - 1 - *start;
            return first >= 0 && (m.frames <= 0 || last < m.frames);
        };
        const QString clip = e.clip_name.isEmpty() ? QFileInfo(e.source_file).fileName() : e.clip_name;
        std::vector<const EdlMedia*> by_name, by_reel;
        for (const auto& m : media) {
            const QString file = from_path(m.path.filename());
            if (!clip.isEmpty() && (file.compare(clip, Qt::CaseInsensitive) == 0 ||
                                    stem_of(file).compare(stem_of(clip), Qt::CaseInsensitive) == 0))
                by_name.push_back(&m);
            if ((!m.reel.isEmpty() && m.reel.compare(e.reel, Qt::CaseInsensitive) == 0) ||
                stem_of(file).compare(e.reel, Qt::CaseInsensitive) == 0 ||
                stem_of(file).startsWith(e.reel + "_", Qt::CaseInsensitive))
                by_reel.push_back(&m);
        }
        const EdlMedia* hit = nullptr;
        long long first = 0, last = 0;
        bool range_wrong = false;
        for (const auto* list : {&by_name, &by_reel}) {
            for (const EdlMedia* m : *list) {
                long long f = 0, l = 0;
                if (holds(*m, f, l)) {
                    hit = m;
                    first = f;
                    last = l;
                    break;
                }
                range_wrong = true;
            }
            if (hit) break;
        }
        if (!hit) {
            r.unmatched << label + (range_wrong ? QStringLiteral(": the movie found does not hold this timecode")
                                                : QStringLiteral(": no movie with this clip name or reel"));
            continue;
        }
        EdlShot s;
        // A retime (M2): the source frames the edit really uses, forward from its in point.
        if (e.speed != 0) {
            const int base = timecode_base(hit->fps);
            const double ratio = e.speed / hit->fps;
            const auto ri = frames_of(e.rec_in, base), ro = frames_of(e.rec_out, base);
            if (ri && ro && *ro > *ri && e.speed > 0) {
                const long long used = std::max<long long>(1, std::llround(double(*ro - *ri) * ratio));
                last = first + used - 1;
                if (hit->frames > 0) last = std::min<long long>(last, hit->frames - 1);
            }
            s.warnings << QStringLiteral("speed %1% (M2 %2 fps)%3: the source frames it uses are rendered; the retime is "
                                         "the conform's")
                              .arg(std::lround(ratio * 100.0))
                              .arg(e.speed, 0, 'f', 1)
                              .arg(e.speed < 0 ? QStringLiteral(", reversed") : QString());
        }
        s.event = e.number;
        s.rec_in = e.rec_in;
        s.rec_out = e.rec_out;
        s.src_in = e.src_in;
        s.shot.title = from_path(hit->path.filename());
        s.shot.kind = "movie";
        s.shot.sources = {hit->path};
        s.first = int(std::max<long long>(0, first - handles));
        s.last = int(hit->frames > 0 ? std::min<long long>(hit->frames - 1, last + handles) : last + handles);
        // Handles the movie cannot give (the edit is near its first or last frame).
        const long long head = first - s.first, tail = s.last - last;
        if (head < handles || tail < handles)
            s.warnings << QStringLiteral("handles short: %1 head, %2 tail of %3").arg(head).arg(tail).arg(handles);
        s.name = QStringLiteral("%1_%2").arg(e.number, 3, 10, QChar('0')).arg(batch_clean_name(e.reel));
        for (const QString& w : s.warnings) r.warnings << label + ": " + w;
        r.shots.push_back(std::move(s));
    }
    return r;
}

}  // namespace rudra::app
