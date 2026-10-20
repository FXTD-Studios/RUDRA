#include "rudra/deliver/conform.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>

namespace rudra {

namespace {

Error parse_error(std::string message, std::string detail = {}) {
    return make_error(ErrorCode::ParseError, std::move(message), std::move(detail));
}

std::string trim(std::string_view s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return std::string(s.substr(a, b - a));
}

std::string upper(std::string s) {
    for (char& c : s) c = char(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> words(std::string_view line) {
    std::istringstream in{std::string(line)};
    std::vector<std::string> w;
    for (std::string x; in >> x;) w.push_back(x);
    return w;
}

bool looks_like_timecode(const std::string& s) {
    if (s.size() != 11) return false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const bool sep = i == 2 || i == 5 || i == 8;
        if (sep ? (s[i] != ':' && s[i] != ';' && s[i] != '.') : !std::isdigit(static_cast<unsigned char>(s[i])))
            return false;
    }
    return true;
}

// Rational seconds as FCPXML writes them: "3600/24s", "10s", "1001/24000s", "0s".
std::optional<double> rational_seconds(std::string_view v) {
    std::string s = trim(v);
    if (s.empty() || s.back() != 's') return std::nullopt;
    s.pop_back();
    const auto slash = s.find('/');
    char* end = nullptr;
    if (slash == std::string::npos) {
        const double x = std::strtod(s.c_str(), &end);
        if (end == s.c_str()) return std::nullopt;
        return x;
    }
    const double n = std::strtod(s.substr(0, slash).c_str(), &end);
    const double d = std::strtod(s.substr(slash + 1).c_str(), &end);
    if (d == 0.0) return std::nullopt;
    return n / d;
}

std::string xml_unescape(std::string s) {
    static const std::pair<const char*, const char*> kEntities[] = {
        {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&amp;", "&"}};
    for (const auto& [from, to] : kEntities) {
        std::string::size_type at = 0;
        const std::string f(from);
        while ((at = s.find(f, at)) != std::string::npos) {
            s.replace(at, f.size(), to);
            at += std::string_view(to).size();
        }
    }
    return s;
}

struct XmlTag {
    std::string name;
    std::map<std::string, std::string> attrs;
    bool closing = false, self_closing = false;
};

// The next tag at or after `pos` (comments, declarations and CDATA skipped);
// false at the end of the text.
bool next_tag(std::string_view text, std::size_t& pos, XmlTag& tag) {
    while (true) {
        const auto lt = text.find('<', pos);
        if (lt == std::string_view::npos) return false;
        if (text.substr(lt, 4) == "<!--") {
            const auto e = text.find("-->", lt + 4);
            if (e == std::string_view::npos) return false;
            pos = e + 3;
            continue;
        }
        if (text.substr(lt, 9) == "<![CDATA[") {
            const auto e = text.find("]]>", lt + 9);
            if (e == std::string_view::npos) return false;
            pos = e + 3;
            continue;
        }
        if (lt + 1 < text.size() && (text[lt + 1] == '?' || text[lt + 1] == '!')) {
            const auto e = text.find('>', lt);
            if (e == std::string_view::npos) return false;
            pos = e + 1;
            continue;
        }
        // The tag's end: the first '>' outside a quoted attribute.
        std::size_t i = lt + 1;
        char quote = 0;
        for (; i < text.size(); ++i) {
            const char ch = text[i];
            if (quote) {
                if (ch == quote) quote = 0;
            } else if (ch == '"' || ch == '\'') {
                quote = ch;
            } else if (ch == '>') {
                break;
            }
        }
        if (i >= text.size()) return false;
        std::string_view body = text.substr(lt + 1, i - lt - 1);
        pos = i + 1;
        tag = XmlTag{};
        if (!body.empty() && body.front() == '/') {
            tag.closing = true;
            body.remove_prefix(1);
        }
        if (!body.empty() && body.back() == '/') {
            tag.self_closing = true;
            body.remove_suffix(1);
        }
        std::size_t k = 0;
        while (k < body.size() && !std::isspace(static_cast<unsigned char>(body[k]))) ++k;
        tag.name = std::string(body.substr(0, k));
        while (k < body.size()) {
            while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
            const auto eq = body.find('=', k);
            if (eq == std::string_view::npos) break;
            const std::string key = trim(body.substr(k, eq - k));
            std::size_t v = eq + 1;
            while (v < body.size() && std::isspace(static_cast<unsigned char>(body[v]))) ++v;
            if (v >= body.size() || (body[v] != '"' && body[v] != '\'')) break;
            const char q = body[v];
            const auto close = body.find(q, v + 1);
            if (close == std::string_view::npos) break;
            tag.attrs[key] = xml_unescape(std::string(body.substr(v + 1, close - v - 1)));
            k = close + 1;
        }
        return true;
    }
}

}  // namespace

int FrameRate::nominal() const { return int(std::lround(value())); }

Result<FrameRate> parse_frame_rate(std::string_view text) {
    const std::string s = trim(text);
    FrameRate r;
    char* end = nullptr;
    if (const auto slash = s.find('/'); slash != std::string::npos) {
        r.num = std::strtoll(s.substr(0, slash).c_str(), &end, 10);
        r.den = std::strtoll(s.substr(slash + 1).c_str(), &end, 10);
    } else {
        const double x = std::strtod(s.c_str(), &end);
        if (end == s.c_str()) return parse_error("Not a frame rate", s);
        // The NTSC rates as editors write them.
        static const std::pair<double, FrameRate> kNtsc[] = {
            {23.976, {24000, 1001}}, {29.97, {30000, 1001}}, {47.952, {48000, 1001}}, {59.94, {60000, 1001}},
            {119.88, {120000, 1001}}};
        bool ntsc = false;
        for (const auto& [v, rate] : kNtsc)
            if (std::fabs(x - v) < 0.01) r = rate, ntsc = true;
        if (!ntsc) {
            if (std::fabs(x - std::round(x)) > 1e-6) return parse_error("Not a frame rate", s);
            r = {static_cast<long long>(std::llround(x)), 1};
        }
    }
    if (r.num <= 0 || r.den <= 0 || r.value() > 1000.0) return parse_error("Not a frame rate", s);
    return r;
}

Result<long long> timecode_frames(std::string_view tc, const FrameRate& rate, bool drop) {
    const std::string s = trim(tc);
    if (!looks_like_timecode(s)) return parse_error("Not a timecode", s);
    const int hh = std::atoi(s.substr(0, 2).c_str()), mm = std::atoi(s.substr(3, 2).c_str());
    const int ss = std::atoi(s.substr(6, 2).c_str()), ff = std::atoi(s.substr(9, 2).c_str());
    const int base = rate.nominal();
    if (mm > 59 || ss > 59 || ff >= base) return parse_error("Not a timecode at this frame rate", s);
    long long frames = (static_cast<long long>(hh) * 3600 + mm * 60 + ss) * base + ff;
    drop = drop || s[8] == ';' || s[8] == '.';
    const bool ntsc = rate.den == 1001 && (base == 30 || base == 60);
    if (drop && ntsc) {
        const int dropped = base == 30 ? 2 : 4;
        const long long minutes = static_cast<long long>(hh) * 60 + mm;
        frames -= dropped * (minutes - minutes / 10);
    }
    return frames;
}

Result<ConformEdit> parse_edl(std::string_view text, const FrameRate& rate) {
    ConformEdit edit;
    edit.source = "edl";
    bool drop = false;
    std::istringstream in{std::string(text)};
    std::string line;
    int line_no = 0;
    // The event a "* FROM CLIP NAME:" line names (an index: events grow).
    std::optional<std::size_t> last;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string t = trim(line);
        if (t.empty()) continue;
        const std::string u = upper(t);
        if (u.rfind("TITLE:", 0) == 0) {
            edit.title = trim(t.substr(6));
            continue;
        }
        if (u.rfind("FCM:", 0) == 0) {
            drop = u.find("NON") == std::string::npos && u.find("DROP") != std::string::npos;
            continue;
        }
        if (t.front() == '*') {
            const std::string up = upper(t);
            const auto at = up.find("FROM CLIP NAME:");
            if (at != std::string::npos && last) edit.events[*last].name = trim(t.substr(at + 15));
            continue;
        }
        const auto w = words(t);
        if (w.size() < 8 || !std::all_of(w[0].begin(), w[0].end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
            continue;   // M2 (speed), SPLIT, comments and the like
        // number reel track transition [duration] src_in src_out rec_in rec_out
        const std::string& track = w[2];
        const std::size_t n = w.size();
        if (!looks_like_timecode(w[n - 1]) || !looks_like_timecode(w[n - 2]))
            return parse_error("An EDL event has no record timecodes", "line " + std::to_string(line_no));
        if (upper(track).find('V') == std::string::npos) {   // audio only
            last.reset();
            continue;
        }
        auto rin = timecode_frames(w[n - 2], rate, drop);
        if (!rin) return rin.error();
        auto rout = timecode_frames(w[n - 1], rate, drop);
        if (!rout) return rout.error();
        if (*rout < *rin) return parse_error("An EDL event ends before it starts", "line " + std::to_string(line_no));
        if (*rout == *rin) {   // the zero-length outgoing half of a dissolve
            last.reset();
            continue;
        }
        // A dissolve's incoming half (same event number, "D nnn") is a shot
        // of its own from the dissolve's start: the cut is placed there.
        ConformEvent e;
        e.record_in = *rin, e.record_out = *rout;
        e.name = w[1] == "BL" || w[1] == "BLACK" ? "event " + w[0] + " (black)" : w[1];
        last = edit.events.size();
        edit.events.push_back(std::move(e));
    }
    if (edit.events.empty()) return parse_error("The EDL has no video events");
    std::sort(edit.events.begin(), edit.events.end(),
              [](const ConformEvent& a, const ConformEvent& b) { return a.record_in < b.record_in; });
    return edit;
}

Result<ConformEdit> parse_fcpxml(std::string_view text, const FrameRate& rate) {
    ConformEdit edit;
    edit.source = "fcpxml";
    std::size_t pos = 0;
    XmlTag tag;
    bool saw_root = false;
    double tc_start = 0.0;
    // To the first <sequence> (its tcStart), then its <spine>.
    while (next_tag(text, pos, tag)) {
        if (tag.closing) continue;
        if (tag.name == "fcpxml") saw_root = true;
        if (tag.name == "project" && edit.title.empty() && tag.attrs.count("name")) edit.title = tag.attrs["name"];
        if (tag.name == "sequence") {
            if (auto t = tag.attrs.find("tcStart"); t != tag.attrs.end())
                if (auto v = rational_seconds(t->second)) tc_start = *v;
        }
        if (tag.name == "spine") break;
    }
    if (!saw_root) return parse_error("This is not an FCPXML file");
    if (tag.name != "spine") return parse_error("The FCPXML has no primary storyline (spine)");
    // The spine's direct children; nested elements (connected clips, effects) skipped.
    int depth = 0;
    const double fps = rate.value();
    int index = 0;
    while (next_tag(text, pos, tag)) {
        if (tag.closing) {
            if (depth == 0) break;   // </spine>
            --depth;
            continue;
        }
        if (depth == 0) {
            ++index;
            const auto off_v = tag.attrs.count("offset") ? rational_seconds(tag.attrs["offset"]) : std::nullopt;
            const auto dur_v = tag.attrs.count("duration") ? rational_seconds(tag.attrs["duration"]) : std::nullopt;
            if (!off_v || !dur_v) return parse_error("An FCPXML clip has no offset or duration", tag.name);
            const double off = off_v.value(), dur = dur_v.value();
            if (tag.name != "gap" && dur > 0.0) {
                ConformEvent e;
                e.record_in = std::llround((off - tc_start) * fps);
                e.record_out = std::llround((off - tc_start + dur) * fps);
                e.name = tag.attrs.count("name") ? tag.attrs["name"] : tag.name + " " + std::to_string(index);
                if (e.record_out > e.record_in) edit.events.push_back(std::move(e));
            }
        }
        if (!tag.self_closing) ++depth;
    }
    if (edit.events.empty()) return parse_error("The FCPXML's storyline has no clips");
    std::sort(edit.events.begin(), edit.events.end(),
              [](const ConformEvent& a, const ConformEvent& b) { return a.record_in < b.record_in; });
    return edit;
}

Result<ConformResult> conform_shots(const ConformEdit& edit, int frame_count, const MasterRequest& base,
                                    std::optional<long long> record_start) {
    if (frame_count <= 0) return make_error(ErrorCode::InvalidArgument, "The programme has no frames");
    if (edit.events.empty()) return make_error(ErrorCode::InvalidArgument, "The edit has no events");
    ConformResult r;
    r.shots.base = base;
    const long long start = record_start ? *record_start : edit.events.front().record_in;
    long long previous_end = 0;
    for (const auto& e : edit.events) {
        long long first = e.record_in - start, end = e.record_out - start;   // [first, end)
        if (end <= 0) {
            r.notes.push_back(e.name + " is before the programme starts; left out");
            continue;
        }
        if (first < 0) {
            r.notes.push_back(e.name + " starts before the programme; cut at its first frame");
            first = 0;
        }
        if (first >= frame_count)
            return make_error(ErrorCode::InvalidArgument, "The edit runs past the end of the movie",
                              e.name + " starts at frame " + std::to_string(first) + " of " + std::to_string(frame_count));
        if (first < previous_end) {
            // Overlapping events (a dissolve the EDL wrote as two cuts): the later one starts where the earlier ends.
            r.notes.push_back(e.name + " overlaps the shot before it; starts where that one ends");
            first = previous_end;
            if (first >= end) continue;
        }
        if (end > frame_count) {
            r.notes.push_back(e.name + " runs past the end of the movie; cut at its last frame");
            end = frame_count;
        }
        ShotGrade g;
        g.name = e.name;
        g.first = int(first);
        g.last = int(end - 1);
        g.source = edit.source;
        r.shots.shots.push_back(std::move(g));
        previous_end = end;
    }
    if (r.shots.shots.empty()) return make_error(ErrorCode::InvalidArgument, "No event of the edit falls in the movie");
    if (auto v = validate_shot_grades(r.shots, frame_count); !v) return v.error();
    return r;
}

}  // namespace rudra
