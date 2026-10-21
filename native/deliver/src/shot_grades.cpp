#include "rudra/deliver/shot_grades.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <system_error>

#include <nlohmann/json.hpp>

namespace rudra {
namespace fs = std::filesystem;

namespace {

Error invalid(std::string message, std::string detail = {}) {
    return make_error(ErrorCode::InvalidArgument, std::move(message), std::move(detail));
}

double lerp(double a, double b, double t) { return a + (b - a) * t; }

bool same_bands(const std::vector<MasterBand>& a, const std::vector<MasterBand>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].label != b[i].label) return false;
    return true;
}

bool same_anchors(const std::vector<CalibrationPoint>& a, const std::vector<CalibrationPoint>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].code != b[i].code) return false;
    return true;
}

// A key's master params with its masks_file as the file should hold it:
// relative to the shots file's folder when it sits under it.
nlohmann::json request_to_json(const MasterRequest& q, const fs::path& dir) {
    auto j = nlohmann::json::parse(master_request_json(q));
    if (j.contains("masks_file") && !dir.empty()) {
        const fs::path m = fs::path(j["masks_file"].get<std::string>());
        std::error_code ec;
        const fs::path rel = fs::relative(m, dir, ec);
        if (!ec && !rel.empty() && *rel.begin() != "..") j["masks_file"] = rel.generic_string();
    }
    return j;
}

Result<MasterRequest> request_from_json(nlohmann::json j, const fs::path& dir) {
    if (!j.is_object()) return make_error(ErrorCode::ParseError, "A grade in the shots file is not an object");
    if (j.contains("masks_file") && j["masks_file"].is_string() && !dir.empty()) {
        const fs::path m = fs::path(j["masks_file"].get<std::string>());
        if (m.is_relative()) j["masks_file"] = (dir / m).lexically_normal().string();
    }
    return master_request_from_json(j.dump());
}

}  // namespace

ShotGrades shot_grades_from_cuts(const std::vector<int>& cuts, int frame_count, const MasterRequest& base) {
    ShotGrades s;
    s.base = base;
    if (frame_count <= 0) return s;
    std::vector<int> starts{0};
    std::vector<int> sorted = cuts;
    std::sort(sorted.begin(), sorted.end());
    for (int c : sorted)
        if (c > starts.back() && c < frame_count) starts.push_back(c);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        ShotGrade g;
        char name[16];
        std::snprintf(name, sizeof name, "sh%03d", int(i + 1) * 10);
        g.name = name;
        g.first = starts[i];
        g.last = (i + 1 < starts.size() ? starts[i + 1] : frame_count) - 1;
        g.source = "detected";
        s.shots.push_back(std::move(g));
    }
    return s;
}

int shot_index(const ShotGrades& s, int frame) {
    // Shots are sorted and disjoint: the last one starting at or before the frame.
    auto it = std::upper_bound(s.shots.begin(), s.shots.end(), frame,
                               [](int f, const ShotGrade& g) { return f < g.first; });
    if (it == s.shots.begin()) return -1;
    --it;
    return frame <= it->last ? int(it - s.shots.begin()) : -1;
}

MasterRequest blend_grades(const MasterRequest& a, const MasterRequest& b, double t) {
    if (t <= 0.0) return a;
    if (t >= 1.0) return b;
    MasterRequest r = a;   // the stepped settings stay the earlier key's
    r.strength = lerp(a.strength, b.strength, t);
    r.exposure_ev = lerp(a.exposure_ev, b.exposure_ev, t);
    r.anchor_knee = lerp(a.anchor_knee, b.anchor_knee, t);
    r.chroma_knee = lerp(a.chroma_knee, b.chroma_knee, t);
    r.region_softness_stops = lerp(a.region_softness_stops, b.region_softness_stops, t);
    if (same_bands(a.regions, b.regions))
        for (std::size_t i = 0; i < r.regions.size(); ++i) {
            r.regions[i].low_nits = lerp(a.regions[i].low_nits, b.regions[i].low_nits, t);
            r.regions[i].high_nits = lerp(a.regions[i].high_nits, b.regions[i].high_nits, t);
            r.regions[i].ev = lerp(a.regions[i].ev, b.regions[i].ev, t);
        }
    if (same_anchors(a.calibration, b.calibration))
        for (std::size_t i = 0; i < r.calibration.size(); ++i)
            r.calibration[i].nits = lerp(a.calibration[i].nits, b.calibration[i].nits, t);
    return r;
}

MasterRequest grade_at(const ShotGrades& s, int frame) {
    const int i = shot_index(s, frame);
    if (i < 0) return s.base;
    const auto& keys = s.shots[std::size_t(i)].keys;
    if (keys.empty()) return s.base;
    if (frame <= keys.front().frame) return keys.front().grade;
    if (frame >= keys.back().frame) return keys.back().grade;
    auto next = std::upper_bound(keys.begin(), keys.end(), frame,
                                 [](int f, const GradeKey& k) { return f < k.frame; });
    const auto& b = *next;
    const auto& a = *std::prev(next);
    if (a.frame == frame) return a.grade;
    const double t = double(frame - a.frame) / double(b.frame - a.frame);
    return blend_grades(a.grade, b.grade, t);
}

Result<void> set_key(ShotGrades& s, int frame, const MasterRequest& grade) {
    const int i = shot_index(s, frame);
    if (i < 0) return invalid("That frame is outside every shot", std::to_string(frame));
    auto& keys = s.shots[std::size_t(i)].keys;
    auto it = std::lower_bound(keys.begin(), keys.end(), frame, [](const GradeKey& k, int f) { return k.frame < f; });
    if (it != keys.end() && it->frame == frame) it->grade = grade;
    else keys.insert(it, GradeKey{frame, grade});
    return {};
}

bool remove_key(ShotGrades& s, int frame) {
    const int i = shot_index(s, frame);
    if (i < 0) return false;
    auto& keys = s.shots[std::size_t(i)].keys;
    auto it = std::find_if(keys.begin(), keys.end(), [&](const GradeKey& k) { return k.frame == frame; });
    if (it == keys.end()) return false;
    keys.erase(it);
    return true;
}

Result<void> validate_shot_grades(const ShotGrades& s, int frame_count) {
    int previous_last = -1;
    for (const auto& g : s.shots) {
        const std::string who = g.name.empty() ? "shot at " + std::to_string(g.first) : g.name;
        if (g.first < 0 || g.last < g.first) return invalid("A shot ends before it starts", who);
        if (frame_count > 0 && g.last >= frame_count)
            return invalid("A shot runs past the end of the programme",
                           who + " ends at " + std::to_string(g.last) + " of " + std::to_string(frame_count));
        if (g.first <= previous_last) return invalid("Shots overlap or are out of order", who);
        previous_last = g.last;
        int previous_key = -1;
        for (const auto& k : g.keys) {
            if (k.frame < g.first || k.frame > g.last) return invalid("A key is outside its shot", who);
            if (k.frame <= previous_key) return invalid("A shot's keys are out of order or repeated", who);
            previous_key = k.frame;
        }
    }
    return {};
}

std::string shot_grades_json(const ShotGrades& s, const fs::path& file) {
    const fs::path dir = file.empty() ? fs::path() : file.parent_path();
    nlohmann::json j{{"version", kShotGradesVersion}, {"base", request_to_json(s.base, dir)}};
    auto shots = nlohmann::json::array();
    for (const auto& g : s.shots) {
        auto keys = nlohmann::json::array();
        for (const auto& k : g.keys) keys.push_back({{"frame", k.frame}, {"grade", request_to_json(k.grade, dir)}});
        shots.push_back({{"name", g.name}, {"first", g.first}, {"last", g.last}, {"source", g.source}, {"keys", keys}});
    }
    j["shots"] = shots;
    return j.dump(2);
}

Result<ShotGrades> parse_shot_grades(const std::string& text, const fs::path& file) {
    const fs::path dir = file.empty() ? fs::path() : file.parent_path();
    ShotGrades s;
    try {
        const auto j = nlohmann::json::parse(text);
        if (!j.is_object() || !j.contains("shots"))
            return make_error(ErrorCode::ParseError, "This is not a shots file", file.string());
        const int version = j.value("version", 0);
        if (version < 1 || version > kShotGradesVersion)
            return make_error(ErrorCode::ContractMismatch, "This shots file is from a newer RUDRA",
                              "version " + std::to_string(version));
        if (j.contains("base")) {
            auto b = request_from_json(j["base"], dir);
            if (!b) return b.error();
            s.base = *b;
        }
        for (const auto& g : j.at("shots")) {
            ShotGrade shot;
            shot.name = g.value("name", std::string());
            shot.first = g.at("first").get<int>();
            shot.last = g.at("last").get<int>();
            shot.source = g.value("source", std::string("manual"));
            for (const auto& k : g.value("keys", nlohmann::json::array())) {
                auto q = request_from_json(k.at("grade"), dir);
                if (!q) return q.error();
                shot.keys.push_back(GradeKey{k.at("frame").get<int>(), *q});
            }
            s.shots.push_back(std::move(shot));
        }
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The shots file could not be read", e.what());
    }
    if (auto v = validate_shot_grades(s); !v) return v.error();
    return s;
}

Result<void> save_shot_grades(const fs::path& path, const ShotGrades& s) {
    if (auto v = validate_shot_grades(s); !v) return v.error();
    const std::string text = shot_grades_json(s, path);
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return make_error(ErrorCode::IoError, "The shots file could not be written", tmp.string());
        out << text;
        if (!out.flush()) return make_error(ErrorCode::IoError, "The shots file could not be written", tmp.string());
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return make_error(ErrorCode::IoError, "The shots file could not be replaced", path.string());
    }
    return {};
}

Result<ShotGrades> load_shot_grades(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The shots file is missing", path.string());
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse_shot_grades(text, path);
}

}  // namespace rudra
