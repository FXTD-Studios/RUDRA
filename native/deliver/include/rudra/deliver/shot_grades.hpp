#pragma once
// Shot keyframes (roadmap 4.3, 20 Oct 2026): the grade of a programme kept per
// shot, keyed at frames inside the shot, and reset at every cut.
//
// A programme is a run of frames (a movie, a frame folder). Its shots are
// inclusive frame ranges [first, last], in order and not overlapping; frames
// outside every shot take the base grade. A shot holds keys: a full master
// request (deliver/master.hpp) at a frame of that shot.
//
// grade_at(frame):
//   - outside every shot, or in a shot with no keys: the base grade;
//   - before the shot's first key or after its last: that key, held;
//   - between two keys: the earlier key, with the continuous settings moved
//     linearly to the later one (strength, exposure_ev, anchor_knee,
//     chroma_knee, region_softness_stops; each Region EV band's low, high and
//     ev when both keys have the same bands; each calibration anchor's nits
//     when both keys have the same anchors at the same codes). Everything else
//     (mode, source curve, preserve, clip fill and its peaks, masks, the
//     reference, the container) steps at the later key.
// Keys never reach across a cut: a shot's grade comes from its own keys only.
//
// The file (<name>.shots.json, written whole through a temporary file):
//   {"version": 1, "base": {master params},
//    "shots": [{"name": "sh010", "first": 0, "last": 47, "source": "edl",
//               "keys": [{"frame": 0, "grade": {master params}}, ...]}, ...]}
// Master params are master_request_json's; a relative masks_file is resolved
// against the file's folder on read and written back relative to it.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/deliver/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kShotGradesVersion = 1;

struct GradeKey {
    int frame = 0;          // a programme frame, inside its shot
    MasterRequest grade;
};

struct ShotGrade {
    std::string name;       // "sh010", an EDL event's name, or "" for an unnamed shot
    int first = 0, last = 0;   // inclusive programme frames
    std::string source;     // where the boundary came from: "edl", "fcpxml", "detected", "manual"
    std::vector<GradeKey> keys;   // sorted by frame, at most one per frame
};

struct ShotGrades {
    MasterRequest base;
    std::vector<ShotGrade> shots;   // sorted by first, not overlapping
};

// Shots from cut frames (each the first frame of a new shot) over a programme
// of `frame_count` frames, named sh010, sh020 ...; no keys. Cuts out of range,
// at 0 and repeated are ignored.
ShotGrades shot_grades_from_cuts(const std::vector<int>& cuts, int frame_count, const MasterRequest& base = {});

// The shot holding `frame`, or -1.
int shot_index(const ShotGrades& s, int frame);

// The grade for one frame (above).
MasterRequest grade_at(const ShotGrades& s, int frame);

// Set (or replace) the key at `frame` in the shot that holds it; refused for
// a frame outside every shot. Remove: false when there was none.
Result<void> set_key(ShotGrades& s, int frame, const MasterRequest& grade);
bool remove_key(ShotGrades& s, int frame);

// Shots in order, inside [0, frame_count) when frame_count > 0, not
// overlapping, first <= last; keys inside their shot, sorted, unique.
Result<void> validate_shot_grades(const ShotGrades& s, int frame_count = 0);

// The interpolation itself: `a` moved toward `b` by t in [0, 1] (above).
MasterRequest blend_grades(const MasterRequest& a, const MasterRequest& b, double t);

std::string shot_grades_json(const ShotGrades& s, const std::filesystem::path& file = {});
Result<ShotGrades> parse_shot_grades(const std::string& json, const std::filesystem::path& file = {});
Result<void> save_shot_grades(const std::filesystem::path& path, const ShotGrades& s);
Result<ShotGrades> load_shot_grades(const std::filesystem::path& path);

}  // namespace rudra
