#pragma once
// Conform by shot (roadmap 5.4, 20 Oct 2026): a programme's shots from its
// edit, so a whole programme grades and delivers in one pass with a grade per
// shot (deliver/shot_grades.hpp, roadmap 4.3).
//
// Three sources of shots:
//   - a CMX3600 EDL: the video events' record in and out, in timecode at the
//     programme's rate (drop frame when the timecode says so: ';' or
//     "FCM: DROP FRAME"); "* FROM CLIP NAME:" names the shot;
//   - an FCPXML (1.8 to 1.11): the primary storyline's clips (<spine>'s
//     direct children), offset and duration as rational seconds, less the
//     sequence's tcStart; gaps are not shots;
//   - cuts detected in the movie itself (video/convert.hpp detect_cuts).
// The record timeline starts at the earliest event unless a record start is
// given (an EDL's 01:00:00:00, say). Frames between events are in no shot and
// take the base grade.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/deliver/shot_grades.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// A frame rate as a ratio: 24000/1001, 25/1.
struct FrameRate {
    long long num = 24, den = 1;
    double value() const { return double(num) / double(den); }
    int nominal() const;   // the timecode base: 24 for 23.976, 30 for 29.97
};

// "24", "25", "24000/1001", "29.97", "23.976" (ffprobe's and editors' forms).
Result<FrameRate> parse_frame_rate(std::string_view text);

// "HH:MM:SS:FF" (or ';' / '.' before the frames for drop frame) to a frame
// count at `rate`. Drop frame applies to 29.97 and 59.94 when `drop` (or the
// separator) says so.
Result<long long> timecode_frames(std::string_view tc, const FrameRate& rate, bool drop = false);

struct ConformEvent {
    std::string name;        // the clip name, else the reel, else "event N"
    long long record_in = 0, record_out = 0;   // timeline frames, [in, out)
};

struct ConformEdit {
    std::string source;      // "edl" or "fcpxml"
    std::string title;
    std::vector<ConformEvent> events;   // in record order
    bool drop_frame = false;  // the edit's timecode is drop frame (FCM: DROP FRAME, tcFormat="DF")
};

Result<ConformEdit> parse_edl(std::string_view text, const FrameRate& rate);
Result<ConformEdit> parse_fcpxml(std::string_view text, const FrameRate& rate);

// The edit as shots over a programme of `frame_count` frames; record_start:
// the timeline frame of programme frame 0 (unset: the earliest record in).
// Refused when a shot would start past the programme's end; a last shot that
// runs past it is cut at the end and `notes` says so.
struct ConformResult {
    ShotGrades shots;
    std::vector<std::string> notes;
};
Result<ConformResult> conform_shots(const ConformEdit& edit, int frame_count, const MasterRequest& base = {},
                                    std::optional<long long> record_start = std::nullopt);

}  // namespace rudra
