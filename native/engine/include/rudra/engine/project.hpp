#pragma once
// A RUDRA project (.rudra): a shot and everything done to it, saved so it can
// be reopened tomorrow or handed to a colleague (product item 3, 10 Oct 2026).
//
// JSON, versioned, written atomically (a temporary file renamed over the
// target). What it holds: the shot (a folder of frames, a movie, or a list of
// stills), the frame on screen, the model package and backend, and the
// session: the grade (mode, strength, preserve, Region EV bands, source curve,
// calibration, reference fit), the view peak and the Deliver settings. Painted
// masks travel beside it as <name>.rudra.masks.png (core/masks
// write_mask_set), so a master <name>.exr beside it keeps its own.
//
// Paths are stored as written and, when they sit under the project's folder,
// also relative to it; on load the relative one wins when it exists, so a
// project moved together with its footage still opens.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/masks.hpp"
#include "rudra/engine/session.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kProjectVersion = 1;
inline constexpr const char* kProjectSuffix = ".rudra";

struct Project {
    // "folder" (a folder of frames), "movie" or "files" (stills); "" for none.
    std::string source_kind;
    std::vector<std::filesystem::path> sources;   // one folder or movie, or the stills in order
    int frame = 0;                                // the frame on screen
    std::filesystem::path package;                // the model package in use, empty for none
    std::string backend;                          // BackendChoice::key(), "" for the default
    GradeSnapshot grade;                          // with its masks, if any
    double peak_ev = 0.0;
    bool anchor = true, carry_chroma = true;
    double anchor_knee = 0.9;
    std::string container = "aces";
    std::string app_version;                      // what wrote it, for the record
    std::string masks_file;                       // the masks file the project names, "" for none (read by load_project)
    std::vector<std::string> notes;               // what load_project could not bring back (missing masks), for the log
};

// The session's part of a project (grade, peak, Deliver settings).
Project project_from_session(const Session& s);

// The project as JSON, with paths relative to `project_file`'s folder where they can be.
std::string project_json(const Project& p, const std::filesystem::path& project_file);

// Parse; paths resolved against `project_file`'s folder. ParseError on a file
// that is not a project or is from a newer version.
Result<Project> parse_project(const std::string& json, const std::filesystem::path& project_file);

// Write <path> (and <path>.masks.png when the grade has masks; a stale one is
// removed when it has none). The project and its masks are each replaced
// whole; when the save fails the files already there are left as they were. The suffix is added when missing; the path
// written is returned.
Result<std::filesystem::path> save_project(const std::filesystem::path& path, const Project& p);

// Read a project and its masks. Masks that are missing or unreadable do not
// refuse the project: the grade opens without them and `notes` says so.
Result<Project> load_project(const std::filesystem::path& path);

// Where a project's masks live.
std::filesystem::path project_masks_path(const std::filesystem::path& project_file);

}  // namespace rudra
