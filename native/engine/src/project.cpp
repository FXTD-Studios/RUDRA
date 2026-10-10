#include "rudra/engine/project.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/source_curve.hpp"
#include "rudra/platform/io_error.hpp"

namespace rudra {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string utf8(const fs::path& p) {
    const std::u8string u = p.generic_u8string();
    return std::string(u.begin(), u.end());
}
fs::path from_utf8(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

// A path for the file: absolute as given, plus relative when it is under the project's folder.
json path_json(const fs::path& p, const fs::path& base) {
    json j = {{"path", utf8(p)}};
    std::error_code ec;
    const fs::path rel = fs::relative(p, base, ec);
    if (!ec && !rel.empty() && *rel.begin() != "..") j["relative"] = utf8(rel);
    return j;
}

fs::path path_from(const json& j, const fs::path& base) {
    if (j.is_string()) return from_utf8(j.get<std::string>());
    if (j.contains("relative")) {
        const fs::path rel = base / from_utf8(j["relative"].get<std::string>());
        std::error_code ec;
        if (fs::exists(rel, ec)) return rel.lexically_normal();
    }
    return from_utf8(j.value("path", std::string()));
}

json reference_json(const ReferenceFit& r) {
    return {{"file", r.file},
            {"target_log2_nits", r.target_log2_nits},
            {"samples_per_code", r.samples_per_code},
            {"residual_mean", r.residual_mean},
            {"residual_p95", r.residual_p95},
            {"samples", r.samples},
            {"codes_seen", r.codes_seen}};
}

ReferenceFit reference_from(const json& j) {
    ReferenceFit r;
    r.file = j.value("file", std::string());
    r.target_log2_nits = j.value("target_log2_nits", std::vector<float>{});
    r.samples_per_code = j.value("samples_per_code", std::vector<int>{});
    r.residual_mean = j.value("residual_mean", 0.0);
    r.residual_p95 = j.value("residual_p95", 0.0);
    r.samples = j.value("samples", 0LL);
    r.codes_seen = j.value("codes_seen", 0);
    return r;
}

}  // namespace

fs::path project_masks_path(const fs::path& project_file) {
    // <name>.rudra.masks.png: not <name>.masks.png, which a master <name>.exr
    // in the same folder writes for its own masks.
    fs::path p = project_file;
    p += ".masks.png";
    return p;
}

Project project_from_session(const Session& s) {
    Project p;
    p.grade = s.grade;
    p.peak_ev = s.peak_ev;
    p.anchor = s.anchor;
    p.carry_chroma = s.carry_chroma;
    p.anchor_knee = s.anchor_knee;
    p.container = s.container;
    return p;
}

std::string project_json(const Project& p, const fs::path& project_file) {
    const fs::path base = project_file.parent_path();
    json sources = json::array();
    for (const auto& s : p.sources) sources.push_back(path_json(s, base));
    json regions = json::array();
    for (const auto& r : p.grade.regions)
        regions.push_back({{"label", r.label}, {"low_nits", r.low_nits}, {"high_nits", r.high_nits}, {"ev", r.ev}});
    json calibration = json::array();
    for (const auto& c : p.grade.calibration) calibration.push_back({{"code", c.code}, {"nits", c.nits}});
    json j = {
        {"rudra_project", kProjectVersion},
        {"app_version", p.app_version},
        {"shot", {{"kind", p.source_kind}, {"sources", sources}, {"frame", p.frame}}},
        {"model",
         {{"package", p.package.empty() ? json() : path_json(p.package, base)},
          {"backend", p.backend},
          {"use", p.use_model}}},
        {"grade",
         {{"mode", p.grade.mode},
          {"strength", p.grade.strength},
          {"preserve", p.grade.preserve},
          {"regions", regions},
          {"source", p.grade.source},
          {"calibration", calibration},
          {"reference", p.grade.reference.empty() ? json() : reference_json(p.grade.reference)},
          {"masks", p.grade.masks && !p.grade.masks->empty()
                        ? json(utf8(project_masks_path(project_file).filename()))
                        : json()}}},
        {"view", {{"peak_ev", p.peak_ev}}},
        {"deliver",
         {{"anchor", p.anchor}, {"anchor_knee", p.anchor_knee}, {"carry_chroma", p.carry_chroma}, {"container", p.container}}},
    };
    return j.dump(2) + "\n";
}

Result<Project> parse_project(const std::string& text, const fs::path& project_file) {
    json j;
    try {
        j = json::parse(text);
    } catch (const json::exception& e) {
        return make_error(ErrorCode::ParseError, "This is not a RUDRA project.", e.what());
    }
    if (!j.is_object() || !j.contains("rudra_project"))
        return make_error(ErrorCode::ParseError, "This is not a RUDRA project.", project_file.string());
    if (!j["rudra_project"].is_number_integer())
        return make_error(ErrorCode::ParseError, "This is not a RUDRA project.", project_file.string());
    const int version = j["rudra_project"].get<int>();
    if (version > kProjectVersion)
        return make_error(ErrorCode::ContractMismatch,
                          "This project was saved by a newer RUDRA (project version " + std::to_string(version) +
                              "); update RUDRA to open it.",
                          project_file.string());
    const fs::path base = project_file.parent_path();
    Project p;
    try {
        p.app_version = j.value("app_version", std::string());
        const json& shot = j.value("shot", json::object());
        p.source_kind = shot.value("kind", std::string());
        for (const auto& s : shot.value("sources", json::array())) p.sources.push_back(path_from(s, base));
        p.frame = shot.value("frame", 0);
        const json& model = j.value("model", json::object());
        if (model.contains("package") && !model["package"].is_null()) p.package = path_from(model["package"], base);
        p.backend = model.value("backend", std::string());
        p.use_model = model.value("use", false);
        const json& g = j.value("grade", json::object());
        p.grade.mode = g.value("mode", std::string("all"));
        p.grade.strength = g.value("strength", 1.0);
        p.grade.preserve = g.value("preserve", true);
        if (g.contains("regions")) {
            p.grade.regions.clear();
            for (const auto& r : g["regions"])
                p.grade.regions.push_back({r.value("label", std::string()), r.value("low_nits", 0.0),
                                           r.value("high_nits", 0.0), r.value("ev", 0.0)});
        }
        p.grade.source = g.value("source", std::string("unknown"));
        for (const auto& c : g.value("calibration", json::array()))
            p.grade.calibration.push_back({c.value("code", 0), c.value("nits", 0.0)});
        if (g.contains("reference") && !g["reference"].is_null()) p.grade.reference = reference_from(g["reference"]);
        if (g.contains("masks") && g["masks"].is_string()) p.masks_file = g["masks"].get<std::string>();
        const json& view = j.value("view", json::object());
        p.peak_ev = view.value("peak_ev", 0.0);
        const json& d = j.value("deliver", json::object());
        p.anchor = d.value("anchor", true);
        p.anchor_knee = d.value("anchor_knee", 0.9);
        p.carry_chroma = d.value("carry_chroma", true);
        p.container = d.value("container", std::string("aces"));
    } catch (const json::exception& e) {
        return make_error(ErrorCode::ParseError, "This RUDRA project is damaged.", e.what());
    }
    // What the window indexes or switches on must be in range: a hand-edited
    // file is refused rather than drawn out of bounds.
    auto damaged = [&](const std::string& what) {
        return make_error(ErrorCode::ParseError, "This RUDRA project is damaged (" + what + ").", project_file.string());
    };
    const auto& modes = {"all", "highlights", "shadows", "off"};
    if (std::find(modes.begin(), modes.end(), p.grade.mode) == modes.end()) return damaged("mode " + p.grade.mode);
    if (!parse_source_curve(p.grade.source)) return damaged("source " + p.grade.source);
    // A slot nobody has clicked yet is code -1 (Session::set_calibration).
    for (const auto& c : p.grade.calibration)
        if (c.code < -1 || c.code > 255 || !(c.nits >= 0.0)) return damaged("a calibration point");
    if (p.grade.calibration.size() > 16) return damaged("the calibration");
    // The ranges the controls have; a hand-edited value outside them is pulled in.
    auto finite_or = [](double v, double fallback) { return std::isfinite(v) ? v : fallback; };
    p.grade.strength = std::clamp(finite_or(p.grade.strength, 1.0), 0.0, 2.0);
    p.peak_ev = std::clamp(finite_or(p.peak_ev, 0.0), -1.0, 5.0);
    p.anchor_knee = std::clamp(finite_or(p.anchor_knee, 0.9), 0.5, 0.99);
    for (auto& r : p.grade.regions) r.ev = std::clamp(finite_or(r.ev, 0.0), -4.0, 4.0);
    if (p.container != "aces" && p.container != "linear") return damaged("container " + p.container);
    if (!p.grade.reference.empty() &&
        (p.grade.reference.target_log2_nits.size() != 256 || p.grade.reference.samples_per_code.size() != 256))
        return damaged("the reference fit");
    if (p.source_kind != "" && p.source_kind != "folder" && p.source_kind != "movie" && p.source_kind != "files")
        return damaged("shot kind " + p.source_kind);
    if (p.frame < 0) p.frame = 0;
    return p;
}

Result<fs::path> save_project(const fs::path& path_in, const Project& p) {
    fs::path path = path_in;
    if (path.extension() != kProjectSuffix) path += kProjectSuffix;
    std::error_code ec;
    if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);
    const std::string text = project_json(p, path);
    // The masks go to a temporary file first and take their place only once
    // the project has: a save that fails leaves the old project and its old
    // masks as they were.
    const fs::path masks = project_masks_path(path);
    fs::path masks_tmp = masks;
    masks_tmp += ".tmp";
    const bool has_masks = p.grade.masks && !p.grade.masks->empty();
    if (has_masks) {
        if (auto w = write_mask_set(masks_tmp, *p.grade.masks); !w) {
            fs::remove(masks_tmp, ec);
            return w.error();
        }
    }
    auto drop_tmp = [&] { if (has_masks) fs::remove(masks_tmp, ec); };
    // The old masks step aside (and come back if the project cannot be
    // written), so a project never sits beside masks from another save: a
    // failure after the project is in place reads as "masks missing", never
    // as the previous save's masks.
    fs::path masks_old = masks;
    masks_old += ".old";
    bool moved_old = false;
    if (fs::is_regular_file(masks, ec)) {
        fs::remove(masks_old, ec);
        fs::rename(masks, masks_old, ec);
        moved_old = !ec;
    }
    auto restore_old = [&] {
        if (moved_old) fs::rename(masks_old, masks, ec);
    };
    // Atomic: a crash mid-save leaves the previous project, never half of one.
    fs::path tmp = path;
    tmp += ".tmp";
    errno = 0;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            const int err = errno;
            drop_tmp();
            restore_old();
            return make_error(ErrorCode::IoError, "The project could not be saved (" + io_failure_reason(tmp, err, text.size()) + ").",
                              path.string());
        }
        f.write(text.data(), std::streamsize(text.size()));
        f.close();
        if (!f) {
            const int err = errno;
            const std::string reason = io_failure_reason(tmp, err, text.size());
            fs::remove(tmp, ec);
            drop_tmp();
            restore_old();
            return make_error(ErrorCode::IoError, "The project could not be saved (" + reason + ").", path.string());
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        const std::string reason = ec.message();
        fs::remove(tmp, ec);
        drop_tmp();
        restore_old();
        return make_error(ErrorCode::IoError, "The project could not be saved (" + reason + ").", path.string());
    }
    if (moved_old) fs::remove(masks_old, ec);
    if (has_masks) {
        fs::rename(masks_tmp, masks, ec);
        if (ec) {
            const std::string reason = ec.message();
            fs::remove(masks_tmp, ec);
            return make_error(ErrorCode::IoError, "The project was saved but its painted masks were not (" + reason + ").",
                              masks.string());
        }
    }
    return path;
}

Result<Project> load_project(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return make_error(ErrorCode::NotFound, "The project could not be opened.", path.string());
    std::stringstream ss;
    ss << f.rdbuf();
    auto p = parse_project(ss.str(), path);
    if (!p) return p;
    if (!p->masks_file.empty()) {
        // The one the project's own name gives, else the name it stored (a
        // bare file name beside it: a project renamed in Explorer or Finder).
        // Own name first: a session kept aside as previous-session.rudra still
        // stores last-session's name, which the next autosave reuses.
        std::error_code ec;
        fs::path masks = project_masks_path(path);
        const fs::path stored = from_utf8(p->masks_file);
        if (!fs::is_regular_file(masks, ec) && stored.has_filename() && stored == stored.filename())
            masks = path.parent_path() / stored;
        if (!fs::is_regular_file(masks, ec)) {
            p->notes.push_back("the painted masks (" + p->masks_file + ") are missing; the grade opens without them");
        } else if (auto m = read_mask_set(masks); !m) {
            p->notes.push_back("the painted masks could not be read (" + m.error().message + "); the grade opens without them");
        } else {
            p->grade.masks = std::make_shared<const MaskSet>(std::move(*m));
        }
    }
    return p;
}

}  // namespace rudra
