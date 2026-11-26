#include "fill_store.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "rudra/core/gamut.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"

namespace rudra::ofx_fill {
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// A running job with no word for this long has stopped (the host closed, the computer slept): ofx_job.py
// writes status.json at least every 30 seconds, loading the model included.
constexpr double kStaleSeconds = 300.0;

fs::path env_path(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? fs::path(v) : fs::path();
}

fs::path utf8_path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::string hex(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Written whole or not at all: <file>.part, then renamed over the file.
bool write_text(const fs::path& path, const std::string& text) {
    const fs::path part = path.string() + ".part";
    {
        std::ofstream f(part, std::ios::binary | std::ios::trunc);
        f << text;
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(part, path, ec);
    if (ec) {   // Windows will not rename over a file another reader holds open: once more after removing it
        fs::remove(path, ec);
        fs::rename(part, path, ec);
    }
    return !ec;
}

std::optional<json> read_json(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    return j;
}

std::string read_text(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::mutex& store_mutex() {
    static std::mutex m;
    return m;
}

// The jobs this host started, by folder; never destroyed (the host may unload the plug-in in any order).
std::map<std::string, std::unique_ptr<Process>>& jobs() {
    static auto* m = new std::map<std::string, std::unique_ptr<Process>>();
    return *m;
}

fs::path master_path(const fs::path& d, int index) { return d / "master" / (frame_stem(index) + ".exr"); }
fs::path codes_path(const fs::path& d, int index) { return d / "codes" / (frame_stem(index) + ".tif"); }
fs::path hash_path(const fs::path& d, int index) { return d / "hash" / (frame_stem(index) + ".txt"); }

bool stored(const fs::path& d, int index) {
    std::error_code ec;
    return fs::is_regular_file(master_path(d, index), ec) && fs::is_regular_file(codes_path(d, index), ec) &&
           fs::is_regular_file(hash_path(d, index), ec);
}

int count_stored(const fs::path& d, int n) {
    int k = 0;
    for (int i = 0; i < n; ++i) k += stored(d, i) ? 1 : 0;
    return k;
}

State state_of(const std::string& s) {
    if (s == "running") return State::Running;
    if (s == "done") return State::Done;
    if (s == "failed") return State::Failed;
    if (s == "cancelled") return State::Cancelled;
    return State::None;
}

Status read_status(const fs::path& d, int frames) {
    Status s;
    std::error_code ec;
    if (!fs::is_directory(d, ec)) return s;
    if (auto j = read_json(d / "status.json")) {
        s.state = state_of(j->value("state", std::string()));
        s.done = j->value("done", 0);
        s.of = j->value("of", frames);
        s.message = j->value("message", std::string());
        if (s.state == State::Done && !fs::is_directory(d / "fill", ec)) s.state = State::None;   // cleared by hand
        if (s.state == State::Running) {
            const auto it = jobs().find(utf8(d));
            const bool ours_ended = it != jobs().end() && it->second && !it->second->running();
            if (ours_ended || now_seconds() - j->value("updated", 0.0) > kStaleSeconds) {
                s.state = State::Failed;
                s.message = ours_ended ? "the AI fill ended without finishing (its log: " + utf8(d / "job.log") + ")"
                                       : "no word from the AI fill for five minutes (the host closed, or the computer slept)";
            }
        }
        if (s.state != State::None) return s;
    }
    if (fs::is_regular_file(d / "collect", ec)) {
        s.state = State::Collecting;
        s.done = count_stored(d, frames);
        s.of = frames;
    }
    return s;
}

Result<void> check(const FillRequest& r) {
    if (r.shot_id.empty() || r.shot_id.find_first_of("/\\.:") != std::string::npos)
        return make_error(ErrorCode::InvalidArgument, "AI fill: this instance has no shot id yet (press Generate).");
    if (r.last < r.first) return make_error(ErrorCode::InvalidArgument, "AI fill: the clip has no frames.");
    if (r.frames() > kMaxFrames)
        return make_error(ErrorCode::InvalidArgument,
                          "AI fill: the clip has " + std::to_string(r.frames()) + " frames; at most " + std::to_string(kMaxFrames) +
                              " (cut the shot shorter).");
    if (r.condition != "bt1886" && r.condition != "aces") return make_error(ErrorCode::InvalidArgument, "AI fill: unknown route.");
    return {};
}

}  // namespace

std::string FillRequest::key() const {
    json j = {{"first", first}, {"last", last},       {"condition", condition}, {"prompt", prompt},
              {"seed", seed},   {"region", region_json}, {"master", master_key}, {"recipe", "ai-fill-v5 rudra-1.3.1"}};
    if (steps != 30) j["steps"] = steps;   // 30, the default, keeps the keys fills were stored under before Steps
    const std::string s = j.dump();
    return hex(xxh64(std::as_bytes(std::span<const char>(s.data(), s.size()))));
}

fs::path root() {
    if (fs::path p = env_path("RUDRA_OFX_FILL_ROOT"); !p.empty()) return p;
#ifdef _WIN32
    const fs::path base = env_path("LOCALAPPDATA");
#elif defined(__APPLE__)
    const fs::path base = env_path("HOME").empty() ? fs::path() : env_path("HOME") / "Library" / "Application Support";
#else
    const fs::path base = !env_path("XDG_DATA_HOME").empty() ? env_path("XDG_DATA_HOME")
                          : env_path("HOME").empty()         ? fs::path()
                                                             : env_path("HOME") / ".local" / "share";
#endif
    return (base.empty() ? fs::temp_directory_path() : base) / "FXTD Studios" / "RUDRA" / "ofx_fill";
}

fs::path folder(const FillRequest& r) { return root() / r.shot_id / r.key(); }

std::string frame_stem(int index) {
    char b[24];
    std::snprintf(b, sizeof b, "f.%06d", index);
    return b;
}

std::uint64_t codes_hash(const SdrImage& codes) {
    const PlanarBuffer& p = codes.buffer();
    std::vector<std::uint8_t> b(p.span().size() + 8);
    std::size_t i = 0;
    for (float v : p.span()) b[i++] = std::uint8_t(std::lround(std::clamp(std::isfinite(v) ? v : 0.0f, 0.0f, 1.0f) * 255.0f));
    const std::uint32_t w = std::uint32_t(codes.width()), h = std::uint32_t(codes.height());
    std::memcpy(b.data() + i, &w, 4);
    std::memcpy(b.data() + i + 4, &h, 4);
    return xxh64(std::as_bytes(std::span<const std::uint8_t>(b)));
}

Result<void> begin_collect(const FillRequest& r) {
    if (auto ok = check(r); !ok) return ok.error();
    std::lock_guard<std::mutex> lock(store_mutex());
    const fs::path d = folder(r);
    std::error_code ec;
    for (const char* sub : {"codes", "master", "hash"}) fs::create_directories(d / sub, ec);
    if (ec) return make_error(ErrorCode::IoError, "AI fill: could not make its folder.", utf8(d));
    const Status s = read_status(d, r.frames());
    if (s.state == State::Done || s.state == State::Running) return {};   // nothing to collect
    if (s.state == State::Failed || s.state == State::Cancelled) fs::remove(d / "status.json", ec);   // a new start
    fs::remove(d / "cancel", ec);
    json req = {{"shot_id", r.shot_id}, {"key", r.key()}, {"first", r.first}, {"last", r.last},   {"frames", r.frames()},
                {"condition", r.condition}, {"prompt", r.prompt}, {"seed", r.seed}, {"steps", r.steps}, {"master", r.master_key},
                {"created", now_seconds()}};
    req["region"] = r.region_json.empty() ? json() : json::parse(r.region_json, nullptr, false);
    if (req["region"].is_discarded()) return make_error(ErrorCode::InvalidArgument, "AI fill: the luminance limit is not valid JSON.");
    if (auto old = read_json(d / "request.json")) {   // the size the first stored frame set
        if (old->contains("width")) req["width"] = (*old)["width"], req["height"] = (*old)["height"];
    }
    if (!write_text(d / "request.json", req.dump(1)) || !write_text(d / "collect", "Generate pressed\n"))
        return make_error(ErrorCode::IoError, "AI fill: could not write to its folder.", utf8(d));
    return {};
}

Status status(const FillRequest& r) {
    if (!check(r)) return {};
    std::lock_guard<std::mutex> lock(store_mutex());
    return read_status(folder(r), r.frames());
}

Result<bool> store_frame(const FillRequest& r, int frame, const SdrImage& codes, const PlanarBuffer& master_ap1) {
    if (auto ok = check(r); !ok) return ok.error();
    const int index = frame - r.first;
    if (index < 0 || index >= r.frames()) return false;   // a frame outside the clip (a handle): not stored
    if (master_ap1.channels() != 3 || master_ap1.width() != codes.width() || master_ap1.height() != codes.height())
        return make_error(ErrorCode::InvalidArgument, "AI fill: the master and the codes differ in size.");
    const fs::path d = folder(r);
    std::error_code ec;
    {
        std::lock_guard<std::mutex> lock(store_mutex());
        if (!fs::is_regular_file(d / "collect", ec)) return false;   // not collecting (Cancel, or done)
        // Every frame of one size: the first stored sets it (a proxy render is a different size, and refused).
        auto req = read_json(d / "request.json");
        if (!req) return make_error(ErrorCode::IoError, "AI fill: its request.json is missing (press Generate again).");
        if (!req->contains("width")) {
            (*req)["width"] = codes.width(), (*req)["height"] = codes.height();
            write_text(d / "request.json", req->dump(1));
        } else if ((*req)["width"] != codes.width() || (*req)["height"] != codes.height()) {
            return make_error(ErrorCode::InvalidArgument,
                              "AI fill: this frame is " + std::to_string(codes.width()) + " x " + std::to_string(codes.height()) +
                                  ", the stored ones " + (*req)["width"].dump() + " x " + (*req)["height"].dump() +
                                  " (render at full size, not a proxy).");
        }
    }
    const std::string h = hex(codes_hash(codes));
    if (!(stored(d, index) && read_text(hash_path(d, index)) == h)) {
        const fs::path tif = codes_path(d, index), exr = master_path(d, index);
        const fs::path tif_part = tif.string() + ".part", exr_part = exr.string() + ".part";
        if (auto ok = ofx_engine::write_tiff16(tif_part, codes); !ok) return ok.error();
        const ExrCompression zip = exr_zip_available() ? ExrCompression::Zip : ExrCompression::None;
        if (auto ok = write_acescg_exr(exr_part, master_ap1, Primaries::Ap1, 1.0, {{"rudra:ofxFrame", std::to_string(frame)}}, true,
                                       nullptr, {}, nullptr, zip);
            !ok)
            return ok.error();
        fs::rename(tif_part, tif, ec);
        if (!ec) fs::rename(exr_part, exr, ec);
        if (ec || !write_text(hash_path(d, index), h)) return make_error(ErrorCode::IoError, "AI fill: could not store the frame.", utf8(exr));
    }
    std::lock_guard<std::mutex> lock(store_mutex());
    return fs::is_regular_file(d / "collect", ec) && !fs::exists(d / "status.json", ec) && count_stored(d, r.frames()) == r.frames();
}

Result<void> start_job(const FillRequest& r, const ofx_engine::GenFillConfig& cfg) {
    if (auto ok = check(r); !ok) return ok.error();
    std::lock_guard<std::mutex> lock(store_mutex());
    const fs::path d = folder(r);
    const Status s = read_status(d, r.frames());
    if (s.state == State::Running || s.state == State::Done) return {};
    std::error_code ec;
    if (count_stored(d, r.frames()) != r.frames()) return make_error(ErrorCode::InvalidArgument, "AI fill: frames are still missing.");
    fs::remove(d / "cancel", ec);
    fs::remove_all(d / "fill.part", ec);
    const json st = {{"state", "running"}, {"done", 0}, {"of", r.frames()}, {"message", "starting"}, {"updated", now_seconds()}};
    if (!write_text(d / "status.json", st.dump())) return make_error(ErrorCode::IoError, "AI fill: could not write status.json.", utf8(d));
    const std::vector<std::string> argv = {utf8(cfg.python), utf8(cfg.repo / "rudra" / "genfill" / "ofx_job.py"), "--dir", utf8(d),
                                           "--repo", utf8(cfg.repo), "--base", utf8(cfg.base), "--lora", utf8(cfg.lora)};
    // The Python finds its DLLs beside itself; the host's inherited DLL folder is cleared for the start
    // (platform/process.cpp), as for a development engine.
    auto p = Process::start_duplex(argv, d / "job.log", utf8(cfg.python.parent_path()));
    if (!p) {
        write_text(d / "status.json", json{{"state", "failed"}, {"message", "the AI fill did not start: " + p.error().message},
                                           {"updated", now_seconds()}}
                                          .dump());
        return p.error();
    }
    (*p)->close_input();
    jobs()[utf8(d)] = std::move(*p);
    fs::remove(d / "collect", ec);   // collected
    return {};
}

Result<void> cancel(const FillRequest& r) {
    if (auto ok = check(r); !ok) return ok.error();
    std::lock_guard<std::mutex> lock(store_mutex());
    const fs::path d = folder(r);
    std::error_code ec;
    fs::remove(d / "collect", ec);
    const Status s = read_status(d, r.frames());
    if (s.state == State::Running && !write_text(d / "cancel", "Cancel pressed\n"))
        return make_error(ErrorCode::IoError, "AI fill: could not ask the fill to stop.", utf8(d));
    return {};
}

Result<void> clear(const std::string& shot_id) {
    if (shot_id.empty() || shot_id.find_first_of("/\\.:") != std::string::npos) return {};
    std::lock_guard<std::mutex> lock(store_mutex());
    const fs::path shot = root() / shot_id;
    std::error_code ec;
    if (!fs::is_directory(shot, ec)) return {};
    for (const auto& e : fs::directory_iterator(shot, ec))
        if (e.is_directory() && read_status(e.path(), 1).state == State::Running)
            return make_error(ErrorCode::InvalidArgument, "AI fill is running for this node: Cancel it first.");
    fs::remove_all(shot, ec);
    if (ec) return make_error(ErrorCode::IoError, "AI fill: could not remove the stored fill.", utf8(shot));
    return {};
}

bool other_fill(const FillRequest& r) {
    if (!check(r)) return false;
    std::lock_guard<std::mutex> lock(store_mutex());
    const fs::path shot = root() / r.shot_id;
    std::error_code ec;
    if (!fs::is_directory(shot, ec)) return false;
    const std::string key = r.key();
    for (const auto& e : fs::directory_iterator(shot, ec))
        if (e.is_directory() && e.path().filename().string() != key && read_status(e.path(), 1).state == State::Done) return true;
    return false;
}

std::optional<FillFrame> fill_frame(const FillRequest& r, int frame, const SdrImage& codes, std::string* why) {
    auto no = [&](std::string w) -> std::optional<FillFrame> {
        if (why) *why = std::move(w);
        return std::nullopt;
    };
    if (!check(r)) return no("no shot id");
    const fs::path d = folder(r);
    const int index = frame - r.first;
    if (index < 0 || index >= r.frames()) return no("this frame is outside the clip the fill was made for");
    if (status(r).state != State::Done) return no("the fill is not done");
    if (read_text(hash_path(d, index)) != hex(codes_hash(codes)))
        return no("the picture coming into the node has changed since Generate: press Generate again");
    auto ex = read_exr(d / "fill" / (frame_stem(index) + ".exr"));
    if (!ex) return no("the stored fill of this frame could not be read: " + ex.error().message);
    if (ex->pixels.channels() < 3 || ex->pixels.width() != codes.width() || ex->pixels.height() != codes.height())
        return no("the stored fill is another size");
    FillFrame f{PlanarBuffer(3, codes.height(), codes.width()), PlanarBuffer(1, codes.height(), codes.width())};
    const std::size_t n = std::size_t(codes.width()) * std::size_t(codes.height());
    for (int c = 0; c < 3; ++c) std::copy(ex->pixels.plane(c), ex->pixels.plane(c) + n, f.ap1.plane(c));
    // AI fill writes the frame in its master's primaries: ACEScg here; an ACES 2065-1 file is converted.
    for (const auto& [name, bytes] : ex->attributes)
        if (name == "chromaticities" && bytes.size() >= 4) {
            float rx = 0.0f;
            std::memcpy(&rx, bytes.data(), 4);
            if (std::fabs(rx - 0.7347f) < 1e-3f) f.ap1 = convert_primaries(f.ap1, Primaries::Ap0, Primaries::Ap1);
        }
    // An EXR with alpha holds RGB premultiplied; the master is written without one, so this is the master's RGB.
    for (const auto& [name, plane] : ex->other)
        if (name == "rudra.generated" && plane.channels() == 1 && plane.width() == codes.width() && plane.height() == codes.height())
            std::copy(plane.plane(0), plane.plane(0) + n, f.generated.plane(0));
    return f;
}

std::optional<PlanarBuffer> master_frame(const FillRequest& r, int frame, const SdrImage& codes) {
    if (!check(r)) return std::nullopt;
    const fs::path d = folder(r);
    const int index = frame - r.first;
    if (index < 0 || index >= r.frames() || read_text(hash_path(d, index)) != hex(codes_hash(codes))) return std::nullopt;
    auto ex = read_exr(master_path(d, index));
    if (!ex || ex->pixels.channels() < 3 || ex->pixels.width() != codes.width() || ex->pixels.height() != codes.height())
        return std::nullopt;
    PlanarBuffer p(3, codes.height(), codes.width());
    const std::size_t n = p.plane_size();
    for (int c = 0; c < 3; ++c) std::copy(ex->pixels.plane(c), ex->pixels.plane(c) + n, p.plane(c));
    return p;
}

std::string status_text(const Status& s) {
    switch (s.state) {
        case State::None: return "AI fill: press Generate, then play or render the shot once at full size.";
        case State::Collecting:
            return "AI fill: collecting frames, " + std::to_string(s.done) + " of " + std::to_string(s.of) +
                   " (play or render those frames at full size; the fill starts when all are in).";
        case State::Running:
            return "AI fill: drawing, frame " + std::to_string(s.done) + " of " + std::to_string(s.of) +
                   (s.message.empty() ? std::string() : " (" + s.message + ")") + ". The master is shown meanwhile.";
        case State::Done: return "AI fill: ready. The stored fill is shown.";
        case State::Failed: return "AI fill failed: " + s.message;
        case State::Cancelled: return "AI fill: cancelled. Generate to start again (the frames stored are kept).";
    }
    return {};
}

}  // namespace rudra::ofx_fill
