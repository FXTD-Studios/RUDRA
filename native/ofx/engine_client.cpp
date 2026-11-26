#include "engine_client.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>

#include <nlohmann/json.hpp>

#include "rudra/platform/process.hpp"

namespace rudra::ofx_engine {
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

fs::path env_path(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? fs::path(v) : fs::path();
}

fs::path utf8_path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

std::string path_list(const std::vector<fs::path>& dirs) {
#ifdef _WIN32
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif
    std::string s;
    for (const auto& d : dirs) {
        if (!s.empty()) s.push_back(sep);
        s += d.string();
    }
    return s;
}

// What an engine's exit code means (Windows NTSTATUS codes for a program that never got going).
std::string exit_text(int code) {
    char hex[16];
    std::snprintf(hex, sizeof hex, "0x%08X", unsigned(code));
    switch (unsigned(code)) {
        case 0xC0000135u: return std::string("exit ") + hex + ": a DLL it needs was not found";
        case 0xC0000139u: return std::string("exit ") + hex + ": a DLL of another version was loaded (entry point not found)";
        case 0xC000007Bu: return std::string("exit ") + hex + ": a 32-bit or broken DLL was loaded";
        case 0xC0000005u: return std::string("exit ") + hex + ": it crashed";
        default: return "exit " + std::to_string(code) + (code < 0 || code > 255 ? std::string(" (") + hex + ")" : std::string());
    }
}

// The last line the engine wrote to its log, if any.
std::string log_tail(const fs::path& log) {
    std::ifstream f(log, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (s.empty()) return ": the log is empty";
    const auto nl = s.find_last_of('\n');
    std::string last = nl == std::string::npos ? s : s.substr(nl + 1);
    if (!last.empty() && last.back() == '\r') last.pop_back();
    if (last.size() > 300) last = last.substr(0, 300) + "...";
    return ": " + last;
}

void put16(std::vector<std::uint8_t>& b, std::uint16_t v) { b.push_back(std::uint8_t(v & 0xff)), b.push_back(std::uint8_t(v >> 8)); }
void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t((v >> (8 * i)) & 0xff));
}

// The engine, one per host process.
class Engine {
public:
    Result<PlanarBuffer> master(const SdrImage& codes, double strength, const std::string& region_json) {
        std::lock_guard<std::mutex> lock(mu_);
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!proc_ || !proc_->running()) {
                auto started = start();
                if (!started) return started.error();
            }
            auto r = request(codes, strength, region_json);
            if (r || (proc_ && proc_->running())) return r;   // a refused frame is an answer; a dead engine is retried once
            proc_.reset();
            if (attempt == 1) return r;
        }
        return make_error(ErrorCode::IoError, "RUDRA's learned engine stopped.");
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mu_);
        if (proc_ && proc_->running()) {
            static const char quit[] = "{\"op\":\"quit\"}\n";
            proc_->write({reinterpret_cast<const std::uint8_t*>(quit), sizeof quit - 1});
            proc_->close_input();
        }
        proc_.reset();
    }

private:
    Result<void> start() {
        auto cfg = load_engine_config();
        if (!cfg) return cfg.error();
        const fs::path log = fs::temp_directory_path() / "rudra_ofx_engine.log";
        auto p = Process::start_duplex({cfg->cli.string(), "aces-v2-serve", cfg->package.string(), "--runtime", cfg->runtime,
                                        "--device", cfg->device},
                                       log, path_list(cfg->path_prepend));
        if (!p) return make_error(ErrorCode::NotFound, "RUDRA's learned engine did not start.", cfg->cli.string());
        proc_ = std::move(*p);
        buffer_.clear();
        // The model loads, then the engine says ready (or why not).
        for (;;) {
            auto line = read_line();
            if (!line) {
                const int code = proc_->wait();
                proc_.reset();
                return make_error(ErrorCode::IoError, "RUDRA's learned engine stopped while loading (" + exit_text(code) + ")" +
                                                          log_tail(log) + "; see " + log.string());
            }
            json j = json::parse(*line, nullptr, false);
            if (j.is_discarded()) continue;
            const std::string ev = j.value("event", std::string());
            if (ev == "ready") return {};
            if (ev == "error") {
                proc_.reset();
                return make_error(ErrorCode::IoError, "RUDRA's learned engine could not load the model: " + j.value("message", j.dump()));
            }
        }
    }

    Result<PlanarBuffer> request(const SdrImage& codes, double strength, const std::string& region_json) {
        const int w = codes.width(), h = codes.height();
        const std::uint64_t id = ++next_id_;
        const fs::path dir = fs::temp_directory_path();
        const std::string stem = "rudra_ofx_" + std::to_string(std::uintptr_t(this)) + "_" + std::to_string(id);
        const fs::path still = dir / (stem + ".tif"), planes = dir / (stem + ".f32");
        struct Cleanup {
            fs::path a, b;
            ~Cleanup() {
                std::error_code ec;
                fs::remove(a, ec), fs::remove(b, ec);
            }
        } cleanup{still, planes};
        if (auto ok = write_tiff16(still, codes); !ok) return ok.error();
        json req = {{"id", id}, {"op", "preview"}, {"source", still.string()}, {"frame", 0}, {"keep", false},
                    {"guard", true}, {"model_out", planes.string()}, {"trims", {{"strength", strength}}}};
        if (!region_json.empty()) {   // 26 Nov 2026: the luminance limit (aces_v2_serve's reconstruction_region)
            json rr = json::parse(region_json, nullptr, false);
            if (rr.is_discarded() || !rr.is_object()) return make_error(ErrorCode::InvalidArgument, "The luminance limit is not valid JSON.");
            req["reconstruction_region"] = rr;
        }
        const std::string line = req.dump() + "\n";
        if (!proc_->write({reinterpret_cast<const std::uint8_t*>(line.data()), line.size()}))
            return make_error(ErrorCode::IoError, "RUDRA's learned engine stopped.");
        for (;;) {
            auto ans = read_line();
            if (!ans) return make_error(ErrorCode::IoError, "RUDRA's learned engine stopped.");
            json j = json::parse(*ans, nullptr, false);
            if (j.is_discarded() || !j.contains("id") || j["id"] != id) continue;
            if (!j.value("ok", false)) return make_error(ErrorCode::IoError, "RUDRA's learned engine: " + j.value("error", j.dump()));
            break;
        }
        const std::size_t n = std::size_t(w) * std::size_t(h);
        std::error_code ec;
        if (fs::file_size(planes, ec) != n * 3 * sizeof(float))
            return make_error(ErrorCode::IoError, "RUDRA's learned engine wrote a master of another size.");
        PlanarBuffer out(3, h, w);
        std::ifstream f(planes, std::ios::binary);
        for (int c = 0; c < 3; ++c) f.read(reinterpret_cast<char*>(out.plane(c)), std::streamsize(n * sizeof(float)));
        if (!f) return make_error(ErrorCode::IoError, "Could not read the learned master.", planes.string());
        return out;
    }

    // One line of the engine's stdout, without its newline; nullopt at the end of the stream.
    std::optional<std::string> read_line() {
        for (;;) {
            const auto nl = buffer_.find('\n');
            if (nl != std::string::npos) {
                std::string line = buffer_.substr(0, nl);
                buffer_.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                return line;
            }
            std::array<std::uint8_t, 4096> chunk{};
            const std::size_t got = proc_ ? proc_->read_some(chunk) : 0;
            if (got == 0) return std::nullopt;
            buffer_.append(reinterpret_cast<const char*>(chunk.data()), got);
        }
    }

    std::mutex mu_;
    std::unique_ptr<Process> proc_;
    std::string buffer_;
    std::atomic<std::uint64_t> next_id_{0};
};

Engine& engine() {
    static Engine* e = new Engine();   // never destroyed: the host may unload in any order
    return *e;
}

}  // namespace

// The settings file as JSON (an object), or why not.
Result<json> read_config_json(fs::path* at_out = nullptr) {
    const fs::path at = engine_config_path();
    if (at_out) *at_out = at;
    std::error_code ec;
    if (at.empty() || !fs::is_regular_file(at, ec))
        return make_error(ErrorCode::NotFound, "no " + (at.empty() ? std::string("ofx_engine.json") : at.string()) +
                                                   " (install RUDRA, or run tmp\\BUILD_OFX.ps1 -Install).");
    std::ifstream f(at);
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return make_error(ErrorCode::InvalidArgument, "Could not read the engine settings.", at.string());
    return j;
}

fs::path engine_config_path() {
    if (fs::path p = env_path("RUDRA_OFX_ENGINE_CONFIG"); !p.empty()) return p;
#ifdef _WIN32
    const fs::path base = env_path("LOCALAPPDATA");
#elif defined(__APPLE__)
    const fs::path base = env_path("HOME").empty() ? fs::path() : env_path("HOME") / "Library" / "Application Support";
#else
    const fs::path base = !env_path("XDG_DATA_HOME").empty() ? env_path("XDG_DATA_HOME")
                          : env_path("HOME").empty()         ? fs::path()
                                                             : env_path("HOME") / ".local" / "share";
#endif
    return base.empty() ? fs::path() : base / "FXTD Studios" / "RUDRA" / "ofx_engine.json";
}

Result<EngineConfig> load_engine_config() {
    fs::path at;
    auto jr = read_config_json(&at);
    if (!jr) {
        // The tests' wording since phase 2: the learned route says what it needs.
        if (jr.error().code == ErrorCode::NotFound)
            return make_error(ErrorCode::NotFound, "The learned route needs RUDRA's engine: " + jr.error().message);
        return jr.error();
    }
    const json& j = *jr;
    std::error_code ec;
    EngineConfig c;
    c.from = at;
    c.cli = utf8_path(j.value("cli", std::string()));
    c.package = utf8_path(j.value("package", std::string()));
    c.runtime = j.value("runtime", c.runtime);
    c.device = j.value("device", c.device);
    if (j.contains("path_prepend") && j["path_prepend"].is_array())
        for (const auto& d : j["path_prepend"])
            if (d.is_string()) c.path_prepend.push_back(utf8_path(d.get<std::string>()));
    if (j.contains("genfill") && j["genfill"].is_object()) {
        const json& g = j["genfill"];
        c.genfill = {utf8_path(g.value("python", std::string())), utf8_path(g.value("repo", std::string())),
                     utf8_path(g.value("base", std::string())), utf8_path(g.value("lora", std::string()))};
    }
    if (c.cli.empty() || !fs::is_regular_file(c.cli, ec))
        return make_error(ErrorCode::NotFound, "RUDRA's learned engine is not where the settings say.", c.cli.string());
    if (c.package.empty() || !fs::is_regular_file(c.package / "manifest.json", ec))
        return make_error(ErrorCode::NotFound, "The learned ACES 2.0 model is not where the settings say (File > Model packages in RUDRA).",
                          c.package.string());
    return c;
}

Result<void> write_tiff16(const fs::path& path, const SdrImage& codes) {
    const int w = codes.width(), h = codes.height();
    if (w <= 0 || h <= 0) return make_error(ErrorCode::InvalidArgument, "No picture.");
    const std::uint32_t data_bytes = std::uint32_t(w) * std::uint32_t(h) * 6;
    // Header, then the pixels, then the directory (and the 3 bits-per-sample values).
    std::vector<std::uint8_t> b;
    b.reserve(std::size_t(data_bytes) + 256);
    b.push_back('I'), b.push_back('I');
    put16(b, 42);
    const std::uint32_t data_at = 8, bps_at = data_at + data_bytes, ifd_at = bps_at + 6;
    put32(b, ifd_at);
    const PlanarBuffer& s = codes.buffer();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = s.at(c, y, x);
                v = v != v ? 0.0f : v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
                put16(b, std::uint16_t(v * 65535.0f + 0.5f));
            }
    put16(b, 16), put16(b, 16), put16(b, 16);
    struct Tag {
        std::uint16_t id, type;
        std::uint32_t count, value;
    };
    const Tag tags[] = {{256, 4, 1, std::uint32_t(w)}, {257, 4, 1, std::uint32_t(h)}, {258, 3, 3, bps_at}, {259, 3, 1, 1},
                        {262, 3, 1, 2},  {273, 4, 1, data_at}, {277, 3, 1, 3},  {278, 4, 1, std::uint32_t(h)},
                        {279, 4, 1, data_bytes}, {284, 3, 1, 1}};
    put16(b, std::uint16_t(std::size(tags)));
    for (const auto& t : tags) {
        put16(b, t.id), put16(b, t.type), put32(b, t.count);
        if (t.type == 3 && t.count == 1) put16(b, std::uint16_t(t.value)), put16(b, 0);
        else put32(b, t.value);
    }
    put32(b, 0);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    if (!f) return make_error(ErrorCode::IoError, "Could not write the frame for the engine.", path.string());
    return {};
}

Result<GenFillConfig> load_genfill_config() {
    auto jr = read_config_json();
    if (!jr) {
        if (jr.error().code == ErrorCode::NotFound)
            return make_error(ErrorCode::NotFound, "AI fill needs RUDRA installed on this computer: " + jr.error().message);
        return jr.error();
    }
    const json& j = *jr;
    if (!j.contains("genfill") || !j["genfill"].is_object())
        return make_error(ErrorCode::NotFound, "AI fill is not set up for the plug-in: run tmp\\BUILD_OFX.ps1 -Install, or set up AI fill in the RUDRA app (Deliver > AI fill > Setup).");
    const json& g = j["genfill"];
    GenFillConfig c{utf8_path(g.value("python", std::string())), utf8_path(g.value("repo", std::string())),
                    utf8_path(g.value("base", std::string())), utf8_path(g.value("lora", std::string()))};
    std::error_code ec;
    if (c.python.empty() || !fs::is_regular_file(c.python, ec))
        return make_error(ErrorCode::NotFound, "AI fill: no Python with PyTorch where the settings say.", c.python.string());
    if (c.repo.empty() || !fs::is_regular_file(c.repo / "rudra" / "genfill" / "ofx_job.py", ec))
        return make_error(ErrorCode::NotFound, "AI fill: RUDRA's Python source (rudra/genfill/ofx_job.py) is not where the settings say.", c.repo.string());
    if (c.base.empty() || !fs::is_regular_file(c.base / "transformer" / "config.json", ec))
        return make_error(ErrorCode::NotFound, "AI fill: the Wan2.1-VACE-1.3B base model is not where the settings say.", c.base.string());
    if (c.lora.empty() || !fs::is_regular_file(c.lora, ec))
        return make_error(ErrorCode::NotFound, "AI fill v5 is not installed (File > Model packages in RUDRA).", c.lora.string());
    return c;
}

Result<PlanarBuffer> learned_master(const SdrImage& codes, double strength, const std::string& region_json) {
    return engine().master(codes, strength, region_json);
}

void stop_engine() { engine().stop(); }

}  // namespace rudra::ofx_engine
