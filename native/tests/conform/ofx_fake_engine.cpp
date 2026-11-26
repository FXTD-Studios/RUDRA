// A stand-in for `rudra-native aces-v2-serve` (25 Nov 2026), for the OFX plug-in's learned route
// tests: no model, the same line protocol. It reads the 16-bit TIFF the plug-in writes
// (ofx/engine_client.cpp) and answers a preview with the planes 2 * code^2.4, ACEScg.
//   RUDRA_OFX_FAKE_ENGINE=error  answers {"event":"error"} at start and exits 2.
// A request with a "reconstruction_region" (the luminance limit, 26 Nov 2026) gets half the planes, so the
// tests see that the limit reached the engine.
// Started as `<it> .../ofx_job.py --dir <folder> ...` (26 Nov 2026), it stands in for AI fill's job
// (rudra/genfill/ofx_job.py): each stored master becomes the "fill" of its frame, and status.json says done
// (RUDRA_OFX_FAKE_JOB=fail: failed).
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace {

std::uint32_t u32(const std::vector<std::uint8_t>& b, std::size_t at) {
    return std::uint32_t(b[at]) | std::uint32_t(b[at + 1]) << 8 | std::uint32_t(b[at + 2]) << 16 | std::uint32_t(b[at + 3]) << 24;
}
std::uint16_t u16(const std::vector<std::uint8_t>& b, std::size_t at) { return std::uint16_t(b[at] | b[at + 1] << 8); }

void say(const json& j) { std::cout << j.dump() << "\n" << std::flush; }

int fake_job(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path dir;
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--dir") dir = argv[i + 1];
    std::ifstream rq(dir / "request.json");
    const json req = json::parse(rq, nullptr, false);
    const int n = req.is_object() ? req.value("frames", 0) : 0;
    const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    json st = {{"state", "done"}, {"done", n}, {"of", n}, {"message", "done (stand-in)"}, {"updated", now}};
    if (const char* f = std::getenv("RUDRA_OFX_FAKE_JOB"); f && std::string(f) == "fail") {
        st = {{"state", "failed"}, {"message", "the stand-in failed"}, {"updated", now}};
    } else {
        std::error_code ec;
        fs::create_directories(dir / "fill", ec);
        for (int i = 0; i < n; ++i) {
            char name[32];
            std::snprintf(name, sizeof name, "f.%06d.exr", i);
            fs::copy_file(dir / "master" / name, dir / "fill" / name, fs::copy_options::overwrite_existing, ec);
            if (ec) st = {{"state", "failed"}, {"message", std::string("missing ") + name}, {"updated", now}};
        }
    }
    std::ofstream(dir / "status.json", std::ios::trunc) << st.dump();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2) {
        const std::string first = argv[1];
        if (first.size() >= 10 && first.compare(first.size() - 10, 10, "ofx_job.py") == 0) return fake_job(argc, argv);
    }
    if (argc < 3 || std::string(argv[1]) != "aces-v2-serve") return 2;
    if (const char* f = std::getenv("RUDRA_OFX_FAKE_ENGINE"); f && std::string(f) == "error") {
        say({{"event", "error"}, {"message", "no model (fake)"}});
        return 2;
    }
    say({{"event", "ready"}, {"package", argv[2]}});
    std::string line;
    while (std::getline(std::cin, line)) {
        json r = json::parse(line, nullptr, false);
        if (r.is_discarded()) continue;
        const std::string op = r.value("op", std::string());
        if (op == "quit") break;
        if (op != "preview") {
            say({{"id", r.value("id", 0)}, {"ok", false}, {"error", "unknown op"}});
            continue;
        }
        std::ifstream f(r.at("source").get<std::string>(), std::ios::binary);
        std::vector<std::uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (b.size() < 8 || b[0] != 'I' || b[1] != 'I') {
            say({{"id", r.value("id", 0)}, {"ok", false}, {"error", "not a little-endian TIFF"}});
            continue;
        }
        const std::size_t ifd = u32(b, 4);
        std::uint32_t w = 0, h = 0, data = 0;
        for (int i = 0; i < u16(b, ifd); ++i) {
            const std::size_t e = ifd + 2 + std::size_t(i) * 12;
            const std::uint16_t tag = u16(b, e), type = u16(b, e + 2);
            const std::uint32_t v = type == 3 ? u16(b, e + 8) : u32(b, e + 8);
            if (tag == 256) w = v;
            if (tag == 257) h = v;
            if (tag == 273) data = v;
        }
        const std::size_t n = std::size_t(w) * h;
        std::vector<float> planes(n * 3);
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                const float code = float(u16(b, data + (i * 3 + std::size_t(c)) * 2)) / 65535.0f;
                planes[std::size_t(c) * n + i] = 2.0f * std::pow(code, 2.4f) * float(r.value("trims", json::object()).value("strength", 1.0)) *
                                                 (r.contains("reconstruction_region") ? 0.5f : 1.0f);
            }
        std::ofstream o(r.at("model_out").get<std::string>(), std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char*>(planes.data()), std::streamsize(planes.size() * sizeof(float)));
        o.close();
        say({{"id", r.value("id", 0)}, {"ok", true}, {"width", w}, {"height", h}});
    }
    return 0;
}
