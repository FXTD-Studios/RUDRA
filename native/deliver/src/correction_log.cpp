#include "rudra/deliver/correction_log.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <system_error>

#include <nlohmann/json.hpp>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/platform/hash.hpp"

#ifndef RUDRA_VERSION_LABEL
#define RUDRA_VERSION_LABEL "dev"
#endif

namespace rudra {
namespace fs = std::filesystem;

namespace {

Error invalid(std::string m, std::string d = {}) { return make_error(ErrorCode::InvalidArgument, std::move(m), std::move(d)); }

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0xFFFFFFFFu) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

void put_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
    o.push_back(std::uint8_t(v >> 24)), o.push_back(std::uint8_t(v >> 16));
    o.push_back(std::uint8_t(v >> 8)), o.push_back(std::uint8_t(v));
}

void chunk(std::vector<std::uint8_t>& o, const char* type, const std::vector<std::uint8_t>& data) {
    put_u32(o, std::uint32_t(data.size()));
    std::vector<std::uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    o.insert(o.end(), td.begin(), td.end());
    put_u32(o, crc32(td.data(), td.size()) ^ 0xFFFFFFFFu);
}

std::string utc_now() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return b;
}

Result<void> write_bytes(const fs::path& path, const std::string& bytes) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return make_error(ErrorCode::IoError, "The correction log could not be written", tmp.string());
        out.write(bytes.data(), std::streamsize(bytes.size()));
        if (!out.flush()) return make_error(ErrorCode::IoError, "The correction log could not be written", tmp.string());
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return make_error(ErrorCode::IoError, "The correction log could not be written", path.string());
    }
    return {};
}

Result<std::string> read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The correction log is missing a file", path.string());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string hash_text(const std::string& s) {
    return sha256_hex(std::as_bytes(std::span(s.data(), s.size())));
}

nlohmann::json consent_json(const CorrectionConsent& c) {
    nlohmann::json j{{"version", kCorrectionLogVersion}, {"opted_in", c.opted_in}, {"by", c.by}, {"at", c.at},
                     {"terms_version", c.terms_version}, {"scope", c.scope}, {"source_id", c.source_id},
                     {"rights", c.rights}, {"note", c.note}};
    if (c.withdrawn()) j["withdrawn"] = {{"by", c.withdrawn_by}, {"at", c.withdrawn_at}};
    return j;
}

// The storage the training loader decodes these targets with (pipeline/hdr_io.py HDRStorage).
const nlohmann::json kStorage = {{"mode", "pq_10000"}, {"diffuse_white_nits", 203.0}, {"floor_nits", 0.005},
                                 {"ceiling_nits", 1000000.0}, {"version", 3}};

Result<void> active(const fs::path& dir, CorrectionConsent* out = nullptr, std::string* consent_sha = nullptr) {
    auto c = read_correction_consent(dir);
    if (!c) return c.error();
    if (!c->opted_in || c->withdrawn())
        return make_error(ErrorCode::Unsupported, "Correction logging is off: there is no active consent", dir.string());
    if (out) *out = *c;
    if (consent_sha) {
        auto text = read_text(dir / "consent.json");
        if (!text) return text.error();
        *consent_sha = hash_text(*text);
    }
    return {};
}

}  // namespace

std::vector<std::uint8_t> png16_rgb_bytes(int width, int height, const std::vector<std::uint16_t>& rgb) {
    std::vector<std::uint8_t> o = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<std::uint8_t> ihdr;
    put_u32(ihdr, std::uint32_t(width));
    put_u32(ihdr, std::uint32_t(height));
    ihdr.insert(ihdr.end(), {16, 2, 0, 0, 0});   // 16-bit, RGB, deflate, no filter method, no interlace
    chunk(o, "IHDR", ihdr);
    // Scanlines: filter 0, then big-endian samples.
    const std::size_t row = std::size_t(width) * 6 + 1;
    std::vector<std::uint8_t> raw(row * std::size_t(height));
    for (int y = 0; y < height; ++y) {
        std::uint8_t* r = raw.data() + row * std::size_t(y);
        r[0] = 0;
        for (int i = 0; i < width * 3; ++i) {
            const std::uint16_t v = rgb[std::size_t(y) * std::size_t(width) * 3 + std::size_t(i)];
            r[1 + 2 * i] = std::uint8_t(v >> 8), r[2 + 2 * i] = std::uint8_t(v);
        }
    }
    std::vector<std::uint8_t> z = {0x78, 0x01};
    std::size_t at = 0;
    do {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - at);
        const bool last = at + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(std::uint8_t(n)), z.push_back(std::uint8_t(n >> 8));
        z.push_back(std::uint8_t(~n)), z.push_back(std::uint8_t((~n) >> 8));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(at), raw.begin() + std::ptrdiff_t(at + n));
        at += n;
    } while (at < raw.size());
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t v : raw) a = (a + v) % 65521u, b = (b + a) % 65521u;
    put_u32(z, (b << 16) | a);
    chunk(o, "IDAT", z);
    chunk(o, "IEND", {});
    return o;
}

Result<void> open_correction_log(const fs::path& dir, const CorrectionConsent& consent) {
    if (!consent.opted_in) return invalid("Correction logging needs an opt-in");
    if (consent.by.empty() || consent.at.empty() || consent.terms_version.empty() || consent.source_id.empty())
        return invalid("The consent must say who agreed, when, to which terms, and who holds the rights");
    if (consent.rights != "owner" && consent.rights != "licensed" && consent.rights != "unknown")
        return invalid("The rights must be owner, licensed or unknown", consent.rights);
    std::error_code ec;
    fs::create_directories(dir / "sdr", ec);
    fs::create_directories(dir / "hdr", ec);
    fs::create_directories(dir / "meta", ec);
    if (ec) return make_error(ErrorCode::IoError, "The correction log folder could not be made", dir.string());
    const fs::path sentinel = dir / "_ingest_config.json";
    if (fs::exists(sentinel, ec)) {
        auto text = read_text(sentinel);
        if (!text) return text.error();
        try {
            const auto j = nlohmann::json::parse(*text);
            if (j.value("storage", nlohmann::json()) != kStorage)
                return invalid("This folder holds pairs stored another way", sentinel.string());
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ParseError, "The folder's ingest config could not be read", e.what());
        }
    } else {
        const nlohmann::json j{{"storage", kStorage}, {"writer", "rudra correction log"},
                               {"sdr", "16-bit PNG code values, Rec.709"},
                               {"hdr", "16-bit PNG, ST 2084 (PQ, 10 000 nits), Rec.2020"}};
        if (auto w = write_bytes(sentinel, j.dump(2)); !w) return w.error();
    }
    CorrectionConsent c = consent;
    c.withdrawn_by.clear(), c.withdrawn_at.clear();
    return write_bytes(dir / "consent.json", consent_json(c).dump(2));
}

Result<CorrectionConsent> read_correction_consent(const fs::path& dir) {
    auto text = read_text(dir / "consent.json");
    if (!text) return make_error(ErrorCode::NotFound, "This folder is not a correction log", dir.string());
    try {
        const auto j = nlohmann::json::parse(*text);
        if (j.value("version", 0) > kCorrectionLogVersion)
            return make_error(ErrorCode::ContractMismatch, "This correction log is from a newer RUDRA");
        CorrectionConsent c;
        c.opted_in = j.value("opted_in", false);
        c.by = j.value("by", std::string()), c.at = j.value("at", std::string());
        c.terms_version = j.value("terms_version", std::string());
        c.scope = j.value("scope", std::string()), c.source_id = j.value("source_id", std::string());
        c.rights = j.value("rights", std::string()), c.note = j.value("note", std::string());
        if (j.contains("withdrawn")) {
            c.withdrawn_by = j["withdrawn"].value("by", std::string());
            c.withdrawn_at = j["withdrawn"].value("at", std::string());
        }
        return c;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The consent could not be read", e.what());
    }
}

Result<void> withdraw_correction_consent(const fs::path& dir, const std::string& by, const std::string& at) {
    auto c = read_correction_consent(dir);
    if (!c) return c.error();
    if (by.empty() || at.empty()) return invalid("A withdrawal must say who and when");
    c->withdrawn_by = by, c->withdrawn_at = at;
    return write_bytes(dir / "consent.json", consent_json(*c).dump(2));
}

Result<CorrectionEntry> log_correction(const fs::path& dir, const SdrImage& sdr, const NitsFrame& accepted,
                                       const MasterRequest& grade, const CorrectionSource& source) {
    std::string consent_sha;
    if (auto a = active(dir, nullptr, &consent_sha); !a) return a.error();
    if (grade.source_space != "rec709")
        return make_error(ErrorCode::Unsupported, "Only Rec.709 frames are logged for training", grade.source_space);
    const int h = sdr.height(), w = sdr.width();
    if (h <= 0 || w <= 0 || accepted.height() != h || accepted.width() != w)
        return invalid("The accepted master and the frame differ in size");
    std::error_code ec;
    if (!fs::is_regular_file(source.file, ec)) return make_error(ErrorCode::NotFound, "The frame's source file is missing", source.file.string());
    auto source_sha = sha256_file(source.file);
    if (!source_sha) return source_sha.error();
    const std::string grade_json = master_request_json(grade);
    const std::string id =
        hash_text(*source_sha + ":" + std::to_string(source.frame) + ":" + grade_json).substr(0, 20);
    CorrectionEntry e;
    e.id = id;
    e.sdr = dir / "sdr" / (id + ".png");
    e.hdr = dir / "hdr" / (id + ".png");
    e.meta = dir / "meta" / (id + ".json");
    e.scene_id = source_sha->substr(0, 16);
    // 1 scene in 10 to val, by the source's hash: frames of one file stay together.
    e.split = std::stoul(source_sha->substr(0, 2), nullptr, 16) % 10 == 0 ? "val" : "train";
    if (fs::exists(e.meta, ec)) {
        e.already_logged = true;
        return e;
    }
    // The SDR, as decoded.
    std::vector<std::uint16_t> code(std::size_t(h) * std::size_t(w) * 3);
    for (int c = 0; c < 3; ++c) {
        const float* p = sdr.buffer().plane(c);
        for (std::size_t i = 0; i < std::size_t(h) * std::size_t(w); ++i)
            code[i * 3 + std::size_t(c)] = std::uint16_t(std::lround(std::clamp(double(p[i]), 0.0, 1.0) * 65535.0));
    }
    // The accepted HDR: Rec.709 nits to Rec.2020, then PQ.
    PlanarBuffer nits709(3, h, w);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < std::size_t(h) * std::size_t(w); ++i) nits709.plane(c)[i] = float(accepted.plane(c)[i]);
    const PlanarBuffer nits2020 = convert_primaries(nits709, Primaries::Rec709, Primaries::Rec2020);
    std::vector<std::uint16_t> pq(code.size());
    double peak = 0.0;
    std::vector<std::uint8_t> clipped(std::size_t(h) * std::size_t(w), 0);   // a channel above PQ's 10 000 nits
    for (int c = 0; c < 3; ++c) {
        const float* p = nits2020.plane(c);
        for (std::size_t i = 0; i < std::size_t(h) * std::size_t(w); ++i) {
            const float v = std::isfinite(p[i]) ? std::max(p[i], 0.0f) : 0.0f;
            peak = std::max(peak, double(v));
            if (v > 10000.0f) clipped[i] = 1;
            pq[i * 3 + std::size_t(c)] = std::uint16_t(std::lround(double(pq_oetf(v)) * 65535.0));
        }
    }
    const double clipped_fraction =
        double(std::count(clipped.begin(), clipped.end(), std::uint8_t(1))) / double(clipped.size());
    const auto sdr_bytes = png16_rgb_bytes(w, h, code), hdr_bytes = png16_rgb_bytes(w, h, pq);
    const std::string sdr_s(sdr_bytes.begin(), sdr_bytes.end()), hdr_s(hdr_bytes.begin(), hdr_bytes.end());
    if (auto r = write_bytes(e.sdr, sdr_s); !r) return r.error();
    if (auto r = write_bytes(e.hdr, hdr_s); !r) return r.error();
    const nlohmann::json meta{
        {"version", kCorrectionLogVersion},
        {"id", id},
        {"logged_at", utc_now()},
        {"app_version", RUDRA_VERSION_LABEL},
        {"source", {{"file", fs::absolute(source.file).string()}, {"sha256", *source_sha}, {"frame", source.frame},
                    {"label", source.label}}},
        {"consent_sha256", consent_sha},
        {"grade", nlohmann::json::parse(grade_json)},
        {"sdr", {{"file", e.sdr.filename().string()}, {"sha256", hash_text(sdr_s)}, {"width", w}, {"height", h},
                 {"primaries", "rec709"}, {"encoding", "code values, 16-bit"}}},
        {"hdr", {{"file", e.hdr.filename().string()}, {"sha256", hash_text(hdr_s)}, {"primaries", "rec2020"},
                 {"encoding", "pq_10000"}, {"peak_nits", peak}, {"clipped_fraction", clipped_fraction}}},
        {"sdr_kind", "real"},
        {"target_kind", "artist_accepted_grade"},
        {"scene_id", e.scene_id},
        {"split", e.split},
    };
    if (auto r = write_bytes(e.meta, meta.dump(2)); !r) return r.error();
    return e;
}

Result<int> export_correction_manifest(const fs::path& dir, const fs::path& manifest) {
    CorrectionConsent consent;
    if (auto a = active(dir, &consent); !a) return a.error();
    std::error_code ec;
    if (fs::exists(manifest, ec)) return invalid("Refusing to overwrite the manifest", manifest.string());
    std::vector<fs::path> metas;
    for (const auto& f : fs::directory_iterator(dir / "meta", ec))
        if (f.path().extension() == ".json") metas.push_back(f.path());
    std::sort(metas.begin(), metas.end());
    std::string out;
    const fs::path root = fs::absolute(dir).lexically_normal();
    const bool commercial = consent.rights == "owner" || consent.rights == "licensed";
    for (const auto& m : metas) {
        auto text = read_text(m);
        if (!text) return text.error();
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(*text);
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ParseError, "A correction record could not be read", m.string() + ": " + e.what());
        }
        const nlohmann::json row{
            {"source_id", consent.source_id},
            {"scene_id", "correction-" + j.value("scene_id", std::string())},
            {"asset_id", "correction-" + j.value("id", std::string())},
            {"split", j.value("split", std::string("train"))},
            {"sdr_path", (root / "sdr" / j["sdr"]["file"].get<std::string>()).string()},
            {"sdr_sha256", j["sdr"]["sha256"]},
            {"hdr_path", (root / "hdr" / j["hdr"]["file"].get<std::string>()).string()},
            {"hdr_sha256", j["hdr"]["sha256"]},
            {"metadata_path", (root / "meta" / m.filename()).string()},
            {"sdr_kind", "real"},
            {"target_kind", "artist_accepted_grade"},
            {"commercial_ok", commercial},
            {"source", "correction_log"},
        };
        out += row.dump() + "\n";
    }
    if (auto w = write_bytes(manifest, out); !w) return w.error();
    return int(metas.size());
}

}  // namespace rudra
