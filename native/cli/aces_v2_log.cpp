#include "aces_v2_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

#include "rudra/core/gamut.hpp"
#include "rudra/core/timecode.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/media/png16.hpp"
#include "rudra/platform/process.hpp"

namespace fs = std::filesystem;

namespace rudra {

const AcesDisplayTarget* aces_display_target(const std::string& id) {
    static const AcesDisplayTarget targets[] = {
        {"pq1000", aces_v2::kOcioConfig, aces_v2::kOcioHdrDisplay, aces_v2::kOcioHdrView, "pq", 1000,
         "PQ (SMPTE ST 2084), ACES 2.0 HDR 1000 nits (P3 D65)"},
        {"pq2000", aces_v2::kOcioStudioConfig, "Rec.2100-PQ - Display", "ACES 2.0 - HDR 2000 nits (P3 D65)", "pq", 2000,
         "PQ (SMPTE ST 2084), ACES 2.0 HDR 2000 nits (P3 D65)"},
        {"pq4000", aces_v2::kOcioStudioConfig, "Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (P3 D65)", "pq", 4000,
         "PQ (SMPTE ST 2084), ACES 2.0 HDR 4000 nits (P3 D65)"},
        {"hlg1000", aces_v2::kOcioStudioConfig, "Rec.2100-HLG - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)", "hlg", 1000,
         "HLG (ARIB STD-B67), ACES 2.0 HDR 1000 nits (P3 D65)"},
        {"sdr709", aces_v2::kOcioConfig, aces_v2::kOcioDisplay, aces_v2::kOcioView, "sdr", 100,
         "Rec.709 on BT.1886, ACES 2.0 SDR 100 nits (the SDR trim pass)"},
    };
    for (const auto& t : targets)
        if (id == t.id) return &t;
    return nullptr;
}

std::string AcesLogOptions::name() const {
    if (pq) return target;
    return curve ? log_curve_name(*curve) : "";
}

Result<AcesLogOptions> parse_aces_log(const std::string& curve, const std::string& formats) {
    AcesLogOptions o;
    if (curve.empty() || curve == "none") return o;
    if (aces_display_target(curve)) {
        o.pq = true;
        o.target = curve;
    } else {
        o.curve = log_curve_from_name(curve);
        if (!o.curve)
            return make_error(ErrorCode::InvalidArgument,
                              "Unknown encoding (logc4, acescct, pq1000, pq2000, pq4000, hlg1000 or sdr709).", curve);
    }
    std::stringstream ss(formats.empty() ? std::string("tiff16") : formats);
    std::string f;
    auto one_movie = [&] {
        return o.movie() ? make_error(ErrorCode::InvalidArgument, "Choose one movie format.", formats) : Result<void>{};
    };
    while (std::getline(ss, f, ',')) {
        if (f == "tiff16") o.tiff16 = true;
        else if (f == "dpx10") o.dpx10 = true;
        else if (f == "prores4444xq" || f == "prores4444" || f == "prores422hq") {
            if (auto r = one_movie(); !r) return r.error();
            o.prores = f.substr(6);
        } else if (f == "mov16") {
            if (auto r = one_movie(); !r) return r.error();
            o.mov16 = true;
        } else if (f == "hevc10") {
            if (auto r = one_movie(); !r) return r.error();
            if (!o.hdr10()) return make_error(ErrorCode::InvalidArgument, "HDR10 HEVC is a PQ deliverable: choose a PQ encoding.", formats);
            o.hevc10 = true;
        } else if (!f.empty())
            return make_error(ErrorCode::InvalidArgument,
                              "Unknown format (tiff16, dpx10, prores4444xq, prores4444, prores422hq, mov16, hevc10).", f);
    }
    return o;
}

PlanarBuffer aces_log_codes(const PlanarBuffer& in, LogCurve curve) {
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, log_curve_primaries(curve));
    PlanarBuffer out(3, in.height(), in.width());
    const std::size_t n = in.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        const double v[3] = {in.plane(0)[i], in.plane(1)[i], in.plane(2)[i]};
        for (int c = 0; c < 3; ++c) {
            const double lin = m[std::size_t(c)][0] * v[0] + m[std::size_t(c)][1] * v[1] + m[std::size_t(c)][2] * v[2];
            const double code = curve == LogCurve::AcesCct ? acescct_encode(lin) : logc4_encode(lin);
            out.plane(c)[i] = static_cast<float>(std::clamp(code, 0.0, 1.0));
        }
    }
    return out;
}

namespace {
// Interleaved 16-bit, in `order` (0,1,2 = RGB; 2,1,0 = BGR).
std::vector<std::uint16_t> to_u16(const PlanarBuffer& codes, const int (&order)[3]) {
    const std::size_t n = codes.plane_size();
    std::vector<std::uint16_t> out(n * 3);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            out[i * 3 + std::size_t(c)] = static_cast<std::uint16_t>(std::lround(codes.plane(order[c])[i] * 65535.0f));
    return out;
}
std::string fps_text(double fps) {
    // 23.976 and friends as the exact fraction ffmpeg expects.
    for (double base : {24.0, 30.0, 60.0, 48.0, 120.0})
        if (std::abs(fps - base * 1000.0 / 1001.0) < 1e-3) return std::to_string(int(base * 1000)) + "/1001";
    std::ostringstream s;
    s.precision(10);
    s << fps;
    return s.str();
}
}  // namespace

Result<std::unique_ptr<AcesLogWriter>> AcesLogWriter::open(const AcesLogOptions& o, const fs::path& out_dir,
                                                           const fs::path& movie, double fps,
                                                           const aces_v2::AcesView* display) {
    if (!o.encoded()) return make_error(ErrorCode::InvalidArgument, "No encoding for the copies.");
    if (o.pq && !display)
        return make_error(ErrorCode::Unsupported, "The ACES 2.0 output needs the OpenColorIO view, which is not loaded.");
    std::unique_ptr<AcesLogWriter> w(new AcesLogWriter());
    w->o_ = o;
    w->display_ = display;
    w->movie_ = movie;
    w->fps_ = fps > 0 ? fps : 24.0;
    const std::string dir = o.pq ? o.name() : "log_" + o.name();
    w->dir_ = o.flat ? out_dir : out_dir / dir;
    w->dpx_dir_ = o.flat ? out_dir : out_dir / (dir + "_dpx");
    std::error_code ec;
    if (o.tiff16) fs::create_directories(w->dir_, ec);
    if (!ec && o.dpx10) fs::create_directories(w->dpx_dir_, ec);
    if (ec) return make_error(ErrorCode::IoError, "The copies' folder could not be made.", out_dir.string());
    if (o.dpx10 || o.movie()) {
        auto ff = require_executable("ffmpeg");
        if (!ff) return ff.error();
    }
    return w;
}

AcesLogWriter::~AcesLogWriter() {
    if (ffmpeg_) {
        ffmpeg_->close_input();
        ffmpeg_->wait();
    }
}

PlanarBuffer AcesLogWriter::codes(const PlanarBuffer& acescg) const {
    if (!o_.pq) return aces_log_codes(acescg, *o_.curve);
    PlanarBuffer out = aces_v2::view_forward(*display_, acescg);
    for (float& v : out.span()) v = std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
    return out;
}

std::vector<std::string> AcesLogWriter::movie_arguments(const AcesLogOptions& o) {
    // PQ is BT.2020 / SMPTE 2084; a camera log has no codes for its gamut or curve.
    // The tags go on the frames too (setparams), and for ProRes into its frame
    // headers (prores_metadata), which Resolve and FCP read: some ffmpeg builds
    // (the 2025 gyan builds on Windows) leave them unknown otherwise.
    // The display targets' tags: PQ and HLG are BT.2020; the SDR trim pass BT.709 / BT.1886 (bt709).
    const std::string tf = o.pq ? o.display_target().transfer : "";
    const bool sdr = tf == "sdr";
    const std::string prim = !o.pq ? "unknown" : sdr ? "bt709" : "bt2020";
    const std::string trc = !o.pq ? "unknown" : tf == "hlg" ? "arib-std-b67" : sdr ? "bt709" : "smpte2084";
    if (o.mov16) {
        std::vector<std::string> a;
        if (o.pq) a = {"-vf", "setparams=range=full:color_primaries=" + prim + ":color_trc=" + trc + ":colorspace=gbr"};
        for (const char* x : {"-c:v", "rawvideo", "-pix_fmt", "rgb48be", "-colorspace", "rgb", "-color_range", "pc"}) a.push_back(x);
        for (const std::string& x : {std::string("-color_primaries"), prim, std::string("-color_trc"), trc,
                                     std::string("-movflags"), std::string("+write_colr")})
            a.push_back(x);
        return a;
    }
    const std::string matrix = o.pq && !sdr ? "bt2020nc" : "bt709";
    // HEVC: a ProRes 4444 XQ intermediate first (the light level is known only at the end).
    const std::string prores = o.hevc10 ? std::string("4444xq") : o.prores;
    const std::string profile = prores == "4444xq" ? "5" : prores == "422hq" ? "3" : "4";
    const std::string pix = prores == "422hq" ? "yuv422p10le" : "yuv444p12le";   // ProRes 4444 is 12-bit
    std::string vf = std::string("scale=out_color_matrix=") + (o.pq && !sdr ? "bt2020" : "bt709") +
                     ":out_range=tv:flags=accurate_rnd+full_chroma_int";
    if (o.pq) vf += ",setparams=range=tv:color_primaries=" + prim + ":color_trc=" + trc + ":colorspace=" + matrix;
    std::vector<std::string> a = {"-vf", vf, "-c:v", "prores_ks", "-profile:v", profile, "-qscale:v", "1", "-vendor", "apl0",
                                  "-pix_fmt", pix, "-colorspace", matrix, "-color_range", "tv",
                                  "-color_primaries", prim, "-color_trc", trc, "-movflags", "+write_colr"};
    if (o.pq)
        a.insert(a.end(), {"-bsf:v", "prores_metadata=color_primaries=" + prim + ":color_trc=" + trc + ":colorspace=" + matrix});
    return a;
}

std::vector<std::string> AcesLogWriter::hevc_arguments(double maxcll, double maxfall, double peak_nits) {
    // SMPTE ST 2086 in x265's units (0.00002 for x,y; 0.0001 nits): P3 D65, 1000 / 0.0001 nits,
    // the display the ACES 2.0 HDR 1000 nits (P3 D65) output is made for. CTA-861.3 light level.
    // CTA-861.3 integers, rounded up as the analytic route's (core/measure maxcll_maxfall) are.
    const long cll = std::lround(std::ceil(std::max(0.0, maxcll))), fall = std::lround(std::ceil(std::max(0.0, std::min(maxfall, maxcll))));
    const std::string x265 = "hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                             "range=limited:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(" +
                             std::to_string(std::lround(peak_nits * 10000.0)) + ",1):"
                             "max-cll=" + std::to_string(cll) + "," + std::to_string(fall);
    return {"-c:v", "libx265", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p10le", "-profile:v", "main10",
            "-x265-params", x265, "-tag:v", "hvc1", "-color_primaries", "bt2020", "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc", "-color_range", "tv"};
}

Result<void> stamp_dpx_header(const fs::path& dpx, const std::string& timecode, double fps, const std::string& reel,
                              int transfer, int colorimetric) {
    std::fstream f(dpx, std::ios::in | std::ios::out | std::ios::binary);
    char magic[4] = {};
    if (!f.read(magic, 4)) return make_error(ErrorCode::IoError, "The DPX cannot be opened to stamp it.", dpx.string());
    const bool big = std::string(magic, 4) == "SDPX";
    if (!big && std::string(magic, 4) != "XPDS") return make_error(ErrorCode::ParseError, "Not a DPX file.", dpx.string());
    auto put32 = [&](std::streamoff at, std::uint32_t v) {
        unsigned char b[4];
        for (int i = 0; i < 4; ++i) b[i] = static_cast<unsigned char>(big ? v >> (24 - 8 * i) : v >> (8 * i));
        f.seekp(at);
        f.write(reinterpret_cast<const char*>(b), 4);
    };
    auto putf = [&](std::streamoff at, float v) {
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        put32(at, u);
    };
    auto put8 = [&](std::streamoff at, int v) {
        const char c = static_cast<char>(v);
        f.seekp(at);
        f.write(&c, 1);
    };
    put8(801, transfer);        // image element 1: transfer characteristic
    put8(802, colorimetric);    // and colorimetric specification
    if (fps > 0) {
        putf(1724, float(fps));   // film header: frame rate
        putf(1940, float(fps));   // TV header: temporal sampling rate
    }
    if (const auto tc = parse_timecode(timecode)) {
        put32(1920, exr_timecode(*tc)[0]);   // TV header: SMPTE time code (BCD, drop flag as 12M)
        put32(1924, 0);                       // user bits
    }
    if (!reel.empty()) {   // orientation header: input device name, 32 bytes
        char name[32] = {};
        std::memcpy(name, reel.data(), std::min<std::size_t>(reel.size(), 31));
        f.seekp(1556);
        f.write(name, 32);
    }
    f.flush();
    if (!f) return make_error(ErrorCode::IoError, "The DPX header could not be stamped.", dpx.string());
    return {};
}

Result<void> AcesLogWriter::write(const PlanarBuffer& acescg, const std::string& name, const std::string& timecode) {
    const PlanarBuffer codes = this->codes(acescg);
    const int h = codes.height(), w = codes.width();
    if (o_.tiff16) {
        const fs::path tif = dir_ / (fs::path(name).stem().string() + ".tif");
        const int bgr[3] = {2, 1, 0};   // OpenCV writes BGR
        const auto px = to_u16(codes, bgr);
        if (auto r = write_png16(tif, px, w, h, 3); !r)
            return make_error(ErrorCode::IoError, "The 16-bit TIFF could not be written.", tif.string());
    }
    if (o_.dpx10) {
        // One ffmpeg per frame: RGB 16-bit in, 10-bit DPX out (gbrp10le), the frame's own name.
        auto ff = require_executable("ffmpeg");
        if (!ff) return ff.error();
        const fs::path dpx = dpx_dir_ / (fs::path(name).stem().string() + ".dpx");
        const fs::path elog = dpx_dir_ / "ffmpeg.log";
        auto p = Process::start_writer({ff->string(), "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                                        "-s", std::to_string(w) + "x" + std::to_string(h), "-i", "pipe:0", "-frames:v", "1",
                                        "-pix_fmt", "gbrp10le", dpx.string()},
                                       elog);
        if (!p) return p.error();
        // Rounded to the 10-bit codes first: ffmpeg's 16-to-10-bit step truncates.
        PlanarBuffer q = codes;
        for (float& v : q.span()) v = std::nearbyint(v * 1023.0f) / 1023.0f;
        const int rgb[3] = {0, 1, 2};
        const auto px = to_u16(q, rgb);
        const bool sent = (*p)->write({reinterpret_cast<const std::uint8_t*>(px.data()), px.size() * 2});
        (*p)->close_input();
        if (!sent || (*p)->wait() != 0)
            return make_error(ErrorCode::IoError, "The 10-bit DPX could not be written.", dpx.string());
        const std::string tf = o_.pq ? o_.display_target().transfer : "";
        const int code = tf == "sdr" ? 6 : 0;   // BT.709 for the SDR pass; user-defined otherwise
        if (auto st = stamp_dpx_header(dpx, timecode, fps_, reel_, code, code); !st) return st.error();
        std::error_code ec;
        if (fs::exists(elog, ec) && fs::file_size(elog, ec) == 0) fs::remove(elog, ec);
    }
    if (o_.movie()) {
        if (!ffmpeg_) {
            if (!o_.mov16 && (w % 2 || h % 2))
                return make_error(ErrorCode::InvalidArgument, "ProRes needs even frame dimensions; use the TIFFs or the 16-bit movie.", name);
            auto ff = require_executable("ffmpeg");
            if (!ff) return ff.error();
            log_ = movie_.parent_path() / (movie_.stem().string() + "_ffmpeg.log");
            // Written beside as <stem>.part.<ext> (an intermediate .mov for HEVC), renamed when whole.
            part_ = movie_.parent_path() / (movie_.stem().string() + ".part" + (o_.hevc10 ? std::string(".mov") : movie_.extension().string()));
            std::vector<std::string> argv = {
                ff->string(), "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s", std::to_string(w) + "x" + std::to_string(h), "-r", fps_text(fps_), "-i", "pipe:0"};
            for (auto& a : movie_arguments(o_)) argv.push_back(std::move(a));
            // The source's timecode and reel (a QuickTime tmcd track), for the conform.
            if (!timecode_.empty()) argv.insert(argv.end(), {"-timecode", timecode_});
            if (!reel_.empty()) argv.insert(argv.end(), {"-metadata:s:v:0", "reel_name=" + reel_});
            argv.push_back(part_.string());
            auto p = Process::start_writer(argv, log_);
            if (!p) return p.error();
            ffmpeg_ = std::move(*p);
            width_ = w, height_ = h;
        }
        if (w != width_ || h != height_)
            return make_error(ErrorCode::InvalidArgument, "Frames of different sizes cannot share one movie.", name);
        const int rgb[3] = {0, 1, 2};
        const auto px = to_u16(codes, rgb);
        if (!ffmpeg_->write({reinterpret_cast<const std::uint8_t*>(px.data()), px.size() * 2}))
            return make_error(ErrorCode::IoError, "ffmpeg stopped while writing the movie.", log_.string());
    }
    ++frames_;
    return {};
}

Result<void> AcesLogWriter::finish(double maxcll, double maxfall) {
    if (!ffmpeg_) return {};
    ffmpeg_->close_input();
    const int code = ffmpeg_->wait();
    ffmpeg_.reset();
    auto log_text = [this] {
        std::ifstream in(log_);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return text.substr(0, 2000);
    };
    if (code != 0) return make_error(ErrorCode::IoError, "ffmpeg could not write the movie.", log_text());
    std::error_code ec;
    if (o_.hevc10) {   // the HDR10 pass: x265 with the run's measured light level, then the intermediate goes
        maxcll_ = maxcll, maxfall_ = maxfall;
        auto ff = require_executable("ffmpeg");
        if (!ff) return ff.error();
        const fs::path hevc_part = movie_.parent_path() / (movie_.stem().string() + ".part" + movie_.extension().string());
        std::vector<std::string> argv = {ff->string(), "-hide_banner", "-v", "error", "-y", "-i", part_.string(),
                                         "-map", "0:v:0"};
        for (auto& a : hevc_arguments(maxcll, maxfall, o_.display_target().peak_nits)) argv.push_back(std::move(a));
        if (!timecode_.empty()) argv.insert(argv.end(), {"-timecode", timecode_});
        argv.insert(argv.end(), {"-f", "mp4", hevc_part.string()});
        auto p = Process::start_writer(argv, log_);
        if (!p) return p.error();
        (*p)->close_input();
        if ((*p)->wait() != 0) return make_error(ErrorCode::IoError, "ffmpeg could not write the HDR10 HEVC.", log_text());
        fs::remove(part_, ec);
        part_ = hevc_part;
    }
    fs::rename(part_, movie_, ec);
    if (ec) return make_error(ErrorCode::IoError, "The finished movie could not be put in place.", movie_.string());
    if (fs::exists(log_, ec) && fs::file_size(log_, ec) == 0) fs::remove(log_, ec);
    return {};
}

nlohmann::json AcesLogWriter::provenance() const {
    nlohmann::json files = nlohmann::json::array();
    if (o_.tiff16) files.push_back({{"kind", "16-bit TIFF per frame (lossless)"}, {"folder", dir_.filename().string()}});
    if (o_.dpx10) files.push_back({{"kind", "10-bit DPX per frame (RGB)"}, {"folder", dpx_dir_.filename().string()}});
    const std::string tf = o_.pq ? o_.display_target().transfer : "";
    const std::string tags = !o_.pq      ? "primaries and transfer unknown (no codes for the log)"
                             : tf == "hlg" ? "BT.2020 primaries, ARIB STD-B67 (HLG)"
                             : tf == "sdr" ? "BT.709 primaries, BT.709 transfer (BT.1886 display)"
                                           : "BT.2020 primaries, SMPTE ST 2084 (PQ)";
    if (o_.hevc10)
        files.push_back({{"kind", "HDR10 HEVC Main 10, 4:2:0 (lossy; x265 CRF 12 from a ProRes 4444 XQ intermediate)"},
                         {"file", movie_.filename().string()}, {"fps", fps_}, {"bits", 10}, {"timecode", timecode_},
                         {"tags", "BT.2020 primaries, SMPTE ST 2084 (PQ), BT.2020nc matrix, limited range"},
                         {"mastering_display", "P3 D65, " + std::to_string(int(o_.display_target().peak_nits)) +
                                                   " / 0.0001 nits (SMPTE ST 2086, in the SEI)"},
                         {"maxcll", std::lround(std::ceil(maxcll_))}, {"maxfall", std::lround(std::ceil(std::min(maxfall_, maxcll_)))}});
    else if (o_.mov16)
        files.push_back({{"kind", "Uncompressed RGB 16-bit QuickTime (b48r, lossless)"}, {"file", movie_.filename().string()},
                         {"fps", fps_}, {"bits", 16}, {"timecode", timecode_}, {"reel", reel_},
                         {"tags", "RGB, full range; " + tags}});
    else if (!o_.prores.empty())
        files.push_back({{"kind", o_.prores == "4444xq" ? "ProRes 4444 XQ, 12-bit 4:4:4 (lossy)"
                                  : o_.prores == "4444" ? "ProRes 4444, 12-bit 4:4:4 (lossy)"
                                                        : "ProRes 422 HQ, 10-bit 4:2:2 (lossy)"},
                         {"file", movie_.filename().string()}, {"fps", fps_}, {"timecode", timecode_}, {"reel", reel_},
                         {"bits", o_.prores == "4444xq" || o_.prores == "4444" ? 12 : 10},
                         {"tags", std::string(o_.pq && tf != "sdr" ? "BT.2020" : "BT.709") + " YCbCr matrix, limited range; " + tags}});
    if (o_.pq) {
        const AcesDisplayTarget& t = o_.display_target();
        const std::string peak = std::to_string(int(t.peak_nits));
        return {{"curve", t.label}, {"encoding", t.id},
                {"gamut", tf == "sdr" ? "Rec.709" : "Rec.2020 (P3 D65 limited)"},
                {"frames", frames_}, {"files", files}, {"display_referred", true},
                {"source", "the ACEScg master of each frame through the ACES 2.0 output transform"},
                {"output_transform", display_ ? display_->identity() : std::string()},
                {"clamp", "codes outside [0, 1] are clamped; peak " + peak + " nits"},
                {"note", tf == "sdr" ? "the SDR trim pass of the same master; the rudra.reconstructed mask stays in the EXR"
                                     : "a display-referred HDR deliverable; the rudra.reconstructed mask stays in the EXR"},
                {"input_transform", tf == "hlg"   ? "set the clip to Rec.2100 HLG (Rec.2020); no further output transform"
                                    : tf == "sdr" ? "set the clip to Rec.709 (BT.1886); no further output transform"
                                                  : "set the clip to Rec.2100 PQ (ST 2084, Rec.2020); no further output transform"}};
    }
    const LogCurve c = *o_.curve;
    return {{"curve", log_curve_label(c)}, {"encoding", o_.name()}, {"gamut", log_gamut_label(c)}, {"frames", frames_},
            {"files", files}, {"display_referred", false},
            {"source", "the ACEScg master of each frame, no exposure change, no peak or knee"},
            {"clamp", c == LogCurve::LogC4 ? "codes above 1.0 (469.8 x diffuse white) are clamped" : "codes above 1.0 are clamped"},
            {"note", "camera log of the master; the rudra.reconstructed mask stays in the EXR"},
            {"input_transform", c == LogCurve::LogC4 ? "set the clip to ARRI LogC4 / ARRI Wide Gamut 4" : "set the clip to ACEScct"}};
}

}  // namespace rudra
