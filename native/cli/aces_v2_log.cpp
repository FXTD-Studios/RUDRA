#include "aces_v2_log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>

#include "rudra/core/gamut.hpp"
#include "rudra/media/png16.hpp"
#include "rudra/platform/process.hpp"

namespace fs = std::filesystem;

namespace rudra {

Result<AcesLogOptions> parse_aces_log(const std::string& curve, const std::string& formats) {
    AcesLogOptions o;
    if (curve.empty() || curve == "none") return o;
    o.curve = log_curve_from_name(curve);
    if (!o.curve) return make_error(ErrorCode::InvalidArgument, "Unknown log curve (logc4 or acescct).", curve);
    std::stringstream ss(formats.empty() ? std::string("tiff16") : formats);
    std::string f;
    while (std::getline(ss, f, ',')) {
        if (f == "tiff16") o.tiff16 = true;
        else if (f == "dpx10") o.dpx10 = true;
        else if (f == "prores4444xq" || f == "prores4444" || f == "prores422hq") {
            if (!o.prores.empty()) return make_error(ErrorCode::InvalidArgument, "Choose one ProRes format.", formats);
            o.prores = f.substr(6);
        } else if (!f.empty())
            return make_error(ErrorCode::InvalidArgument,
                              "Unknown log format (tiff16, dpx10, prores4444xq, prores4444, prores422hq).", f);
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
                                                           const fs::path& movie, double fps) {
    std::unique_ptr<AcesLogWriter> w(new AcesLogWriter());
    w->o_ = o;
    w->movie_ = movie;
    w->fps_ = fps > 0 ? fps : 24.0;
    w->dir_ = out_dir / ("log_" + std::string(log_curve_name(*o.curve)));
    w->dpx_dir_ = out_dir / ("log_" + std::string(log_curve_name(*o.curve)) + "_dpx");
    std::error_code ec;
    if (o.tiff16) fs::create_directories(w->dir_, ec);
    if (!ec && o.dpx10) fs::create_directories(w->dpx_dir_, ec);
    if (ec) return make_error(ErrorCode::IoError, "The log folder could not be made.", out_dir.string());
    if (o.dpx10 || !o.prores.empty()) {
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

Result<void> AcesLogWriter::write(const PlanarBuffer& acescg, const std::string& name) {
    const PlanarBuffer codes = aces_log_codes(acescg, *o_.curve);
    const int h = codes.height(), w = codes.width();
    if (o_.tiff16) {
        const fs::path tif = dir_ / (fs::path(name).stem().string() + ".tif");
        const int bgr[3] = {2, 1, 0};   // OpenCV writes BGR
        const auto px = to_u16(codes, bgr);
        if (auto r = write_png16(tif, px, w, h, 3); !r)
            return make_error(ErrorCode::IoError, "The 16-bit log TIFF could not be written.", tif.string());
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
            return make_error(ErrorCode::IoError, "The 10-bit log DPX could not be written.", dpx.string());
        std::error_code ec;
        if (fs::exists(elog, ec) && fs::file_size(elog, ec) == 0) fs::remove(elog, ec);
    }
    if (!o_.prores.empty()) {
        if (!ffmpeg_) {
            if (w % 2 || h % 2)
                return make_error(ErrorCode::InvalidArgument, "ProRes needs even frame dimensions; use the TIFFs.", name);
            auto ff = require_executable("ffmpeg");
            if (!ff) return ff.error();
            const std::string profile = o_.prores == "4444xq" ? "5" : o_.prores == "4444" ? "4" : "3";
            const std::string pix = o_.prores == "422hq" ? "yuv422p10le" : "yuv444p12le";
            log_ = movie_.parent_path() / (movie_.stem().string() + "_ffmpeg.log");
            const std::vector<std::string> argv = {
                ff->string(), "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                "-s", std::to_string(w) + "x" + std::to_string(h), "-r", fps_text(fps_), "-i", "pipe:0",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int",
                "-c:v", "prores_ks", "-profile:v", profile, "-qscale:v", "1", "-vendor", "apl0",
                "-pix_fmt", pix, "-colorspace", "bt709", "-color_range", "tv",
                "-color_primaries", "unknown", "-color_trc", "unknown", movie_.string()};
            auto p = Process::start_writer(argv, log_);
            if (!p) return p.error();
            ffmpeg_ = std::move(*p);
            width_ = w, height_ = h;
        }
        if (w != width_ || h != height_)
            return make_error(ErrorCode::InvalidArgument, "Frames of different sizes cannot share one ProRes movie.", name);
        const int rgb[3] = {0, 1, 2};
        const auto px = to_u16(codes, rgb);
        if (!ffmpeg_->write({reinterpret_cast<const std::uint8_t*>(px.data()), px.size() * 2}))
            return make_error(ErrorCode::IoError, "ffmpeg stopped while writing the ProRes movie.", log_.string());
    }
    ++frames_;
    return {};
}

Result<void> AcesLogWriter::finish() {
    if (!ffmpeg_) return {};
    ffmpeg_->close_input();
    const int code = ffmpeg_->wait();
    ffmpeg_.reset();
    if (code != 0) {
        std::ifstream in(log_);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return make_error(ErrorCode::IoError, "ffmpeg could not write the ProRes movie.", text.substr(0, 2000));
    }
    std::error_code ec;
    if (fs::exists(log_, ec) && fs::file_size(log_, ec) == 0) fs::remove(log_, ec);
    return {};
}

nlohmann::json AcesLogWriter::provenance() const {
    const LogCurve c = *o_.curve;
    nlohmann::json files = nlohmann::json::array();
    if (o_.tiff16) files.push_back({{"kind", "16-bit TIFF per frame (lossless)"}, {"folder", dir_.filename().string()}});
    if (o_.dpx10) files.push_back({{"kind", "10-bit DPX per frame (RGB)"}, {"folder", dpx_dir_.filename().string()}});
    if (!o_.prores.empty())
        files.push_back({{"kind", o_.prores == "4444xq" ? "ProRes 4444 XQ, 12-bit 4:4:4 (lossy)"
                                  : o_.prores == "4444" ? "ProRes 4444, 12-bit 4:4:4 (lossy)"
                                                        : "ProRes 422 HQ, 10-bit 4:2:2 (lossy)"},
                         {"file", movie_.filename().string()}, {"fps", fps_},
                         {"tags", "BT.709 YCbCr matrix, limited range; primaries and transfer unknown (no codes for the log)"}});
    return {{"curve", log_curve_label(c)}, {"gamut", log_gamut_label(c)}, {"frames", frames_}, {"files", files},
            {"source", "the ACEScg master of each frame, no exposure change, no peak or knee"},
            {"clamp", c == LogCurve::LogC4 ? "codes above 1.0 (469.8 x diffuse white) are clamped" : "codes above 1.0 are clamped"},
            {"note", "camera log of the master; the rudra.reconstructed mask stays in the EXR"},
            {"input_transform", c == LogCurve::LogC4 ? "set the clip to ARRI LogC4 / ARRI Wide Gamut 4" : "set the clip to ACEScct"}};
}

}  // namespace rudra
