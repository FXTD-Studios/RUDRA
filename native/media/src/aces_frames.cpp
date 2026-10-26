#include "rudra/media/aces_frames.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <vector>

#include "rudra/platform/process.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {
std::string lower_ext(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}
}  // namespace

bool aces_is_still(const fs::path& p) {
    const std::string e = lower_ext(p);
    return e == ".png" || e == ".tif" || e == ".tiff" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".webp";
}

bool aces_is_video(const fs::path& p) {
    const std::string e = lower_ext(p);
    for (const char* v : {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mxf", ".webm", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".wmv"})
        if (e == v) return true;
    return false;
}

VideoArgs aces_video_args(const VideoStreamInfo& stream) {
    VideoArgs a;
    // Code values pass through unchanged; transfer and primaries only gate the
    // generic HDR10 path, so they are fixed here (HDR input is still refused).
    a.input_transfer = "rec709";
    a.input_primaries = "rec709";
    const std::string& pix = stream.pix_fmt;
    const bool rgb = pix.starts_with("rgb") || pix.starts_with("bgr") || pix.starts_with("gbr");
    if (!rgb) {
        const auto& cs = stream.color_space;
        if (!cs || *cs == "unknown" || *cs == "reserved" || cs->empty()) a.input_matrix = "bt709";
        const auto& cr = stream.color_range;
        if (!cr || (*cr != "tv" && *cr != "pc")) a.input_range = "limited";
    }
    if (has_alpha(pix)) {   // the alpha is decoded and dropped
        a.format = "prores4444";
        a.alpha_mode = "straight";
    }
    return a;
}

Result<VideoSource> aces_open_video(const fs::path& file) {
    auto probe = probe_video(file);
    if (!probe) return probe.error();
    if (probe->video_streams.size() != 1)
        return make_error(ErrorCode::InvalidArgument, "Select a source containing exactly one video stream", file.string());
    const VideoStreamInfo& stream = probe->video_streams.front();
    auto contract = input_contract(stream, aces_video_args(stream));
    if (!contract) return contract.error();
    for (const auto& f : probe->frames)
        if (f.width != stream.width || f.height != stream.height)
            return make_error(ErrorCode::Unsupported, "Changing frame dimensions are unsupported", file.string());
    // Every decoded frame is mastered as it comes (no audio, no retiming), so a
    // variable frame rate is accepted here; the count is the frames ffprobe saw.
    VideoClock clock;
    if (auto c = video_timing(stream, probe->frames)) clock = *c;
    else {
        clock.fps = stream.avg_frame_rate;
        clock.frames = static_cast<int>(probe->frames.size());
    }
    return VideoSource{stream, *contract, clock, has_alpha(stream.pix_fmt), probe->streams_json};
}

int aces_video_bits(const std::string& pix_fmt) {
    for (int b : {16, 12, 10})
        if (pix_fmt.find(std::to_string(b)) != std::string::npos) return b;
    return 8;
}

std::string aces_video_exr_name(const fs::path& video, int index) {
    char n[16];
    std::snprintf(n, sizeof n, "_%06d.exr", index);
    return video.stem().string() + n;
}

namespace {
// The decode tools/rudra_aces_export.py uses (swscale, accurate rounding, full
// chroma interpolation), so a frame reaches the model as the Python reads it.
std::vector<std::string> aces_decoder_command(const std::string& ffmpeg, const fs::path& source, const VideoSource& video,
                                              int only_frame) {
    const InputContract& c = video.contract;
    std::string vf;
    if (only_frame >= 0) vf = "select=eq(n\\," + std::to_string(only_frame) + "),";
    if (c.matrix == "gbr") vf += "scale=flags=accurate_rnd+full_chroma_int";
    else
        vf += std::string("scale=in_color_matrix=") + (c.matrix == "bt2020nc" ? "bt2020" : "bt709") +
              ":in_range=" + (c.range == "full" ? "pc" : "tv") + ":out_range=pc:flags=accurate_rnd+full_chroma_int";
    vf += video.alpha ? ",format=rgba64le" : ",format=rgb48le";
    std::vector<std::string> a = {ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-noautorotate", "-i", source.string(),
                                  "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough"};
    if (only_frame >= 0) a.insert(a.end(), {"-frames:v", "1"});
    a.insert(a.end(), {"-pix_fmt", video.alpha ? "rgba64le" : "rgb48le", "-f", "rawvideo", "pipe:1"});
    return a;
}
}  // namespace

Result<std::unique_ptr<VideoDecoder>> aces_open_video_decoder(const fs::path& file, const VideoSource& video,
                                                              const fs::path& log) {
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    return VideoDecoder::open_command(aces_decoder_command(ffmpeg->string(), file, video, -1), video.stream.width,
                                      video.stream.height, video.alpha ? 4 : 3, video.clock.frames, log);
}

Result<RawFrame16> aces_read_video_frame(const fs::path& file, const VideoSource& video, int index, const fs::path& log) {
    if (index < 0 || index >= video.clock.frames)
        return make_error(ErrorCode::InvalidArgument, "No such frame in the video.", std::to_string(index));
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    auto d = VideoDecoder::open_command(aces_decoder_command(ffmpeg->string(), file, video, index), video.stream.width,
                                        video.stream.height, video.alpha ? 4 : 3, 1, log);
    if (!d) return d.error();
    auto f = (*d)->next();
    (*d)->close();
    return f;
}

}  // namespace rudra
