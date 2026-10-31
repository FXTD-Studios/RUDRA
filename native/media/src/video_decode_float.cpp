#include "rudra/media/video_decode_float.hpp"

#include <bit>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <limits>
#include <new>
#include <stdexcept>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "rudra/platform/process.hpp"

namespace rudra {
namespace fs = std::filesystem;
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4 &&
              std::numeric_limits<float>::is_iec559, "gbrpf32le requires little-endian IEEE float32");
namespace {
Error io_error(std::string message) { return make_error(ErrorCode::IoError, std::move(message)); }
Error cancelled() { return make_error(ErrorCode::Cancelled, "Float decoder was cancelled"); }

std::string log_tail(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    if (in.tellg() > std::streampos(2000)) in.seekg(-2000, std::ios::end);
    else in.seekg(0);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}  // namespace

Result<std::vector<std::string>> float_decoder_command(const std::string& ffmpeg,
    const fs::path& source, const VideoSource& video, std::optional<FloatFrameRange> range) {
    const auto& s = video.stream;
    if (video.alpha || has_alpha(s.pix_fmt))
        return make_error(ErrorCode::Unsupported, "Float decoder prototype does not support alpha");
    if (s.pix_fmt != "yuv444p" && s.pix_fmt != "yuv444p10le")
        return make_error(ErrorCode::Unsupported, "Float decoder prototype requires 8-bit or 10-bit YUV444");
    auto inferred = input_contract(s, VideoArgs{});
    if (!inferred) return inferred.error();
    if (*inferred != video.contract)
        return make_error(ErrorCode::ContractMismatch, "Float decoder requires a contract matching the source tags");
    const auto& c = video.contract;
    if (c.range != "limited" || c.transfer != "rec709" ||
        (c.primaries != "rec709" && c.primaries != "rec2020") ||
        (c.matrix != "bt709" && c.matrix != "bt2020nc"))
        return make_error(ErrorCode::Unsupported, "Float decoder prototype requires tagged limited-range Rec.709-transfer video");
    if (s.width <= 0 || s.height <= 0 || video.clock.frames <= 0)
        return make_error(ErrorCode::InvalidArgument, "Invalid float video dimensions or frame count");
    if (range && (range->first < 0 || range->first >= video.clock.frames || range->count <= 0 ||
                  range->count > video.clock.frames - range->first))
        return make_error(ErrorCode::InvalidArgument, "Float frame range is outside the movie");
    const std::string matrix = c.matrix == "bt709" ? "709" : "2020_ncl";
    std::string filter = "zscale=matrixin=" + matrix + ":rangein=limited:matrix=gbr:range=full,format=gbrpf32le";
    if (range) filter = "trim=start_frame=" + std::to_string(range->first) +
        ":end_frame=" + std::to_string(range->first + range->count) + "," + filter;
    // Measured FFV1 profile: two input workers retain much less memory than
    // four with comparable pipe throughput. Other codecs retain their existing
    // profile until measured. These options do not change conversion or pixels.
    const bool bounded_ffv1 = s.codec_name == "ffv1";
    std::vector<std::string> command{ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-nostdin",
        "-noautorotate", "-threads", bounded_ffv1 ? "2" : "4", "-i", source.string(), "-map", "0:v:0", "-vf",
        filter};
    if (range) command.insert(command.end(), {"-frames:v", std::to_string(range->count)});
    // Output-scoped rawvideo controls: one packing worker and one queued packet.
    // This bounds these components, not FFmpeg's entire working set/frame pools.
    if (bounded_ffv1) command.insert(command.end(), {"-threads:v", "1", "-thread_queue_size", "1"});
    command.insert(command.end(), {
        "-fps_mode", "passthrough", "-pix_fmt", "gbrpf32le", "-f", "rawvideo", "pipe:1"});
    return command;
}

Result<std::vector<std::string>> float_playback_decoder_command(const std::string& ffmpeg,
    const fs::path& source, const VideoSource& video, FloatFrameRange range) {
    auto command = float_decoder_command(ffmpeg, source, video, range);
    if (!command) return command.error();
    // Only the lossless codec/container combination covered by seek parity
    // fixtures is enabled. The indexed path remains the reference/fallback.
    auto extension = source.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    const auto rate = parse_fraction(video.clock.fps);
    if (video.stream.codec_name != "ffv1" || extension != ".mkv" || range.first == 0 ||
        !rate || rate->num <= 0 || !std::isfinite(video.clock.start) || video.clock.start < 0 ||
        !std::isfinite(video.clock.tolerance) || video.clock.tolerance < 0) return command;
    const double period = double(rate->den) / double(rate->num);
    // video_timing verifies EVERY PTS within tolerance of start + i*period.
    // Quarter-period slack leaves both rounding and timestamp jitter safely
    // inside the half-period gap. Reject unsafe numeric ranges to indexed trim.
    const double begin = video.clock.start + (double(range.first) - .5) * period;
    const double end = video.clock.start + (double(range.first + range.count) - .5) * period;
    if (!std::isfinite(period) || !std::isfinite(begin) || !std::isfinite(end) ||
        period <= 0 || video.clock.tolerance + 0.000002 >= period / 4 ||
        begin < 0 || end > 1e8 || end <= begin) return command;
    double container_start = 0;
    try {
        const auto metadata = nlohmann::json::parse(video.streams_json);
        const auto& format = metadata.at("format");
        if (format.at("format_name").get<std::string>() != "matroska,webm") return command;
        const auto value = format.at("start_time").get<std::string>();
        std::size_t used = 0; container_start = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(container_start)) return command;
    } catch (const std::exception&) { return command; }
    const double seek_offset = begin - container_start;
    if (!std::isfinite(seek_offset) || seek_offset < 0 || seek_offset > 1e8) return command;
    auto seconds = [](double value) {
        std::ostringstream out; out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(9) << value; return out.str();
    };
    auto at = std::find(command->begin(), command->end(), "-i");
    // Relative input seek uses the container start, which may differ from the
    // first video PTS (e.g. earlier audio). copyts keeps absolute PTS for trim.
    // FFmpeg 7.1.1 absolute seek_timestamp + accurate_seek lost offset fixtures.
    command->insert(at, {"-copyts", "-accurate_seek", "-ss", seconds(seek_offset)});
    at = std::find(command->begin(), command->end(), "-vf") + 1;
    const auto conversion = at->substr(at->find(',') + 1);
    *at = "trim=start=" + seconds(begin) + ":end=" + seconds(end) + "," + conversion;
    return command;
}

FloatVideoDecoder::~FloatVideoDecoder() { close(); }

Result<std::unique_ptr<FloatVideoDecoder>> FloatVideoDecoder::open(const fs::path& source,
    const VideoSource& video, const fs::path& log, std::optional<FloatFrameRange> range) {
    // Refuse unsupported media before looking for or launching FFmpeg.
    auto command = float_decoder_command("ffmpeg", source, video, range);
    if (!command) return command.error();
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    (*command)[0] = ffmpeg->string();
    return open_command(*command, video.stream.width, video.stream.height,
                        range ? range->count : video.clock.frames, video.contract, log);
}

Result<std::unique_ptr<FloatVideoDecoder>> FloatVideoDecoder::open_command(const std::vector<std::string>& argv,
    int width, int height, int frames, const InputContract& contract, const fs::path& log) {
    // Check the entire allocation in bytes, before multiplying dimensions.
    if (width <= 0 || height <= 0 || frames < 0 ||
        std::size_t(width) > std::numeric_limits<std::size_t>::max() / std::size_t(height) / (3 * sizeof(float)) ||
        std::size_t(width) > std::vector<float>().max_size() / std::size_t(height) / 3)
        return make_error(ErrorCode::InvalidArgument, "Invalid float decoded frame shape");
    auto process = Process::start(argv, log);
    if (!process) return process.error();
    std::unique_ptr<FloatVideoDecoder> decoder(new FloatVideoDecoder());
    decoder->process_ = std::move(*process);
    decoder->width_ = width;
    decoder->height_ = height;
    decoder->frames_ = frames;
    decoder->contract_ = contract;
    decoder->log_ = log;
    return decoder;
}

Error FloatVideoDecoder::remember(Error error) {
    failure_ = std::move(error);
    return *failure_;
}

Result<ExtendedVideoFrame> FloatVideoDecoder::next() {
    if (failure_) return *failure_;
    if (interrupted_) return remember(cancelled());
    if (!process_ || finished_) return io_error("Float decoder is closed");
    if (index_ >= frames_)
        return make_error(ErrorCode::InvalidArgument, "All expected float frames have been read; call finish");
    PlanarBuffer rgb;
    try {
        rgb = PlanarBuffer(3, height_, width_);
    } catch (const std::bad_alloc&) {
        return remember(io_error("Could not allocate a float decoded frame"));
    } catch (const std::length_error&) {
        return remember(io_error("Float decoded frame exceeds the buffer size limit"));
    }
    // FFmpeg sends G, B, R. Read directly into their final planes, avoiding
    // a second full-sized frame and a separate permutation/copy pass.
    std::size_t total = 0;
    for (int channel : {1, 2, 0}) {
        auto bytes = std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(rgb.plane(channel)),
                                           rgb.plane_size() * sizeof(float));
        const auto got = process_->read(bytes);
        total += got;
        if (interrupted_) return remember(cancelled());
        if (got != bytes.size())
            return remember(io_error(total == 0 ? "Float decoder ended before expected frame count"
                                                : "Truncated float decoded frame"));
    }
    for (float value : rgb.span())
        if (!std::isfinite(value)) return remember(io_error("Nonfinite sample in float decoded frame"));
    ++index_;
    return ExtendedVideoFrame{std::move(rgb), contract_};
}

Result<void> FloatVideoDecoder::finish() {
    if (interrupted_ && !failure_) failure_ = cancelled();
    if (failure_) {
        stop_process();
        return *failure_;
    }
    if (!process_ || finished_) return io_error("Float decoder is closed");
    if (index_ != frames_)
        return make_error(ErrorCode::InvalidArgument, "Read all expected float frames before finish");
    std::uint8_t extra[1];
    const auto extra_bytes = process_->read(extra);
    if (interrupted_) {
        stop_process();
        return remember(cancelled());
    }
    if (extra_bytes != 0) {
        stop_process();
        return remember(io_error("Float decoder produced unexpected extra frames"));
    }
    int code = -1;
    for (;;) {
        {
            std::lock_guard lock(process_mutex_);
            if (!process_->running()) { code = process_->wait(); break; }
        }
        // Keep interrupt available even if stdout closed before child exit.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    finished_ = true;
    if (interrupted_) return remember(cancelled());
    if (code != 0)
        return remember(make_error(ErrorCode::IoError, "Float decoder process failed", log_tail(log_)));
    return {};
}

void FloatVideoDecoder::interrupt() {
    std::lock_guard lock(process_mutex_);
    interrupted_ = true;
    if (process_) process_->kill();
}

void FloatVideoDecoder::stop_process() {
    std::lock_guard lock(process_mutex_);
    if (!process_) return;
    if (process_->running()) process_->kill();
    process_->wait();
    finished_ = true;
}

void FloatVideoDecoder::close() {
    stop_process();
    process_.reset();
}
}  // namespace rudra
