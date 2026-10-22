#include "rudra/video/headroom_source.hpp"

#include <cmath>
#include <mutex>
#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"
#include "headroom_read_lease.hpp"

namespace rudra {
namespace {
std::string utf8(const std::filesystem::path& path) {
    const auto text = path.generic_u8string();
    return {text.begin(), text.end()};
}
}  // namespace

Result<std::string> headroom_source_cache_id(const std::filesystem::path& source,
    const VideoSource& video, HeadroomPolicy policy, const std::filesystem::path& decoder_executable,
    std::stop_token stop) {
    if (stop.stop_requested()) return make_error(ErrorCode::Cancelled, "Headroom preparation was cancelled");
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "A float source cache requires an explicit supported headroom policy");
    const auto rate = parse_fraction(video.clock.fps);
    if (!rate || rate->num <= 0 || !std::isfinite(video.clock.start) || !std::isfinite(video.clock.duration) ||
        video.clock.duration <= 0 || !std::isfinite(video.clock.tolerance) || video.clock.tolerance < 0)
        return make_error(ErrorCode::InvalidArgument, "Invalid clock for headroom source cache identity");
    std::error_code ec;
    const auto input = std::filesystem::canonical(source, ec);
    if (ec) return make_error(ErrorCode::NotFound, "Cannot resolve the headroom source", ec.message());
    const auto decoder = std::filesystem::canonical(decoder_executable, ec);
    if (ec) return make_error(ErrorCode::NotFound, "Cannot resolve the headroom decoder", ec.message());
    auto command = float_decoder_command(utf8(decoder), input, video);
    if (!command) return command.error();
    auto playback = float_playback_decoder_command(utf8(decoder), input, video, {video.clock.frames / 2, 1});
    if (!playback) return playback.error();
    auto source_hash = sha256_file(input, stop);
    if (!source_hash) return source_hash.error();
    auto decoder_hash = sha256_file(decoder, stop);
    if (!decoder_hash) return decoder_hash.error();
    const auto& c = video.contract;
    const auto& s = video.stream;
    const auto& clock = video.clock;
    const nlohmann::json identity = {
        {"schema", "headroom-float-source-v1"}, {"policy", headroom_policy_id(policy)},
        {"source", utf8(input)}, {"source_sha256", *source_hash},
        {"decoder", utf8(decoder)}, {"decoder_sha256", *decoder_hash},
        {"decode_argv", *command}, {"storage", "planar-rgb-f32-source-encoded"},
        {"playback_selection", kFloatPlaybackSelection},
        {"playback_argv_sample", *playback},
        {"contract", {c.transfer, c.primaries, c.matrix, c.range}},
        {"stream", {s.index, s.width, s.height, s.pix_fmt, s.codec_name}},
        {"clock", {clock.fps, clock.frames, clock.start, clock.duration, clock.tolerance}}
    };
    const auto bytes = identity.dump();
    if (stop.stop_requested()) return make_error(ErrorCode::Cancelled, "Headroom preparation was cancelled");
    return "headroom-float-v1-" + sha256_hex(std::as_bytes(std::span(bytes.data(), bytes.size())));
}

struct PreparedHeadroomSource::Impl {
    headroom_detail::ReadLease source, decoder;
    VideoSource video;
    std::string id;
    mutable std::mutex mutex;
    std::optional<Error> failure;
};
PreparedHeadroomSource::PreparedHeadroomSource() : d_(std::make_unique<Impl>()) {}
PreparedHeadroomSource::~PreparedHeadroomSource() = default;

Result<PreparedHeadroomSource::Handle> PreparedHeadroomSource::prepare(
    const std::filesystem::path& source, HeadroomPolicy policy,
    const std::filesystem::path& decoder, HeadroomPrepareOptions options) {
    if (options.stop.stop_requested()) return make_error(ErrorCode::Cancelled, "Headroom preparation was cancelled");
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Headroom preparation requires an explicit supported policy");
    try {
        std::shared_ptr<PreparedHeadroomSource> out(new PreparedHeadroomSource());
        auto& d = *out->d_;
        auto s = d.source.open(source); if (!s) return s.error();
        auto e = d.decoder.open(decoder); if (!e) return e.error();
        auto tool = options.probe_executable.empty() ? require_executable("ffprobe") :
            Result<std::filesystem::path>(options.probe_executable);
        if (!tool) return tool.error();
        // The probing tool cannot change between the two passes. Only source/
        // decoder leases need to persist after preparation has completed.
        headroom_detail::ReadLease probe_lease;
        auto p = probe_lease.open(*tool); if (!p) return p.error();
        auto probe = probe_video(d.source.path, options.stop, probe_lease.path);
        if (!probe) return probe.error();
        auto video = open_video_source(*probe, VideoArgs{}); if (!video) return video.error();
        auto id = headroom_source_cache_id(d.source.path, *video, policy, d.decoder.path, options.stop);
        if (!id) return id.error();
        auto valid = out->verify(); if (!valid) return valid.error();
        valid = probe_lease.verify(); if (!valid) return valid.error();
        if (options.stop.stop_requested()) return make_error(ErrorCode::Cancelled, "Headroom preparation was cancelled");
        d.video = std::move(*video); d.id = std::move(*id);
        return Handle(std::move(out));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::IoError, "Cannot prepare headroom source", e.what());
    }
}
const VideoSource& PreparedHeadroomSource::video() const { return d_->video; }
const std::string& PreparedHeadroomSource::cache_id() const { return d_->id; }
const std::filesystem::path& PreparedHeadroomSource::source_path() const { return d_->source.path; }
const std::filesystem::path& PreparedHeadroomSource::decoder_path() const { return d_->decoder.path; }
Result<void> PreparedHeadroomSource::verify() const {
    std::lock_guard lock(d_->mutex);
    if (d_->failure) return *d_->failure;
    auto valid = d_->source.verify();
    if (valid) valid = d_->decoder.verify();
    if (!valid) d_->failure = valid.error();
    return valid;
}
}  // namespace rudra
