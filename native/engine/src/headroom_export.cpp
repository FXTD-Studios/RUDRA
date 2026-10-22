#include "rudra/engine/headroom_export.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sequence_encode.hpp"
#include "rudra/deliver/video_encode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/video/headroom_render.hpp"
#include "rudra/video/headroom_source.hpp"
#include "rudra/video/headroom_stream.hpp"

#ifndef RUDRA_BUILD_COMMIT
#define RUDRA_BUILD_COMMIT "unknown"
#endif

namespace rudra {
namespace fs = std::filesystem;

fs::path headroom_export_sidecar(const fs::path& file) {
    fs::path s = file;
    s += ".json";
    return s;
}

namespace {
double fps_of(const std::string& text) {
    const auto slash = text.find('/');
    try {
        if (slash == std::string::npos) return std::stod(text);
        const double n = std::stod(text.substr(0, slash)), d = std::stod(text.substr(slash + 1));
        return d > 0 ? n / d : 0.0;
    } catch (...) {
        return 0.0;
    }
}
Error cancelled() { return make_error(ErrorCode::Cancelled, "The headroom export was cancelled"); }
}  // namespace

Result<HeadroomExportResult> export_headroom_movie(const HeadroomExportRequest& req,
    const std::function<void(const HeadroomExportProgress&)>& progress, const std::atomic<bool>* cancel) {
    auto say = [&](const char* phase, int done, int total) {
        if (progress) progress({phase, done, total});
    };
    auto stopped = [&] { return cancel && cancel->load(); };

    const SequenceTarget* target = find_sequence_target(req.target);
    if (!target) return make_error(ErrorCode::InvalidArgument, "Unknown headroom export format: " + req.target);
    if (!(req.peak_nits > 0.0)) return make_error(ErrorCode::InvalidArgument, "The export peak must be positive");
    if (req.source.empty() || req.output.empty())
        return make_error(ErrorCode::InvalidArgument, "The headroom export needs a source and an output");
    fs::path final_file = req.output;
    final_file.replace_extension(target->suffix);
    const fs::path final_sidecar = headroom_export_sidecar(final_file);
    std::error_code ec;
    if (fs::exists(final_file, ec) || fs::exists(final_sidecar, ec))
        return make_error(ErrorCode::InvalidArgument, "The export already exists; choose another name", final_file.string());
    if (auto valid = validate_headroom_grade(req.grade); !valid) return valid.error();

    fs::path decoder = req.decoder;
    if (decoder.empty()) {
        auto found = require_executable("ffmpeg");
        if (!found) return found.error();
        decoder = *found;
    }
    auto staging = StagingDir::create(final_file.parent_path(), ".rudra-headroom-");
    if (!staging) return staging.error();
    const fs::path frames_dir = staging->path() / "frames";
    fs::create_directories(frames_dir, ec);
    if (ec) return make_error(ErrorCode::IoError, "Could not create the export staging folder", ec.message());

    // The source: hashed for provenance, prepared and protected for the decode.
    say("prepare", 0, 0);
    auto sha_before = sha256_file(req.source);
    if (!sha_before) return sha_before.error();
    auto prepared = PreparedHeadroomSource::prepare(req.source, HeadroomPolicy::Bt1886V1, decoder);
    if (!prepared) return prepared.error();
    const VideoSource& video = (*prepared)->video();
    const int count = video.clock.frames;
    if (count <= 0) return make_error(ErrorCode::InvalidArgument, "The movie has no frames to export");
    const double fps = req.fps > 0.0 ? req.fps : fps_of(video.clock.fps);
    if (!(fps > 0.0)) return make_error(ErrorCode::InvalidArgument, "The movie's frame rate is unknown; choose one");

    auto stream = HeadroomStream::open(*prepared);
    if (!stream) return stream.error();
    if (auto s = (*stream)->start({0, count}); !s) return s.error();

    MasterRequest q = req.grade;
    q.source_space = video.contract.primaries;
    std::vector<fs::path> frames;
    frames.reserve(std::size_t(count));
    int width = 0, height = 0;
    for (int i = 0; i < count; ++i) {
        if (stopped()) {
            (*stream)->cancel();
            return cancelled();
        }
        auto frame = (*stream)->next();
        for (int busy = 0; !frame && frame.error().code == ErrorCode::Busy && busy < 2000; ++busy) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (stopped()) {
                (*stream)->cancel();
                return cancelled();
            }
            frame = (*stream)->next();
        }
        if (!frame) return frame.error();
        if (frame->index() != i)
            return make_error(ErrorCode::IntegrityError, "The decoder returned frames out of order",
                              std::to_string(frame->index()) + " for " + std::to_string(i));
        auto master = render_headroom_frame(frame->pixels(), q, HeadroomPolicy::Bt1886V1);
        if (!master) return master.error();
        if (auto valid = (*stream)->validate_preview(*frame); !valid) return valid.error();
        width = master->linear2020.width();
        height = master->linear2020.height();
        char name[32];
        std::snprintf(name, sizeof name, "%08d.exr", i);
        const fs::path exr = frames_dir / name;
        if (auto w = write_exr(exr, master->linear2020.buffer(), false); !w) return w.error();
        frames.push_back(exr);
        say("render", i + 1, count);
    }
    // Decoder completion and the source identity, before anything is encoded.
    auto receipt = (*stream)->finish();
    if (!receipt) return receipt.error();
    if (auto valid = (*stream)->validate_completion(*receipt); !valid) return valid.error();
    if (auto valid = (*prepared)->verify(); !valid) return valid.error();
    (*stream)->close();
    if (stopped()) return cancelled();

    // Encode from the staged frames (two passes: measure, then encode).
    SequenceEncodeOptions opt;
    opt.target = req.target;
    opt.fps = fps;
    opt.peak_nits = req.peak_nits;
    opt.source = Primaries::Rec2020;
    opt.verify_tags = true;
    std::vector<std::string> notes;
    opt.note = [&](const std::string& n) { notes.push_back(n); };
    const fs::path staged_movie = staging->path() / ("movie" + target->suffix);
    auto delivered = deliver_linear_frames(frames, staged_movie, opt, 203.0, [&](const LinearDeliveryProgress& p) {
        say("encode", int(p.done), int(p.total));
    }, cancel);
    if (!delivered) return delivered.error();
    if (stopped()) return cancelled();

    // QC of the staged file: every frame, the rendered size, the tags (verified above).
    say("qc", 0, 1);
    auto probe = probe_video(delivered->file);
    if (!probe) return probe.error();
    if (probe->video_streams.empty()) return make_error(ErrorCode::IntegrityError, "The export has no video stream");
    const auto& vs = probe->video_streams.front();
    const int probed = int(probe->frames.size());
    if (vs.width != width || vs.height != height || probed != count)
        return make_error(ErrorCode::IntegrityError, "The export failed its check",
                          std::to_string(vs.width) + "x" + std::to_string(vs.height) + ", " + std::to_string(probed) +
                              " frames; expected " + std::to_string(width) + "x" + std::to_string(height) + ", " +
                              std::to_string(count));
    auto sha_after = sha256_file(req.source);
    if (!sha_after) return sha_after.error();
    if (*sha_after != *sha_before) return make_error(ErrorCode::IntegrityError, "The source changed during the export");

    nlohmann::json side{
        {"schema", "rudra-headroom-export-v1"},
        {"policy", std::string(headroom_policy_id(HeadroomPolicy::Bt1886V1))},
        {"reconstruction", "analytic headroom (bt1886-headroom-v1), no model"},
        {"source", req.source.string()},
        {"source_sha256", *sha_before},
        {"source_cache_id", (*prepared)->cache_id()},
        {"decoder", decoder.string()},
        {"target", req.target},
        {"fps", fps},
        {"peak_nits", req.peak_nits},
        {"frames", count},
        {"width", width},
        {"height", height},
        {"maxcll", delivered->metadata.maxcll},
        {"maxfall", delivered->metadata.maxfall},
        {"grade", nlohmann::json::parse(master_request_json(q), nullptr, false)},
        {"app_version", req.app_version},
        {"build_commit", RUDRA_BUILD_COMMIT},
        {"qc", {{"frames", probed}, {"width", vs.width}, {"height", vs.height}, {"tags_verified", true},
                {"decoder_completion_verified", true}, {"source_unchanged", true}}},
        {"notes", notes},
        {"limits", "no audio, no temporal smoothing, no dynamic metadata; not a calibrated-review result"}};
    const fs::path staged_sidecar = staging->path() / "movie.json";
    {
        std::ofstream out(staged_sidecar, std::ios::binary);
        out << side.dump(2) << '\n';
        if (!out) return make_error(ErrorCode::IoError, "Could not write the export sidecar");
    }
    if (stopped()) return cancelled();
    say("publish", 0, 1);
    if (auto pub = publish_video(delivered->file, staged_sidecar, final_file, final_sidecar); !pub) {
        // A half-published pair is not left behind: the movie goes if its sidecar could not.
        // Only when our movie was renamed and our sidecar was not (never a file that appeared meanwhile).
        if (pub.error().message == "Could not publish the sidecar") fs::remove(final_file, ec);
        return pub.error();
    }
    say("publish", 1, 1);
    HeadroomExportResult r;
    r.file = final_file;
    r.sidecar = final_sidecar;
    r.frames = count;
    r.width = width;
    r.height = height;
    r.metadata = delivered->metadata;
    r.source_sha256 = *sha_before;
    return r;
}

}  // namespace rudra
