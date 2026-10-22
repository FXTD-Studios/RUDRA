// Bounded diagnostic, not installed. Compares the session bridge with the
// existing direct CPU renderer, using the same protected source and pixels.
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include "rudra/engine/headroom_presentation.hpp"
#include "rudra/platform/hash.hpp"

using namespace rudra;
using Clock = std::chrono::steady_clock;
namespace {
template<class T> T checked(Result<T> r) {
    if (!r) throw std::runtime_error(r.error().message + ": " + r.error().detail);
    return std::move(r).value();
}
void checked(Result<void> r) { if (!r) throw std::runtime_error(r.error().message); }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
std::uint64_t hash(const Image<space::SceneLinear2020>& image) {
    return xxh64(std::as_bytes(image.buffer().span()));
}
}
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "headroom-presentation-bench source.mkv report.json\n"; return 2; }
    nlohmann::json report{{"schema", "headroom-presentation-benchmark-v1"}, {"ok", false},
                          {"runs", nlohmann::json::array()}};
    try {
        if (std::filesystem::exists(argv[2])) throw std::runtime_error("Choose a fresh report path");
        const auto before = checked(sha256_file(argv[1]));
        const auto preparing = Clock::now();
        auto prepared = checked(PreparedHeadroomSource::prepare(argv[1], HeadroomPolicy::Bt1886V1,
                                checked(require_executable("ffmpeg"))));
        report["prepare_ms"] = ms(preparing);
        const auto& video = prepared->video();
        const int count = std::min(8, video.clock.frames);
        report["source"] = argv[1]; report["source_sha256"] = before;
        report["cache_id"] = prepared->cache_id(); report["frames_per_run"] = count;
        report["width"] = video.stream.width; report["height"] = video.stream.height;
        Session session(Session::SourceSupport::HeadroomV1);
        checked(session.set_headroom_policy(HeadroomPolicy::Bt1886V1));
        session.anchor = session.carry_chroma = session.grade.clip_fill = false;
        session.nudge_exposure(1);
        MasterRequest q; q.analytic = true; q.source_curve = "bt1886";
        q.source_space = video.contract.primaries; q.exposure_ev = 1;
        q.anchor = q.carry_chroma = q.settle_grain = q.clip_fill = false;
        nlohmann::json reference;
        // Reverse the second pair to limit ordering bias; still a small sample.
        for (bool bridge : {false, true, true, false}) {
            nlohmann::json hashes = nlohmann::json::array();
            nlohmann::json run{{"mode", bridge ? "session_preview" : "direct_cpu"}};
            const auto start = Clock::now();
            if (bridge) {
                auto preview = checked(HeadroomPresentation::create());
                checked(preview->set_session(session, false, false, false));
                checked(preview->open(prepared, {0, count}));
                std::optional<HeadroomPresentation::Frame> last;
                for (int i = 0; i < count; ++i) {
                    auto frame = checked(preview->render_next(std::chrono::seconds(30)));
                    if (frame.index() != i) throw std::runtime_error("Wrong presented frame");
                    if (!i) run["first_frame_ms"] = ms(start);
                    checked(preview->present(frame, [&](const auto& shown) {
                        hashes.push_back({hash(shown.master().linear2020), hash(shown.baseline().linear2020)});
                    }));
                    last.emplace(std::move(frame));
                }
                const auto deadline = Clock::now() + std::chrono::seconds(5);
                while (!preview->stats().complete && Clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (!preview->stats().complete) throw std::runtime_error("Preview decode did not complete");
                checked(preview->present(*last, [](const auto&) {}));
                run["decoder_starts"] = preview->stats().decoder_starts;
                run["source_pixel_peak_bytes"] = preview->stats().pixels.peak;
                run["source_pixel_limit_bytes"] = preview->stats().pixels.limit;
            } else {
                auto stream = checked(HeadroomStream::open(prepared));
                checked(stream->start({0, count}));
                for (int i = 0; i < count; ++i) {
                    auto frame = checked(stream->next());
                    auto master = checked(render_headroom_frame(frame.pixels(), q, HeadroomPolicy::Bt1886V1));
                    auto baseline = checked(reconstruct_headroom_baseline(frame.pixels(), q, HeadroomPolicy::Bt1886V1));
                    checked(stream->validate_preview(frame));
                    if (!i) run["first_frame_ms"] = ms(start);
                    hashes.push_back({hash(master.linear2020), hash(baseline.linear2020)});
                }
                checked(stream->validate_completion(checked(stream->finish())));
                run["decoder_starts"] = stream->stats().decoder_starts;
            }
            run["feed_render_hash_ms"] = ms(start); run["hashes"] = hashes;
            if (reference.is_null()) reference = hashes;
            if (hashes != reference) throw std::runtime_error("Session output differs from direct CPU baseline");
            run["pixels_match"] = true;
            report["runs"].push_back(run);
        }
        checked(prepared->verify());
        if (checked(sha256_file(argv[1])) != before) throw std::runtime_error("Source changed");
        report["ok"] = true;
    } catch (const std::exception& e) { report["error"] = e.what(); }
    // Do not replace an existing report, including on an argument error.
    if (std::filesystem::exists(argv[2])) { std::cerr << report.dump(2); return 1; }
    std::ofstream out(argv[2]); out << report.dump(2) << '\n';
    if (!out) return 1;
    return report["ok"].get<bool>() ? 0 : 1;
}
