// rudra-native aces-v2-serve: the ACES v2 staged master (core/aces_v2.hpp) as a
// long-running engine for the app (26 Oct 2026). The view and the network load
// once; requests then come as one JSON object per line on stdin and every
// answer is one JSON object per line on stdout, so the app can show a frame
// through the model at any exposure, compare it with the plain ACES inverse,
// and master with nothing loaded twice. The masters are the ones
// `rudra-native aces-v2` writes (same stages, same EXR, same provenance).
//
//   rudra-native aces-v2-serve <package> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml]
//                              [--ocio-threads N] [--no-smoothing]
// On start:  {"event":"ready", ...}  or  {"event":"error", ...} and exit 2.
// Requests (id is echoed):
//   {"id":1,"op":"preview","source":"<still or video>","frame":0,"guard":true}
//       runs the frame, keeps it, answers its size and statistics. Options for the
//       app's main viewer: "keep":false leaves the kept frame alone; "model_out" and
//       "baseline_out" write the master and the plain ACES inverse as raw float32
//       ACEScg, planar R, G, B (width x height each), native byte order
//       "trims":{"strength":1,"saturation":1,"peak_nits":0} trims the master (core/aces_v2_trims.hpp);
//       "display":"sdr100|hdr1000" writes the outputs through that ACES 2.0 output, in display
//       light (aces_v2_monitor.hpp); "max_side":960 runs a proxy; "codes_out" writes the SDR's
//       largest code per pixel (one float plane)
//   {"id":2,"op":"view","ev":-2,"mode":"model|plain|source|mask|split|fill","max_side":1600,"out":"<file.ppm>"}
//       the kept frame through the ACES 2.0 SDR view at that exposure, 8-bit PPM; "fill" shows the
//       AI fill's regions (rudra/genfill/masks.py): source codes max(R,G,B) >= "tau_high" (250) in
//       orange, <= "tau_low" (4) in cyan, over the dimmed model view
//   {"id":3,"op":"master","sources":["<still or video>",...],"out":"<new dir>","guard":true}
//       a video is mastered frame by frame as <stem>_<NNNNNN>.exr; "log":"logc4" and
//       "log_formats":"tiff16,dpx10,prores4444xq" add camera-log copies (aces_v2_log.hpp).
//       Delivery options (aces_v2_deliver.hpp): "exr":"float|half|none",
//       "exr_gamut":"acescg|ap0", "frames":[n,...] (of one video), "fps":24,
//       "name":"shot" with "start":1001 (shot.001001.exr ...), "log":"pq1000"
//       (the ACES 2.0 HDR 1000-nit Rec.2100 PQ output), "log_formats":"mov16"
//       (uncompressed 16-bit RGB QuickTime).
//   {"id":4,"op":"probe","source":"<still or video>"}  -> frames, width, height, fps
//       {"id":3,"progress":{...}} per frame, then {"id":3,"ok":true,"done":true,...}
//   {"op":"cancel"}  stops a master after the frame in hand
//   {"op":"cancel","target":id}  stops that master (running or queued) only
//   {"op":"quit"}

#include "aces_v2_serve.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include <nlohmann/json.hpp>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/aces_v2_manifest.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/media/still.hpp"
#include "aces_v2_inputs.hpp"
#include "aces_v2_deliver.hpp"
#include "aces_v2_log.hpp"
#include "aces_v2_monitor.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace rudra;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

std::mutex g_out;
void send(const json& j) {
    std::lock_guard l(g_out);
    std::fputs((j.dump() + "\n").c_str(), stdout);
    std::fflush(stdout);
}
json err(const json& id, const std::string& message, const std::string& detail = {}) {
    return {{"id", id}, {"ok", false}, {"error", message}, {"detail", detail}};
}
json err(const json& id, const Error& e) { return err(id, e.message, e.detail); }

// The kept frame for "view".
struct Kept {
    fs::path source;
    PlanarBuffer codes, baseline, master, mask;
};

std::size_t count_positive(const PlanarBuffer& m) {
    std::size_t n = 0;
    for (float v : m.span()) n += v > 0.0f;
    return n;
}

double peak_luma(const PlanarBuffer& rgb) {
    double peak = 0;
    for (std::size_t i = 0; i < rgb.plane_size(); ++i)
        peak = std::max(peak, aces_v2::kLumaAp1[0] * rgb.plane(0)[i] + aces_v2::kLumaAp1[1] * rgb.plane(1)[i] +
                                  aces_v2::kLumaAp1[2] * rgb.plane(2)[i]);
    return peak;
}

// The display image of `mode` at `ev`, box-averaged down to max_side, as 8-bit RGB.
std::vector<std::uint8_t> render_view(const Kept& k, const aces_v2::AcesView& view, const std::string& mode, double ev,
                                      int max_side, int& out_w, int& out_h, int tau_high = 250, int tau_low = 4) {
    const int h = k.codes.height(), w = k.codes.width();
    const float gain = static_cast<float>(std::exp2(ev));
    auto shown = [&](const PlanarBuffer& lin) {
        PlanarBuffer s = lin;
        for (float& v : s.span()) v *= gain;
        return aces_v2::view_forward(view, s);
    };
    PlanarBuffer disp;
    if (mode == "source") disp = k.codes;
    else if (mode == "plain") disp = shown(k.baseline);
    else {
        disp = shown(k.master);
        if (mode == "mask") {
            for (std::size_t i = 0; i < disp.plane_size(); ++i) {
                const float m = k.mask.plane(0)[i] > 0.0f ? std::min(1.0f, 0.35f + 0.65f * k.mask.plane(0)[i]) : 0.0f;
                const float magenta[3] = {1.0f, 0.0f, 1.0f};
                for (int c = 0; c < 3; ++c) disp.plane(c)[i] = disp.plane(c)[i] * 0.35f * (1 - m) + magenta[c] * m;
            }
        } else if (mode == "fill") {
            const float* c[3] = {k.codes.plane(0), k.codes.plane(1), k.codes.plane(2)};
            const float hi_t = (float(tau_high) - 0.5f) / 255.0f, lo_t = (float(tau_low) + 0.5f) / 255.0f;
            for (std::size_t i = 0; i < disp.plane_size(); ++i) {
                const float mx = std::max({c[0][i], c[1][i], c[2][i]});
                const bool hi = tau_high <= 255 && mx >= hi_t, lo = !hi && tau_low >= 0 && mx <= lo_t;
                const float orange[3] = {1.0f, 0.55f, 0.0f}, cyan[3] = {0.0f, 0.85f, 1.0f};
                for (int ch = 0; ch < 3; ++ch)
                    disp.plane(ch)[i] = hi ? orange[ch] : lo ? cyan[ch] : disp.plane(ch)[i] * 0.5f;
            }
        } else if (mode == "split") {
            const PlanarBuffer plain = shown(k.baseline);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w / 2; ++x)
                    for (int c = 0; c < 3; ++c) disp.at(c, y, x) = plain.at(c, y, x);
            for (int y = 0; y < h; ++y)
                for (int c = 0; c < 3; ++c) disp.at(c, y, w / 2) = 1.0f;
        }
    }
    const int f = std::max(1, (std::max(h, w) + std::max(1, max_side) - 1) / std::max(1, max_side));
    out_w = w / f;
    out_h = h / f;
    std::vector<std::uint8_t> rgb(std::size_t(out_w) * out_h * 3);
    for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x)
            for (int c = 0; c < 3; ++c) {
                double s = 0;
                for (int dy = 0; dy < f; ++dy)
                    for (int dx = 0; dx < f; ++dx) s += disp.at(c, y * f + dy, x * f + dx);
                const double v = std::clamp(s / (f * f), 0.0, 1.0);
                rgb[(std::size_t(y) * out_w + x) * 3 + c] = static_cast<std::uint8_t>(std::lround(v * 255.0));
            }
    return rgb;
}

bool write_planes(const fs::path& p, const PlanarBuffer& b) {
    std::ofstream o(p, std::ios::binary);
    for (int c = 0; c < 3; ++c)
        o.write(reinterpret_cast<const char*>(b.plane(c)), std::streamsize(b.plane_size() * sizeof(float)));
    return bool(o);
}

bool write_ppm(const fs::path& p, const std::vector<std::uint8_t>& rgb, int w, int h) {
    std::ofstream o(p, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    o.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
    return bool(o);
}

// Requests from stdin on their own thread, so "cancel" reaches a running master.
struct Inbox {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> q;
    bool closed = false;
    std::atomic<bool> cancel{false};   // {"op":"cancel"}: the master running now
    // {"op":"cancel","target":id}: that master only, running or still queued (28 Oct 2026).
    std::mutex cmu;
    std::set<std::string> targets;
    void target(const json& id) {
        std::lock_guard l(cmu);
        targets.insert(id.dump());
    }
    bool cancelled(const json& id) {
        std::lock_guard l(cmu);
        return targets.count(id.dump()) > 0;
    }
    void forget(const json& id) {
        std::lock_guard l(cmu);
        targets.erase(id.dump());
    }
};

}  // namespace

int cmd_aces_v2_serve(const std::vector<std::string>& args) {
    if (args.empty() || args[0] == "--help") {
        std::fprintf(stderr, "usage: rudra-native aces-v2-serve <package> [--runtime libtorch|onnxruntime] [--device cpu|cuda|directml] [--ocio-threads N] [--no-smoothing]\n");
        return 64;
    }
    const auto t0 = Clock::now();
    const fs::path pkg = args[0];
    std::string device = "cpu", runtime = "libtorch";
    int threads = 8;
    bool smoothing = true;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--device" && i + 1 < args.size()) device = args[++i];
        else if (args[i] == "--runtime" && i + 1 < args.size()) runtime = args[++i];
        else if (args[i] == "--ocio-threads" && i + 1 < args.size()) threads = std::atoi(args[++i].c_str());
        else if (args[i] == "--no-smoothing") smoothing = false;
        else {
            send({{"event", "error"}, {"error", "Unknown option."}, {"detail", args[i]}});
            return 2;
        }
    }
    auto fail_start = [](const Error& e) {
        send({{"event", "error"}, {"error", e.message}, {"detail", e.detail}});
        return 2;
    };
    auto m = aces_v2::read_manifest(pkg);
    if (!m) return fail_start(m.error());
    if (auto v = aces_v2::verify_files(*m); !v) return fail_start(v.error());
    if (smoothing && m->smoothing != "hybrid64")
        return fail_start(make_error(ErrorCode::ContractMismatch, "This package declares no clipped smoothing."));
    auto view = aces_v2::make_ocio_view(threads);
    if (!view) return fail_start(view.error());
    Result<aces_v2::G1Fn> g1 = make_error(ErrorCode::Unsupported, "Unknown runtime (libtorch or onnxruntime).", runtime);
    if (runtime == "libtorch") {
#ifdef RUDRA_HAVE_LIBTORCH_ACES
        g1 = aces_v2::load_g1_libtorch(m->g1, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no LibTorch.");
#endif
    } else if (runtime == "onnxruntime") {
        if (m->g1_onnx.empty()) return fail_start(make_error(ErrorCode::Unsupported, "This package has no ONNX model."));
#ifdef RUDRA_HAVE_ORT_ACES
        g1 = aces_v2::load_g1_onnxruntime(m->g1_onnx, device);
#else
        g1 = make_error(ErrorCode::Unsupported, "This build has no ONNX Runtime.");
#endif
    }
    if (!g1) return fail_start(g1.error());
    // Warm-up on a small grey frame: the first CUDA call is the slow one.
    {
        PlanarBuffer warm(3, 64, 64, 0.5f);
        (void)aces_v2::run_frame(warm, **view, *g1, smoothing, true, m->forbid_new_negative_y);
    }
    send({{"event", "ready"}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256}, {"checkpoint_sha256", m->checkpoint_sha256},
          {"runtime", runtime}, {"device", device}, {"view", (*view)->identity()}, {"luminance_guard_default", m->forbid_new_negative_y},
          {"release_authorized", m->release_authorized}, {"load_ms", ms_since(t0)}});

    Inbox in;
    std::thread reader([&in] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            json j = json::parse(line, nullptr, false);
            if (j.is_discarded() || !j.is_object()) {
                send(err(nullptr, "A request must be one JSON object per line.", line.substr(0, 200)));
                continue;
            }
            const std::string op = j.value("op", std::string());
            if (op == "cancel") {
                if (j.contains("target")) in.target(j["target"]);
                else in.cancel = true;
                continue;
            }
            std::lock_guard l(in.mu);
            in.q.push_back(std::move(j));
            in.cv.notify_one();
        }
        std::lock_guard l(in.mu);
        in.closed = true;
        in.cv.notify_one();
    });

    std::optional<Kept> kept;
    std::unique_ptr<aces_v2::AcesView> hdr_view;   // the ACES 2.0 HDR output (pq1000 copies), made when first asked for
    std::map<std::string, std::unique_ptr<aces_v2::AcesView>> target_views;   // pq2000, pq4000, hlg1000, sdr709
    for (;;) {
        json req;
        {
            std::unique_lock l(in.mu);
            in.cv.wait(l, [&] { return in.closed || !in.q.empty(); });
            if (in.q.empty()) break;
            req = std::move(in.q.front());
            in.q.pop_front();
        }
        const json id = req.value("id", json());
        const std::string op = req.value("op", std::string());
        try {
            if (op == "quit") break;
            const bool guard = req.value("guard", m->forbid_new_negative_y);
            if (op == "preview") {
                const fs::path src = req.at("source").get<std::string>();
                const int frame_index = req.value("frame", 0);
                const auto t = Clock::now();
                auto still = aces_read_one(src, frame_index, fs::temp_directory_path() / "rudra_aces_preview_decode.log");
                if (!still) {
                    send(err(id, still.error()));
                    continue;
                }
                const std::string source_display = req.value("source_display", std::string("bt1886"));
                if (!aces_valid_source_display(source_display)) {
                    send(err(id, "source_display is bt1886, srgb or gamma22.", source_display));
                    continue;
                }
                aces_adapt_source(still->codes, source_display);   // the same display light, as BT.1886 codes
                // A playback proxy: the codes area-averaged down before the model (the app asks while playing).
                const PlanarBuffer codes = aces_area_downscale(still->codes, req.value("max_side", 0));
                bool monitor_ok = true;
                const AcesMonitor monitor = aces_monitor_from(req.value("display", std::string()), &monitor_ok);
                if (!monitor_ok) {
                    send(err(id, "display is scene, sdr100 or hdr1000."));
                    continue;
                }
                if (monitor == AcesMonitor::Hdr1000 && !hdr_view) {   // the ACES 2.0 HDR output, made once
                    auto hv = aces_v2::make_ocio_display_view(aces_v2::kOcioHdrDisplay, aces_v2::kOcioHdrView, threads);
                    if (!hv) {
                        send(err(id, hv.error()));
                        continue;
                    }
                    hdr_view = std::move(*hv);
                }
                auto r = aces_v2::run_frame(codes, **view, *g1, smoothing, true, guard);
                if (!r) {
                    send(err(id, r.error()));
                    continue;
                }
                Kept k{src, codes, aces_v2::view_inverse(**view, codes), std::move(r->acescg), std::move(r->mask)};
                // The colorist's trims (28 Oct 2026): the viewer shows what a render writes.
                auto trims = aces_trims_from_json(req.value("trims", json()));
                if (!trims) {
                    send(err(id, trims.error()));
                    continue;
                }
                if (!trims->identity()) k.master = aces_v2::apply_trims(k.master, k.baseline, k.mask, *trims);
                bool wrote = true;
                // The monitoring transform (scene, or an ACES 2.0 output decoded to display light).
                const aces_v2::AcesView& mview = monitor == AcesMonitor::Hdr1000 ? *hdr_view : **view;
                if (req.contains("model_out"))
                    wrote = wrote && write_planes(req["model_out"].get<std::string>(), aces_monitor_planes(k.master, mview, monitor));
                if (req.contains("baseline_out"))
                    wrote = wrote && write_planes(req["baseline_out"].get<std::string>(), aces_monitor_planes(k.baseline, mview, monitor));
                if (req.contains("codes_out")) {   // the SDR's largest code per pixel: the Invented layer
                    const PlanarBuffer cp = aces_code_plane(k.codes);
                    std::ofstream o(req["codes_out"].get<std::string>(), std::ios::binary);
                    o.write(reinterpret_cast<const char*>(cp.plane(0)), std::streamsize(cp.plane_size() * sizeof(float)));
                    wrote = wrote && bool(o);
                }
                if (!wrote) {
                    send(err(id, "The frame could not be written for the viewer."));
                    continue;
                }
                const double ms = ms_since(t);
                const std::size_t recon = count_positive(k.mask);
                send({{"id", id}, {"ok", true}, {"source", src.string()}, {"frame", frame_index}, {"label", still->source},
                      {"width", codes.width()}, {"height", codes.height()},
                      {"bits", still->bits}, {"ms", ms}, {"reconstructed_fraction", double(recon) / double(k.mask.plane_size())},
                      {"peak", peak_luma(k.master)}, {"peak_plain", peak_luma(k.baseline)},
                      {"fully_clipped_pixels", r->smoothing.full_pixels}, {"remaining_display_errors", r->constraint.remaining_display_errors},
                      {"new_negative_luminance", r->constraint.new_negative_luminance}, {"luminance_guard", guard}});
                if (req.value("keep", true)) kept = std::move(k);
            } else if (op == "probe") {
                const fs::path src = req.at("source").get<std::string>();
                if (aces_is_video(src)) {
                    auto vs = aces_open_video(src);
                    if (!vs) {
                        send(err(id, vs.error()));
                        continue;
                    }
                    send({{"id", id}, {"ok", true}, {"video", true}, {"frames", vs->clock.frames}, {"width", vs->stream.width},
                          {"height", vs->stream.height}, {"fps", vs->clock.fps}, {"bits", aces_video_bits(vs->stream.pix_fmt)}});
                } else {
                    send({{"id", id}, {"ok", true}, {"video", false}, {"frames", aces_is_still(src) ? 1 : 0}});
                }
            } else if (op == "view") {
                if (!kept) {
                    send(err(id, "Preview a frame first."));
                    continue;
                }
                const auto t = Clock::now();
                int w = 0, h = 0;
                const auto rgb = render_view(*kept, **view, req.value("mode", std::string("model")), req.value("ev", 0.0),
                                             req.value("max_side", 1600), w, h, req.value("tau_high", 250), req.value("tau_low", 4));
                const fs::path out = req.at("out").get<std::string>();
                if (!write_ppm(out, rgb, w, h)) {
                    send(err(id, "The preview image could not be written.", out.string()));
                    continue;
                }
                send({{"id", id}, {"ok", true}, {"path", out.string()}, {"width", w}, {"height", h}, {"ms", ms_since(t)}});
            } else if (op == "master") {
                in.cancel = false;
                if (in.cancelled(id)) {   // stopped while it waited: nothing made, nothing written
                    in.forget(id);
                    send({{"id", id}, {"ok", true}, {"done", true}, {"cancelled", true}, {"frames", 0}});
                    continue;
                }
                std::vector<fs::path> sources;
                for (const auto& s : req.at("sources")) sources.emplace_back(s.get<std::string>());
                const fs::path out = req.at("out").get<std::string>();
                if (sources.empty()) {
                    send(err(id, "No stills or videos to master."));
                    continue;
                }
                auto opts = aces_deliver_from_json(req);
                if (!opts) {
                    send(err(id, opts.error()));
                    continue;
                }
                auto frames = AcesFrameFeed::open(sources, opts->frames, fs::temp_directory_path() / "rudra_aces_serve_decode.log", opts->range_first, opts->range_last);
                if (!frames) {
                    send(err(id, frames.error()));
                    continue;
                }
                // A new folder, or (Nuke's Write) the folder given, with nothing in it replaced.
                const std::string movie_stem = sources.size() == 1 ? sources.front().stem().string() : out.filename().string();
                const std::string movie_file = opts->copies.any() && opts->copies.movie() ? aces_movie_file(*opts, movie_stem) : "";
                if (auto ready = aces_prepare_out(*opts, out, (*frames)->total(), movie_file); !ready) {
                    send(err(id, ready.error()));
                    continue;
                }
                // The ACES 2.0 HDR output (PQ copies, and the HDR10 light level of every master):
                // built the first time it is asked for, then kept.
                if (!hdr_view) {
                    auto hv = aces_v2::make_ocio_display_view(aces_v2::kOcioHdrDisplay, aces_v2::kOcioHdrView, threads);
                    if (!hv) {
                        send(err(id, hv.error()));
                        continue;
                    }
                    hdr_view = std::move(*hv);
                }
                json run = {{"contract", aces_v2::kContractId}, {"package", pkg.string()}, {"g1_sha256", m->g1_sha256},
                            {"checkpoint_sha256", m->checkpoint_sha256}, {"model_contract", m->model_contract_id},
                            {"view", (*view)->identity()}, {"device", device}, {"runtime", runtime},
                            {"g1_file", runtime == "onnxruntime" ? m->g1_onnx.filename().string() : m->g1.filename().string()},
                            {"clipped_smoothing", smoothing ? m->smoothing : "none"},
                            {"appearance_constraint", guard ? "1 code, include fully clipped, no new negative luminance"
                                                            : "1 code, include fully clipped"},
                            {"luminance_guard", guard}, {"luminance_guard_package_default", m->forbid_new_negative_y},
                            {"master", aces_exr_label(*opts)}, {"trims", aces_trims_json(opts->trims)},
                            {"source_display", aces_source_display_label(opts->source_display)},
                            {"mask", "rudra.reconstructed = highlight reconstruction weight; 0 = from source (incl. analytic shadows)"},
                            {"not", "not camera RAW; reconstructed highlights are estimates"},
                            {"release_authorized", m->release_authorized}, {"engine", "aces-v2-serve"}, {"frames", json::array()}};
                if (!opts->name.empty()) run["name"] = opts->name, run["start"] = opts->start;
                if (req.contains("edl_event") && req["edl_event"].is_object()) run["edl_event"] = req["edl_event"];   // the conform's event
                bool all_ok = true, stopped = false;
                std::optional<Error> failure;
                std::unique_ptr<AcesLogWriter> logw;
                if (opts->copies.any()) {
                    // A display target other than PQ 1000 (2000, 4000, HLG, the SDR pass): its own view, made once.
                    const aces_v2::AcesView* copy_view = hdr_view.get();
                    if (opts->copies.pq && opts->copies.target != "pq1000") {
                        auto& tv = target_views[opts->copies.target];
                        if (!tv) {
                            auto made = aces_target_view(opts->copies.display_target(), threads);
                            if (made) tv = std::move(*made);
                            else failure = made.error();
                        }
                        copy_view = tv.get();
                    }
                    auto w = failure ? Result<std::unique_ptr<AcesLogWriter>>(*failure)
                                     : AcesLogWriter::open(opts->copies, out, out / aces_movie_file(*opts, movie_stem),
                                                           opts->fps > 0 ? opts->fps : (*frames)->fps(), copy_view);
                    if (!w) failure = w.error();
                    else logw = std::move(*w);
                }
                const std::size_t total = failure ? 0 : (*frames)->total();
                const double run_fps = opts->fps > 0 ? opts->fps : (*frames)->fps();
                double run_maxcll = 0, run_maxfall = 0, run_scene_peak = 0;
                std::vector<FrameStats> run_stats;   // the PQ output's, for Dolby Vision
                int run_w = 0, run_h = 0;
                for (std::size_t i = 0; i < total; ++i) {
                    if (in.cancel || in.cancelled(id)) {
                        stopped = true;
                        break;
                    }
                    auto f = (*frames)->next();
                    if (!f) {
                        failure = f.error();
                        break;
                    }
                    aces_adapt_source(f->codes, opts->source_display);   // an sRGB or 2.2 source, as BT.1886 codes
                    const auto t = Clock::now();
                    auto r = aces_v2::run_frame(f->codes, **view, *g1, smoothing, true, guard);
                    if (!r) {
                        failure = r.error();
                        break;
                    }
                    const double ms = ms_since(t);
                    // The colorist's trims, then the frame's timecode and reel (28 Oct 2026).
                    const PlanarBuffer master = aces_trimmed(*opts, r->acescg, r->mask, **view, f->codes);
                    const AcesFrameStamp stamp = aces_frame_stamp(*opts, i, *f, run_fps);
                    // HDR10 numbers on the PQ deliverable (its own peak), else on the PQ 1000 output.
                    const aces_v2::AcesView* light_view =
                        opts->copies.hdr10() && opts->copies.target != "pq1000" ? target_views[opts->copies.target].get() : hdr_view.get();
                    const AcesLightLevel light = aces_light_level(master, light_view, int(i));
                    if (opts->copies.hdr10()) run_stats.push_back(light.stats);
                    run_w = master.width(), run_h = master.height();
                    run_maxcll = std::max(run_maxcll, light.maxcll);
                    run_maxfall = std::max(run_maxfall, light.maxfall);
                    run_scene_peak = std::max(run_scene_peak, light.scene_peak);
                    if (logw && i == 0) logw->set_movie_stamp(stamp.timecode, stamp.reel);
                    const std::string stem = aces_frame_stem(*opts, i, f->exr);
                    const fs::path exr = opts->writes_exr() ? out / (stem + ".exr") : fs::path();
                    const ExrAttributes prov = {{"contract", aces_v2::kContractId}, {"g1_sha256", m->g1_sha256},
                                                {"source", f->source}, {"view", (*view)->identity()}};
                    if (opts->writes_exr())
                        if (auto w = write_aces_master_exr(exr, master, r->mask, *opts, prov,
                                                                      exr_timecode_attributes(stamp.timecode, stamp.fps, stamp.reel)); !w) {
                            failure = w.error();
                            break;
                        }
                    if (logw)
                        if (auto lw = logw->write(master, stem + ".exr", stamp.timecode); !lw) {
                            failure = lw.error();
                            break;
                        }
                    const std::size_t recon = count_positive(r->mask);
                    const bool ok = r->constraint.remaining_display_errors == 0 && (!guard || r->constraint.new_negative_luminance == 0);
                    all_ok = all_ok && ok;
                    const json frame = {{"source", f->source_path}, {"frame", f->source}, {"exr", exr.filename().string()},
                                        {"name", stem}, {"timecode", stamp.timecode}, {"reel", stamp.reel}, {"maxcll", light.maxcll}, {"maxfall", light.maxfall}, {"scene_peak_nits", light.scene_peak}, {"scene_average_nits", light.scene_average}, {"bits", f->bits}, {"ms", ms},
                                        {"reconstructed_fraction", double(recon) / double(r->mask.plane_size())},
                                        {"fully_clipped_pixels", r->smoothing.full_pixels},
                                        {"smoothed_components", r->smoothing.components},
                                        {"constraint_support", r->constraint.support_pixels},
                                        {"constraint_changed", r->constraint.changed_pixels},
                                        {"constraint_baseline_fallback", r->constraint.baseline_fallback_pixels},
                                        {"remaining_display_errors", r->constraint.remaining_display_errors},
                                        {"new_negative_luminance", r->constraint.new_negative_luminance}};
                    run["frames"].push_back(frame);
                    send({{"id", id}, {"progress", {{"index", i + 1}, {"total", total}, {"source", f->source},
                                                    {"exr", opts->writes_exr() ? exr.filename().string() : stem}, {"ms", ms},
                                                    {"remaining_display_errors", r->constraint.remaining_display_errors}}}});
                }
                if (logw) {
                    if (auto fin = logw->finish(run_maxcll, run_maxfall); !fin && !failure) failure = fin.error();
                    run["log_delivery"] = logw->provenance();
                }
                if (opts->copies.any() && opts->copies.hdr10() && !failure) {   // the HDR10 numbers beside the PQ files
                    run["hdr10"] = aces_hdr10_json(run_maxcll, run_maxfall, opts->copies.display_target().peak_nits);
                    if (auto hw = aces_write_whole(aces_hdr10_file(*opts, out), run["hdr10"].dump(1)); !hw) failure = hw.error();
                    else run["hdr10_file"] = aces_hdr10_file(*opts, out).filename().string();
                    // Dolby Vision CM v4.0 of the same PQ output: the XML Resolve imports, and dovi_tool's config.
                    if (!failure && !run_stats.empty()) {
                        auto dv = aces_dolby_vision(run_stats, opts->copies.display_target().peak_nits,
                                                    opts->name.empty() ? out.filename().string() : opts->name, run_fps, run_w, run_h);
                        if (!dv) failure = dv.error();
                        else if (auto a = aces_write_whole(aces_dovi_xml_file(*opts, out), dv->xml); !a) failure = a.error();
                        else if (auto b = aces_write_whole(aces_dovi_generate_file(*opts, out), dv->generate_json); !b) failure = b.error();
                        else
                            run["dolby_vision"] = {{"xml", aces_dovi_xml_file(*opts, out).filename().string()},
                                                   {"generate", aces_dovi_generate_file(*opts, out).filename().string()},
                                                   {"shots", dv->shots}, {"cm", "4.0"}, {"profile", "8.1"},
                                                   {"note", "check with Dolby's tools before a licensed delivery"}};
                    }
                }
                run["maxcll"] = run_maxcll;
                run["maxfall"] = run_maxfall;
                run["scene_peak_nits"] = run_scene_peak;
                run["light_level"] = kAcesLightLevelNote;
                run["appearance_constraint_pass"] = all_ok;
                run["cancelled"] = stopped;
                if (auto pw = aces_write_whole(aces_provenance_file(*opts, out), run.dump(1)); !pw && !failure) failure = pw.error();
                in.forget(id);
                if (failure) send(err(id, *failure));
                else {
                    json done = {{"id", id}, {"ok", true}, {"done", true}, {"cancelled", stopped}, {"frames", run["frames"].size()},
                                 {"appearance_pass", all_ok}, {"provenance", aces_provenance_file(*opts, out).string()},
                                 {"maxcll", run_maxcll}, {"maxfall", run_maxfall}, {"scene_peak_nits", run_scene_peak}};
                    if (run.contains("hdr10_file")) done["hdr10"] = aces_hdr10_file(*opts, out).string();
                    if (run.contains("dolby_vision")) done["dolby_vision"] = aces_dovi_xml_file(*opts, out).string();
                    send(done);
                }
            } else {
                send(err(id, "Unknown op.", op));
            }
        } catch (const std::exception& e) {
            send(err(id, "Bad request.", e.what()));
        }
    }
    // stdin stays open in the reader until the app closes it or exits.
    if (reader.joinable()) reader.detach();
    // Leave at once: the reader may still be blocked on stdin, and runtime
    // thread pools can hold up a normal exit (the app waits for this process).
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
}
