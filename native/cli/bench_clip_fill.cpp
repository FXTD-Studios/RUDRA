#include "bench_clip_fill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "rudra/core/baseline.hpp"
#include "rudra/core/clip_fill.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/media/still.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: rudra-native bench-clip-fill <bench> [--corpus-ev EV] [--tree NAME] [--glow K] [--force]\n"
                 "  For every <bench>/ref/<scene>/<asset>.exr, reads <bench>/sdr/<scene>/<asset>.png (an export\n"
                 "  made with --write-sdr) and writes <bench>/<tree>/<scene>/<asset>.exr: the analytic baseline at\n"
                 "  corpus EV (0, the shared benches' convention) with the clip fill, scene-linear, diffuse white\n"
                 "  1.0, half -- the units of the export's baseline/ tree. Then:\n"
                 "    python training/score_regions.py <bench> --test-dir <tree> baseline\n");
}

}  // namespace

int cmd_bench_clip_fill(const std::vector<std::string>& args) {
    if (args.empty()) {
        usage();
        return 64;
    }
    const fs::path bench = args[0];
    float corpus_ev = 0.0f;
    std::string tree = "clipfill";
    bool force = false;
    ClipFillOptions options;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--force") force = true;
        else if (i + 1 < args.size() && args[i] == "--corpus-ev") corpus_ev = float(std::atof(args[++i].c_str()));
        else if (i + 1 < args.size() && args[i] == "--tree") tree = args[++i];
        else if (i + 1 < args.size() && args[i] == "--glow") options.glow_scale = std::atof(args[++i].c_str());
        else {
            usage();
            return 64;
        }
    }
    std::error_code ec;
    if (!fs::is_directory(bench / "ref", ec) || !fs::is_directory(bench / "sdr", ec)) {
        std::fprintf(stderr, "%s needs ref/ and sdr/ trees (export_bench_pairs.py --write-sdr)\n", bench.string().c_str());
        return 66;
    }
    std::vector<fs::path> refs;
    for (const auto& e : fs::recursive_directory_iterator(bench / "ref", ec))
        if (e.is_regular_file() && e.path().extension() == ".exr") refs.push_back(e.path());
    std::sort(refs.begin(), refs.end());
    const auto t0 = std::chrono::steady_clock::now();
    int written = 0, skipped = 0;
    for (std::size_t k = 0; k < refs.size(); ++k) {
        const std::string scene = refs[k].parent_path().filename().string(), asset = refs[k].stem().string();
        const fs::path out = bench / tree / scene / (asset + ".exr");
        if (!force && fs::exists(out, ec)) {
            ++skipped;
            continue;
        }
        auto d = decode_sdr_file(bench / "sdr" / scene / (asset + ".png"));
        if (!d) {
            std::fprintf(stderr, "%s/%s: %s %s\n", scene.c_str(), asset.c_str(), d.error().message.c_str(), d.error().detail.c_str());
            return 1;
        }
        const ClipFill fill = reconstruct_clipped(d->rgb, corpus_ev, options);
        PlanarBuffer px = analytic_baseline(d->rgb, corpus_ev).buffer();
        const float to_scene = 10000.0f / 203.0f;   // network units -> diffuse white 1.0
        for (std::size_t i = 0; i < px.span().size(); ++i) {
            const float g = fill.gain_log2.empty() ? 0.0f : fill.gain_log2.span()[i];
            px.span()[i] = px.span()[i] * std::exp2(g) * to_scene;
        }
        fs::create_directories(out.parent_path(), ec);
        if (auto w = write_exr(out, px, true); !w) {
            std::fprintf(stderr, "%s: %s %s\n", out.string().c_str(), w.error().message.c_str(), w.error().detail.c_str());
            return 1;
        }
        ++written;
        if (written % 25 == 0)
            std::printf("%zu/%zu  %.0f s\n", k + 1, refs.size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("%s: %d written, %d already there, tree %s\n", bench.string().c_str(), written, skipped, tree.c_str());
    return 0;
}

}  // namespace rudra
