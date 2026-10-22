// clip_fill_probe: the shipped clip fill (core/clip_fill.cpp) as a file-in /
// file-out tool, so Python can train and score against the exact code the app
// runs (PeakNet, claude/TASK_COMMERCIAL_MODEL_PLAN.md, 17 Oct 2026).
//
//   clip_fill_probe <in.f32> <out.f32> <regions.tsv> [--corpus-ev EV] [--glow K] [--overrides file]
//
// in.f32:  int32 h, int32 w, then 3*h*w float32 SDR codes, planar RGB.
// out.f32: int32 h, int32 w, then 3*h*w float32 filled values in network
//          units (nits / 10,000): baseline_value(code) * 2^gain_log2.
// regions.tsv: one line per fully clipped region.
// overrides: lines "x y nits" (x, y normalised 0..1), as ClipOverride.
//
// Build: the clip_fill_probe target (native/cli/CMakeLists.txt), or on Linux from the repo root:
//   g++ -std=c++20 -O2 -pthread -Inative/core/include -Inative/platform/include -Inative/core/src
//       native/tools/clip_fill_probe.cpp native/core/src/clip_fill.cpp native/core/src/baseline.cpp -o clip_fill_probe
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rudra/core/baseline.hpp"
#include "rudra/core/clip_fill.hpp"

using namespace rudra;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: clip_fill_probe in.f32 out.f32 regions.tsv [--corpus-ev EV] [--glow K] [--overrides f]\n");
        return 2;
    }
    float corpus_ev = 0.0f;
    ClipFillOptions opt;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--corpus-ev")) corpus_ev = std::stof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--glow")) opt.glow_scale = std::stod(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--overrides")) {
            std::ifstream f(argv[i + 1]);
            ClipOverride o;
            while (f >> o.x >> o.y >> o.nits) opt.overrides.push_back(o);
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) return 3;
    std::int32_t hw[2];
    if (std::fread(hw, 4, 2, in) != 2) return 3;
    const int h = hw[0], w = hw[1];
    std::vector<float> data(std::size_t(3) * h * w);
    if (std::fread(data.data(), 4, data.size(), in) != data.size()) return 3;
    std::fclose(in);

    SdrImage sdr(PlanarBuffer(3, h, w, std::move(data)));
    ClipFill fill = reconstruct_clipped(sdr, corpus_ev, opt);

    std::vector<float> out(std::size_t(3) * h * w);
    const PlanarBuffer& s = sdr.buffer();
    for (int c = 0; c < 3; ++c) {
        const float* p = s.plane(c);
        const float* g = fill.gain_log2.empty() ? nullptr : fill.gain_log2.plane(c);
        for (std::size_t i = 0; i < std::size_t(h) * w; ++i) {
            const float b = baseline_value(std::min(std::max(p[i], 0.0f), 1.0f), corpus_ev);
            out[std::size_t(c) * h * w + i] = g ? b * std::exp2(g[i]) : b;
        }
    }
    std::FILE* o = std::fopen(argv[2], "wb");
    if (!o) return 4;
    std::fwrite(hw, 4, 2, o);
    std::fwrite(out.data(), 4, out.size(), o);
    std::fclose(o);

    std::FILE* r = std::fopen(argv[3], "w");
    if (!r) return 4;
    std::fprintf(r, "pixels\tfraction\tradius_px\tlarge\tedge_nits\tpeak_nits\tglow_ev_per_px\toverridden\tx\ty\n");
    for (const ClipRegion& g : fill.regions)
        std::fprintf(r, "%d\t%.9g\t%.6g\t%d\t%.9g\t%.9g\t%.9g\t%d\t%d\t%d\n", g.pixels, g.fraction, g.radius_px, int(g.large),
                     g.edge_nits, g.peak_nits, g.glow_ev_per_px, int(g.overridden), g.x, g.y);
    std::fclose(r);
    return 0;
}
