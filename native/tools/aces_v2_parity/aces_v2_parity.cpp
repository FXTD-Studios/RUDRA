// Cloud parity harness: in.f32 (int32 H, int32 W, HWC float32 codes) ->
// out_<stage>.f32 (HWC float32) for composed, smoothed, final, mask; with --lock S also
// locked.f32, the lock (stage 6b, dequantized target, strength S) on the final master.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "rudra/core/aces_v2.hpp"
#include "rudra/core/aces_lock.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
using namespace rudra;
using namespace rudra::aces_v2;
static void dump(const std::string& path, const PlanarBuffer& b) {
    std::ofstream f(path, std::ios::binary);
    int32_t hw[3] = {b.height(), b.width(), b.channels()};
    f.write((const char*)hw, sizeof hw);
    std::vector<float> v(b.plane_size() * b.channels());
    for (int c = 0; c < b.channels(); ++c) for (size_t i = 0; i < b.plane_size(); ++i) v[i * b.channels() + c] = b.plane(c)[i];
    f.write((const char*)v.data(), v.size() * 4);
}
int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "parity in.f32 g1.ts outdir [--forbid-new-negative-y] [--lock S]\n"; return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    int32_t h, w; f.read((char*)&h, 4); f.read((char*)&w, 4);
    std::vector<float> packed((size_t)h * w * 3); f.read((char*)packed.data(), packed.size() * 4);
    PlanarBuffer src = from_packed(packed, h, w);
    auto view = make_ocio_view(1);
    if (!view) { std::cerr << view.error().message << " " << view.error().detail << "\n"; return 1; }
    auto g1 = load_g1_libtorch(argv[2]);
    if (!g1) { std::cerr << g1.error().detail << "\n"; return 1; }
    std::string out = argv[3];
    // stage by stage (mirrors run_frame) so each can be compared
    PlanarBuffer baseline = view_inverse(**view, src);
    auto net = (*g1)(src, baseline);
    if (!net) { std::cerr << net.error().detail << "\n"; return 1; }
    PlanarBuffer restored = view_inverse(**view, net->restored_source);
    auto comp = compose_v2(src, baseline, restored, *net);
    SmoothingStats ss; ConstraintStats cs;
    auto sm = smooth_clipped_hybrid(comp.acescg, src, 64.0, &ss);
    bool guard = false;
    LockSettings lock;   // off unless --lock S: the registered dequantized lock at strength S
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--forbid-new-negative-y") guard = true;
        else if (a == "--lock" && i + 1 < argc) { lock = LockSettings::registered(); lock.strength = std::atof(argv[++i]); }
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    auto fin = constrain_colour_v2(src, sm, baseline, net->weight, **view, true, &cs, guard);
    LockStats ls;
    PlanarBuffer locked = lock.on ? lock_observed(src, fin, net->weight, **view, lock, &ls) : fin;
    auto rf = run_frame(src, **view, *g1, true, true, guard, {}, {}, lock);
    double maxdiff = 0; for (size_t i = 0; i < locked.span().size(); ++i) maxdiff = std::max(maxdiff, (double)std::abs(locked.span()[i] - rf->acescg.span()[i]));
    if (lock.on) dump(out + "/locked.f32", locked);
    dump(out + "/composed.f32", comp.acescg); dump(out + "/smoothed.f32", sm); dump(out + "/final.f32", fin); dump(out + "/mask.f32", comp.mask);
    dump(out + "/weight_g1.f32", net->weight);
    // core reconstruction_weight vs G1 weight
    auto wc = reconstruction_weight(src); double wd = 0;
    for (size_t i = 0; i < wc.span().size(); ++i) wd = std::max(wd, (double)std::abs(wc.span()[i] - net->weight.span()[i]));
    std::printf("{\"view\":\"%s\",\"full\":%zu,\"interior\":%zu,\"components\":%zu,\"support\":%zu,\"changed\":%zu,\"fallback\":%zu,\"remaining\":%zu,\"new_negative_y\":%zu,\"run_frame_vs_staged\":%g,\"core_weight_vs_g1\":%g,\"lock_support\":%zu,\"lock_changed\":%zu,\"lock_kept\":%zu,\"lock_raised\":%zu,\"lock_ms\":%.1f}\n",
                (*view)->identity().c_str(), ss.full_pixels, ss.interior_pixels, ss.components, cs.support_pixels, cs.changed_pixels,
                cs.baseline_fallback_pixels, cs.remaining_display_errors, cs.new_negative_luminance, maxdiff, wd,
                ls.support_pixels, ls.changed_pixels, ls.kept_pixels, ls.strength_raised_pixels, ls.ms_dequant + ls.ms_solve);
}
