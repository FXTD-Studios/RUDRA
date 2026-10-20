#include "deliver.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "rudra/core/measure.hpp"
#include "rudra/deliver/sequence_encode.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {
namespace fs = std::filesystem;

// rudra/delivery/cli.py _cmd_deliver: the frames into Rec.2020 first, then the
// roll-off into the peak (the shoulder weighs Rec.2020 luminance; 9 Oct 2026),
// MaxCLL and MaxFALL measured from that, and the frames encoded as they are.
int cmd_deliver(const std::vector<std::string>& args) {
    auto usage = [](const std::string& m) {
        std::fprintf(stderr, "rudra-native deliver: error: %s\n", m.c_str());
        return 64;
    };
    if (args.empty()) return usage("the following arguments are required: input");
    fs::path input = args[0], output;
    SequenceEncodeOptions o;
    o.source = Primaries::Rec709;   // `deliver`'s default --source-space
    double nits_scale = 203.0;
    std::string space = "rec709";
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& k = args[i];
        if (k == "--no-verify-tags") { o.verify_tags = false; continue; }
        if (i + 1 >= args.size()) return usage("argument " + k + ": expected one argument");
        const std::string v = args[++i];
        char* end = nullptr;
        const double d = std::strtod(v.c_str(), &end);
        const bool num = end && *end == '\0' && !v.empty();
        if (k == "--output") output = v;
        else if (k == "--target") {
            if (!find_sequence_target(v)) return usage("invalid --target: " + v);
            o.target = v;
        } else if (k == "--fps" && num) o.fps = d;
        else if (k == "--peak-nits" && num) o.peak_nits = d;
        else if (k == "--min-nits" && num) o.min_nits = d;
        else if (k == "--nits-scale" && num) nits_scale = d;
        else if (k == "--source-space") {
            if (v == "rec2020") o.source = Primaries::Rec2020;
            else if (v == "rec709") o.source = Primaries::Rec709;
            else if (v == "p3d65") o.source = Primaries::P3D65;
            else return usage("invalid --source-space: " + v);
            space = v;
        } else return usage("unrecognized or invalid argument: " + k + " " + v);
    }
    if (output.empty()) return usage("the following arguments are required: --output");
    auto fail = [](const Error& e) {
        std::fprintf(stderr, "error: %s\n", e.message.c_str());
        return 2;
    };
    auto paths = list_linear_frames(input);
    if (!paths) return fail(paths.error());

    o.note = [](const std::string& n) { std::fprintf(stderr, "%s\n", n.c_str()); };
    auto delivered = deliver_linear_frames(*paths, output, o, nits_scale);
    if (!delivered) return fail(delivered.error());
    const fs::path* out = &delivered->file;
    const StaticMetadata sm = delivered->metadata;

    auto tags_value = [](const std::optional<ColourTags>& t) -> pyjson::Value {
        if (!t) return nullptr;
        pyjson::Dict d;
        for (const auto& [k, v] : *t) d.emplace_back(k, v);
        return d;
    };
    auto tags = colour_tags(*out);
    const SequenceTarget* t = find_sequence_target(o.target);
    // Only ProRes has frame headers: scanning HEVC for 'icpf' can match
    // compressed data and report tags that do not exist.
    const bool prores = std::find(t->codec.begin(), t->codec.end(), "prores_ks") != t->codec.end();
    const pyjson::Value report = pyjson::Dict{
        {"file", out->string()},
        {"frames", int(paths->size())},
        {"target", o.target},
        {"fps", o.fps},
        {"peak_nits", o.peak_nits},
        {"maxcll", sm.maxcll},
        {"maxfall", sm.maxfall},
        {"colour_tags", tags ? tags_value(*tags) : pyjson::Value()},
        {"prores_frame_tags", prores ? tags_value(prores_frame_tags(*out)) : pyjson::Value()},
        {"tags_verified", o.verify_tags},
        {"note", t->note},
    };
    std::printf("%s\n", pyjson::dumps(report, 2).c_str());
    return 0;
}

}  // namespace rudra
