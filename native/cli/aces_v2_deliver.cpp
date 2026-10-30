#include "aces_v2_deliver.hpp"

#include <cctype>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/metadata.hpp"
#include "rudra/core/timecode.hpp"
#include "rudra/deliver/dolby_vision.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
#include "rudra/platform/pyjson.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

namespace rudra {
namespace {

double parse_fps(const std::string& f) {
    const auto slash = f.find('/');
    try {
        const double v = slash == std::string::npos ? std::stod(f) : std::stod(f.substr(0, slash)) / std::stod(f.substr(slash + 1));
        if (v > 0) return v;
    } catch (...) {
    }
    return 24.0;
}

bool good_name(const std::string& n) {
    if (n.empty() || n.size() > 120 || n == "." || n == "..") return false;
    for (unsigned char c : n)
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == ' ')) return false;
    return n.back() != '.' && n.back() != ' ';
}

}  // namespace

Result<AcesDeliverOptions> aces_deliver_from_json(const nlohmann::json& req) {
    AcesDeliverOptions o;
    try {
        o.exr = req.value("exr", std::string("float"));
        o.exr_gamut = req.value("exr_gamut", std::string("acescg"));
        if (req.contains("frames"))
            for (const auto& f : req.at("frames")) o.frames.push_back(f.get<int>());
        o.fps = req.value("fps", 0.0);
        if (req.contains("range")) {
            const auto& r = req.at("range");
            if (!r.is_array() || r.size() != 2) return make_error(ErrorCode::InvalidArgument, "range is [first, last].");
            o.range_first = r[0].get<int>();
            o.range_last = r[1].get<int>();
        }
        o.name = req.value("name", std::string());
        o.start = req.value("start", 0);
        o.timecode = req.value("timecode", std::string());
        o.padding = req.value("padding", 6);
        o.separator = req.value("separator", std::string("."));
        o.into_existing = req.value("into_existing", false);
        o.create_dirs = req.value("create_dirs", true);
        o.reel = req.value("reel", std::string());
        o.source_display = req.value("source_display", std::string("bt1886"));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::InvalidArgument, "Bad delivery options.", e.what());
    }
    if (!aces_valid_source_display(o.source_display))
        return make_error(ErrorCode::InvalidArgument, "source_display is bt1886, srgb or gamma22.", o.source_display);
    if (o.exr != "float" && o.exr != "half" && o.exr != "none")
        return make_error(ErrorCode::InvalidArgument, "exr is float, half or none.", o.exr);
    if (o.exr_gamut != "acescg" && o.exr_gamut != "ap0")
        return make_error(ErrorCode::InvalidArgument, "exr_gamut is acescg or ap0.", o.exr_gamut);
    // ACES 2065-1 is written as the SMPTE ST 2065-4 container: half float RGB (28 Oct 2026 review).
    if (o.exr_gamut == "ap0" && o.exr == "float")
        return make_error(ErrorCode::InvalidArgument, "An ACES 2065-1 (ST 2065-4) master is half float: choose exr half, or ACEScg for float32.");
    for (int f : o.frames)
        if (f < 0) return make_error(ErrorCode::InvalidArgument, "Frame numbers start at 0.", std::to_string(f));
    if (req.contains("range") && (o.range_first < 0 || o.range_last < o.range_first))
        return make_error(ErrorCode::InvalidArgument, "range is [first, last], first >= 0.");
    if (o.range_first >= 0 && !o.frames.empty())
        return make_error(ErrorCode::InvalidArgument, "Choose frames or a range, not both.");
    if (o.fps < 0 || o.fps > 1000) return make_error(ErrorCode::InvalidArgument, "fps is out of range.");
    if (!o.name.empty() && !good_name(o.name))
        return make_error(ErrorCode::InvalidArgument, "A render name uses letters, digits, space, '.', '_' and '-'.", o.name);
    if (o.start < 0 || o.start > 99999999) return make_error(ErrorCode::InvalidArgument, "start is out of range.");
    if (!o.timecode.empty() && !parse_timecode(o.timecode))
        return make_error(ErrorCode::InvalidArgument, "timecode is HH:MM:SS:FF (or ;FF for drop frame).", o.timecode);
    if (o.padding < 1 || o.padding > 10) return make_error(ErrorCode::InvalidArgument, "padding is 1 to 10 digits.");
    if (o.separator != "." && o.separator != "_" && !o.separator.empty())
        return make_error(ErrorCode::InvalidArgument, "separator is '.', '_' or nothing.");
    if (o.into_existing && o.name.empty())
        return make_error(ErrorCode::InvalidArgument, "Writing into an existing folder needs a name.");
    if (o.reel.size() > 120) return make_error(ErrorCode::InvalidArgument, "The reel name is too long.");
    auto trims = aces_trims_from_json(req.value("trims", nlohmann::json()));
    if (!trims) return trims.error();
    o.trims = *trims;
    auto copies = parse_aces_log(req.value("log", std::string()), req.value("log_formats", std::string()));
    if (!copies) return copies.error();
    o.copies = *copies;
    o.copies.flat = o.into_existing;
    if (!o.writes_exr() && !o.copies.any())
        return make_error(ErrorCode::InvalidArgument, "Nothing to write: choose an EXR or an encoded copy.");
    return o;
}

std::string aces_frame_stem(const AcesDeliverOptions& o, std::size_t i, const std::string& exr_name) {
    if (o.name.empty()) return fs::path(exr_name).stem().string();
    char num[24];
    std::snprintf(num, sizeof num, "%0*lld", o.padding, static_cast<long long>(o.start) + static_cast<long long>(i));
    return o.name + o.separator + num;
}

Result<void> aces_prepare_out(const AcesDeliverOptions& o, const fs::path& out, std::size_t total, const std::string& movie) {
    std::error_code ec;
    if (!o.into_existing) {
        if (fs::exists(out, ec)) return make_error(ErrorCode::InvalidArgument, "Choose a new output folder.", out.string());
        fs::create_directories(out, ec);
        if (ec) return make_error(ErrorCode::IoError, "The output folder could not be made.", out.string());
        return {};
    }
    if (!fs::is_directory(out, ec)) {
        if (fs::exists(out, ec)) return make_error(ErrorCode::InvalidArgument, "That path is a file, not a folder.", out.string());
        if (!o.create_dirs)
            return make_error(ErrorCode::InvalidArgument, "The folder does not exist (turn on Create directories).", out.string());
        fs::create_directories(out, ec);
        if (ec) return make_error(ErrorCode::IoError, "The output folder could not be made.", out.string());
    }
    auto taken = [&](const fs::path& p) -> Result<void> {
        if (fs::exists(p, ec))
            return make_error(ErrorCode::InvalidArgument, "Already there; nothing is replaced. Change the name or version up.",
                              p.string());
        return {};
    };
    for (std::size_t i = 0; i < total; ++i) {
        const std::string stem = aces_frame_stem(o, i, "");
        if (o.writes_exr())
            if (auto r = taken(out / (stem + ".exr")); !r) return r;
        if (o.writes_exr() && o.exr_gamut == "ap0")
            if (auto r = taken(aces_mask_file(out / (stem + ".exr"))); !r) return r;
        if (o.copies.encoded() && o.copies.dpx10)
            if (auto r = taken(out / (stem + ".dpx")); !r) return r;
        if (o.copies.encoded() && o.copies.tiff16)
            if (auto r = taken(out / (stem + ".tif")); !r) return r;
    }
    if (!movie.empty())
        if (auto r = taken(out / movie); !r) return r;
    if (o.copies.hdr10()) {
        if (auto r = taken(aces_hdr10_file(o, out)); !r) return r;
        if (auto r = taken(aces_dovi_xml_file(o, out)); !r) return r;
        if (auto r = taken(aces_dovi_generate_file(o, out)); !r) return r;
    }
    return taken(aces_provenance_file(o, out));
}

fs::path aces_provenance_file(const AcesDeliverOptions& o, const fs::path& out) {
    return o.into_existing ? out / (o.name + "_provenance.json") : out / "provenance.json";
}

fs::path aces_hdr10_file(const AcesDeliverOptions& o, const fs::path& out) {
    return o.into_existing ? out / (o.name + "_hdr10.json") : out / "hdr10.json";
}

nlohmann::json aces_hdr10_json(double maxcll, double maxfall, double peak_nits) {
    // CTA-861.3 integers, rounded up as the analytic route's (core/measure maxcll_maxfall) are.
    const long cll = std::lround(std::ceil(std::max(0.0, maxcll))), fall = std::lround(std::ceil(std::max(0.0, std::min(maxfall, maxcll))));
    return {{"what", "HDR10 static metadata of the PQ deliverables of this render (enter it in the delivery or the encoder)"},
            {"output_transform", "ACES 2.0 - HDR " + std::to_string(std::lround(peak_nits)) + " nits (P3 D65) on Rec.2100-PQ"},
            {"container", "Rec.2020 primaries, SMPTE ST 2084 (PQ), P3 D65 limited"},
            {"mastering_display", {{"primaries", "P3 D65"},
                                   {"red", {0.680, 0.320}}, {"green", {0.265, 0.690}}, {"blue", {0.150, 0.060}},
                                   {"white", {0.3127, 0.3290}}, {"max_nits", std::lround(peak_nits)}, {"min_nits", 0.0001}}},
            {"maxcll", cll}, {"maxfall", fall},
            {"measured", "CTA-861.3 on the PQ output in display nits: MaxCLL the brightest max(R,G,B), MaxFALL the brightest frame's mean"},
            {"x265", "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(" +
                         std::to_string(std::lround(peak_nits * 10000.0)) + ",1):max-cll=" + std::to_string(cll) + "," +
                         std::to_string(fall)},
            {"resolve", "Project Settings > Color Management: HDR mastering " + std::to_string(std::lround(peak_nits)) +
                            " nits, P3-D65; Deliver: MaxCLL " +
                            std::to_string(cll) + ", MaxFALL " + std::to_string(fall)}};
}

std::string aces_movie_file(const AcesDeliverOptions& o, const std::string& stem) {
    return (o.name.empty() ? stem + "_" + o.copies.name() : o.name) + o.copies.movie_extension();
}

std::string aces_exr_label(const AcesDeliverOptions& o) {
    if (!o.writes_exr()) return "none (encoded copies only)";
    if (o.exr_gamut == "ap0")
        return "ACES 2065-1 (AP0), SMPTE ST 2065-4 container: half float RGB, uncompressed, acesImageContainerFlag 1; "
               "scene-linear, signed, unclamped; the reconstruction mask in mask/<frame>.exr";
    return std::string("ACEScg scene-linear, signed, unclamped, ") + (o.exr == "half" ? "half float (16-bit)" : "float32") +
           "; the reconstruction mask as the rudra.reconstructed channel";
}

Result<void> write_aces_master_exr(const fs::path& path, const PlanarBuffer& acescg, const PlanarBuffer& mask,
                                   const AcesDeliverOptions& o, const ExrAttributes& provenance,
                                   const ExrTypedAttributes& typed) {
    const bool half = o.exr == "half";
    const fs::path part = aces_part_file(path);
    // ST 2065-4: the AP0 file is RGB only, so its mask is a file of its own (mask/<frame>.exr, R=G=B).
    const bool ap0 = o.exr_gamut == "ap0";
    auto w = ap0 ? write_aces_exr(part, acescg, Primaries::Ap1, 1.0, provenance, true, nullptr, typed)
                 : write_acescg_exr(part, acescg, Primaries::Ap1, 1.0, provenance, half, &mask, typed);
    std::error_code ec;
    if (!w) {
        fs::remove(part, ec);
        return w;
    }
    if (ap0) {
        const fs::path mf = aces_mask_file(path);
        PlanarBuffer m3(3, mask.height(), mask.width());
        for (int c = 0; c < 3; ++c) std::copy(mask.plane(0), mask.plane(0) + mask.plane_size(), m3.plane(c));
        const fs::path mpart = aces_part_file(mf);
        auto mw = write_exr(mpart, m3, true, std::nullopt,
                            {{"rudra:mask", "rudra.reconstructed of " + path.filename().string() +
                                             ": highlight reconstruction weight, 0 = from source"}});
        if (!mw) {
            fs::remove(part, ec);
            fs::remove(mpart, ec);
            return mw;
        }
        if (auto r = aces_finish_part(mpart, mf); !r) {
            fs::remove(part, ec);
            return r;
        }
    }
    return aces_finish_part(part, path);
}

bool aces_valid_source_display(const std::string& s) { return s == "bt1886" || s == "srgb" || s == "gamma22"; }

void aces_adapt_source(PlanarBuffer& codes, const std::string& sd) {
    if (sd != "srgb" && sd != "gamma22") return;
    // One table per 16-bit code (the decoders' codes are 8 to 16 bits), then linear between.
    static constexpr int kN = 65536;
    std::vector<float> lut(kN);
    for (int i = 0; i < kN; ++i) {
        const double v = double(i) / (kN - 1);
        const double light = sd == "srgb" ? (v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4)) : std::pow(v, 2.2);
        lut[std::size_t(i)] = float(std::pow(light, 1.0 / 2.4));
    }
    for (float& c : codes.span()) {
        if (!std::isfinite(c)) continue;
        const double x = std::clamp(double(c), 0.0, 1.0) * (kN - 1);
        const int i = std::min(int(x), kN - 2);
        const double t = x - i;
        c = float(lut[std::size_t(i)] * (1.0 - t) + lut[std::size_t(i) + 1] * t);
    }
}

std::string aces_source_display_label(const std::string& s) {
    if (s == "srgb") return "sRGB display (piecewise 2.2), re-encoded to BT.1886 2.4 before the model";
    if (s == "gamma22") return "gamma 2.2 display, re-encoded to BT.1886 2.4 before the model";
    return "Rec.709 on BT.1886 (2.4), read as the ACES 2.0 SDR 100 nits output (the model's contract)";
}

Result<AcesDolbyVision> aces_dolby_vision(const std::vector<FrameStats>& stats, double peak_nits, const std::string& name,
                                          double fps, int width, int height) {
    if (stats.empty()) return make_error(ErrorCode::InvalidArgument, "No frames for Dolby Vision.");
    const std::vector<Shot> shots = detect_shots(stats);
    AcesDolbyVision d;
    d.shots = int(shots.size());
    d.generate_json = pyjson::dumps(to_dovi_generate_json(stats, shots, peak_nits, 0.0001), 2);
    auto meta = parse_dovi_generate(d.generate_json);
    if (!meta) return meta.error();
    // fps as a fraction: 23.976 and friends exactly.
    long long num = std::llround(fps * 1000.0), den = 1000;
    for (double base : {24.0, 30.0, 60.0, 48.0, 120.0})
        if (std::abs(fps - base * 1000.0 / 1001.0) < 1e-3) num = std::llround(base * 1000.0), den = 1001;
    if (den == 1000 && num % 1000 == 0) num /= 1000, den = 1;
    auto xml = dovi_metadata_xml(*meta, name, num, den, width, height);
    if (!xml) return xml.error();
    d.xml = *xml;
    return d;
}

fs::path aces_dovi_xml_file(const AcesDeliverOptions& o, const fs::path& out) {
    return o.into_existing ? out / (o.name + "_dolby_vision.xml") : out / "dolby_vision.xml";
}

fs::path aces_dovi_generate_file(const AcesDeliverOptions& o, const fs::path& out) {
    return o.into_existing ? out / (o.name + "_dovi_generate.json") : out / "dovi_generate.json";
}

fs::path aces_mask_file(const fs::path& exr) { return exr.parent_path() / "mask" / exr.filename(); }

fs::path aces_part_file(const fs::path& file) { return fs::path(file).concat(".part"); }

Result<void> aces_finish_part(const fs::path& part, const fs::path& file) {
    std::error_code ec;
    fs::rename(part, file, ec);
    if (ec) {
        fs::remove(part, ec);
        return make_error(ErrorCode::IoError, "The finished file could not be put in place.", file.string());
    }
    return {};
}

Result<void> aces_write_whole(const fs::path& file, const std::string& bytes) {
    const fs::path part = aces_part_file(file);
    {
        std::ofstream o(part, std::ios::binary | std::ios::trunc);
        o.write(bytes.data(), std::streamsize(bytes.size()));
        o.flush();
        if (!o) {
            o.close();
            std::error_code ec;
            fs::remove(part, ec);
            return make_error(ErrorCode::IoError, "The file could not be written (disk full or no access?).", file.string());
        }
    }
    return aces_finish_part(part, file);
}

AcesFrameStamp aces_frame_stamp(const AcesDeliverOptions& o, std::size_t i, const AcesFrameIn& f, double run_fps) {
    AcesFrameStamp s;
    s.fps = f.fps > 0 ? f.fps : (o.fps > 0 ? o.fps : run_fps);
    // A typed start timecode wins (a movie's frame k is start + k); else the movie's own.
    const long long offset = f.frame >= 0 ? f.frame : static_cast<long long>(i);
    s.timecode = !o.timecode.empty() ? timecode_at(o.timecode, offset, s.fps) : f.timecode;
    s.reel = !o.reel.empty() ? o.reel : f.reel;
    return s;
}

PlanarBuffer aces_trimmed(const AcesDeliverOptions& o, const PlanarBuffer& acescg, const PlanarBuffer& mask,
                          const aces_v2::AcesView& view, const PlanarBuffer& codes) {
    if (o.trims.identity()) return acescg;
    return aces_v2::apply_trims(acescg, aces_v2::view_inverse(view, codes), mask, o.trims);
}

AcesLightLevel aces_light_level(const PlanarBuffer& acescg, const aces_v2::AcesView* hdr_pq, int index) {
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Rec2020);
    AcesLightLevel l;
    if (hdr_pq) {   // the deliverable's own light: PQ codes -> display nits
        const PlanarBuffer codes = aces_v2::view_forward(*hdr_pq, acescg);
        const std::size_t n = codes.plane_size();
        double sum = 0, mn = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            double mx = 0;
            for (int c = 0; c < 3; ++c) {
                const float v = codes.plane(c)[i];
                if (std::isfinite(v)) mx = std::max(mx, double(pq_eotf(std::clamp(v, 0.0f, 1.0f))));
            }
            l.maxcll = std::max(l.maxcll, mx);
            mn = std::min(mn, mx);
            sum += mx;
        }
        l.maxfall = n ? sum / double(n) : 0.0;
        // The frame's statistics for Dolby Vision: the histogram (cut detection) on every 4th
        // pixel each way, the min / mean / max of MaxRGB on every pixel.
        const int step = 4, h = codes.height(), w = codes.width();
        NitsFrame small((h + step - 1) / step, (w + step - 1) / step);
        for (int y = 0, sy = 0; y < h; y += step, ++sy)
            for (int x = 0, sx = 0; x < w; x += step, ++sx)
                for (int c = 0; c < 3; ++c) {
                    const float v = codes.plane(c)[std::size_t(y) * std::size_t(w) + std::size_t(x)];
                    small.at(c, sy, sx) = std::isfinite(v) ? double(pq_eotf(std::clamp(v, 0.0f, 1.0f))) : 0.0;
                }
        if (small.plane_size() > 0) l.stats = analyze_frame(small, index);
        l.stats.index = index;
        l.stats.min_nits = n ? mn : 0.0;
        l.stats.avg_nits = l.maxfall;
        l.stats.max_nits = l.maxcll;
    }
    double sum = 0;
    const std::size_t n = acescg.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        const double v[3] = {acescg.plane(0)[i], acescg.plane(1)[i], acescg.plane(2)[i]};
        double mx = 0;
        for (int c = 0; c < 3; ++c) {
            const double x = m[std::size_t(c)][0] * v[0] + m[std::size_t(c)][1] * v[1] + m[std::size_t(c)][2] * v[2];
            if (std::isfinite(x)) mx = std::max(mx, x);
        }
        l.scene_peak = std::max(l.scene_peak, mx);
        sum += mx;
    }
    l.scene_peak *= aces_v2::kAcesDiffuseNits;
    l.scene_average = n ? sum / double(n) * aces_v2::kAcesDiffuseNits : 0.0;
    return l;
}

Result<aces_v2::AcesTrims> aces_trims_from_json(const nlohmann::json& j) {
    aces_v2::AcesTrims t;
    if (j.is_null()) return t;
    if (!j.is_object()) return make_error(ErrorCode::InvalidArgument, "trims is an object.");
    try {
        t.strength = j.value("strength", 1.0);
        t.saturation = j.value("saturation", 1.0);
        t.peak_nits = j.value("peak_nits", 0.0);
        t.exposure_ev = j.value("exposure_ev", 0.0);
        t.contrast = j.value("contrast", 1.0);
    } catch (const std::exception& e) {
        return make_error(ErrorCode::InvalidArgument, "Bad trims.", e.what());
    }
    if (!(t.strength >= 0 && t.strength <= 1.5)) return make_error(ErrorCode::InvalidArgument, "strength is 0 to 1.5.");
    if (!(t.saturation >= 0 && t.saturation <= 2)) return make_error(ErrorCode::InvalidArgument, "saturation is 0 to 2.");
    if (!(t.peak_nits == 0 || (t.peak_nits >= 100 && t.peak_nits <= 10000)))
        return make_error(ErrorCode::InvalidArgument, "peak_nits is 0 (none) or 100 to 10000.");
    if (!(t.exposure_ev >= -3 && t.exposure_ev <= 3)) return make_error(ErrorCode::InvalidArgument, "exposure_ev is -3 to 3.");
    if (!(t.contrast >= 0.5 && t.contrast <= 1.5)) return make_error(ErrorCode::InvalidArgument, "contrast is 0.5 to 1.5.");
    return t;
}

nlohmann::json aces_trims_json(const aces_v2::AcesTrims& t) {
    return {{"strength", t.strength}, {"saturation", t.saturation}, {"peak_nits", t.peak_nits},
            {"exposure_ev", t.exposure_ev}, {"contrast", t.contrast}, {"identity", t.identity()},
            {"text", aces_v2::trims_text(t)}};
}

Result<std::unique_ptr<AcesFrameFeed>> AcesFrameFeed::open(const std::vector<fs::path>& sources, const std::vector<int>& frames,
                                                           const fs::path& decode_log, int range_first, int range_last) {
    std::unique_ptr<AcesFrameFeed> f(new AcesFrameFeed());
    f->log_ = decode_log;
    if (range_first >= 0) {
        if (sources.size() != 1 || !aces_is_video(sources.front()))
            return make_error(ErrorCode::InvalidArgument, "A frame range is for one video.");
        auto all = AcesInputs::open(sources, decode_log);
        if (!all) return all.error();
        if (auto r = (*all)->set_range(range_first, range_last); !r) return r.error();
        f->fps_ = (*all)->fps();
        f->all_ = std::move(*all);
        return f;
    }
    if (frames.empty() || (sources.size() == 1 && !aces_is_video(sources.front()))) {
        auto all = AcesInputs::open(sources, decode_log);
        if (!all) return all.error();
        f->fps_ = (*all)->fps();
        f->all_ = std::move(*all);
        return f;
    }
    if (sources.size() != 1) return make_error(ErrorCode::InvalidArgument, "Frames are chosen from one video.");
    auto vs = aces_open_video(sources.front());
    if (!vs) return vs.error();
    for (int n : frames)
        if (n >= vs->clock.frames)
            return make_error(ErrorCode::InvalidArgument, "That frame is past the end of the video.",
                              std::to_string(n) + " of " + std::to_string(vs->clock.frames));
    f->video_ = sources.front();
    f->frames_ = frames;
    f->fps_ = parse_fps(vs->clock.fps);
    return f;
}

std::size_t AcesFrameFeed::total() const noexcept { return all_ ? all_->total() : frames_.size(); }

Result<AcesFrameIn> AcesFrameFeed::next() {
    if (all_) return all_->next();
    if (at_ >= frames_.size()) return make_error(ErrorCode::InvalidArgument, "No more frames.");
    return aces_read_one(video_, frames_[at_++], log_);
}

}  // namespace rudra
