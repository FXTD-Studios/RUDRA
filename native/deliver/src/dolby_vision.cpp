#include "rudra/deliver/dolby_vision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <sstream>

#include <nlohmann/json.hpp>

namespace rudra {

namespace {

Error refuse(std::string m, std::string d = {}) { return make_error(ErrorCode::Unsupported, std::move(m), std::move(d)); }

class BitWriter {
public:
    void u(std::uint64_t v, int n) {
        for (int i = n - 1; i >= 0; --i) bit(int((v >> i) & 1u));
    }
    void ue(std::uint64_t v) {
        const std::uint64_t x = v + 1;
        int len = 0;
        while ((x >> len) > 1) ++len;
        u(0, len);
        u(x, len + 1);
    }
    void se(std::int64_t v) { ue(v > 0 ? std::uint64_t(2 * v - 1) : std::uint64_t(-2 * v)); }
    void align() {
        while (n_ % 8) bit(0);
    }
    const std::vector<std::uint8_t>& bytes() const { return out_; }
    std::size_t bits() const { return n_; }

private:
    void bit(int b) {
        if (n_ % 8 == 0) out_.push_back(0);
        if (b) out_.back() |= std::uint8_t(0x80u >> (n_ % 8));
        ++n_;
    }
    std::vector<std::uint8_t> out_;
    std::size_t n_ = 0;
};

std::uint32_t crc32_mpeg2(const std::uint8_t* p, std::size_t n) {
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c ^= std::uint32_t(p[i]) << 24;
        for (int k = 0; k < 8; ++k) c = (c & 0x80000000u) ? (c << 1) ^ 0x04C11DB7u : c << 1;
    }
    return c;
}

// The source display the RPU names: dovi_tool snaps the mastering display to
// its presets (measured against dovi_tool 2.3: 2 000, 4 000 and 10 000 nits by
// their own PQ codes, any other peak as 1 000 nits; a minimum of 0.005 nits as
// 62, one of 0.001 nits or less as 7, any other as 0).
int source_max_pq(int max_display_nits) {
    switch (max_display_nits) {
        case 2000: return 3388;
        case 4000: return 3696;
        case 10000: return 4095;
        default: return 3079;
    }
}
int source_min_pq(int min_display_ten_thousandths) {
    if (min_display_ten_thousandths == 50) return 62;
    return min_display_ten_thousandths <= 10 ? 7 : 0;
}

// One extension block: length (ue), level (8 bits), payload padded to length bytes.
template <class F>
void ext_block(BitWriter& w, int level, int length, F payload) {
    w.ue(std::uint64_t(length));
    w.u(std::uint64_t(level), 8);
    const std::size_t start = w.bits();
    payload(w);
    const std::size_t used = w.bits() - start;
    w.u(0, int(std::size_t(length) * 8 - used));
}

// The profile 8.1 constants dovi_tool writes (BT.2020 YCbCr to RGB, RGB to
// LMS, a 12-bit PQ RGB signal).
constexpr std::array<std::int16_t, 9> kYccToRgb = {9574, 0, 13802, 9574, -1540, -5348, 9574, 17610, 0};
constexpr std::array<std::uint32_t, 3> kYccOffsets = {16777216u, 134217728u, 134217728u};
constexpr std::array<std::int16_t, 9> kRgbToLms = {7222, 8771, 390, 2654, 12430, 1300, 0, 422, 15962};

std::string fmt(double v, const char* f = "%.6f") {
    char b[48];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

std::string xml_escape(std::string_view s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '&': o += "&amp;"; break;
            case '<': o += "&lt;"; break;
            case '>': o += "&gt;"; break;
            case '"': o += "&quot;"; break;
            default: o += c;
        }
    }
    return o;
}

// A stable UUID-shaped id from text (the XML's UniqueIDs).
std::string uuid_of(std::string_view text) {
    std::uint64_t a = 1469598103934665603ull, b = 1099511628211ull ^ 0x9E3779B97F4A7C15ull;
    for (unsigned char c : text) {
        a = (a ^ c) * 1099511628211ull;
        b = (b ^ c) * 0x100000001B3ull + 0x7F4A7C15ull;
    }
    char o[40];
    std::snprintf(o, sizeof o, "%08x-%04x-4%03x-a%03x-%012llx", unsigned(a >> 32), unsigned((a >> 16) & 0xFFFF),
                  unsigned(a & 0xFFF), unsigned((b >> 52) & 0xFFF), static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFull));
    return o;
}

}  // namespace

Result<DoviMetadata> parse_dovi_generate(std::string_view text) {
    DoviMetadata m;
    try {
        const auto j = nlohmann::json::parse(text);
        if (j.value("cm_version", std::string()) != "V40") return refuse("Only CM v4.0 configs are written", j.value("cm_version", std::string()));
        if (j.value("profile", std::string()) != "8.1") return refuse("Only profile 8.1 is written", j.value("profile", std::string()));
        m.length = j.at("length").get<int>();
        const auto& l6 = j.at("level6");
        m.max_display_nits = l6.at("max_display_mastering_luminance").get<int>();
        m.min_display_ten_thousandths = l6.at("min_display_mastering_luminance").get<int>();
        m.max_cll = l6.at("max_content_light_level").get<int>();
        m.max_fall = l6.at("max_frame_average_light_level").get<int>();
        for (const auto& s : j.at("shots")) {
            DoviShot shot;
            shot.start = s.at("start").get<int>();
            shot.duration = s.at("duration").get<int>();
            bool l1 = false;
            for (const auto& b : s.value("metadata_blocks", nlohmann::json::array())) {
                if (!b.contains("Level1") || b.size() != 1) return refuse("Only Level 1 shot blocks are written", b.dump());
                shot.min_pq = b["Level1"].at("min_pq").get<int>();
                shot.max_pq = b["Level1"].at("max_pq").get<int>();
                shot.avg_pq = b["Level1"].at("avg_pq").get<int>();
                l1 = true;
            }
            if (!l1) return refuse("A shot has no Level 1 block", std::to_string(shot.start));
            m.shots.push_back(shot);
        }
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The Dolby Vision config could not be read", e.what());
    }
    // Level 6's limits, as dovi_tool checks them.
    for (int v : {m.max_display_nits, m.min_display_ten_thousandths, m.max_cll, m.max_fall})
        if (v < 0 || v > 10000) return make_error(ErrorCode::InvalidArgument, "A Level 6 value is outside 0 to 10 000", std::to_string(v));
    int next = 0;
    for (const auto& s : m.shots) {
        if (s.start != next || s.duration <= 0)
            return make_error(ErrorCode::InvalidArgument, "The shots must cover the frames in order", std::to_string(s.start));
        for (int v : {s.min_pq, s.max_pq, s.avg_pq})
            if (v < 0 || v > 4095) return make_error(ErrorCode::InvalidArgument, "An L1 value is outside 12 bits");
        next = s.start + s.duration;
    }
    if (next != m.length) return make_error(ErrorCode::InvalidArgument, "The shots do not cover the frames", std::to_string(next));
    return m;
}

std::vector<std::uint8_t> dovi_rpu_payload(const DoviMetadata& m, const DoviShot& shot, bool scene_refresh) {
    BitWriter w;
    w.u(0x19, 8);   // rpu_nal_prefix
    // rpu_data_header
    w.u(2, 6);      // rpu_type
    w.u(18, 11);    // rpu_format
    w.u(1, 4);      // vdr_rpu_profile
    w.u(0, 4);      // vdr_rpu_level
    w.u(1, 1);      // vdr_seq_info_present_flag
    w.u(0, 1);      // chroma_resampling_explicit_filter_flag
    w.u(0, 2);      // coefficient_data_type: fixed point
    w.ue(23);       // coefficient_log2_denom
    w.u(1, 2);      // vdr_rpu_normalized_idc
    w.u(0, 1);      // bl_video_full_range_flag
    w.ue(2);        // bl_bit_depth_minus8: a 10-bit base layer
    w.ue(2);        // el_bit_depth_minus8
    w.ue(4);        // vdr_bit_depth_minus8
    w.u(0, 1);      // spatial_resampling_filter_flag
    w.u(0, 3);      // reserved_zero_3bits
    w.u(0, 1);      // el_spatial_resampling_filter_flag
    w.u(1, 1);      // disable_residual_flag
    w.u(1, 1);      // vdr_dm_metadata_present_flag
    w.u(0, 1);      // use_prev_vdr_rpu_flag
    // rpu_data_mapping: identity, one piece per component
    w.ue(0);        // vdr_rpu_id
    w.ue(0);        // mapping_color_space
    w.ue(0);        // mapping_chroma_format_idc
    for (int c = 0; c < 3; ++c) {
        w.ue(0);         // num_pivots_minus2
        w.u(0, 10);      // pivot 0
        w.u(1023, 10);   // pivot 1 - pivot 0
    }
    w.ue(0);        // num_x_partitions_minus1
    w.ue(0);        // num_y_partitions_minus1
    for (int c = 0; c < 3; ++c) {
        w.ue(0);         // mapping_idc: polynomial
        w.ue(0);         // poly_order_minus1: linear
        w.u(0, 1);       // linear_interp_flag
        w.se(0), w.u(0, 23);   // a0 = 0
        w.se(1), w.u(0, 23);   // a1 = 1
    }
    // vdr_dm_data_payload
    w.ue(0);        // affected_dm_metadata_id
    w.ue(0);        // current_dm_metadata_id
    w.ue(scene_refresh ? 1 : 0);
    for (auto v : kYccToRgb) w.u(std::uint16_t(v), 16);
    for (auto v : kYccOffsets) w.u(v, 32);
    for (auto v : kRgbToLms) w.u(std::uint16_t(v), 16);
    w.u(65535, 16);  // signal_eotf: PQ
    w.u(0, 16), w.u(0, 16), w.u(0, 32);   // signal_eotf_param0..2
    w.u(12, 5);      // signal_bit_depth
    w.u(0, 2);       // signal_color_space: RGB
    w.u(0, 2);       // signal_chroma_format
    w.u(1, 2);       // signal_full_range_flag
    w.u(std::uint64_t(source_min_pq(m.min_display_ten_thousandths)), 12);
    w.u(std::uint64_t(source_max_pq(m.max_display_nits)), 12);
    w.u(std::uint64_t(m.source_diagonal_inches), 10);
    // CM v2.9 blocks: L1, L5, L6
    w.ue(3);
    w.align();
    ext_block(w, 1, 5, [&](BitWriter& b) {
        b.u(std::uint64_t(shot.min_pq), 12), b.u(std::uint64_t(shot.max_pq), 12), b.u(std::uint64_t(shot.avg_pq), 12);
    });
    ext_block(w, 5, 7, [](BitWriter& b) {
        for (int i = 0; i < 4; ++i) b.u(0, 13);   // no letterbox
    });
    ext_block(w, 6, 8, [&](BitWriter& b) {
        b.u(std::uint64_t(m.max_display_nits), 16), b.u(std::uint64_t(m.min_display_ten_thousandths), 16);
        b.u(std::uint64_t(m.max_cll), 16), b.u(std::uint64_t(m.max_fall), 16);
    });
    // CM v4.0 blocks: L9, L11, L254
    w.ue(3);
    w.align();
    ext_block(w, 9, 1, [](BitWriter& b) { b.u(0, 8); });               // source primaries: index 0
    ext_block(w, 11, 4, [](BitWriter& b) {
        b.u(1, 8);       // content type: film
        b.u(0x10, 8);    // reference mode, D65 white
        b.u(0, 16);
    });
    ext_block(w, 254, 2, [](BitWriter& b) { b.u(0, 8), b.u(2, 8); });   // dm_mode 0, dm_version_index 2
    w.align();
    std::vector<std::uint8_t> out = w.bytes();
    const std::uint32_t crc = crc32_mpeg2(out.data() + 1, out.size() - 1);
    for (int s = 24; s >= 0; s -= 8) out.push_back(std::uint8_t(crc >> s));
    out.push_back(0x80);   // rbsp trailing bits
    return out;
}

Result<std::vector<std::uint8_t>> dovi_rpu_file(const DoviMetadata& m) {
    if (m.shots.empty() || m.length <= 0) return make_error(ErrorCode::InvalidArgument, "No frames to write an RPU for");
    std::vector<std::uint8_t> file;
    for (const auto& shot : m.shots)
        for (int f = 0; f < shot.duration; ++f) {
            const auto payload = dovi_rpu_payload(m, shot, f == 0);
            file.insert(file.end(), {0, 0, 0, 1});
            int zeros = 0;
            for (std::uint8_t b : payload) {   // emulation prevention
                if (zeros >= 2 && b <= 3) {
                    file.push_back(3);
                    zeros = 0;
                }
                file.push_back(b);
                zeros = b == 0 ? zeros + 1 : 0;
            }
        }
    return file;
}

Result<std::string> dovi_metadata_xml(const DoviMetadata& m, std::string_view name, long long fps_num, long long fps_den,
                                      int width, int height) {
    if (m.shots.empty()) return make_error(ErrorCode::InvalidArgument, "No shots to write");
    if (fps_num <= 0 || fps_den <= 0 || width <= 0 || height <= 0)
        return make_error(ErrorCode::InvalidArgument, "The XML needs the frame rate and the size");
    const double min_nits = m.min_display_ten_thousandths / 10000.0;
    const std::string aspect = fmt(double(width) / double(height), "%.7g");
    const std::string seed = std::string(name) + ":" + std::to_string(m.length);
    std::ostringstream x;
    const std::string encoding =
        "        <ColorEncoding>\n"
        "          <Primaries>\n"
        "            <Red>0.708 0.292</Red>\n"
        "            <Green>0.17 0.797</Green>\n"
        "            <Blue>0.131 0.046</Blue>\n"
        "          </Primaries>\n"
        "          <WhitePoint>0.3127 0.329</WhitePoint>\n"
        "          <PeakBrightness>" + std::to_string(m.max_display_nits) + "</PeakBrightness>\n"
        "          <MinimumBrightness>" + fmt(min_nits, "%.4g") + "</MinimumBrightness>\n"
        "          <Encoding>pq</Encoding>\n"
        "          <BitDepth>16</BitDepth>\n"
        "          <ColorSpace>rgb</ColorSpace>\n"
        "          <ChromaFormat>444</ChromaFormat>\n"
        "          <SignalRange>computer</SignalRange>\n"
        "        </ColorEncoding>\n";
    x << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      << "<DolbyLabsMDF xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
         "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" version=\"4.0.2\" "
         "xmlns=\"http://www.dolby.com/schemas/dvmd/4_0_2\">\n"
      << "  <SourceList>\n    <Source type=\"Video\">\n"
      << "      <Name>" << xml_escape(name) << "</Name>\n"
      << "      <UniqueID>" << uuid_of(seed + ":source") << "</UniqueID>\n"
      << encoding
      << "      <Level6 level=\"6\">\n        <MaxCLL>" << m.max_cll << "</MaxCLL>\n        <MaxFALL>" << m.max_fall
      << "</MaxFALL>\n      </Level6>\n"
      << "    </Source>\n  </SourceList>\n"
      << "  <RevisionHistory>\n    <Revision>\n      <DateTime>1970-01-01T00:00:00Z</DateTime>\n"
      << "      <Author>RUDRA</Author>\n      <Software>RUDRA</Software>\n      <SoftwareVersion>1.0</SoftwareVersion>\n"
      << "    </Revision>\n  </RevisionHistory>\n"
      << "  <Outputs>\n    <Output>\n"
      << "      <CompositionName>" << xml_escape(name) << "</CompositionName>\n"
      << "      <UniqueID>" << uuid_of(seed + ":output") << "</UniqueID>\n"
      << "      <NumberVideoTracks>1</NumberVideoTracks>\n"
      << "      <CanvasAspectRatio>" << aspect << "</CanvasAspectRatio>\n"
      << "      <ImageAspectRatio>" << aspect << "</ImageAspectRatio>\n"
      << "      <Video>\n        <Track>\n"
      << "          <TrackName>V1</TrackName>\n"
      << "          <UniqueID>" << uuid_of(seed + ":track") << "</UniqueID>\n"
      << "          <Rate>\n            <n>" << fps_num << "</n>\n            <d>" << fps_den << "</d>\n          </Rate>\n";
    {
        std::string e = encoding;   // indented two more for the track
        std::string indented;
        std::istringstream in(e);
        for (std::string line; std::getline(in, line);) indented += "  " + line + "\n";
        x << indented;
    }
    x << "          <Level6 level=\"6\">\n            <MaxCLL>" << m.max_cll << "</MaxCLL>\n            <MaxFALL>"
      << m.max_fall << "</MaxFALL>\n          </Level6>\n"
      << "          <PluginNode>\n"
      << "            <DVGlobalData>\n"
      << "              <MasteringDisplay level=\"0\">\n"
      << "                <ID>-1</ID>\n"
      << "                <Name>RUDRA master " << m.max_display_nits << " nits P3/PQ</Name>\n"
      << "                <Primaries>\n"
      << "                  <Red>0.68 0.32</Red>\n                  <Green>0.265 0.69</Green>\n"
      << "                  <Blue>0.15 0.06</Blue>\n                </Primaries>\n"
      << "                <WhitePoint>0.3127 0.329</WhitePoint>\n"
      << "                <PeakBrightness>" << m.max_display_nits << "</PeakBrightness>\n"
      << "                <MinimumBrightness>" << fmt(min_nits, "%.4g") << "</MinimumBrightness>\n"
      << "                <DiagonalSize>" << m.source_diagonal_inches << "</DiagonalSize>\n"
      << "              </MasteringDisplay>\n"
      << "              <Level9 level=\"9\">\n                <SourceColorPrimary>0</SourceColorPrimary>\n              </Level9>\n"
      << "              <Level11 level=\"11\">\n                <ContentType>1</ContentType>\n"
      << "                <IntendedWhitePoint>0</IntendedWhitePoint>\n                <ExtensionProperties>1</ExtensionProperties>\n"
      << "              </Level11>\n"
      << "            </DVGlobalData>\n"
      << "            <Level254 level=\"254\">\n              <DMMode>0</DMMode>\n              <DMVersion>2</DMVersion>\n"
      << "            </Level254>\n"
      << "          </PluginNode>\n";
    for (std::size_t i = 0; i < m.shots.size(); ++i) {
        const auto& s = m.shots[i];
        x << "          <Shot>\n"
          << "            <UniqueID>" << uuid_of(seed + ":shot:" + std::to_string(i)) << "</UniqueID>\n"
          << "            <Record>\n              <In>" << s.start << "</In>\n              <Duration>" << s.duration
          << "</Duration>\n            </Record>\n"
          << "            <PluginNode>\n              <DVDynamicData>\n"
          << "                <Level1 level=\"1\">\n                  <ImageCharacter>" << fmt(s.min_pq / 4095.0) << " "
          << fmt(s.avg_pq / 4095.0) << " " << fmt(s.max_pq / 4095.0) << "</ImageCharacter>\n"
          << "                </Level1>\n"
          << "              </DVDynamicData>\n            </PluginNode>\n"
          << "          </Shot>\n";
    }
    x << "        </Track>\n      </Video>\n    </Output>\n  </Outputs>\n</DolbyLabsMDF>\n";
    return x.str();
}

}  // namespace rudra
