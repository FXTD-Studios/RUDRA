#pragma once
// Dolby Vision from RUDRA itself (roadmap 5.5, 20 Oct 2026): the per-shot
// metadata a PQ master already carries beside it as <stem>_dovi_generate.json
// (core/metadata.hpp to_dovi_generate_json), written as
//   - a binary RPU, profile 8.1 (an HDR10 base layer, no enhancement layer),
//     CM v4.0: one RPU per frame, Annex B (start code 00 00 00 01 before each),
//     each the RPU NAL payload (prefix 0x19), emulation-prevented, with its
//     CRC32 -- the file `dovi_tool inject-rpu` and x265's --dolby-vision-rpu
//     take; and
//   - a Dolby Vision metadata XML (CM v4.0, the DolbyLabsMDF 4.0.2 schema), the
//     file Resolve and the Dolby tools import.
// The RPU carries what `dovi_tool generate` puts in a profile 8.1 CM v4.0 RPU
// from the same config: identity mapping, the BT.2020 PQ signal, per frame L1
// (its shot's min/avg/max PQ) with a scene refresh at each shot's first frame,
// L5 (no letterbox), L6 (the mastering display and MaxCLL/MaxFALL), and the
// CM v4.0 blocks L9 (source primaries index 0), L11 (content type film,
// reference mode) and L254 (DM version 2). It is byte for byte the RPU
// dovi_tool 2.3 generates (tests/conform/test_dolby_vision.cpp, against files
// it made). No trims (L2/L8) and no L3/L4: the target displays' tone mapping
// is the Dolby decoder's, from L1 and L6.
//
// Not a Dolby certification: a deliverable for a licensed workflow is still
// checked with Dolby's own tools (metafier / the Dolby Vision Professional
// Tools) before it is delivered.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct DoviShot {
    int start = 0, duration = 0;
    int min_pq = 0, max_pq = 0, avg_pq = 0;   // L1, 12-bit PQ codes
};

struct DoviMetadata {
    int length = 0;                            // frames
    int max_display_nits = 1000;               // L6, nits
    int min_display_ten_thousandths = 50;      // L6, 0.0001 nits (50 = 0.005 nits)
    int max_cll = 0, max_fall = 0;             // L6, nits
    int source_diagonal_inches = 42;
    std::vector<DoviShot> shots;               // in order, covering [0, length)
};

// The generate config RUDRA writes (and dovi_tool reads): cm_version V40,
// profile 8.1, length, level6 and shots of Level1 blocks. Anything else is
// refused rather than dropped.
Result<DoviMetadata> parse_dovi_generate(std::string_view json);

// One RPU NAL payload per frame (before emulation prevention).
std::vector<std::uint8_t> dovi_rpu_payload(const DoviMetadata& m, const DoviShot& shot, bool scene_refresh);

// The whole RPU file, every frame, Annex B.
Result<std::vector<std::uint8_t>> dovi_rpu_file(const DoviMetadata& m);

// The metadata XML, CM v4.0. `name`: the composition's; `fps` as num/den.
Result<std::string> dovi_metadata_xml(const DoviMetadata& m, std::string_view name, long long fps_num,
                                      long long fps_den, int width, int height);

}  // namespace rudra
