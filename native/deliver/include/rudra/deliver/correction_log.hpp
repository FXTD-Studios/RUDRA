#pragma once
// Opt-in correction logging (roadmap 5.2, 20 Oct 2026): when the artist
// accepts a grade on real SDR, the frame and the accepted HDR are kept as a
// training pair, with where the frame came from and the consent that allowed
// keeping it. Nothing is logged without a recorded opt-in, and a withdrawn
// consent stops both new records and every later export.
//
// A log is a folder:
//   consent.json            the opt-in: who, when, the terms' version, the
//                           scope, the rights holder (source_id) and the
//                           rights claimed; a withdrawal is added to it
//   _ingest_config.json     the HDR storage (pq_10000, diffuse white 203
//                           nits), as the training loader and the commercial
//                           preflight look for it beside the pairs
//   sdr/<id>.png            the SDR frame as decoded: 16-bit RGB code values,
//                           Rec.709 primaries
//   hdr/<id>.png            the accepted master: Rec.2020, ST 2084 (PQ, 10 000
//                           nits) 16-bit, what pipeline/hdr_io decodes
//   meta/<id>.json          the record: the source file and its SHA-256, the
//                           frame, the grade (master params), the consent's
//                           SHA-256 and a copy of its rights, the app
//                           version, the pair's hashes
//   consent_history/<sha256>.json  every consent a record was logged under or
//                           that was replaced or withdrawn, as it was written
// export_correction_manifest writes a training manifest (JSONL) from the
// records, with absolute paths, as pipeline/commercial_preflight.py reads it:
// source_id, scene_id (the source file's hash: frames of one file share a
// split), asset_id, split (1 in 10 scenes to val, by that hash), sdr/hdr paths
// and SHA-256, metadata_path, sdr_kind "real", target_kind
// "artist_accepted_grade" (the target is RUDRA's output as an artist accepted
// it; trainers may weight or filter it on that), commercial_ok when the
// rights the record was logged under are "owner" or "licensed" (a later
// consent does not change an earlier record's rights).
//
// Only Rec.709 frames are logged: the corpus's SDR is Rec.709 code. A frame in
// other primaries is refused, not converted.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/core/master.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kCorrectionLogVersion = 1;

struct CorrectionConsent {
    bool opted_in = false;
    std::string by;               // who agreed (a name or account)
    std::string at;               // when, ISO 8601 UTC
    std::string terms_version;    // the text they agreed to, by version
    std::string scope = "training RUDRA reconstruction models";
    std::string source_id;        // the rights holder, as the licence review names it
    std::string rights;           // "owner", "licensed" or "unknown"
    std::string note;
    // A withdrawal (read back; set by withdraw_correction_consent).
    std::string withdrawn_by, withdrawn_at;
    bool withdrawn() const { return !withdrawn_at.empty(); }
};

// Make the folder a log with this consent (an existing log keeps its records
// and takes the new consent; a withdrawn one stays withdrawn until opened
// again with opted_in). Opening a withdrawn log again re-enables export of the
// records it kept, each still with the rights of the consent it was logged
// under; the replaced consent, withdrawal included, stays in consent_history. Refused unless opted_in, by, at, terms_version and
// source_id are given and rights is one of the three.
Result<void> open_correction_log(const std::filesystem::path& dir, const CorrectionConsent& consent);
Result<CorrectionConsent> read_correction_consent(const std::filesystem::path& dir);
Result<void> withdraw_correction_consent(const std::filesystem::path& dir, const std::string& by,
                                         const std::string& at);

struct CorrectionSource {
    std::filesystem::path file;   // the still or movie the frame came from
    int frame = 0;                // its frame (0 for a still)
    std::string label;            // the shot's name, if any
};

struct CorrectionEntry {
    std::string id;
    std::filesystem::path sdr, hdr, meta;
    std::string scene_id, split;
    bool already_logged = false;   // the same frame, source and grade were logged before
};

// Log one accepted frame: `sdr` the decoded frame (Rec.709 code in [0, 1]),
// `accepted` the master's picture in absolute nits (GradedMaster::nits, in
// the request's source_space, which must be rec709). Refused without an
// active consent. The id is the hash of the source file, frame and grade, so
// accepting the same thing twice logs it once.
Result<CorrectionEntry> log_correction(const std::filesystem::path& dir, const SdrImage& sdr, const NitsFrame& accepted,
                                       const MasterRequest& grade, const CorrectionSource& source);

// Every record as a manifest row; the file is written whole (refused if it
// exists). Refused when the consent is withdrawn. Returns the rows written.
Result<int> export_correction_manifest(const std::filesystem::path& dir, const std::filesystem::path& manifest);

// The 16-bit RGB PNG the log writes (zlib stored blocks, no library); empty
// when the size is not positive or rgb_hwc is not width * height * 3 samples.
std::vector<std::uint8_t> png16_rgb_bytes(int width, int height, const std::vector<std::uint16_t>& rgb_hwc);

}  // namespace rudra
