#pragma once
// ACES Metadata File (AMF) v2.0 for RUDRA's learned ACES 2.0 masters
// (1 Nov 2026): the interchange sidecar Resolve, Nuke and Baselight read to set
// up the viewing pipeline. Checked against the Academy's schema
// (github.com/ampas/aces-amf, schema/acesMetadataFile.xsd; vendored in
// tests/data/aces_amf for the test that validates every file RUDRA writes).
//
// A master is scene-linear, so nothing is baked in: both transforms are
// applied="false". ACES2065-1 masters need no input transform; ACEScg masters
// get the Academy's ACEScg-to-ACES conversion. The output transform is the one
// RUDRA's appearance constraint holds the master to: ACES 2.0 SDR, 100 nits,
// Rec.709 primaries, BT.1886.
#include <cstdint>
#include <optional>
#include <random>
#include <string>

namespace rudra {

inline constexpr const char* kAmfNamespace = "urn:ampas:aces:amf:v2.0";
inline constexpr const char* kAcesCgToAcesId = "urn:ampas:aces:transformId:v2.0:CSC.Academy.ACEScg_to_ACES.a2.v1";
inline constexpr const char* kAcesRec709Bt1886OutputId =
    "urn:ampas:aces:transformId:v2.0:Output.Academy.Rec709-D65_100nit_in_Rec709-D65_BT1886.a2.v1";

struct AmfSequence {
    std::string pattern;   // e.g. "shot010.######.exr", '#' the frame digits
    std::int64_t first = 0, last = 0;
};

struct AmfRequest {
    std::string description;            // amfInfo description
    // The schema's author needs a name AND an email; without an email none is written.
    std::string author_name, author_email;
    std::string created_utc;            // ISO 8601, e.g. 2026-11-02T10:15:00Z
    std::string amf_uuid, pipeline_uuid;   // urn:uuid:... (make_amf_uuid)
    bool acescg = true;                 // false: ACES2065-1 (AP0), no input transform
    std::optional<std::string> clip_name;   // with sequence or file: the clipId
    std::optional<AmfSequence> sequence;
    std::optional<std::string> file;
};

// The AMF document, UTF-8, every text value XML-escaped.
std::string aces_amf_xml(const AmfRequest& r);
// A random (version 4) UUID as urn:uuid:..., the schema's pattern.
std::string make_amf_uuid(std::mt19937_64& rng);
// Now in UTC as ISO 8601 with a Z.
std::string amf_now_utc();

}  // namespace rudra
