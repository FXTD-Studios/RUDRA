#include "rudra/core/aces_amf.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>

namespace rudra {
namespace {

std::string esc(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': o += "&amp;"; break;
            case '<': o += "&lt;"; break;
            case '>': o += "&gt;"; break;
            case '"': o += "&quot;"; break;
            case '\'': o += "&apos;"; break;
            default: o += c;
        }
    }
    return o;
}

void datetime(std::string& x, const std::string& indent, const std::string& when) {
    x += indent + "<aces:dateTime>\n";
    x += indent + "    <aces:creationDateTime>" + esc(when) + "</aces:creationDateTime>\n";
    x += indent + "    <aces:modificationDateTime>" + esc(when) + "</aces:modificationDateTime>\n";
    x += indent + "</aces:dateTime>\n";
}

}  // namespace

std::string aces_amf_xml(const AmfRequest& r) {
    std::string x = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    x += std::string("<aces:acesMetadataFile xmlns:aces=\"") + kAmfNamespace +
         "\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:cdl=\"urn:ASC:CDL:v1.01\" version=\"2.0\">\n";
    x += "    <aces:amfInfo>\n";
    x += "        <aces:description>" + esc(r.description) + "</aces:description>\n";
    if (!r.author_name.empty() && !r.author_email.empty())
        x += "        <aces:author>\n            <aces:name>" + esc(r.author_name) + "</aces:name>\n            <aces:emailAddress>" +
             esc(r.author_email) + "</aces:emailAddress>\n        </aces:author>\n";
    datetime(x, "        ", r.created_utc);
    x += "        <aces:uuid>" + esc(r.amf_uuid) + "</aces:uuid>\n";
    x += "    </aces:amfInfo>\n";
    if (r.clip_name && (r.sequence || r.file)) {
        x += "    <aces:clipId>\n        <aces:clipName>" + esc(*r.clip_name) + "</aces:clipName>\n";
        if (r.sequence)
            x += "        <aces:sequence idx=\"#\" min=\"" + std::to_string(r.sequence->first) + "\" max=\"" +
                 std::to_string(r.sequence->last) + "\">" + esc(r.sequence->pattern) + "</aces:sequence>\n";
        else
            x += "        <aces:file>" + esc(*r.file) + "</aces:file>\n";
        x += "    </aces:clipId>\n";
    }
    x += "    <aces:pipeline>\n        <aces:pipelineInfo>\n";
    datetime(x, "            ", r.created_utc);
    x += "            <aces:uuid>" + esc(r.pipeline_uuid) + "</aces:uuid>\n";
    x += "            <aces:systemVersion>\n                <aces:majorVersion>2</aces:majorVersion>\n"
         "                <aces:minorVersion>0</aces:minorVersion>\n                <aces:patchVersion>0</aces:patchVersion>\n"
         "            </aces:systemVersion>\n        </aces:pipelineInfo>\n";
    if (r.acescg)
        x += std::string("        <aces:inputTransform applied=\"false\">\n            <aces:description>ACEScg to ACES2065-1 "
                         "(the master is ACEScg)</aces:description>\n            <aces:transformId>") +
             kAcesCgToAcesId + "</aces:transformId>\n        </aces:inputTransform>\n";
    x += std::string("        <aces:outputTransform applied=\"false\">\n            <aces:description>ACES 2.0 SDR 100 nits "
                     "Rec.709 BT.1886: the view RUDRA's appearance constraint holds the master to</aces:description>\n"
                     "            <aces:transformId>") +
         kAcesRec709Bt1886OutputId + "</aces:transformId>\n        </aces:outputTransform>\n";
    x += "    </aces:pipeline>\n</aces:acesMetadataFile>\n";
    return x;
}

std::string make_amf_uuid(std::mt19937_64& rng) {
    std::uint64_t a = rng(), b = rng();
    a = (a & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;   // version 4
    b = (b & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;   // RFC 4122 variant
    char s[64];
    std::snprintf(s, sizeof s, "urn:uuid:%08x-%04x-%04x-%04x-%012llx", unsigned(a >> 32), unsigned((a >> 16) & 0xFFFF),
                  unsigned(a & 0xFFFF), unsigned(b >> 48), static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFull));
    return s;
}

std::string amf_now_utc() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm g{};
#ifdef _WIN32
    gmtime_s(&g, &t);
#else
    gmtime_r(&t, &g);
#endif
    char s[32];
    std::strftime(s, sizeof s, "%Y-%m-%dT%H:%M:%SZ", &g);
    return s;
}

}  // namespace rudra
