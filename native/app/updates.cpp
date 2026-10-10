#include "updates.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

#include <nlohmann/json.hpp>

namespace rudra::app {

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

bool numeric(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

int cmp_numeric(const std::string& a, const std::string& b) {
    // Without converting: leading zeros aside, a longer digit string is larger.
    std::string x = a, y = b;
    x.erase(0, std::min(x.find_first_not_of('0'), x.size()));
    y.erase(0, std::min(y.find_first_not_of('0'), y.size()));
    if (x.size() != y.size()) return x.size() < y.size() ? -1 : 1;
    return x < y ? -1 : x > y ? 1 : 0;
}

}  // namespace

int compare_versions(const std::string& a_in, const std::string& b_in) {
    auto clean = [](std::string v) {
        if (!v.empty() && (v[0] == 'v' || v[0] == 'V')) v.erase(0, 1);
        if (const auto plus = v.find('+'); plus != std::string::npos) v.erase(plus);   // build metadata
        return v;
    };
    const std::string a = clean(a_in), b = clean(b_in);
    const auto da = a.find('-'), db = b.find('-');
    const auto core_a = split(a.substr(0, da), '.'), core_b = split(b.substr(0, db), '.');
    for (std::size_t i = 0; i < std::max(core_a.size(), core_b.size()); ++i) {
        const std::string x = i < core_a.size() ? core_a[i] : "0", y = i < core_b.size() ? core_b[i] : "0";
        if (const int c = cmp_numeric(numeric(x) ? x : "0", numeric(y) ? y : "0"); c) return c;
    }
    const bool pa = da != std::string::npos, pb = db != std::string::npos;
    if (!pa || !pb) return pa == pb ? 0 : (pa ? -1 : 1);   // a release is newer than its pre-releases
    const auto ia = split(a.substr(da + 1), '.'), ib = split(b.substr(db + 1), '.');
    for (std::size_t i = 0; i < std::min(ia.size(), ib.size()); ++i) {
        const bool na = numeric(ia[i]), nb = numeric(ib[i]);
        int c = 0;
        if (na && nb) c = cmp_numeric(ia[i], ib[i]);
        else if (na != nb) c = na ? -1 : 1;   // numeric identifiers sort first
        else c = ia[i] < ib[i] ? -1 : ia[i] > ib[i] ? 1 : 0;
        if (c) return c;
    }
    return ia.size() == ib.size() ? 0 : (ia.size() < ib.size() ? -1 : 1);
}

std::optional<ReleaseInfo> newest_release(const std::string& text, bool include_prereleases) {
    nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_array()) return std::nullopt;
    std::optional<ReleaseInfo> best;
    for (const auto& r : j) {
        if (!r.is_object() || r.value("draft", false)) continue;
        const bool pre = r.value("prerelease", false);
        if (pre && !include_prereleases) continue;
        std::string tag = r.value("tag_name", std::string());
        if (tag.empty()) continue;
        if (tag[0] == 'v' || tag[0] == 'V') tag.erase(0, 1);
        if (tag.empty() || !std::isdigit(static_cast<unsigned char>(tag[0]))) continue;   // not a version tag
        if (!best || compare_versions(tag, best->version) > 0) best = ReleaseInfo{tag, r.value("html_url", std::string()), pre};
    }
    return best;
}

}  // namespace rudra::app
