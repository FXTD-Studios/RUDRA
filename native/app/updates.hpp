#pragma once
// Is there a newer RUDRA? (product item 4, 10 Oct 2026.) The releases feed is
// GitHub's (api.github.com/repos/FXTD-Studios/RUDRA/releases), read once a day
// at most and on Help > Check for updates. Nothing is downloaded or installed:
// the app says what is newer and where it is.

#include <optional>
#include <string>

namespace rudra::app {

// Semantic-version precedence (semver.org 2.0, 11): -1, 0 or 1. A leading "v"
// is ignored; "0.9.0" is newer than "0.9.0-beta.5", "beta.10" than "beta.9".
int compare_versions(const std::string& a, const std::string& b);

struct ReleaseInfo {
    std::string version;   // "0.9.0-beta.6"
    std::string url;       // the release page
    bool prerelease = false;
};

// The newest published release in a GitHub releases JSON array (drafts are
// skipped; pre-releases only when `include_prereleases`). nullopt when there
// is none or the text is not that JSON.
std::optional<ReleaseInfo> newest_release(const std::string& github_json, bool include_prereleases);

// A beta user is offered betas; a release user only releases.
inline bool is_prerelease(const std::string& version) { return version.find('-') != std::string::npos; }

}  // namespace rudra::app
