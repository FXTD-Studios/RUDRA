#include "ofx_install.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace rudra::app {

namespace fs = std::filesystem;

namespace {

std::string utf8(const fs::path& p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

fs::path env_path(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? fs::path(v) : fs::path();
}

bool is_dir(const fs::path& p) {
    std::error_code ec;
    return !p.empty() && fs::is_directory(p, ec);
}

std::string lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// A path in single quotes for PowerShell ('' for ').
[[maybe_unused]] std::string ps_quote(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) out += c == '\'' ? std::string("''") : std::string(1, c);
    return out + "'";
}

// A path in single quotes for sh ('\'' for ').
[[maybe_unused]] std::string sh_quote(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

// A string inside AppleScript's double quotes.
[[maybe_unused]] std::string applescript_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
}

std::vector<std::string> version_parts(const std::string& v) {
    std::vector<std::string> parts;
    std::stringstream ss(v);
    for (std::string part; std::getline(ss, part, '.');) parts.push_back(part);
    return parts;
}

// <0, 0, >0 as a < b, a == b, a > b: number by number, text where a part is not a number.
int compare_versions(const std::string& a, const std::string& b) {
    const auto pa = version_parts(a), pb = version_parts(b);
    for (std::size_t i = 0; i < std::max(pa.size(), pb.size()); ++i) {
        const std::string x = i < pa.size() ? pa[i] : "0", y = i < pb.size() ? pb[i] : "0";
        const bool nx = !x.empty() && std::all_of(x.begin(), x.end(), ::isdigit);
        const bool ny = !y.empty() && std::all_of(y.begin(), y.end(), ::isdigit);
        if (nx && ny) {
            const long long u = std::stoll(x), w = std::stoll(y);
            if (u != w) return u < w ? -1 : 1;
        } else if (x != y) {
            return x < y ? -1 : 1;
        }
    }
    return 0;
}

}  // namespace

fs::path shipped_ofx_bundle(const fs::path& app_dir) {
    for (const fs::path& p : {app_dir / "ofx" / "RUDRA.ofx.bundle", app_dir / ".." / "PlugIns" / "ofx" / "RUDRA.ofx.bundle"})
        if (is_dir(p / "Contents")) return p.lexically_normal();
    return {};
}

fs::path shipped_nuke_menu(const fs::path& app_dir) {
    std::error_code ec;
    for (const fs::path& p : {app_dir / "ofx" / "nuke", app_dir / ".." / "PlugIns" / "ofx" / "nuke"})
        if (fs::is_regular_file(p / "menu.py", ec)) return p.lexically_normal();
    return {};
}

fs::path system_ofx_dir() {
    if (fs::path p = env_path("RUDRA_OFX_PLUGIN_DIR"); !p.empty()) return p;
#ifdef _WIN32
    const fs::path common = env_path("CommonProgramFiles");
    return (common.empty() ? fs::path("C:\\Program Files\\Common Files") : common) / "OFX" / "Plugins";
#elif defined(__APPLE__)
    return "/Library/OFX/Plugins";
#else
    return "/usr/OFX/Plugins";
#endif
}

std::string ofx_bundle_version(const fs::path& bundle) {
    std::ifstream f(bundle / "Contents" / "Resources" / "version.txt");
    std::string v;
    std::getline(f, v);
    while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back();
    return v;
}

OfxInstalled ofx_installed(const fs::path& shipped, const fs::path& installed) {
    if (!is_dir(installed / "Contents")) return OfxInstalled::None;
    const std::string have = ofx_bundle_version(installed), ours = ofx_bundle_version(shipped);
    if (have.empty()) return OfxInstalled::Older;   // a build from before version.txt (1.3.1's tests)
    const int c = compare_versions(have, ours);
    return c < 0 ? OfxInstalled::Older : c == 0 ? OfxInstalled::Same : OfxInstalled::Newer;
}

std::string ofx_engine_json(const OfxEngineSettings& s) {
    nlohmann::ordered_json j;
    j["cli"] = utf8(s.cli);
    j["package"] = utf8(s.package);
    j["runtime"] = s.runtime;
    j["device"] = s.device;
    // The engine runs from its own folder, which holds its DLLs (an installed RUDRA's): nothing to prepend.
    j["path_prepend"] = nlohmann::json::array();
    if (s.genfill) {
        j["genfill"] = {{"python", utf8(s.genfill->python)},
                        {"repo", utf8(s.genfill->repo)},
                        {"base", utf8(s.genfill->base)},
                        {"lora", utf8(s.genfill->lora)}};
        if (!s.genfill->florence.empty()) j["genfill"]["florence"] = utf8(s.genfill->florence);   // Suggest prompt
    }
    j["written_by"] = "RUDRA, Help > Install the OFX plug-in";
    return j.dump(2) + "\n";
}

std::vector<std::string> elevated_copy_command(const fs::path& bundle, const fs::path& dest_dir, const fs::path& error_file) {
    const fs::path target = dest_dir / "RUDRA.ofx.bundle";
#ifdef _WIN32
    (void)target;
    const std::string script =
        "$ErrorActionPreference='Stop'; try { New-Item -ItemType Directory -Force -Path " + ps_quote(dest_dir) +
        " | Out-Null; $t = Join-Path " + ps_quote(dest_dir) +
        " 'RUDRA.ofx.bundle'; if (Test-Path -LiteralPath $t) { Remove-Item -LiteralPath $t -Recurse -Force }; "
        "Copy-Item -LiteralPath " + ps_quote(bundle) + " -Destination " + ps_quote(dest_dir) +
        " -Recurse -Force } catch { $_ | Out-String | Set-Content -Encoding UTF8 -LiteralPath " + ps_quote(error_file) +
        "; exit 1 }";
    return {"powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script};
#else
    (void)error_file;
    const std::string script = "mkdir -p " + sh_quote(dest_dir) + " && rm -rf " + sh_quote(target) + " && cp -R " +
                               sh_quote(bundle) + " " + sh_quote(dest_dir) + "/";
#ifdef __APPLE__
    return {"osascript", "-e", "do shell script " + applescript_string(script) + " with administrator privileges"};
#else
    return {"pkexec", "/bin/sh", "-c", script};
#endif
#endif
}

std::vector<std::string> ofx_hosts_running(const std::vector<std::string>& process_names) {
    static const char* const kHosts[] = {"nuke", "resolve", "fuscript", "fusion", "natron", "flame", "baselight", "vegas", "afterfx"};
    std::vector<std::string> out;
    for (const auto& name : process_names) {
        const std::string n = lower(fs::path(name).filename().string());
        for (const char* h : kHosts)
            if (n.rfind(h, 0) == 0 && std::find(out.begin(), out.end(), name) == out.end()) {
                out.push_back(name);
                break;
            }
    }
    return out;
}

std::string nuke_init_line() { return "nuke.pluginAddPath(\"./RUDRA\")  # RUDRA's toolbar icon (Help > Install the OFX plug-in)"; }

bool nuke_init_has_rudra(const std::string& init_text) { return init_text.find("pluginAddPath(\"./RUDRA\")") != std::string::npos; }

}  // namespace rudra::app
