#include "rudra/platform/io_error.hpp"

#include <cerrno>
#include <cstdio>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace rudra {

namespace fs = std::filesystem;

std::string human_bytes(std::uintmax_t bytes) {
    char buf[32];
    const double b = double(bytes);
    if (b >= 1e9) std::snprintf(buf, sizeof buf, "%.1f GB", b / 1e9);
    else if (b >= 1e6) std::snprintf(buf, sizeof buf, "%.1f MB", b / 1e6);
    else if (b >= 1e3) std::snprintf(buf, sizeof buf, "%.0f KB", b / 1e3);
    else std::snprintf(buf, sizeof buf, "%llu bytes", static_cast<unsigned long long>(bytes));
    return buf;
}

namespace {

#ifdef _WIN32
std::string last_windows_error() {
    const DWORD code = GetLastError();
    if (code == 0) return {};
    char* text = nullptr;
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                   nullptr, code, 0, reinterpret_cast<LPSTR>(&text), 0, nullptr);
    std::string s = n && text ? std::string(text, n) : "Windows error " + std::to_string(code);
    if (text) LocalFree(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s;
}
#endif

// The nearest folder that exists, for asking about free space.
fs::path existing_folder(const fs::path& target) {
    std::error_code ec;
    fs::path p = target.has_parent_path() ? target.parent_path() : fs::current_path(ec);
    while (!p.empty() && !fs::exists(p, ec)) {
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return p;
}

}  // namespace

std::string io_failure_reason(const fs::path& target, int err, std::uintmax_t bytes_needed) {
    std::error_code ec;
    const fs::path folder = existing_folder(target);
    const fs::space_info sp = folder.empty() ? fs::space_info{} : fs::space(folder, ec);
    const bool have_space = !folder.empty() && !ec;
    const std::string where = have_space ? " on " + folder.root_path().string() : std::string();
    const std::string needed = bytes_needed ? ", " + human_bytes(bytes_needed) + " needed" : std::string();

    if (err == ENOSPC || (have_space && bytes_needed && sp.available < bytes_needed))
        return "the disk is full: " + (have_space ? human_bytes(sp.available) + " free" + where : std::string("no space")) +
               needed;

    std::string reason;
    if (err == EACCES || err == EPERM) {
        reason = "access denied";
#ifdef _WIN32
        reason += " (a read-only folder, or Windows Security's Controlled folder access blocking RUDRA: "
                  "Protection history names it)";
#endif
    } else if (err == ENOENT) {
        reason = "the folder does not exist or the drive is not connected";
    } else if (err != 0) {
        reason = std::generic_category().message(err);
    } else {
#ifdef _WIN32
        reason = last_windows_error();
#endif
        if (reason.empty()) reason = "the reason was not reported";
    }
    // Space is worth saying even when it was not the stated reason: a nearly
    // full drive fails writes in ways the OS reports as other things.
    if (have_space && sp.available < (std::uintmax_t(256) << 20))
        reason += "; only " + human_bytes(sp.available) + " free" + where;
    return reason;
}

}  // namespace rudra
