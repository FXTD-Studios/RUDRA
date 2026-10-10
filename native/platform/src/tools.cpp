#include "rudra/platform/tools.hpp"

#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace rudra {

namespace fs = std::filesystem;

fs::path executable_dir() {
#ifdef _WIN32
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), DWORD(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    return fs::path(buf).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    std::error_code ec;
    const fs::path p = fs::weakly_canonical(fs::path(buf.c_str()), ec);
    return (ec ? fs::path(buf.c_str()) : p).parent_path();
#else
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path() : p.parent_path();
#endif
}

namespace {

bool has_ffmpeg(const fs::path& dir) {
    std::error_code ec;
#ifdef _WIN32
    return fs::is_regular_file(dir / "ffmpeg.exe", ec) && fs::is_regular_file(dir / "ffprobe.exe", ec);
#else
    return fs::is_regular_file(dir / "ffmpeg", ec) && fs::is_regular_file(dir / "ffprobe", ec);
#endif
}

void prepend_path(const fs::path& dir) {
#ifdef _WIN32
    // Wide, so a folder with non-ASCII letters survives (the narrow CRT is the code page).
    const wchar_t* old = _wgetenv(L"PATH");
    const std::wstring value = dir.wstring() + (old && *old ? L";" + std::wstring(old) : std::wstring());
    _wputenv_s(L"PATH", value.c_str());   // the CRT's copy and the process environment both
#else
    const char* old = std::getenv("PATH");
    const std::string value = dir.string() + (old && *old ? ":" + std::string(old) : std::string());
    setenv("PATH", value.c_str(), 1);
#endif
}

// RUDRA_FFMPEG_DIR, empty when unset.
fs::path chosen_tools_dir() {
#ifdef _WIN32
    const wchar_t* v = _wgetenv(L"RUDRA_FFMPEG_DIR");
    return v && *v ? fs::path(v) : fs::path();
#else
    const char* v = std::getenv("RUDRA_FFMPEG_DIR");
    return v && *v ? fs::path(v) : fs::path();
#endif
}

}  // namespace

fs::path bundled_tools_dir(const fs::path& exe_dir) {
    if (exe_dir.empty()) return {};
    for (const fs::path& d : {exe_dir / "ffmpeg", exe_dir.parent_path() / "Resources" / "ffmpeg"})
        if (has_ffmpeg(d)) return d;
    return {};
}

std::optional<fs::path> use_bundled_tools(const fs::path& exe_dir) {
    // RUDRA_FFMPEG_DIR: a folder to use instead, or "path" for the PATH's own.
    if (const fs::path chosen = chosen_tools_dir(); !chosen.empty()) {
        if (chosen == "path") return std::nullopt;
        if (has_ffmpeg(chosen)) {
            prepend_path(chosen);
            return chosen;
        }
    }
    const fs::path dir = bundled_tools_dir(exe_dir);
    if (dir.empty()) return std::nullopt;
    prepend_path(dir);
    return dir;
}

}  // namespace rudra
