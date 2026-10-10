#pragma once
// The executables RUDRA runs (ffmpeg, ffprobe) and where they come from
// (product item 2, 10 Oct 2026). A package can carry a qualified ffmpeg
// beside the app, so movies work without the user installing one; a user's
// own build stays reachable when they ask for it.

#include <filesystem>
#include <optional>
#include <string>

namespace rudra {

// The folder the running executable is in.
std::filesystem::path executable_dir();

// Where a package keeps its own ffmpeg and ffprobe: <exe>/ffmpeg (the Windows
// package; on macOS Contents/MacOS/ffmpeg, where package_mac.sh puts them so
// they are signed as the bundle's code), else Contents/Resources/ffmpeg beside
// Contents/MacOS. Empty when the package carries none.
std::filesystem::path bundled_tools_dir(const std::filesystem::path& exe_dir);

// Puts the tools RUDRA should run first on this process's PATH, so every
// lookup and every child process finds them. RUDRA_FFMPEG_DIR names a folder
// to use instead of the bundled one; RUDRA_FFMPEG_DIR=path keeps the PATH as
// it is (the user's own ffmpeg). Returns the folder put first, if any.
std::optional<std::filesystem::path> use_bundled_tools(const std::filesystem::path& exe_dir);

}  // namespace rudra
