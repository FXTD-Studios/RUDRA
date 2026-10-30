#pragma once
// Paths between Qt and std::filesystem through UTF-16 both ways (28 Oct 2026
// review): std::string paths are the ANSI code page on Windows, so a folder
// called "Été" or "撮影" must never pass through toStdString()/string().

#include <QString>

#include <filesystem>

namespace rudra::app {

inline std::filesystem::path to_path(const QString& s) { return std::filesystem::path(s.toStdU16String()); }
inline QString from_path(const std::filesystem::path& p) { return QString::fromStdU16String(p.u16string()); }

}  // namespace rudra::app
