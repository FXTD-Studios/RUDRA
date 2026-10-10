#pragma once
// What RUDRA does before and around QApplication so that it starts the same
// on every machine and leaves a record when it does not.

#include <filesystem>
#include <string>
#include <vector>

#include <QString>

namespace rudra::app {

// The folder the running executable is in (before QApplication exists).
std::filesystem::path executable_dir();

// The platform plugins this build ships: <exe>/platforms on Windows and Linux
// packages, <bundle>/Contents/PlugIns/platforms on macOS. Empty when there is
// none beside the executable (a development build on the system's Qt).
std::filesystem::path shipped_platforms_dir(const std::filesystem::path& exe_dir);

// A packaged RUDRA carries its own Qt plugins, so Qt settings left in the
// environment by other software (a conda env with PyQt, a Qt SDK, an earlier
// test run's QT_QPA_PLATFORM=offscreen) must not steer it: on 9 Oct 2026 one
// stopped beta 3 with "no Qt platform plugin could be initialized". Call
// before QApplication. QT_PLUGIN_PATH and QT_QPA_PLATFORM_PLUGIN_PATH are
// dropped, and QT_QPA_PLATFORM too unless it names a plugin in
// `platforms_dir`. Nothing changes when `platforms_dir` is empty. Returns one
// line per variable ignored, for the log.
std::vector<std::string> ignore_foreign_qt_environment(const std::filesystem::path& platforms_dir);

// The app's log file: <dir>/rudra.log, rotated at startup to rudra.1.log ...
// rudra.<keep>.log once it passes `max_bytes`. Qt's own messages and every
// line of the window's log go to it with a time stamp. Returns the file's
// path, or empty if it could not be opened (the app runs on without one).
QString install_app_log(const QString& dir, qint64 max_bytes = 5 << 20, int keep = 3);

// One line to the log file, if one is installed; a no-op otherwise.
void app_log_line(const QString& line);

// The installed log file, or empty.
QString app_log_path();

}  // namespace rudra::app
