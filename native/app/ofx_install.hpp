#pragma once
// Help > Install the OFX plug-in for Resolve and Nuke (26 Nov 2026).
//
// Every RUDRA package carries the plug-in beside the app (ofx/RUDRA.ofx.bundle, and ofx/nuke for Nuke's toolbar;
// on a Mac in RUDRA.app/Contents/PlugIns/ofx, where codesign signs it with the app). Installing it is three steps, the first needing an administrator:
//   1. the bundle into the system's OFX folder, where Resolve and Nuke look (C:\Program Files\Common Files\OFX\Plugins,
//      /Library/OFX/Plugins, /usr/OFX/Plugins);
//   2. ofx_engine.json, the per-user file that tells the plug-in where this RUDRA's engine, learned model and AI fill
//      are (ofx/engine_client.cpp engine_config_path, read the same way);
//   3. Nuke's toolbar icon: ofx/nuke into ~/.nuke/RUDRA and one line in ~/.nuke/init.py.
// This file has no Qt: what to copy, what to write and the command that copies, each testable alone. The window
// (main_window.cpp install_ofx) asks, runs the command and says what happened.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rudra::app {

// The plug-in this RUDRA carries: <app>/ofx/RUDRA.ofx.bundle, or <app>/../PlugIns/ofx/RUDRA.ofx.bundle (a Mac
// app's Contents/MacOS). Empty when the package has none (a development build: tmp\BUILD_OFX.ps1).
std::filesystem::path shipped_ofx_bundle(const std::filesystem::path& app_dir);
// Nuke's toolbar files beside it (ofx/nuke: menu.py and the RUDRA mark); empty when there are none.
std::filesystem::path shipped_nuke_menu(const std::filesystem::path& app_dir);

// Where OFX hosts look for plug-ins on this system (RUDRA_OFX_PLUGIN_DIR first, for tests).
std::filesystem::path system_ofx_dir();

// The version written into a bundle at build time (Contents/Resources/version.txt); "" when it has none.
std::string ofx_bundle_version(const std::filesystem::path& bundle);

enum class OfxInstalled { None, Older, Same, Newer };
// The installed bundle against the shipped one, by their version.txt ("1.3.10" > "1.3.9").
OfxInstalled ofx_installed(const std::filesystem::path& shipped, const std::filesystem::path& installed);

// What the plug-in needs from this RUDRA. Empty paths are left out; the plug-in then says what is missing.
struct OfxEngineSettings {
    std::filesystem::path cli, package;   // rudra-native and the learned ACES 2.0 package (aces_engine_config)
    std::string runtime = "onnxruntime", device = "cpu";
    struct GenFill {
        std::filesystem::path python, repo, base, lora;   // genfill_setup(), AI fill v5 only
        std::filesystem::path florence;                    // 1.3.2: Florence-2 for Suggest prompt; empty when not set up
    };
    std::optional<GenFill> genfill;
};
// The file's text, as ofx/engine_client.cpp reads it (cli, package, runtime, device, path_prepend, genfill).
std::string ofx_engine_json(const OfxEngineSettings& s);

// The command that copies `bundle` into `dest_dir` as an administrator, replacing an older RUDRA.ofx.bundle there:
// Windows, the arguments of an elevated powershell.exe (ShellExecute "runas"); a Mac, osascript's "with
// administrator privileges"; Linux, pkexec. `error_file`: where the command writes why it failed (Windows).
std::vector<std::string> elevated_copy_command(const std::filesystem::path& bundle, const std::filesystem::path& dest_dir,
                                               const std::filesystem::path& error_file);

// The OFX hosts among running processes' names (Nuke13.2.exe, Resolve, ...): they lock a loaded plug-in.
std::vector<std::string> ofx_hosts_running(const std::vector<std::string>& process_names);

// Nuke: the line ~/.nuke/init.py needs to load ~/.nuke/RUDRA, and whether `init_text` has it.
std::string nuke_init_line();
bool nuke_init_has_rudra(const std::string& init_text);

}  // namespace rudra::app
