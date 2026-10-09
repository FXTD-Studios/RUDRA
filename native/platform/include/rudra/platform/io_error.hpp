#pragma once
// Why a file could not be written, in words a user can act on.
//
// Every writer used to say only "The EXR file could not be written." A master
// that failed on 9 Oct 2026 left nothing to tell a full disk from a protected
// folder from a vanished drive. This names the operating system's reason and,
// when space is the problem or close to it, how much is free and how much the
// file needed.

#include <cstdint>
#include <filesystem>
#include <string>

namespace rudra {

// `err` is errno as it was right after the failure (capture it before anything
// else runs). `bytes_needed` is the size of what was being written, 0 if not
// known. Never empty: "the reason was not reported" when the OS gave none.
std::string io_failure_reason(const std::filesystem::path& target, int err, std::uintmax_t bytes_needed = 0);

// "1.5 GB", "52.0 MB", "812 KB", "17 bytes".
std::string human_bytes(std::uintmax_t bytes);

}  // namespace rudra
