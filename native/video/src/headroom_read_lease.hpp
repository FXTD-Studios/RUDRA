#pragma once

#include "rudra/video/headroom_source.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rudra::headroom_detail {
namespace fs = std::filesystem;
// Deny writes/deletion for the complete identity/probe/decode/cache lifetime.
// Metadata-only polling cannot detect a same-size, same-mtime replacement.
// A path check also detects a retargeted ancestor/junction. No source writes.
class ReadLease {
public:
    ReadLease() = default;
    ReadLease(const ReadLease&) = delete;
    ReadLease& operator=(const ReadLease&) = delete;
    ~ReadLease() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#endif
    }
    Result<void> open(const fs::path& file) {
        std::error_code ec;
        path = fs::canonical(file, ec);
        if (ec) return make_error(ErrorCode::NotFound, "Cannot resolve headroom input", ec.message());
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            return make_error(ErrorCode::Busy, "Cannot protect headroom input from changes", path.string());
        if (!GetFileInformationByHandle(handle_, &identity_) ||
            (identity_.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            return make_error(ErrorCode::IoError, "Cannot identify headroom input", path.string());
        return {};
#else
        return make_error(ErrorCode::Unsupported, "Headroom source read leases require Windows in this build");
#endif
    }
    Result<void> verify() const {
#ifdef _WIN32
        const auto check = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);
        BY_HANDLE_FILE_INFORMATION now{};
        const bool read = check != INVALID_HANDLE_VALUE && GetFileInformationByHandle(check, &now);
        if (check != INVALID_HANDLE_VALUE) CloseHandle(check);
        if (read && now.dwVolumeSerialNumber == identity_.dwVolumeSerialNumber &&
            now.nFileIndexHigh == identity_.nFileIndexHigh && now.nFileIndexLow == identity_.nFileIndexLow &&
            now.nFileSizeHigh == identity_.nFileSizeHigh && now.nFileSizeLow == identity_.nFileSizeLow &&
            CompareFileTime(&now.ftLastWriteTime, &identity_.ftLastWriteTime) == 0) return {};
#endif
        return make_error(ErrorCode::IntegrityError, "Headroom input changed; reopen the source", path.string());
    }
    fs::path path;
private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    BY_HANDLE_FILE_INFORMATION identity_{};
#endif
};

}  // namespace rudra::headroom_detail
