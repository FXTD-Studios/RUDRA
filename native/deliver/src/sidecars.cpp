#include "rudra/deliver/sidecars.hpp"

#include <cerrno>
#include <fstream>

#include "rudra/core/metadata.hpp"
#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/io_error.hpp"

namespace rudra {

Result<void> write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    errno = 0;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        const int err = errno;
        return make_error(ErrorCode::IoError, "The file could not be created (" + io_failure_reason(path, err, text.size()) + ").",
                          path.string());
    }
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    f.close();   // the last bytes land on close: check it, not the destructor
    if (!f) {
        const int err = errno;
        return make_error(ErrorCode::IoError, "The file could not be written (" + io_failure_reason(path, err, text.size()) + ").",
                          path.string());
    }
    return {};
}

Result<SidecarPaths> write_all_sidecars(std::span<const FrameStats> stats, const std::filesystem::path& stem,
                                        double peak, double threshold) {
    if (stats.empty()) return make_error(ErrorCode::InvalidArgument, "no frames analyzed");
    const auto shots = detect_shots(stats, threshold);
    const auto base = stem.filename().string();
    SidecarPaths p{stem.parent_path() / (base + "_hdr_analysis.json"), stem.parent_path() / (base + "_dovi_generate.json"),
                   stem.parent_path() / (base + "_hdr10plus_scenes.json")};
    if (auto r = write_text_file(p.rudra, pyjson::dumps(to_rudra_sidecar(stats, shots), 2)); !r) return r.error();
    if (auto r = write_text_file(p.dovi, pyjson::dumps(to_dovi_generate_json(stats, shots, peak), 2)); !r) return r.error();
    if (auto r = write_text_file(p.hdr10plus, pyjson::dumps(to_hdr10plus_json(stats, shots), 2)); !r) return r.error();
    return p;
}

}  // namespace rudra
