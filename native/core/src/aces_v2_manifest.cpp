#include "rudra/core/aces_v2_manifest.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"

namespace rudra::aces_v2 {

Result<Manifest> read_manifest(const std::filesystem::path& package_root) {
    const auto path = package_root / "manifest.json";
    std::ifstream in(path);
    if (!in) return make_error(ErrorCode::NotFound, "This folder is not a RUDRA model package.", path.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The model manifest is not valid JSON.", e.what());
    }
    Manifest m;
    m.root = package_root;
    try {
        if (j.at("contract").get<std::string>() != kContractId || j.at("contract_major").get<int>() != kContractMajor)
            return make_error(ErrorCode::ContractMismatch, "This model package was written for a different version of RUDRA.",
                              "need " + std::string(kContractId) + " major " + std::to_string(kContractMajor));
        m.g1 = package_root / j.at("g1_file").get<std::string>();
        m.g1_sha256 = j.at("g1_sha256").get<std::string>();
        m.checkpoint_sha256 = j.at("checkpoint_sha256").get<std::string>();
        if (j.contains("onnx_file") && j.at("onnx_file").is_string()) {
            m.g1_onnx = package_root / j.at("onnx_file").get<std::string>();
            m.g1_onnx_sha256 = j.at("onnx_sha256").get<std::string>();
            m.onnx_limit_display_codes = j.at("onnx_tolerance").at("limit_display_codes").get<double>();
        }
        const auto& mc = j.at("model_contract");
        m.model_contract_id = mc.at("id").get<std::string>();
        if (mc.value("full_context", std::string()) != "independent_pixel_rms")
            return make_error(ErrorCode::ContractMismatch, "This ACES package needs a newer model contract.", m.model_contract_id);
        m.ocio_version = mc.at("ocio_version").get<std::string>();
        m.ocio_config = mc.at("config").get<std::string>();
        m.ocio_view = mc.at("view").get<std::string>();
        m.ocio_display = mc.at("display").get<std::string>();
        const auto& sm = j.at("clipped_smoothing");
        if (sm.is_null()) {
            m.smoothing = "none";
        } else {
            m.smoothing = sm.at("name").get<std::string>();
            if (m.smoothing != "hybrid64" || sm.at("sigma").get<double>() != 64.0 || sm.at("ramp").get<double>() != 1.0 ||
                sm.at("start").get<double>() != 0.0 || sm.at("mode").get<std::string>() != "hybrid")
                return make_error(ErrorCode::ContractMismatch, "This package asks for a smoothing this build does not implement.",
                                  sm.dump());
        }
        const auto& ac = j.at("appearance_constraint");
        if (ac.at("max_display_error_codes").get<double>() != 1.0 || !ac.at("include_fully_clipped").get<bool>())
            return make_error(ErrorCode::ContractMismatch, "This package asks for a constraint this build does not implement.",
                              ac.dump());
        m.forbid_new_negative_y = ac.value("forbid_new_negative_y", false);
        m.parity_pass = j.at("parity_pass").get<bool>();
        m.release_authorized = j.value("release_authorized", false);
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The model manifest is missing a field.", e.what());
    }
    if (!m.parity_pass)
        return make_error(ErrorCode::ParityError, "This model package failed its export parity check.", path.string());
    return m;
}

Result<void> verify_files(const Manifest& m) {
    auto got = sha256_file(m.g1);
    if (!got) return got.error();
    if (*got != m.g1_sha256)
        return make_error(ErrorCode::IntegrityError, "The model file is damaged or was changed.", m.g1.string());
    if (!m.g1_onnx.empty()) {
        auto o = sha256_file(m.g1_onnx);
        if (!o) return o.error();
        if (*o != m.g1_onnx_sha256)
            return make_error(ErrorCode::IntegrityError, "The model file is damaged or was changed.", m.g1_onnx.string());
    }
    return {};
}

}  // namespace rudra::aces_v2
