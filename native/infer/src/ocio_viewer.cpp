// The colour-managed viewer's OpenColorIO side (infer/ocio_viewer.hpp): configs, their
// displays and views, OpenColorIO's GLSL (Vulkan flavour, so QShaderBaker takes it as it
// is) and CPU processor for each transform the viewer runs.

#include "rudra/infer/ocio_viewer.hpp"

#include <OpenColorIO/OpenColorIO.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "rudra/core/gamut.hpp"

namespace OCIO = OCIO_NAMESPACE;

namespace rudra::ocio_viewer {

bool available() noexcept { return true; }

struct Config::Impl {
    OCIO::ConstConfigRcPtr config;
    ConfigSource source;
    std::string name;
    std::string aces;   // the ACES2065-1 colour space (role aces_interchange)
    std::string cg;     // the ACEScg colour space; "" when the config has none (a matrix from ACES2065-1 then)
    std::string xyz;    // the display-referred CIE XYZ D65 (role cie_xyz_d65_interchange); "" none
};

namespace {

using Cpu = std::function<void(float*, std::size_t)>;

int thread_count(int threads) {
    if (threads > 0) return std::min(threads, 16);
    return int(std::clamp(std::thread::hardware_concurrency(), 1u, 16u));
}

// OpenColorIO's CPU processor over packed RGB, split across threads for long spans
// (the pointwise transforms give the same values however they are split).
Cpu cpu_of(const OCIO::ConstProcessorRcPtr& proc, int threads) {
    OCIO::ConstCPUProcessorRcPtr cpu = proc->getDefaultCPUProcessor();
    const int t = thread_count(threads);
    return [cpu, t](float* rgb, std::size_t pixels) {
        auto run = [&cpu](float* p, std::size_t n) {
            if (n == 0) return;
            OCIO::PackedImageDesc desc(p, static_cast<long>(n), 1, 3);
            cpu->apply(desc);
        };
        if (t <= 1 || pixels < 65536) {
            run(rgb, pixels);
            return;
        }
        std::vector<std::thread> pool;
        const std::size_t k = std::size_t(t);
        for (std::size_t i = 1; i < k; ++i)
            pool.emplace_back([&, i] { run(rgb + pixels * i / k * 3, pixels * (i + 1) / k - pixels * i / k); });
        run(rgb, pixels / k);
        for (auto& th : pool) th.join();
    };
}

// One stage: OpenColorIO's GLSL for `proc` as `vec4 function(vec4)`, its textures from `binding` on.
Result<OcioStage> stage_of(const OCIO::ConstProcessorRcPtr& proc, const char* function, const char* prefix,
                           unsigned binding, int threads) {
    OcioStage st;
    st.function = function;
    OCIO::GpuShaderDescRcPtr desc = OCIO::GpuShaderDesc::CreateShaderDesc();
    desc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_VK_4_6);
    desc->setFunctionName(function);
    desc->setResourcePrefix(prefix);
    desc->setAllowTexture1D(false);            // QRhi has 2D and 3D textures everywhere, 1D not
    desc->setDescriptorSetIndex(0, binding);
    proc->getDefaultGPUProcessor()->extractGpuShaderInfo(desc);
    if (desc->getNumUniforms() > 0)
        return make_error(ErrorCode::Unsupported, "This view has live parameters, which the viewer does not run.",
                          std::string(function) + ": " + std::to_string(desc->getNumUniforms()) + " uniforms");
    st.glsl = desc->getShaderText();
    for (unsigned i = 0; i < desc->getNumTextures(); ++i) {
        const char *tex = nullptr, *sampler = nullptr;
        unsigned w = 0, h = 0;
        OCIO::GpuShaderDesc::TextureType channel{};
        OCIO::GpuShaderDesc::TextureDimensions dims{};
        OCIO::Interpolation interp{};
        desc->getTexture(i, tex, sampler, w, h, channel, dims, interp);
        const float* values = nullptr;
        desc->getTextureValues(i, values);
        OcioTexture t;
        t.sampler = sampler ? sampler : "";
        t.binding = int(desc->getTextureShaderBindingIndex(i));
        t.width = int(w);
        t.height = int(std::max(h, 1u));
        t.channels = channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1 : 3;
        t.linear = interp != OCIO::INTERP_NEAREST;
        const std::size_t n = std::size_t(t.width) * std::size_t(t.height) * std::size_t(t.channels);
        if (values) t.values.assign(values, values + n);
        st.textures.push_back(std::move(t));
    }
    for (unsigned i = 0; i < desc->getNum3DTextures(); ++i) {
        const char *tex = nullptr, *sampler = nullptr;
        unsigned edge = 0;
        OCIO::Interpolation interp{};
        desc->get3DTexture(i, tex, sampler, edge, interp);
        const float* values = nullptr;
        desc->get3DTextureValues(i, values);
        OcioTexture t;
        t.sampler = sampler ? sampler : "";
        t.binding = int(desc->get3DTextureShaderBindingIndex(i));
        t.width = t.height = t.depth = int(edge);
        t.channels = 3;
        t.linear = interp != OCIO::INTERP_NEAREST;
        const std::size_t n = std::size_t(edge) * edge * edge * 3;
        if (values) t.values.assign(values, values + n);
        st.textures.push_back(std::move(t));
    }
    st.cpu = cpu_of(proc, threads);
    return st;
}

// AP1 -> AP0 (or back) as OpenColorIO's 4x4, for a config without an ACEScg space.
OCIO::MatrixTransformRcPtr ap_matrix(bool ap1_to_ap0) {
    const Mat3 m = ap1_to_ap0 ? rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Ap0) : rgb_to_rgb_matrix(Primaries::Ap0, Primaries::Ap1);
    double m44[16] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m44[r * 4 + c] = m[std::size_t(r)][std::size_t(c)];
    m44[15] = 1.0;
    auto t = OCIO::MatrixTransform::Create();
    t->setMatrix(m44);
    return t;
}

std::string first_space(const OCIO::ConstConfigRcPtr& c, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (auto cs = c->getColorSpace(n)) return cs->getName();
    return {};
}

bool display_referred(const OCIO::ConstConfigRcPtr& c, const std::string& name) {
    if (name.empty()) return false;
    auto cs = c->getColorSpace(name.c_str());
    return cs && !cs->isData() && cs->getReferenceSpaceType() == OCIO::REFERENCE_SPACE_DISPLAY;
}

// The colour space a display's codes are in, for the decode to light: the view's own
// when it is display referred, else (Raw, a scene-referred view) the display's.
std::string display_space(const OCIO::ConstConfigRcPtr& c, const std::string& display, const std::string& view) {
    auto resolve = [&](const char* cs) {
        std::string n = cs ? cs : "";
        return n == "<USE_DISPLAY_NAME>" ? display : n;
    };
    const std::string own = resolve(c->getDisplayViewColorSpaceName(display.c_str(), view.c_str()));
    if (display_referred(c, own)) return own;
    if (display_referred(c, display)) return display;
    for (int i = 0; i < c->getNumViews(display.c_str()); ++i) {
        const std::string n = resolve(c->getDisplayViewColorSpaceName(display.c_str(), c->getView(display.c_str(), i)));
        if (display_referred(c, n)) return n;
    }
    return {};
}

}  // namespace

Result<std::shared_ptr<const Config>> Config::open(const ConfigSource& source) {
    auto d = std::make_shared<Impl>();
    d->source = source;
    try {
        switch (source.kind) {
            case ConfigKind::Builtin:
                d->config = OCIO::Config::CreateFromBuiltinConfig(kBuiltinConfig);
                d->name = std::string("ACES 2.0 studio config (built in) · OpenColorIO ") + OCIO::GetVersion();
                break;
            case ConfigKind::Environment: {
                const char* env = std::getenv("OCIO");
                if (!env || !*env)
                    return make_error(ErrorCode::NotFound, "The OCIO environment variable is not set on this computer.");
                d->config = OCIO::Config::CreateFromEnv();
                d->name = std::string("$OCIO · ") + env;
                break;
            }
            case ConfigKind::File:
                if (source.path.empty()) return make_error(ErrorCode::InvalidArgument, "No config file was chosen.");
                d->config = OCIO::Config::CreateFromFile(source.path.c_str());
                d->name = source.path;
                break;
        }
        d->config->validate();
    } catch (const OCIO::Exception& e) {
        return make_error(ErrorCode::ParseError, "OpenColorIO could not open this config.", e.what());
    }
    const auto& c = d->config;
    if (auto cs = c->getColorSpace(OCIO::ROLE_INTERCHANGE_SCENE)) d->aces = cs->getName();
    if (d->aces.empty()) d->aces = first_space(c, {"ACES2065-1", "ACES - ACES2065-1", "aces2065_1", "ACES2065_1"});
    d->cg = first_space(c, {"ACEScg", "ACES - ACEScg", "acescg"});
    if (auto cs = c->getColorSpace(OCIO::ROLE_INTERCHANGE_DISPLAY)) d->xyz = cs->getName();
    if (d->aces.empty() && d->cg.empty())
        return make_error(ErrorCode::Unsupported,
                          "This config has no ACES2065-1 or ACEScg colour space, so RUDRA cannot hand it the picture.",
                          "need role aces_interchange or a colour space named ACES2065-1 or ACEScg");
    return std::shared_ptr<const Config>(std::make_shared<Config>(std::move(d)));
}

const std::string& Config::name() const { return d_->name; }
const ConfigSource& Config::source() const { return d_->source; }

std::vector<std::string> Config::displays() const {
    std::vector<std::string> out;
    for (int i = 0; i < d_->config->getNumDisplays(); ++i) out.emplace_back(d_->config->getDisplay(i));
    return out;
}

std::vector<std::string> Config::views(const std::string& display) const {
    std::vector<std::string> out;
    for (int i = 0; i < d_->config->getNumViews(display.c_str()); ++i) out.emplace_back(d_->config->getView(display.c_str(), i));
    return out;
}

std::vector<ViewChoice> Config::presets() const {
    static const ViewChoice kWanted[] = {
        {"SDR · sRGB · ACES 2.0 100 nits", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"},
        {"SDR · Rec.709 (BT.1886) · ACES 2.0 100 nits", "Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"},
        {"HDR · Rec.2100 PQ · ACES 2.0 1000 nits (P3-D65)", "Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"},
        {"HDR · Rec.2100 PQ · ACES 2.0 1000 nits (Rec.2020)", "Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (Rec.2020)"},
        {"P3-D65 · ACES 2.0 100 nits (cinema, gamma 2.6)", "P3-D65 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"},
        {"Display P3 · ACES 2.0 100 nits", "Display P3 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"},
        {"Raw · the working-space values (sRGB display)", "sRGB - Display", "Raw"},
    };
    std::vector<ViewChoice> out;
    auto has = [&](const std::string& display, const std::string& view) {
        for (const std::string& v : views(display))
            if (v == view) return true;
        return false;
    };
    for (const ViewChoice& w : kWanted)
        if (has(w.display, w.view)) out.push_back(w);
    if (out.empty() && d_->config->getNumDisplays() > 0) {
        const std::string display = d_->config->getDefaultDisplay();
        const std::string view = d_->config->getDefaultView(display.c_str());
        out.push_back({"Config default · " + display + " · " + view, display, view});
    }
    return out;
}

std::vector<InputChoice> Config::inputs() const {
    std::vector<InputChoice> out;
    const auto& c = d_->config;
    auto has_view = [&](const char* display, const char* view) {
        for (int i = 0; i < c->getNumViews(display); ++i)
            if (std::strcmp(c->getView(display, i), view) == 0) return true;
        return false;
    };
    struct Inv { const char *display, *view, *label; };
    static const Inv kInverse[] = {
        {"sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", "Inverse ACES 2.0 SDR view, sRGB display (looks as graded)"},
        {"Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", "Inverse ACES 2.0 SDR view, Rec.709 BT.1886 display"},
        {"Gamma 2.2 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", "Inverse ACES 2.0 SDR view, gamma 2.2 display"},
    };
    for (const Inv& i : kInverse)
        if (has_view(i.display, i.view)) out.push_back({std::string("inverse:") + i.display + "\t" + i.view, i.label});
    struct Sp { const char *name, *label; };
    static const Sp kSpaces[] = {
        {"sRGB Encoded Rec.709 (sRGB)", "sRGB texture: the codes as linear light (no tone curve)"},
        {"Gamma 2.4 Encoded Rec.709", "Rec.709 gamma 2.4: the codes as linear light"},
        {"Gamma 2.2 Encoded Rec.709", "Rec.709 gamma 2.2: the codes as linear light"},
    };
    for (const Sp& s : kSpaces)
        if (c->getColorSpace(s.name)) out.push_back({std::string("space:") + s.name, s.label});
    if (out.empty()) {
        // Another config: the inverse of its default view, then every colour space with pixels in it.
        if (c->getNumDisplays() > 0) {
            const std::string display = c->getDefaultDisplay();
            const std::string view = c->getDefaultView(display.c_str());
            out.push_back({"inverse:" + display + "\t" + view, "Inverse " + view + " (" + display + ")"});
        }
        for (int i = 0; i < c->getNumColorSpaces(); ++i) {
            const char* n = c->getColorSpaceNameByIndex(i);
            auto cs = c->getColorSpace(n);
            if (cs && !cs->isData()) out.push_back({std::string("space:") + n, n});
        }
    }
    return out;
}

Result<std::shared_ptr<const OcioViewTransform>> Config::view(const std::string& display, const std::string& view,
                                                             WorkingSpace working, int threads) const {
    const auto& c = d_->config;
    auto t = std::make_shared<OcioViewTransform>();
    t->config = d_->name;
    t->display = display;
    t->view = view;
    t->working = working;
    t->id = d_->name + "|" + display + "|" + view + "|" + working_name(working) + "|" + OCIO::GetVersion();
    try {
        // From the working space: the config's own ACEScg or ACES2065-1, so the picture is what
        // Nuke shows for an EXR read as that space; a matrix where the config lacks one.
        auto dvt = OCIO::DisplayViewTransform::Create();
        dvt->setDisplay(display.c_str());
        dvt->setView(view.c_str());
        auto group = OCIO::GroupTransform::Create();
        if (working == WorkingSpace::AcesCg) {
            if (!d_->cg.empty()) {
                dvt->setSrc(d_->cg.c_str());
            } else {
                group->appendTransform(ap_matrix(true));
                dvt->setSrc(d_->aces.c_str());
            }
        } else {
            if (!d_->aces.empty()) {
                dvt->setSrc(d_->aces.c_str());
            } else {
                group->appendTransform(ap_matrix(false));
                dvt->setSrc(d_->cg.c_str());
            }
        }
        group->appendTransform(dvt);
        auto forward = stage_of(c->getProcessor(group), "ocioView", "ocioV", kFirstOcioBinding, threads);
        if (!forward) return forward.error();
        t->forward = std::move(forward.value());
        // The codes back to light, for HDR swapchains.
        const std::string space = display_space(c, display, view);
        if (!space.empty() && !d_->xyz.empty()) {
            const unsigned next = kFirstOcioBinding + unsigned(t->forward.textures.size());
            auto decode = stage_of(c->getProcessor(space.c_str(), d_->xyz.c_str()), "ocioDecode", "ocioD", next, threads);
            if (decode) {
                t->decode = std::move(decode.value());
                t->decodes = true;
            }
            auto cs = c->getColorSpace(space.c_str());
            const std::string enc = cs && cs->getEncoding() ? cs->getEncoding() : "";
            std::string lower = space;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
            const bool named_hdr = lower.find("pq") != std::string::npos || lower.find("2084") != std::string::npos ||
                                   lower.find("hlg") != std::string::npos;
            if (enc == "hdr-video" || enc == "edr-video" || (enc.empty() && named_hdr)) {
                t->hdr = true;
                t->nits_per_y = 100.0;   // OpenColorIO's display-referred XYZ: 1.0 is 100 nits
            }
            if (enc == "edr-video") t->clip_code = 1e30;   // codes above 1 are the extended range
        }
    } catch (const OCIO::Exception& e) {
        return make_error(ErrorCode::InvalidArgument, "OpenColorIO cannot make this view.", e.what());
    }
    return std::shared_ptr<const OcioViewTransform>(std::move(t));
}

Result<std::shared_ptr<const OcioInputTransform>> Config::input(const std::string& id, WorkingSpace working,
                                                               int threads) const {
    const auto& c = d_->config;
    auto t = std::make_shared<OcioInputTransform>();
    t->id = d_->name + "|" + id + "|" + working_name(working);
    t->working = working;
    try {
        // To the working space: the config's own when it has it, else through the other and a matrix.
        const bool cg = working == WorkingSpace::AcesCg;
        const std::string target = cg ? (d_->cg.empty() ? d_->aces : d_->cg) : (d_->aces.empty() ? d_->cg : d_->aces);
        const bool matrix_after = cg ? d_->cg.empty() : d_->aces.empty();
        auto group = OCIO::GroupTransform::Create();
        if (id.rfind("inverse:", 0) == 0) {
            const std::string rest = id.substr(8);
            const auto tab = rest.find('\t');
            if (tab == std::string::npos) return make_error(ErrorCode::InvalidArgument, "Unknown input transform.", id);
            auto dvt = OCIO::DisplayViewTransform::Create();
            dvt->setSrc(target.c_str());
            dvt->setDisplay(rest.substr(0, tab).c_str());
            dvt->setView(rest.substr(tab + 1).c_str());
            dvt->setDirection(OCIO::TRANSFORM_DIR_INVERSE);
            group->appendTransform(dvt);
            t->name = "Inverse " + rest.substr(tab + 1) + " (" + rest.substr(0, tab) + ")";
        } else if (id.rfind("space:", 0) == 0) {
            auto cst = OCIO::ColorSpaceTransform::Create();
            cst->setSrc(id.substr(6).c_str());
            cst->setDst(target.c_str());
            group->appendTransform(cst);
            t->name = id.substr(6);
        } else {
            return make_error(ErrorCode::InvalidArgument, "Unknown input transform.", id);
        }
        if (matrix_after) group->appendTransform(ap_matrix(!cg));
        t->cpu = cpu_of(c->getProcessor(group), threads);
    } catch (const OCIO::Exception& e) {
        return make_error(ErrorCode::InvalidArgument, "OpenColorIO cannot make this input transform.", e.what());
    }
    return std::shared_ptr<const OcioInputTransform>(std::move(t));
}

}  // namespace rudra::ocio_viewer
