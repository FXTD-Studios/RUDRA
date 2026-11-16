#pragma once
// The colour-managed viewer's OpenColorIO side (17 Nov 2026, docs/view.spec.md revision 7):
// a config (the built-in ACES 2.0 studio config, $OCIO, or a config.ocio), its displays
// and views, and the transforms the viewer runs (core/view_ocio.hpp): a view from the
// working space (ACEScg or ACES2065-1) with OpenColorIO's own GLSL for the GPU and its CPU
// processor for core/view.cpp, its decode to light for HDR swapchains, and the input
// transforms the original SDR comes in through.
//
// OpenColorIO stays in src/ocio_viewer.cpp (RUDRA_WITH_OCIO); without it every call says
// so and the viewer keeps its own view.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/view_ocio.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::ocio_viewer {

// The config RUDRA ships with: OpenColorIO's built-in ACES 2.0 studio config.
inline constexpr const char* kBuiltinConfig = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5";

// The display pass's last binding is 5; OpenColorIO's samplers start after it.
inline constexpr unsigned kFirstOcioBinding = 6;

enum class ConfigKind { Builtin, Environment, File };
struct ConfigSource {
    ConfigKind kind = ConfigKind::Builtin;
    std::string path;   // File: the config.ocio
};

// Built with OpenColorIO (the viewer can be colour managed at all).
bool available() noexcept;

// A display and view the viewer offers first: the views the 17 Nov 2026 request named
// (SDR sRGB and Rec.709, HDR Rec.2100 PQ 1000, P3-D65, Raw), those the config has.
struct ViewChoice {
    std::string label;      // "SDR · sRGB · ACES 2.0 100 nits"
    std::string display, view;
};

// A way in for the original SDR: an id the app keeps ("inverse:<display>\t<view>" or
// "space:<colour space>") and its words.
struct InputChoice {
    std::string id, label;
};

class Config {
public:
    static Result<std::shared_ptr<const Config>> open(const ConfigSource& source);

    // "ACES 2.0 studio config (built in) · OpenColorIO 2.5.2" or the file's path.
    const std::string& name() const;
    const ConfigSource& source() const;
    std::vector<std::string> displays() const;
    std::vector<std::string> views(const std::string& display) const;
    std::vector<ViewChoice> presets() const;
    std::vector<InputChoice> inputs() const;

    // The view from `working`, its textures bound from kFirstOcioBinding on (the display
    // pass's own end at 5). threads: the CPU processor's split, 0 for the machine's cores.
    Result<std::shared_ptr<const OcioViewTransform>> view(const std::string& display, const std::string& view,
                                                         WorkingSpace working, int threads = 0) const;
    // The original's way in (an InputChoice id) to `working`.
    Result<std::shared_ptr<const OcioInputTransform>> input(const std::string& id, WorkingSpace working,
                                                           int threads = 0) const;

    struct Impl;
    explicit Config(std::shared_ptr<Impl> d) : d_(std::move(d)) {}

private:
    std::shared_ptr<Impl> d_;
};

}  // namespace rudra::ocio_viewer
