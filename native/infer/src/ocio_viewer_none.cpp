// infer/ocio_viewer.hpp without OpenColorIO (RUDRA_WITH_OCIO off): the viewer keeps its own view.

#include "rudra/infer/ocio_viewer.hpp"

namespace rudra::ocio_viewer {

namespace {
Error missing() {
    return make_error(ErrorCode::Unsupported, "This build of RUDRA has no OpenColorIO, so the viewer has its own view only.",
                      "built without RUDRA_WITH_OCIO");
}
}  // namespace

struct Config::Impl {
    ConfigSource source;
    std::string name;
};

bool available() noexcept { return false; }

Result<std::shared_ptr<const Config>> Config::open(const ConfigSource&) { return missing(); }
const std::string& Config::name() const { return d_->name; }
const ConfigSource& Config::source() const { return d_->source; }
std::vector<std::string> Config::displays() const { return {}; }
std::vector<std::string> Config::views(const std::string&) const { return {}; }
std::vector<ViewChoice> Config::presets() const { return {}; }
std::vector<InputChoice> Config::inputs() const { return {}; }
Result<std::shared_ptr<const OcioViewTransform>> Config::view(const std::string&, const std::string&, WorkingSpace, int) const {
    return missing();
}
Result<std::shared_ptr<const OcioInputTransform>> Config::input(const std::string&, WorkingSpace, int) const {
    return missing();
}

}  // namespace rudra::ocio_viewer
