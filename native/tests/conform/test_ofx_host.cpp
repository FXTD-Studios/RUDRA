// The OFX plug-in in a host (roadmap 5.1): a small host on the OpenFX SDK's
// HostSupport library (BSD) finds RUDRA.ofx.bundle, describes it, makes an
// instance in the filter context and renders a float frame through it. The
// frame must be what render_sdr_to_hdr makes of the same pixels, with the
// parameters the host gives.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// HostSupport's headers depend on each other in this order (its hostDemo's).
#include "ofxCore.h"
#include "ofxColour.h"
#include "ofxImageEffect.h"
#include "ofxPixels.h"
#include "ofxhBinary.h"
#include "ofxhPropertySuite.h"
#include "ofxhClip.h"
#include "ofxhParam.h"
#include "ofxhMemory.h"
#include "ofxhImageEffect.h"
#include "ofxhPluginAPICache.h"
#include "ofxhPluginCache.h"
#include "ofxhHost.h"
#include "ofxhImageEffectAPI.h"

#define RUDRA_OFX_PLAIN
#include "ofx_process.hpp"

namespace {

using namespace OFX::Host;
constexpr int kW = 48, kH = 27;

// What the test sets before an instance is made: parameter values by name.
std::map<std::string, double> g_values;
// The Mask input (27 Nov 2026): every pixel's weight, or below 0 for nothing connected.
float g_mask = -1.0f;
// 1.3.2: the host abandons the render (abort() answers 1 and the source gives no picture), or the source gives none.
bool g_abort = false, g_no_source = false;

std::vector<float> source_pixels() {
    std::vector<float> p(std::size_t(kW) * kH * 4);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float* q = &p[(std::size_t(y) * kW + std::size_t(x)) * 4];
            q[0] = float(x) / float(kW - 1);
            q[1] = float(y) / float(kH - 1);
            q[2] = (x + y) % 7 == 0 ? 1.0f : 0.35f;   // some clipped highlights
            q[3] = 0.5f + 0.5f * float(x % 2);
        }
    return p;
}

class Effect;

class FloatImage : public ImageEffect::Image {
public:
    FloatImage(ImageEffect::ClipInstance& clip, std::vector<float> pixels) : ImageEffect::Image(clip), data_(std::move(pixels)) {
        setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 0);
        setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 1);
        setPointerProperty(kOfxImagePropData, data_.data());
        for (const char* p : {kOfxImagePropBounds, kOfxImagePropRegionOfDefinition}) {
            setIntProperty(p, 0, 0), setIntProperty(p, 0, 1), setIntProperty(p, kW, 2), setIntProperty(p, kH, 3);
        }
        setIntProperty(kOfxImagePropRowBytes, kW * 4 * int(sizeof(float)));
    }
    const std::vector<float>& pixels() const { return data_; }

private:
    std::vector<float> data_;
};

class Clip : public ImageEffect::ClipInstance {
public:
    Clip(ImageEffect::Instance* effect, ImageEffect::ClipDescriptor* desc)
        : ImageEffect::ClipInstance(effect, *desc), name_(desc->getName()) {}
    ~Clip() {
        if (output_) output_->releaseReference();
    }
    FloatImage* output() { return output_; }
    const std::string& getUnmappedBitDepth() const {
        static const std::string v(kOfxBitDepthFloat);
        return v;
    }
    const std::string& getUnmappedComponents() const {
        static const std::string v(kOfxImageComponentRGBA);
        return v;
    }
    const std::string& getPremult() const {
        static const std::string v(kOfxImageUnPreMultiplied);
        return v;
    }
    double getAspectRatio() const { return 1.0; }
    double getFrameRate() const { return 24.0; }
    void getFrameRange(double& a, double& b) const { a = 0, b = 1; }
    const std::string& getFieldOrder() const {
        static const std::string v(kOfxImageFieldNone);
        return v;
    }
    bool getConnected() const { return name_ != "Mask" || g_mask >= 0.0f; }
    double getUnmappedFrameRate() const { return 24.0; }
    void getUnmappedFrameRange(double& a, double& b) const { a = 0, b = 1; }
    bool getContinuousSamples() const { return false; }
    OfxRectD getRegionOfDefinition(OfxTime) const { return OfxRectD{0, 0, double(kW), double(kH)}; }
    ImageEffect::Image* getImage(OfxTime, const OfxRectD*) {
        if (name_ == kOfxImageEffectOutputClipName) {
            if (!output_) output_ = new FloatImage(*this, std::vector<float>(std::size_t(kW) * kH * 4, -1.0f));
            output_->addReference();
            return output_;
        }
        if (name_ == "Mask") return new FloatImage(*this, std::vector<float>(std::size_t(kW) * kH * 4, g_mask));
        if (g_abort || g_no_source) return nullptr;
        return new FloatImage(*this, source_pixels());
    }
#ifdef OFX_SUPPORTS_OPENGLRENDER
    ImageEffect::Texture* loadTexture(OfxTime, const char*, const OfxRectD*) { return nullptr; }
#endif

private:
    std::string name_;
    FloatImage* output_ = nullptr;
};

// ParamT, not Base: inside the class, `Base` finds the SDK's own Param::Base
// (every parameter instance derives from it), and MSVC then rejects Base(d).
template <class ParamT, class T>
class Value : public ParamT {
public:
    // `owner`: the effect the parameter belongs to, which a value the plug-in sets is reported to (26 Nov 2026).
    Value(Param::Descriptor& d, Param::SetInstance* owner, const std::string& name, T fallback) : ParamT(d, owner), value_(fallback) {
        auto it = g_values.find(name);
        if (it != g_values.end()) value_ = T(it->second);
    }
    OfxStatus get(T& v) { return v = value_, kOfxStatOK; }
    OfxStatus get(OfxTime, T& v) { return v = value_, kOfxStatOK; }
    OfxStatus set(T v) { return value_ = v, kOfxStatOK; }
    OfxStatus set(OfxTime, T v) { return value_ = v, kOfxStatOK; }

protected:
    T value_;
};

class DoubleValue : public Value<Param::DoubleInstance, double> {
public:
    using Value::Value;
    OfxStatus derive(OfxTime, double& v) { return v = 0.0, kOfxStatOK; }
    OfxStatus integrate(OfxTime a, OfxTime b, double& v) { return v = value_ * (b - a), kOfxStatOK; }
};
using BoolValue = Value<Param::BooleanInstance, bool>;
using ChoiceValue = Value<Param::ChoiceInstance, int>;
using IntValue = Value<Param::IntegerInstance, int>;   // 26 Nov 2026 (phase 3): AI fill's seed

// The plug-in's text parameters (AI fill's prompt, status and shot id), 26 Nov 2026.
class StringValue : public Param::StringInstance {
public:
    StringValue(Param::Descriptor& d, Param::SetInstance* owner, std::string fallback)
        : Param::StringInstance(d, owner), value_(std::move(fallback)) {}
    OfxStatus get(std::string& v) { return v = value_, kOfxStatOK; }
    OfxStatus get(OfxTime, std::string& v) { return v = value_, kOfxStatOK; }
    OfxStatus set(const char* v) { return value_ = v ? v : "", kOfxStatOK; }
    OfxStatus set(OfxTime, const char* v) { return value_ = v ? v : "", kOfxStatOK; }

private:
    std::string value_;
};

class Effect : public ImageEffect::Instance {
public:
    Effect(ImageEffect::ImageEffectPlugin* plugin, ImageEffect::Descriptor& desc, const std::string& context)
        : ImageEffect::Instance(plugin, desc, context, false) {}
    const std::string& getDefaultOutputFielding() const {
        static const std::string v(kOfxImageFieldNone);
        return v;
    }
    ImageEffect::ClipInstance* newClipInstance(ImageEffect::Instance* plugin, ImageEffect::ClipDescriptor* d, int) {
        return new Clip(plugin, d);
    }
    OfxStatus vmessage(const char*, const char*, const char* format, va_list args) {
        char b[1024];
        std::vsnprintf(b, sizeof b, format, args);
        messages.push_back(b);
        return kOfxStatOK;
    }
    OfxStatus setPersistentMessage(const char* t, const char* id, const char* f, va_list a) { return vmessage(t, id, f, a); }
    OfxStatus clearPersistentMessage() { return kOfxStatOK; }
    void getProjectSize(double& x, double& y) const { x = kW, y = kH; }
    void getProjectOffset(double& x, double& y) const { x = y = 0; }
    void getProjectExtent(double& x, double& y) const { x = kW, y = kH; }
    double getProjectPixelAspectRatio() const { return 1.0; }
    double getEffectDuration() const { return 1.0; }
    double getFrameRate() const { return 24.0; }
    double getFrameRecursive() const { return 0.0; }
    void getRenderScaleRecursive(double& x, double& y) const { x = y = 1.0; }
    Param::Instance* newParam(const std::string& name, Param::Descriptor& d) {
        auto& props = d.getProperties();
        if (d.getType() == kOfxParamTypeDouble) return new DoubleValue(d, this, name, props.getDoubleProperty(kOfxParamPropDefault));
        if (d.getType() == kOfxParamTypeBoolean) return new BoolValue(d, this, name, props.getIntProperty(kOfxParamPropDefault) != 0);
        if (d.getType() == kOfxParamTypeChoice) return new ChoiceValue(d, this, name, props.getIntProperty(kOfxParamPropDefault));
        // Phase 3 (26 Nov 2026): the limit's and AI fill's groups, the seed, the texts, the buttons.
        if (d.getType() == kOfxParamTypeInteger) return new IntValue(d, this, name, props.getIntProperty(kOfxParamPropDefault));
        if (d.getType() == kOfxParamTypeString) return new StringValue(d, this, props.getStringProperty(kOfxParamPropDefault));
        if (d.getType() == kOfxParamTypePushButton) return new Param::PushbuttonInstance(d, this);
        if (d.getType() == kOfxParamTypeGroup) return new Param::GroupInstance(d, this);
        if (d.getType() == kOfxParamTypePage) return new Param::PageInstance(d, this);
        return nullptr;
    }
    OfxStatus editBegin(const std::string&) { return kOfxStatOK; }
    OfxStatus editEnd() { return kOfxStatOK; }
    void progressStart(const std::string&, const std::string&) {}
    void progressEnd() {}
    bool progressUpdate(double) { return true; }
    double timeLineGetTime() { return 0.0; }
    void timeLineGotoTime(double) {}
    void timeLineGetBounds(double& a, double& b) { a = 0, b = 1; }
    int abort() { return g_abort ? 1 : 0; }
    std::vector<std::string> messages;
};

class TestHost : public ImageEffect::Host {
public:
    TestHost() {
        _properties.setStringProperty(kOfxPropName, "rudra-test-host");
        _properties.setStringProperty(kOfxPropLabel, "RUDRA test host");
        _properties.setIntProperty(kOfxPropAPIVersion, 1, 0);
        _properties.setIntProperty(kOfxPropAPIVersion, 4, 1);
        _properties.setIntProperty(kOfxImageEffectHostPropIsBackground, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsOverlays, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultiResolution, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsTiles, 0);
        _properties.setIntProperty(kOfxImageEffectPropTemporalClipAccess, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextFilter, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGeneral, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipDepths, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipPARs, 0);
        _properties.setIntProperty(kOfxImageEffectPropSetableFrameRate, 0);
        _properties.setIntProperty(kOfxImageEffectPropSetableFielding, 0);
        _properties.setIntProperty(kOfxParamHostPropMaxParameters, -1);
        _properties.setIntProperty(kOfxParamHostPropMaxPages, 0);
    }
    ImageEffect::Instance* newInstance(void*, ImageEffect::ImageEffectPlugin* plugin, ImageEffect::Descriptor& desc,
                                       const std::string& context) {
        return new Effect(plugin, desc, context);
    }
    ImageEffect::Descriptor* makeDescriptor(ImageEffect::ImageEffectPlugin* plugin) {
        return new ImageEffect::Descriptor(plugin);
    }
    ImageEffect::Descriptor* makeDescriptor(const ImageEffect::Descriptor& root, ImageEffect::ImageEffectPlugin* plugin) {
        return new ImageEffect::Descriptor(root, plugin);
    }
    ImageEffect::Descriptor* makeDescriptor(const std::string& path, ImageEffect::ImageEffectPlugin* plugin) {
        return new ImageEffect::Descriptor(path, plugin);
    }
    OfxStatus vmessage(const char*, const char*, const char*, va_list) { return kOfxStatOK; }
    OfxStatus setPersistentMessage(const char*, const char*, const char*, va_list) { return kOfxStatOK; }
    OfxStatus clearPersistentMessage() { return kOfxStatOK; }
#ifdef OFX_SUPPORTS_OPENGLRENDER
    OfxStatus flushOpenGLResources() const { return kOfxStatFailed; }
#endif
};

struct Loaded {
    TestHost host;
    std::unique_ptr<ImageEffect::PluginCache> cache;
    ImageEffect::ImageEffectPlugin* plugin = nullptr;
};

Loaded& loaded() {
    static Loaded l = [] {
        Loaded x;
        // The plug-in's log aside (1.3.2): not the user's %TEMP%\rudra_ofx.log.
        const std::filesystem::path logs = std::filesystem::temp_directory_path() / "rudra_ofx_test_logs";
        std::error_code ec;
        std::filesystem::create_directories(logs, ec);
#ifdef _WIN32
        _putenv_s("RUDRA_OFX_LOG_DIR", logs.string().c_str());
#else
        setenv("RUDRA_OFX_LOG_DIR", logs.string().c_str(), 1);
#endif
        PluginCache::getPluginCache()->setCacheVersion("rudra-test-host-1");
        x.cache = std::make_unique<ImageEffect::PluginCache>(x.host);
        x.cache->registerInCache(*PluginCache::getPluginCache());
        // Only the bundle just built (26 Nov 2026): the cache starts with the system's OFX folders, and a RUDRA
        // installed there earlier (an older build) was the one the tests loaded on Windows.
        const_cast<std::list<std::string>&>(PluginCache::getPluginCache()->getPluginPath()).clear();
        PluginCache::getPluginCache()->addFileToPath(RUDRA_OFX_BUNDLE_DIR);
        PluginCache::getPluginCache()->scanPluginFiles();
        x.plugin = x.cache->getPluginById("com.fxtdstudios.rudra.SdrToHdr");
        return x;
    }();
    return l;
}

// One frame through the plug-in, with these parameter values.
std::vector<float> render(const std::map<std::string, double>& values, std::vector<std::string>* messages = nullptr,
                          const char* context = kOfxImageEffectContextFilter, OfxStatus* status = nullptr) {
    g_values = values;
    auto* plugin = loaded().plugin;
    if (!plugin) return {};
    std::unique_ptr<ImageEffect::Instance> instance(plugin->createInstance(context, nullptr));
    if (!instance) return {};
    const OfxStatus created = instance->createInstanceAction();
    if (created != kOfxStatOK && created != kOfxStatReplyDefault) return {};
    if (!instance->getClipPreferences()) return {};
    const OfxPointD scale{1.0, 1.0};
    const OfxRectI window{0, 0, kW, kH};
    instance->beginRenderAction(0, 0, 1.0, false, scale, true, false);
    const OfxStatus s = instance->renderAction(0, kOfxImageFieldNone, window, scale, true, false, false);
    instance->endRenderAction(0, 0, 1.0, false, scale, true, false);
    auto* out = dynamic_cast<Clip*>(instance->getClip(kOfxImageEffectOutputClipName));
    if (messages) *messages = static_cast<Effect*>(instance.get())->messages;
    if (status) *status = s;
    if (s != kOfxStatOK || !out || !out->output()) return {};
    return out->output()->pixels();
}

struct Grade {
    std::string source = "bt1886";
    double exposure = 0.0;
    bool fill = true;
    std::string output = "linear_rec2020";
};

// The plug-in hands RUDRA the rows top first (26 Nov 2026): OFX's rows run bottom up, so the top row is the
// last in memory and the stride is negative, for the source and the output alike.
std::vector<float> expected(const Grade& g) {
    const auto src = source_pixels();
    std::vector<float> out(src.size());
    const std::ptrdiff_t last = std::ptrdiff_t(kH - 1) * kW * 4;
    EXPECT_EQ(rudra_ofx::render("analytic", g.source, g.exposure, 1.0, "all", g.fill, true, g.output, src.data() + last, -kW * 4,
                                out.data() + last, -kW * 4, kW, kH),
              "");
    return out;
}

}  // namespace

TEST(OfxHost, TheBundleLoadsAndDescribesItself) {
    auto* plugin = loaded().plugin;
    ASSERT_NE(plugin, nullptr) << "RUDRA.ofx.bundle not found in " << RUDRA_OFX_BUNDLE_DIR;
    auto& desc = plugin->getDescriptor();
    EXPECT_EQ(desc.getProps().getStringProperty(kOfxPropLabel), "RUDRA SDR to HDR");
    EXPECT_EQ(desc.getProps().getStringProperty(kOfxImageEffectPluginPropGrouping), "RUDRA");
}

TEST(OfxHost, ARenderIsTheMasterChainsPicture) {
    const auto got = render({});   // the defaults: BT.1886, 0 EV, scene-linear Rec.2020
    ASSERT_EQ(got.size(), std::size_t(kW) * kH * 4);
    const auto want = expected(Grade{});
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], want[i]) << "float " << i;
    // Alpha came through.
    EXPECT_EQ(got[3], 0.5f);
    EXPECT_EQ(got[7], 1.0f);
}

TEST(OfxHost, TheParametersReachThePicture) {
    Grade g;
    g.exposure = 1.0;
    g.output = "pq_rec2020";
    g.fill = false;
    g.source = "aces";
    const auto ids = rudra_ofx::source_ids();
    const int aces = int(std::find(ids.begin(), ids.end(), "aces") - ids.begin());
    ASSERT_LT(aces, int(ids.size()));
    const auto got = render({{"exposure", 1.0}, {"output", 2}, {"clipFill", 0}, {"source", aces}});
    ASSERT_FALSE(got.empty());
    const auto want = expected(g);
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], want[i]) << "float " << i;
    for (std::size_t i = 0; i < got.size(); i += 4) ASSERT_LE(got[i], 1.0f);   // PQ code
}

// Phase 3 (26 Nov 2026): the luminance limit and the Alpha choice reach the picture; AI fill is off by default
// and leaves the picture as it was.
TEST(OfxHost, TheLuminanceLimitReachesThePictureAndItsAlpha) {
    // Limit: Highlights above 0.6; Alpha: the limit (index 1).
    const auto got = render({{"limit", 1}, {"highlightsFrom", 0.6}, {"limitSoftness", 0.05}, {"alphaOut", 1}});
    ASSERT_EQ(got.size(), std::size_t(kW) * kH * 4);
    rudra_ofx::Settings s;
    s.limit = "highlights", s.highlights_from = 0.6, s.limit_softness = 0.05, s.alpha = "limit";
    const auto src = source_pixels();
    std::vector<float> want(src.size());
    const std::ptrdiff_t last = std::ptrdiff_t(kH - 1) * kW * 4;
    ASSERT_EQ(rudra_ofx::render(s, src.data() + last, -kW * 4, want.data() + last, -kW * 4, kW, kH).error, "");
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], want[i]) << "float " << i;
    // The alpha is the limit: 0 on a dark pixel, 1 on a bright one.
    for (std::size_t i = 0; i < src.size(); i += 4) {
        const float y = 0.2126f * src[i] + 0.7152f * src[i + 1] + 0.0722f * src[i + 2];
        if (y < 0.5f) {
            ASSERT_EQ(got[i + 3], 0.0f) << i;
        }
        if (y > 0.7f) {
            ASSERT_EQ(got[i + 3], 1.0f) << i;
        }
    }
}

// 26 Nov 2026: a host renders a frame again only when a parameter changes, and a button press changes none.
// Refresh status (and Generate, Cancel, Clear) change a hidden number, so the host renders the clip again:
// frames are collected after a Generate, and the stored fill replaces the master the host had kept.
TEST(OfxHost, RefreshStatusMakesTheHostRenderAgain) {
    auto* plugin = loaded().plugin;
    ASSERT_NE(plugin, nullptr);
    auto shown_after = [&](const std::map<std::string, double>& values, const char* button) {
        g_values = values;
        std::unique_ptr<ImageEffect::Instance> instance(plugin->createInstance(kOfxImageEffectContextFilter, nullptr));
        if (!instance) return -1;
        const OfxStatus created = instance->createInstanceAction();
        if (created != kOfxStatOK && created != kOfxStatReplyDefault) return -1;
        const OfxPointD scale{1.0, 1.0};
        instance->beginInstanceChangedAction(kOfxChangeUserEdited);
        instance->paramInstanceChangedAction(button, kOfxChangeUserEdited, 0.0, scale);
        instance->endInstanceChangedAction(kOfxChangeUserEdited);
        int shown = -1;
        if (auto* p = dynamic_cast<IntValue*>(instance->getParam("aiShown"))) p->get(shown);
        return shown;
    };
    EXPECT_EQ(shown_after({{"aiFill", 1.0}}, "aiRefresh"), 1);
    EXPECT_EQ(shown_after({}, "aiRefresh"), 0);   // AI fill off: nothing new to show
}

// 27 Nov 2026: in the general context (Nuke) the effect has a Mask input. Unconnected, the picture is as in the filter
// context (Resolve); connected, its alpha blends RUDRA into the plain inverse (OfxProcess.TheMaskBlends...).
TEST(OfxHost, TheMaskInputIsNukesAndBlendsThePicture) {
    const auto filter = render({});
    ASSERT_EQ(filter.size(), std::size_t(kW) * kH * 4);
    g_mask = -1.0f;
    EXPECT_EQ(render({}, nullptr, kOfxImageEffectContextGeneral), filter);
    const auto src = source_pixels();
    for (float m : {0.0f, 0.5f}) {
        g_mask = m;
        const auto got = render({}, nullptr, kOfxImageEffectContextGeneral);
        rudra_ofx::Settings s;
        s.mask.assign(std::size_t(kW) * kH, m);
        std::vector<float> want(src.size());
        const std::ptrdiff_t last = std::ptrdiff_t(kH - 1) * kW * 4;
        ASSERT_EQ(rudra_ofx::render(s, src.data() + last, -kW * 4, want.data() + last, -kW * 4, kW, kH).error, "");
        ASSERT_EQ(got.size(), want.size()) << m;
        for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], want[i]) << "mask " << m << ", float " << i;
    }
    g_mask = -1.0f;
}

#ifdef RUDRA_OFX_FAKE_ENGINE
// 1.3.2: Suggest prompt writes a suggestion into Prompt and says where it came from (the stand-in suggester answers;
// the frame is the one the host gives in changedParam).
TEST(OfxHost, SuggestPromptPutsASuggestionInThePrompt) {
    namespace fs = std::filesystem;
    auto* plugin = loaded().plugin;
    ASSERT_NE(plugin, nullptr);
    const auto set_env = [](const char* name, const std::string& value) {
#ifdef _WIN32
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    };
    const fs::path root = fs::temp_directory_path() / "rudra_ofx_host_suggest";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "repo" / "rudra" / "genfill");
    std::ofstream(root / "repo" / "rudra" / "genfill" / "ofx_job.py") << "# stand-in\n";
    fs::create_directories(root / "base" / "transformer");
    std::ofstream(root / "base" / "transformer" / "config.json") << "{}";
    std::ofstream(root / "lora.safetensors") << "x";
    const auto q = [](const fs::path& p) { return "\"" + p.generic_string() + "\""; };
    std::ofstream(root / "ofx_engine.json") << "{\"cli\": \"\", \"package\": \"\", \"genfill\": {\"python\": "
                                            << q(RUDRA_OFX_FAKE_ENGINE) << ", \"repo\": " << q(root / "repo") << ", \"base\": "
                                            << q(root / "base") << ", \"lora\": " << q(root / "lora.safetensors") << "}}";
    set_env("RUDRA_OFX_ENGINE_CONFIG", (root / "ofx_engine.json").string());
    set_env("RUDRA_OFX_FILL_ROOT", (root / "store").string());
    g_values = {};
    std::unique_ptr<ImageEffect::Instance> instance(plugin->createInstance(kOfxImageEffectContextFilter, nullptr));
    ASSERT_TRUE(instance);
    const OfxStatus created = instance->createInstanceAction();
    ASSERT_TRUE(created == kOfxStatOK || created == kOfxStatReplyDefault);
    const OfxPointD scale{1.0, 1.0};
    instance->beginInstanceChangedAction(kOfxChangeUserEdited);
    instance->paramInstanceChangedAction("aiSuggest", kOfxChangeUserEdited, 0.0, scale);
    instance->endInstanceChangedAction(kOfxChangeUserEdited);
    std::string prompt, status;
    if (auto* p = dynamic_cast<StringValue*>(instance->getParam("aiPrompt"))) p->get(prompt);
    if (auto* p = dynamic_cast<StringValue*>(instance->getParam("aiStatus"))) p->get(status);
    EXPECT_EQ(prompt, "bright hazy sky, stand-in");
    EXPECT_NE(status.find("Prompt suggested from frame 0: From this shot"), std::string::npos) << status;
    EXPECT_NE(status.find("Others: Overcast sky"), std::string::npos) << status;
    instance.reset();
    set_env("RUDRA_OFX_ENGINE_CONFIG", "");
    set_env("RUDRA_OFX_FILL_ROOT", "");
    fs::remove_all(root, ec);
}
#endif

// 1.3.2 (27 Nov 2026): Nuke abandons a render when a knob moves while it runs (a Grade's gain dragged above RUDRA)
// and hands it no picture. That render ends quietly: the error it used to leave on the node ("no picture from the
// input") stayed after the picture came back. A picture missing for any other reason is still an error that says so.
TEST(OfxHost, ARenderTheHostAbandonsLeavesNoErrorOnTheNode) {
    std::vector<std::string> messages;
    OfxStatus status = kOfxStatErrUnknown;
    g_abort = true;
    render({}, &messages, kOfxImageEffectContextFilter, &status);
    g_abort = false;
    EXPECT_EQ(status, kOfxStatOK);
    EXPECT_TRUE(messages.empty()) << messages.front();
    g_no_source = true;
    render({}, &messages, kOfxImageEffectContextFilter, &status);
    g_no_source = false;
    EXPECT_EQ(status, kOfxStatFailed);
    ASSERT_FALSE(messages.empty());
    EXPECT_NE(messages.back().find("no picture from the input"), std::string::npos) << messages.back();
    EXPECT_FALSE(render({}).empty());   // and the next render draws
}

// 1.3.2 (27 Nov 2026): OpenFX colour management through the plug-in's own entry point, as a colour-managing host (Resolve
// with "Enable OpenFX Color Management") calls it: the source asked for as Rec.709 display codes, the output named by
// Output. A host without colour management is not asked anything (the other tests).
TEST(OfxHost, AColourManagingHostIsAskedForTheSdrAndToldTheOutput) {
    auto* plugin = loaded().plugin;
    ASSERT_NE(plugin, nullptr);
    g_values = {{"output", 1.0}};   // Scene-linear ACEScg
    std::unique_ptr<ImageEffect::Instance> instance(plugin->createInstance(kOfxImageEffectContextFilter, nullptr));
    ASSERT_TRUE(instance);
    const OfxStatus created = instance->createInstanceAction();
    ASSERT_TRUE(created == kOfxStatOK || created == kOfxStatReplyDefault);
    static const Property::PropSpec style = {kOfxImageEffectPropColourManagementStyle, Property::eString, 1, true,
                                             kOfxImageEffectColourManagementCore};
    instance->getProps().createProperty(style);

    const std::string preferred = std::string(kOfxImageClipPropPreferredColourspaces) + "_Source";
    const Property::PropSpec pref = {preferred.c_str(), Property::eString, 0, false, ""};
    Property::Set prefs;
    prefs.createProperty(pref);
    EXPECT_EQ(instance->mainEntry(kOfxImageEffectActionGetClipPreferences, instance->getHandle(), nullptr, &prefs), kOfxStatOK);
    ASSERT_EQ(prefs.getDimension(preferred), 2);
    EXPECT_EQ(prefs.getStringProperty(preferred, 0), "rec1886_rec709_display");
    EXPECT_EQ(prefs.getStringProperty(preferred, 1), "ofx_display_sdr");

    static const Property::PropSpec ask = {kOfxImageClipPropPreferredColourspaces, Property::eString, 0, false, ""};
    static const Property::PropSpec got = {kOfxImageClipPropColourspace, Property::eString, 1, false, ""};
    Property::Set in, out;
    in.createProperty(ask);
    out.createProperty(got);
    EXPECT_EQ(instance->mainEntry(kOfxImageEffectActionGetOutputColourspace, instance->getHandle(), &in, &out), kOfxStatOK);
    EXPECT_EQ(out.getStringProperty(kOfxImageClipPropColourspace), "ACEScg");
    g_values = {};
}
