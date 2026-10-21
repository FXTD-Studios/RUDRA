// The OFX plug-in in a host (roadmap 5.1): a small host on the OpenFX SDK's
// HostSupport library (BSD) finds RUDRA.ofx.bundle, describes it, makes an
// instance in the filter context and renders a float frame through it. The
// frame must be what render_sdr_to_hdr makes of the same pixels, with the
// parameters the host gives.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// HostSupport's headers depend on each other in this order (its hostDemo's).
#include "ofxCore.h"
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
    bool getConnected() const { return true; }
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
        return new FloatImage(*this, source_pixels());
    }
#ifdef OFX_SUPPORTS_OPENGLRENDER
    ImageEffect::Texture* loadTexture(OfxTime, const char*, const OfxRectD*) { return nullptr; }
#endif

private:
    std::string name_;
    FloatImage* output_ = nullptr;
};

template <class Base, class T>
class Value : public Base {
public:
    Value(Param::Descriptor& d, const std::string& name, T fallback) : Base(d), value_(fallback) {
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
        if (d.getType() == kOfxParamTypeDouble) return new DoubleValue(d, name, props.getDoubleProperty(kOfxParamPropDefault));
        if (d.getType() == kOfxParamTypeBoolean) return new BoolValue(d, name, props.getIntProperty(kOfxParamPropDefault) != 0);
        if (d.getType() == kOfxParamTypeChoice) return new ChoiceValue(d, name, props.getIntProperty(kOfxParamPropDefault));
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
        PluginCache::getPluginCache()->setCacheVersion("rudra-test-host-1");
        x.cache = std::make_unique<ImageEffect::PluginCache>(x.host);
        x.cache->registerInCache(*PluginCache::getPluginCache());
        PluginCache::getPluginCache()->addFileToPath(RUDRA_OFX_BUNDLE_DIR);
        PluginCache::getPluginCache()->scanPluginFiles();
        x.plugin = x.cache->getPluginById("com.fxtdstudios.rudra.SdrToHdr");
        return x;
    }();
    return l;
}

// One frame through the plug-in, with these parameter values.
std::vector<float> render(const std::map<std::string, double>& values, std::vector<std::string>* messages = nullptr) {
    g_values = values;
    auto* plugin = loaded().plugin;
    if (!plugin) return {};
    std::unique_ptr<ImageEffect::Instance> instance(plugin->createInstance(kOfxImageEffectContextFilter, nullptr));
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
    if (s != kOfxStatOK || !out || !out->output()) return {};
    return out->output()->pixels();
}

struct Grade {
    std::string source = "bt1886";
    double exposure = 0.0;
    bool fill = true;
    std::string output = "linear_rec2020";
};

std::vector<float> expected(const Grade& g) {
    const auto src = source_pixels();
    std::vector<float> out(src.size());
    EXPECT_EQ(rudra_ofx::render(g.source, g.exposure, 1.0, "all", g.fill, true, g.output, src.data(), kW * 4, out.data(),
                                kW * 4, kW, kH),
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
