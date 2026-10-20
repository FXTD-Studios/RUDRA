// RUDRA as an OFX image effect (roadmap 5.1, 20 Oct 2026): "RUDRA SDR to HDR"
// for DaVinci Resolve, Nuke, Baselight, Flame and any OFX 1.4 host. The
// analytic reconstruction and the master chain (ofx_process.hpp), on the CPU,
// a whole frame at a time; float RGBA in and out.
//
// Built on the OFX C++ Support library (BSD, the OpenFX SDK's Support/).
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#define RUDRA_OFX_PLAIN   // C++14: only the plain interface
#include "ofx_process.hpp"

namespace {

constexpr const char* kPluginId = "com.fxtdstudios.rudra.SdrToHdr";
constexpr const char* kSource = "source", *kExposure = "exposure", *kStrength = "strength", *kRecovery = "recovery",
                      *kClipFill = "clipFill", *kAnchor = "anchor", *kOutput = "output";
constexpr const char* kRecoveryModes[] = {"all", "highlights", "shadows", "off"};
constexpr const char* kOutputs[] = {"linear_rec2020", "linear_acescg", "pq_rec2020"};

int clamp_index(int i, int n) { return std::max(0, std::min(i, n - 1)); }

class RudraEffect : public OFX::ImageEffect {
public:
    explicit RudraEffect(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
        dst_ = fetchClip(kOfxImageEffectOutputClipName);
        src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
        source_ = fetchChoiceParam(kSource);
        exposure_ = fetchDoubleParam(kExposure);
        strength_ = fetchDoubleParam(kStrength);
        recovery_ = fetchChoiceParam(kRecovery);
        clip_fill_ = fetchBooleanParam(kClipFill);
        anchor_ = fetchBooleanParam(kAnchor);
        output_ = fetchChoiceParam(kOutput);
    }

    void render(const OFX::RenderArguments& args) {
        std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
        std::unique_ptr<OFX::Image> src(src_->fetchImage(args.time));
        if (!dst || !src) OFX::throwSuiteStatusException(kOfxStatFailed);
        if (src->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelDepth() != OFX::eBitDepthFloat ||
            src->getPixelComponents() != OFX::ePixelComponentRGBA || dst->getPixelComponents() != OFX::ePixelComponentRGBA)
            OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
        int source = 0, recovery = 0, output = 0;
        double exposure = 0.0, strength = 1.0;
        bool fill = true, anchor = true;
        source_->getValueAtTime(args.time, source);
        exposure_->getValueAtTime(args.time, exposure);
        strength_->getValueAtTime(args.time, strength);
        recovery_->getValueAtTime(args.time, recovery);
        clip_fill_->getValueAtTime(args.time, fill);
        anchor_->getValueAtTime(args.time, anchor);
        output_->getValueAtTime(args.time, output);
        const std::vector<std::string> ids = rudra_ofx::source_ids();

        // The whole source frame (the anchor and the grain settling look at
        // neighbours), then the render window out of it.
        const OfxRectI sb = src->getBounds();
        const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
        if (w <= 0 || h <= 0) return;
        std::vector<float> out(std::size_t(w) * std::size_t(h) * 4);
        const auto* first = static_cast<const float*>(src->getPixelAddress(sb.x1, sb.y1));
        const std::ptrdiff_t src_row = src->getRowBytes() / std::ptrdiff_t(sizeof(float));
        const std::string why = rudra_ofx::render(ids.at(std::size_t(clamp_index(source, int(ids.size())))), exposure, strength,
                                                  kRecoveryModes[clamp_index(recovery, 4)], fill, anchor,
                                                  kOutputs[clamp_index(output, 3)], first, src_row, out.data(),
                                                  std::ptrdiff_t(w) * 4, w, h);
        if (!why.empty()) {
            setPersistentMessage(OFX::Message::eMessageError, "", "RUDRA: " + why);
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
        clearPersistentMessage();
        const OfxRectI win = args.renderWindow;
        for (int y = std::max(win.y1, sb.y1); y < std::min(win.y2, sb.y2); ++y) {
            auto* row = static_cast<float*>(dst->getPixelAddress(std::max(win.x1, sb.x1), y));
            if (!row) continue;
            const float* from = out.data() + (std::size_t(y - sb.y1) * std::size_t(w) + std::size_t(std::max(win.x1, sb.x1) - sb.x1)) * 4;
            std::copy(from, from + std::ptrdiff_t(std::min(win.x2, sb.x2) - std::max(win.x1, sb.x1)) * 4, row);
        }
    }

    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) {
        rois.setRegionOfInterest(*src_, src_->getRegionOfDefinition(args.time));
    }

private:
    OFX::Clip *dst_ = nullptr, *src_ = nullptr;
    OFX::ChoiceParam *source_ = nullptr, *recovery_ = nullptr, *output_ = nullptr;
    OFX::DoubleParam *exposure_ = nullptr, *strength_ = nullptr;
    OFX::BooleanParam *clip_fill_ = nullptr, *anchor_ = nullptr;
};

mDeclarePluginFactory(RudraFactory, {}, {});

void RudraFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels("RUDRA SDR to HDR", "RUDRA SDR to HDR", "RUDRA SDR to HDR");
    desc.setPluginGrouping("RUDRA");
    desc.setPluginDescription(
        "SDR to scene-linear or PQ HDR by RUDRA's analytic reconstruction (FXTD Studios). Non-commercial use "
        "under the PolyForm Noncommercial License 1.0.0.");
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
}

void RudraFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum) {
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    OFX::ChoiceParamDescriptor* source = desc.defineChoiceParam(kSource);
    source->setLabels("Source", "Source", "Source curve");
    source->setHint("The tone curve that made the SDR: BT.1886 for camera and broadcast video; ACES, filmic or AgX for CG and games.");
    for (const std::string& label : rudra_ofx::source_labels()) source->appendOption(label);
    source->setDefault(rudra_ofx::default_source());

    OFX::DoubleParamDescriptor* exposure = desc.defineDoubleParam(kExposure);
    exposure->setLabels("Exposure", "Exposure", "Exposure (stops)");
    exposure->setDefault(0.0);
    exposure->setRange(-3.0, 3.0);
    exposure->setDisplayRange(-3.0, 3.0);
    exposure->setIncrement(0.05);

    OFX::DoubleParamDescriptor* strength = desc.defineDoubleParam(kStrength);
    strength->setLabels("Strength", "Strength", "Reconstruction strength");
    strength->setDefault(1.0);
    strength->setRange(0.0, 1.0);
    strength->setDisplayRange(0.0, 1.0);

    OFX::ChoiceParamDescriptor* recovery = desc.defineChoiceParam(kRecovery);
    recovery->setLabels("Recovery", "Recovery", "Recovery");
    for (const char* m : {"All", "Highlights", "Shadows", "Off"}) recovery->appendOption(m);
    recovery->setDefault(0);

    OFX::BooleanParamDescriptor* fill = desc.defineBooleanParam(kClipFill);
    fill->setLabels("Fill clipped highlights", "Fill clipped", "Fill clipped highlights");
    fill->setDefault(true);

    OFX::BooleanParamDescriptor* anchor = desc.defineBooleanParam(kAnchor);
    anchor->setLabels("Anchor to the SDR", "Anchor", "Anchor to the SDR's mid-tones");
    anchor->setDefault(true);

    OFX::ChoiceParamDescriptor* output = desc.defineChoiceParam(kOutput);
    output->setLabels("Output", "Output", "Output encoding");
    output->appendOption("Scene-linear Rec.2020 (1.0 = 203 nits)");
    output->appendOption("Scene-linear ACEScg (1.0 = 203 nits)");
    output->appendOption("PQ Rec.2020 (ST 2084)");
    output->setDefault(0);
}

OFX::ImageEffect* RudraFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) {
    return new RudraEffect(handle);
}

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
    static RudraFactory factory(kPluginId, 1, 0);
    ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
