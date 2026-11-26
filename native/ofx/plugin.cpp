// RUDRA as an OFX image effect (roadmap 5.1, 20 Oct 2026): "RUDRA SDR to HDR"
// for DaVinci Resolve, Nuke, Baselight, Flame and any OFX host. Two routes
// (ofx_process.hpp): the analytic reconstruction and the master chain, on the
// CPU, and (25 Nov 2026, phase 2) the learned ACES 2.0 master through RUDRA's
// own engine run beside the host. A whole frame at a time; float RGBA in and out.
// 26 Nov 2026 (phase 3): the luminance limit (the app's Reconstruct > Region), an Alpha choice (the source's,
// the limit, what AI drew), and AI fill: Generate, the shot collected as it plays, the fill drawn beside the
// host by the RUDRA app's AI fill and shown from then on (fill_store.hpp).
//
// Built on the OFX C++ Support library (BSD, the OpenFX SDK's Support/).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#define RUDRA_OFX_PLAIN   // only the plain interface: no RUDRA types in the plug-in
#include "ofx_process.hpp"

namespace {

constexpr const char* kPluginId = "com.fxtdstudios.rudra.SdrToHdr";
constexpr const char* kRoute = "route";
constexpr const char* kRoutes[] = {"analytic", "learned"};
constexpr const char* kSource = "source", *kExposure = "exposure", *kStrength = "strength", *kRecovery = "recovery",
                      *kClipFill = "clipFill", *kAnchor = "anchor", *kOutput = "output";
constexpr const char* kRecoveryModes[] = {"all", "highlights", "shadows", "off"};
constexpr const char* kOutputs[] = {"linear_rec2020", "linear_acescg", "pq_rec2020"};
// Phase 3 (26 Nov 2026).
constexpr const char* kLimitGroup = "limitGroup", *kLimit = "limit", *kHighlightsFrom = "highlightsFrom",
                      *kShadowsBelow = "shadowsBelow", *kBandFrom = "bandFrom", *kBandTo = "bandTo", *kLimitSoftness = "limitSoftness";
constexpr const char* kLimits[] = {"off", "highlights", "shadows", "both", "band"};
constexpr const char* kAlpha = "alphaOut";
constexpr const char* kAlphas[] = {"source", "limit", "ai_fill"};
constexpr const char* kAiGroup = "aiGroup", *kAiFill = "aiFill", *kAiPrompt = "aiPrompt", *kAiSeed = "aiSeed",
                      *kAiGenerate = "aiGenerate", *kAiCancel = "aiCancel", *kAiRefresh = "aiRefresh", *kAiClear = "aiClear",
                      *kAiStatus = "aiStatus", *kAiShot = "aiShot", *kAiShown = "aiShown", *kAiOffset = "aiOffset", *kAiSteps = "aiSteps",
                      *kAiFrom = "aiFrom", *kAiTo = "aiTo";
// The Mask input (27 Nov 2026, the general context: Nuke) and its invert switch.
constexpr const char* kMaskClip = "Mask", *kMaskInvert = "maskInvert";

int clamp_index(int i, int n) { return std::max(0, std::min(i, n - 1)); }

// Why a render failed, where it can be read (25 Nov 2026): Nuke shows only "Effect raised an
// error during render", not the persistent message, so the reason also goes to the host's
// console (stderr) and to <temp>/rudra_ofx.log.
void log_failure(const std::string& why) {
    std::fprintf(stderr, "RUDRA SDR to HDR: %s\n", why.c_str());
    std::fflush(stderr);
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) return;
    std::ofstream f(dir / "rudra_ofx.log", std::ios::app);
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char stamp[32] = {};
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    f << stamp << "  " << why << "\n";
}

class RudraEffect : public OFX::ImageEffect {
public:
    explicit RudraEffect(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
        dst_ = fetchClip(kOfxImageEffectOutputClipName);
        src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
        route_ = fetchChoiceParam(kRoute);
        source_ = fetchChoiceParam(kSource);
        exposure_ = fetchDoubleParam(kExposure);
        strength_ = fetchDoubleParam(kStrength);
        recovery_ = fetchChoiceParam(kRecovery);
        clip_fill_ = fetchBooleanParam(kClipFill);
        anchor_ = fetchBooleanParam(kAnchor);
        output_ = fetchChoiceParam(kOutput);
        limit_ = fetchChoiceParam(kLimit);
        highlights_from_ = fetchDoubleParam(kHighlightsFrom);
        shadows_below_ = fetchDoubleParam(kShadowsBelow);
        band_from_ = fetchDoubleParam(kBandFrom);
        band_to_ = fetchDoubleParam(kBandTo);
        limit_softness_ = fetchDoubleParam(kLimitSoftness);
        alpha_ = fetchChoiceParam(kAlpha);
        ai_fill_ = fetchBooleanParam(kAiFill);
        ai_prompt_ = fetchStringParam(kAiPrompt);
        ai_seed_ = fetchIntParam(kAiSeed);
        ai_status_ = fetchStringParam(kAiStatus);
        ai_shot_ = fetchStringParam(kAiShot);
        ai_shown_ = fetchIntParam(kAiShown);
        ai_offset_ = fetchIntParam(kAiOffset);
        ai_steps_ = fetchIntParam(kAiSteps);
        ai_from_ = fetchIntParam(kAiFrom);
        ai_to_ = fetchIntParam(kAiTo);
        mask_invert_ = fetchBooleanParam(kMaskInvert);
        if (getContext() == OFX::eContextGeneral) mask_ = fetchClip(kMaskClip);   // Resolve (filter): the node's own key
        update_enabled();
    }

    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) {
        if (name == kAiStatus || name == kAiShot || name == kAiShown || name == kAiOffset) return;   // the plug-in's own writes
        if (name == kRoute || name == kLimit) update_enabled();
        rudra_ofx::Settings s = settings_at(args.time);
        std::string why;
        bool render_again = false;
        if (name == kAiGenerate) {
            if (s.shot_id.empty()) {
                s.shot_id = rudra_ofx::new_shot_id();
                ai_shot_->setValue(s.shot_id);
            }
            if (!s.ai_fill) {
                s.ai_fill = true;
                ai_fill_->setValue(true);
            }
            // Resolve (1.3.2): it renders a clip at its source media's frame numbers (129, 130, ...) but gives the
            // clip's range from 0 (0 to 96), so no frame was ever inside the range and nothing was collected. There
            // the range starts at the playhead: From (or the clip's first frame) is the frame Generate is pressed on.
            const int offset = resolve_host() ? int(std::lround(args.time)) - (s.first_frame - current_offset()) : 0;
            if (offset != current_offset()) {
                ai_offset_->setValue(offset);
                s = settings_at(args.time);
            }
            why = rudra_ofx::ai_generate(s);
            render_again = why.empty();
        } else if (name == kAiCancel) {
            why = rudra_ofx::ai_cancel(s);
            render_again = true;
        } else if (name == kAiClear) {
            why = rudra_ofx::ai_clear(s);
            if (why.empty()) ai_offset_->setValue(0);
            render_again = why.empty();
        } else if (name == kAiRefresh) {
            render_again = s.ai_fill;
        }
        if (!why.empty()) {
            log_failure(why);
            sendMessage(OFX::Message::eMessageError, "", "RUDRA: " + why);
        }
        const std::string now = (s.ai_fill || name == kAiFill) ? rudra_ofx::ai_status(s) : std::string();
        if (s.ai_fill || name == kAiFill) ai_status_->setValue(now);
        // What a press did, in the log (27 Nov 2026): the host may not show the status field.
        if (name == kAiGenerate || name == kAiCancel || name == kAiClear || name == kAiRefresh)
            log_failure(name + " pressed (frames " + std::to_string(s.first_frame) + " to " + std::to_string(s.last_frame) +
                        ", steps " + std::to_string(s.steps) + "): " + (why.empty() ? now : why));
        // The host keeps the frames it rendered and asks again only when a parameter changes; a button
        // press changes none. Generate, Cancel, Clear and Refresh status change this hidden number, so the
        // clip is rendered again: its frames are collected after a Generate, and the stored fill replaces
        // the master once it is ready (26 Nov 2026).
        if (render_again) {
            int shown = 0;
            ai_shown_->getValue(shown);
            ai_shown_->setValue(shown % 1000000 + 1);
        }
    }

    void render(const OFX::RenderArguments& args) {
        std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
        std::unique_ptr<OFX::Image> src(src_->fetchImage(args.time));
        if (!dst || !src) fail(!src ? "no picture from the input (check the node above RUDRA for its own error)."
                                    : "the host gave no output image.");
        if (src->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelDepth() != OFX::eBitDepthFloat ||
            src->getPixelComponents() != OFX::ePixelComponentRGBA || dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
            log_failure("the host gave a picture that is not 32-bit float RGBA.");
            OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
        }
        rudra_ofx::Settings s = settings_at(args.time);
        s.mask = mask_plane(args.time, src->getBounds());
        s.full_size = std::fabs(args.renderScale.x - 1.0) < 1e-9 && std::fabs(args.renderScale.y - 1.0) < 1e-9;

        // The whole source frame (the anchor and the grain settling look at neighbours), then the render
        // window out of it. The rows go to RUDRA top first (26 Nov 2026): OFX images run bottom up, and the
        // learned model and AI fill drew the picture upside down before. One row up in OFX is one row
        // down the picture: a stride of minus the row bytes, from the top row's address.
        const OfxRectI sb = src->getBounds();
        const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
        if (w <= 0 || h <= 0) return;
        std::vector<float> out(std::size_t(w) * std::size_t(h) * 4);
        const auto* top = static_cast<const float*>(src->getPixelAddress(sb.x1, sb.y2 - 1));
        const std::ptrdiff_t src_row = -(src->getRowBytes() / std::ptrdiff_t(sizeof(float)));
        rudra_ofx::Outcome result;
        try {
            result = rudra_ofx::render(s, top, src_row, out.data(), std::ptrdiff_t(w) * 4, w, h);
        } catch (const std::bad_alloc&) {
            result.error = "out of memory for a " + std::to_string(w) + " x " + std::to_string(h) + " frame.";
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        if (!result.error.empty())
            fail(result.error + " (" + s.route + " route, " + std::to_string(w) + " x " + std::to_string(h) + ")");
        if (!result.note.empty() && (result.note.rfind("AI fill failed", 0) == 0 || result.note.find("out of date") != std::string::npos ||
                                     result.note.find("not ") != std::string::npos)) {
            log_note_once(result.note);   // in the log as well: Nuke shows a node's warning only on the node
            setPersistentMessage(OFX::Message::eMessageWarning, "", "RUDRA: " + result.note);
        } else {
            clearPersistentMessage();
        }
        const OfxRectI win = args.renderWindow;
        const int x0 = std::max(win.x1, sb.x1), x1 = std::min(win.x2, sb.x2);
        for (int y = std::max(win.y1, sb.y1); y < std::min(win.y2, sb.y2); ++y) {
            auto* row = static_cast<float*>(dst->getPixelAddress(x0, y));
            if (!row || x1 <= x0) continue;
            const std::size_t r = std::size_t(sb.y2 - 1 - y);   // RUDRA's row: the top is 0
            const float* from = out.data() + (r * std::size_t(w) + std::size_t(x0 - sb.x1)) * 4;
            std::copy(from, from + std::ptrdiff_t(x1 - x0) * 4, row);
        }
    }

    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) {
        rois.setRegionOfInterest(*src_, src_->getRegionOfDefinition(args.time));
    }

private:
    // The Mask input as weights over the source's bounds, the top row first (RUDRA's order): its alpha (or its only
    // channel), 0 outside the mask's own bounds, inverted when asked. Empty when nothing is connected.
    std::vector<float> mask_plane(double time, const OfxRectI& sb) {
        if (!mask_ || !mask_->isConnected()) return {};
        std::unique_ptr<OFX::Image> m(mask_->fetchImage(time));
        if (!m || m->getPixelDepth() != OFX::eBitDepthFloat) return {};
        bool invert = false;
        mask_invert_->getValueAtTime(time, invert);
        const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
        const int comps = m->getPixelComponents() == OFX::ePixelComponentAlpha ? 1 : 4;
        const OfxRectI mb = m->getBounds();
        std::vector<float> out(std::size_t(w) * std::size_t(h), 0.0f);
        for (int r = 0; r < h; ++r) {
            const int y = sb.y2 - 1 - r;   // RUDRA's row r is OFX's row y
            for (int x = 0; x < w; ++x) {
                const int X = sb.x1 + x;
                float v = 0.0f;
                if (X >= mb.x1 && X < mb.x2 && y >= mb.y1 && y < mb.y2)
                    if (const auto* p = static_cast<const float*>(m->getPixelAddress(X, y))) v = p[comps - 1];
                v = std::clamp(std::isfinite(v) ? v : 0.0f, 0.0f, 1.0f);
                out[std::size_t(r) * std::size_t(w) + std::size_t(x)] = invert ? 1.0f - v : v;
            }
        }
        return out;
    }

    // A warning once per change, not once per frame (renders run on several threads).
    void log_note_once(const std::string& note) {
        std::lock_guard<std::mutex> lock(note_mutex_);
        if (note == last_note_) return;
        last_note_ = note;
        log_failure(note);
    }

    [[noreturn]] void fail(const std::string& why) {
        log_failure(why);
        setPersistentMessage(OFX::Message::eMessageError, "", "RUDRA: " + why);
        OFX::throwSuiteStatusException(kOfxStatFailed);
        throw std::runtime_error(why);   // not reached
    }

    rudra_ofx::Settings settings_at(double time) {
        rudra_ofx::Settings s;
        int route = 0, source = 0, recovery = 0, output = 0, limit = 0, alpha = 0, seed = 0;
        route_->getValueAtTime(time, route);
        source_->getValueAtTime(time, source);
        exposure_->getValueAtTime(time, s.exposure_ev);
        strength_->getValueAtTime(time, s.strength);
        recovery_->getValueAtTime(time, recovery);
        clip_fill_->getValueAtTime(time, s.clip_fill);
        anchor_->getValueAtTime(time, s.anchor);
        output_->getValueAtTime(time, output);
        limit_->getValueAtTime(time, limit);
        highlights_from_->getValueAtTime(time, s.highlights_from);
        shadows_below_->getValueAtTime(time, s.shadows_below);
        band_from_->getValueAtTime(time, s.band_from);
        band_to_->getValueAtTime(time, s.band_to);
        limit_softness_->getValueAtTime(time, s.limit_softness);
        alpha_->getValueAtTime(time, alpha);
        ai_fill_->getValueAtTime(time, s.ai_fill);
        ai_prompt_->getValue(s.prompt);
        ai_seed_->getValue(seed);
        ai_steps_->getValue(s.steps);
        ai_shot_->getValue(s.shot_id);
        const std::vector<std::string> ids = rudra_ofx::source_ids();
        s.route = kRoutes[clamp_index(route, 2)];
        s.source_curve = ids.at(std::size_t(clamp_index(source, int(ids.size()))));
        s.recovery_mode = kRecoveryModes[clamp_index(recovery, 4)];
        s.output = kOutputs[clamp_index(output, 3)];
        s.limit = kLimits[clamp_index(limit, 5)];
        s.alpha = kAlphas[clamp_index(alpha, 3)];
        s.seed = seed;
        // The clip's frames, for AI fill: the source's frame range in the host's own frame numbers.
        const OfxRangeD range = src_->getFrameRange();
        s.first_frame = int(std::lround(range.min));
        s.last_frame = int(std::lround(range.max));
        // AI fill's frames (27 Nov 2026): From / To inside the clip; 0 is the clip's own first or last frame. A test
        // on a long clip draws a few frames, not all 823.
        int from = 0, to = 0;
        ai_from_->getValue(from);
        ai_to_->getValue(to);
        if (from != 0) s.first_frame = std::clamp(from, s.first_frame, s.last_frame);
        if (to != 0) s.last_frame = std::clamp(to, s.first_frame, s.last_frame);
        // Resolve's frames from the playhead at Generate (changedParam); 0 in every other host.
        const int offset = current_offset();
        s.first_frame += offset;
        s.last_frame += offset;
        s.frame = int(std::lround(time));
        return s;
    }

    int current_offset() const {
        int offset = 0;
        ai_offset_->getValue(offset);
        return offset;
    }

    static bool resolve_host() {
        const OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
        return host && host->hostName.find("DaVinci") != std::string::npos;
    }

    // The analytic route's own controls are greyed out on the learned route; each limit's edges only for it.
    void update_enabled() {
        int route = 0, limit = 0;
        route_->getValue(route);
        limit_->getValue(limit);
        const bool analytic = clamp_index(route, 2) == 0;
        source_->setEnabled(analytic);
        recovery_->setEnabled(analytic);
        clip_fill_->setEnabled(analytic);
        anchor_->setEnabled(analytic);
        const std::string l = kLimits[clamp_index(limit, 5)];
        highlights_from_->setEnabled(l == "highlights" || l == "both");
        shadows_below_->setEnabled(l == "shadows" || l == "both");
        band_from_->setEnabled(l == "band");
        band_to_->setEnabled(l == "band");
        limit_softness_->setEnabled(l != "off");
    }

    OFX::Clip *dst_ = nullptr, *src_ = nullptr, *mask_ = nullptr;
    OFX::ChoiceParam *route_ = nullptr, *source_ = nullptr, *recovery_ = nullptr, *output_ = nullptr, *limit_ = nullptr,
                     *alpha_ = nullptr;
    OFX::DoubleParam *exposure_ = nullptr, *strength_ = nullptr, *highlights_from_ = nullptr, *shadows_below_ = nullptr,
                     *band_from_ = nullptr, *band_to_ = nullptr, *limit_softness_ = nullptr;
    OFX::BooleanParam *clip_fill_ = nullptr, *anchor_ = nullptr, *ai_fill_ = nullptr, *mask_invert_ = nullptr;
    OFX::StringParam *ai_prompt_ = nullptr, *ai_status_ = nullptr, *ai_shot_ = nullptr;
    OFX::IntParam *ai_seed_ = nullptr, *ai_shown_ = nullptr, *ai_offset_ = nullptr, *ai_steps_ = nullptr, *ai_from_ = nullptr, *ai_to_ = nullptr;
    std::mutex note_mutex_;
    std::string last_note_;
};

mDeclarePluginFactory(RudraFactory, {}, { rudra_ofx::stop_engine(); });

void RudraFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels("RUDRA SDR to HDR", "RUDRA SDR to HDR", "RUDRA SDR to HDR");
    desc.setPluginGrouping("RUDRA");
    // The icon files beside the binary (Contents/Resources, by the build): the SVG first, then the PNG. A host
    // without the property finds them by the plug-in's identifier anyway.
    desc.getPropertySet().propSetString(kOfxPropIcon, "com.fxtdstudios.rudra.SdrToHdr.svg", 0, false);
    desc.getPropertySet().propSetString(kOfxPropIcon, "com.fxtdstudios.rudra.SdrToHdr.png", 1, false);
    desc.setPluginDescription(
        "SDR to scene-linear or PQ HDR by RUDRA (FXTD Studios): the analytic reconstruction, or the learned "
        "ACES 2.0 model through the RUDRA engine installed on this computer, with AI fill (experimental) drawn by "
        "the RUDRA app's AI fill. Licensed under the Apache License 2.0.");
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

void RudraFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) {
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);
    // The Mask (27 Nov 2026): where RUDRA acts, reconstruction and AI fill alike; outside it the plain inverse.
    // The general context only (Nuke's mask input); in Resolve the node's own window or qualifier limits it.
    if (context == OFX::eContextGeneral) {
        OFX::ClipDescriptor* mask = desc.defineClip(kMaskClip);
        mask->addSupportedComponent(OFX::ePixelComponentAlpha);
        mask->addSupportedComponent(OFX::ePixelComponentRGBA);
        mask->setTemporalClipAccess(false);
        mask->setSupportsTiles(false);
        mask->setOptional(true);
        mask->setIsMask(true);
    }

    OFX::ChoiceParamDescriptor* route = desc.defineChoiceParam(kRoute);
    route->setLabels("Route", "Route", "Reconstruction route");
    route->setHint("Analytic: in the plug-in, on the CPU, no model needed. Learned ACES 2.0: RUDRA's learned model, run by "
                   "the RUDRA engine on this computer (install RUDRA and its learned ACES 2.0 model); best for SDR made "
                   "through ACES 2.0.");
    route->appendOption("Analytic");
    route->appendOption("Learned ACES 2.0");
    route->setDefault(0);
    route->setAnimates(false);

    OFX::ChoiceParamDescriptor* source = desc.defineChoiceParam(kSource);
    source->setLabels("Source", "Source", "Source curve");
    source->setHint("The tone curve that made the SDR: BT.1886 for camera and broadcast video; sRGB for an SDR read as "
                    "sRGB; ACES, filmic or AgX for CG and games; BT.1886 shoulder to roll an ACES-style shoulder back out.");
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

    // The luminance limit (26 Nov 2026): the app's Reconstruct > Region, on both routes and for AI fill.
    OFX::GroupParamDescriptor* limits = desc.defineGroupParam(kLimitGroup);
    limits->setLabels("Luminance limit", "Luminance limit", "Luminance limit");
    limits->setOpen(false);
    OFX::ChoiceParamDescriptor* limit = desc.defineChoiceParam(kLimit);
    limit->setLabels("Limit", "Limit", "Limit the reconstruction by the SDR's brightness");
    limit->setHint("Where RUDRA reconstructs and where AI fill may draw, by the SDR's brightness (Rec.709 luma, 0 to 1): "
                   "the whole frame (Off), the highlights above a level, the shadows below one, both, or one band. "
                   "Outside it the picture is the plain inverse of the Source curve. The same as the RUDRA app's Region.");
    for (const char* m : {"Off", "Highlights above", "Shadows below", "Highlights and shadows", "A band"}) limit->appendOption(m);
    limit->setDefault(0);
    limit->setParent(*limits);
    struct Level { const char* name; const char* label; double def; const char* hint; };
    for (const Level& l : {Level{kHighlightsFrom, "Highlights from", 0.75, "SDR luma where the highlights start (the edge's middle)."},
                           Level{kShadowsBelow, "Shadows below", 0.15, "SDR luma where the shadows end (the edge's middle)."},
                           Level{kBandFrom, "Band from", 0.50, "The band's lower edge (SDR luma)."},
                           Level{kBandTo, "Band to", 0.90, "The band's upper edge (SDR luma)."}}) {
        OFX::DoubleParamDescriptor* d = desc.defineDoubleParam(l.name);
        d->setLabels(l.label, l.label, l.label);
        d->setHint(l.hint);
        d->setDefault(l.def);
        d->setRange(0.0, 1.0);
        d->setDisplayRange(0.0, 1.0);
        d->setParent(*limits);
    }
    OFX::DoubleParamDescriptor* softness = desc.defineDoubleParam(kLimitSoftness);
    softness->setLabels("Softness", "Softness", "Edge softness");
    softness->setHint("How wide each edge of the limit is, in SDR luma (0 = a hard edge).");
    softness->setDefault(0.10);
    softness->setRange(0.0, 0.5);
    softness->setDisplayRange(0.0, 0.5);
    softness->setParent(*limits);
    OFX::BooleanParamDescriptor* invert = desc.defineBooleanParam(kMaskInvert);
    invert->setLabels("Invert mask", "Invert mask", "Invert the Mask input");
    invert->setHint("With a Mask connected (Nuke): RUDRA acts outside the mask instead of inside it.");
    invert->setDefault(false);
    invert->setAnimates(false);
    invert->setParent(*limits);

    OFX::ChoiceParamDescriptor* alpha = desc.defineChoiceParam(kAlpha);
    alpha->setLabels("Alpha", "Alpha", "Output alpha");
    alpha->setHint("The output's alpha: the source's (as it came in), the luminance limit as a matte (1 where RUDRA "
                   "reconstructs), or what AI drew (rudra.generated: 1 where AI fill drew).");
    for (const char* m : {"Source alpha", "Luminance limit", "What AI drew"}) alpha->appendOption(m);
    alpha->setDefault(0);

    // AI fill (26 Nov 2026), experimental: drawn by the RUDRA app's AI fill v5 beside the host.
    OFX::GroupParamDescriptor* ai = desc.defineGroupParam(kAiGroup);
    ai->setLabels("AI fill (experimental)", "AI fill", "AI fill (experimental)");
    ai->setOpen(false);
    OFX::BooleanParamDescriptor* on = desc.defineBooleanParam(kAiFill);
    on->setLabels("AI fill", "AI fill", "AI fill");
    on->setHint("Shows the stored AI fill of this clip once it is made (Generate). AI draws inside the clipped "
                "highlights; every pixel it draws is generated, not recovered: review it before delivery. Needs the "
                "RUDRA app with AI fill v5 installed and a 16 GB+ NVIDIA GPU.");
    on->setDefault(false);
    on->setAnimates(false);
    on->setParent(*ai);
    OFX::StringParamDescriptor* prompt = desc.defineStringParam(kAiPrompt);
    prompt->setLabels("Prompt", "Prompt", "What to draw in the highlights");
    prompt->setHint("Optional: what the clipped areas show (\"overcast sky over the sea\"). Empty draws from the picture alone.");
    prompt->setDefault("");
    prompt->setAnimates(false);
    prompt->setParent(*ai);
    struct Edge { const char* name; const char* label; const char* hint; };
    for (const Edge& e : {Edge{kAiFrom, "From frame", "The first frame AI fill draws (0: the clip's first). Generate collects and draws only "
                                                    "From to To: a few frames to try it, the shot to deliver it."},
                          Edge{kAiTo, "To frame", "The last frame AI fill draws (0: the clip's last)."}}) {
        OFX::IntParamDescriptor* d = desc.defineIntParam(e.name);
        d->setLabels(e.label, e.label, e.label);
        d->setHint(e.hint);
        d->setDefault(0);
        d->setRange(-1000000, 1000000);
        d->setDisplayRange(0, 1000);
        d->setAnimates(false);
        d->setParent(*ai);
    }
    OFX::IntParamDescriptor* steps = desc.defineIntParam(kAiSteps);
    steps->setLabels("Steps", "Steps", "Denoising steps");
    steps->setHint("How many steps the fill takes: 30 is what AI fill v5 was tested with (about 20 minutes a 100 frames). "
                   "Fewer is faster; below 20 it can draw stripes. Changing it makes a stored fill out of date.");
    steps->setDefault(30);
    steps->setRange(10, 50);
    steps->setDisplayRange(10, 50);
    steps->setAnimates(false);
    steps->setParent(*ai);
    OFX::IntParamDescriptor* seed = desc.defineIntParam(kAiSeed);
    seed->setLabels("Seed", "Seed", "Seed");
    seed->setHint("Another seed draws another fill.");
    seed->setDefault(0);
    seed->setRange(0, 999999);
    seed->setDisplayRange(0, 100);
    seed->setAnimates(false);
    seed->setParent(*ai);
    struct Button { const char* name; const char* label; const char* hint; };
    for (const Button& b : {Button{kAiGenerate, "Generate", "Start AI fill for this clip: then play or render the whole clip once at full "
                                                             "size; when every frame is in, the fill is drawn beside the host (about 20 "
                                                             "minutes a 100 frames of 1080p) and shown from then on. In Resolve, put the playhead on "
                                                             "the first frame to draw (From, or the clip's first) before pressing it."},
                            Button{kAiCancel, "Cancel", "Stop collecting, or stop the fill being drawn."},
                            Button{kAiRefresh, "Refresh status", "Read the AI fill's progress again."},
                            Button{kAiClear, "Clear stored fill", "Delete this node's stored frames and fills from the disk."}}) {
        OFX::PushButtonParamDescriptor* d = desc.definePushButtonParam(b.name);
        d->setLabels(b.label, b.label, b.label);
        d->setHint(b.hint);
        d->setParent(*ai);
    }
    OFX::StringParamDescriptor* status = desc.defineStringParam(kAiStatus);
    status->setLabels("Status", "Status", "AI fill status");
    status->setDefault("AI fill is off.");
    // A one-line field, greyed (27 Nov 2026): Nuke draws a label parameter once, from its default, and never showed
    // what the plug-in wrote into it ("AI fill is off." stayed after Generate).
    status->setStringType(OFX::eStringTypeSingleLine);
    status->setEnabled(false);
    status->setAnimates(false);
    status->setEvaluateOnChange(false);
    status->setParent(*ai);
    OFX::StringParamDescriptor* shot = desc.defineStringParam(kAiShot);
    shot->setLabels("Shot id", "Shot id", "AI fill shot id");
    shot->setDefault("");
    shot->setIsSecret(true);
    shot->setAnimates(false);
    shot->setEvaluateOnChange(false);
    shot->setParent(*ai);
    // A number the host sees change when the AI fill's picture may have (changedParam): it renders again.
    OFX::IntParamDescriptor* shown = desc.defineIntParam(kAiShown);
    shown->setLabels("AI fill picture", "AI fill picture", "AI fill picture");
    shown->setDefault(0);
    shown->setIsSecret(true);
    shown->setAnimates(false);
    shown->setParent(*ai);
    // Resolve's frame numbers against the clip's range (1.3.2): set by Generate, cleared by Clear stored fill.
    OFX::IntParamDescriptor* offset = desc.defineIntParam(kAiOffset);
    offset->setLabels("AI fill frame offset", "AI fill frame offset", "AI fill frame offset");
    offset->setDefault(0);
    offset->setIsSecret(true);
    offset->setAnimates(false);
    offset->setEvaluateOnChange(false);
    offset->setParent(*ai);
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
