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
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ofxColour.h"
#include "ofx-native-v1.5_aces-v1.3_ocio-v2.3.h"
#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsSupportPrivate.h"   // the property suite and an instance from its handle, for colour management (1.3.2)

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
                      *kAiStatus = "aiStatus", *kAiShot = "aiShot", *kAiShown = "aiShown", *kAiStart = "aiStart", *kAiSpan = "aiSpan", *kAiSteps = "aiSteps",
                      *kAiFrom = "aiFrom", *kAiTo = "aiTo", *kAiSuggest = "aiSuggest", *kAiStrength = "aiStrength";
// Suggest prompt (1.3.2): how long the button waits for Florence-2 before it leaves the answer to Refresh status.
constexpr double kSuggestWait = 45.0;
// The Mask input (27 Nov 2026, the general context: Nuke) and its invert switch.
constexpr const char* kMaskClip = "Mask", *kMaskInvert = "maskInvert";

int clamp_index(int i, int n) { return std::max(0, std::min(i, n - 1)); }

// Why a render failed, where it can be read (25 Nov 2026): Nuke shows only "Effect raised an
// error during render", not the persistent message, so the reason also goes to the host's
// console (stderr) and to <temp>/rudra_ofx.log (RUDRA_OFX_LOG_DIR instead, when set: the tests', 1.3.2).
void log_failure(const std::string& why) {
    std::fprintf(stderr, "RUDRA SDR to HDR: %s\n", why.c_str());
    std::fflush(stderr);
    std::error_code ec;
    const char* aside = std::getenv("RUDRA_OFX_LOG_DIR");
    const std::filesystem::path dir = aside && *aside ? std::filesystem::path(aside) : std::filesystem::temp_directory_path(ec);
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
        // The parameters added after the first AI fill build (27 Nov 2026 on): fetched only when the host has them, so
        // an instance a host made from an older description of the plug-in (its plug-in cache) still opens (1.3.2).
        ai_shown_ = paramExists(kAiShown) ? fetchIntParam(kAiShown) : nullptr;
        ai_start_ = paramExists(kAiStart) ? fetchIntParam(kAiStart) : nullptr;
        ai_span_ = paramExists(kAiSpan) ? fetchIntParam(kAiSpan) : nullptr;
        ai_steps_ = paramExists(kAiSteps) ? fetchIntParam(kAiSteps) : nullptr;
        ai_from_ = paramExists(kAiFrom) ? fetchIntParam(kAiFrom) : nullptr;
        ai_to_ = paramExists(kAiTo) ? fetchIntParam(kAiTo) : nullptr;
        ai_strength_ = paramExists(kAiStrength) ? fetchDoubleParam(kAiStrength) : nullptr;
        mask_invert_ = paramExists(kMaskInvert) ? fetchBooleanParam(kMaskInvert) : nullptr;
        if (!ai_strength_ || !ai_span_ || !ai_steps_)
            log_failure("this host describes RUDRA without some of its parameters (an older description in its plug-in "
                        "cache): clear the host's OFX plug-in cache and open the project again.");
        if (getContext() == OFX::eContextGeneral) mask_ = fetchClip(kMaskClip);   // Resolve (filter): the node's own key
        update_enabled();
    }

    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) {
        if (name == kAiStatus || name == kAiShot || name == kAiShown || name == kAiStart || name == kAiSpan) return;   // its own writes
        if (name == kAiPrompt && args.reason == OFX::eChangeUserEdit) suggest_pending_ = false;   // typed: no late suggestion over it
        if (name == kAiSuggest) {   // 1.3.2: it writes the Prompt and the Status itself
            suggest_prompt(args);
            return;
        }
        if (name == kRoute || name == kLimit) update_enabled();
        rudra_ofx::Settings s = settings_at(args.time, true);
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
            // Resolve (1.3.2): it renders a clip at its source media's frame numbers (129, 130, ...) and does not say
            // where a trimmed clip starts, so Generate there fixes the frames from From (else the playhead) to To (else
            // the clip's end), kept in two hidden parameters for the renders that collect them (resolve_frames).
            if (resolve_host() && ai_start_ && ai_span_) {
                const rudra_ofx::FrameSpan r = resolve_span(args.time);
                ai_start_->setValue(r.first);
                ai_span_->setValue(r.last - r.first + 1);
                s = settings_at(args.time, true);
            }
            log_failure(host_frames(args.time, s));
            why = rudra_ofx::ai_generate(s);
            render_again = true;   // a refusal too: the guide in the picture says why (1.3.2)
        } else if (name == kAiCancel) {
            why = rudra_ofx::ai_cancel(s);
            render_again = true;
        } else if (name == kAiClear) {
            why = rudra_ofx::ai_clear(s);
            if (why.empty() && ai_span_) ai_span_->setValue(0);
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
        if (render_again && ai_shown_) {
            int shown = 0;
            ai_shown_->getValue(shown);
            ai_shown_->setValue(shown % 1000000 + 1);
        }
        // A suggestion that was still being made when Suggest prompt stopped waiting (1.3.2).
        if (name == kAiRefresh && suggest_pending_) finish_suggestion(rudra_ofx::ai_suggestion(s), suggest_frame_);
    }

    void render(const OFX::RenderArguments& args) {
        const std::uint64_t ticket = ++renders_;
        std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
        std::unique_ptr<OFX::Image> src(src_->fetchImage(args.time));
        if (!dst || !src) {
            // A render the host abandons says nothing (1.3.2, 27 Nov 2026): Nuke abandons one when a knob moves while
            // it runs (a Grade's gain dragged above RUDRA) and hands it no picture, and the error then put on the node
            // ("no picture from the input") could stay after the picture had come back.
            if (abort()) return;
            fail(ticket, !src ? "no picture from the input (check the node above RUDRA for its own error)."
                              : "the host gave no output image.");
        }
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
        // A clipped picture that arrives re-encoded (1.3.2): said on the node, and to AI fill (its guide and Generate).
        s.flat_top = rudra_ofx::flat_top_below_white(top, src_row, w, h);
        if (s.full_size) flat_top_ = s.flat_top;
        rudra_ofx::Outcome result;
        try {
            result = rudra_ofx::render(s, top, src_row, out.data(), std::ptrdiff_t(w) * 4, w, h);
        } catch (const std::bad_alloc&) {
            result.error = "out of memory for a " + std::to_string(w) + " x " + std::to_string(h) + " frame.";
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        if (!result.error.empty())
            fail(ticket, result.error + " (" + s.route + " route, " + std::to_string(w) + " x " + std::to_string(h) + ")");
        if (s.full_size) keep_frame(top, src_row, w, h, s.frame);   // Suggest prompt's frame where the host gives none there
        if (!result.log.empty()) log_note_once(result.log);           // each frame AI fill stored (1.3.2)
        if (result.log.empty()) log_guide(result.guide);              // not each frame's count while collecting
        if (const std::string problem = rudra_ofx::input_problem(s); !problem.empty()) {
            log_note_once(problem);
            say(ticket, OFX::Message::eMessageWarning, "RUDRA: " + problem);
        } else if (!result.note.empty() && (result.note.rfind("AI fill failed", 0) == 0 || result.note.find("out of date") != std::string::npos ||
                                     result.note.find("not ") != std::string::npos ||
                                     result.note.find("full-size renders only") != std::string::npos)) {
            log_note_once(result.note);   // in the log as well: Nuke shows a node's warning only on the node
            say(ticket, OFX::Message::eMessageWarning, "RUDRA: " + result.note);
        } else {
            say(ticket, OFX::Message::eMessageWarning, "");
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

    // OpenFX colour management (1.3.2, 27 Nov 2026): the style the host chose for this instance ("" without it).
    std::string colour_style() { return getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false); }

    // kOfxImageEffectActionGetClipPreferences, after the support library's own: the source as the SDR's Rec.709 display
    // codes. False when the host manages no colour.
    bool prefer_colourspaces(OfxPropertySetHandle out) {
        const std::vector<std::string> want = rudra_ofx::colour_source_preferences(colour_style());
        if (want.empty() || !out || !OFX::Private::gPropSuite) return false;
        std::vector<const char*> names;
        for (const auto& w : want) names.push_back(w.c_str());
        const std::string prop = std::string(kOfxImageClipPropPreferredColourspaces) + "_" + kOfxImageEffectSimpleSourceClipName;
        return OFX::Private::gPropSuite->propSetStringN(out, prop.c_str(), int(names.size()), names.data()) == kOfxStatOK;
    }

    // kOfxImageEffectActionGetOutputColourspace: what Output puts out, in the host's colour management.
    OfxStatus output_colourspace(OfxPropertySetHandle out) {
        int output = 0;
        output_->getValue(output);
        const std::string cs = rudra_ofx::colour_output(kOutputs[clamp_index(output, 3)], colour_style());
        if (cs.empty() || !out || !OFX::Private::gPropSuite) return kOfxStatReplyDefault;
        return OFX::Private::gPropSuite->propSetString(out, kOfxImageClipPropColourspace, 0, cs.c_str()) == kOfxStatOK ? kOfxStatOK
                                                                                                                      : kOfxStatFailed;
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
        if (mask_invert_) mask_invert_->getValueAtTime(time, invert);
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

    // What AI fill's guide said in the picture (1.3.2), each time it changes: a session's log shows what its user saw.
    void log_guide(const std::string& guide) {
        std::lock_guard<std::mutex> lock(note_mutex_);
        if (guide == last_guide_) return;
        last_guide_ = guide;
        if (!guide.empty()) log_failure("Shown in the picture: " + guide);
    }

    [[noreturn]] void fail(std::uint64_t ticket, const std::string& why) {
        log_failure(why);
        say(ticket, OFX::Message::eMessageError, "RUDRA: " + why);
        OFX::throwSuiteStatusException(kOfxStatFailed);
        throw std::runtime_error(why);   // not reached
    }

    // The node's message (1.3.2): the word of the render that started last, as renders run side by side
    // (eRenderFullySafe) and one that ends late must not put back a message a later one cleared. "" clears it. Only
    // renders call it, and the host is called outside the lock: a host may answer these calls on its own UI thread,
    // and a lock held across them, taken by changedParam too, could stop both (27 Nov 2026: Resolve stopped
    // responding with that build).
    void say(std::uint64_t ticket, OFX::Message::MessageTypeEnum type, const std::string& text) {
        {
            std::lock_guard<std::mutex> lock(message_mutex_);
            if (ticket < said_) return;
            said_ = ticket;
        }
        if (text.empty())
            clearPersistentMessage();
        else
            setPersistentMessage(type, "", text);
    }

    // `main_thread`: changedParam's (it may ask the host's timeline where Resolve's clip ends); a render's is false.
    rudra_ofx::Settings settings_at(double time, bool main_thread = false) {
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
        if (ai_steps_) ai_steps_->getValue(s.steps);
        if (ai_strength_) ai_strength_->getValue(s.prompt_strength);
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
        if (ai_from_) ai_from_->getValue(from);
        if (ai_to_) ai_to_->getValue(to);
        if (from != 0) s.first_frame = std::clamp(from, s.first_frame, s.last_frame);
        if (to != 0) s.last_frame = std::clamp(to, s.first_frame, s.last_frame);
        s.to_frame = to;
        s.frame = int(std::lround(time));
        s.host_label = host_label();
        s.flat_top = flat_top_;   // the last full-size picture's (a render measures its own)
        s.colour_style = colour_style();
        if (!s.colour_style.empty()) s.input_colourspace = src_->getPropertySet().propGetString(kOfxImageClipPropColourspace, false);
        // Resolve (1.3.2): the frames Generate fixed; before one, what it would take now (a render cannot ask the host's
        // timeline, and has nothing to collect yet: its own frame).
        if (resolve_host()) {
            int start = 0, span = 0;
            if (ai_start_) ai_start_->getValue(start);
            if (ai_span_) ai_span_->getValue(span);
            if (span > 0) {
                s.first_frame = start, s.last_frame = start + span - 1;
            } else if (main_thread) {
                const rudra_ofx::FrameSpan r = resolve_span(time);
                s.first_frame = r.first, s.last_frame = r.last;
            } else {
                s.first_frame = s.last_frame = s.frame;
            }
        }
        return s;
    }

    // Resolve's AI fill frames for Generate at `time` (rudra_ofx::resolve_frames): From, else the playhead, to To, else
    // the clip's end as the host's timeline gives it, else its frame range's end.
    rudra_ofx::FrameSpan resolve_span(double time) {
        int from = 0, to = 0;
        if (ai_from_) ai_from_->getValue(from);
        if (ai_to_) ai_to_->getValue(to);
        double t1 = 0.0, t2 = 0.0;
        timeLineGetBounds(t1, t2);
        return rudra_ofx::resolve_frames(int(std::lround(time)), from, to, t2, src_->getFrameRange().max);
    }

    // What the host said about its frame numbers when Generate was pressed (1.3.2), for the log.
    std::string host_frames(double time, const rudra_ofx::Settings& s) {
        const OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
        const OfxRangeD range = src_->getFrameRange();
        double t1 = 0.0, t2 = 0.0;
        timeLineGetBounds(t1, t2);
        char b[512];
        std::snprintf(b, sizeof b,
                      "Generate in %s %d.%d at frame %d: AI fill frames %d to %d (%d); the clip's frame range %.0f to %.0f, "
                      "the timeline %.0f to %.0f",
                      host ? host->hostName.c_str() : "?", host ? host->versionMajor : 0, host ? host->versionMinor : 0,
                      int(std::lround(time)), s.first_frame, s.last_frame, s.last_frame - s.first_frame + 1, range.min, range.max,
                      t1, t2);
        return b;
    }

    // The SDR as 8-bit RGB codes, the top row first (Suggest prompt's frame): RGBA floats, a row `row` floats apart.
    static void to_rgb8(const float* top, std::ptrdiff_t row, int w, int h, std::vector<unsigned char>& rgb) {
        rgb.resize(std::size_t(w) * std::size_t(h) * 3);
        for (int y = 0; y < h; ++y) {
            const float* r = top + std::ptrdiff_t(y) * row;
            unsigned char* o = rgb.data() + std::size_t(y) * std::size_t(w) * 3;
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    const float v = r[std::ptrdiff_t(x) * 4 + c];
                    o[std::size_t(x) * 3 + std::size_t(c)] =
                        static_cast<unsigned char>(std::lround((std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f) * 255.0f));
                }
        }
    }

    // The last frame rendered at full size, for Suggest prompt in a host that gives no image outside a render.
    void keep_frame(const float* top, std::ptrdiff_t row, int w, int h, int frame) {
        std::vector<unsigned char> rgb;
        to_rgb8(top, row, w, h, rgb);
        std::lock_guard<std::mutex> lock(frame_mutex_);
        last_rgb_.swap(rgb);
        last_w_ = w, last_h_ = h, last_frame_ = frame;
    }

    // This frame from the host here, outside a render (the OFX spec allows it in changedParam; Nuke gives it). False
    // when the host gives none.
    bool frame_now(double time, std::vector<unsigned char>& rgb, int& w, int& h) {
        try {
            std::unique_ptr<OFX::Image> img(src_->fetchImage(time));
            if (!img || img->getPixelDepth() != OFX::eBitDepthFloat || img->getPixelComponents() != OFX::ePixelComponentRGBA)
                return false;
            const OfxRectI b = img->getBounds();
            w = b.x2 - b.x1, h = b.y2 - b.y1;
            if (w <= 0 || h <= 0) return false;
            const auto* top = static_cast<const float*>(img->getPixelAddress(b.x1, b.y2 - 1));
            if (!top) return false;
            to_rgb8(top, -(img->getRowBytes() / std::ptrdiff_t(sizeof(float))), w, h, rgb);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Suggest prompt (1.3.2): this frame (or the last one shown at full size) to RUDRA's prompt suggester, waiting up
    // to kSuggestWait with the host's progress bar; Prompt takes the first suggestion.
    void suggest_prompt(const OFX::InstanceChangedArgs& args) {
        rudra_ofx::Settings s = settings_at(args.time);
        if (s.shot_id.empty()) {
            s.shot_id = rudra_ofx::new_shot_id();
            ai_shot_->setValue(s.shot_id);
        }
        std::vector<unsigned char> rgb;
        int w = 0, h = 0, frame = int(std::lround(args.time));
        // Resolve: the frame its viewer last rendered (the playhead's), not an image fetched outside a render, which
        // Resolve may not give or may stall its interface for.
        if (resolve_host() || !frame_now(args.time, rgb, w, h)) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            rgb = last_rgb_;
            w = last_w_, h = last_h_, frame = last_frame_;
        }
        if (rgb.empty()) {
            say_suggestion("Suggest prompt: show a frame of this clip at full size first, then press it again.");
            return;
        }
        progressStart("RUDRA: suggesting a prompt");
        const auto r = rudra_ofx::ai_suggest(s, rgb.data(), w, h, kSuggestWait, [this](double f) { return progressUpdate(f); });
        progressEnd();
        finish_suggestion(r, frame);
    }

    void finish_suggestion(const rudra_ofx::PromptSuggestion& r, int frame) {
        const std::string at = "frame " + std::to_string(frame);
        if (!r.ready && r.error.empty()) {
            suggest_pending_ = true;
            suggest_frame_ = frame;
            say_suggestion("Suggest prompt: still reading " + at + " (Florence-2 loads the first time): press Refresh status in a few seconds.");
            return;
        }
        suggest_pending_ = false;
        if (!r.error.empty()) {
            say_suggestion("Suggest prompt (" + at + "): " + r.error);
            return;
        }
        ai_prompt_->setValue(r.text);   // as if typed: the stored fill's key follows it
        std::string msg = "Prompt suggested from " + at + ": " + r.label;
        if (!r.reason.empty()) msg += " (" + r.reason + ")";
        for (std::size_t i = 0; i < r.others.size(); ++i) msg += (i ? ", " : ". Others: ") + r.others[i];
        if (!r.note.empty()) msg += ". " + r.note;
        say_suggestion(msg + ". Edit it as you like.");
    }

    void say_suggestion(const std::string& m) {
        ai_status_->setValue(m);
        log_failure(m);   // the log too: Resolve may not show the status field
    }

    static bool resolve_host() {
        const OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
        return host && host->hostName.find("DaVinci") != std::string::npos;
    }

    // The host's name for AI fill's guide in the picture ("Keep Resolve open"); "" for one it does not know.
    static std::string host_label() {
        const OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
        if (!host) return {};
        if (host->hostName.find("DaVinci") != std::string::npos) return "Resolve";
        std::string name = host->hostName;
        for (char& c : name) c = char(std::tolower(static_cast<unsigned char>(c)));
        return name.find("nuke") != std::string::npos ? "Nuke" : std::string();
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
    OFX::IntParam *ai_seed_ = nullptr, *ai_shown_ = nullptr, *ai_start_ = nullptr, *ai_span_ = nullptr, *ai_steps_ = nullptr,
                  *ai_from_ = nullptr, *ai_to_ = nullptr;
    OFX::DoubleParam* ai_strength_ = nullptr;
    std::mutex note_mutex_;
    std::string last_note_;
    std::string last_guide_;
    std::atomic<std::uint64_t> renders_{0};   // each render's ticket (say)
    std::atomic<int> flat_top_{0};            // the last full-size picture's flat_top_below_white, for Generate
    std::mutex message_mutex_;
    std::uint64_t said_ = 0;                  // the ticket of the render whose message the node shows
    std::mutex frame_mutex_;                 // the last full-size frame (keep_frame), for Suggest prompt
    std::vector<unsigned char> last_rgb_;
    int last_w_ = 0, last_h_ = 0, last_frame_ = 0;
    bool suggest_pending_ = false;           // a suggestion still being made when the button stopped waiting
    int suggest_frame_ = 0;
};

// The factory (mDeclarePluginFactory, written out): its entry point answers OpenFX colour management (1.3.2), which the
// support library predates, and hands every other action to the support library.
class RudraFactory : public OFX::PluginFactoryHelper<RudraFactory> {
public:
    RudraFactory(const std::string& id, unsigned int verMaj, unsigned int verMin) : OFX::PluginFactoryHelper<RudraFactory>(id, verMaj, verMin) {}
    void load() override {}
    void unload() override { rudra_ofx::stop_engine(); }
    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override;
    OfxPluginEntryPoint* getMainEntry() override { return &entry; }

private:
    static RudraEffect* effect_of(const void* handle) {
        try {
            return dynamic_cast<RudraEffect*>(OFX::Private::retrieveImageEffectPointer(static_cast<OfxImageEffectHandle>(const_cast<void*>(handle))));
        } catch (...) {
            return nullptr;
        }
    }
    static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
        const bool output_cs = action && std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0;
        if (output_cs) {
            try {
                RudraEffect* fx = effect_of(handle);
                return fx ? fx->output_colourspace(out) : kOfxStatReplyDefault;
            } catch (...) {
                return kOfxStatFailed;
            }
        }
        const OfxStatus st = FactoryMainEntryHelper<RudraFactory>::mainEntry(action, handle, in, out);
        if (action && std::strcmp(action, kOfxImageEffectActionGetClipPreferences) == 0 && (st == kOfxStatOK || st == kOfxStatReplyDefault)) {
            try {
                if (RudraEffect* fx = effect_of(handle); fx && fx->prefer_colourspaces(out)) return kOfxStatOK;
            } catch (...) {
            }
        }
        return st;
    }
};

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
    // OpenFX colour management (1.3.2): a host that has it hands RUDRA the SDR's Rec.709 display codes and takes its
    // output for what Output says. Hosts without it do not know these properties: their refusal is not an error.
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementStyle, kOfxImageEffectColourManagementCore, 0, false);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementAvailableConfigs, kOfxConfigIdentifier, 0, false);
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
    desc.addClipPreferencesSlaveParam(*output);   // 1.3.2: Output decides the colourspace the host takes the output in

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
    // 1.3.2: the app's Suggest a prompt and Deliver's Prompt strength, here too.
    OFX::PushButtonParamDescriptor* suggest = desc.definePushButtonParam(kAiSuggest);
    suggest->setLabels("Suggest prompt", "Suggest prompt", "Suggest a prompt from this frame");
    suggest->setHint("Writes a prompt for this frame's clipped highlights into Prompt, as RUDRA's Suggest a prompt does: "
                     "Florence-2 describes the frame when RUDRA's AI fill setup installed it, else the presets the clipped "
                     "areas point to. Edit it as you like.");
    suggest->setParent(*ai);
    OFX::DoubleParamDescriptor* prompt_strength = desc.defineDoubleParam(kAiStrength);
    prompt_strength->setLabels("Prompt strength", "Prompt strength", "Prompt strength");
    prompt_strength->setHint("How strongly the Prompt steers what AI draws in the clipped areas: 1 is how AI fill v5 was tested, "
                      "0 ignores the prompt, above 1 follows it more closely (experimental). Changing it makes a stored "
                      "fill out of date.");
    prompt_strength->setDefault(1.0);
    prompt_strength->setRange(0.0, 2.0);
    prompt_strength->setDisplayRange(0.0, 2.0);
    prompt_strength->setIncrement(0.1);
    prompt_strength->setDigits(2);
    prompt_strength->setAnimates(false);
    prompt_strength->setParent(*ai);
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
    shot->setEvaluateOnChange(true);    // 1.3.2: as aiStart / aiSpan, the renders need it
    shot->setParent(*ai);
    // A number the host sees change when the AI fill's picture may have (changedParam): it renders again.
    OFX::IntParamDescriptor* shown = desc.defineIntParam(kAiShown);
    shown->setLabels("AI fill picture", "AI fill picture", "AI fill picture");
    shown->setDefault(0);
    shown->setIsSecret(true);
    shown->setAnimates(false);
    shown->setParent(*ai);
    // Resolve's AI fill frames (1.3.2): the first and how many, set by Generate (resolve_frames), cleared by Clear stored fill.
    for (const char* name : {kAiStart, kAiSpan}) {
        OFX::IntParamDescriptor* d = desc.defineIntParam(name);
        d->setLabels(name, name, name);
        d->setDefault(0);
        d->setRange(-100000000, 100000000);
        d->setIsSecret(true);
        d->setAnimates(false);
        d->setEvaluateOnChange(true);   // a host may pass the renders only what evaluates: they collect by these
        d->setParent(*ai);
    }
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
