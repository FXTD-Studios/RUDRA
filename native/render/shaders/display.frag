#version 440
// The display pass (docs/view.spec.md section 2). A line-for-line port of
// native/core/src/view.cpp, the reference it is held to (rudra-gpu-parity),
// and of the DISPLAY shader in ui/compositor.js, which it replaces.
//
// Texels are fetched by pixel, so output row r is image row r on every API,
// as in composite.frag. Presenting to a swapchain puts the flip, if the API
// needs one, in the vertex stage (render/, Phase 2 step 9), never here.
//
// Revision 7 (17 Nov 2026): built a second time at run time for an OpenColorIO view
// (render/src/viewer_window.cpp, QShaderBaker): RUDRA_OCIO defined, and OpenColorIO's
// own GLSL for the view (vec4 ocioView(vec4)) and its decode to light
// (vec4 ocioDecode(vec4)) put in at the line below, with their samplers from
// binding 6 on. The build-time shader has neither and never takes that path.
// @OCIO@

layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform View {
    vec4 view;    // x mode (0 image, 1 false colour, 2 difference, 3 invented, 4 range), y exposure (10000 / view peak), z wipe (< 0 off), w wipe half width
    vec4 extra;   // x log2(1 + diff gain), y frame width, z 1 to show the baseline, w 1 to tint the changes
    vec4 target;  // x path (0 SDR, 1 scRGB, 2 HDR10, 3 EDR), y ceiling nits, z nits per 1.0 (linear paths)
    vec4 pic[3];  // rows of source primaries -> swapchain primaries (xyz)
    vec4 gfx[3];  // rows of Rec.709 -> swapchain primaries, for the overlays
    vec4 anchor;  // roadmap 3.3: x 1 to anchor the model's picture, y knee (SDR max code), z softness, w hold gain
    vec4 paint;   // roadmap 3.3: x the painted mask's channel (< 0 none), yzw its tint (sRGB)
    vec4 look;    // revision 6: x 2^view_ev, y 1 / view gamma, z 1 when the gamma is on, w 1 when sdrLight is bound
    vec4 ab;      // revision 6: x the shown source, y the wipe's left, z its right (0 model, 1 baseline, 2 sdr), w zebras (1 clamp + 2 clip + 4 gamut + 8 invalid)
    vec4 region;  // RUDRA 1.1: x 1 to tint the reconstruction region, yzw its tint (sRGB)
    // Revision 7 (core/view_ocio.hpp).
    vec4 ocio;    // x 1: an OCIO view is set, y nits per decoded Y, z the clip code (huge: none, EDR), w 1: its light is decoded
    vec4 work[3]; // rows of source -> working space x 10000 / 203
    vec4 dec[3];  // rows of CIE XYZ -> Rec.709 x nits per Y
    vec4 chk[3];  // rows of source -> the primaries the gamut check reads
    vec4 ap0[3];  // rows of source -> AP0
};

layout(binding = 3) uniform sampler2D masks;      // painted masks (3.3), RGBA8, a band per channel
layout(binding = 1) uniform sampler2D model;      // rgb reconstruction, network units; a the SDR's Rec.2020 luma, linearised (the anchor's target)
layout(binding = 2) uniform sampler2D baseline;   // rgb analytic baseline, a the SDR's max code (Invented)
layout(binding = 5) uniform sampler2D regionW;    // RUDRA 1.1: r the region's weight (core region_overlay_rgba8)
layout(binding = 4) uniform sampler2D sdrLight;   // revision 6: rgb the SDR as light (core/view.cpp sdr_light_rgba), a its clamp flags

const float kPeak = 10000.0;
// core/compare.hpp change_weight() and core/view.hpp kChange*.
const float kChangeFloor = 0.05;
const vec3 kChangeUp = vec3(0.95, 0.62, 0.28);
const vec3 kChangeDown = vec3(0.32, 0.56, 0.95);
const float kChangeTintMix = 0.55;
// core/view.hpp kMap*, core/compare.hpp kSdrClipCode / kSdrCrushCode.
const float kMapGrey = 0.6;
const float kMapMix = 0.85;
const vec3 kInventedColour = vec3(0.92, 0.30, 0.86);
const vec3 kReinterpretedColour = vec3(0.25, 0.78, 0.86);
const float kSdrClipCode = 254.0 / 255.0;
const float kSdrCrushCode = 1.0 / 255.0;

float lum2020(vec3 c) { return dot(c, vec3(0.2627, 0.6780, 0.0593)); }

// Rows in the order core/view.cpp sums them.
vec3 rowsOf(vec4 r0, vec4 r1, vec4 r2, vec3 n) {
    return vec3(r0.x * n.x + r0.y * n.y + r0.z * n.z,
                r1.x * n.x + r1.y * n.y + r1.z * n.z,
                r2.x * n.x + r2.y * n.y + r2.z * n.z);
}

// core/view_ocio.cpp below_floor: a channel below -(floor + relative x the largest).
bool belowFloor(vec3 v) {
    float mx = max(max(v.r, v.g), v.b);
    float mn = min(min(v.r, v.g), v.b);
    return mn < -(1e-7 + 1e-3 * max(mx, 0.0));
}

float linearToSrgb(float x) {
    x = clamp(x, 0.0, 1.0);
    return x > 0.0031308 ? 1.055 * pow(x, 1.0 / 2.4) - 0.055 : 12.92 * x;
}

float srgbToLinear(float c) {
    c = clamp(c, 0.0, 1.0);
    return c > 0.04045 ? pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}

// SMPTE ST 2084 inverse EOTF of absolute nits, as core/hdr10.cpp pq_oetf.
float pq(float nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float p = pow(clamp(nits / 10000.0, 0.0, 1.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

float encodeNits(float n) { return int(target.x + 0.5) == 2 ? pq(n) : n / target.z; }

float changeWeight(vec3 m, vec3 b) {
    float mn = max(max(m.r, m.g), m.b) * kPeak;
    float bn = max(max(b.r, b.g), b.b) * kPeak;
    float d = log2((mn + kChangeFloor) / (bn + kChangeFloor));
    float w = clamp((abs(d) - 0.05) / 0.1, 0.0, 1.0);
    return d < 0.0 ? -w : w;
}

// core/view.hpp revision 6.
const vec3 kClampAllColour = vec3(0.95, 0.20, 0.20);
const vec3 kClampSomeColour = vec3(0.98, 0.55, 0.15);
const vec3 kCrushColour = vec3(0.20, 0.45, 0.98);
const vec3 kScreenClipColour = vec3(0.98, 0.90, 0.15);
const float kZebraMix = 0.6;
const float kRangeHatchMix = 0.7;
// core/view_ocio.hpp revision 7.
const vec3 kGamutColour = vec3(0.90, 0.25, 0.95);
const vec3 kInvalidColour = vec3(0.15, 1.00, 0.35);
const float kHalfMax = 65504.0;

// The overlay colour col in what c holds: the code itself, or light at the SDR white.
vec3 overlay(vec3 col, bool codes) {
    return codes ? col : 203.0 * vec3(srgbToLinear(col.r), srgbToLinear(col.g), srgbToLinear(col.b));
}

vec3 rangeColour(float s) {
    if (!(s >= -6.0)) return vec3(0.08, 0.08, 0.10);
    if (s < 0.0) return vec3(0.15 + 0.45 * (s + 6.0) / 6.0);
    if (s < 1.0) return vec3(0.20, 0.55, 0.85);
    if (s < 2.0) return vec3(0.25, 0.75, 0.45);
    if (s < 3.0) return vec3(0.95, 0.85, 0.25);
    if (s < 4.0) return vec3(0.95, 0.55, 0.20);
    return vec3(0.90, 0.25, 0.25);
}

vec3 falseColour(float n) {
    if (n <     0.1) return vec3(0.169, 0.122, 0.239);
    if (n <     1.0) return vec3(0.184, 0.294, 0.561);
    if (n <    10.0) return vec3(0.184, 0.561, 0.722);
    if (n <   100.0) return vec3(0.200, 0.627, 0.416);
    if (n <   160.0) return vec3(0.604, 0.655, 0.698);
    if (n <   250.0) return vec3(0.914, 0.929, 0.945);
    if (n <  1000.0) return vec3(0.910, 0.765, 0.290);
    if (n <  4000.0) return vec3(0.910, 0.529, 0.227);
    if (n < 10000.0) return vec3(0.816, 0.263, 0.184);
    return vec3(0.780, 0.290, 0.780);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float u = (float(p.x) + 0.5) / extra.y;
    bool wiping = view.z >= 0.0;
    vec4 b4 = texelFetch(baseline, p, 0);
    vec3 b = b4.rgb;
    vec4 m4 = texelFetch(model, p, 0);
    vec3 m = m4.rgb;
    // Revision 6: sources 0 model, 1 baseline, 2 the SDR as light (the
    // baseline when no SDR is bound), the wipe's A | B.
    bool haveSdr = look.w > 0.5;
    vec4 l4 = haveSdr ? texelFetch(sdrLight, p, 0) : vec4(0.0);
    int flags = int(l4.a + 0.5);
    int shown = int(ab.x + 0.5);
    int which = wiping ? (u < view.z ? int(ab.y + 0.5) : int(ab.z + 0.5)) : shown;
    if (which == 2 && !haveSdr) which = 1;
    if (shown == 2 && !haveSdr) shown = 1;
    vec3 src = wiping ? m : (shown == 0 ? m : shown == 1 ? b : l4.rgb);
    vec3 h = which == 0 ? m : which == 1 ? b : l4.rgb;
    // The anchor (core/view.cpp anchor_gain_f) on the model's picture only.
    if (anchor.x > 0.5 && which == 0) {
        float eps = 1e-4;
        float target = m4.a * 203.0;
        float actual = lum2020(m) * kPeak;
        float gain = (target + eps) / (actual + eps);
        float t = clamp((b4.a - (anchor.y - anchor.z)) / (2.0 * anchor.z), 0.0, 1.0);
        float ramp = t * t * (3.0 - 2.0 * t);
        gain = gain * (1.0 - ramp) + anchor.w * ramp;
        h = m * gain;
    }
    int mode = int(view.x + 0.5);
    vec3 c;
    bool screenClip = false;
    // What c holds on an HDR path: 0 nits in the source primaries (the picture), 1 nits
    // in Rec.709 (an OCIO view's light), 2 an SDR code (graphics; an OCIO view's codes
    // when its display cannot be decoded). core/view.cpp's `kind`.
    int kind = mode == 0 ? 0 : 2;
    if (mode == 4) {
        c = rangeColour(log2(max(max(h.r, h.g), h.b) * kPeak / 203.0));
        if ((flags & 7) != 0 && ((p.x + p.y) & 7) == 0) c = c * (1.0 - kRangeHatchMix) + vec3(kRangeHatchMix);
    } else if (mode == 1) {
        c = falseColour(lum2020(h) * kPeak);
    } else if (mode == 2) {
        float d = lum2020(abs(src - b));
        float v = clamp(log2(1.0 + d * kPeak) / extra.x, 0.0, 1.0);
        c = v * vec3(0.95, 0.62, 0.28);
    } else if (mode == 3) {
        float g = kMapGrey * linearToSrgb(lum2020(h) * view.y);
        float a = kMapMix * abs(changeWeight(m, b));
        vec3 col = (b4.a >= kSdrClipCode || b4.a <= kSdrCrushCode) ? kInventedColour : kReinterpretedColour;
        c = vec3(g) * (1.0 - a) + col * a;
    } else {
        // Revision 6: the viewer exposure, then the gamma on the view peak, then the clip.
        vec3 hv = h * look.x;
        bool sdrPath = target.x < 0.5;
        bool done = false;
#ifdef RUDRA_OCIO
        if (ocio.x > 0.5) {
            // Revision 7: the working space, the view's codes, Nuke's viewer gamma on the
            // codes, then the codes (SDR swapchain) or their light (HDR swapchain).
            vec3 e = ocioView(vec4(rowsOf(work[0], work[1], work[2], hv), 1.0)).rgb;
            if (look.z > 0.5) e = sign(e) * pow(abs(e), vec3(look.y));
            screenClip = e.r >= ocio.z || e.g >= ocio.z || e.b >= ocio.z;
            if (!sdrPath && ocio.w > 0.5) {
                // Codes above 1 only where the display has them (EDR: the clip code is out of reach).
                vec3 xyz = ocioDecode(vec4(clamp(e, 0.0, max(1.0, ocio.z)), 1.0)).rgb;
                c = rowsOf(dec[0], dec[1], dec[2], xyz);
                screenClip = screenClip || max(max(c.r, c.g), c.b) >= target.y;
                kind = 1;
            } else {
                c = clamp(e, 0.0, 1.0);
                kind = 2;
            }
            done = true;
        }
#endif
        if (!done) {
            vec3 v = sdrPath ? hv * view.y : hv * kPeak;
            if (look.z > 0.5) {
                float pivot = sdrPath ? 1.0 : target.y;
                v = sign(v) * pivot * pow(abs(v) / pivot, vec3(look.y));
            }
            if (sdrPath) {
                screenClip = v.r >= 1.0 || v.g >= 1.0 || v.b >= 1.0;
                c = vec3(linearToSrgb(v.r), linearToSrgb(v.g), linearToSrgb(v.b));
            } else {
                screenClip = v.r >= target.y || v.g >= target.y || v.b >= target.y;
                c = clamp(v, 0.0, target.y);
            }
        }
    }
    // Overlays mix in as codes on the SDR path and over an OCIO view's codes, else as light.
    bool codesOut = target.x < 0.5 || kind == 2;
    if (extra.w > 0.5 && mode == 0) {
        // Model against baseline at this pixel, whichever side is shown.
        float w = changeWeight(m, b);
        if (w != 0.0) {
            vec3 col = w > 0.0 ? kChangeUp : kChangeDown;
            float a = kChangeTintMix * abs(w);
            vec3 t = overlay(col, codesOut);
            c = c * (1.0 - a) + t * a;
        }
    }
    if (paint.x >= 0.0 && mode == 0) {
        // The mask being painted (3.3), tinted over whichever side is shown.
        vec4 mv = texelFetch(masks, p, 0);   // at the frame's size (passes.hpp masks_rgba8)
        int ch = int(paint.x + 0.5);
        float a = 0.35 * (ch == 0 ? mv.x : ch == 1 ? mv.y : ch == 2 ? mv.z : mv.w);
        if (a > 0.0) {
            vec3 t = overlay(paint.yzw, codesOut);
            c = c * (1.0 - a) + t * a;
        }
    }
    if (region.x > 0.5 && mode == 0) {
        // RUDRA 1.1: the reconstruction region, as the composite gates by it (core/view.cpp).
        // 8 Nov 2026: g says the shadow side; shadows blue (core ViewParams::region_shadow_tint),
        // highlights in the uniform's colour (region_tint, warm).
        vec2 rw = texelFetch(regionW, p, 0).rg;
        float a = 0.40 * rw.r;
        if (a > 0.0) {
            vec3 tint = rw.g > 0.5 ? vec3(0.30, 0.52, 1.00) : region.yzw;
            vec3 t = overlay(tint, codesOut);
            c = c * (1.0 - a) + t * a;
        }
    }
    int zebras = int(ab.w + 0.5);
    if (mode == 0 && zebras != 0) {
        // Revision 6: the source's clamp on one diagonal, the screen's clip on the other.
        int d1 = (p.x + p.y) & 7;
        int d2 = (p.x - p.y) & 7;
        vec3 col = vec3(0.0);
        bool on = false;
        if ((zebras & 1) != 0) {
            int clip = flags & 7;
            if (clip == 7) { on = d1 < 3; col = kClampAllColour; }
            else if (clip != 0) { on = d1 < 1; col = kClampSomeColour; }
            else if ((flags & 8) != 0) { on = d1 < 3; col = kCrushColour; }
            if (on) c = c * (1.0 - kZebraMix) + overlay(col, codesOut) * kZebraMix;
        }
        if ((zebras & 2) != 0 && screenClip && d2 < 2) c = c * (1.0 - kZebraMix) + overlay(kScreenClipColour, codesOut) * kZebraMix;
        // Revision 7: out of the checked gamut (bands every 8 rows), invalid values (solid).
        if ((zebras & 4) != 0 && (p.y & 7) < 3 && belowFloor(rowsOf(chk[0], chk[1], chk[2], h)))
            c = c * (1.0 - kZebraMix) + overlay(kGamutColour, codesOut) * kZebraMix;
        if ((zebras & 8) != 0) {
            bool bad = any(isnan(h)) || any(isinf(h)) || belowFloor(rowsOf(ap0[0], ap0[1], ap0[2], h));
            vec3 wk = abs(rowsOf(work[0], work[1], work[2], h));
            bad = bad || max(max(wk.r, wk.g), wk.b) > kHalfMax;
            if (bad) c = overlay(kInvalidColour, codesOut);
        }
    }
    bool handle = wiping && abs(u - view.z) < view.w;
    if (target.x < 0.5) {
        if (handle) c = vec3(1.0) - c;
        frag = vec4(c, 1.0);
        return;
    }
    // HDR: the picture in nits (source primaries, or Rec.709 through an OCIO view), or an
    // overlay at the SDR white.
    vec3 n;
    vec4 r0, r1, r2;
    if (kind == 0) {
        n = handle ? vec3(target.y) - c : c;
        r0 = pic[0]; r1 = pic[1]; r2 = pic[2];
    } else if (kind == 1) {
        n = handle ? vec3(target.y) - c : c;
        r0 = gfx[0]; r1 = gfx[1]; r2 = gfx[2];
    } else {
        vec3 g = handle ? vec3(1.0) - c : c;
        n = 203.0 * vec3(srgbToLinear(g.r), srgbToLinear(g.g), srgbToLinear(g.b));
        r0 = gfx[0]; r1 = gfx[1]; r2 = gfx[2];
    }
    // Row by row in the order core/view.cpp sums them.
    vec3 o = rowsOf(r0, r1, r2, n);
    frag = vec4(encodeNits(o.r), encodeNits(o.g), encodeNits(o.b), 1.0);
}
