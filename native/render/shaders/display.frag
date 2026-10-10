#version 440
// The display pass (docs/view.spec.md section 2). A line-for-line port of
// native/core/src/view.cpp, the reference it is held to (rudra-gpu-parity),
// and of the DISPLAY shader in ui/compositor.js, which it replaces.
//
// Texels are fetched by pixel, so output row r is image row r on every API,
// as in composite.frag. Presenting to a swapchain puts the flip, if the API
// needs one, in the vertex stage (render/, Phase 2 step 9), never here.

layout(location = 0) out vec4 frag;

layout(std140, binding = 0) uniform View {
    vec4 view;    // x mode (0 image, 1 false colour, 2 difference, 3 invented), y exposure (10000 / view peak), z wipe (< 0 off), w wipe half width
    vec4 extra;   // x log2(1 + diff gain), y frame width, z 1 to show the baseline, w 1 to tint the changes
    vec4 target;  // x path (0 SDR, 1 scRGB, 2 HDR10, 3 EDR), y ceiling nits, z nits per 1.0 (linear paths)
    vec4 pic[3];  // rows of source primaries -> swapchain primaries (xyz)
    vec4 gfx[3];  // rows of Rec.709 -> swapchain primaries, for the overlays
    vec4 anchor;  // roadmap 3.3: x 1 to anchor the model's picture, y knee (SDR max code), z softness, w hold gain
    vec4 paint;   // roadmap 3.3: x the painted mask's channel (< 0 none), yzw its tint (sRGB)
};

layout(binding = 3) uniform sampler2D masks;      // painted masks (3.3), RGBA8, a band per channel
layout(binding = 1) uniform sampler2D model;      // rgb reconstruction, network units; a the SDR's Rec.2020 luma, linearised (the anchor's target)
layout(binding = 2) uniform sampler2D baseline;   // rgb analytic baseline, a the SDR's max code (Invented)

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
    vec3 src = (!wiping && extra.z > 0.5) ? b : m;
    bool left = wiping && u < view.z;
    vec3 h = left ? b : src;
    // The anchor (core/view.cpp anchor_gain_f) on the model's picture only.
    if (anchor.x > 0.5 && !left && !(!wiping && extra.z > 0.5)) {
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
    if (mode == 1) {
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
    } else if (target.x < 0.5) {
        c = vec3(linearToSrgb(h.r * view.y), linearToSrgb(h.g * view.y), linearToSrgb(h.b * view.y));
    } else {
        c = clamp(h * kPeak, 0.0, target.y);
    }
    if (extra.w > 0.5 && mode == 0) {
        // Model against baseline at this pixel, whichever side is shown.
        float w = changeWeight(m, b);
        if (w != 0.0) {
            vec3 col = w > 0.0 ? kChangeUp : kChangeDown;
            float a = kChangeTintMix * abs(w);
            vec3 t = target.x < 0.5 ? col : 203.0 * vec3(srgbToLinear(col.r), srgbToLinear(col.g), srgbToLinear(col.b));
            c = c * (1.0 - a) + t * a;
        }
    }
    if (paint.x >= 0.0 && mode == 0) {
        // The mask being painted (3.3), tinted over whichever side is shown.
        vec4 mv = texelFetch(masks, p, 0);   // at the frame's size (passes.hpp masks_rgba8)
        int ch = int(paint.x + 0.5);
        float a = 0.35 * (ch == 0 ? mv.x : ch == 1 ? mv.y : ch == 2 ? mv.z : mv.w);
        if (a > 0.0) {
            vec3 t = target.x < 0.5 ? paint.yzw : 203.0 * vec3(srgbToLinear(paint.y), srgbToLinear(paint.z), srgbToLinear(paint.w));
            c = c * (1.0 - a) + t * a;
        }
    }
    bool handle = wiping && abs(u - view.z) < view.w;
    if (target.x < 0.5) {
        if (handle) c = vec3(1.0) - c;
        frag = vec4(c, 1.0);
        return;
    }
    // HDR: the picture in nits (source primaries), or an overlay at the SDR white.
    vec3 n;
    vec4 r0, r1, r2;
    if (mode == 0) {
        n = handle ? vec3(target.y) - c : c;
        r0 = pic[0]; r1 = pic[1]; r2 = pic[2];
    } else {
        vec3 g = handle ? vec3(1.0) - c : c;
        n = 203.0 * vec3(srgbToLinear(g.r), srgbToLinear(g.g), srgbToLinear(g.b));
        r0 = gfx[0]; r1 = gfx[1]; r2 = gfx[2];
    }
    // Row by row in the order core/view.cpp sums them.
    vec3 o = vec3(r0.x * n.x + r0.y * n.y + r0.z * n.z,
                  r1.x * n.x + r1.y * n.y + r1.z * n.z,
                  r2.x * n.x + r2.y * n.y + r2.z * n.z);
    frag = vec4(encodeNits(o.r), encodeNits(o.g), encodeNits(o.b), 1.0);
}
