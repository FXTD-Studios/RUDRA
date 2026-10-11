#version 440
// The static hold while playing (11 Oct 2026): ShadowSmoother::hold_pixels
// (native/core/src/video_predict.cpp, rudra/video.py hold_pixels) on the GPU,
// after the composite and before the display pass, so a still part of a shot
// plays as the export holds it. It is held to the C++ reference by
// rudra-gpu-parity (GpuCompositor::hold_sequence).
//
// Per pixel, the HDR's gain over the source, G = (Yh + 1e-6) / (Ys + 1e-4),
// is held across frames where the source is still (3x3 mean |dLuma| under
// 1.5/255; its own over 6/255; smoothstep between). Where the source is
// clipped in both frames the HDR's own change (3x3 mean |dlog2 Yh|, held under
// 0.05, its own over 0.25) decides; a pixel clipped in one frame only, a
// crushed one and one whose HDR is negative are never held.
//
// Two outputs: the held picture (rgb times held/G, alpha as the composite
// wrote it) and the state the next frame reads (x luma code, y largest code,
// z held gain, w Yh). The state ping-pongs between two textures.

layout(location = 0) out vec4 held;
layout(location = 1) out vec4 state;

layout(std140, binding = 0) uniform Hold {
    vec4 hold;   // x retention^frames since the previous one shown, y 1: the shot's first frame
};

layout(binding = 1) uniform sampler2D model;       // the composite: rgb network units, a anchor target
layout(binding = 2) uniform sampler2D sdrShadow;   // rgb SDR code values (a, the shadow mask, unused)
layout(binding = 3) uniform sampler2D previous;    // the previous frame's state

const float kStill = 1.5 / 255.0, kMoved = 6.0 / 255.0;
const float kSdrFloor = 1e-4, kHdrFloor = 1e-6;
const float kClipped = 0.98, kCrushed = 0.02;
const float kHdrStill = 0.05, kHdrMoved = 0.25;
const vec3 kLuma709 = vec3(0.2126, 0.7152, 0.0722);
const vec3 kLuma2020 = vec3(0.2627, 0.6780, 0.0593);

float srgbToLinear(float x) {
    x = clamp(x, 0.0, 1.0);
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

ivec2 clampTo(ivec2 q, ivec2 size) { return clamp(q, ivec2(0), size - 1); }

float codeLuma(ivec2 q) { return dot(texelFetch(sdrShadow, q, 0).rgb, kLuma709); }
float hdrLuma(ivec2 q) { return dot(texelFetch(model, q, 0).rgb, kLuma2020); }
float logHdr(float yh) { return log2(max(yh, 0.0) + kHdrFloor); }
float release(float t) { return 1.0 - t * t * (3.0 - 2.0 * t); }

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(model, 0);
    vec4 m = texelFetch(model, p, 0);
    vec3 s = texelFetch(sdrShadow, p, 0).rgb;
    float luma = dot(s, kLuma709);
    float peak = max(s.r, max(s.g, s.b));
    float ys = dot(vec3(srgbToLinear(s.r), srgbToLinear(s.g), srgbToLinear(s.b)), kLuma709);
    float yh = dot(m.rgb, kLuma2020);
    float gain = (yh + kHdrFloor) / (ys + kSdrFloor);

    if (hold.y > 0.5) {   // the shot's first frame: its own, and the state starts
        held = m;
        state = vec4(luma, peak, gain, yh);
        return;
    }
    vec4 prev = texelFetch(previous, p, 0);
    bool bothClipped = min(peak, prev.y) >= kClipped;
    float moved = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 q = clampTo(p + ivec2(dx, dy), size);
            if (bothClipped)
                moved += abs(logHdr(hdrLuma(q)) - logHdr(texelFetch(previous, q, 0).w));
            else
                moved += abs(codeLuma(q) - texelFetch(previous, q, 0).x);
        }
    moved /= 9.0;
    float t = bothClipped ? clamp((moved - kHdrStill) / (kHdrMoved - kHdrStill), 0.0, 1.0)
                          : clamp((moved - kStill) / (kMoved - kStill), 0.0, 1.0);
    float h = hold.x * release(t);
    if ((max(peak, prev.y) >= kClipped && !bothClipped) || min(peak, prev.y) <= kCrushed || yh < 0.0) h = 0.0;
    float heldGain = h * prev.z + (1.0 - h) * gain;
    held = vec4(m.rgb * (heldGain / gain), m.a);
    state = vec4(luma, peak, heldGain, yh);
}
