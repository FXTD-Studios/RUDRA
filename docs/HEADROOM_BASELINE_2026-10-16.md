# Opt-in BT.1886 headroom baseline

16 October 2026. Second implementation gate from the
[super-white review](SUPER_WHITES_REVIEW_2026-10-16.md), following the
[isolated float decoder](FLOAT_DECODER_2026-10-16.md).

## Implemented

`native/video/{include/rudra/video,src}/headroom_baseline.*` adds a separate
CPU baseline API. The stable policy identifier is `bt1886-headroom-v1`;
the default policy is Disabled. Disabled or unknown policies refuse, without
falling back or modifying a saved request. No existing caller selects this API.

The function takes ExtendedVideoFrame and checks the relevant MasterRequest
settings. It requires analytic BT.1886, clip fill/peak overrides off, no
calibration/reference fitting, a supported limited-range Rec.709-transfer
source contract and matching declared primaries. It leaves source samples and
settings intact. Unsupported combinations return errors.

For each source RGB channel, it computes `max(E, 0)^2.4` in double precision,
then uses the existing primary-conversion matrix to transform linear light to
Rec.2020. The returned image uses the existing linear-master unit tag:
**1 = 203 nits**. No clamp occurs at nominal white. Original negative source
samples remain available; this policy maps their contribution to zero light.
Nonfinite input, malformed buffers and output beyond finite float range refuse.

The result carries its policy and source contract for later provenance work.
It is explicitly **before grading**: exposure, anchor, source-chroma carry,
grain, shoulder and temporal processing are not applied by this function.
Neither NetworkLinearImage nor SdrImage is used to disguise extended inputs
or the output's different units. This API is not a finished master/export.

## Verification

Release build and **283/283 native tests pass** (38.473 seconds). The full
suite includes legacy regression checks. App/GPU tests were not rerun: no UI
or shader integration is present in this increment.

19 focused tests pass: the 10 decoder tests plus 9 new baseline tests.
Coverage includes explicit opt-in/version identity, absolute brightness
references, exact black/nominal white, monotonicity through all tested 8/10-bit
headroom levels, continuity immediately around white, source-primary colors,
order of EOTF and gamut conversion, preserved source data, unsupported settings,
metadata disagreement, invalid buffers, NaN/Inf and numerical overflow.

Absolute brightness targets use independently evaluated 60-digit Decimal
references; primary-color targets use the existing frozen Python gamut golden,
not a second call to the production conversion routine. Tested errors stay
within 1e-4 nits for the fixed brightness samples and 2e-7 relative-white units
for colored matrix targets. These are numerical gates, not visual judgments.

The four real FFV1 decoder fixtures now pass through the baseline too:
8/10-bit YUV444, BT.709 and BT.2020 matrices/primaries. Every tested headroom
level remains distinct after reconstruction; output agrees with the neutral
reference equation within 3e-4 nits. Recorded final samples, both primaries:

| Original neutral source | Baseline output before grade |
| --- | ---: |
| 8-bit code 254 | 247.864758 nits |
| 10-bit code 1019 | 249.743557 nits |

The same clips through the unchanged integer decoder still lose these levels.
The existing BT.1886 inverse still gives approximately 201.795706 nits at
nominal white. v1 gives 203 nits for exact E=1. This intentional policy change
is tested and versioned; legacy parity is not claimed when opting into v1.

Evidence:
- `reports/headroom_baseline_focused_20261016.{json,log}`
- `reports/headroom_baseline_build_fix_20261016.log`
- `reports/headroom_baseline_native_20261016.{json,log}`

Initial build caught a test-only ReferenceFit member-name typo, corrected to
`target_log2_nits`. The failed build log is retained; final build is clean of
compiler diagnostics. No existing pixel formula or golden was changed.

## Remaining / next action

Shared paused/playback/master/movie-export integration comes next, including
versioned cache keys, project policy persistence and truthful output provenance.
Do not expose the option in the app until the same policy is used across these
paths and the user's grading controls have a validated implementation.
The new function's baseline output must not bypass those controls or be fed
back through the old SDR clamps, clip-fill residuals or network-unit scaling.

Python compatibility, GPU parity, scopes/comparison/invented-map behavior,
temporal holds, real-footage quality, throughput and calibrated HDR assessment
remain outstanding. No production enablement, release clearance, package
refresh, training, data change, download, commit or app restart occurred.

Reference: [ITU-R BT.1886, Annex 1](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1886-0-201103-I!!PDF-E.pdf)
defines the display EOTF. RUDRA's 203-nit normalization and use above nominal
white are explicit reconstruction policy; they do not establish original
scene luminance or make an unknown tone-mapped source invertible.
