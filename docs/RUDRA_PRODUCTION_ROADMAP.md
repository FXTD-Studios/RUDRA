# RUDRA Commercial SDR-to-HDR: Claude Execution Plan
Date: 2026-10-13
Repo: https://github.com/FXTD-Studios/RUDRA

## Strategic recommendation
Keep RUDRA's fast source-aware analytic inverse and artist controls as the production foundation. Prove v10 before redesigning the model. Build a clean commercial dataset and checkpoint lineage. Introduce optional, confidence-gated sky, fire, specular and practical-light enhancement, first with deterministic techniques, then small neural models if justified by measurements. Separate **Faithful HDR** from opt-in **Creative HDR**. Fully clipped cloud/flame details cannot be recovered as ground truth from a single SDR frame.

## Verified repository state
- `STATUS.md`: v7 underperformed the analytic inverse on all 287 Meridian held-out real frames, -4.13 dB and -0.328 CVVDP JOD. BT.1886 source-aware mapping subsequently improved markedly over ACES inverse on real SDR. RUDRA 1.0 defaults to analytic inverse; neural inference is optional.
- `docs/MODEL_PLAN_V10.md`: v10 is prepared, not confirmed passing. It changes the model baseline to the source picker, protects crushed blacks, uses live-action validation and full-frame checkpoint selection. Run `scripts/run_v10_2026_10_13.ps1` with smoke, train, bench -ClipOnly, bench in that order, subject to approval and hardware availability.
- `docs/RELEASE_V1.md`: open commercial-tool comparison, macOS, signing and clean-PC release checks. Existing `training/compare_market.py` supports benchmark orchestration.
- `docs/RESULTS.md`: quadrupling network capacity did not resolve earlier performance deficits. Current ~1.2M-parameter weights correspond to ~4.8MB FP32, before overhead.
- Published Hugging Face weights are non-commercial, citing academic-only HdM data; the code has a separate license. Do not fine-tune these weights and call them commercially clean.

## Competitive references
- Topaz Video Hyperion: production SDR-to-HDR filter; measure visually and quantitatively with licensed access: https://www.topazlabs.com/learn/how-to-convert-sdr-to-hdr-topaz-video
- HDRTVNet++: efficient global mapping, local enhancement and highlight generation: https://github.com/xiaom233/hdrtvnet-plus . Its dataset is explicitly academic-only, not commercial training material without permission.
- HDRTVDM: HDR/WCG diversity and realistic degradation modeling: https://github.com/AndreGuo/HDRTVDM . Verify rights for code, weights and datasets separately.
- RUDRA must beat the *correct source-aware* BT.1886/BT.2408 baseline, not only the old ACES inverse.

## Architecture
SDR input and transfer-function picker -> calibrated analytic inverse -> confidence and clipped-region maps -> optional compact scene-aware refinement -> sky/fire/emissive/specular region processing -> motion-aware temporal hold -> artist grade -> scene-linear EXR/ACEScg and HDR10/HLG delivery.

**Faithful HDR**: no generated geometry, no invented texture, preserve skin, neutral surfaces, black levels and original scene. **Creative HDR**: explicit opt-in, clearly label any synthesized clouds/flames or light structure, preserve temporal consistency and provide region strengths.

Candidate weight-size experiments: 4 MB compact gate, 16 MB spatial multi-head refiner, 32 MB only if 16 MB gives statistically meaningful gains. These are targets, not verified requirements. No default diffusion or LoRA. Benchmark real runtime, VRAM, quality and temporal behavior.

## Commercial dataset strategy
1. Read-only inventory of local `G:\` when available. Scan `G:\datasets`, corpora, manifests and checkpoint metadata. No remote access assumption and no deletion.
2. Hash files; record origin, license evidence, rights to commercial ML training and distribution, scene ID, frame format, PQ/HLG/linear interpretation, mastering peak and splits.
3. Quarantine academic-only HdM and all unknown-license records. Verify the claimed 40% proportion from actual manifests. Do not reuse weights descended from restricted training for the commercial checkpoint absent appropriate rights.
4. Acquire rights-cleared FXTD HDR/RAW/EXR and commercially licensed real SDR/HDR pairs. CC0 HDR panoramas supplement but cannot substitute for diverse real video scenes. Check permissions individually.
5. Rebuild scene-disjoint train/validation/test splits; audit nit scaling, EOTF, highlight clipping, leakage and near-duplicates. Initialize commercial model weights from scratch unless every ancestor is cleared.

## Phase plan and exit criteria
**P0, read-only audit:** identify active code, git commit, model sizes, licenses, G: access, v10 readiness, release blockers. Output `docs/ai/RUDRA_EXECUTION_AUDIT.md`. No code changes.

**P1, existing v10:** run smoke and synthetic tests, then authorized train/bench. Use predeclared gates: improve real Meridian PU21 and CVVDP with 95% scene-level confidence intervals above zero; avoid regressions on aces/oog/mix; beat clip fill at +0/+1/+2 EV; protect crushed blacks. Stop if gates fail.

**P2, commercial lineage:** source-by-source license matrix, G: manifest, provenance report, clean training plan. No commercial checkpoint until approved.

**P3, competitor evaluation:** same SDR input, reference HDR, color management and metrics for RUDRA, BT.1886, BT.2408, permitted HDRTVNet++/HDRTVDM and Topaz. Test skies, fire, fireworks, neon, specular, interiors, skin, motion and compression. Report PU21-PSNR, CVVDP, Delta E_ITP, clipped stop error, flicker, latency and VRAM. HDR monitor blind review.

**P4, scene-aware prototype:** deterministic masks and local processing first; compact learned region gating second. Use uncertainty masks and non-destructive controls. Keep original pixels outside target ROI. Separate faithful and creative modes.

**P5, clean commercial training:** train from scratch on rights-cleared data; identical splits/seeds across 4/16/32 MB ablations. Promotion only with significant measured gains and no clean-scene or temporal regression.

**P6, native integration and release:** picker-aware tile graph, GPU parity, EXR/ACEScg/PQ/HLG correctness, Windows/macOS tests, crash handling, rollback and analytic fallback. Publish model card, licensing manifest and measured limitations.

## Paste into Claude Code

You are FXTD Studios' lead HDR color scientist, ML engineer and production VFX software architect. Work in the existing RUDRA Git repository. Prioritize evidence, commercial safety and low token usage.

Goal: make RUDRA a best-in-class, commercially licensable SDR-to-HDR system with high-quality real-video reconstruction, compact models, sky/fire/emissive enhancement, temporal stability and artist control. Do not claim recovery of irreversibly clipped source detail.

1. Read `CLAUDE.md` if present, `STATUS.md`, `docs/MODEL_PLAN_V10.md`, `docs/RELEASE_V1.md`, `docs/RESULTS.md`, training scripts, and relevant tests. Confirm commit, branch and clean worktree. Do not repeat completed work or confuse paused HDR VAE decoders with the active SDR-to-HDR product.
2. **First phase is READ-ONLY AUDIT.** No code edits, installs, downloads, long GPU runs, deletions, license changes or production promotions without explicit GO. Check whether G: is mounted locally; if not, report unavailable, do not fabricate inventory.
3. Produce `docs/ai/RUDRA_EXECUTION_AUDIT.md` with evidence-backed current architecture, model parameter and byte counts, v10 readiness, active test gates, release blockers, G: inventory feasibility, exact dataset/checkpoint lineage and license risk. Cite paths and line numbers. Include top five actions and estimated resources.
4. Preserve the analytic inverse production path and release work. Audit v10's existing smoke/train/clip-only/full-bench scripts before proposing new architecture. Follow predeclared stop rules; do not relax gates after seeing results.
5. Treat commercial lineage as a hard gate. Quarantine HdM/academic-only and unknown-license data and descendant weights; fine-tuning restricted weights is not a clean restart. Verify all data, code and dependency licenses separately.
6. Compare to source-aware BT.1886/BT.2408 and permitted external tools on identical color-managed test sets. Use full-frame PU21, CVVDP, clipped stop error, shadow preservation, flicker, Delta E_ITP, runtime and VRAM. Record confidence intervals and scene IDs.
7. Propose Faithful HDR and opt-in Creative HDR. Start with deterministic sky/fire/emissive confidence and regional corrections. Consider compact 4/16/32 MB models only if baseline error analysis warrants it; no default diffusion or LoRA.
8. For every later change, require a reproducer, focused test, before/after benchmark, rollback path and atomic commit. Do not run expensive training without an approved dataset, budget and abort criterion.
9. Keep conversation concise: Findings / Evidence / Decision / Next action. Store long results in `docs/ai/` rather than repeating them in chat.

**START NOW: Perform only P0 read-only audit. Create the audit report and give me the five highest-priority next steps. Wait for GO before implementation or training.**
