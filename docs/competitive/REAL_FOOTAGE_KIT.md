# Real-footage training for the 0 EV fix (option 1: keep the gate, fix the model)

Decided 4 Nov 2026 (Ahmed: "need to fix it"): the 0 EV gate stays as it is; the
model is fixed. Evidence so far (RESULT_BIN_HINGE_2026_11_02.md): w4 passes every
original development gate, but every variant stays at 0.45 code RMS on the fresh
FXTD set (limit 0.338). That set is real camera footage with many PARTIALLY clipped
highlights, which the training corpora barely contain.

**The FXTD camera corpus is used up**: the fresh set took every scene with an
unclipped reference (85 scenes), and the rest have sensor-clipped references.
Training on fresh-set scenes would make the fresh numbers meaningless. So this
needs **new footage, split by project**: 2-3 projects for training, 4-5 others for
the holdout (HOLDOUT_KIT.md). Never the same project in both.

1. Ingest the training projects as run05 plates (decoded, with `_ingest_config.json`),
   one JSONL row per frame (asset_id, scene_id, project, hdr_path, plate fields).
2. Count first, on G: (nothing is written):

       python tools\build_real_partial_corpus.py --sources <rows.jsonl> --plates <plates> ^
           --exclude G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102\report.json <holdout registrations> ^
           --out G:\datasets\corpora\real_partial_v1 --dry-run

   Aim for 300+ pairs from 20+ scenes; fewer will not move a model.
3. Build it (same command without --dry-run). The report records every refused
   evaluation scene and the hashes of what it excluded against.
4. Register before training (written from the report): arm **w4 + real**
   (continue w4, 1600 steps, its recipe, `--extra-pilot G:\datasets\corpora\real_partial_v1\report.json`)
   and control **w4 continued** (identical, no extra pilot); the original gates plus
   the fresh tolerant rule; selection on the fresh rule with every gate held.
5. Train and evaluate with RUN_BIN_HINGE.ps1's sequence. Confirm on the holdout once.
