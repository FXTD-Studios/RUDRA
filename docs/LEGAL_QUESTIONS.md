# Questions for counsel before RUDRA 1.0 is sold (roadmap 0.3)

16 October 2026. Prepared by FXTD Studios. This is not legal advice: these
are the questions to send to counsel. Evidence sits in the repository's local
audits (docs/ai/).

1. **Netflix Open Content basis.** The current source folder offers CC BY 4.0
   (the AOM licence file beside the archives, and opencontent.netflix.com).
   An older README in the same folder lists NC-ND for Sparks, Meridian,
   Sol Levante and Nocturne.
   - May we rely on CC BY 4.0 for training, and for shipping values measured
     from these titles? The app's "BT.1886 shadows" curve holds per-code
     medians measured on Nocturne, Sol Levante and Cosmos Laundromat.
   - Should we ask Netflix for written confirmation?
2. **Cosmos Laundromat.** The original film is CC BY 3.0 (Blender
   Foundation) and the HDR re-grade is Netflix-provided. Is the attribution
   in NOTICE sufficient for both layers?
3. **Measured values against trained weights.** Does a 39-number curve
   measured from footage carry the licence conditions of that footage
   differently from trained model weights?
4. **HdM-HDR-2014 / HFR-2017.** Released weights v8 and earlier are
   non-commercial because of HdM's academic-only term. Confirm that the
   analytic app, which ships no weights by default, is unaffected, and that
   weights trained only on cleared sources would be.
5. **PanDORA** allows use for "any purpose, including academic research,
   education, and commercial research". Does that cover a commercial
   product?
6. **Linköping HDRv** is CC BY-SA 4.0. Does ShareAlike reach weights trained
   on it?
7. **LIVE-TMHDR** requires its copyright notice "in all copies of this
   database". What is owed if it is only used for training?
8. **HEVC.** The Windows package bundles the unmodified gyan.dev ffmpeg
   (GPLv3, run as a separate program) with x265. What is our patent-pool
   exposure when selling software that encodes HEVC?
9. **Dolby Vision and HDR10+ names.** RUDRA writes metadata files labelled
   for Dolby Vision (for `dovi_tool generate`) and HDR10+. May a commercial
   product name them so without certification or a licence?
10. **Code licence.** The code is PolyForm Noncommercial 1.0.0; copies taken
    before 24 Sep 2026 are Apache 2.0. Is the commercial licence we sell
    alongside it worded correctly?
