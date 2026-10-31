"""RUDRA generative fill (layer 3, 27 Oct 2026).

A Wan2.1-VACE video model (Apache-2.0) with a RUDRA-trained LoRA fills only the regions the
8-bit source cannot know: fully clipped highlights and fully crushed shadows. Everything else
stays the master the exact (analytic) and learned layers wrote.

    encoding   scene-linear ACEScg <-> the VAE's [-1, 1] pixels (normalised ACEScct)
    masks      the fill regions from the SDR codes (tau_high / tau_low), dilated, steadied in time
    composite  the round-trip-safe blend: no 8-bit code of the master's SDR view changes
    model      the Wan VACE components, the conditioning, the denoising loop, LoRA
    run        a master folder (EXR) -> a filled folder, JSON progress lines for the app

Nothing here uses DiffHDR code or weights. The training data are RUDRA's commercial-clean
sources only (tools/build_genfill_clips.py); the LoRA is trained by training/genfill_train.py.
"""
