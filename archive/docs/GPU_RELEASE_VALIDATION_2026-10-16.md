# GPU and release-dependency validation — 16 October 2026

Dependency validation and four-backend parity/latency gates complete. Primary-display HDR swapchain checks pass; physical/calibrated review remains pending. Latest source rebuilt into the existing Gate B configuration; no training or source datasets modified.

## Latest rerun — workload removed, all latency gates pass

Evidence: `reports/gpu_idle_20261016_153330/` (per-backend JSON/logs,
timing summary, GPU load before/after). All four tools return exit 0 and
`pass=true`; original budgets unchanged.

| API | 1080p composite + view wall ms | Budget ms | 4K wall ms | Budget ms |
| --- | ---: | ---: | ---: | ---: |
| D3D12 | 0.34 | 1.1 | 1.41 | 4.0 |
| D3D11 | 0.28 | 0.9 | 1.24 | 3.6 |
| Vulkan | 0.30 | 1.0 | 1.48 | 3.7 |
| OpenGL | 0.28 | 0.9 | 1.16 | 3.3 |

These are measured median wall times for composite + display, not neural
inference or complete video delivery. Numerical parity passes on all APIs.
Memory use fell to about 1.6 GB; utilization was 33% before and 32% after,
so this was not a literally idle desktop. Nevertheless all predefined gates
pass with that background activity. The earlier loaded runs remain preserved.

Next: calibrated HDR/black-level review on the primary display, then remaining
release workflow, installer/signing, clean-machine and licensing checks.
Secondary-display HDR remains off; OpenGL HDR is not supported by this run.

## First run: image correctness passes, release gate not passed

All four backends (D3D12, D3D11, Vulkan, OpenGL) pass 28 composite cases and 30 view cases each. The real viewer SDR parity checks also pass on all four.

| API | 1080p composite + view wall ms | Budget ms | 4K wall ms | Budget ms |
| --- | ---: | ---: | ---: | ---: |
| D3D12 | 4.952 | 1.1 | 1.863 | 4.0 |
| D3D11 | 15.875 | 0.9 | 15.540 | 3.6 |
| Vulkan | 2.194 | 1.0 | 11.033 | 3.7 |
| OpenGL | 3.653 | 0.9 | 11.794 | 3.3 |

Timings are the first run under concurrent GPU load, not clean performance results. NVIDIA reports 100% utilization and approximately 15.6 GB occupied after the probes finish; another Python compute process and graphics applications are present. No other processes were stopped. Exact timing values are retained in `reports/gpu_validation_20261016/timing_summary.json`.

Both displays (PA279CRV and LG TV SSCR2) report Windows HDR off. D3D HDR probes fall back to SDR and fail. The Vulkan viewer card reports a valid scRGB readback, which does not establish physical HDR output when Windows reports HDR off. Calibrated visual confirmation remains necessary.

The user offered to make the GPU idle and enable HDR; a clean rerun remains pending.

## Release FFmpeg

The user approved downloading the repository-pinned FFmpeg 7.1.1 archive (177,650,503 bytes). Download, checksum verification and extraction completed. Verified archive SHA-256:
`d760e1b3574402ed18b4865851f87d87e73965a982e6453212df8621fed1c508`.

Full Release suites were then run with the extracted 7.1.1 `bin` directory first on the test process PATH:

- **264/264 native tests passed**, 25.023 s.
- **57/57 application tests passed**, 32.989 s.
- Version records and JSON/log evidence: `reports/ffmpeg_release_711_20261016/`.

No system PATH change or deployment occurred. The archive and extracted tools remain in `tmp/native_deps/`. These passes validate the dependency with the local full build, not a signed installer or a clean-machine installation.

## Continuation

### Rerun after user enabled HDR

Saved separately in `reports/gpu_ready_20261016_153101/`.

- Primary PA279CRV now reports Windows HDR on. D3D12 scRGB, D3D12 HDR10
  and D3D11 scRGB swapchain probes all pass.
- Real-viewer HDR cards pass on D3D12, D3D11 and Vulkan. OpenGL returns
  unsupported/no HDR (exit 2), rather than an HDR pass.
- Secondary LG TV SSCR2 still reports Windows HDR off and falls back to SDR.
- All 28 composite and 30 view cases still pass on each of the four APIs.
- Performance gates still fail: GPU utilization was 100% before and after
  the run, with roughly 15–15.8 GB occupied. Another Python compute process
  (PID 45712) remained present. No process was stopped. These are loaded
  measurements, not an isolated performance regression diagnosis.

The workload-removal and latency rerun above supersede this pending step.
Physical/calibrated display review is still separate from successful
swapchain readback.

Keep both runs as evidence and use a separate directory for the idle-GPU rerun. Do not change thresholds or stop unrelated processes. Physical HDR and calibrated shadow review remain open. No commit, push, release or training was performed.
