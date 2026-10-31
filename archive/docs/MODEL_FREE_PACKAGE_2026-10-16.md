# Model-free Windows package — 16 October 2026

Implemented `-NoModels` in `scripts/PACKAGE_WINDOWS.ps1`. The existing research
beta mode is preserved. Model-free mode skips model requirements, catalogue,
weights and weight-license copying, and stages into a new GUID-named directory
under `dist/analytic/`, keeping existing packages intact. `-BuildDirectory`
allows reuse of an explicitly selected local build with `-SkipBuild`.

Model-free mode runs `scripts/TEST_MODEL_FREE_PACKAGE.ps1` before ZIP creation.
It rejects nonempty model directories or weight files, clears Python and Qt
environment overrides, limits PATH to the staged package and Windows, starts
the app's theme check and performs analytic HDR10 and ProRes deliveries. It
checks both output reports identify reconstruction as analytic with no model.
Environment variables are restored afterward. It does not install or publish.

## Created and tested

Package:
`dist/analytic/dced1f5d4ad94866818d217cefc2bb65/RUDRA-1.0.0-beta.3-windows-x64.zip`

SHA-256:
`c438a8969f02a76174541eecba8af69eb6ea3e564f6259b1f88a24c9a83d8221`

The existing executable version, 1.0.0-beta.3, was retained. This is an unsigned
local evaluation ZIP, not a new tagged release or commercial clearance.

- Packaging succeeded using `build/native_phase3` and pinned FFmpeg 7.1.1.
- No weights/model catalogue bundled; empty models directory retained.
- Restricted-PATH startup/theme, analytic HDR10 and ProRes deliveries passed.
- A synthetic package containing a model file was correctly rejected.
- PowerShell parse and diff whitespace checks passed.
- The ZIP was built before integrating the smoke-test invocation into the
  packager; the exact staged content passed that same smoke script afterward.
  The resulting tested content was not changed.

Evidence: `reports/model_free_package_20261016.log` and
`reports/model_free_smoke_d9999221e3504fd39d3f7c74adb72612/`.

Reproduction:

```powershell
pwsh -NoProfile -File scripts/PACKAGE_WINDOWS.ps1 -NoModels -SkipBuild -BuildDirectory build/native_phase3 -NoInstaller
```

Visual review remains pending: the owner said they are not ready to assess.
Next: inspect/stage the installer without installing over an existing RUDRA,
then clean-machine validation and signing when available. Restricted-PATH
testing is useful but does not replace a clean-machine test. Commercial rights
for code, derived shadow values and third-party components remain unresolved;
omitting weights alone does not resolve them. No training, dataset changes,
commit, push, installation or publishing occurred.
