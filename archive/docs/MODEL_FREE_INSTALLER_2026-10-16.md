# Model-free installer staging — 16 October 2026

Compiled successfully with Inno Setup in 88.890 seconds from the previously
tested model-free folder. No host installation or uninstallation performed.

Artifact:
`dist/analytic/installer-review-9f880d42954b4e0bae524a29abc5722c/RUDRA-1.0.0-beta.3-windows-x64-setup.exe`

Size: 143,886,082 bytes. SHA-256:
`2aa492ca7cd777c25835b50ab9318764e3d762d571c33143002ca7efd3150beb`.
A checksum file sits beside the executable. Authenticode reports NotSigned.

## Changes

The `ModelFreeEvaluation` compiler define creates a distinct installer AppId,
Analytic Evaluation display name, install folder and shortcuts. It omits file
association registration and does not reuse a previous installer directory.
`PACKAGE_WINDOWS.ps1 -NoModels` passes that define automatically. Existing
research beta installer behavior remains in the default branch.

The application still shares RUDRA's preference namespace. This is installer
separation, not complete application-profile isolation. Use a clean Windows
user/VM for evaluation. The payload's startup and analytic HDR10/ProRes smoke
checks already passed; that does not establish install/uninstall behavior.

## Evidence and remaining gates

Compile log: `reports/model_free_installer_20261016.log`.
Artifact record: `reports/model_free_installer_20261016.json`.
Source diff whitespace check passed. Installer source reviewed for distinct
identity/paths and omission of associations in evaluation mode. No dynamic
registry or uninstall test is claimed.

Windows Sandbox was not found on this host. Follow
[CLEAN_MACHINE_ACCEPTANCE.md](CLEAN_MACHINE_ACCEPTANCE.md) on a separate
disposable Windows VM/PC. Every row remains pending. Signing, calibrated HDR
assessment and commercial licensing also remain pending; this unsigned local
evaluation installer is not release-cleared. No training, source-dataset
changes, commit, push or publication occurred.
