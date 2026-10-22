# Model-free evaluation installer: clean-machine acceptance

Use a disposable Windows x64 VM or a separate test PC with no RUDRA, Python,
Qt, OpenCV or developer tools installed. Do not substitute a restricted PATH
run on the development machine for this check. Windows Sandbox was not found
on the development host on 16 October 2026.

The evaluation installer has its own AppId, install directory, shortcuts and
uninstall name; it does not register `.rudra` file associations. The executable
still uses the normal RUDRA preference namespace, so run in a fresh Windows
user profile. It is unsigned and has no commercial clearance.

Record OS version, GPU/driver, display/HDR state, installer SHA-256 and every
result below. Preserve installer and application logs before uninstalling.

| Check | Required evidence | Result |
| --- | --- | --- |
| Identity | Installer says RUDRA Analytic Evaluation; hash matches supplied SHA-256 | Pending |
| Installation | Installs for current user into its separate directory | Pending |
| Contents | No `.pt`, `.pth`, `.onnx`, `.safetensors` or model catalogue; bundled dependencies present | Pending |
| Startup | Launch from Start menu with no dependency or Python errors | Pending |
| Source workflow | Open still and clip, scrub, change grade, undo, save/reopen project | Pending |
| Analytic delivery | Export HDR10 and ProRes without a model; sidecars say analytic (no model) | Pending |
| Output | Play in a separate HDR-capable player; check tags and visible output | Pending |
| Associations | `.rudra` default association is unchanged | Pending |
| Uninstall | Evaluation entry, binaries and shortcuts removed; user's media/projects retained | Pending |
| Isolation | No existing RUDRA install modified; use a separate disposable test for coexistence | Pending |

Installation success alone does not pass this checklist. Hardware HDR/display
review, signing and licensing remain separate release gates. Return the
completed table and logs before calling the installer validated.
