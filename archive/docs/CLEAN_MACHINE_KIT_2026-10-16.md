# Portable clean-machine acceptance kit — 16 October 2026

Owner confirmed **no clean environment is available yet**. Installation and
uninstallation acceptance are pending, not passed. No host software or Windows
features were installed/enabled to work around this limitation.

Kit folder:
`dist/analytic/acceptance-kit-9e8583faa1e14d88872cd3779c8b1975/`

Contains the unsigned model-free installer and checksum, a tiny existing
synthetic video fixture, standalone smoke-test script, instructions and manual
acceptance checklist. It can be copied to a future clean PC/VM without the
repository or developer runtimes.

`TEST_MODEL_FREE_PACKAGE.ps1` now accepts explicit input and report paths and
writes BOM-free JSON using an API supported by Windows PowerShell 5.1. Ran
the copied kit script using the actual Windows PowerShell 5.1 executable:
startup/theme, HDR10 and ProRes analytic export all passed against the staged
package with restricted PATH. This tests script portability on the development
host; it does not certify a clean machine or installer lifecycle.

Development evidence: kit `results/model_free_smoke_69d05f9d744f49a3bdf0b6dadad41dbf/`.
Future clean results must go in a separate directory as documented in the kit.

Next external prerequisite: a disposable Windows x64 VM/PC with a fresh user
profile. Until available, preserve this kit and do not repeat local smoke tests
as a substitute. Signing, owner visual review and commercial clearance remain
open. No training, source-data edits, installation, release or push performed.
