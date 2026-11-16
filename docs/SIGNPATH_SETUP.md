# Signing the Windows release with SignPath (maintainers)

Free code signing for open source, from SignPath Foundation, set up for 1.2.5. The public page is
[CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md); this is how to set it up and run it.

## 1. Apply (once)

1. Fill in the team roles in [CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md) and publish that
   page on the website too (www.fxtdstudios.com, with a link from the RUDRA page). The README's
   "Code signing policy" section links it on GitHub.
2. Turn on multi-factor authentication for every member of the FXTD-Studios GitHub organisation
   (Settings > Authentication security > Require two-factor authentication).
3. Apply at https://signpath.org/apply with: the repository (github.com/FXTD-Studios/RUDRA), the
   licence (Apache-2.0), the release page, the code signing policy page, and the privacy policy
   (docs/PRIVACY.md). RUDRA's packages carry no model weights; every model is a separate
   download with its own licence.

SignPath's conditions and where RUDRA meets them: OSI licence (Apache-2.0, no commercial
dual-licensing); own binaries only (RUDRA.exe, rudra-native.exe, the installer; the libraries
others make are passed through unsigned); the privacy rule (docs/PRIVACY.md, the installer's
Privacy page and its "Check for updates automatically" choice, Help > Check for updates
automatically); uninstall (the installer's uninstaller); product name and version in every
signed file (native/app/windows/rudra.rc.in, native/cli/windows/rudra-native.rc.in, rudra.iss).

## 2. Set up SignPath (once, after the approval)

1. In SignPath: the project **RUDRA**, linked to the predefined **GitHub.com** trusted build
   system. Install the SignPath GitHub App on the repository.
2. Two artifact configurations, slugs **app** and **installer**: paste
   `.signpath/artifact-configurations/app.xml` and `installer.xml`.
3. The signing policy **release-signing** (the Foundation's certificate), with the approvers of
   CODE_SIGNING_POLICY.md and manual approval of each request.
4. A CI user with the Submitter role on that policy; create its API token.
5. In GitHub (Settings > Secrets and variables > Actions):
   - secret `SIGNPATH_API_TOKEN`: the CI user's token;
   - variable `SIGNPATH_ORGANIZATION_ID`: SignPath's organisation id;
   - only if the slugs differ from the above: variables `SIGNPATH_PROJECT_SLUG`,
     `SIGNPATH_SIGNING_POLICY_SLUG`.
   The old `WINDOWS_CERT_PFX` / `WINDOWS_CERT_PASSWORD` secrets are no longer read; delete them.

## 3. Each release

Push the tag as before. The Windows job of `release.yml` then:

1. builds and checks the package folder (`PACKAGE_WINDOWS.ps1 -NoArchive`);
2. sends it to SignPath (configuration **app**) and waits: **approve the request in SignPath**;
3. puts the signed RUDRA.exe and rudra-native.exe back, checks their signatures, and makes the
   ZIP and the installer (`PACKAGE_WINDOWS_ARCHIVE.ps1`);
4. sends the installer (configuration **installer**) and waits: **approve the second request**;
5. checks the installer's signature and writes its SHA-256 again.

Each wait lasts up to two hours (`wait-for-completion-timeout-in-seconds: 7200`); a request not
approved by then fails the job, and the release is not published. Re-run the job to try again.
Without the secret and the variable, or on a run that is not a tag, nothing is signed and the
job says so in a warning, as before 1.2.5.

The `signing-input-*` artifacts are the unsigned files SignPath fetched; they are kept one day
and are never published (the release job takes only `rudra-*`).

## Not covered

- The uninstaller that the installer writes (`unins000.exe`) is not signed.
- macOS: Apple's Developer ID, see [MACOS_SIGNING.md](MACOS_SIGNING.md).
- Local builds: `PACKAGE_WINDOWS.ps1` still signs with a .pfx through `RUDRA_SIGN_PFX` and
  `RUDRA_SIGN_PASSWORD` when they are set.
