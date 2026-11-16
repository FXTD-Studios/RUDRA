# Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io/), certificate by
[SignPath Foundation](https://signpath.org/).

RUDRA's Windows releases from 1.2.5 on are signed this way. This page says what is signed, how,
by whom, and what RUDRA sends over the network.

## What is signed

- `RUDRA.exe` and `rudra-native.exe`, RUDRA's own programs, in both the installer and the
  portable ZIP, and the installer `RUDRA-<version>-windows-x64-setup.exe`.
- Only the files of a release published on
  [GitHub](https://github.com/FXTD-Studios/RUDRA/releases). They are built from the tagged
  commit of this repository by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
  on GitHub-hosted runners, never on a team member's computer.
- A file is signed only when its version resource names the product RUDRA and the release's
  version ([`.signpath/artifact-configurations/`](../.signpath/artifact-configurations/)).
- Every release is two signing requests (the programs, then the installer), and each one is
  approved by hand by an approver below.

Not signed by us: the libraries in the package that others make (Qt, ONNX Runtime, DirectML,
OpenCV, OpenColorIO, ffmpeg, the Microsoft Visual C++ runtime) are shipped as their makers built
them; their own signatures, where they have one, are theirs. The macOS app is signed separately
([MACOS_SIGNING.md](MACOS_SIGNING.md)) and is not covered here.

## Team roles

| Role | Who |
|---|---|
| Committers and reviewers | FXTD Studios: [fxtdstudios](https://github.com/fxtdstudios) |
| Approvers | FXTD Studios: [fxtdstudios](https://github.com/fxtdstudios) |

Contact for signing and security questions: research@fxtdstudios.com.

- **Committers** may change the code. A change from anyone outside the team is reviewed by a
  committer before it is merged.
- **Approvers** approve each signing request in SignPath, after checking that it comes from the
  release workflow of a tag on `main`.
- Every team member signs in to GitHub and SignPath with multi-factor authentication.

## Privacy policy

This program will not transfer any information to other networked systems unless specifically
requested by the user or the person installing or operating it, with two exceptions, neither of
which sends anything about you: at most once a day it reads RUDRA's public release list on GitHub
to say when a newer version exists, and while its Check this computer window is open (the first
start opens it) it reads the public list of models it can download.
The installer shows this and asks whether to keep it; **Help > Check for updates
automatically** turns it off at any time. Nothing about you or your files is sent.

The whole policy, with every address RUDRA can connect to and when: [PRIVACY.md](PRIVACY.md).
The installer shows it as a page of its own.

## Checking a download

- Windows: right-click the file > **Properties** > **Digital Signatures**: the signer is
  SignPath Foundation. In PowerShell: `Get-AuthenticodeSignature .\RUDRA-<version>-windows-x64-setup.exe`.
- Each file on the release page has its SHA-256 beside it (`.sha256`).
- A new certificate gathers SmartScreen reputation over the first downloads, so Windows can
  still ask once ("More info > Run anyway") for a while after a release.

## Reporting

A signed file that is not ours, or one that behaves in a way this page does not describe:
report it privately as [SECURITY.md](../SECURITY.md) says. You can also write to SignPath
Foundation through [signpath.org](https://signpath.org/).
