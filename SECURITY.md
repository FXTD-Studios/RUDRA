# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: on the repository page, **Security › Report a vulnerability**
(https://github.com/FXTD-Studios/RUDRA/security/advisories/new). Say which version of RUDRA you
used, on which system, and how to reproduce it; a file that triggers the problem helps most.

We aim to answer within 7 days and to ship a fix, or say why not, within 30.

## Supported versions

Fixes go into the newest release. Older releases are not patched: update to the newest one.

## What RUDRA downloads and runs

- **Model packages** (File › Model packages) come over HTTPS and every file is checked against a
  SHA-256 before it is put in place.
- **AI fill's "Set up automatically"** downloads one fixed release of `uv` (checked by SHA-256),
  a private Python, and Python packages at the fixed versions in `scripts/genfill_constraints.txt`.
- **ffmpeg** in the Windows package is one pinned build, checked by SHA-256 when the package is made.
- The Windows and macOS packages are not yet signed with a paid certificate. Download RUDRA only
  from the project's GitHub releases page and compare the SHA-256 listed there.
