# RUDRA privacy policy

RUDRA runs on your computer. It has no account, no telemetry, no analytics and no crash
reporting. The pictures you open, grade and render, your projects and your settings stay on
your computer.

RUDRA connects to the internet only in the cases below. Each request carries what any web
request carries (your IP address, and "RUDRA/<version>" as the user agent) and nothing else:
no name, no files, no file names, no device identifier.

## 1. The update check (automatic, can be turned off)

At most once a day, when it starts, RUDRA reads the list of RUDRA releases from GitHub
(api.github.com/repos/FXTD-Studios/RUDRA/releases) to tell you when a newer version exists.
Nothing is downloaded or installed.

To turn it off, do any one of these:

- untick "Check for updates automatically" in the installer;
- in RUDRA, untick Help > Check for updates automatically;
- set the environment variable RUDRA_NO_UPDATE_CHECK=1.

Help > Check for updates... then checks once, only when you choose it.

## 2. Everything else happens only when you ask for it

- The model list: while the Check this computer window or the File > Model packages window is
  open, RUDRA reads the list of models you can download from
  www.fxtdstudios.com/rudra-docs/models.json. The first start opens Check this computer, and so
  does the first start after an update, so the list is read then too. Nothing is downloaded until
  you tick a model and press Install selected (or Download).
- Model downloads: when you press Download or Resume, the model's files come from
  huggingface.co. Every file is checked against a checksum built into RUDRA.
- AI fill, "Set up automatically": when you press it, RUDRA downloads uv
  (github.com/astral-sh/uv), Python 3.12 and the fixed Python packages the fill was tested
  with (pypi.org, download.pytorch.org), and the Wan2.1-VACE-1.3B base model from
  huggingface.co.
- Links you click (the user guide, a release page, a licence) open in your web browser.

GitHub, Hugging Face, PyPI, PyTorch and the host of www.fxtdstudios.com have their own
privacy policies for the requests they receive.

## Questions

Ask at https://github.com/FXTD-Studios/RUDRA/issues. This policy is part of RUDRA's code
signing policy (docs/CODE_SIGNING_POLICY.md).
