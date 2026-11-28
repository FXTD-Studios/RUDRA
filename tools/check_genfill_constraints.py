"""Whether AI fill's pinned packages can be installed together (28 Nov 2026).

scripts/genfill_constraints.txt pins every package "Set up automatically" installs. A version bump
(Dependabot's tokenizers 0.22.2 -> 0.23.3, #78) made the set impossible: transformers 5.13.1 asks
for tokenizers <= 0.23.0, uv refused the install, and the setup stopped at "the AI fill's packages
did not install" on every new computer. This resolves the bootstrap's package list against the
constraints for Windows, Linux and macOS with uv (nothing is installed) and fails when it cannot.

    python tools/check_genfill_constraints.py            # uv on PATH (pip install uv)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / 'scripts' / 'genfill_bootstrap.ps1'
CONSTRAINTS = ROOT / 'scripts' / 'genfill_constraints.txt'


def bootstrap_packages(text: str) -> list[str]:
    """The package list the Windows bootstrap installs ($pkgs = @(...)), and torch (installed on its own)."""
    m = re.search(r"\$pkgs\s*=\s*@\((.*?)\)", text, re.S)
    if not m:
        raise SystemExit(f'no $pkgs list in {BOOTSTRAP}')
    return re.findall(r"'([^']+)'", m.group(1)) + ['torch']


def main() -> int:
    uv = shutil.which('uv')
    if not uv:
        print('uv is not on PATH (pip install uv)', file=sys.stderr)
        return 2
    pkgs = bootstrap_packages(BOOTSTRAP.read_text(encoding='utf-8'))
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / 'genfill.in'
        req.write_text('\n'.join(pkgs) + '\n', encoding='utf-8')
        for platform in ('windows', 'linux', 'macos'):
            r = subprocess.run([uv, 'pip', 'compile', str(req), '-c', str(CONSTRAINTS), '--python-platform', platform,
                                '--python-version', '3.12', '--quiet', '--no-header', '-o', str(Path(tmp) / f'{platform}.txt')],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f'{platform}: the pinned packages install together')
            else:
                failed = True
                print(f'{platform}: NOT INSTALLABLE\n{r.stderr.strip()}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
