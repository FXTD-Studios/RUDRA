#!/usr/bin/env python3
"""Write AI fill v5's package SHA-256 values where the app and the publish script check them (RUDRA 1.3).

    python tools/finalize_genfill_v5.py            # after docs/models/genfill_v5/report_v5.json is written

The app offers a published package only when every file's SHA-256 is one it was built with
(native/app/model_downloads.cpp kKnownModelFiles), and scripts/PUBLISH_HF_MODELS.ps1 refuses a stage whose files
differ. Both read the values of docs/models/genfill_v5/registration_v5.json and report_v5.json from here. Run it
again after either file changes; it replaces the values it wrote before."""
import hashlib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/models/genfill_v5'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reg, rep = DOCS / 'registration_v5.json', DOCS / 'report_v5.json'
    for p in (reg, rep):
        if not p.is_file():
            sys.exit(f'missing {p}')
    hr, hp = sha(reg), sha(rep)
    cpp = ROOT / 'native/app/model_downloads.cpp'
    s = cpp.read_bytes().decode('utf-8')
    nl = '\r\n' if '\r\n' in s else '\n'
    block = (f'    "{hr}",   // registration_v5.json (docs/models/genfill_v5){nl}'
             f'    "{hp}",   // report_v5.json (docs/models/genfill_v5){nl}')
    marker = re.compile(r'    // @V5_DOCS@[^\r\n]*\r?\n')
    done = re.compile(r'    "[0-9a-f]{64}",   // registration_v5\.json \(docs/models/genfill_v5\)\r?\n'
                      r'    "[0-9a-f]{64}",   // report_v5\.json \(docs/models/genfill_v5\)\r?\n')
    if marker.search(s):
        s = marker.sub(lambda m: block, s, count=1)
    elif done.search(s):
        s = done.sub(lambda m: block, s, count=1)
    else:
        sys.exit('model_downloads.cpp has neither the @V5_DOCS@ marker nor the v5 lines')
    ps = ROOT / 'scripts/PUBLISH_HF_MODELS.ps1'
    t = ps.read_bytes().decode('utf-8')
    for name, h in (('registration_v5.json', hr), ('report_v5.json', hp)):
        key = re.escape('"models\\rudra_genfill_v5\\' + name + '"')
        t, n = re.subn('(' + key + r'\s+= ")[^"]*(")', lambda m, h=h: m.group(1) + h + m.group(2), t)
        if n != 1:
            sys.exit(f'PUBLISH_HF_MODELS.ps1: the {name} line was not found')
    cpp.write_bytes(s.encode('utf-8'))       # both found: write both
    ps.write_bytes(t.encode('utf-8'))
    print(f'registration_v5.json {hr}\nreport_v5.json       {hp}')


if __name__ == '__main__':
    main()
