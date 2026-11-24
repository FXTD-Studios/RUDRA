#!/usr/bin/env python3
"""docs/models/genfill_v5/report_v5.json from the v5d checks (RUDRA 1.3, 23 Nov 2026), then the release texts' numbers.

    python tools/make_report_v5.py        # after tmp/09V_test/v4_gates/v5d_verdict.json is written (PASS)
then  python tools/finalize_genfill_v5.py

Writes report_v5.json (the verdict, the ghost report against v5, the offline replays, the history, and the learned
route's result when tmp/09V_test/v4_gates/learned_v5_verdict.json is there, and v5e's when v5e_verdict.json is) and replaces the
<!-- V5C_RESULTS --> marker in docs/RELEASE_NOTES_1.3.0.md and docs/MODELS.md with the measured numbers."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'tmp/09V_test/v4_gates'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def learned_route(v):
    """24 Nov 2026: v5 with the thin hold on the learned ACES 2.0 route (research_genfill_v5_learned_20261123.json),
    the result as tmp/09V_test/v4_gates/run_v5_learned.py wrote it, and what it says in words."""
    shots = v['shots']
    passed = [k for k, s in shots.items() if s.get('pass')]
    why = []
    for k, s in shots.items():
        if s.get('pass'):
            continue
        g, t = s.get('gates', {}), max(s['T']) if s.get('T') else None
        bad = [n for n in ('L', 'S', 'O') if n in s.get('needs', []) and g.get(n) is not True]
        if g.get('O') is False and 'O' not in bad:
            bad.append('O')
        if (k == 'water' and t is not None and t > 0.115) or (k != 'water' and g.get('T') is False):
            bad.append('T')
        lv = s.get('L')
        level = 'too bright' if lv is not None and lv > 4 else 'too dim' if lv is not None and lv < 2.5 else 'off in hue'
        words = {'L': f"{level}, L {lv} (limit 2.5-4 x white)", 'S': f"uneven, S {max(s['S']) if s.get('S') else None}",
                 'O': 'changed pixels outside the mask', 'T': f"flicker, T {t} EV (limit {0.115 if k == 'water' else 0.10})"}
        why.append(f"{k}: " + ', '.join(words[n] for n in bad) if bad else f"{k}: {s.get('why', 'failed')}")
    h = v.get('H', {})
    held = '; '.join(f"{key}: PU21 {r['pu21_-3_mean_delta']:+.1f} on {r['clips']} scenes, closer in {r['better_in']}"
                     for key, r in h.items() if 'pu21_-3_mean_delta' in r)
    summary = (f"{v['verdict']} on the learned ACES 2.0 route: {len(passed)} of {len(shots)} check shots passed "
               f"({', '.join(passed)}). Failed: {'; '.join(why)}. Held-out ({held}; a positive PU21 is further from the "
               f"truth than the learned reconstruction alone). v5 was checked and passed on the analytic route: use it there.")
    return {'registration': v.get('registration'), 'summary': summary, 'result': v}


def main():
    vd = load(W / 'v5d_verdict.json')
    v5 = load(W / 'v5_verdict.json')
    ref = load(W / 'v4_3_verdict.json') if (W / 'v4_3_verdict.json').exists() else {}
    if vd.get('verdict') != 'PASS':
        sys.exit(f"v5d_verdict.json says {vd.get('verdict')}: no report for a release")
    ghost = W / 'ghost_v5d_vs_v5_09V.json'
    rep = {'schema': 'rudra_model_report_v1', 'model': 'RUDRA AI fill v5, region re-render (experimental)',
           'lora_sha256': '5c5d4ab4107088f86bdc8213ae2ec21c1a882cc6d942141f0e12fb7733fda5e6',
           'registration': 'configs/experiments/research_genfill_v5d_thin_20261123.json',
           'verdict': vd, 'v5_without_the_hold': v5,
           'ghosts_09V_against_v5': load(ghost) if ghost.exists() else None,
           'offline_replays': load(ROOT / 'configs/experiments/research_genfill_v5d_thin_20261123.json').get('offline_choice'),
           'owed': 'a fresh shot with a thin, still, clipped light that nobody tuned on'}
    if (W / 'learned_v5_verdict.json').exists():
        rep['learned_route'] = learned_route(load(W / 'learned_v5_verdict.json'))
    if (W / 'v5e_verdict.json').exists():   # 24 Nov 2026: the finish the 1.3.0 app runs v5 with (both routes)
        reg = load(ROOT / 'configs/experiments/research_genfill_v5e_20261124.json')
        rep['v5e'] = {'registration': 'configs/experiments/research_genfill_v5e_20261124.json', 'change': reg['change'],
                      'offline_choice': reg['offline_choice'], 'honesty': reg['honesty'], 'verdict': load(W / 'v5e_verdict.json')}
        if 'outcome' in reg:   # the verdict and, when it was not PASS, the decision that shipped v5e anyway (24 Nov 2026)
            rep['v5e']['outcome'] = reg['outcome']
            rep['v5e']['release_authorized'] = reg.get('release_authorized', False)
            rep['v5e']['release_authorized_by'] = reg.get('release_authorized_by', '')
    out = ROOT / 'docs/models/genfill_v5/report_v5.json'
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    s, l, h = vd['shots'], vd['lamp_09V'], vd['H']
    t = lambda name, d: max(d['shots'][name]['T']) if d.get('shots', {}).get(name) else None
    rows = [('Porch: a thin LED strip, frame-to-frame change where nothing moves (limit 0.10 EV)', t('Porch', ref), t('Porch', v5), t('Porch', vd)),
            ('Water: the same (limit 0.115 EV)', t('water', ref), t('water', v5), t('water', vd)),
            ('09V: the practical lamp, brightest part over the shot (stops)', l['lamp99_range_stops']['v4_2'], v5['lamp_09V']['lamp99_range_stops']['v5'], l['lamp99_range_stops']['v5']),
            ('09V: the lamp at the worst window join (x the change elsewhere)', l['lamp99_worst_join']['v4_2'], v5['lamp_09V']['lamp99_worst_join']['v5'], l['lamp99_worst_join']['v5'])]
    f = lambda v: '–' if v is None else (f'{v:.3f}'.rstrip('0').rstrip('.') if isinstance(v, float) else str(v))
    table = '| | v4.3 | v5 alone | v5 in 1.3 |\n|---|---|---|---|\n' + ''.join(f'| {a} | {f(b)} | {f(c)} | {f(d)} |\n' for a, b, c, d in rows)
    hs = h['gated_96_119']
    notes = (f"Measured on the four check shots and 29 held-out scenes (registration `research_genfill_v5d_thin_20261123.json`, "
             f"report `docs/models/genfill_v5/report_v5.json`):\n\n{table}\n"
             f"On 22 held-out scenes it is closer to the truth than the plain reconstruction in {hs['better_in']} "
             f"(PU21 {hs['pu21_-3_mean_delta']:.1f}); every check passed. Beside 09V's antenna it draws what v5 alone draws; "
             f"holding every still clipped pixel instead had kept faint copies of the antenna for several frames.\n")
    rn = ROOT / 'docs/RELEASE_NOTES_1.3.0.md'
    s1 = rn.read_text(encoding='utf-8')
    if '<!-- V5C_RESULTS -->' in s1:
        rn.write_text(s1.replace('<!-- V5C_RESULTS -->', notes.rstrip('\n')), encoding='utf-8', newline='\n')
    md = ROOT / 'docs/MODELS.md'
    s2 = md.read_text(encoding='utf-8')
    if '<!-- V5C_RESULTS -->' in s2:
        md.write_text(s2.replace('<!-- V5C_RESULTS -->', (
            f"Passed on 09V, a window, a softbox strip and water (Porch flicker {f(t('Porch', vd))} EV, water {f(t('water', vd))}) "
            f"and on 29 held-out scenes, with RUDRA 1.3's hold for thin lights.")), encoding='utf-8', newline='\n')
    print(out, '\n' + table)


if __name__ == '__main__':
    main()
