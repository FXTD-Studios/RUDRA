"""Document a completed CC0 collection and its overlap with older auxiliary data.

Only writes inside the new source collection. Never changes a training corpus.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from PIL import Image, ImageDraw

from collect_commercial_highlights import OUT, EXISTING, dump, file_hash

AUX = Path('G:/datasets/corpora/rudra_commercial_review_20261018_v1/auxiliary_fxtd_camera_polyhaven.jsonl')


def capture(aid):
    return aid.removesuffix('_puresky')


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    if summary['stage'] != 'collection_verified' or (OUT/'collector.lock').exists():
        raise RuntimeError('Collection must finish before review')
    selected = json.loads((OUT/'selection.json').read_text())
    existing = defaultdict(list)
    for p in EXISTING.glob('*_2k.exr'):
        existing[capture(p.stem.removesuffix('_2k'))].append(str(p))
    old = defaultdict(list)
    for line in AUX.read_text().splitlines():
        row = json.loads(line)
        if row.get('licence_source') != 'polyhaven':
            continue
        source = row['scene_id'].split('::')[-1]
        aid = re.sub(r'_\d+k$', '', source)
        old[capture(aid)].append({'asset_id': row['asset_id'], 'split': row['split']})
    overlaps = []
    for a in selected:
        aid = a['asset_id']; group = capture(aid)
        a['same_capture_resolution_group'] = group
        a['existing_capture_paths'] = existing.get(group, []) if a['kind'] == 'hdri' else []
        if a['kind'] == 'hdri':
            a['source_colour_interpretation_verified'] = False
            a['radiometric_scale'] = 'relative scene-linear; not calibrated absolute nits'
        if aid == 'fireplace':
            a['category'] = 'fireplace_environment'
        elif aid == 'veld_fire':
            a['category'] = 'burned_landscape_smoke_reference'
        rows = old.get(group, []) if a['kind'] == 'hdri' else []
        splits = sorted({r['split'] for r in rows})
        a['previous_auxiliary_splits'] = splits
        a['split_eligibility'] = 'blocked_pending_capture_group_reconciliation' if len(splits)>1 else 'pending_capture_and_location_split_review'
        a['split_group_policy'] = 'All resolutions, puresky variants, crops and renders of a capture stay together; also review location groups. No split assigned yet.'
        overlaps.append(dict(asset_id=aid, capture_group=group,
            previous_source_paths=a['existing_capture_paths'], previous_auxiliary_rows=rows,
            previous_auxiliary_splits=splits, mixed_prior_splits=len(splits)>1))
    dump(OUT/'selection.json', selected)
    categories = {a['asset_id']: a['category'] for a in selected}
    manifest = OUT/'files_manifest.jsonl'
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    for r in rows:
        r['category'] = categories[r['asset_id']]
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    plan = json.loads((OUT/'download_plan.json').read_text())
    for r in plan['files']:
        r['category'] = categories[r['asset_id']]
    dump(OUT/'download_plan.json', plan)
    dump(OUT/'existing_corpus_overlap.json', dict(
        reference_manifest=str(AUX), reference_sha256=file_hash(AUX), assets=overlaps,
        policy='Older auxiliary pairs are not in the active Z run. Do not reuse their scene-linear targets as display-referred targets. Freeze capture/location groups before future rendering.'))
    summary.update(categories=dict(Counter(a['category'] for a in selected)),
        unique_hdri_capture_groups=len({capture(a['asset_id']) for a in selected if a['kind']=='hdri'}),
        hdri_assets_overlapping_existing_capture=sum(bool(a['existing_capture_paths']) for a in selected),
        hdri_assets_overlapping_previous_auxiliary=sum(bool(a['previous_auxiliary_splits']) for a in selected),
        prior_mixed_split_groups=sorted({r['capture_group'] for r in overlaps if r['mixed_prior_splits']}),
        manifest_sha256=file_hash(manifest))
    dump(OUT/'summary.json', summary); dump(OUT/'status.json', summary)
    panoramas=[a for a in selected if a['kind']=='hdri']
    sheet=Image.new('RGB',(1536,218*((len(panoramas)+3)//4)),(24,27,32))
    draw=ImageDraw.Draw(sheet)
    for i,a in enumerate(panoramas):
        x=(i%4)*384;y=(i//4)*218
        with Image.open(OUT/'qc_previews'/(a['asset_id']+'.jpg')) as im:
            sheet.paste(im.resize((384,192)),(x,y))
        draw.text((x+5,y+198),a['asset_id'],fill=(240,240,240))
    sheet.save(OUT/'qc_previews/contact_sheet.jpg',quality=90)
    dump(OUT/'excluded_sources.json', dict(downloaded_from_these_sources=False, sources=[
        dict(provider='ActionVFX', status='Excluded: AI training requires prior written consent; no agreement obtained.', evidence='https://www.actionvfx.com/faq'),
        dict(provider='RawFilm', status='Excluded: published licence prohibits AI training.', evidence='https://raw.film/legal'),
        dict(provider='ARRI sample footage', status='Not downloaded: no verified commercial AI training grant for exact files.'),
        dict(provider='Shutterstock Data Licensing', status='Not downloaded: bespoke data licence and RAW availability not agreed.', evidence='https://www.shutterstock.com/data-licensing')]))
    report = f'''# Commercial highlight source collection — 18 October 2026

Downloaded and verified on G:. Sources from **Poly Haven**, CC0-1.0.
The provider's [asset licence](https://polyhaven.com/license) permits commercial
use, and its [FAQ](https://docs.polyhaven.com/en/faq) explicitly permits AI training.
These asset permissions do not certify the complete model or authorize release.

## Contents

| Content | Assets | Files / purpose |
| --- | ---: | --- |
| Clear sun, cloud detail, sunrise/sunset, overcast | 32 | 8K floating-point EXR panoramas; eight in each category |
| Fireplace environment | 1 | Static indoor HDR panorama; not fire video |
| Burned landscape / smoke reference | 1 | Veld Fire panorama; not a flame dataset |
| Metal materials | 12 | 52 selected 2K PNG material maps for future synthetic scene rendering |

**{summary['assets']} assets, {summary['files']} media files, {summary['bytes']/1e9:.3f} GB.**
All publisher MD5 checksums and sizes match; local SHA-256 hashes are recorded.
All files fully decode and contain finite numeric values. EXR geometry and floating
point storage were checked. These checks do not prove unclipped sun/fire, absolute
luminance calibration or model quality. The panoramas are photographic HDRIs,
not Alexa/ARRIRAW footage. Material maps alone are not HDR target images.

## Commercial-use evidence

- `licence_review.json`: URLs, retrieval times and SHA-256 of saved evidence.
- `licence_evidence/`: licence, explicit AI permission FAQ and API terms snapshots.
- `metadata/`: original official API catalogue and per-asset file records.
- `selection.json`: asset pages, authors, CC0 status, category and capture groups.
- `files_manifest.jsonl`: media paths, publisher MD5, local SHA-256 and decode QC.
- `excluded_sources.json`: sources not downloaded because training permission is
  absent, prohibited or requires a separate contract.

Downloads used the [official API](https://polyhaven.com/our-api), with a unique
User-Agent and two concurrent transfers under its published terms. Only asset
media are intended as future training inputs; website text and licence evidence
are documentation, not training data. No paid source was purchased.

## Training readiness and overlap

This is a **verified source pack**, not a new paired training corpus.
No training pairs were created and the active `fxtd_z_hdr_run01` was not changed.
All new records retain `training_enabled: false` and an unassigned split.

The 34 HDRI assets represent {summary['unique_hdri_capture_groups']} capture groups
after grouping normal/puresky variants. {summary['hdri_assets_overlapping_existing_capture']}
selected HDRI assets overlap captures already present at 2K on G:;
{summary['hdri_assets_overlapping_previous_auxiliary']} overlap the older auxiliary
manifest. Higher resolution and alternate panoramas do not create independent
scenes. See `existing_corpus_overlap.json` for exact previous rows and splits.
Location grouping is an additional heuristic, not exhaustive visual deduplication.
Prior mixed-split capture groups: **{', '.join(summary['prior_mixed_split_groups']) or 'none found'}**.
These groups are explicitly blocked pending reconciliation; older auxiliary
validation results must not be treated as independent for these captures.
This finding does not affect the current Z-footage run, which excludes that auxiliary corpus.

Before a later training run: freeze scene/location groups across resolutions and
all derivatives; review representative views and highlight clipping; confirm source
colour interpretation; render perspective views and consistent SDR/HDR pairs using
the chosen display target pipeline. Do not feed scene-linear panoramas directly
into the current display-referred objective. For metal maps, render physically
based scenes with varied geometry, lighting and motion before pairing.

## Remaining coverage gaps

No commercially licensed camera RAW fire, spark or moving reflective-metal video
was acquired. The fireplace and burned-landscape panoramas cannot validate flame
or spark motion or temporal stability. Obtain owned footage or a written licence
explicitly allowing commercial ML training and use of resulting model weights.
Do not infer training permission from a standard stock-footage licence.

## Files and next operator

`summary.json` is the completion receipt; `qc_previews/` contains exposure-compressed
512-pixel previews generated from the downloaded EXRs for inspection only.
`visual_review.json` records the contact-sheet review and its limits. Provider
category tags are not precise manual labels. The puresky lower hemisphere needs
exclusion when generating natural photographic views.
G: is sufficient for this collection; Z: is not needed. Preserve original Z: camera
sources for the separate existing corpus. No model promotion, deployment or release
was performed. Re-read live trainer status before any future training action.
'''
    (OUT/'README.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
