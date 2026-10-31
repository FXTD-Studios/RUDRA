import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from pipeline.collect_next_hdr_sources import *
out=ROOT/'tears_of_steel';rows=json.loads((out/'samples.json').read_text());files=[]
# Keep the rejected original acquisition for audit; never repair nonfinite pixels.
initial=out/'initial_sequence_manifest.json'
if not initial.exists() and (out/'sequence_manifest.json').exists():
 initial.write_bytes((out/'sequence_manifest.json').read_bytes())
for r in rows:
 if r['scene'] not in ['03_2a','04_4k','04_5f','06_1a']:continue
 names=r['all_names'];start=max(0,len(names)//2-24)
 for name in names[start:start+48]:
  url=r['frame_url'].rsplit('/',1)[0]+'/'+name;path=out/'exr'/r['scene']/name
  info=download(url,path,200*1024**2);files.append(dict(scene=r['scene'],path=str(path),receipt=info,licence='CC-BY-3.0',training_enabled=False,split='unassigned',role='sparse_still_samples' if r['scene']=='04_4k' else 'continuous_sequence_candidate'))
  if len(files)%12==0:print('[sequence]',len(files),'/166 (three 48-frame sections and 22 sparse still samples)',flush=True);dump(out/'sequence_manifest.json',files)
 dump(out/'sequence_manifest.json',files)
dump(out/'status.json',dict(stage='downloaded_pending_full_decode_colour_and_split_review',frames=len(files),continuous_frames=144,sparse_still_frames=22,scenes=4,training_enabled=False))
