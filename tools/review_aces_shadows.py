"""Full-resolution shadow crops centred on largest candidate change at +3EV."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','new','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new directory required')
    old=json.loads(a.old.read_text());new=json.loads(a.new.read_text());view=ACES2View()
    a.out.mkdir(parents=True);records=[]
    for i in (5,6,7,8):
        o=old['rows'][i];n=new['rows'][i];assert o['asset_id']==n['asset_id']
        for item in (o,n):assert sha(item['target_path'])==item['target_sha256']
        assert sha(n['sdr_path'])==n['sdr_sha256']
        source=canonical_to_source(load_rgb(n['sdr_path'],hdr=False))
        before=np.load(o['target_path']);after=np.load(n['target_path'])
        support=source.max(-1)<5/255
        error=np.abs(view.forward(before*8)-view.forward(after*8)).max(-1)*support
        y,x=np.unravel_index(error.argmax(),error.shape)
        y0=max(0,min(y-64,source.shape[0]-128));x0=max(0,min(x-64,source.shape[1]-128))
        arrays=[view.inverse(source[y0:y0+128,x0:x0+128]),before[y0:y0+128,x0:x0+128],after[y0:y0+128,x0:x0+128]]
        sheet=Image.new('RGB',(768,560),'white');draw=ImageDraw.Draw(sheet)
        for j,ev in enumerate((0,3)):
            for k,(label,value) in enumerate(zip(('analytic','previous','candidate'),arrays)):
                draw.text((k*256+4,j*280+4),f'{label} {ev:+}EV',fill='black')
                pixels=np.rint(np.clip(view.forward(value*2.**ev),0,1)*255).astype(np.uint8)
                sheet.paste(Image.fromarray(pixels).resize((256,256),Image.Resampling.NEAREST),(k*256,j*280+24))
        sheet.save(a.out/f'case_{i:02d}.png')
        records.append({'case':i,'asset_id':n['asset_id'],'crop_xy':[int(x0),int(y0)],
                        'max_change_at_plus3_codes':float(error.max()*255)})
    (a.out/'crops.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
