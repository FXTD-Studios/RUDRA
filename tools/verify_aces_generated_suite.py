"""Verify each completed registered export while the local review is running."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.verify_aces_generated_delivery import verify
from tools.audit_aces_generated_colour import audit
from training.aces_reference import reference_view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('G:/rudra_training/aces_v1_generated_review_20261024'))
    p.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_v1_generated_review_20261024.json'))
    a=p.parse_args();root=a.root
    reg=json.loads(a.registration.read_text())
    reference_view().set_threads(4);deadline=time.monotonic()+3600
    deliveries=[];colours=[]
    for source in reg['expected_inputs']:
        folder=root/source['clip'];report=folder/'report.json'
        while not report.exists():
            if time.monotonic()>deadline:raise TimeoutError(f'export not completed: {source["clip"]}')
            time.sleep(2)
        delivery_path=folder/'delivery_verification.json';colour_path=folder/'colour_audit.json'
        if delivery_path.exists() or colour_path.exists():raise ValueError('verification already exists; inspect before retry')
        delivery=verify(report);delivery_path.write_text(json.dumps(delivery,indent=2));deliveries.append(delivery)
        colour=audit(report);colour_path.write_text(json.dumps(colour,indent=2));colours.append(colour)
        print(source['clip'],'transport',delivery['pass'],flush=True)
    (root/'delivery_verification.json').write_text(json.dumps({'clips':deliveries,'pass':len(deliveries)==len(reg['expected_inputs']) and all(r['pass'] for r in deliveries),
        'scope':'decoded frame counts, tags, and exact sampled EXR-to-PQ codes; not perceptual quality'},indent=2))
    (root/'colour_audit.json').write_text(json.dumps({'clips':colours,'scope':'saved EXR sample colour fidelity including reconstructed regions; not HDR truth'},indent=2))


if __name__=='__main__':main()
