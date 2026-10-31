"""Validate explicit paired schedules before any optimizer updates."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registered_schedule(directory, output, *, steps, batch, crop, border, paired_motion):
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory/'manifest.json').read_text())
    for key, expected in dict(steps=steps, batch=batch, crop=crop, border=border, paired_motion=paired_motion).items():
        if manifest[key] != expected:
            raise ValueError('schedule geometry mismatch: '+key)
    if not paired_motion:
        raise ValueError('registered schedule requires paired motion')
    for name in ('crop_pool', 'motion_pool'):
        if file_sha(output/(name+'.json')) != manifest[name+'_sha256']:
            raise ValueError('schedule pool identity: '+name)
    for path, digest in manifest['pilot_hashes'].items():
        if file_sha(path) != digest:
            raise ValueError('schedule pilot identity')
    for path, digest in ((directory/'schedule.json', manifest['schedule_sha256']),
                         (Path(manifest['control_schedule_path']), manifest['control_schedule_sha256'])):
        if file_sha(path) != digest:
            raise ValueError('schedule content identity')
    schedule = json.loads((directory/'schedule.json').read_text())
    control = json.loads(Path(manifest['control_schedule_path']).read_text())
    lengths = [len(json.loads((output/(name+'.json')).read_text())) for name in ('crop_pool', 'motion_pool')]
    motion_pool = json.loads((output/'motion_pool.json').read_text())
    if len(schedule) != steps or len(control) != steps:
        raise ValueError('schedule step count')
    for step, indices in enumerate(schedule):
        motion = step % 2 == 0
        if (len(indices) != (batch//2 if motion else batch) or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < lengths[int(motion)] for i in indices)):
            raise ValueError('invalid schedule batch')
        if not motion and indices != control[step]:
            raise ValueError('still control schedule changed')
        if motion and manifest.get('preserve_motion_groups'):
            if [motion_pool[i]['group'] for i in indices] != [motion_pool[i]['group'] for i in control[step]]:
                raise ValueError('motion scene/exposure slot changed')
    return schedule
