"""Read downloaded scene dependencies and volume headers; never render or save scenes.

Run using Blender --background --factory-startup --disable-autoexec --threads 2
--python pipeline/inspect_blender_gap_sources.py. No downloaded scripts execute.
"""
import json
import re
from pathlib import Path

import bpy

ROOT = Path(r"G:\datasets\sources\blender_jangafx_gap_sources_20261018_v1")
report = {"blender_version": bpy.app.version_string, "renders_performed": 0,
          "training_enabled": False, "scenes": [], "volume_sequences": []}
for path in sorted(ROOT.rglob("*.blend")):
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
    images = []
    for im in bpy.data.images:
        packed = bool(im.packed_file or im.packed_files)
        resolved = bpy.path.abspath(im.filepath, library=im.library) if im.filepath else ""
        images.append({"name": im.name, "source": im.source, "packed": packed,
                       "filepath": im.filepath, "external_exists": Path(resolved).is_file() if resolved else None})
    libraries = [{"path": lib.filepath, "exists": Path(bpy.path.abspath(lib.filepath)).is_file()}
                 for lib in bpy.data.libraries]
    report["scenes"].append({"file": str(path), "opened": True,
        "scenes": [{"name": s.name, "frame_start": s.frame_start, "frame_end": s.frame_end,
                    "engine": s.render.engine} for s in bpy.data.scenes],
        "material_count": len(bpy.data.materials), "object_count": len(bpy.data.objects),
        "images": images, "libraries": libraries,
        "missing_external_images": [i for i in images if i["source"] in ("FILE", "SEQUENCE", "MOVIE")
                                    and not i["packed"] and not i["external_exists"]]})
    print("INSPECTED", path.name, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
for name in ("campfire", "simple_fire"):
    numbered = []
    for p in (ROOT / "unpacked" / name).rglob("*.vdb"):
        m = re.search(r"(\d+)\.vdb$", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    frames = [p for _, p in sorted(numbered)]
    if not frames:
        # Recorded, not fatal: the scene results above must still be saved.
        report["volume_sequences"].append({"id": name, "frame_count": 0, "sampled_headers": [],
                                           "error": "no numbered .vdb frames found (archive not unpacked?)"})
        continue
    samples = []
    for p in [frames[i] for i in sorted({0, len(frames) // 2, len(frames) - 1})]:
        volume = bpy.data.volumes.new("source_validation")
        volume.filepath = str(p)
        volume.grids.load()
        samples.append({"file": str(p), "loaded": volume.grids.is_loaded,
                        "error": volume.grids.error_message,
                        "grids": [{"name": g.name, "type": g.data_type} for g in volume.grids]})
        bpy.data.volumes.remove(volume)
    report["volume_sequences"].append({"id": name, "frame_count": len(frames), "sampled_headers": samples})
(ROOT / "source_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("SOURCE INSPECTION SAVED; render quality and numeric voxel checks not performed.", flush=True)
