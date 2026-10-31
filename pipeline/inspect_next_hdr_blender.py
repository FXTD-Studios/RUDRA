import bpy,json
from pathlib import Path
root=Path('G:/datasets/sources/next_hdr_sources_20261019_v3/blender')
paths=[root/'unpacked/classroom/classroom.blend',root/'downloads/basic_particle_simulation.blend'];records=[]
for p in paths:
 bpy.ops.wm.open_mainfile(filepath=str(p),use_scripts=False)
 missing=[]
 for im in bpy.data.images:
  if im.source in ('FILE','SEQUENCE','MOVIE') and not (im.packed_file or im.packed_files) and im.filepath:
   actual=Path(bpy.path.abspath(im.filepath,library=im.library))
   if not actual.is_file():missing.append(dict(image=im.name,path=str(actual)))
 records.append(dict(path=str(p),opened=True,version=bpy.app.version_string,scenes=[dict(name=s.name,engine=s.render.engine,width=s.render.resolution_x,height=s.render.resolution_y,frames=[s.frame_start,s.frame_end]) for s in bpy.data.scenes],objects=len(bpy.data.objects),missing_images=missing,missing_libraries=[l.filepath for l in bpy.data.libraries if not Path(bpy.path.abspath(l.filepath)).is_file()],script_execution_enabled=False))
(root/'scene_inspection.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
