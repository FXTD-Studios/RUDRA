"""CPU-only, authored synthetic scenes. Never open or execute downloaded blends.

Blender --background --factory-startup --disable-autoexec --threads 2 --python
pipeline/render_highlight_scenes.py -- --plan PLAN --out OUT
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector


def material(name, color, metal=0, rough=.35, emission=0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metal
    p.inputs['Roughness'].default_value = rough
    p.inputs['Emission Color'].default_value = (*color, 1)
    p.inputs['Emission Strength'].default_value = emission
    return m


def cube(name, pos, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    o = bpy.context.object; o.name = name; o.scale = scale
    o.data.materials.append(mat)
    return o


def aim(obj, point):
    obj.rotation_euler = (Vector(point)-obj.location).to_track_quat('-Z', 'Y').to_euler()


def area(pos, power, size, color=(1,1,1)):
    d = bpy.data.lights.new('softbox', 'AREA'); d.energy = power; d.shape = 'DISK'; d.size = size; d.color = color
    o = bpy.data.objects.new('softbox', d); bpy.context.collection.objects.link(o); o.location = pos; aim(o, (0,0,1))


def sphere(pos, mat, radius=1):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=pos)
    o = bpy.context.object; o.data.materials.append(mat)
    for p in o.data.polygons: p.use_smooth = True
    return o


def texture_metal(paths):
    m = material('approved_polyhaven_metal', (.5,.5,.5), 1, .24)
    n = m.node_tree.nodes; links = m.node_tree.links; p = n.get('Principled BSDF')
    for role, socket in [('Diffuse','Base Color'),('Rough','Roughness'),('Metal','Metallic')]:
        if role not in paths: continue
        t = n.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(paths[role])
        t.image.colorspace_settings.name = 'sRGB' if role == 'Diffuse' else 'Non-Color'
        links.new(t.outputs['Color'], p.inputs[socket])
    return m


def setup(job, frame, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene; s.render.engine = 'CYCLES'; s.cycles.device = 'CPU'
    s.render.threads_mode = 'FIXED'; s.render.threads = 2
    s.cycles.samples = 32; s.cycles.use_denoising = True; s.cycles.seed = job['seed']
    s.cycles.max_bounces = 6; s.cycles.volume_bounces = 1
    s.render.resolution_x = 640; s.render.resolution_y = 360; s.render.resolution_percentage = 100
    s.render.image_settings.file_format = 'OPEN_EXR'; s.render.image_settings.color_depth = '32'
    s.render.image_settings.color_mode = 'RGB'; s.render.image_settings.exr_codec = 'ZIP'
    s.render.film_transparent = False; s.render.fps = 24
    s.view_settings.view_transform = 'Standard'; s.view_settings.look = 'None'
    s.view_settings.exposure = 0; s.view_settings.gamma = 1
    world = bpy.data.worlds.new('controlled_world'); world.use_nodes = True; s.world = world
    world.node_tree.nodes.get('Background').inputs[0].default_value = (.06,.07,.1,1)
    world.node_tree.nodes.get('Background').inputs[1].default_value = .12
    floor = material('floor',(.10,.12,.15), .2, .25)
    cube('floor', (0,0,-.1),(12,12,.2),floor)
    camd = bpy.data.cameras.new('camera'); cam = bpy.data.objects.new('camera', camd)
    bpy.context.collection.objects.link(cam); s.camera = cam; camd.lens = 45
    cam.location = (4.5+frame*.025,-7,3.3); aim(cam,(0,0,1.2))
    area((2,-3,6), 1100, 3); area((-4,0,3), 700, 2,(.5,.7,1))
    kind = job['category']; animated = []
    if kind in ('metal_polished','metal_textured'):
        if job.get('hdri_path'):
            n = world.node_tree.nodes; t = n.new('ShaderNodeTexEnvironment')
            t.image = bpy.data.images.load(job['hdri_path']); t.image.colorspace_settings.name = 'Linear Rec.709'
            world.node_tree.links.new(t.outputs['Color'],n.get('Background').inputs[0])
            n.get('Background').inputs[1].default_value = .4
        m = texture_metal(job['material_maps']) if kind == 'metal_textured' else material('chrome',(.8,.82,.85),1,.08)
        animated.append(sphere((-1,0,1.1),m))
        bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=.28, location=(1.3,.1,1.3), rotation=(1.2,frame*.08,.4))
        bpy.context.object.data.materials.append(m)
        cube('bright_strip',(-2,2,3),(1,.1,3),material('strip',(1,.85,.65),emission=45))
    elif kind.startswith('sparks'):
        bright = kind == 'sparks_bright'
        cube('background',(0,2,2),(10,.2,6),material('back',(.55,.6,.65) if bright else (.008,.009,.012)))
        sphere((0,0,.7),material('metal_bowl',(.3,.3,.3),1,.25),.6)
        rng = random.Random(job['seed'])
        mats = [material('spark_'+str(i),c,emission=e) for i,(c,e) in enumerate([
            ((1,.19,.015),30),((1,.55,.08),70),((1,.85,.5),180)])]
        for i in range(65):
            x,y,z = rng.uniform(-1.5,1.5),rng.uniform(-.6,.6),rng.uniform(.6,3.4)
            velocity = (rng.uniform(-.5,.5),rng.uniform(-.1,.1),rng.uniform(.4,1.4))
            pos = Vector((x,y,z))+Vector(velocity)*(frame/24)
            curve = bpy.data.curves.new('spark_trail','CURVE'); curve.dimensions='3D'; curve.bevel_depth=rng.uniform(.003,.012); curve.bevel_resolution=2
            spline=curve.splines.new('POLY'); spline.points.add(1)
            spline.points[0].co=(*pos,1); spline.points[1].co=(*(pos-Vector(velocity)*.07),1)
            o=bpy.data.objects.new('spark',curve); bpy.context.collection.objects.link(o); curve.materials.append(mats[i%3])
    elif kind == 'neon':
        cube('wall',(0,1.8,2),(9,.2,6),material('wall',(.035,.04,.05)))
        for i,c in enumerate([(1,.01,.02),(.02,.05,1),(.01,1,.15)]):
            bpy.ops.mesh.primitive_torus_add(major_radius=.6,minor_radius=.035,location=((i-1)*1.6,0,1.5),rotation=(math.pi/2,0,0))
            bpy.context.object.data.materials.append(material('neon'+str(i),c,emission=35+frame*2))
        sphere((0,-1,.55),material('reflector',(.7,.7,.7),1,.1),.5)
    elif kind == 'bright_window':
        wall=material('room_wall',(.25,.22,.18)); emitter=material('window_light',(.7,.85,1),emission=65)
        cube('window',(0,1.6,2.4),(3.4,.1,2.5),emitter)
        for x in [-1.8,0,1.8]:cube('frame',(x,1.4,2.4),(.12,.2,2.8),wall)
        cube('frame',(0,1.4,2.4),(3.6,.2,.12),wall)
        sphere((-.8,-.5,.8),material('diffuse_subject',(.35,.12,.05)),.8)
        sphere((1,-.2,.6),material('reflective_subject',(.8,.8,.8),1,.12),.6)
    else:
        path=job['vdb_frames'][frame]
        bpy.ops.object.volume_import(filepath=path)
        o=bpy.context.object; o.data.grids.load(); bpy.context.view_layer.update()
        bounds=[Vector(p) for p in o.bound_box]
        lo=Vector([min(p[i] for p in bounds) for i in range(3)]); hi=Vector([max(p[i] for p in bounds) for i in range(3)])
        scale=2.8/max(hi-lo); o.scale=(scale,)*3; o.location=(-(lo.x+hi.x)/2*scale,-(lo.y+hi.y)/2*scale,-lo.z*scale)
        m=bpy.data.materials.new('flame_volume'); m.use_nodes=True; n=m.node_tree.nodes; n.clear(); l=m.node_tree.links
        outnode=n.new('ShaderNodeOutputMaterial'); volume=n.new('ShaderNodeVolumePrincipled')
        volume.inputs['Density'].default_value=0
        volume.inputs['Density Attribute'].default_value=''
        volume.inputs['Color'].default_value=(.15,.1,.07,1)
        flame=n.new('ShaderNodeAttribute'); flame.attribute_name='flames'
        power=n.new('ShaderNodeMath'); power.operation='MULTIPLY'; power.inputs[1].default_value=55
        l.new(flame.outputs['Fac'],power.inputs[0]); l.new(power.outputs[0],volume.inputs['Emission Strength'])
        ramp=n.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color=(1,.025,.001,1); ramp.color_ramp.elements[1].position=.8; ramp.color_ramp.elements[1].color=(1,.65,.12,1)
        l.new(flame.outputs['Fac'],ramp.inputs[0]); l.new(ramp.outputs[0],volume.inputs['Emission Color'])
        if kind=='campfire':
            density=n.new('ShaderNodeAttribute'); density.attribute_name='density'; l.new(density.outputs['Fac'],volume.inputs['Density'])
        l.new(volume.outputs[0],outnode.inputs['Volume']); o.data.materials.append(m)
        cam.location=(4,-6,2.8); aim(cam,(0,0,1.2))
    path=out/'linear_exr'/f"{job['id']}_f{frame:04d}.exr"; path.parent.mkdir(parents=True,exist_ok=True)
    s.render.filepath=str(path)
    bpy.ops.render.render(write_still=True)
    return {'id':job['id'],'category':kind,'frame':frame,'path':str(path),'scene_linear_primaries':'Rec.709',
            'renderer':'Cycles CPU','samples':32,'denoising':True,'blender':bpy.app.version_string,'fps':24,'source_job':job}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--plan',required=True); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(a.plan).read_text())['cg_jobs']; done=0
    for j in jobs:
        for frame in range(4):
            record=out/'render_records'/f"{j['id']}_f{frame:04d}.json"
            if record.exists(): continue
            r=setup(j,frame,out); record.parent.mkdir(exist_ok=True); record.write_text(json.dumps(r,indent=2),encoding='utf-8')
            print('HIGHLIGHT_RENDER',j['id'],frame,flush=True); done+=1
            if a.limit and done>=a.limit:sys.exit(0)
