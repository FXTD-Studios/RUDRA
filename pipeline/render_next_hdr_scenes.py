"""Native-1080p CPU re-renders of reviewed authored setups; no downloaded code."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
from mathutils import Vector
from render_highlight_scenes import material,cube,sphere,aim,area,texture_metal

def setup(job,frame,out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU'
    s.render.threads_mode='FIXED';s.render.threads=8
    s.cycles.samples=64;s.cycles.use_denoising=True;s.cycles.seed=job['seed']
    s.cycles.max_bounces=8;s.cycles.transmission_bounces=6;s.cycles.volume_bounces=1
    s.render.resolution_x=1920;s.render.resolution_y=1080;s.render.resolution_percentage=100
    s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
    s.render.image_settings.color_mode='RGB';s.render.image_settings.exr_codec='ZIP'
    s.render.fps=24;s.view_settings.view_transform='Standard';s.view_settings.look='None'
    s.view_settings.exposure=0;s.view_settings.gamma=1
    world=bpy.data.worlds.new('world');world.use_nodes=True;s.world=world
    wn=world.node_tree.nodes;bg=wn.get('Background');bg.inputs[0].default_value=(.02,.025,.035,1);bg.inputs[1].default_value=.15
    if job.get('hdri_path'):
        t=wn.new('ShaderNodeTexEnvironment');t.image=bpy.data.images.load(job['hdri_path']);t.image.colorspace_settings.name='Linear Rec.709'
        world.node_tree.links.new(t.outputs['Color'],bg.inputs[0]);bg.inputs[1].default_value=.3
    t=frame/24
    floor=material('floor',(.07,.085,.1),.4,.18)
    cube('floor',(0,0,-.12),(14,14,.2),floor)
    camd=bpy.data.cameras.new('camera');cam=bpy.data.objects.new('camera',camd);bpy.context.collection.objects.link(cam);s.camera=cam;camd.lens=45
    cam.location=(4.6+t*.7,-7,3.1);aim(cam,(0,0,1.25))
    area((2,-3,6),850,3);area((-4,0,3),450,2,(.55,.7,1))
    # Moving emissive strips make specular highlights move independently of the camera.
    for i,(col,power) in enumerate([((1,.75,.4),35),((.2,.5,1),55)]):
        cube('light_strip',((-1 if i==0 else 1)*(2.3-.4*t),1.5,2.3),(.2,.1,3),material('emitter',col,emission=power))
    kind=job['category'];grid_names=[]
    if kind=='metal':
        m=texture_metal(job['material_maps'])
        if 'NormalGL' in job['material_maps']:
            n=m.node_tree.nodes;l=m.node_tree.links
            tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(job['material_maps']['NormalGL']);tex.image.colorspace_settings.name='Non-Color'
            normal=n.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=.5
            l.new(tex.outputs['Color'],normal.inputs['Color']);l.new(normal.outputs[0],n.get('Principled BSDF').inputs['Normal'])
        ball=sphere((-1,0,1),m,.95);ball.rotation_euler.z=t*.6
        bpy.ops.mesh.primitive_torus_add(major_radius=.85,minor_radius=.27,location=(1.2,0,1.25),rotation=(1.1,t*.8,.3))
        o=bpy.context.object;o.data.materials.append(m)
        for p in o.data.polygons:p.use_smooth=True
    elif kind=='fire_reflection':
        bpy.ops.object.volume_import(filepath=job['vdb_frames'][frame]);o=bpy.context.object;o.data.grids.load()
        grid_names=[g.name for g in o.data.grids]
        if 'flames' not in grid_names and 'temperature' not in grid_names:raise ValueError('No fire field: '+str(grid_names))
        # One fixed transform for the whole sequence, derived in the plan inspection.
        o.scale=(job['volume_scale'],)*3;o.location=job['volume_location']
        m=bpy.data.materials.new('fire');m.use_nodes=True;n=m.node_tree.nodes;n.clear();l=m.node_tree.links
        output=n.new('ShaderNodeOutputMaterial');v=n.new('ShaderNodeVolumePrincipled')
        v.inputs['Density'].default_value=0;v.inputs['Density Attribute'].default_value=''
        v.inputs['Color'].default_value=(.12,.09,.07,1)
        attr=n.new('ShaderNodeAttribute');attr.attribute_name='flames' if 'flames' in grid_names else 'temperature'
        power=n.new('ShaderNodeMath');power.operation='MULTIPLY';power.inputs[1].default_value=job.get('emission_strength',30)
        l.new(attr.outputs['Fac'],power.inputs[0]);l.new(power.outputs[0],v.inputs['Emission Strength'])
        ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(1,.04,.002,1);ramp.color_ramp.elements[1].position=.7;ramp.color_ramp.elements[1].color=(1,.75,.2,1)
        l.new(attr.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],v.inputs['Emission Color'])
        if 'density' in grid_names:
            d=n.new('ShaderNodeAttribute');d.attribute_name='density'
            mult=n.new('ShaderNodeMath');mult.operation='MULTIPLY';mult.inputs[1].default_value=.35
            l.new(d.outputs['Fac'],mult.inputs[0]);l.new(mult.outputs[0],v.inputs['Density'])
        l.new(v.outputs[0],output.inputs['Volume']);o.data.materials.append(m)
        chrome=material('reflective_metal',(.75,.78,.8),1,.07)
        sphere((-1.7,0,.65),chrome,.65);cube('reflective_panel',(1.7,1,1.1),(1.5,.1,2.2),chrome)
        cam.location=(4.7,-7,3.2);aim(cam,(0,0,1.3))
    elif kind=='glass':
        m=material('glass',(.95,.98,1),0,.035)
        p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Transmission Weight'].default_value=1;p.inputs['IOR'].default_value=1.45
        sphere((-1,0,1.05),m,1);sphere((1.25,.2,.8),m,.75)
    elif kind in ('water','wet_floor'):
        m=material(kind,(.025,.05,.065),0,.08 if kind=='water' else .15)
        p=m.node_tree.nodes.get('Principled BSDF');p.inputs['IOR'].default_value=1.333
        p.inputs['Coat Weight'].default_value=1;p.inputs['Coat Roughness'].default_value=.035
        if kind=='water':p.inputs['Transmission Weight'].default_value=.65
        n=m.node_tree.nodes;l=m.node_tree.links;tex=n.new('ShaderNodeTexNoise');tex.noise_dimensions='4D'
        tex.inputs['Scale'].default_value=3.5;tex.inputs['Detail'].default_value=3;tex.inputs['W'].default_value=t*.6
        bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.24;bump.inputs['Distance'].default_value=.07
        l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],p.inputs['Normal'])
        cube('surface',(0,0,.02),(10,10,.08),m)
        sphere((-1.25,.2,.7),material('chrome',(.8,.8,.8),1,.08),.65)
        sphere((1.2,.4,.75),material('diffuse',(.2,.08,.035)),.7)
        cam.location=(3.6,-6,2.2);aim(cam,(0,.5,.6))
    elif kind=='sparks':
        rng=random.Random(job['seed']);mats=[material('spark'+str(i),c,emission=e) for i,(c,e) in enumerate([((1,.15,.01),30),((1,.5,.1),80),((1,.85,.5),160)])]
        for i in range(85):
            vel=Vector((rng.uniform(-2.5,2.5),rng.uniform(-.6,.6),rng.uniform(1.7,4)))
            age=t+rng.uniform(0,.4);pos=Vector((0,0,.3))+vel*age+Vector((0,0,-2*age*age))
            if pos.z<.03:continue
            curve=bpy.data.curves.new('spark','CURVE');curve.dimensions='3D';curve.bevel_depth=rng.uniform(.003,.009);curve.bevel_resolution=2
            spline=curve.splines.new('POLY');spline.points.add(1);spline.points[0].co=(*pos,1);spline.points[1].co=(*(pos-vel*.025),1)
            o=bpy.data.objects.new('spark',curve);bpy.context.collection.objects.link(o);curve.materials.append(mats[i%3])
        sphere((1,.2,.6),material('steel',(.65,.7,.75),1,.1),.6)
    else:raise ValueError(kind)
    path=out/'linear_exr'/f"{job['id']}_f{frame:04d}.exr";path.parent.mkdir(parents=True,exist_ok=True);s.render.filepath=str(path)
    start=time.time();bpy.ops.render.render(write_still=True)
    return dict(id=job['id'],frame=frame,path=str(path),category=kind,samples=64,renderer='Cycles CPU',threads=8,width=1920,height=1080,
        scene_linear_primaries='Rec.709',fps=24,source_job=job,grid_names=grid_names,render_seconds=time.time()-start,blender=bpy.app.version_string)

def inspect(plan,out):
    for j in plan['cg_jobs']:
        if j['category']!='fire_reflection':continue
        lows=[];highs=[];grids=set()
        for path in j['vdb_frames']:
            bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.object.volume_import(filepath=path)
            o=bpy.context.object;o.data.grids.load();bpy.context.view_layer.update()
            grids.update(g.name for g in o.data.grids)
            b=[Vector(x) for x in o.bound_box]
            lows.append([min(x[i] for x in b) for i in range(3)]);highs.append([max(x[i] for x in b) for i in range(3)])
        lo=Vector([min(x[i] for x in lows) for i in range(3)]);hi=Vector([max(x[i] for x in highs) for i in range(3)])
        if not ('flames' in grids or 'temperature' in grids):raise ValueError('Missing fire grids')
        scale=2.8/max(hi-lo);j['volume_scale']=scale;j['volume_location']=[-(lo.x+hi.x)/2*scale,-(lo.y+hi.y)/2*scale,-lo.z*scale]
        j['grid_names']=sorted(grids);j['fixed_sequence_bounds']=[list(lo),list(hi)]
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--inspect',action='store_true');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);plan=json.loads(Path(a.plan).read_text());done=0
    if a.inspect:inspect(plan,out);sys.exit(0)
    for j in plan['cg_jobs']:
        for frame in [j['sample_frame']]:
            rec=out/'render_records'/f"{j['id']}_f{frame:04d}.json"
            if rec.exists():continue
            r=setup(j,frame,out);rec.parent.mkdir(parents=True,exist_ok=True);rec.write_text(json.dumps(r,indent=2),encoding='utf-8')
            print('V3_NATIVE_RENDER',j['id'],frame,round(r['render_seconds'],1),flush=True);done+=1
            if a.limit and done>=a.limit:sys.exit(0)
