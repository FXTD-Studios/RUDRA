"""Authored training scenes for the AI fill v5 (22 Nov 2026): what CC0 panoramas rarely hold sharply.

    blender -b --factory-startup --python-exit-code 1 -P tools/blender/genfill_v5_scenes.py -- --skies G:/datasets/sources/genfill_v5/manifest.jsonl
            --out G:/datasets/sources/genfill_v5/blender [--count 200] [--start 0] [--device GPU]

Every scene is procedural geometry (nothing downloaded but the CC0 sky it is lit by) with a moving camera,
rendered by Cycles to scene-linear EXR (Linear Rec.709, half float), 33 frames at 1024 x 576 (the builder
takes 512 x 288 from it). Five kinds, in turn:

  thin       antennas and masts, power cables (catenaries), railings, chain-link and wire grids, bare
             branches, against the brightest part of the sky; 0.3-6 px wide at the clip's size, so their
             edges blend into the clipped sky the way the antenna on 09V does
  door       a dark interior (a cargo bay, a corridor) opening onto a bright exterior: sky, a sea or ground
             plane with its glints, figures and an antenna in the opening, a floor grating (09V's shot)
  window     a room lit only by the sky through windows with mullions, glazing bars or blinds
  practical  a dim room with lamps: bare bulbs, shaded lamps, fluorescent tubes, LED panels, neon in colour
  glare      the sun itself in frame (only skies whose sun is not clipped: manifest light.plateau false),
             with branches and wires across it (the builder adds the lens glare: recipe v5)

Each scene draws its own parameters from --seed + its index (a re-run makes the same scene). Variety in
what the clip edges look like: object width and distance, depth of field on a quarter of the scenes,
motion blur on a third, the camera's lens (24-85 mm) and move (pan, tilt, dolly, handheld wobble).

Written: <out>/<kind>_<index>/f.0001.exr ... f.0033.exr and scene.json, and a row in <out>/manifest.jsonl
when the scene is complete (asset_id, scene_id 'blender:<kind>_<index>', kind, the sky's scene_id, licence:
authored geometry, FXTD Studios, lit by a CC0 sky). A scene already in the manifest is skipped (resume).
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

KINDS = ('thin', 'door', 'window', 'practical', 'glare')
SKY_THEMES = {'thin': ('sky', 'sun', 'thin'), 'door': ('sky', 'sun', 'thin'), 'window': ('sky', 'sun', 'thin'),
              'practical': ('sky', 'sun', 'thin', 'window'), 'glare': ('sky', 'sun')}


# ------------------------------------------------------------------------------- the world
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if hasattr(bpy.data, 'colorspace') and hasattr(bpy.data.colorspace, 'working_space'):
        try:
            bpy.data.colorspace.working_space = 'Linear Rec.709'
        except Exception:                                   # noqa: BLE001 - older or custom OCIO config
            pass
    return bpy.context.scene


def working_space() -> str:
    try:
        return str(bpy.data.colorspace.working_space)
    except Exception:                                       # noqa: BLE001 - Blender < 5: always Linear Rec.709
        return 'Linear Rec.709'


def set_world(scene, sky_path, strength=1.0):
    w = bpy.data.worlds.new('sky')
    scene.world = w
    try:
        w.use_nodes = True
    except Exception:                                       # noqa: BLE001 - Blender 5: always nodes
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    if sky_path:
        env = nt.nodes.new('ShaderNodeTexEnvironment')
        env.image = bpy.data.images.load(str(sky_path), check_existing=True)
        env.interpolation = 'Cubic'
        nt.links.new(env.outputs['Color'], bg.inputs['Color'])
    else:
        bg.inputs['Color'].default_value = (0.02, 0.02, 0.025, 1)
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    return env.image if sky_path else None


def brightest_direction(image, stride=8):
    """The direction (world, unit) of the image's brightest area: Blender's equirectangular lookup
    (u = -atan2(y, x) / 2pi + 0.5, v = atan2(z, hypot(x, y)) / pi + 0.5; rows from the bottom), on a
    coarse grid, blurred by averaging a block."""
    import numpy as np
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    y = px.reshape(h, w, 4)[..., :3] @ np.array([.2126, .7152, .0722], np.float32)
    hh, ww = h // stride, w // stride
    y = np.nan_to_num(y[:hh * stride, :ww * stride].reshape(hh, stride, ww, stride).mean((1, 3)))
    r, c = np.unravel_index(int(np.argmax(y)), y.shape)
    u, v = (c + 0.5) / ww, (r + 0.5) / hh
    return equirect_to_dir(u, v), float(y.max()), float(np.median(y))


def equirect_to_dir(u, v):
    phi = -(u - 0.5) * 2 * math.pi
    el = (v - 0.5) * math.pi
    return Vector((math.cos(el) * math.cos(phi), math.cos(el) * math.sin(phi), math.sin(el)))


# ------------------------------------------------------------------------------- materials
def material(name, color=(0.1, 0.1, 0.1), rough=0.6, metal=0.0, emit=None, emit_strength=0.0, glossy_water=False):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:                                       # noqa: BLE001
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    if emit is not None:
        e = nt.nodes.new('ShaderNodeEmission')
        e.inputs['Color'].default_value = (*emit, 1)
        e.inputs['Strength'].default_value = emit_strength
        nt.links.new(e.outputs['Emission'], out.inputs['Surface'])
        return m
    p = nt.nodes.new('ShaderNodeBsdfPrincipled')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    if glossy_water:
        p.inputs['IOR'].default_value = 1.33
    nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
    return m


def dark_paint(rng, name='paint'):
    g = rng.uniform(0.015, 0.35) if rng.random() < 0.85 else rng.uniform(0.5, 0.85)    # some white masts
    tint = [g * rng.uniform(0.85, 1.15) for _ in range(3)]
    return material(name, tint, rough=rng.uniform(0.25, 0.9), metal=rng.choice([0.0, 0.0, 1.0]))


# ------------------------------------------------------------------------------- geometry
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def tubes(name, lines, mat, resolution=3):
    """Thin round things, many in one object: lines = [[(x, y, z, radius), ...], ...] (poly splines)."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 1.0
    cu.bevel_resolution = resolution
    cu.use_fill_caps = True
    for pts in lines:
        if len(pts) < 2:
            continue
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for p, (x, y, z, r) in zip(sp.points, pts):
            p.co = (x, y, z, 1.0)
            p.radius = r
    ob = link(bpy.data.objects.new(name, cu))
    ob.data.materials.append(mat)
    return ob


def box(name, center, size, mat, rot_z=0.0):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    v = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
         (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    me.update()
    ob = link(bpy.data.objects.new(name, me))
    ob.location = (cx, cy, cz)
    ob.rotation_euler = (0, 0, rot_z)
    ob.data.materials.append(mat)
    return ob


def sphere(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center, segments=24, ring_count=12)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.materials.append(mat)
    return ob


def plane(name, center, size, mat):
    return box(name, center, (size[0], size[1], 0.001), mat)


def catenary(a, b, sag, n=48):
    a, b = Vector(a), Vector(b)
    out = []
    for i in range(n + 1):
        t = i / n
        p = a.lerp(b, t)
        p.z -= sag * 4 * t * (1 - t)
        out.append(p)
    return out


def branch_lines(rng, base, direction, length, radius, depth, out):
    """A bare tree: each branch a tapering poly line, children at its end and along it."""
    d = Vector(direction).normalized()
    pts, p, r = [], Vector(base), radius
    segs = 6
    for k in range(segs + 1):
        pts.append((p.x, p.y, p.z, max(r, 0.0015)))
        if k < segs:
            jitter = Vector((rng.gauss(0, 0.15), rng.gauss(0, 0.15), rng.gauss(0, 0.1)))
            d = (d + jitter).normalized()
            p = p + d * (length / segs)
            r *= 0.93
    out.append(pts)
    if depth <= 0:
        return
    for _ in range(rng.randint(2, 3)):
        k = rng.randint(segs // 2, segs)
        x, y, z, rr = pts[k]
        axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.uniform(0.2, 1.0))).normalized()
        child = (d * 0.5 + axis).normalized()
        branch_lines(rng, (x, y, z), child, length * rng.uniform(0.55, 0.75), rr * rng.uniform(0.55, 0.7), depth - 1, out)


# ------------------------------------------------------------------------------- the camera
def camera(scene, rng, origin, target, frames, lens=None):
    cam = bpy.data.cameras.new('cam')
    cam.lens = lens or rng.uniform(24, 85)
    cam.sensor_width = 36
    ob = link(bpy.data.objects.new('cam', cam))
    scene.camera = ob
    origin, target = Vector(origin), Vector(target)
    fwd = (target - origin).normalized()
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd)
    pan = math.radians(rng.uniform(-0.35, 0.35))           # per frame
    tilt = math.radians(rng.uniform(-0.12, 0.12))
    dolly = rng.uniform(-0.03, 0.03) * (target - origin).length / 20
    wob = math.radians(rng.uniform(0.0, 0.12))
    ph = [rng.uniform(0, 6.28) for _ in range(3)]
    still = rng.random() < 0.2                             # a locked-off camera: the fill must hold still
    prev = None
    for f in range(frames):
        t = 0 if still else f
        jy = wob * (math.sin(0.37 * t + ph[0]) + 0.5 * math.sin(0.91 * t + ph[1]))
        jx = wob * (math.sin(0.29 * t + ph[2]) + 0.5 * math.sin(0.83 * t + ph[0]))
        d = (fwd + right * math.tan(pan * t + jy) + up * math.tan(tilt * t + jx)).normalized()
        ob.location = origin + fwd * dolly * t
        # 22 Nov 2026: each frame's Euler angles the closest to the previous frame's. Plain to_euler() can
        # jump by 360 degrees between two frames (an angle crossing +-180), and Cycles' motion blur then
        # sweeps the whole turn into the frame (9 of the first 61 motion-blurred scenes smeared in a swirl).
        q = d.to_track_quat('-Z', 'Y')
        prev = q.to_euler('XYZ', prev) if prev is not None else q.to_euler('XYZ')
        ob.rotation_euler = prev
        ob.keyframe_insert('location', frame=f + 1)
        ob.keyframe_insert('rotation_euler', frame=f + 1)
    return ob, fwd, right, up, {'lens': round(cam.lens, 1), 'pan_deg_per_frame': round(math.degrees(pan), 3),
                                'tilt_deg_per_frame': round(math.degrees(tilt), 3), 'dolly_m_per_frame': round(dolly, 4),
                                'wobble_deg': round(math.degrees(wob), 3), 'still': still}


def in_view(origin, fwd, right, up, lens, dist, sx, sy, aspect=16 / 9):
    """A point at `dist` along the view, sx, sy in -1..1 across the frame's width and height."""
    half = 18.0 / lens
    return Vector(origin) + fwd * dist + right * (sx * half * dist) + up * (sy * half / aspect * dist)


# ------------------------------------------------------------------------------- the kinds
def thin_things(rng, cam_o, fwd, right, up, lens, n=None, near=3.0, far=60.0):
    """Antennas, cables, railings, wire grids and bare branches spread across the view."""
    made = []
    for i in range(n or rng.randint(2, 4)):
        what = rng.choice(['antenna', 'cables', 'railing', 'grid', 'branches', 'antenna', 'cables', 'branches'])
        dist = math.exp(rng.uniform(math.log(near), math.log(far)))
        p = in_view(cam_o, fwd, right, up, lens, dist, rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.3))
        mat = dark_paint(rng, f'{what}_{i}')
        px = dist * 36.0 / lens / 1024                       # metres per pixel at this distance (1024 wide)
        width_px = math.exp(rng.uniform(math.log(0.3), math.log(6.0)))
        r = width_px * px / 2
        if what == 'antenna':
            h = rng.uniform(1.5, 8.0) * dist / 10
            lines = [[(p.x, p.y, p.z - h * 0.6, r * 4), (p.x, p.y, p.z + h * 0.4, r * 2.5)],      # the mast
                     [(p.x, p.y, p.z + h * 0.4, r), (p.x + rng.gauss(0, 0.02) * h, p.y, p.z + h * rng.uniform(0.8, 1.4), r * 0.6)]]
            for k in range(rng.randint(0, 6)):                                                  # yagi / cross bars
                z = p.z + h * rng.uniform(0.0, 0.4)
                L = h * rng.uniform(0.05, 0.25)
                a = rng.uniform(0, math.pi)
                lines.append([(p.x - L * math.cos(a), p.y - L * math.sin(a), z, r * 0.7),
                              (p.x + L * math.cos(a), p.y + L * math.sin(a), z, r * 0.7)])
            made.append(tubes(f'antenna_{i}', lines, mat))
        elif what == 'cables':
            span = rng.uniform(1.5, 4.0) * dist * 36 / lens
            a = p - right * span / 2 + Vector((0, 0, rng.uniform(-0.1, 0.1) * span))
            b = p + right * span / 2 + Vector((0, 0, rng.uniform(-0.1, 0.1) * span))
            sag = rng.uniform(0.01, 0.12) * span
            lines = []
            for k in range(rng.randint(1, 5)):
                off = Vector((0, 0, k * rng.uniform(0.3, 1.2) * dist / 20))
                lines.append([(q.x, q.y, q.z, r) for q in catenary(a + off, b + off, sag)])
            made.append(tubes(f'cables_{i}', lines, mat, resolution=2))
        elif what == 'railing':
            w = rng.uniform(2, 6)
            gap = rng.uniform(0.08, 0.2)
            top = p.z
            a, b = p - right * w / 2, p + right * w / 2
            lines = [[(a.x, a.y, top, r * 3), (b.x, b.y, top, r * 3)], [(a.x, a.y, top - 0.5, r * 2), (b.x, b.y, top - 0.5, r * 2)]]
            for k in range(int(w / gap)):
                q = a.lerp(b, (k + 0.5) / int(w / gap))
                lines.append([(q.x, q.y, top - 1.0, r * 1.5), (q.x, q.y, top, r * 1.5)])
            made.append(tubes(f'railing_{i}', lines, mat))
        elif what == 'grid':
            w, gap = rng.uniform(1.5, 4.0), rng.uniform(0.04, 0.15)
            lines = []
            a = p - right * w / 2
            lo, hi = p.z - 3 * w, p.z + w * 0.3              # a fence: from below the frame up into the sky
            for k in range(int(w / gap) + 1):
                q = a + right * (k * gap)
                lines.append([(q.x, q.y, lo, r), (q.x, q.y, hi, r)])
            for k in range(int((hi - lo) / gap) + 1):
                z = lo + k * gap
                lines.append([(a.x, a.y, z, r), ((a + right * w).x, (a + right * w).y, z, r)])
            for q in (a, a + right * w):                     # its posts
                lines.append([(q.x, q.y, lo, r * 6), (q.x, q.y, hi + 0.1, r * 6)])
            made.append(tubes(f'grid_{i}', lines, mat, resolution=1))
        else:
            lines = []
            base = p - Vector((0, 0, rng.uniform(2, 6) * dist / 10)) + right * rng.uniform(-1, 1) * dist * 0.2
            branch_lines(rng, base, (rng.gauss(0, 0.3), rng.gauss(0, 0.3), 1.0), rng.uniform(1.5, 4) * dist / 10,
                         max(r * 8, 0.01), rng.randint(3, 5), lines)
            made.append(tubes(f'branches_{i}', lines, mat, resolution=2))
        made[-1]['rudra_what'] = what
        made[-1]['rudra_width_px'] = round(width_px, 2)
    return made


def ground(rng, z=0.0, water=None):
    water = rng.random() < 0.5 if water is None else water
    if water:
        m = material('water', (0.01, 0.02, 0.025), rough=rng.uniform(0.02, 0.25), glossy_water=True)
    else:
        g = rng.uniform(0.03, 0.25)
        m = material('ground', (g, g * 0.95, g * 0.85), rough=0.9)
    plane('ground', (0, 0, z), (4000, 4000), m)
    return 'water' if water else 'ground'


def build_thin(scene, rng, sky_dir, frames):
    o = Vector((0, 0, rng.uniform(1.2, 20)))
    aim = sky_dir.copy()
    aim.z = max(aim.z, math.sin(math.radians(rng.uniform(-3, 25))))
    aim = (aim + Vector((rng.gauss(0, 0.15), rng.gauss(0, 0.15), 0))).normalized()
    cam, f, r, u, mv = camera(scene, rng, o, o + aim * 10, frames)
    gnd = ground(rng) if rng.random() < 0.6 else None
    things = thin_things(rng, o, f, r, u, cam.data.lens)
    return cam, {'ground': gnd, 'things': [(t['rudra_what'], t['rudra_width_px']) for t in things], **mv}


def build_door(scene, rng, sky_dir, frames):
    W, H, D = rng.uniform(3, 8), rng.uniform(2.4, 5), rng.uniform(6, 20)
    wall = material('inside', [rng.uniform(0.02, 0.12)] * 3, rough=rng.uniform(0.3, 0.9), metal=rng.choice([0, 1]))
    t = 0.2
    box('floor', (0, D / 2, -t / 2), (W, D, t), wall)
    box('roof', (0, D / 2, H + t / 2), (W, D, t), wall)
    box('left', (-W / 2 - t / 2, D / 2, H / 2), (t, D, H), wall)
    box('right', (W / 2 + t / 2, D / 2, H / 2), (t, D, H), wall)
    box('back', (0, -t / 2, H / 2), (W, t, H), wall)
    ow, oh = W * rng.uniform(0.5, 1.0), H * rng.uniform(0.6, 1.0)          # the opening at y = D
    side = (W - ow) / 2
    if side > 0.01:
        box('jamb_l', (-W / 2 + side / 2, D + t / 2, H / 2), (side, t, H), wall)
        box('jamb_r', (W / 2 - side / 2, D + t / 2, H / 2), (side, t, H), wall)
    if H - oh > 0.01:
        box('lintel', (0, D + t / 2, oh + (H - oh) / 2), (ow, t, H - oh), wall)
    if rng.random() < 0.6:                                                    # a floor grating: thin over bright
        gm = dark_paint(rng, 'grating')
        gap = rng.uniform(0.05, 0.15)
        lines = [[(x, D * 0.6, 0.02, 0.008), (x, D + 2.0, 0.02, 0.008)] for x in [(-ow / 2 + k * gap) for k in range(int(ow / gap) + 1)]]
        tubes('grating', lines, gm, resolution=1)
    gz = -rng.uniform(1, 30)
    gnd = ground(rng, gz)
    figs = []
    for k in range(rng.randint(0, 3)):                                        # figures in the opening
        x, y = rng.uniform(-ow / 2.5, ow / 2.5), D - rng.uniform(0.5, 3)
        m = material(f'fig{k}', [rng.uniform(0.01, 0.06)] * 3, rough=0.8)
        tubes(f'fig{k}', [[(x, y, 0.0, 0.17), (x, y, 1.45, 0.2)], [(x - 0.25, y, 1.4, 0.06), (x - 0.3, y, 0.8, 0.05)],
                          [(x + 0.25, y, 1.4, 0.06), (x + 0.45, y + 0.2, 1.1, 0.05)]], m)
        sphere(f'head{k}', (x, y, 1.65), 0.12, m)
        if rng.random() < 0.5:
            tubes(f'whip{k}', [[(x - 0.2, y - 0.1, 1.3, 0.004), (x - 0.25, y - 0.1, 2.6, 0.002)]], m)
        figs.append(round(x, 2))
    o = Vector((rng.uniform(-W / 4, W / 4), rng.uniform(0.5, D * 0.5), rng.uniform(1.2, min(2.2, H - 0.3))))
    target = Vector((rng.uniform(-ow / 4, ow / 4), D + 5, o.z + rng.uniform(-1.5, 1.0)))
    cam, f, r, u, mv = camera(scene, rng, o, target, frames, lens=rng.uniform(18, 50))
    return cam, {'room_w_h_d': [round(W, 2), round(H, 2), round(D, 2)], 'opening': [round(ow, 2), round(oh, 2)],
                 'ground': gnd, 'ground_z': round(gz, 1), 'figures': len(figs), **mv}


def build_window(scene, rng, sky_dir, frames):
    W, H, D = rng.uniform(3, 7), rng.uniform(2.4, 3.5), rng.uniform(3, 8)
    c = rng.uniform(0.15, 0.7)
    wall = material('wall', (c, c * rng.uniform(0.9, 1.0), c * rng.uniform(0.8, 1.0)), rough=0.85)
    t = 0.25
    box('floor', (0, D / 2, -t / 2), (W, D, t), material('floor', [rng.uniform(0.05, 0.4)] * 3, rough=0.6))
    box('ceiling', (0, D / 2, H + t / 2), (W, D, t), wall)
    box('left', (-W / 2 - t / 2, D / 2, H / 2), (t, D, H), wall)
    box('right', (W / 2 + t / 2, D / 2, H / 2), (t, D, H), wall)
    box('back', (0, -t / 2, H / 2), (W, t, H), wall)
    n = rng.randint(1, 3)                                                    # windows in the far wall (y = D)
    sill, top = rng.uniform(0.3, 1.0), H - rng.uniform(0.2, 0.6)
    ww = W / n * rng.uniform(0.4, 0.8)
    xs = [(-W / 2 + (k + 0.5) * W / n) for k in range(n)]
    edges = [-W / 2] + [x for c_ in xs for x in (c_ - ww / 2, c_ + ww / 2)] + [W / 2]
    for k in range(0, len(edges), 2):
        a, b = edges[k], edges[k + 1]
        if b - a > 0.01:
            box(f'pier{k}', ((a + b) / 2, D + t / 2, H / 2), (b - a, t, H), wall)
    box('below', (0, D + t / 2, sill / 2), (W, t, sill), wall)
    box('above', (0, D + t / 2, top + (H - top) / 2), (W, t, H - top), wall)
    frame_m = dark_paint(rng, 'frame') if rng.random() < 0.6 else material('frame', [rng.uniform(0.6, 0.9)] * 3, rough=0.4)
    style = rng.choice(['mullions', 'grid', 'blinds', 'plain'])
    lines = []
    for c_ in xs:
        a, b = c_ - ww / 2, c_ + ww / 2
        if style in ('mullions', 'grid'):
            for k in range(1, rng.randint(2, 4)):
                x = a + k * (b - a) / rng.randint(2, 4)
                lines.append([(x, D, sill, 0.02), (x, D, top, 0.02)])
        if style == 'grid':
            for k in range(1, rng.randint(2, 5)):
                z = sill + k * (top - sill) / 4
                lines.append([(a, D, z, 0.012), (b, D, z, 0.012)])
        if style == 'blinds':
            gap = rng.uniform(0.04, 0.09)
            for k in range(int((top - sill) / gap)):
                z = sill + k * gap
                lines.append([(a, D - 0.05, z, 0.004), (b, D - 0.05, z, 0.004)])
    if lines:
        tubes('frames', lines, frame_m, resolution=1)
    if rng.random() < 0.5:                                                   # something outside: trees, wires
        o_ = Vector((0, D + 1, 1.5))
        thin_things(rng, o_, Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1)), 35, n=rng.randint(1, 3), near=8, far=60)
    gnd = ground(rng, -rng.uniform(0, 15))
    o = Vector((rng.uniform(-W / 4, W / 4), rng.uniform(0.3, D * 0.4), rng.uniform(1.2, 1.8)))
    target = Vector((rng.choice(xs) + rng.uniform(-0.5, 0.5), D, (sill + top) / 2 + rng.uniform(-0.4, 0.4)))
    cam, f, r, u, mv = camera(scene, rng, o, target, frames, lens=rng.uniform(20, 50))
    return cam, {'windows': n, 'style': style, 'ground': gnd, **mv}


def build_practical(scene, rng, sky_dir, frames):
    W, H, D = rng.uniform(3, 10), rng.uniform(2.4, 5), rng.uniform(4, 15)
    c = rng.uniform(0.05, 0.5)
    wall = material('wall', (c, c * rng.uniform(0.85, 1.0), c * rng.uniform(0.7, 1.0)), rough=rng.uniform(0.4, 0.95))
    t = 0.25
    for nm, ctr, sz in (('floor', (0, D / 2, -t / 2), (W, D, t)), ('ceiling', (0, D / 2, H + t / 2), (W, D, t)),
                        ('left', (-W / 2 - t / 2, D / 2, H / 2), (t, D, H)), ('right', (W / 2 + t / 2, D / 2, H / 2), (t, D, H)),
                        ('back', (0, -t / 2, H / 2), (W, t, H)), ('front', (0, D + t / 2, H / 2), (W, t, H))):
        box(nm, ctr, sz, wall)
    o = Vector((rng.uniform(-W / 4, W / 4), rng.uniform(0.3, D * 0.3), rng.uniform(1.2, 1.9)))
    target = Vector((rng.uniform(-W / 4, W / 4), D, rng.uniform(1.0, H - 0.3)))
    cam, f, r, u, mv = camera(scene, rng, o, target, frames, lens=rng.uniform(20, 60))
    lamps = []
    for k in range(rng.randint(1, 6)):
        kind = rng.choice(['bulb', 'shaded', 'tube', 'panel', 'neon', 'bulb'])
        if k == 0 or rng.random() < 0.6:                    # the first lamp, and most, where the camera sees them
            kind = 'bulb' if kind in ('panel', 'shaded') and k == 0 else kind
            q = in_view(o, f, r, u, cam.data.lens, rng.uniform(1.5, max(2.0, (D - o.y) * 0.9)),
                        rng.uniform(-0.75, 0.75), rng.uniform(-0.6, 0.6))
            x, y, z = min(max(q.x, -W / 2 + 0.3), W / 2 - 0.3), min(max(q.y, 0.3), D - 0.2), min(max(q.z, 0.3), H - 0.15)
        else:
            x, y, z = rng.uniform(-W / 2.5, W / 2.5), rng.uniform(D * 0.4, D * 0.95), rng.uniform(0.8, H - 0.2)
        temp = rng.choice([(1.0, 0.62, 0.32), (1.0, 0.8, 0.6), (1.0, 0.95, 0.9), (0.85, 0.92, 1.0)])
        s = 10 ** rng.uniform(1.0, 3.5)
        if kind == 'neon':
            temp = rng.choice([(1, 0.05, 0.1), (0.1, 0.4, 1), (0.1, 1, 0.3), (1, 0.2, 0.9), (1, 0.5, 0.0)])
        em = material(f'lamp{k}', emit=temp, emit_strength=s)
        if kind == 'bulb':
            sphere(f'bulb{k}', (x, y, z), rng.uniform(0.02, 0.07), em)
            tubes(f'cord{k}', [[(x, y, z, 0.004), (x, y, H, 0.004)]], dark_paint(rng))
        elif kind == 'shaded':
            sphere(f'bulb{k}', (x, y, z), 0.04, em)
            sh = material(f'shade{k}', [rng.uniform(0.3, 0.9)] * 3, rough=0.7)
            tubes(f'shade{k}', [[(x, y, z - 0.12, 0.25), (x, y, z + 0.1, 0.12)]], sh, resolution=6)
        elif kind == 'tube':
            L = rng.uniform(0.6, 1.5)
            tubes(f'tube{k}', [[(x - L / 2, y, z, 0.013), (x + L / 2, y, z, 0.013)]], em)
        elif kind == 'panel':                              # on the ceiling, or a light box on the wall facing the camera
            if rng.random() < 0.5:
                plane(f'panel{k}', (x, y, H - 0.01), (rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2)), em)
            else:
                box(f'panel{k}', (x, D - 0.03, z), (rng.uniform(0.3, 1.5), 0.02, rng.uniform(0.2, 0.9)), em)
        else:
            pts = [(x + 0.3 * math.cos(a_) * rng.uniform(0.8, 1.2), y - 0.05, z + 0.2 * math.sin(2 * a_), 0.008)
                   for a_ in [i_ * 0.35 for i_ in range(18)]]
            tubes(f'neon{k}', [pts], em)
        lamps.append({'kind': kind, 'strength': round(s, 1), 'colour': temp})
    return cam, {'lamps': lamps, 'room_w_h_d': [round(W, 2), round(H, 2), round(D, 2)], **mv}


def build_glare(scene, rng, sky_dir, frames):
    o = Vector((0, 0, rng.uniform(1.5, 10)))
    aim = (sky_dir + Vector((rng.gauss(0, 0.08), rng.gauss(0, 0.08), rng.gauss(0, 0.05)))).normalized()
    cam, f, r, u, mv = camera(scene, rng, o, o + aim * 10, frames, lens=rng.uniform(24, 70))
    gnd = ground(rng) if rng.random() < 0.6 else None
    things = thin_things(rng, o, f, r, u, cam.data.lens, n=rng.randint(1, 3), near=5, far=80)
    return cam, {'ground': gnd, 'things': [(t['rudra_what'], t['rudra_width_px']) for t in things], **mv}


BUILD = {'thin': build_thin, 'door': build_door, 'window': build_window, 'practical': build_practical, 'glare': build_glare}


# ------------------------------------------------------------------------------- rendering
def setup_render(scene, rng, a, folder):
    scene.render.engine = 'CYCLES'
    cyc = scene.cycles
    if a.device == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for kind in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
            try:
                prefs.compute_device_type = kind
                prefs.refresh_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    cyc.device = 'GPU'
                    break
            except Exception:                               # noqa: BLE001 - that backend is not here
                continue
    cyc.samples = a.samples
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.01
    cyc.use_denoising = a.denoise
    cyc.sample_clamp_direct = 0.0
    cyc.sample_clamp_indirect = 20.0
    cyc.max_bounces = 6
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = a.width, a.height, 100
    r.film_transparent = False
    r.use_motion_blur = rng.random() < 0.33
    r.motion_blur_shutter = rng.uniform(0.25, 0.6)
    scene.frame_start, scene.frame_end = 1, a.frames
    im = r.image_settings
    if hasattr(im, 'media_type'):
        try:
            im.media_type = 'IMAGE'
        except Exception:                                   # noqa: BLE001
            pass
    im.file_format = 'OPEN_EXR'
    im.color_mode = 'RGB'
    im.color_depth = '16'
    im.exr_codec = 'ZIP'
    r.filepath = str(folder / 'f.')
    r.use_file_extension = True
    scene.view_settings.view_transform = 'Standard'          # the EXR stays scene-linear either way
    return {'motion_blur': r.use_motion_blur, 'shutter': round(r.motion_blur_shutter, 3) if r.use_motion_blur else 0,
            'device': str(cyc.device)}


def focus(scene, rng, cam):
    if rng.random() >= 0.25:
        return {'dof': False}
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = math.exp(rng.uniform(math.log(1.0), math.log(40.0)))
    cam.data.dof.aperture_fstop = rng.uniform(1.4, 8.0)
    return {'dof': True, 'focus_m': round(cam.data.dof.focus_distance, 2), 'fstop': round(cam.data.dof.aperture_fstop, 2)}


def load_skies(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    return [r for r in rows if r.get('path') and Path(r['path']).exists()]


def pick_sky(rng, skies, kind):
    want = SKY_THEMES[kind]
    pool = [s for s in skies if set(s.get('themes', [])) & set(want)]
    if kind == 'glare':
        pool = [s for s in pool if not (s.get('light') or {}).get('plateau') and (s.get('light') or {}).get('dynamic_range_stops', 0) >= 12]
    pool = pool or skies
    return pool[rng.randrange(len(pool))] if pool else None


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='genfill_v5_scenes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--skies', default='', help='the collector manifest (tools/collect_genfill_v5.py); none: a plain dark world')
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261122)
    ap.add_argument('--kinds', default=','.join(KINDS))
    ap.add_argument('--frames', type=int, default=33)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=576)
    ap.add_argument('--samples', type=int, default=96)
    ap.add_argument('--no-denoise', dest='denoise', action='store_false')
    ap.add_argument('--device', choices=('GPU', 'CPU'), default='GPU')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = [k for k in a.kinds.split(',') if k]
    for k in kinds:
        if k not in BUILD:
            sys.exit(f'unknown kind {k}; one of {", ".join(KINDS)}')
    skies = load_skies(a.skies) if a.skies else []
    man = out / 'manifest.jsonl'
    done = {json.loads(l)['scene_id'] for l in man.read_text(encoding='utf-8').splitlines() if l.strip()} if man.exists() else set()
    for idx in range(a.start, a.start + a.count):
        kind = kinds[idx % len(kinds)]
        name = f'{kind}_{idx:05d}'
        sid = f'blender:{name}'
        if sid in done:
            continue
        rng = random.Random(a.seed * 1000003 + idx)
        t0 = time.time()
        scene = reset()
        sky = pick_sky(rng, skies, kind) if skies else None
        strength = 1.0 if kind != 'practical' else 10 ** rng.uniform(-2.5, -1.0)
        img = set_world(scene, sky['path'] if sky else None, strength)
        if img is not None:
            sky_dir, peak, med = brightest_direction(img)
            rot = rng.uniform(0, 2 * math.pi)                # turn the world, not the sky's own direction
            sky_dir = Vector((sky_dir.x * math.cos(rot) - sky_dir.y * math.sin(rot),
                              sky_dir.x * math.sin(rot) + sky_dir.y * math.cos(rot), sky_dir.z))
            nt = scene.world.node_tree
            env = next(n for n in nt.nodes if n.type == 'TEX_ENVIRONMENT')
            tc, mp = nt.nodes.new('ShaderNodeTexCoord'), nt.nodes.new('ShaderNodeMapping')
            mp.inputs['Rotation'].default_value = (0, 0, -rot)
            nt.links.new(tc.outputs['Generated'], mp.inputs['Vector'])
            nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
        else:
            sky_dir, peak, med, rot = Vector((0, 1, 0.2)).normalized(), 0.0, 0.0, 0.0
        folder = out / name
        folder.mkdir(exist_ok=True)
        cam, params = BUILD[kind](scene, rng, sky_dir, a.frames)
        params.update(focus(scene, rng, cam))
        params.update(setup_render(scene, rng, a, folder))
        bpy.ops.render.render(animation=True)
        frames = sorted(folder.glob('f.*.exr'))
        if len(frames) != a.frames:
            print(json.dumps({'failed': sid, 'frames': len(frames)}), flush=True)
            continue
        row = {'asset_id': f'blender_v5:{name}', 'scene_id': sid, 'kind': kind, 'folder': str(folder).replace('\\', '/'),
               'frames': a.frames, 'size': [a.width, a.height], 'space': 'rec709', 'working_space': working_space(),
               'sky_scene_id': sky['scene_id'] if sky else None, 'sky_rotation_deg': round(math.degrees(rot), 2),
               'sky_peak': round(peak, 3), 'seed': a.seed, 'index': idx, 'samples': a.samples,
               'licence': 'authored geometry (FXTD Studios), sky CC0' if sky else 'authored geometry (FXTD Studios)',
               'blender': bpy.app.version_string, 'params': params, 'seconds': round(time.time() - t0, 1)}
        (folder / 'scene.json').write_text(json.dumps(row, indent=1), encoding='utf-8')
        with open(man, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({'done': sid, 'seconds': row['seconds'], 'sky': row['sky_scene_id']}), flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
