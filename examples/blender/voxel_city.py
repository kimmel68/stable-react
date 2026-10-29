"""Voxel board city: an audio-reactive example scene for the 3D-passes source of Stable React (Blender 5.x).

A drone view over a city of voxel blocks that reads as skyscrapers from one world and as chips on a circuit board from
the next: 10 m blocks, 3 m streets with light traffic, landmark towers, a diagonal canal, two plazas and a far skyline.
Every count 1 of the example track rebuilds the city (new heights, arriving as a ring wave from the camera to the
horizon); counts 2–4 make blocks jump in whole metres, streets and windows flash on every beat, the camera pushes on
count 1. The jumps are real geometry, so the EXR vector pass carries every jump into the loop.

Timing: 30 fps (the app reads source frame i as output frame i at 30 fps); beats come from signal_decay_beats.json
(the analysis of examples/signal_decay_20s.m4a), frame F <-> t = (F - 1) / 30.

    blender -b -P examples/blender/voxel_city.py -- <out_dir> [frames]

writes <out>/beauty/, <out>/depth/ (16-bit, white = near), <out>/exr/ (Depth, Vector, Object Index), <out>/meta.json.
"""
import json
import math
import sys
from pathlib import Path

import bpy

FPS, FRAMES, RES = 30, 300, (768, 432)
NEAR, FAR = 25.0, 420.0
BEATS = Path(__file__).resolve().with_name("signal_decay_beats.json")
HOEHE, NEIGUNG, LENS = 32.0, 42.0, 24          # drone: 32 m up, 42° down
YAW = (25.0, 33.0)                             # degrees against the street grid, drifting over the clip
TEMPO = 0.133                                  # m per frame (4 m/s)


# --- beat table: one point per frame with stepped values -------------------------------------------------------------

def takt_tabelle(n):
    a = json.loads(BEATS.read_text(encoding="utf-8"))
    beats = [(round(t * FPS) + 1, (i - a["takt_phase_index"]) % 4 + 1) for i, t in enumerate(a["beat_zeiten"])]
    T = {k: [0.0] * n for k in ("bar", "c1", "c2", "c3", "c4", "puls")}

    def setze(k, a_, b_, v):
        for f in range(max(0, a_), min(n, b_)):
            T[k][f] = v
    bar = 0
    for F, count in beats:
        if F >= n:
            break
        if count == 1:
            bar += 1
            setze("bar", F, n, float(bar))                   # new city from here
            setze("c1", F, F + 6, 3.0)                        # surge 3 -> 1.5 -> 0 m
            setze("c1", F + 6, F + 12, 1.5)
        elif count == 2:
            setze("c2", F, F + 5, 1.0)                        # half of the blocks +2 m, hold 5 frames
        elif count == 3:
            setze("c3", F, F + 8, 1.0)                        # ring wave runs away from the camera
        else:
            setze("c4", F, F + 4, 1.0)                        # foreground pop
        for j, v in enumerate((3.0, 2.0, 1.0)):
            setze("puls", F + j, F + j + 1, v)                # streets and windows flash on every beat
    return T, [F for F, c in beats if c == 1 and F < n]


def takt_objekt(T):
    n = len(T["bar"])
    me = bpy.data.meshes.new("Takt")
    me.from_pydata([(0.0, 0.0, -50.0)] * n, [], [])          # points only: never rendered, read by geometry nodes
    for k, v in T.items():
        me.attributes.new(k, "FLOAT", "POINT").data.foreach_set("value", v)
    ob = bpy.data.objects.new("Takt", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# --- geometry nodes ---------------------------------------------------------------------------------------------------

class Baum:
    def __init__(self, name):
        self.ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        self.n, self.L = self.ng.nodes, self.ng.links

    def node(self, t, **kw):
        x = self.n.new(t)
        for k, v in kw.items():
            setattr(x, k, v)
        return x

    def _ein(self, sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif v is not None:
            self.L.new(v, sock)

    def m(self, op, a=None, b=None, c=None):
        x = self.node("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b, c)):
            self._ein(x.inputs[i], v)
        return x.outputs[0]

    def vec(self, x, y, z):
        c = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self._ein(c.inputs[i], v)
        return c.outputs[0]

    def hash(self, v):
        w = self.node("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        self.L.new(v, w.inputs["Vector"])
        return w.outputs["Value"]

    def und(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.m("MULTIPLY", r, x)
        return r

    def oder(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.m("MAXIMUM", r, x)
        return r

    def waehle(self, maske, ja, nein):
        return self.m("ADD", self.m("MULTIPLY", ja, maske), self.m("MULTIPLY", nein, self.m("SUBTRACT", 1.0, maske)))

    def ausgang(self, geo):
        self.ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        out = self.node("NodeGroupOutput")
        self.L.new(geo, out.inputs[0])
        return self.ng


def stadt_baum(takt, kamera, mat, name, groesse, mitte, voxel, block, strasse_m, skyline=False):
    """Grid of voxel columns (heights in metres). Per-block hash of the bar -> new city on every count 1, arriving as a
    ring wave from the camera; landmark towers stay; canal and plazas stay low; stepped jumps on counts 1–4."""
    b = Baum(name)
    zeit = b.node("GeometryNodeInputSceneTime")
    frame = zeit.outputs["Frame"]
    grid = b.node("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = groesse[0] - voxel
    grid.inputs["Size Y"].default_value = groesse[1] - voxel
    grid.inputs["Vertices X"].default_value = int(groesse[0] / voxel)
    grid.inputs["Vertices Y"].default_value = int(groesse[1] / voxel)
    tr = b.node("GeometryNodeTransform")
    b.L.new(grid.outputs["Mesh"], tr.inputs["Geometry"])
    tr.inputs["Translation"].default_value = (mitte[0], mitte[1], 0)
    pts = b.node("GeometryNodeMeshToPoints")
    b.L.new(tr.outputs["Geometry"], pts.inputs["Mesh"])
    sep = b.node("ShaderNodeSeparateXYZ")
    b.L.new(b.node("GeometryNodeInputPosition").outputs[0], sep.inputs[0])
    x, y = sep.outputs["X"], sep.outputs["Y"]
    ki = b.node("GeometryNodeObjectInfo")
    ki.inputs["Object"].default_value = kamera
    ks = b.node("ShaderNodeSeparateXYZ")
    b.L.new(ki.outputs["Location"], ks.inputs[0])
    dx, dy = b.m("SUBTRACT", x, ks.outputs["X"]), b.m("SUBTRACT", y, ks.outputs["Y"])
    dist = b.m("SQRT", b.m("ADD", b.m("MULTIPLY", dx, dx), b.m("MULTIPLY", dy, dy)))   # ground distance to the camera
    delay = b.m("MINIMUM", b.m("FLOOR", b.m("DIVIDE", dist, 18.0)), 10.0)
    ti = b.node("GeometryNodeObjectInfo")
    ti.inputs["Object"].default_value = takt

    def tab(name_, mit_delay):
        s = b.node("GeometryNodeSampleIndex", data_type="FLOAT", domain="POINT")
        b.L.new(ti.outputs["Geometry"], s.inputs["Geometry"])
        na = b.node("GeometryNodeInputNamedAttribute", data_type="FLOAT")
        na.inputs["Name"].default_value = name_
        b.L.new(na.outputs["Attribute"], s.inputs["Value"])
        b.L.new(b.m("MAXIMUM", b.m("SUBTRACT", frame, delay) if mit_delay else frame, 0.0), s.inputs["Index"])
        return s.outputs["Value"]

    bx = b.m("FLOOR", b.m("DIVIDE", x, block))
    by = b.m("FLOOR", b.m("DIVIDE", y, block))
    v = b.hash(b.vec(bx, by, tab("bar", True)))
    hoch = b.m("GREATER_THAN", v, 0.7)                                       # 25–30 % tall blocks
    if skyline:
        hb = b.m("ADD", 40.0, b.m("MULTIPLY", b.m("FLOOR", b.m("MULTIPLY", v, 16.0)), 4.0))    # 40–100 m
    else:
        hb = b.waehle(hoch, b.m("ADD", 10.0, b.m("FLOOR", b.m("MULTIPLY", b.m("SUBTRACT", v, 0.7), 30.0))),
                      b.m("ADD", 3.0, b.m("FLOOR", b.m("MULTIPLY", v, 8.6))))                  # 3–8 m / 10–18 m
    hc = b.m("FLOOR", b.m("MULTIPLY", b.hash(b.vec(x, y, 0.0)), 2.0))       # roof detail 0–1 m
    m2 = b.m("LESS_THAN", b.hash(b.vec(b.m("ADD", bx, 0.5), b.m("ADD", by, 0.5), 7.0)), 0.5)
    m3 = b.m("LESS_THAN", b.m("FLOORED_MODULO", b.m("SUBTRACT", b.m("FLOOR", b.m("DIVIDE", dist, 3.0)), frame), 8.0), 0.5)
    m4 = b.und(b.m("LESS_THAN", dist, 60.0), b.m("LESS_THAN", b.hash(b.vec(b.m("ADD", x, 7.0), b.m("ADD", y, 7.0), 3.0)), 0.3))
    h = b.m("ADD", b.m("ADD", hb, hc), tab("c1", True))
    h = b.m("ADD", h, b.m("MULTIPLY", b.und(tab("c2", False), m2), 2.0))
    h = b.m("ADD", h, b.m("MULTIPLY", b.und(tab("c3", False), m3), 2.0))
    h = b.m("ADD", h, b.und(tab("c4", False), m4))
    strasse = b.oder(b.m("LESS_THAN", b.m("FRACT", b.m("DIVIDE", x, block)), strasse_m / block),
                     b.m("LESS_THAN", b.m("FRACT", b.m("DIVIDE", y, block)), strasse_m / block))
    tief = strasse
    if not skyline:
        # landmarks: every 7th block in x, 6th in y; never jump, 45–65 m
        lm = b.und(b.m("LESS_THAN", b.m("ABSOLUTE", b.m("SUBTRACT", b.m("FLOORED_MODULO", bx, 7.0), 3.0)), 0.5),
                   b.m("LESS_THAN", b.m("ABSOLUTE", b.m("SUBTRACT", b.m("FLOORED_MODULO", by, 6.0), 2.0)), 0.5))
        h = b.waehle(lm, b.m("ADD", 45.0, b.m("FLOOR", b.m("MULTIPLY", b.hash(b.vec(bx, by, 99.0)), 20.0))), h)
        # canal 8 m wide, diagonal; two plazas 20 x 20 m
        kanal = b.m("LESS_THAN", b.m("ABSOLUTE", b.m("SUBTRACT", b.m("SUBTRACT", b.m("MULTIPLY", x, 0.8),
                                                                                     b.m("MULTIPLY", y, 0.6)), -20.0)), 4.0)
        platz = b.oder(*[b.und(b.m("LESS_THAN", b.m("ABSOLUTE", b.m("SUBTRACT", x, px)), 10.0),
                               b.m("LESS_THAN", b.m("ABSOLUTE", b.m("SUBTRACT", y, py)), 10.0)) for px, py in ((25, 45), (-35, 110))])
        tief = b.oder(strasse, platz)
        h = b.waehle(tief, 1.0, h)
        h = b.waehle(kanal, 0.3, h)
    else:
        h = b.waehle(strasse, 1.0, h)
    wuerfel = b.node("GeometryNodeMeshCube")
    wuerfel.inputs["Size"].default_value = (voxel * 0.94, voxel * 0.94, 1.0)
    hochs = b.node("GeometryNodeTransform")
    b.L.new(wuerfel.outputs["Mesh"], hochs.inputs["Geometry"])
    hochs.inputs["Translation"].default_value = (0, 0, 0.5)
    sm = b.node("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    b.L.new(hochs.outputs["Geometry"], sm.inputs["Geometry"])
    inst = b.node("GeometryNodeInstanceOnPoints")
    b.L.new(pts.outputs["Points"], inst.inputs["Points"])
    b.L.new(sm.outputs["Geometry"], inst.inputs["Instance"])
    b.L.new(b.vec(1.0, 1.0, h), inst.inputs["Scale"])
    geo = inst.outputs["Instances"]
    for an, wert in (("strasse", tief), ("zelle", b.hash(b.vec(x, y, 11.0))), ("puls", tab("puls", False))):
        st = b.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="INSTANCE")
        st.inputs["Name"].default_value = an
        b.L.new(geo, st.inputs["Geometry"])
        b.L.new(wert, st.inputs["Value"])
        geo = st.outputs["Geometry"]
    real = b.node("GeometryNodeRealizeInstances")                       # constant topology -> deformation vectors
    b.L.new(geo, real.inputs["Geometry"])
    return b.ausgang(real.outputs[0])


def verkehr_baum(mat):
    """Light cubes flowing along the streets (north-south and east-west), seen from above as lines of light."""
    b = Baum("traffic")
    frame = b.node("GeometryNodeInputSceneTime").outputs["Frame"]
    join = b.node("GeometryNodeJoinGeometry")
    lanes = [((x0 + 1.5, -100.0, 1.2), (0, 3.0, 0), 1.0 if i % 2 else -1.0, "Y") for i, x0 in enumerate(range(-60, 70, 20))]
    lanes += [((-100.0, y0 + 1.5, 1.2), (3.0, 0, 0), 1.0 if i % 2 else -1.0, "X") for i, y0 in enumerate(range(-40, 180, 30))]
    for start, off, richtung, achse in lanes:
        line = b.node("GeometryNodeMeshLine")
        line.inputs["Count"].default_value = 90
        line.inputs["Start Location"].default_value = start
        line.inputs["Offset"].default_value = off
        pts = b.node("GeometryNodeMeshToPoints")
        b.L.new(line.outputs[0], pts.inputs["Mesh"])
        sp = b.node("GeometryNodeSetPosition")
        b.L.new(pts.outputs[0], sp.inputs["Geometry"])
        d = b.m("MULTIPLY", b.m("FLOORED_MODULO", b.m("MULTIPLY", frame, 0.6), 3.0), richtung)
        b.L.new(b.vec(d, 0.0, 0.0) if achse == "X" else b.vec(0.0, d, 0.0), sp.inputs["Offset"])
        b.L.new(sp.outputs[0], join.inputs[0])
    cube = b.node("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.9, 0.9, 0.6)
    sm = b.node("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    b.L.new(cube.outputs[0], sm.inputs["Geometry"])
    inst = b.node("GeometryNodeInstanceOnPoints")
    b.L.new(join.outputs[0], inst.inputs["Points"])
    b.L.new(sm.outputs[0], inst.inputs["Instance"])
    real = b.node("GeometryNodeRealizeInstances")
    b.L.new(inst.outputs[0], real.inputs[0])
    return b.ausgang(real.outputs[0])


# --- materials --------------------------------------------------------------------------------------------------------

def material_stadt(eins):
    """Mid-grey, contrasty base (each world brings its own palette in the app); white emission only: window bands on
    the tower sides and glowing streets, both flash on every beat; overall light level steps per world."""
    m = bpy.data.materials.new("board_city")
    nt = m.node_tree
    bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")

    def attr(name):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_type, a.attribute_name = "GEOMETRY", name
        return a.outputs["Fac"]

    def mm(op, a, b_):
        x = nt.nodes.new("ShaderNodeMath")
        x.operation = op
        for i, v in enumerate((a, b_)):
            if isinstance(v, float):
                x.inputs[i].default_value = v
            else:
                nt.links.new(v, x.inputs[i])
        return x.outputs[0]

    strasse, zelle, puls = attr("strasse"), attr("zelle"), attr("puls")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sp.inputs[0])
    sn = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sn.inputs[0])
    band = nt.nodes.new("ShaderNodeMath")
    band.operation = "FRACT"
    nt.links.new(mm("DIVIDE", sp.outputs["Z"], 1.5), band.inputs[0])          # one window band per 1.5 m storey
    fenster = mm("MULTIPLY", mm("MULTIPLY", mm("GREATER_THAN", band.outputs[0], 0.35), mm("LESS_THAN", band.outputs[0], 0.7)),
                 mm("MULTIPLY", mm("LESS_THAN", zelle, 0.55), mm("LESS_THAN", mm("ABSOLUTE", sn.outputs["Z"], 0.0), 0.5)))
    fenster = mm("MULTIPLY", fenster, mm("SUBTRACT", 1.0, strasse))
    leuchten = mm("ADD", fenster, mm("MULTIPLY", strasse, 0.6))
    welt = nt.nodes.new("ShaderNodeValue")
    welt.name = "welt_licht"
    for F, wert in zip([1] + eins, (2.0, 4.0, 2.5, 3.5, 4.0)):
        welt.outputs[0].default_value = wert
        welt.outputs[0].keyframe_insert("default_value", frame=F)
    staerke = mm("MULTIPLY", mm("MULTIPLY", leuchten, welt.outputs[0]), mm("ADD", 1.0, puls))
    farbe = nt.nodes.new("ShaderNodeMix")
    farbe.data_type = "RGBA"
    nt.links.new(strasse, farbe.inputs["Factor"])
    farbe.inputs["A"].default_value = (0.30, 0.30, 0.31, 1)
    farbe.inputs["B"].default_value = (0.15, 0.15, 0.16, 1)
    nt.links.new(farbe.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    nt.links.new(staerke, bsdf.inputs["Emission Strength"])
    return m


def material_licht():
    m = bpy.data.materials.new("traffic_light")
    bsdf = next(nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Emission Strength"].default_value = 8.0
    return m


def material_wasser():
    m = bpy.data.materials.new("canal_water")
    bsdf = next(nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.03, 0.035, 0.04, 1)
    bsdf.inputs["Roughness"].default_value = 0.05
    return m


# --- scene ------------------------------------------------------------------------------------------------------------

def _fcurves(obj_or_id):
    ad = obj_or_id.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


def _interpolation(id_, art):
    for fc in _fcurves(id_):
        for kp in fc.keyframe_points:
            kp.interpolation = art


def leeren():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.cameras, bpy.data.lights,
                 bpy.data.worlds):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def szene(frames=FRAMES):
    leeren()
    s = bpy.context.scene
    s.render.fps, s.frame_start, s.frame_end = FPS, 1, frames
    s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = *RES, 100
    T, eins = takt_tabelle(frames + 16)
    takt = takt_objekt(T)
    # drone camera: diagonal to the grid, slow forward drift, yaw turns 25° -> 33°, +0.8 m push on every count 1
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    s.collection.objects.link(cam)
    s.camera = cam
    cam.data.lens, cam.data.clip_start, cam.data.clip_end = LENS, 0.5, 900
    px, py = -40.0, -70.0
    for F in range(1, frames + 1):
        yaw = math.radians(YAW[0] + (YAW[1] - YAW[0]) * (F - 1) / max(1, frames - 1))
        schritt = TEMPO + (0.8 / 3 if any(0 <= F - e <= 2 for e in eins) else 0.0)
        px, py = px + math.sin(yaw) * schritt, py + math.cos(yaw) * schritt
        cam.location = (px, py, HOEHE)
        cam.rotation_euler = (math.radians(90 - NEIGUNG), 0, -yaw)
        cam.keyframe_insert("location", frame=F)
        cam.keyframe_insert("rotation_euler", frame=F)
    _interpolation(cam, "LINEAR")
    mat = material_stadt(eins)
    _interpolation(mat.node_tree, "CONSTANT")
    stadt = bpy.data.objects.new("BoardCity", bpy.data.meshes.new("city"))
    s.collection.objects.link(stadt)
    stadt.modifiers.new("city", "NODES").node_group = stadt_baum(takt, cam, mat, "board_city", (200, 240), (0.0, 40.0),
                                                                 1.0, 10.0, 3.0)
    stadt.pass_index = 1
    sky = bpy.data.objects.new("Skyline", bpy.data.meshes.new("skyline"))
    s.collection.objects.link(sky)
    sky.modifiers.new("skyline", "NODES").node_group = stadt_baum(takt, cam, mat, "skyline", (320, 60), (40.0, 190.0),
                                                                  2.0, 14.0, 4.0, skyline=True)
    sky.pass_index = 2
    for F, xx in ((1, 0.0), (frames, -20.0)):                          # slow parallax drift of the skyline
        sky.location = (xx, 0, 0)
        sky.keyframe_insert("location", frame=F)
    _interpolation(sky, "LINEAR")
    verkehr = bpy.data.objects.new("Traffic", bpy.data.meshes.new("traffic"))
    s.collection.objects.link(verkehr)
    verkehr.modifiers.new("traffic", "NODES").node_group = verkehr_baum(material_licht())
    verkehr.pass_index = 3
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 40, 0.6))
    wasser = bpy.context.active_object
    wasser.name, wasser.scale, wasser.pass_index = "Canal", (200, 240, 1), 4
    wasser.data.materials.append(material_wasser())
    sonne = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    s.collection.objects.link(sonne)
    sonne.data.energy, sonne.data.angle = 2.2, math.radians(2)
    sonne.rotation_euler = (math.radians(55), 0, math.radians(-60))
    welt = bpy.data.worlds.new("Sky")
    s.world = welt
    bg = next(nd for nd in welt.node_tree.nodes if nd.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.25, 0.25, 0.27, 1)
    bg.inputs["Strength"].default_value = 1.0
    return s


def ausgabe(s, out: Path):
    s.render.engine = "CYCLES"
    s.cycles.samples, s.cycles.use_denoising = 24, True
    try:
        s.cycles.device = "GPU"
    except TypeError:
        pass
    s.render.use_motion_blur = False                       # Cycles writes the Vector pass only without motion blur
    s.render.image_settings.file_format = "PNG"
    s.render.filepath = str(out / "beauty" / "frame_")
    vl = s.view_layers[0]
    vl.use_pass_z = vl.use_pass_vector = vl.use_pass_object_index = True
    ng = bpy.data.node_groups.new("passes", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    s.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    go = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl.outputs["Image"], go.inputs[0])
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory, fo.file_name = str(out / "exr"), "frame_"
    fo.format.file_format = "OPEN_EXR_MULTILAYER"
    fo.format.color_depth = "32"
    fo.file_output_items.clear()
    for name, typ in (("Depth", "FLOAT"), ("Vector", "VECTOR"), ("Object Index", "FLOAT")):
        it = fo.file_output_items.new(typ, name)
        if typ == "VECTOR":
            it.vector_socket_dimensions = 4
        ng.links.new(rl.outputs[name], fo.inputs[name])

    def rechne(op, a, b):
        x = ng.nodes.new("ShaderNodeMath")
        x.operation, x.use_clamp = op, op == "MULTIPLY"
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                x.inputs[i].default_value = v
            else:
                ng.links.new(v, x.inputs[i])
        return x.outputs[0]
    disp = rechne("MULTIPLY", rechne("SUBTRACT", rechne("DIVIDE", 1.0, rl.outputs["Depth"]), 1.0 / FAR),
                  1.0 / (1.0 / NEAR - 1.0 / FAR))
    fd = ng.nodes.new("CompositorNodeOutputFile")
    fd.directory, fd.file_name = str(out / "depth"), "frame_"
    fd.format.media_type = "IMAGE"
    fd.format.file_format = "PNG"
    fd.format.color_mode = "BW"
    fd.format.color_depth = "16"
    fd.save_as_render = False
    fd.file_output_items.clear()
    fd.file_output_items.new("FLOAT", "depth")
    ng.links.new(disp, fd.inputs["depth"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "fps": FPS, "resolution": list(RES), "frames": [s.frame_start, s.frame_end],
        "normalisation": {"depth_near_m": NEAR, "depth_far_m": FAR},
        "source": "examples/blender/voxel_city.py", "track": "examples/signal_decay_20s.m4a"}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    frames = int(argv[1]) if len(argv) > 1 else FRAMES
    scene = szene(frames)
    if argv:
        ausgabe(scene, Path(argv[0]).resolve())
        bpy.ops.render.render(animation=True)
