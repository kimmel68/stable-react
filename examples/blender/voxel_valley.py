"""Voxel valley: a small animated example scene for the 3D-passes source of Stable React (Blender 5.x).

A valley of voxel columns whose height follows a slowly moving 4D noise (smooth, no popping camera), a river, and a
camera gliding down the valley. Renders exactly the folder layout the apps read:

    <out>/beauty/frame_0001.png ...   rendered frames
    <out>/depth/frame_0001.png ...    16-bit depth, white = near (fixed range from meta.json)
    <out>/exr/frame_0001.exr ...      multilayer EXR with Depth, Vector and Object Index (exact motion for the loop)
    <out>/meta.json                   fps, resolution and the fixed depth range

Run inside Blender (Scripting tab) to build the scene, or render it headless:

    blender -b -P examples/blender/voxel_valley.py -- <out_dir> [frames]
"""
import json
import math
import sys
from pathlib import Path

import bpy

FPS, FRAMES, RES = 24, 192, (768, 432)
VOXEL = 0.5                     # edge length of one voxel in metres
NEAR, FAR = 5.0, 90.0           # fixed depth range for meta.json (nearest ground ~5.5 m from the camera)


def leeren():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.cameras, bpy.data.lights,
                 bpy.data.worlds):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def material_gelaende():
    m = bpy.data.materials.new("voxel_terrain")
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "INSTANCER"
    attr.attribute_name = "hoehe01"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.80, 0.70, 0.45, 1)          # sand at the river
    els[1].position, els[1].color = 1.0, (0.95, 0.96, 1.0, 1)           # snow on top
    for pos, col in ((0.12, (0.22, 0.45, 0.12, 1)), (0.45, (0.16, 0.32, 0.10, 1)), (0.68, (0.42, 0.38, 0.34, 1)),
                     (0.85, (0.55, 0.53, 0.52, 1))):
        e = els.new(pos)
        e.color = col
    ramp.color_ramp.interpolation = "CONSTANT"                           # hard bands = voxel look
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    return m


def material_wasser():
    m = bpy.data.materials.new("voxel_water")
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.25, 0.35, 1)
    bsdf.inputs["Roughness"].default_value = 0.08
    return m


def geonodes(mat):
    """Grid -> points -> height = valley profile + 4D noise (w = time) -> quantised to voxels -> one column each."""
    ng = bpy.data.node_groups.new("voxel_valley", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, L = ng.nodes, ng.links

    def node(t, **kw):
        x = n.new(t)
        for k, v in kw.items():
            setattr(x, k, v)
        return x

    def math_(op, a=None, b=None):
        x = node("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                x.inputs[i].default_value = v
            else:
                L.new(v, x.inputs[i])
        return x.outputs[0]

    grid = node("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = 60.0
    grid.inputs["Size Y"].default_value = 140.0
    grid.inputs["Vertices X"].default_value = int(60 / VOXEL)
    grid.inputs["Vertices Y"].default_value = int(140 / VOXEL)
    pts = node("GeometryNodeMeshToPoints")
    L.new(grid.outputs["Mesh"], pts.inputs["Mesh"])
    pos = node("GeometryNodeInputPosition")
    sep = node("ShaderNodeSeparateXYZ")
    L.new(pos.outputs[0], sep.inputs[0])
    # valley profile: 7 m * (|x| / 18)^1.7, slight bend of the river along y
    biege = math_("MULTIPLY", math_("SINE", math_("MULTIPLY", sep.outputs["Y"], 0.045)), 4.0)
    xr = math_("ABSOLUTE", math_("SUBTRACT", sep.outputs["X"], biege))
    profil = math_("MULTIPLY", math_("POWER", math_("DIVIDE", xr, 18.0), 1.7), 7.0)
    # animated 4D noise: w follows scene time slowly -> smooth, flowing change
    zeit = node("GeometryNodeInputSceneTime")
    noise = node("ShaderNodeTexNoise", noise_dimensions="4D")
    noise.inputs["Scale"].default_value = 0.06
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.45
    L.new(pos.outputs[0], noise.inputs["Vector"])
    L.new(math_("MULTIPLY", zeit.outputs["Seconds"], 0.12), noise.inputs["W"])
    rel = math_("MULTIPLY", math_("SUBTRACT", noise.outputs["Fac"], 0.5), 7.0)
    hang = math_("MULTIPLY", math_("MINIMUM", math_("DIVIDE", xr, 6.0), 1.0), rel)   # flat river bed, noisy slopes
    h = math_("MAXIMUM", math_("ADD", math_("ADD", profil, hang), 0.6), VOXEL)
    hq = math_("MULTIPLY", math_("CEIL", math_("DIVIDE", h, VOXEL)), VOXEL)          # quantised to voxels
    # instance: one column per point, origin at the bottom
    wuerfel = node("GeometryNodeMeshCube")
    wuerfel.inputs["Size"].default_value = (VOXEL * 0.96, VOXEL * 0.96, 1.0)
    rueck = node("GeometryNodeTransform")
    L.new(wuerfel.outputs["Mesh"], rueck.inputs["Geometry"])
    rueck.inputs["Translation"].default_value = (0, 0, 0.5)
    setmat = node("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    L.new(rueck.outputs["Geometry"], setmat.inputs["Geometry"])
    inst = node("GeometryNodeInstanceOnPoints")
    L.new(pts.outputs["Points"], inst.inputs["Points"])
    L.new(setmat.outputs["Geometry"], inst.inputs["Instance"])
    skal = node("ShaderNodeCombineXYZ")
    skal.inputs["X"].default_value = 1.0
    skal.inputs["Y"].default_value = 1.0
    L.new(hq, skal.inputs["Z"])
    L.new(skal.outputs[0], inst.inputs["Scale"])
    store = node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="INSTANCE")
    store.inputs["Name"].default_value = "hoehe01"
    L.new(inst.outputs["Instances"], store.inputs["Geometry"])
    L.new(math_("MINIMUM", math_("DIVIDE", hq, 14.0), 1.0), store.inputs["Value"])
    out = node("NodeGroupOutput")
    L.new(store.outputs["Geometry"], out.inputs[0])
    return ng


def szene():
    leeren()
    s = bpy.context.scene
    s.render.fps, s.frame_start, s.frame_end = FPS, 1, FRAMES
    s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = *RES, 100
    # terrain
    mesh = bpy.data.meshes.new("valley")
    tal = bpy.data.objects.new("VoxelValley", mesh)
    s.collection.objects.link(tal)
    mod = tal.modifiers.new("voxels", "NODES")
    mod.node_group = geonodes(material_gelaende())
    tal.pass_index = 1
    # river
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 1.3))     # river bed is 1.0 m (two voxels)
    wasser = bpy.context.active_object
    wasser.name, wasser.scale, wasser.pass_index = "River", (60, 140, 1), 2
    wasser.data.materials.append(material_wasser())
    # light and sky
    sonne = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    s.collection.objects.link(sonne)
    sonne.data.energy, sonne.data.angle = 3.5, math.radians(3)
    sonne.data.color = (1.0, 0.93, 0.82)
    sonne.rotation_euler = (math.radians(52), 0, math.radians(-35))
    welt = bpy.data.worlds.new("Sky")
    s.world = welt
    bg = next(nd for nd in welt.node_tree.nodes if nd.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.55, 0.70, 0.90, 1)
    bg.inputs["Strength"].default_value = 0.9
    # camera: glides down the valley, gentle sway, linear = constant speed
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    s.collection.objects.link(cam)
    s.camera = cam
    cam.data.lens, cam.data.clip_start, cam.data.clip_end = 24, 0.1, 200
    for f, y, x, z in ((1, -60, -1.5, 5.0), (FRAMES, -20, 1.5, 5.6)):
        cam.location = (x, y, z)
        cam.rotation_euler = (math.radians(80), 0, math.radians(-2 if f == 1 else 2))
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    for fc in _fcurves(cam):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return s


def _fcurves(obj):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []                                  # Blender 5: layered actions
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


def ausgabe(s, out: Path):
    s.render.engine = "CYCLES"
    s.cycles.samples, s.cycles.use_denoising = 32, True
    try:
        s.cycles.device = "GPU"
    except TypeError:
        pass
    s.render.use_motion_blur = False                       # Cycles writes the Vector pass only without motion blur
    s.render.image_settings.file_format = "PNG"
    s.render.filepath = str(out / "beauty" / "frame_")
    s.render.use_file_extension = True
    vl = s.view_layers[0]
    vl.use_pass_z = vl.use_pass_vector = vl.use_pass_object_index = True
    ng = bpy.data.node_groups.new("passes", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    s.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    go = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl.outputs["Image"], go.inputs[0])
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(out / "exr")
    fo.file_name = "frame_"
    fo.format.file_format = "OPEN_EXR_MULTILAYER"
    fo.format.color_depth = "32"
    fo.file_output_items.clear()
    for name, typ in (("Depth", "FLOAT"), ("Vector", "VECTOR"), ("Object Index", "FLOAT")):
        it = fo.file_output_items.new(typ, name)
        if typ == "VECTOR":
            it.vector_socket_dimensions = 4
        ng.links.new(rl.outputs[name], fo.inputs[name])
    # depth/: 16-bit PNG, white = near, fixed disparity range (1/z between 1/FAR and 1/NEAR)
    def rechne(op, a, b):
        m = ng.nodes.new("ShaderNodeMath")
        m.operation, m.use_clamp = op, op == "MULTIPLY"
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                m.inputs[i].default_value = v
            else:
                ng.links.new(v, m.inputs[i])
        return m.outputs[0]
    disp = rechne("MULTIPLY", rechne("SUBTRACT", rechne("DIVIDE", 1.0, rl.outputs["Depth"]), 1.0 / FAR),
                  1.0 / (1.0 / NEAR - 1.0 / FAR))
    fd = ng.nodes.new("CompositorNodeOutputFile")
    fd.directory = str(out / "depth")
    fd.file_name = "frame_"
    fd.format.media_type = "IMAGE"
    fd.format.file_format = "PNG"
    fd.format.color_mode = "BW"
    fd.format.color_depth = "16"
    fd.save_as_render = False                               # data, not a picture: no view transform
    fd.file_output_items.clear()
    fd.file_output_items.new("FLOAT", "depth")
    ng.links.new(disp, fd.inputs["depth"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "fps": FPS, "resolution": list(RES), "frames": [s.frame_start, s.frame_end],
        "normalisation": {"depth_near_m": NEAR, "depth_far_m": FAR},
        "source": "examples/blender/voxel_valley.py"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    scene = szene()
    if argv:
        if len(argv) > 1:
            scene.frame_end = int(argv[1])
        ausgabe(scene, Path(argv[0]).resolve())
        bpy.ops.render.render(animation=True)
