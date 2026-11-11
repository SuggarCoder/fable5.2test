"""Rebuild concept.png in Blender: a terraced field of frosted acrylic cards.

    blender -b -P blender/build_scene.py -- [--preview] [--no-export] [--samples N]

Reads   public/assets/params.json   (shared with the three.js app)
Writes  public/assets/card.glb      one card, 3 materials, for instancing
        blender/scene.blend         the full scene
        blender/render.png          Cycles still to compare against concept.png
"""
import bpy, bmesh, json, math, os, sys
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW = "--preview" in argv
EXPORT = "--no-export" not in argv
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else (48 if PREVIEW else 256)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "public", "assets", "params.json")) as f:
    P = json.load(f)

C = P["card"]
L, H, T = C["length"], C["height"], C["thickness"]
T_FRONT, T_CORE, END_STEP = C["frontSlab"], C["core"], C["endStep"]
T_BACK = T - T_FRONT - T_CORE
PITCH, ROW_PITCH = P["pitchCard"], P["pitchRow"]
ROWS, COLS = P["rows"], P["cols"]
F = P["field"]


# ------------------------------------------------------------------ height field
def ripple(i, j):
    a = math.sin(j * 0.21 + i * 1.3) * 0.5 + 0.5
    b = math.sin(j * 0.057 - i * 0.6 + 1.0) * 0.5 + 0.5
    return F["ripple"] * (0.6 * a + 0.8 * b)


def ridge(d):
    """Profile of the focused row: a tent with the pulled-out card on top."""
    if d < 0:
        return max(0.0, F["front0"] - F["frontSlope"] * (-d - 1))
    if d > 0:
        return max(0.0, F["back0"] - F["backSlope"] * (d - 1))
    return F["front0"]


def offset_of(i, j, fi=0, fj=0):
    """Vertical offset of card (row i, index j) when card (fi, fj) is the focus."""
    r, d = i - fi, j - fj
    if r == 0:
        h = F["peak"] if d == 0 else ridge(d)
    elif r == -1:
        h = max(0.0, F["prevTop"] - (F["prevFrontSlope"] if d < 0 else F["prevBackSlope"]) * abs(d))
        h = min(h, ridge(d))
    else:
        h = F["rowScale"].get(str(r), 0.0) * ridge(d)
    return h + ripple(i, j)


def glow_of(i, j, fi=0, fj=0):
    r, d = i - fi, j - fj
    if r == 0 and d == 0:
        return 1.0
    if r == -1 and d == 0:
        return 0.75
    return 0.0


# ------------------------------------------------------------------ scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree, m.node_tree.nodes["Principled BSDF"]


# frosted acrylic: milky subsurface body with a faint cloudy variation
acrylic, nt, b = new_mat("Acrylic")
b.inputs["Base Color"].default_value = (1.0, 0.93, 0.84, 1)
b.inputs["Roughness"].default_value = 0.5
b.inputs["IOR"].default_value = 1.49
b.inputs["Subsurface Weight"].default_value = 1.0
b.inputs["Subsurface Radius"].default_value = (1.0, 1.0, 1.0)
b.inputs["Subsurface Scale"].default_value = 0.08
b.inputs["Transmission Weight"].default_value = 0.25
b.inputs["Coat Weight"].default_value = 0.1
b.inputs["Coat Roughness"].default_value = 0.35
tc = nt.nodes.new("ShaderNodeTexCoord")
oi = nt.nodes.new("ShaderNodeObjectInfo")
add = nt.nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"
noise = nt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 1.6
noise.inputs["Detail"].default_value = 3.0
ramp = nt.nodes.new("ShaderNodeMapRange")
ramp.inputs["From Min"].default_value = 0.3
ramp.inputs["From Max"].default_value = 0.7
mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
mixc.inputs["A"].default_value = (0.84, 0.63, 0.43, 1)
mixc.inputs["B"].default_value = (0.88, 0.68, 0.49, 1)
nt.links.new(tc.outputs["Object"], add.inputs[0])
nt.links.new(oi.outputs["Random"], add.inputs[1])
stretch = nt.nodes.new("ShaderNodeVectorMath"); stretch.operation = "MULTIPLY"
nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
nt.links.new(ramp.outputs["Result"], mixc.inputs["Factor"])

def math_node(nt, op, a=None, b_=None, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for k, v in enumerate((a, b_, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[k].default_value = v
        else:
            nt.links.new(v, n.inputs[k])
    return n.outputs[0]


# inner spacer frame seen through the frosted face (card.png): a band along the top and the block end
xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], xyz.inputs[0])
frame = math_node(nt, "MAXIMUM",
                  math_node(nt, "GREATER_THAN", xyz.outputs["Z"], C["height"] - 0.2),
                  math_node(nt, "LESS_THAN", xyz.outputs["X"], -C["length"] / 2 + 0.26))
mixf = nt.nodes.new("ShaderNodeMix"); mixf.data_type = "RGBA"
mixf.inputs["B"].default_value = (0.90, 0.72, 0.54, 1)
nt.links.new(math_node(nt, "MULTIPLY", frame, 0.55), mixf.inputs["Factor"])
nt.links.new(mixc.outputs["Result"], mixf.inputs["A"])
nt.links.new(mixf.outputs["Result"], b.inputs["Base Color"])
nt.links.new(math_node(nt, "MULTIPLY_ADD", frame, -0.06, 0.26), b.inputs["Subsurface Scale"])

# light-guide core between the slabs. Object colour (R) switches its edge light on; the LED sits at
# the block end, so the emission falls off along the card: glow * (base + k * exp(-(x + L/2) / d))
core, nt, b = new_mat("Core")
b.inputs["Base Color"].default_value = (0.9, 0.74, 0.56, 1)
b.inputs["Roughness"].default_value = 0.5
b.inputs["Emission Color"].default_value = (1.0, 0.74, 0.48, 1)
b.inputs["Emission Strength"].default_value = 1.0
oi = nt.nodes.new("ShaderNodeObjectInfo")
sep = nt.nodes.new("ShaderNodeSeparateColor")
nt.links.new(oi.outputs["Color"], sep.inputs["Color"])
tc = nt.nodes.new("ShaderNodeTexCoord")
xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], xyz.inputs[0])
fall = math_node(nt, "EXPONENT", math_node(nt, "MULTIPLY_ADD", xyz.outputs["X"], -1 / 0.35, -C["length"] / 2 / 0.35))
nt.links.new(math_node(nt, "MULTIPLY_ADD", sep.outputs["Red"], math_node(nt, "MULTIPLY_ADD", fall, 2.4, 0.14), 0.05),
             b.inputs["Emission Strength"])

# pale satin block (frosted white in concept.png, champagne in card.png) and a polished eyelet rivet
metal, nt, b = new_mat("Metal")
b.inputs["Base Color"].default_value = (0.94, 0.89, 0.83, 1)
b.inputs["Metallic"].default_value = 0.0
b.inputs["Roughness"].default_value = 0.45

rivet, nt, b = new_mat("Rivet")
b.inputs["Base Color"].default_value = (0.95, 0.93, 0.90, 1)
b.inputs["Metallic"].default_value = 1.0
b.inputs["Roughness"].default_value = 0.14

MAT_ACRYLIC, MAT_CORE, MAT_METAL, MAT_RIVET = 0, 1, 2, 3


# ------------------------------------------------------------------ card mesh
def add_box(bm, x0, x1, y0, y1, z0, z1, mat, bevel=0.0, segments=2):
    verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    for v in verts:
        v.co = Vector((x0 + (v.co.x + 0.5) * (x1 - x0),
                       y0 + (v.co.y + 0.5) * (y1 - y0),
                       z0 + (v.co.z + 0.5) * (z1 - z0)))
    faces = {f for v in verts for f in v.link_faces}
    if bevel > 0:
        edges = list({e for v in verts for e in v.link_edges})
        res = bmesh.ops.bevel(bm, geom=edges, offset=bevel, offset_type="OFFSET",
                              segments=segments, profile=0.5, affect="EDGES")
        faces = set(res["faces"]) | {f for v in res["verts"] for f in v.link_faces}
    for f in faces:
        f.material_index = mat
        f.smooth = True


def add_lathe_y(bm, cx, cz, y0, profile, mat, segments=24):
    """Revolve (radius, protrusion) pairs around an axis parallel to Y; protrusion grows towards -Y."""
    rings = []
    for radius, out in profile:
        if radius <= 1e-6:
            rings.append([bm.verts.new((cx, y0 - out, cz))])
        else:
            rings.append([bm.verts.new((cx + radius * math.cos(2 * math.pi * k / segments), y0 - out,
                                        cz + radius * math.sin(2 * math.pi * k / segments)))
                          for k in range(segments)])
    for a, b_ in zip(rings, rings[1:]):
        for k in range(segments):
            k2 = (k + 1) % segments
            if len(a) == 1:
                f = bm.faces.new((a[0], b_[k2], b_[k]))
            elif len(b_) == 1:
                f = bm.faces.new((a[k], a[k2], b_[0]))
            else:
                f = bm.faces.new((a[k], a[k2], b_[k2], b_[k]))
            f.material_index = mat
            f.smooth = True


bm = bmesh.new()
hl, ht = L / 2, T / 2
y_front0, y_front1 = -ht, -ht + T_FRONT          # front slab (carries block + rivet)
y_core0, y_core1 = y_front1, y_front1 + T_CORE    # light-guide core
y_back0, y_back1 = y_core1, ht                    # back slab
BEV = 0.003
# back slab runs the full length; the front slab stops END_STEP short at the block end (card.png)
add_box(bm, -hl, hl, y_back0, y_back1, 0.0, H, MAT_ACRYLIC, BEV)
add_box(bm, -hl + END_STEP, hl, y_front0, y_front1, 0.0, H, MAT_ACRYLIC, BEV)
# core: flush with the back slab at the block end, open slot along the bottom
add_box(bm, -hl + 0.003, hl - 0.003, y_core0 + 0.0015, y_core1 - 0.0015, 0.06, H - 0.002, MAT_CORE)
# metal block inlaid flush into the top corner: through the front slab and the core, level with the
# face and the top edge (1 mm proud only so its faces never z-fight with the slab)
bx0 = -hl + 0.11
add_box(bm, bx0, bx0 + 0.12, y_front0 - 0.001, y_core1 + 0.0005, H - 0.15, H + 0.001, MAT_METAL, 0.0015, 1)
# eyelet rivet on the front face, far top corner
add_lathe_y(bm, hl - 0.065, H - 0.09, y_front0, [
    (0.000, -0.004), (0.011, -0.004), (0.016, 0.009), (0.022, 0.014),
    (0.029, 0.014), (0.034, 0.009), (0.036, -0.001)], MAT_RIVET)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

mesh = bpy.data.meshes.new("CardBase")
bm.to_mesh(mesh)
bm.free()
for m in (acrylic, core, metal, rivet):
    mesh.materials.append(m)

# weighted normals keep the big faces flat while the bevels stay rounded
tmp = bpy.data.objects.new("CardTmp", mesh)
scene.collection.objects.link(tmp)
wn = tmp.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
wn.keep_sharp = True
wn.weight = 100
dg = bpy.context.evaluated_depsgraph_get()
card_mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
card_mesh.name = "Card"
bpy.data.objects.remove(tmp)
bpy.data.meshes.remove(mesh)
print("card tris:", sum(len(p.vertices) - 2 for p in card_mesh.polygons))

# ------------------------------------------------------------------ export one card
if EXPORT:
    card = bpy.data.objects.new("Card", card_mesh)
    scene.collection.objects.link(card)
    bpy.ops.object.select_all(action="DESELECT")
    card.select_set(True)
    bpy.context.view_layer.objects.active = card
    bpy.ops.export_scene.gltf(filepath=os.path.join(ROOT, "public", "assets", "card.glb"),
                              export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=True)
    bpy.data.objects.remove(card)

# ------------------------------------------------------------------ the field
field = bpy.data.collections.new("Cards")
scene.collection.children.link(field)
for i in range(ROWS[0], ROWS[1] + 1):
    for j in range(COLS[0], COLS[1] + 1):
        o = bpy.data.objects.new(f"Card_{i}_{j}", card_mesh)
        o.location = (i * ROW_PITCH, j * PITCH, offset_of(i, j))
        g = glow_of(i, j)
        o.color = (g, g, g, 1.0)
        field.objects.link(o)

# warm glowing floor far below, so any sliver between cards reads as light, not as a black void
fm = bpy.data.meshes.new("Floor")
fb = bmesh.new()
bmesh.ops.create_grid(fb, x_segments=1, y_segments=1, size=80.0)
fb.to_mesh(fm); fb.free()
floor_mat, nt, b = new_mat("FloorGlow")
b.inputs["Base Color"].default_value = (1.0, 0.85, 0.66, 1)
b.inputs["Emission Color"].default_value = (1.0, 0.60, 0.30, 1)
b.inputs["Emission Strength"].default_value = 0.9
fm.materials.append(floor_mat)
floor = bpy.data.objects.new("Floor", fm)
floor.location = (0, 0, -0.05)
scene.collection.objects.link(floor)

# ------------------------------------------------------------------ camera
K = P["camera"]
aim = Vector(K["aim"])
az, el = math.radians(K["azimuthDeg"]), math.radians(K["elevationDeg"])
view_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)))
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = K["lensMm"]
cam_data.sensor_width = K["sensorMm"]
cam_data.clip_end = 500
cam = bpy.data.objects.new("Camera", cam_data)
cam.location = aim - view_dir * K["distance"]
cam.rotation_euler = view_dir.to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(cam)
scene.camera = cam
fi_, fj_ = K["focusCard"]
focus_pt = Vector((fi_ * ROW_PITCH - hl, fj_ * PITCH, offset_of(fi_, fj_) + H))
cam_data.dof.use_dof = True
cam_data.dof.focus_distance = (focus_pt - cam.location).dot(view_dir)
cam_data.dof.aperture_fstop = K["fstop"]

# ------------------------------------------------------------------ light
world = bpy.data.worlds.new("World")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (1.0, 0.96, 0.92, 1)
bg.inputs["Strength"].default_value = 0.5
scene.world = world


def area_light(name, loc, energy, size, color):
    d = bpy.data.lights.new(name, "AREA")
    d.shape, d.size, d.energy, d.color = "DISK", size, energy, color
    o = bpy.data.objects.new(name, d)
    o.location = loc
    o.rotation_euler = (Vector((0, 0, 1.5)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(o)
    return o


# key from the back-left: ends and tops catch it, cards standing clear of the stack are lit through from behind, gaps stay in warm shade
area_light("Key", (-13.0, 9.0, 13.0), 5600.0, 7.0, (1.0, 0.965, 0.92))

# ------------------------------------------------------------------ render
scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 24
scene.render.resolution_x, scene.render.resolution_y = 1672, 941
scene.render.resolution_percentage = 50 if PREVIEW else 100
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.filepath = os.path.join(ROOT, "blender", "preview.png" if PREVIEW else "render.png")

try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = dev_type
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == dev_type
            scene.cycles.device = "GPU"
            print("Cycles device:", dev_type)
            break
except Exception as e:  # CPU fallback
    print("GPU setup skipped:", e)

# white haze towards the back of the field (mist pass mixed over the beauty pass)
bpy.context.view_layer.use_pass_mist = True
world.mist_settings.start = K["distance"] - 0.5
world.mist_settings.depth = 8.5
world.mist_settings.falloff = "QUADRATIC"
scene.use_nodes = True
ct = scene.node_tree
ct.nodes.clear()
rl = ct.nodes.new("CompositorNodeRLayers")
mix = ct.nodes.new("CompositorNodeMixRGB")
mix.inputs[2].default_value = (1.0, 0.985, 0.965, 1.0)
out = ct.nodes.new("CompositorNodeComposite")
ct.links.new(rl.outputs["Mist"], mix.inputs[0])
ct.links.new(rl.outputs["Image"], mix.inputs[1])
ct.links.new(mix.outputs[0], out.inputs[0])

if not PREVIEW:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "blender", "scene.blend"))
bpy.ops.render.render(write_still=True)
print("DONE")
