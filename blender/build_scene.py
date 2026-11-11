"""Rebuild concept.png in Blender: a terraced field of frosted acrylic cards.

    blender -b -P blender/build_scene.py -- [--preview] [--no-export] [--no-dof] [--samples N]
    blender -b -P blender/build_scene.py -- --bake [--samples N]
    either mode:  --set light.skyStrength=0.6   override any value of params.json for one run
                  --out file.png                where the still goes

Reads   public/assets/params.json       (shared with the three.js app)
Writes  public/assets/tex/frost.png     tileable vein / cloud map used by the acrylic (both modes)

scene mode (default)
        public/assets/card.glb          one card (body + rivet), for instancing
        blender/scene.blend             the full scene
        blender/render.png              Cycles still to compare against concept.png

card mode (--card): one card on a white sweep, framed like card.png, to judge the material on its own
        blender/card_check.png

bake mode (--bake): light the card with the very same rig and store what Cycles sees, looking along
the scene camera, as textures
        public/assets/tex/open_*.png    the faces of the card (front / end / top), unoccluded
        public/assets/tex/lut_*.png     how that light fades with depth inside the slot between cards
        public/assets/tex/glow_*.png    the card lit only by its own edge light
"""
import bpy, bmesh, json, math, os, sys
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW = "--preview" in argv
BAKE = "--bake" in argv
EXPORT = "--no-export" not in argv and not BAKE and "--card" not in argv
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else (48 if PREVIEW else 128 if BAKE else 256)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "public", "assets", "tex")
os.makedirs(TEX, exist_ok=True)
with open(os.path.join(ROOT, "public", "assets", "params.json")) as f:
    P = json.load(f)
# experiments without touching the file:  --set light.skyStrength=1.0
for k, a in enumerate(argv):
    if a == "--set":
        path, val = argv[k + 1].split("=", 1)
        node = P
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = json.loads(val)
OUT_NAME = argv[argv.index("--out") + 1] if "--out" in argv else None

C = P["card"]
L, H, T = C["length"], C["height"], C["thickness"]
T_FRONT, T_CORE, END_STEP = C["frontSlab"], C["core"], C["endStep"]
PITCH, ROW_PITCH = P["pitchCard"], P["pitchRow"]
ROWS, COLS = P["rows"], P["cols"]
F = P["field"]
LIGHT, BK = P["light"], P["bake"]
hl, ht = L / 2, T / 2


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


# ------------------------------------------------------------------ image helpers
def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def write_png(path, rgb):
    """rgb: (h, w, 3) display-encoded floats, row 0 at the bottom. Stored byte for byte."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new("_out", w, h, alpha=False)
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = np.clip(rgb, 0.0, 1.0)
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    print("wrote", os.path.relpath(path, ROOT), f"{w}x{h}")


def periodic_noise(h, w, fx, fy, seed, beta=3.0):
    """Tileable smooth noise: white noise low-passed in the frequency domain (fx, fy = cycles per tile)."""
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))
    ky = np.fft.fftfreq(h)[:, None] * h
    kx = np.fft.fftfreq(w)[None, :] * w
    amp = 1.0 / (1.0 + (kx / fx) ** 2 + (ky / fy) ** 2) ** (beta / 2)
    amp[0, 0] = 0.0
    n = np.fft.ifft2(spec * amp).real
    return (n - n.mean()) / n.std()


def make_frost(w=1024, h=512):
    """Cloudy veins of a cast frosted sheet: long soft wisps along the card, drifting slowly across it."""
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    warp = periodic_noise(h, w, 1.2, 1.5, 1)
    warp2 = periodic_noise(h, w, 3.0, 4.0, 2)
    wisps = np.sin(2 * np.pi * (2 * v + 1 * u + 0.22 * warp + 0.05 * warp2))
    clouds = periodic_noise(h, w, 0.9, 3.5, 3)
    lines = periodic_noise(h, w, 0.6, 11.0, 5)                 # soft streaks left by the casting
    haze = periodic_noise(h, w, 5.0, 12.0, 4)
    f = 0.34 * wisps + 0.36 * clouds + 0.22 * lines + 0.08 * haze
    lo, hi = np.percentile(f, [1, 99])
    f = np.clip((f - lo) / (hi - lo), 0, 1)
    f = f * f * (3 - 2 * f)
    return (f - f.mean() + 0.5).astype(np.float32)


frost = make_frost()
FROST_PATH = os.path.join(TEX, "frost.png")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
write_png(FROST_PATH, np.repeat(frost[..., None], 3, axis=2))


# ------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree, m.node_tree.nodes["Principled BSDF"]


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


M = P["material"]
Y_SEAM = -ht + T_FRONT + T_CORE / 2      # the two cast sheets meet here (T_CORE = width of the seam line)
BX0 = -hl + END_STEP + C["blockInset"]   # the block sits this far in from the end of the front sheet

# Frosted acrylic. One solid body, so everything that tells the layers apart is drawn by the material:
#   veins    frost.png, u along the card, v up it and drifting through the thickness, so faces show
#            clouds and the top edge shows streaks; every card slides the map by its own random offset
#   frame    the inner spacer seen through the face (card.png): a paler band along the top and block end
#   seam     pale line where the sheets meet, visible on the top edge and the ends
#   glow     object colour (R) switches the edge light on; the LED sits at the block end, so it falls
#            off along the card and is strongest in the seam
acrylic, nt, b = new_mat("Acrylic")
b.inputs["Roughness"].default_value = M["roughness"]
b.inputs["IOR"].default_value = 1.49
b.inputs["Specular IOR Level"].default_value = M["specular"]
b.inputs["Coat Weight"].default_value = M["coat"]
b.inputs["Coat Roughness"].default_value = M["coatRoughness"]
tc = nt.nodes.new("ShaderNodeTexCoord")
oi = nt.nodes.new("ShaderNodeObjectInfo")
xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], xyz.inputs[0])
uv = nt.nodes.new("ShaderNodeCombineXYZ")
nt.links.new(math_node(nt, "MULTIPLY_ADD", oi.outputs["Random"], 7.13, math_node(nt, "DIVIDE", xyz.outputs["X"], L)), uv.inputs["X"])
vv = math_node(nt, "MULTIPLY_ADD", xyz.outputs["Y"], M["veinAcross"], math_node(nt, "DIVIDE", xyz.outputs["Z"], H))
nt.links.new(math_node(nt, "MULTIPLY_ADD", oi.outputs["Random"], 3.71, vv), uv.inputs["Y"])
img = nt.nodes.new("ShaderNodeTexImage")
img.image = bpy.data.images.load(FROST_PATH)
img.image.colorspace_settings.name = "Non-Color"
nt.links.new(uv.outputs[0], img.inputs["Vector"])
frost_gain = nt.nodes.new("ShaderNodeValue")            # 0 while baking: the app adds the veins per card
frost_gain.outputs[0].default_value = 1.0
fac = math_node(nt, "MULTIPLY_ADD", math_node(nt, "SUBTRACT", img.outputs["Color"], 0.5), frost_gain.outputs[0], 0.5)
mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
mixc.inputs["A"].default_value = (*M["veinDark"], 1)
mixc.inputs["B"].default_value = (*M["veinLight"], 1)
nt.links.new(fac, mixc.inputs["Factor"])
frame = math_node(nt, "MAXIMUM",
                  math_node(nt, "GREATER_THAN", xyz.outputs["Z"], H - M["frameTop"]),
                  math_node(nt, "LESS_THAN", xyz.outputs["X"], BX0 + C["block"][0]))
mixf = nt.nodes.new("ShaderNodeMix"); mixf.data_type = "RGBA"
mixf.inputs["B"].default_value = (*M["frame"], 1)
nt.links.new(math_node(nt, "MULTIPLY", frame, M["frameMix"]), mixf.inputs["Factor"])
nt.links.new(mixc.outputs["Result"], mixf.inputs["A"])
seam = math_node(nt, "LESS_THAN", math_node(nt, "ABSOLUTE", math_node(nt, "SUBTRACT", xyz.outputs["Y"], Y_SEAM)), T_CORE / 2)
mixs = nt.nodes.new("ShaderNodeMix"); mixs.data_type = "RGBA"
mixs.inputs["B"].default_value = (*M["seam"], 1)
nt.links.new(math_node(nt, "MULTIPLY", seam, 0.85), mixs.inputs["Factor"])
# the back sheet is the whiter, more opaque of the two (card.png, and the pale strip along every top edge)
back = math_node(nt, "GREATER_THAN", xyz.outputs["Y"], Y_SEAM + T_CORE / 2)
mixk = nt.nodes.new("ShaderNodeMix"); mixk.data_type = "RGBA"
mixk.inputs["B"].default_value = (*M["backSheet"], 1)
nt.links.new(math_node(nt, "MULTIPLY", back, M["backMix"]), mixk.inputs["Factor"])
nt.links.new(mixf.outputs["Result"], mixk.inputs["A"])
# looking down into the edge of the clearer front sheet the eye sees into its depth, so its top edge reads
# darker than the opaque back sheet next to it
front_top = math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", xyz.outputs["Z"], H - 0.003),
                      math_node(nt, "LESS_THAN", xyz.outputs["Y"], Y_SEAM - T_CORE / 2))
mixt = nt.nodes.new("ShaderNodeMix"); mixt.data_type = "RGBA"; mixt.blend_type = "MULTIPLY"
mixt.inputs["B"].default_value = (*M["frontTop"], 1)
nt.links.new(front_top, mixt.inputs["Factor"])
nt.links.new(mixk.outputs["Result"], mixt.inputs["A"])
nt.links.new(mixt.outputs["Result"], mixs.inputs["A"])
# pale block inlaid at the top of the front sheet, level with its face and the top edge: flush on every
# side, so it is a region of the surface rather than geometry
bw, bh, bd = C["block"]
detail_gain = nt.nodes.new("ShaderNodeValue")           # 0 while measuring pure lighting for the bakes
detail_gain.outputs[0].default_value = 1.0
block = math_node(nt, "MULTIPLY",
                  math_node(nt, "MULTIPLY", math_node(nt, "LESS_THAN", xyz.outputs["X"], BX0 + bw),
                            math_node(nt, "GREATER_THAN", xyz.outputs["Z"], H - bh)),
                  math_node(nt, "MULTIPLY", math_node(nt, "LESS_THAN", xyz.outputs["Y"], -ht + bd),
                            math_node(nt, "GREATER_THAN", xyz.outputs["X"], BX0)))
block = math_node(nt, "MULTIPLY", block, detail_gain.outputs[0])
mixb = nt.nodes.new("ShaderNodeMix"); mixb.data_type = "RGBA"
mixb.inputs["B"].default_value = (*M["block"], 1)
nt.links.new(block, mixb.inputs["Factor"])
nt.links.new(mixs.outputs["Result"], mixb.inputs["A"])
mixs = mixb
nt.links.new(mixs.outputs["Result"], b.inputs["Base Color"])
# the block is an opaque inlay: no light travels through it, so it keeps crisp edges and its own shading
# (the sheets themselves also pass part of the light straight through: frosted, not opaque)
# The two big faces are close to clear: the eye goes through them into the stack. The sawn edges (top and
# ends) are properly frosted: they scatter the light that travels inside the sheet and read as pale strips.
nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Normal"], nrm.inputs[0])
edge = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "ABSOLUTE", nrm.outputs["Y"]))
trans = math_node(nt, "MULTIPLY_ADD", edge, M["transmissionEdge"] - M["transmission"], M["transmission"])
nt.links.new(math_node(nt, "MULTIPLY", trans, math_node(nt, "SUBTRACT", 1.0, block)), b.inputs["Transmission Weight"])
nt.links.new(math_node(nt, "MULTIPLY_ADD", edge, M["roughnessEdge"] - M["roughness"], M["roughness"]), b.inputs["Roughness"])
nt.links.new(math_node(nt, "MULTIPLY_ADD", block, -M["sss"], M["sss"]), b.inputs["Subsurface Weight"])
b.inputs["Subsurface Radius"].default_value = M["sssRadius"]
# the back sheet and the block are denser: light does not travel as far inside them
dense = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY_ADD", block, M["blockScale"] - 1.0, 1.0),
                  math_node(nt, "MULTIPLY_ADD", back, M["backScale"] - 1.0, 1.0))
nt.links.new(math_node(nt, "MULTIPLY", dense, M["sssScale"]), b.inputs["Subsurface Scale"])
b.inputs["Subsurface Anisotropy"].default_value = M["sssAniso"]
sep = nt.nodes.new("ShaderNodeSeparateColor")
nt.links.new(oi.outputs["Color"], sep.inputs["Color"])
fall = math_node(nt, "EXPONENT", math_node(nt, "MULTIPLY_ADD", xyz.outputs["X"], -1 / M["glowFalloff"], -hl / M["glowFalloff"]))
along = math_node(nt, "MULTIPLY_ADD", fall, M["glowEnd"], M["glowBase"])
across = math_node(nt, "MULTIPLY_ADD", seam, M["glowSeam"], 1.0)
b.inputs["Emission Color"].default_value = (*M["glow"], 1)
lit = math_node(nt, "MULTIPLY", sep.outputs["Red"], math_node(nt, "MULTIPLY", along, across))
# the block reads as the whitest thing on the card even in shade
nt.links.new(math_node(nt, "MULTIPLY_ADD", block, M["blockLift"], lit), b.inputs["Emission Strength"])
# polished eyelet rivet
rivet, nt, b = new_mat("Rivet")
b.inputs["Base Color"].default_value = (0.78, 0.75, 0.70, 1)
b.inputs["Metallic"].default_value = 1.0
b.inputs["Roughness"].default_value = 0.3

MAT_ACRYLIC, MAT_RIVET = 0, 1


# ------------------------------------------------------------------ card mesh
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


def build_card(name, detail=True, top_shear=None):
    """detail=False leaves out block and rivet (used to measure pure lighting); top_shear(x) lifts the top."""
    bm = bmesh.new()
    bev = C["bevel"]
    # one solid body: the back sheet runs the full length, the front sheet stops END_STEP short at the
    # block end (card.png), which leaves an L-shaped footprint
    outline = [(-hl + END_STEP, -ht), (hl, -ht), (hl, ht), (-hl, ht), (-hl, Y_SEAM), (-hl + END_STEP, Y_SEAM)]
    bot = [bm.verts.new((x, y, 0.0)) for x, y in outline]
    top = [bm.verts.new((x, y, H)) for x, y in outline]
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    n = len(outline)
    for k in range(n):
        bm.faces.new((bot[k], bot[(k + 1) % n], top[(k + 1) % n], top[k]))
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bev, offset_type="OFFSET", segments=2, profile=0.5, affect="EDGES")
    for f in bm.faces:
        f.material_index = MAT_ACRYLIC
        f.smooth = True
    if detail:
        # eyelet rivet sitting on the front face, far top corner
        add_lathe_y(bm, hl - 0.065, H - 0.09, -ht, [
            (0.000, 0.0005), (0.011, 0.0005), (0.016, 0.009), (0.022, 0.014),
            (0.029, 0.014), (0.034, 0.009), (0.036, 0.0005)], MAT_RIVET)
    if top_shear:
        for v in bm.verts:
            if v.co.z > H / 2:
                v.co.z += top_shear(v.co.x)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name + "Base")
    bm.to_mesh(mesh)
    bm.free()
    for m in (acrylic, rivet):
        mesh.materials.append(m)
    # weighted normals keep the big faces flat while the bevels stay rounded
    tmp = bpy.data.objects.new("CardTmp", mesh)
    scene.collection.objects.link(tmp)
    wn = tmp.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True
    wn.weight = 100
    dg = bpy.context.evaluated_depsgraph_get()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
    out.name = name
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(mesh)
    return out


card_mesh = build_card("Card")
print("card tris:", sum(len(p.vertices) - 2 for p in card_mesh.polygons))

if EXPORT:
    card = bpy.data.objects.new("Card", card_mesh)
    scene.collection.objects.link(card)
    bpy.ops.object.select_all(action="DESELECT")
    card.select_set(True)
    bpy.context.view_layer.objects.active = card
    bpy.ops.export_scene.gltf(filepath=os.path.join(ROOT, "public", "assets", "card.glb"),
                              export_format="GLB", use_selection=True, export_yup=True, export_image_format="NONE",
                              export_apply=True)
    bpy.data.objects.remove(card)


# ------------------------------------------------------------------ light rig (shared by scene and bakes)
# Everything is directional or ambient on purpose: every card in the field is lit exactly like the one
# that gets baked, so the baked textures stay valid anywhere in the field.
world = bpy.data.worlds.new("World")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (*LIGHT["sky"], 1)
bg.inputs["Strength"].default_value = LIGHT["skyStrength"]
scene.world = world

sun_data = bpy.data.lights.new("Key", "SUN")
sun_data.energy = LIGHT["keyStrength"]
sun_data.color = LIGHT["keyColor"]
sun_data.angle = math.radians(LIGHT["keyAngleDeg"])
sun = bpy.data.objects.new("Key", sun_data)
sun.rotation_euler = (-Vector(LIGHT["keyDir"])).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(sun)

# white fill from above the camera's left shoulder: it is what makes the frosted edges read pale
fill_data = bpy.data.lights.new("Fill", "SUN")
fill_data.energy = LIGHT["fillStrength"]
fill_data.color = LIGHT["fillColor"]
fill_data.angle = math.radians(LIGHT["fillAngleDeg"])
fill = bpy.data.objects.new("Fill", fill_data)
fill.rotation_euler = (-Vector(LIGHT["fillDir"])).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(fill)

# warm glowing floor far below, so any sliver between cards reads as light, not as a black void
fm = bpy.data.meshes.new("Floor")
fb = bmesh.new()
bmesh.ops.create_grid(fb, x_segments=1, y_segments=1, size=80.0)
fb.to_mesh(fm); fb.free()
floor_mat, nt, b = new_mat("FloorGlow")
b.inputs["Base Color"].default_value = (1.0, 0.85, 0.66, 1)
b.inputs["Emission Color"].default_value = (*LIGHT["floor"], 1)
b.inputs["Emission Strength"].default_value = LIGHT["floorStrength"]
fm.materials.append(floor_mat)
floor = bpy.data.objects.new("Floor", fm)
floor.location = (0, 0, -0.05)
scene.collection.objects.link(floor)

# ------------------------------------------------------------------ render settings
scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 24
# AgX rolls the highlights off towards a creamy white the way concept.png does; three.js has the same
# transform (AgXToneMapping), so the app can reproduce it from linear textures
VIEW = P["view"]
scene.view_settings.view_transform = VIEW["transform"]
scene.view_settings.look = VIEW["look"]
scene.view_settings.exposure = VIEW["exposure"]
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


# ================================================================== bake mode
def bake():
    """The frosted sheets are partly see-through, so what a face looks like depends on where the eye is.
    Every shot is therefore taken along the scene camera's own direction (orthographic, so it holds for
    every card in the field) and then straightened onto the face it shows."""
    stage = bpy.data.collections.new("Stage")
    scene.collection.children.link(stage)
    plain_mesh = build_card("CardPlain", detail=False)
    d0, d1 = BK["topRange"]
    wedge_mesh = build_card("CardWedge", detail=False, top_shear=lambda x: d0 + (d1 - d0) * (x + hl) / L)

    K = P["camera"]
    az, el = math.radians(K["azimuthDeg"]), math.radians(K["elevationDeg"])
    view_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)))
    cam_data = bpy.data.cameras.new("BakeCam")
    cam_data.type = "ORTHO"
    cam_data.clip_start, cam_data.clip_end = 0.1, 100
    cam = bpy.data.objects.new("BakeCam", cam_data)
    cam.rotation_euler = view_dir.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    bpy.context.view_layer.update()
    right = np.array(cam.matrix_world.col[0][:3])
    up = np.array(cam.matrix_world.col[1][:3])
    corners = np.array([(x, y, z) for x in (-hl, hl) for y in (-ht - 0.02, ht) for z in (0.0, H)])
    span = 1.04 * max(np.ptp(corners @ right), np.ptp(corners @ up))
    mid = np.array([(corners @ right).max() + (corners @ right).min(), (corners @ up).max() + (corners @ up).min()]) / 2
    N = BK["shotSize"]
    cam_data.ortho_scale = span
    scene.render.resolution_x = scene.render.resolution_y = N
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    tmp_exr = os.path.join(bpy.app.tempdir, "bake.exr")

    def build(mesh, front=0.0, back=0.0, rows=0.0, wedge=None, glow=0.0):
        """The whole field around one test card. front / back / rows give the level of the cards in front
        of it, behind it and in the other rows relative to the test card (0 = flush, negative = lower).
        Returns the shot: linear (N, N, 3), bottom row first, plus where it was taken from."""
        for o in list(stage.objects):
            bpy.data.objects.remove(o)
        raise_ = max(0.0, -front, -back, -rows)
        for i in range(ROWS[0], ROWS[1] + 1):
            for j in range(COLS[0], COLS[1] + 1):
                me = mesh if (i, j) == (0, 0) else (wedge[1] if wedge and i == 0 and j * wedge[0] > 0 else card_mesh)
                o = bpy.data.objects.new(f"S_{i}_{j}", me)
                level = 0.0 if (i, j) == (0, 0) else rows if i != 0 else front if j < 0 else back
                o.location = (i * ROW_PITCH, j * PITCH, raise_ + level)
                o.visible_camera = (i, j) == (0, 0)       # neighbours still shade, reflect and show through
                g = glow if (i, j) == (0, 0) else 0.0
                o.color = (g, g, g, 1.0)
                stage.objects.link(o)
        origin = np.array((0.0, 0.0, raise_)) + mid[0] * right + mid[1] * up
        cam.location = Vector(origin) - view_dir * 30
        scene.render.filepath = tmp_exr
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(tmp_exr)
        buf = np.empty(N * N * 4, np.float32)
        im.pixels.foreach_get(buf)
        bpy.data.images.remove(im)
        a = buf.reshape(N, N, 4)[..., :3].astype(np.float64)
        print(f"shot front={front} back={back} rows={rows} mean {a.mean(axis=(0, 1)).round(3)} p99.5 {np.percentile(a, 99.5):.3f}")
        return a, origin

    def face(shot, view, width):
        """Straighten one face out of a shot. Rows run bottom to top; columns as the app's uv expects."""
        img, origin = shot
        z0 = origin[2] - (mid[0] * right + mid[1] * up)[2]
        if view == "front":      # u = x, v = z
            w, h = width, width // 2
            a, b = np.meshgrid(-hl + (np.arange(w) + 0.5) / w * L, z0 + (np.arange(h) + 0.5) / h * H)
            pts = np.stack([a, np.full_like(a, -ht), b], axis=-1)
        elif view == "end":      # block end: u runs from the back sheet to the front one, v = z
            w, h = width // 8, width
            a, b = np.meshgrid(ht - (np.arange(w) + 0.5) / w * T, z0 + (np.arange(h) + 0.5) / h * H)
            pts = np.stack([np.full_like(a, -hl), a, b], axis=-1)
        else:                    # top: u = x, v from the front edge to the back one
            w, h = width, width // 16
            a, b = np.meshgrid(-hl + (np.arange(w) + 0.5) / w * L, -ht + (np.arange(h) + 0.5) / h * T)
            pts = np.stack([a, b, np.full_like(a, z0 + H)], axis=-1)
        # stay a bevel away from the outline: past it the shot shows whatever lies behind the card
        m = 1.5 * C["bevel"]
        lo, hi = np.array((-hl + m, -ht + m, z0 + m)), np.array((hl - m, ht - m, z0 + H - m))
        rel = np.clip(pts, lo, hi) - origin
        fx = np.clip(((rel @ right) / span + 0.5) * N - 0.5, 0, N - 1.001)
        fy = np.clip(((rel @ up) / span + 0.5) * N - 0.5, 0, N - 1.001)
        x0, y0 = fx.astype(int), fy.astype(int)
        tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
        return (img[y0, x0] * (1 - tx) * (1 - ty) + img[y0, x0 + 1] * tx * (1 - ty)
                + img[y0 + 1, x0] * (1 - tx) * ty + img[y0 + 1, x0 + 1] * tx * ty)

    def ratio(num, den):
        return num / np.maximum(den, 1e-3)

    def save(name, a):
        write_png(os.path.join(TEX, name + ".png"), srgb_encode(a))

    OPEN, STEP = BK["openRaise"], BK["stepRaise"]
    scale = BK["lutScale"]
    frost_gain.outputs[0].default_value = 0.0
    floor.visible_camera = False                    # it still lights the slots, but the shots need not trace it

    # 1. the card as the eye sees it: the faces with nothing in front of them (and the next card standing
    #    behind, as it does all along the ridge), the top level with its neighbours
    is_open = dict(front=-OPEN, rows=-OPEN)
    k = BK["openScale"]      # headroom above 1.0, so the app's tone mapping has highlights to roll off
    s = build(card_mesh, **is_open)
    save("open_front", face(s, "front", 2048) * k)
    save("open_end", face(s, "end", 1024) * k)
    save("open_top", face(build(card_mesh), "top", 2048) * k)

    # 2. slot light, relative to that unoccluded card. Row v of each map is a depth below the tops of the
    #    cards concerned, not a place on the card:
    #      lut_front   the cards in front come up        (how deep the face sits in the slot before it)
    #      lut_far     the cards behind go down          (the sheet is see-through: what stands behind shows)
    #      lut_end     the neighbouring rows come up
    detail_gain.outputs[0].default_value = 0.0
    o = build(plain_mesh, **is_open)
    save("lut_front", ratio(face(build(plain_mesh, front=-STEP, rows=-OPEN), "front", 512), face(o, "front", 512)) / scale)
    save("lut_far", ratio(face(build(plain_mesh, front=-OPEN, back=-STEP, rows=-OPEN), "front", 512), face(o, "front", 512)) / scale)
    save("lut_end", ratio(face(build(plain_mesh, front=-OPEN, rows=-STEP), "end", 256), face(o, "end", 256)) / scale)

    # 3. top faces: the cards in front (then those behind) get a top edge that climbs from d0 below to d1
    #    above the test card, which turns the length of the card into a depth axis; relative to the level
    #    stack. All of them, not just the nearest: on the ridge every card on one side is lower or higher
    flat = face(build(plain_mesh), "top", 512)
    for name, where in (("lut_top", -1), ("lut_top_back", 1)):
        save(name, ratio(face(build(plain_mesh, wedge=(where, wedge_mesh)), "top", 512), flat) / scale)

    # 4. edge light only
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sun.hide_render = True
    fill.hide_render = True
    floor.hide_render = True
    g = BK["glowScale"]
    detail_gain.outputs[0].default_value = 1.0
    s = build(card_mesh, glow=1.0, **is_open)
    save("glow_front", face(s, "front", 1024) * g)
    save("glow_end", face(s, "end", 512) * g)
    save("glow_top", face(build(card_mesh, glow=1.0), "top", 1024) * g)


if BAKE:
    bake()
    print("DONE")
    sys.exit(0)

# ================================================================== card mode
def card_shot():
    """One card on a white sweep, framed like card.png: checks the material on its own."""
    floor.hide_render = True
    sun.hide_render = True
    fill.hide_render = True
    bg.inputs["Color"].default_value = (0.97, 0.96, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.85
    sm = bpy.data.meshes.new("Sweep")
    sb = bmesh.new()
    bmesh.ops.create_grid(sb, x_segments=1, y_segments=1, size=40.0)
    sb.to_mesh(sm); sb.free()
    sweep_mat, _, sbsdf = new_mat("Sweep")
    sbsdf.inputs["Base Color"].default_value = (0.93, 0.92, 0.91, 1)
    sbsdf.inputs["Roughness"].default_value = 0.6
    sm.materials.append(sweep_mat)
    scene.collection.objects.link(bpy.data.objects.new("Sweep", sm))
    card = bpy.data.objects.new("Card", card_mesh)
    card.color = (0.6, 0.6, 0.6, 1.0)
    scene.collection.objects.link(card)
    kd = bpy.data.lights.new("Soft", "AREA")
    kd.shape, kd.size, kd.energy = "DISK", 6.0, 160.0
    ko = bpy.data.objects.new("Soft", kd)
    ko.location = (-3.0, -5.0, 5.0)
    ko.rotation_euler = (Vector((0, 0, 0.5)) - Vector(ko.location)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(ko)
    d = Vector((0.42, 0.86, -0.17)).normalized()
    cd = bpy.data.cameras.new("CardCam")
    cd.lens = 85
    co = bpy.data.objects.new("CardCam", cd)
    co.location = Vector((0.0, 0.0, 0.52)) - d * 7.2
    co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co
    scene.render.resolution_x, scene.render.resolution_y = 1448, 1086
    scene.render.resolution_percentage = 50 if PREVIEW else 100
    scene.render.filepath = OUT_NAME or os.path.join(ROOT, "blender", "card_check.png")
    bpy.ops.render.render(write_still=True)


if "--card" in argv:
    card_shot()
    print("DONE")
    sys.exit(0)

# ================================================================== scene mode
field = bpy.data.collections.new("Cards")
scene.collection.children.link(field)
for i in range(ROWS[0], ROWS[1] + 1):
    for j in range(COLS[0], COLS[1] + 1):
        o = bpy.data.objects.new(f"Card_{i}_{j}", card_mesh)
        o.location = (i * ROW_PITCH, j * PITCH, offset_of(i, j))
        g = glow_of(i, j)
        o.color = (g, g, g, 1.0)
        field.objects.link(o)

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
cam_data.dof.use_dof = "--no-dof" not in argv
cam_data.dof.focus_distance = (focus_pt - cam.location).dot(view_dir)
cam_data.dof.aperture_fstop = K["fstop"]

# pool of light around the focused card, from low in front of it so that it catches the faces and ends
# that look at the camera (the app does the same with a gain that follows the focus, see uSpot)
SP = P["spot"]
spot_data = bpy.data.lights.new("Pool", "SPOT")
spot_from = Vector(SP["from"])
spot_target = Vector((SP["offset"][0], SP["offset"][1], offset_of(0, 0) + H * 0.6))
spot_data.energy = SP["gain"] * SP["watts"]
spot_data.color = SP["color"]
spot_data.spot_size = 2 * math.atan(SP["radius"] * 1.7 / spot_from.length)
spot_data.spot_blend = 1.0
spot_data.shadow_soft_size = 2.0
spot = bpy.data.objects.new("Pool", spot_data)
spot.location = spot_target + spot_from
spot.rotation_euler = (-spot_from).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(spot)

scene.render.resolution_x, scene.render.resolution_y = 1672, 941
scene.render.resolution_percentage = 50 if PREVIEW else 100
scene.render.filepath = OUT_NAME or os.path.join(ROOT, "blender", "preview.png" if PREVIEW else "render.png")

# white haze towards the back of the field (mist pass mixed over the beauty pass)
HZ = P["haze"]
bpy.context.view_layer.use_pass_mist = True
world.mist_settings.start = K["distance"] + HZ["start"]
world.mist_settings.depth = HZ["depth"]
world.mist_settings.falloff = "QUADRATIC"
scene.use_nodes = True
ct = scene.node_tree
ct.nodes.clear()
rl = ct.nodes.new("CompositorNodeRLayers")
mix = ct.nodes.new("CompositorNodeMixRGB")
mix.inputs[2].default_value = (*HZ["color"], 1.0)
out = ct.nodes.new("CompositorNodeComposite")
veil = ct.nodes.new("CompositorNodeMath")           # mist, on top of a faint veil over everything
veil.operation = "MULTIPLY_ADD"
veil.inputs[1].default_value = 1.0 - HZ["base"]
veil.inputs[2].default_value = HZ["base"]
ct.links.new(rl.outputs["Mist"], veil.inputs[0])
ct.links.new(veil.outputs[0], mix.inputs[0])
ct.links.new(rl.outputs["Image"], mix.inputs[1])
glare = ct.nodes.new("CompositorNodeGlare")      # the soft bloom of the concept (UnrealBloomPass in the app)
glare.glare_type = "FOG_GLOW"
glare.threshold = 0.9
glare.size = 7
glare.mix = -0.75
ct.links.new(mix.outputs[0], glare.inputs[0])
ct.links.new(glare.outputs[0], out.inputs[0])

if not PREVIEW:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "blender", "scene.blend"))
bpy.ops.render.render(write_still=True)
print("DONE")
