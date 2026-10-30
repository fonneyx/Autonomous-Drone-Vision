"""Build the simulator's people, vehicles and drones in Blender and export them.

Run inside Blender (Scripting tab -> Open -> Run Script), or from a terminal:
    blender --background --python scripts/blender/build_sim_models.py

Every model is low-poly and split by material role (paint, glass, skin, shirt,
...). Each role is written to its own OBJ in assets/sim_models/, and
manifest.json records the class, parts and default colours. The simulator
loads them as meshes and re-colours the role parts randomly every episode.

Coordinates: metres, X forward, Y left, Z up. People and vehicles stand on
z = 0; drones are centred on their body.
"""

import json
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

if "OUT_DIR" not in globals():
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "sim_models")
OUT_DIR = os.path.abspath(OUT_DIR)

MATS = {}
VARIANTS = {}  # name -> object class (person / vehicle / drone)


# --------------------------------------------------------------------- helpers
def mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    MATS[name] = m
    return m


def collection(name, object_class):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    VARIANTS[name] = object_class
    return col


def link(ob, col):
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)


def new_object(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def finish(ob, material, col, bevel=0.0, segments=2, smooth=False):
    ob.data.materials.clear()
    ob.data.materials.append(MATS[material])
    link(ob, col)
    if bevel > 0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
    for p in ob.data.polygons:
        p.use_smooth = smooth
    return ob


def profile_extrude(name, pts, y0, y1):
    """Polygon in the XZ plane extruded between y0 and y1."""
    bm = bmesh.new()
    front = [bm.verts.new((x, y0, z)) for x, z in pts]
    back = [bm.verts.new((x, y1, z)) for x, z in pts]
    bm.faces.new(front[::-1])
    bm.faces.new(back)
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        bm.faces.new([front[i], front[j], back[j], back[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return new_object(name, bm)


def box(name, center, size, rot_z=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot_z, 3, "Z"), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(center), verts=bm.verts)
    return new_object(name, bm)


def cylinder(name, center, radius, depth, axis="Z", segments=20, radius2=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=segments, radius1=radius,
        radius2=radius if radius2 is None else radius2, depth=depth,
    )
    if axis == "Y":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"), verts=bm.verts)
    elif axis == "X":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(center), verts=bm.verts)
    return new_object(name, bm)


def uv_sphere(name, center, scale, segs=16, rings=10):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(center), verts=bm.verts)
    return new_object(name, bm)


def bar(name, start, end, width, height):
    """Box stretched between two points in the XY plane."""
    start, end = Vector((*start[:2], 0.0)), Vector((*end[:2], 0.0))
    mid = (start + end) / 2
    direction = end - start
    angle = math.atan2(direction.y, direction.x)
    return box(name, mid, (direction.length, width, height), angle)


# --------------------------------------------------------------------- vehicles
def vehicle(name, width, body_pts, glass_pts, glass_w, roof, wheel_r, wheel_x, extra=None):
    col = collection(name, "vehicle")
    b = profile_extrude(f"{name}_body", body_pts, -width / 2, width / 2)
    finish(b, "paint", col, bevel=0.06, segments=3)
    g = profile_extrude(f"{name}_glass", glass_pts, -width * glass_w / 2, width * glass_w / 2)
    finish(g, "glass", col, bevel=0.04)
    if roof:
        x0, x1, z = roof
        r = box(f"{name}_roof", ((x0 + x1) / 2, 0, z), (x1 - x0, width * glass_w + 0.04, 0.06))
        finish(r, "paint", col, bevel=0.025)
    for i, (wx, side) in enumerate([(wheel_x[0], 1), (wheel_x[0], -1), (wheel_x[1], 1), (wheel_x[1], -1)]):
        t = cylinder(f"{name}_tire{i}", (wx, side * (width / 2 - 0.05), wheel_r), wheel_r, 0.26, "Y", 24)
        finish(t, "tire", col, bevel=0.04, smooth=True)
        rim = cylinder(f"{name}_rim{i}", (wx, side * (width / 2 - 0.02), wheel_r), wheel_r * 0.6, 0.22, "Y", 16)
        finish(rim, "rim", col)
    front = max(x for x, _ in body_pts)
    rear = min(x for x, _ in body_pts)
    for side in (1, -1):
        h = box(f"{name}_head{side}", (front + 0.01, side * (width / 2 - 0.32), 0.8), (0.06, 0.34, 0.12))
        finish(h, "headlight", col, bevel=0.02)
        t = box(f"{name}_tail{side}", (rear - 0.01, side * (width / 2 - 0.25), 0.85), (0.06, 0.26, 0.16))
        finish(t, "taillight", col, bevel=0.02)
    for x in (front, rear):
        bp = box(f"{name}_bumper{int(x > 0)}", (x + (0.03 if x > 0 else -0.03), 0, 0.45), (0.1, width * 0.96, 0.18))
        finish(bp, "trim", col, bevel=0.03)
    if extra:
        extra(col)


def build_vehicles():
    vehicle("sedan", 1.8,
            [(-2.25, 0.32), (2.25, 0.32), (2.3, 0.62), (2.2, 0.82), (1.0, 0.95), (-1.45, 0.98), (-2.2, 0.95), (-2.3, 0.62)],
            [(1.05, 0.93), (0.25, 1.40), (-0.9, 1.40), (-1.55, 0.96)], 0.86, (-0.92, 0.27, 1.42), 0.33, (1.4, -1.4))
    vehicle("suv", 1.9,
            [(-2.35, 0.38), (2.35, 0.38), (2.4, 0.7), (2.3, 0.98), (1.25, 1.08), (-2.3, 1.10), (-2.4, 0.7)],
            [(1.25, 1.06), (0.65, 1.68), (-2.15, 1.68), (-2.3, 1.08)], 0.9, (-2.2, 0.68, 1.70), 0.38, (1.5, -1.5))

    def van_extra(col):
        up = box("van_upper", (-1.05, 0, 1.62), (2.9, 1.96, 0.78))
        finish(up, "paint", col, bevel=0.06, segments=3)

    vehicle("van", 2.0,
            [(-2.5, 0.36), (2.5, 0.36), (2.55, 0.72), (2.45, 1.05), (1.75, 1.22), (-2.5, 1.24), (-2.55, 0.72)],
            [(1.75, 1.2), (1.0, 2.0), (0.38, 2.0), (0.38, 1.22)], 0.94, (-2.5, 1.05, 2.03), 0.36, (1.6, -1.65),
            van_extra)

    def pickup_extra(col):
        for side in (1, -1):
            w = box(f"pickup_bedwall{side}", (-1.45, side * 0.94, 1.2), (2.3, 0.08, 0.3))
            finish(w, "paint", col, bevel=0.02)
        tg = box("pickup_tailgate", (-2.6, 0, 1.2), (0.08, 1.9, 0.3))
        finish(tg, "paint", col, bevel=0.02)

    vehicle("pickup", 2.0,
            [(-2.65, 0.42), (2.65, 0.42), (2.7, 0.78), (2.6, 1.08), (1.4, 1.15), (-2.65, 1.08), (-2.7, 0.78)],
            [(1.4, 1.13), (0.75, 1.8), (-0.2, 1.8), (-0.3, 1.10)], 0.92, (-0.3, 0.8, 1.83), 0.4, (1.75, -1.75),
            pickup_extra)


# ----------------------------------------------------------------------- people
JOINTS = {
    "pelvis": (0, 0, 0.95), "spine": (0, 0, 1.16), "chest": (0, 0, 1.36), "neck": (0, 0, 1.5),
    "shoulderL": (0, 0.19, 1.42), "shoulderR": (0, -0.19, 1.42),
    "hipL": (0, 0.1, 0.9), "hipR": (0, -0.1, 0.9),
}
POSES = {
    "stand": {"elbowL": (0.0, 0.23, 1.14), "handL": (0.03, 0.25, 0.87), "elbowR": (0.0, -0.23, 1.14),
              "handR": (0.03, -0.25, 0.87), "kneeL": (0.02, 0.11, 0.5), "ankleL": (0.0, 0.11, 0.09),
              "toeL": (0.15, 0.11, 0.04), "kneeR": (0.02, -0.11, 0.5), "ankleR": (0.0, -0.11, 0.09),
              "toeR": (0.15, -0.11, 0.04)},
    "walk": {"elbowL": (-0.12, 0.23, 1.15), "handL": (-0.2, 0.24, 0.9), "elbowR": (0.12, -0.23, 1.15),
             "handR": (0.25, -0.24, 0.95), "kneeL": (0.16, 0.1, 0.52), "ankleL": (0.24, 0.1, 0.09),
             "toeL": (0.38, 0.1, 0.04), "kneeR": (-0.06, -0.1, 0.5), "ankleR": (-0.26, -0.1, 0.14),
             "toeR": (-0.14, -0.1, 0.04)},
    "wave": {"elbowL": (0.0, 0.23, 1.14), "handL": (0.03, 0.25, 0.87), "elbowR": (0.02, -0.4, 1.58),
             "handR": (0.04, -0.42, 1.86), "kneeL": (0.02, 0.12, 0.5), "ankleL": (0.0, 0.14, 0.09),
             "toeL": (0.15, 0.15, 0.04), "kneeR": (0.02, -0.12, 0.5), "ankleR": (0.0, -0.14, 0.09),
             "toeR": (0.15, -0.15, 0.04)},
}
RADII = {"pelvis": (0.15, 0.12), "spine": (0.13, 0.1), "chest": (0.17, 0.11), "neck": (0.05, 0.05),
         "shoulder": (0.065, 0.065), "elbow": (0.045, 0.045), "hand": (0.04, 0.03), "hip": (0.085, 0.085),
         "knee": (0.06, 0.06), "ankle": (0.045, 0.045), "toe": (0.04, 0.035)}
BONES = [("pelvis", "spine", "shirt"), ("spine", "chest", "shirt"), ("chest", "neck", "skin"),
         ("chest", "shoulderL", "shirt"), ("shoulderL", "elbowL", "shirt"), ("elbowL", "handL", "skin"),
         ("chest", "shoulderR", "shirt"), ("shoulderR", "elbowR", "shirt"), ("elbowR", "handR", "skin"),
         ("pelvis", "hipL", "pants"), ("hipL", "kneeL", "pants"), ("kneeL", "ankleL", "pants"),
         ("ankleL", "toeL", "shoes"), ("pelvis", "hipR", "pants"), ("hipR", "kneeR", "pants"),
         ("kneeR", "ankleR", "pants"), ("ankleR", "toeR", "shoes")]


def _segment_distance(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
    return (p - (a + t * ab)).length


def person(name, pose):
    col = collection(name, "person")
    joints = dict(JOINTS)
    joints.update(POSES[pose])
    names = list(joints)
    skeleton = bpy.data.meshes.new(name + "_skeleton")
    skeleton.from_pydata([joints[n] for n in names], [(names.index(a), names.index(b)) for a, b, _ in BONES], [])
    ob = bpy.data.objects.new(name + "_skeleton", skeleton)
    bpy.context.scene.collection.objects.link(ob)
    ob.modifiers.new("skin", "SKIN")
    for i, n in enumerate(names):
        key = n[:-1] if n[-1] in "LR" else n
        ob.data.skin_vertices[0].data[i].radius = RADII[key]
        ob.data.skin_vertices[0].data[i].use_root = n == "pelvis"
    ob.modifiers.new("subsurf", "SUBSURF").levels = 1
    body_mesh = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(ob, do_unlink=True)
    body = bpy.data.objects.new(name + "_body", body_mesh)
    bpy.context.scene.collection.objects.link(body)
    link(body, col)

    roles = ["skin", "shirt", "pants", "shoes"]
    for role in roles:
        body_mesh.materials.append(MATS[role])
    bones = [(Vector(joints[a]), Vector(joints[b]), role) for a, b, role in BONES]
    for polygon in body_mesh.polygons:
        nearest = min(bones, key=lambda bone: _segment_distance(polygon.center, bone[0], bone[1]))
        polygon.material_index = roles.index(nearest[2])
        polygon.use_smooth = True

    finish(uv_sphere(name + "_head", (0.01, 0, 1.66), (0.1, 0.082, 0.118)), "skin", col, smooth=True)
    finish(uv_sphere(name + "_hair", (-0.025, 0, 1.70), (0.098, 0.088, 0.1)), "hair", col, smooth=True)


def build_people():
    for pose in POSES:
        person(f"person_{pose}", pose)


# ----------------------------------------------------------------------- drones
def propeller(name, center, radius, blades, col, angle):
    for k in range(blades):
        a = angle + k * 2 * math.pi / blades
        tip = (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
        blade = bar(f"{name}_b{k}", (center[0], center[1]), tip, radius * 0.16, 0.006)
        blade.location.z = center[2]
        finish(blade, "props", col, bevel=0.003)


def drone(name, arms, arm_len, body_size, prop_r, blades, legs, gimbal, round_body=False):
    col = collection(name, "drone")
    bx, by, bz = body_size
    if round_body:
        body = cylinder(f"{name}_body", (0, 0, 0), bx / 2, bz, "Z", 24)
        finish(body, "body", col, bevel=0.015, smooth=True)
        cap = uv_sphere(f"{name}_cap", (0, 0, bz / 2), (bx * 0.4, bx * 0.4, bz * 0.6))
        finish(cap, "body", col, smooth=True)
    else:
        finish(box(f"{name}_body", (0, 0, 0), body_size), "body", col, bevel=min(bx, by, bz) * 0.3, segments=3)
    offset = math.pi / 4 if arms == 4 else 0.0
    for k in range(arms):
        a = offset + k * 2 * math.pi / arms
        tip = (arm_len * math.cos(a), arm_len * math.sin(a), 0.0)
        arm = bar(f"{name}_arm{k}", (0, 0), tip[:2], max(0.018, arm_len * 0.09), max(0.012, bz * 0.3))
        finish(arm, "frame", col, bevel=0.004)
        motor_h = max(0.02, prop_r * 0.25)
        motor = cylinder(f"{name}_motor{k}", (tip[0], tip[1], motor_h / 2), prop_r * 0.2, motor_h, "Z", 14)
        finish(motor, "motor", col, smooth=True)
        propeller(f"{name}_prop{k}", (tip[0], tip[1], motor_h + 0.004), prop_r, blades, col, a * 1.7)
    if legs == "skids":
        for side in (1, -1):
            skid = cylinder(f"{name}_skid{side}", (0, side * by * 0.9, -bz * 2.2), 0.008 + bx * 0.03, bx * 1.6, "X", 10)
            finish(skid, "frame", col, smooth=True)
            for sx in (1, -1):
                strut = cylinder(f"{name}_strut{side}{sx}", (sx * bx * 0.35, side * by * 0.7, -bz * 1.1),
                                 0.006 + bx * 0.02, bz * 2.2, "Z", 8)
                finish(strut, "frame", col)
    elif legs == "feet":
        for k in range(arms):
            a = offset + k * 2 * math.pi / arms
            foot = cylinder(f"{name}_foot{k}", (arm_len * 0.8 * math.cos(a), arm_len * 0.8 * math.sin(a), -bz * 0.9),
                            0.007, bz * 1.4, "Z", 8)
            finish(foot, "frame", col)
    if gimbal:
        cam = uv_sphere(f"{name}_gimbal", (bx * 0.35, 0, -bz * 0.8), (bz * 0.45, bz * 0.45, bz * 0.45))
        finish(cam, "gimbal", col, smooth=True)


def build_drones():
    # Consumer camera quad (DJI-style)
    drone("drone_quad", 4, 0.18, (0.22, 0.11, 0.07), 0.11, 2, "feet", True)
    # Industrial hexacopter with landing skids
    drone("drone_hexa", 6, 0.42, (0.26, 0.26, 0.1), 0.17, 2, "skids", True, round_body=True)
    # FPV racing quad
    drone("drone_fpv", 4, 0.12, (0.1, 0.05, 0.035), 0.065, 3, None, False)


# ----------------------------------------------------------------------- export
def export_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    manifest = {}
    for variant, object_class in VARIANTS.items():
        parts = {}  # role -> (vertices, normals, faces)
        lo = Vector((1e9, 1e9, 1e9))
        hi = Vector((-1e9, -1e9, -1e9))
        for ob in bpy.data.collections[variant].objects:
            evaluated = ob.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            matrix = ob.matrix_world
            normal_matrix = matrix.to_3x3().inverted().transposed()
            for tri in mesh.loop_triangles:
                role = mesh.materials[tri.material_index].name
                vertices, normals, faces = parts.setdefault(role, ([], [], []))
                face = []
                for loop_index, vertex_index in zip(tri.loops, tri.vertices):
                    co = matrix @ mesh.vertices[vertex_index].co
                    normal = (normal_matrix @ mesh.corner_normals[loop_index].vector).normalized()
                    vertices.append(co)
                    normals.append(normal)
                    face.append(len(vertices))
                    for axis in range(3):
                        lo[axis] = min(lo[axis], co[axis])
                        hi[axis] = max(hi[axis], co[axis])
                faces.append(face)
            evaluated.to_mesh_clear()

        entries = []
        for role, (vertices, normals, faces) in sorted(parts.items()):
            filename = f"{variant}_{role}.obj"
            with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as obj:
                obj.write(f"# {variant} {role}\n")
                obj.writelines(f"v {v.x:.5f} {v.y:.5f} {v.z:.5f}\n" for v in vertices)
                obj.writelines(f"vn {n.x:.4f} {n.y:.4f} {n.z:.4f}\n" for n in normals)
                obj.writelines(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in faces)
            entries.append({"file": filename, "role": role, "rgba": list(MATS[role].diffuse_color),
                            "triangles": len(faces)})
        manifest[variant] = {"class": object_class, "parts": entries,
                             "bounds": [list(lo), list(hi)]}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def layout_for_preview():
    rows = {"vehicle": 0.0, "person": -5.0, "drone": -8.0}
    counters = {}
    for variant, object_class in VARIANTS.items():
        i = counters.get(object_class, 0)
        counters[object_class] = i + 1
        spacing = {"vehicle": 3.5, "person": 1.2, "drone": 1.2}[object_class]
        z = 1.0 if object_class == "drone" else 0.0
        for ob in bpy.data.collections[variant].objects:
            ob.location += Vector((rows[object_class], i * spacing, z))


def main():
    for ob in list(bpy.data.objects):
        if ob.type == "MESH":
            bpy.data.objects.remove(ob, do_unlink=True)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # Default colours; role colours are randomised in the simulator
    for name, rgba in {
        "paint": (0.55, 0.05, 0.05, 1), "glass": (0.08, 0.11, 0.14, 1), "tire": (0.04, 0.04, 0.04, 1),
        "rim": (0.6, 0.6, 0.62, 1), "headlight": (1.0, 0.96, 0.82, 1), "taillight": (0.75, 0.04, 0.04, 1),
        "trim": (0.07, 0.07, 0.08, 1), "skin": (0.78, 0.57, 0.42, 1), "shirt": (0.18, 0.33, 0.68, 1),
        "pants": (0.13, 0.14, 0.2, 1), "shoes": (0.06, 0.06, 0.06, 1), "hair": (0.1, 0.07, 0.05, 1),
        "body": (0.85, 0.86, 0.88, 1), "frame": (0.15, 0.15, 0.16, 1), "motor": (0.05, 0.05, 0.06, 1),
        "props": (0.12, 0.12, 0.13, 1), "gimbal": (0.03, 0.03, 0.03, 1),
    }.items():
        mat(name, rgba)

    build_vehicles()
    build_people()
    build_drones()
    manifest = export_all()
    layout_for_preview()
    for variant, info in manifest.items():
        triangles = sum(p["triangles"] for p in info["parts"])
        print(f"{variant:<16} {info['class']:<8} parts={len(info['parts'])} triangles={triangles}")
    print(f"Exported to {OUT_DIR}")


main()
