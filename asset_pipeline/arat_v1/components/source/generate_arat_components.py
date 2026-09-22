"""Generate the ARAT component Blender source, meshes, colliders, and URDFs.

Run ``prepare_specs.py`` first, then execute this file with Blender:

    blender --background --python generate_arat_components.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Vector


INCH = 0.0254
CM = 0.01
MM = 0.001
UNIT_SCALE = {"inch": INCH, "centimetre": CM, "millimetre": MM}
ROOT = Path(__file__).resolve().parents[1]
RESOLVED_SPECS = Path(__file__).resolve().parent / "resolved_specs.json"
TEXTURE_PATH = ROOT / "textures" / "wood_basecolor.png"
MODELS_DIR = ROOT / "models"
BLEND_PATH = ROOT / "arat_components.blend"
PREVIEW_PATH = ROOT / "previews" / "arat_components_contact_sheet.png"
MANIFEST_PATH = ROOT / "validation" / "asset_manifest.json"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def make_collection(name: str, parent: bpy.types.Collection | None = None, hide_render: bool = False):
    collection = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(collection)
    collection.hide_render = hide_render
    return collection


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def make_material(name: str, rgba, metallic: float, roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    material.diffuse_color = rgba
    return material


def make_wood_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ARAT_light_maple")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(TEXTURE_PATH), check_existing=True)
    texture.extension = "REPEAT"
    texture.interpolation = "Linear"
    shader.inputs["Roughness"].default_value = 0.48
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.diffuse_color = (0.72, 0.49, 0.25, 1.0)
    return material


def apply_bevel(obj: bpy.types.Object, width: float, segments: int = 2) -> None:
    modifier = obj.modifiers.new("small_manufactured_edge", "BEVEL")
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def unwrap(obj: bpy.types.Object, cube: bool = False) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if cube:
        bpy.ops.uv.cube_project(cube_size=0.08, correct_aspect=True)
    else:
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def tag(obj: bpy.types.Object, model: str, role: str, component: str) -> None:
    obj["arat_model"] = model
    obj["arat_role"] = role
    obj["component"] = component
    if role == "collision":
        obj.display_type = "WIRE"
        obj.hide_render = True


def parent_local(obj: bpy.types.Object, root: bpy.types.Object, location) -> None:
    obj.parent = root
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = location


def add_box(
    name,
    model,
    dimensions,
    root,
    collection,
    material,
    *,
    location=(0.0, 0.0, 0.0),
    role="visual",
    bevel=True,
):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = name
    move_to_collection(obj, collection)
    parent_local(obj, root, (location[0], location[1], location[2] + dimensions[2] / 2.0))
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.data.materials.append(material)
    if bevel:
        apply_bevel(obj, min(0.0015, min(dimensions) * 0.08), segments=3)
    if role == "visual":
        unwrap(obj, cube=True)
    tag(obj, model, role, name)
    return obj


def add_cylinder(
    name,
    model,
    radius,
    height,
    root,
    collection,
    material,
    *,
    location=(0.0, 0.0, 0.0),
    role="visual",
    vertices=48,
    bevel=True,
):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=height)
    obj = bpy.context.object
    obj.name = name
    move_to_collection(obj, collection)
    parent_local(obj, root, (location[0], location[1], location[2] + height / 2.0))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.data.materials.append(material)
    if bevel:
        apply_bevel(obj, min(0.001, radius * 0.08, height * 0.08), segments=2)
    if role == "visual":
        unwrap(obj)
    tag(obj, model, role, name)
    return obj


def add_sphere(name, model, radius, root, collection, material, *, role="visual"):
    if role == "visual":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
    obj = bpy.context.object
    if role != "visual":
        # Blender's icosphere has vertices at the nominal radius, but its
        # axis-aligned extents are not identical on X/Y/Z. Normalize the
        # collider so the physical diameter matches the specification.
        obj.dimensions = (2.0 * radius, 2.0 * radius, 2.0 * radius)
    obj.name = name
    move_to_collection(obj, collection)
    parent_local(obj, root, (0.0, 0.0, radius))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.data.materials.append(material)
    if role == "visual":
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        unwrap(obj)
    tag(obj, model, role, name)
    return obj


def annulus_mesh(name: str, outer_radius: float, inner_radius: float, height: float, segments: int = 64):
    vertices = []
    for z in (0.0, height):
        for radius in (outer_radius, inner_radius):
            for index in range(segments):
                angle = 2.0 * math.pi * index / segments
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))

    ob0 = 0
    ib0 = segments
    ot0 = 2 * segments
    it0 = 3 * segments
    faces = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            [
                (ob0 + index, ob0 + nxt, ot0 + nxt, ot0 + index),
                (ib0 + nxt, ib0 + index, it0 + index, it0 + nxt),
                (ot0 + index, ot0 + nxt, it0 + nxt, it0 + index),
                (ob0 + nxt, ob0 + index, ib0 + index, ib0 + nxt),
            ]
        )
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def tapered_annular_wedge_mesh(
    name,
    inner_bottom_radius,
    inner_top_radius,
    outer_bottom_radius,
    outer_top_radius,
    height,
    angle_start,
    angle_end,
):
    def points_2d(inner_radius, outer_radius):
        return [
            (inner_radius * math.cos(angle_start), inner_radius * math.sin(angle_start)),
            (outer_radius * math.cos(angle_start), outer_radius * math.sin(angle_start)),
            (outer_radius * math.cos(angle_end), outer_radius * math.sin(angle_end)),
            (inner_radius * math.cos(angle_end), inner_radius * math.sin(angle_end)),
        ]

    vertices = [(x, y, 0.0) for x, y in points_2d(inner_bottom_radius, outer_bottom_radius)]
    vertices.extend((x, y, height) for x, y in points_2d(inner_top_radius, outer_top_radius))
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def annular_wedge_mesh(name, inner_radius, outer_radius, height, angle_start, angle_end):
    return tapered_annular_wedge_mesh(
        name,
        inner_radius,
        inner_radius,
        outer_radius,
        outer_radius,
        height,
        angle_start,
        angle_end,
    )


def cup_mesh(
    name: str,
    outer_bottom_radius: float,
    inner_bottom_radius: float,
    height: float,
    base_height: float,
    segments=64,
    *,
    outer_top_radius: float | None = None,
    inner_top_radius: float | None = None,
):
    outer_top_radius = outer_bottom_radius if outer_top_radius is None else outer_top_radius
    inner_top_radius = inner_bottom_radius if inner_top_radius is None else inner_top_radius
    vertices = []
    for radius, z in (
        (outer_bottom_radius, 0.0),
        (outer_top_radius, height),
        (inner_top_radius, height),
        (inner_bottom_radius, base_height),
    ):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    floor_center = len(vertices)
    vertices.append((0.0, 0.0, base_height))
    outer_bottom = 0
    outer_top = segments
    inner_top = 2 * segments
    inner_bottom = 3 * segments
    faces = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            [
                (outer_bottom + index, outer_bottom + nxt, outer_top + nxt, outer_top + index),
                (outer_top + index, outer_top + nxt, inner_top + nxt, inner_top + index),
                (inner_top + nxt, inner_top + index, inner_bottom + index, inner_bottom + nxt),
                (outer_bottom + nxt, outer_bottom + index, bottom_center),
                (inner_bottom + index, inner_bottom + nxt, floor_center),
            ]
        )
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def add_custom_mesh(name, model, mesh, root, collection, material, *, role="visual", bevel=False):
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    parent_local(obj, root, (0.0, 0.0, 0.0))
    obj.data.materials.append(material)
    if bevel:
        apply_bevel(obj, 0.0007, segments=2)
    if role == "visual":
        unwrap(obj)
    tag(obj, model, role, name)
    return obj


def export_visual(model: str, root: bpy.types.Object, sources: list[bpy.types.Object]) -> Path:
    mesh_dir = MODELS_DIR / model / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    temp = make_collection(f"__EXPORT_VISUAL_{model}")
    duplicates = []
    root_inverse = root.matrix_world.inverted()
    for source in sources:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        duplicate.parent = None
        duplicate.matrix_world = root_inverse @ source.matrix_world
        temp.objects.link(duplicate)
        duplicates.append(duplicate)
    bpy.ops.object.select_all(action="DESELECT")
    for duplicate in duplicates:
        duplicate.select_set(True)
    bpy.context.view_layer.objects.active = duplicates[0]
    bpy.ops.object.join()
    combined = bpy.context.object
    combined.name = model
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    output = mesh_dir / "visual.obj"
    bpy.ops.wm.obj_export(
        filepath=str(output),
        check_existing=False,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_uv=True,
        export_normals=True,
        export_materials=True,
        export_pbr_extensions=True,
        path_mode="COPY",
        export_triangulated_mesh=True,
    )
    bpy.data.objects.remove(combined, do_unlink=True)
    bpy.data.collections.remove(temp)
    return output


def export_collision(model: str, root: bpy.types.Object, source: bpy.types.Object, collision_name: str) -> Path:
    collision_dir = MODELS_DIR / model / "meshes" / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)
    temp = make_collection(f"__EXPORT_COLLISION_{model}_{collision_name}")
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.parent = None
    duplicate.matrix_world = root.matrix_world.inverted() @ source.matrix_world
    temp.objects.link(duplicate)
    bpy.ops.object.select_all(action="DESELECT")
    duplicate.select_set(True)
    bpy.context.view_layer.objects.active = duplicate
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    output = collision_dir / f"{collision_name}.obj"
    bpy.ops.wm.obj_export(
        filepath=str(output),
        check_existing=False,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_uv=False,
        export_normals=False,
        export_materials=False,
        export_triangulated_mesh=True,
    )
    bpy.data.objects.remove(duplicate, do_unlink=True)
    bpy.data.collections.remove(temp)
    return output


def export_fillable_volume(
    model: str, root: bpy.types.Object, source: bpy.types.Object, local_z_offset: float
) -> Path:
    """Export a fillable meta-link mesh in the meta link's local frame."""
    mesh_dir = MODELS_DIR / model / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    temp = make_collection(f"__EXPORT_FILLABLE_{model}")
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.parent = None
    duplicate.matrix_world = (
        Matrix.Translation((0.0, 0.0, -local_z_offset)) @ root.matrix_world.inverted() @ source.matrix_world
    )
    temp.objects.link(duplicate)
    bpy.ops.object.select_all(action="DESELECT")
    duplicate.select_set(True)
    bpy.context.view_layer.objects.active = duplicate
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    output = mesh_dir / "fillable_volume.obj"
    bpy.ops.wm.obj_export(
        filepath=str(output),
        check_existing=False,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_uv=False,
        export_normals=False,
        export_materials=False,
        export_triangulated_mesh=True,
    )
    bpy.data.objects.remove(duplicate, do_unlink=True)
    bpy.data.collections.remove(temp)
    return output


def set_origin(node: ET.Element, xyz=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(node, "origin", xyz=" ".join(f"{value:.9g}" for value in xyz), rpy="0 0 0")


def write_urdf(model: str, mass: float, bbox, collision_names: list[str], fillable=None) -> Path:
    robot = ET.Element("robot", name=model)
    link = ET.SubElement(robot, "link", name="base_link")
    sx, sy, sz = bbox
    inertial = ET.SubElement(link, "inertial")
    set_origin(inertial, (0.0, 0.0, sz / 2.0))
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{mass * (sy * sy + sz * sz) / 12.0:.9g}",
        ixy="0",
        ixz="0",
        iyy=f"{mass * (sx * sx + sz * sz) / 12.0:.9g}",
        iyz="0",
        izz=f"{mass * (sx * sx + sy * sy) / 12.0:.9g}",
    )
    visual = ET.SubElement(link, "visual", name="visual")
    set_origin(visual)
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geometry, "mesh", filename="meshes/visual.obj", scale="1 1 1")
    for name in collision_names:
        collision = ET.SubElement(link, "collision", name=name)
        set_origin(collision)
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", filename=f"meshes/collision/{name}.obj", scale="1 1 1")
    if fillable is not None:
        meta_name = "meta__base_link_fillable_0_0_link"
        meta_link = ET.SubElement(robot, "link", name=meta_name)
        meta_inertial = ET.SubElement(meta_link, "inertial")
        set_origin(meta_inertial, (0.0, 0.0, fillable["height"] / 2.0))
        ET.SubElement(meta_inertial, "mass", value="0.0001")
        ET.SubElement(
            meta_inertial,
            "inertia",
            ixx="1e-8",
            ixy="0",
            ixz="0",
            iyy="1e-8",
            iyz="0",
            izz="1e-8",
        )
        meta_visual = ET.SubElement(meta_link, "visual", name="fillable_volume")
        set_origin(meta_visual)
        meta_geometry = ET.SubElement(meta_visual, "geometry")
        ET.SubElement(meta_geometry, "mesh", filename="meshes/fillable_volume.obj", scale="1 1 1")
        meta_joint = ET.SubElement(robot, "joint", name="base_link_to_fillable", type="fixed")
        ET.SubElement(meta_joint, "parent", link="base_link")
        ET.SubElement(meta_joint, "child", link=meta_name)
        set_origin(meta_joint, (0.0, 0.0, fillable["z_min"]))
    text = xml.dom.minidom.parseString(ET.tostring(robot, encoding="unicode")).toprettyxml(indent="  ")
    output = MODELS_DIR / model / f"{model}.urdf"
    output.write_text(text, encoding="utf-8")
    return output


def look_at(obj: bpy.types.Object, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    if not RESOLVED_SPECS.is_file() or not TEXTURE_PATH.is_file():
        raise FileNotFoundError("Run source/prepare_specs.py before Blender generation")
    specs = json.loads(RESOLVED_SPECS.read_text(encoding="utf-8"))
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    bpy.context.preferences.filepaths.save_version = 0

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 850
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(PREVIEW_PATH)
    scene.world.color = (0.025, 0.028, 0.035)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.7

    visual_parent = make_collection("VISUAL")
    collision_parent = make_collection("COLLISION", hide_render=True)
    fillable_parent = make_collection("FILLABLE", hide_render=True)
    roots_collection = make_collection("MODEL_ROOTS")
    preview_collection = make_collection("PREVIEW")

    wood = make_wood_material()
    material_specs = specs["materials"]
    materials = {"light_maple_wood": wood}
    for key in (
        "brushed_tin",
        "crimson_red",
        "dark_steel",
        "sharpening_stone",
        "turquoise_marble",
        "cup_blue",
        "cup_red",
    ):
        spec = material_specs[key]
        materials[key] = make_material(
            f"ARAT_{key}", tuple(spec["base_color_rgba"]), float(spec["metallic"]), float(spec["roughness"])
        )
    collision_material = make_material("ARAT_collision_debug", (0.8, 0.02, 0.02, 0.18), 0.0, 0.8)
    collision_material.surface_render_method = "DITHERED"
    floor_material = make_material("ARAT_preview_floor", (0.07, 0.08, 0.095, 1.0), 0.0, 0.65)

    records = []
    roots = {}

    def begin_model(model: str):
        index = len(records)
        root = bpy.data.objects.new(f"ROOT_{model}", None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.04
        # Preview placement only; exported meshes are transformed back into
        # their model-local frames.
        root.location = (-0.72 + (index % 6) * 0.29, 0.50 - (index // 6) * 0.34, 0.0)
        root["arat_model"] = model
        roots_collection.objects.link(root)
        roots[model] = root
        visual_collection = make_collection(model, visual_parent)
        collision_collection = make_collection(model, collision_parent, hide_render=True)
        return root, visual_collection, collision_collection

    def finish_model(
        model,
        category,
        mass,
        bbox,
        visual_objects,
        collision_objects,
        source_unit,
        fillable=None,
        fillable_source=None,
    ):
        root = roots[model]
        export_visual(model, root, visual_objects)
        collision_names = []
        for index, collision in enumerate(collision_objects):
            name = collision.get("collision_name") or f"collision_{index:02d}"
            export_collision(model, root, collision, name)
            collision_names.append(name)
        if fillable is not None:
            if fillable_source is None:
                raise ValueError(f"A fillable mesh source is required for {model}")
            export_fillable_volume(model, root, fillable_source, fillable["z_min"])
        write_urdf(model, mass, bbox, collision_names, fillable=fillable)
        record = {
            "category": category,
            "model": model,
            "source_unit": source_unit,
            "bbox_metres": list(bbox),
            "mass_kg": mass,
            "visual_object_count": len(visual_objects),
            "collision_mesh_count": len(collision_objects),
            "collision_names": collision_names,
            "urdf": str(MODELS_DIR / model / f"{model}.urdf"),
        }
        if fillable is not None:
            record["fillable_volume_metres"] = fillable
        records.append(record)

    block_spec = specs["assets"]["wooden_blocks"]
    block_scale = UNIT_SCALE[block_spec["unit"]]
    for block in block_spec["models"]:
        model = block["model"]
        edge = float(block["edge"]) * block_scale
        root, vis_col, col_col = begin_model(model)
        visual = [add_box("wooden_block", model, (edge, edge, edge), root, vis_col, wood)]
        collision = [
            add_box(
                "COL_box",
                model,
                (edge, edge, edge),
                root,
                col_col,
                collision_material,
                role="collision",
                bevel=False,
            )
        ]
        collision[0]["collision_name"] = "box"
        finish_model(
            model,
            block_spec["category"],
            float(block["mass_grams"]) / 1000.0,
            (edge, edge, edge),
            visual,
            collision,
            block_spec["unit"],
        )

    cricket_ball_spec = specs["assets"]["cricket_ball"]
    model = cricket_ball_spec["model"]
    diameter = float(cricket_ball_spec["diameter"]) * UNIT_SCALE[cricket_ball_spec["unit"]]
    root, vis_col, col_col = begin_model(model)
    visual = [
        add_sphere(
            "cricket_ball", model, diameter / 2.0, root, vis_col, materials[cricket_ball_spec["material"]]
        )
    ]
    collision = [add_sphere("COL_sphere", model, diameter / 2.0, root, col_col, collision_material, role="collision")]
    collision[0]["collision_name"] = "sphere"
    finish_model(
        model,
        cricket_ball_spec["category"],
        float(cricket_ball_spec["mass_grams"]) / 1000.0,
        (diameter,) * 3,
        visual,
        collision,
        cricket_ball_spec["unit"],
    )

    tin = specs["assets"]["tin_lid"]
    tin_segments = int(tin["collision_segments"])
    tin_scale = UNIT_SCALE[tin["unit"]]
    model = tin["model"]
    outer_diameter = float(tin["outside_diameter"]) * tin_scale
    inner_diameter = float(tin["inside_diameter"]) * tin_scale
    height = float(tin["rim_height"]) * tin_scale
    base_height = float(tin["floor_thickness"]) * tin_scale
    outer_radius = outer_diameter / 2.0
    inner_radius = inner_diameter / 2.0
    root, vis_col, col_col = begin_model(model)
    mesh = cup_mesh(f"{model}_visual_mesh", outer_radius, inner_radius, height, base_height)
    visual = [add_custom_mesh("tin_lid", model, mesh, root, vis_col, materials["brushed_tin"])]
    floor = add_cylinder(
        "COL_floor",
        model,
        outer_radius,
        base_height,
        root,
        col_col,
        collision_material,
        role="collision",
        vertices=64,
        bevel=False,
    )
    floor["collision_name"] = "floor"
    collision = [floor]
    for index in range(tin_segments):
        start = 2.0 * math.pi * index / tin_segments
        end = 2.0 * math.pi * (index + 1) / tin_segments
        wedge_mesh = annular_wedge_mesh(f"{model}_wall_{index:02d}", inner_radius, outer_radius, height, start, end)
        wedge = add_custom_mesh(
            f"COL_wall_{index:02d}", model, wedge_mesh, root, col_col, collision_material, role="collision"
        )
        wedge["collision_name"] = f"wall_{index:02d}"
        collision.append(wedge)
    material_volume = math.pi * outer_radius**2 * height - math.pi * inner_radius**2 * (height - base_height)
    finish_model(
        model,
        tin["category"],
        7850.0 * material_volume,
        (outer_diameter, outer_diameter, height),
        visual,
        collision,
        tin["unit"],
    )

    stone = specs["assets"]["sharpening_stone"]
    model = stone["model"]
    stone_dims = tuple(float(value) * UNIT_SCALE[stone["unit"]] for value in stone["dimensions_xyz"])
    root, vis_col, col_col = begin_model(model)
    visual = [add_box("sharpening_stone", model, stone_dims, root, vis_col, materials["sharpening_stone"])]
    collision = [
        add_box(
            "COL_stone",
            model,
            stone_dims,
            root,
            col_col,
            collision_material,
            role="collision",
            bevel=False,
        )
    ]
    collision[0]["collision_name"] = "stone"
    finish_model(
        model,
        stone["category"],
        float(stone["mass_grams"]) / 1000.0,
        stone_dims,
        visual,
        collision,
        stone["unit"],
    )

    for asset_key in ("marble", "ball_bearing"):
        ball = specs["assets"][asset_key]
        model = ball["model"]
        diameter = float(ball["diameter"]) * UNIT_SCALE[ball["unit"]]
        root, vis_col, col_col = begin_model(model)
        visual = [add_sphere(asset_key, model, diameter / 2.0, root, vis_col, materials[ball["material"]])]
        collision = [
            add_sphere(
                f"COL_{asset_key}",
                model,
                diameter / 2.0,
                root,
                col_col,
                collision_material,
                role="collision",
            )
        ]
        collision[0]["collision_name"] = "sphere"
        mass = float(ball["mass_grams"]) / 1000.0
        finish_model(model, ball["category"], mass, (diameter,) * 3, visual, collision, ball["unit"])

    fixtures = []
    wood_density = float(specs["assumptions"]["wood_density_kg_m3"])
    for fixture_key in ("tube_starting_fixture", "tube_target_fixture", "washer_target_fixture"):
        fixture = specs["assets"][fixture_key]
        fixtures.append(fixture)
        scale = UNIT_SCALE[fixture["unit"]]
        model = fixture["model"]
        base_dims = tuple(float(value) * scale for value in fixture["plank_dimensions_xyz"])
        root, vis_col, col_col = begin_model(model)
        visual = [add_box("wood_plank", model, base_dims, root, vis_col, wood)]
        collision = [
            add_box(
                "COL_plank", model, base_dims, root, col_col, collision_material, role="collision", bevel=False
            )
        ]
        collision[0]["collision_name"] = "plank"
        max_height = 0.0
        peg_mass = 0.0
        for peg in fixture["pegs"]:
            peg_id = peg["id"]
            radius = float(peg["diameter"]) * scale / 2.0
            height = float(peg["height"]) * scale
            x = float(peg["x"]) * scale
            y = float(peg["y"]) * scale
            visual.append(
                add_cylinder(
                    f"{peg_id}_peg", model, radius, height, root, vis_col, wood, location=(x, y, base_dims[2])
                )
            )
            collider = add_cylinder(
                f"COL_{peg_id}_peg",
                model,
                radius,
                height,
                root,
                col_col,
                collision_material,
                location=(x, y, base_dims[2]),
                role="collision",
                vertices=32,
                bevel=False,
            )
            collider["collision_name"] = f"{peg_id}_peg"
            collision.append(collider)
            max_height = max(max_height, height)
            peg_mass += wood_density * math.pi * radius * radius * height
        base_mass = wood_density * math.prod(base_dims)
        fixture_bbox = (base_dims[0], base_dims[1], base_dims[2] + max_height)
        finish_model(
            model,
            fixture["category"],
            base_mass + peg_mass,
            fixture_bbox,
            visual,
            collision,
            fixture["unit"],
        )

    components = specs["assets"]["fixture_components"]
    component_scale = UNIT_SCALE[components["unit"]]
    for plank in components["planks"]:
        model = plank["model"]
        dimensions = tuple(float(value) * component_scale for value in plank["dimensions_xyz"])
        root, vis_col, col_col = begin_model(model)
        visual = [add_box("wood_plank", model, dimensions, root, vis_col, wood)]
        collider = add_box(
            "COL_plank",
            model,
            dimensions,
            root,
            col_col,
            collision_material,
            role="collision",
            bevel=False,
        )
        collider["collision_name"] = "plank"
        finish_model(
            model,
            components["plank_category"],
            wood_density * math.prod(dimensions),
            dimensions,
            visual,
            [collider],
            components["unit"],
        )
    for bolt in components["bolts"]:
        model = bolt["model"]
        diameter = float(bolt["diameter"]) * component_scale
        height = float(bolt["height"]) * component_scale
        radius = diameter / 2.0
        root, vis_col, col_col = begin_model(model)
        visual = [add_cylinder("wood_bolt", model, radius, height, root, vis_col, wood)]
        collider = add_cylinder(
            "COL_bolt",
            model,
            radius,
            height,
            root,
            col_col,
            collision_material,
            role="collision",
            vertices=32,
            bevel=False,
        )
        collider["collision_name"] = "bolt"
        finish_model(
            model,
            components["bolt_category"],
            wood_density * math.pi * radius * radius * height,
            (diameter, diameter, height),
            visual,
            [collider],
            components["unit"],
        )

    washer = specs["assets"]["washer"]
    model = washer["model"]
    washer_scale = UNIT_SCALE[washer["unit"]]
    outer_radius = float(washer["outside_diameter"]) * washer_scale / 2.0
    inner_radius = float(washer["inside_diameter"]) * washer_scale / 2.0
    washer_height = float(washer["thickness"]) * washer_scale
    root, vis_col, col_col = begin_model(model)
    visual_mesh = annulus_mesh("washer_visual_mesh", outer_radius, inner_radius, washer_height)
    visual = [add_custom_mesh("washer", model, visual_mesh, root, vis_col, materials[washer["material"]])]
    collision = []
    for index in range(16):
        start = 2.0 * math.pi * index / 16.0
        end = 2.0 * math.pi * (index + 1) / 16.0
        wedge_mesh = annular_wedge_mesh(
            f"washer_wedge_{index:02d}", inner_radius, outer_radius, washer_height, start, end
        )
        wedge = add_custom_mesh(
            f"COL_wedge_{index:02d}", model, wedge_mesh, root, col_col, collision_material, role="collision"
        )
        wedge["collision_name"] = f"wedge_{index:02d}"
        collision.append(wedge)
    finish_model(
        model,
        washer["category"],
        float(washer["mass_grams"]) / 1000.0,
        (2 * outer_radius, 2 * outer_radius, washer_height),
        visual,
        collision,
        washer["unit"],
    )

    cups = specs["assets"]["cups"]
    cup_dims = cups["dimensions"]
    cup_scale = UNIT_SCALE[cups["unit"]]
    cup_outer_bottom = float(cup_dims["lower_outside_diameter"]) * cup_scale
    cup_outer_top = float(cup_dims["upper_outside_diameter"]) * cup_scale
    cup_inner_bottom = float(cup_dims["lower_inside_diameter"]) * cup_scale
    cup_inner_top = float(cup_dims["upper_inside_diameter"]) * cup_scale
    cup_height = float(cup_dims["height"]) * cup_scale
    cup_base = float(cup_dims["base_thickness"]) * cup_scale
    cup_segments = int(cups["collision_segments"])
    fillable_specs = cups["fillable"]
    cup_fillable = {
        "radius": float(fillable_specs["volume_diameter"]) * cup_scale / 2.0,
        "height": float(fillable_specs["volume_height"]) * cup_scale,
        "z_min": float(fillable_specs["z_min"]) * cup_scale,
    }
    for cup in cups["models"]:
        model = cup["model"]
        root, vis_col, col_col = begin_model(model)
        mesh = cup_mesh(
            f"{model}_visual_mesh",
            cup_outer_bottom / 2.0,
            cup_inner_bottom / 2.0,
            cup_height,
            cup_base,
            outer_top_radius=cup_outer_top / 2.0,
            inner_top_radius=cup_inner_top / 2.0,
        )
        visual = [add_custom_mesh("cup", model, mesh, root, vis_col, materials[cup["material"]])]
        floor = add_cylinder(
            "COL_floor",
            model,
            cup_outer_bottom / 2.0,
            cup_base,
            root,
            col_col,
            collision_material,
            role="collision",
            vertices=64,
            bevel=False,
        )
        floor["collision_name"] = "floor"
        collision = [floor]
        for index in range(cup_segments):
            start = 2.0 * math.pi * index / cup_segments
            end = 2.0 * math.pi * (index + 1) / cup_segments
            wedge_mesh = tapered_annular_wedge_mesh(
                f"{model}_wall_{index:02d}",
                cup_inner_bottom / 2.0,
                cup_inner_top / 2.0,
                cup_outer_bottom / 2.0,
                cup_outer_top / 2.0,
                cup_height,
                start,
                end,
            )
            wedge = add_custom_mesh(
                f"COL_wall_{index:02d}",
                model,
                wedge_mesh,
                root,
                col_col,
                collision_material,
                role="collision",
            )
            wedge["collision_name"] = f"wall_{index:02d}"
            collision.append(wedge)
        fillable_source = add_cylinder(
            "fillable_volume",
            model,
            cup_fillable["radius"],
            cup_fillable["height"],
            root,
            fillable_parent,
            collision_material,
            location=(0.0, 0.0, cup_fillable["z_min"]),
            role="fillable",
            vertices=64,
            bevel=False,
        )
        fillable_source.display_type = "WIRE"
        fillable_source.hide_render = True
        finish_model(
            model,
            cups["category"],
            float(cups["mass_grams"]) / 1000.0,
            (cup_outer_top, cup_outer_top, cup_height),
            visual,
            collision,
            cups["unit"],
            fillable=cup_fillable,
            fillable_source=fillable_source,
        )

    tubes = specs["assets"]["alloy_tubes"]
    tube_scale = UNIT_SCALE[tubes["unit"]]
    tube_segments = int(tubes["collision_segments"])
    for tube in tubes["models"]:
        model = tube["model"]
        tube_height = float(tube["height"]) * tube_scale
        inner_radius = float(tube["inside_diameter"]) * tube_scale / 2.0
        outer_radius = float(tube["outside_diameter"]) * tube_scale / 2.0
        root, vis_col, col_col = begin_model(model)
        visual_mesh = annulus_mesh(f"{model}_visual_mesh", outer_radius, inner_radius, tube_height)
        visual = [add_custom_mesh("alloy_tube", model, visual_mesh, root, vis_col, materials[tubes["material"]])]
        collision = []
        axial_segments = int(tube["collision_axial_segments"])
        segment_height = tube_height / axial_segments
        for axial_index in range(axial_segments):
            for radial_index in range(tube_segments):
                start = 2.0 * math.pi * radial_index / tube_segments
                end = 2.0 * math.pi * (radial_index + 1) / tube_segments
                collision_name = (
                    f"wall_{radial_index:02d}"
                    if axial_segments == 1
                    else f"wall_{axial_index:02d}_{radial_index:02d}"
                )
                wedge_mesh = annular_wedge_mesh(
                    f"{model}_{collision_name}", inner_radius, outer_radius, segment_height, start, end
                )
                wedge = add_custom_mesh(
                    f"COL_{collision_name}",
                    model,
                    wedge_mesh,
                    root,
                    col_col,
                    collision_material,
                    role="collision",
                )
                wedge.location.z = axial_index * segment_height
                wedge["collision_name"] = collision_name
                collision.append(wedge)
        finish_model(
            model,
            tubes["category"],
            float(tube["mass_grams"]) / 1000.0,
            (2.0 * outer_radius, 2.0 * outer_radius, tube_height),
            visual,
            collision,
            tubes["unit"],
        )

    # Standardized 75 cm high, 76 cm wide, 49 cm deep ARAT table. The table
    # owns no room floor and is the only object stored in the base scene.
    table = specs["assets"]["table"]
    table_scale = UNIT_SCALE[table["unit"]]
    model = table["model"]
    top_dims = tuple(float(value) * table_scale for value in table["top_dimensions_xyz"])
    top_bottom_z = float(table["top_bottom_z"]) * table_scale
    leg_dims = tuple(float(value) * table_scale for value in table["leg_dimensions_xyz"])
    root, vis_col, col_col = begin_model(model)
    visual = [add_box("tabletop", model, top_dims, root, vis_col, wood, location=(0.0, 0.0, top_bottom_z))]
    collision = [
        add_box(
            "COL_tabletop",
            model,
            top_dims,
            root,
            col_col,
            collision_material,
            location=(0.0, 0.0, top_bottom_z),
            role="collision",
            bevel=False,
        )
    ]
    collision[0]["collision_name"] = "tabletop"
    for index, (x, y) in enumerate(table["leg_centers_xy"], start=1):
        location = (float(x) * table_scale, float(y) * table_scale, 0.0)
        visual.append(add_box(f"leg_{index}", model, leg_dims, root, vis_col, wood, location=location))
        collider = add_box(
            f"COL_leg_{index}",
            model,
            leg_dims,
            root,
            col_col,
            collision_material,
            location=location,
            role="collision",
            bevel=False,
        )
        collider["collision_name"] = f"leg_{index}"
        collision.append(collider)
    overall_dims = tuple(float(value) * table_scale for value in table["overall_dimensions_xyz"])
    table_mass = wood_density * (math.prod(top_dims) + 4 * math.prod(leg_dims))
    finish_model(model, table["category"], table_mass, overall_dims, visual, collision, table["unit"])

    # Purpose-built shelf. The top surface is exactly 37 cm above its root;
    # four posts leave the work area open and avoid introducing a toolbox.
    shelf = specs["assets"]["shelf"]
    shelf_scale = UNIT_SCALE[shelf["unit"]]
    model = shelf["model"]
    frame_dims = tuple(float(value) * shelf_scale for value in shelf["support_frame_dimensions_xyz"])
    plank_dims = tuple(float(value) * shelf_scale for value in shelf["top_plank_dimensions_xyz"])
    post_width = float(shelf["post_square"]) * shelf_scale
    post_height = frame_dims[2]
    root, vis_col, col_col = begin_model(model)
    visual = [
        add_box("top_plank", model, plank_dims, root, vis_col, wood, location=(0.0, 0.0, post_height))
    ]
    collision = [
        add_box(
            "COL_top_plank",
            model,
            plank_dims,
            root,
            col_col,
            collision_material,
            location=(0.0, 0.0, post_height),
            role="collision",
            bevel=False,
        )
    ]
    collision[0]["collision_name"] = "top_plank"
    inset_x = (frame_dims[0] - post_width) / 2.0
    inset_y = (frame_dims[1] - post_width) / 2.0
    post_dims = (post_width, post_width, post_height)
    post_centers = ((-inset_x, -inset_y), (-inset_x, inset_y), (inset_x, -inset_y), (inset_x, inset_y))
    for index, (x, y) in enumerate(post_centers, start=1):
        visual.append(add_box(f"post_{index}", model, post_dims, root, vis_col, wood, location=(x, y, 0.0)))
        collider = add_box(
            f"COL_post_{index}",
            model,
            post_dims,
            root,
            col_col,
            collision_material,
            location=(x, y, 0.0),
            role="collision",
            bevel=False,
        )
        collider["collision_name"] = f"post_{index}"
        collision.append(collider)
    shelf_bbox = (frame_dims[0], frame_dims[1], float(shelf["overall_height"]) * shelf_scale)
    shelf_mass = wood_density * (math.prod(plank_dims) + 4 * math.prod(post_dims))
    finish_model(model, shelf["category"], shelf_mass, shelf_bbox, visual, collision, shelf["unit"])

    bpy.ops.mesh.primitive_plane_add(size=2.2, location=(0.0, 0.0, -0.003))
    floor = bpy.context.object
    floor.name = "preview_floor_not_exported"
    move_to_collection(floor, preview_collection)
    floor.data.materials.append(floor_material)
    bpy.ops.object.camera_add(location=(1.25, -1.55, 1.12))
    camera = bpy.context.object
    camera.name = "ARAT_components_preview_camera"
    camera.data.lens = 58
    look_at(camera, (0.0, 0.02, 0.08))
    move_to_collection(camera, preview_collection)
    scene.camera = camera
    for name, location, energy, size in (
        ("key", (-0.6, -0.65, 1.45), 420.0, 0.9),
        ("fill", (1.1, -0.2, 0.9), 260.0, 0.8),
        ("rim", (0.2, 1.0, 1.2), 350.0, 0.7),
    ):
        data = bpy.data.lights.new(f"ARAT_components_{name}", "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(data.name, data)
        light.location = location
        look_at(light, (0.0, 0.0, 0.08))
        preview_collection.objects.link(light)

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "status": "generated",
                "asset_count": len(records),
                "source_specs": str(ROOT.parent / "arat_components_specs.yaml"),
                "source_units": ["centimetre", "millimetre"],
                "runtime_unit": "metre",
                "assets": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for image in bpy.data.images:
        if image.source == "FILE" and Path(bpy.path.abspath(image.filepath)).resolve() == TEXTURE_PATH.resolve():
            image.filepath = bpy.path.relpath(str(TEXTURE_PATH))
    bpy.ops.file.pack_all()
    scene["arat_spec_file"] = "../arat_components_specs.yaml"
    scene["source_linear_units"] = "centimetre except ball bearing in millimetres"
    scene["runtime_linear_unit"] = "metre"
    scene["asset_count"] = len(records)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    bpy.ops.render.render(write_still=True)
    print(f"ARAT_COMPONENT_BUILD_COMPLETE assets={len(records)} blend={BLEND_PATH} preview={PREVIEW_PATH}")


if __name__ == "__main__":
    main()
