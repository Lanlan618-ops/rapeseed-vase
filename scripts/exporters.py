"""GLB and OBJ exporters with the PBR extensions used by the asset."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

from geo import MeshObject, triangulate_face
from materials import MATERIALS, MaterialSpec

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT = 5126
UINT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def linear_rgb(rgb: Sequence[float]) -> list[float]:
    return [round(srgb_to_linear(float(x)), 6) for x in rgb]


def _align4(data: bytearray) -> None:
    while len(data) % 4:
        data.append(0)


def _material_document(spec: MaterialSpec) -> tuple[dict, set[str]]:
    used: set[str] = set()
    doc: dict = {
        "name": spec.name,
        "pbrMetallicRoughness": {
            "baseColorFactor": linear_rgb(spec.base) + [round(float(spec.opacity), 6)],
            "metallicFactor": float(spec.metallic),
            "roughnessFactor": float(spec.roughness),
        },
        "doubleSided": False,
    }
    if spec.opacity < 0.999:
        doc["alphaMode"] = "BLEND"
    extensions: dict = {}
    if spec.transmission > 0.001:
        extensions["KHR_materials_transmission"] = {
            "transmissionFactor": round(float(spec.transmission), 6)
        }
        extensions["KHR_materials_ior"] = {"ior": round(float(spec.ior), 6)}
        used.update({"KHR_materials_transmission", "KHR_materials_ior"})
        if spec.thickness > 0.0:
            extensions["KHR_materials_volume"] = {
                "thicknessFactor": round(float(spec.thickness), 6),
                "attenuationDistance": round(float(spec.attenuation_distance), 6),
                "attenuationColor": linear_rgb(spec.attenuation),
            }
            used.add("KHR_materials_volume")
    if spec.clearcoat > 0.001:
        extensions["KHR_materials_clearcoat"] = {
            "clearcoatFactor": round(float(spec.clearcoat), 6),
            "clearcoatRoughnessFactor": round(float(spec.clearcoat_roughness), 6),
        }
        used.add("KHR_materials_clearcoat")
    if extensions:
        doc["extensions"] = extensions
    if spec.extras:
        doc["extras"] = dict(spec.extras)
    return doc, used


class GlbBuilder:
    def __init__(self):
        self.doc: dict = {
            "asset": {
                "version": "2.0",
                "generator": "Rapeseed Vase procedural studio (Python / NumPy)",
                "copyright": "Generated for the user's hologram study, 2026",
            },
            "scene": 0,
            "scenes": [{"name": "RAPESEED_VASE", "nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [{"byteLength": 0}],
        }
        self.bin = bytearray()
        self.used_extensions: set[str] = set()
        self.material_indices: dict[str, int] = {}

    def add_material(self, spec: MaterialSpec) -> int:
        if spec.name in self.material_indices:
            return self.material_indices[spec.name]
        index = len(self.doc["materials"])
        mat, used = _material_document(spec)
        self.doc["materials"].append(mat)
        self.used_extensions.update(used)
        self.material_indices[spec.name] = index
        return index

    def _add_buffer_view(self, data: bytes, target: int | None = None) -> int:
        _align4(self.bin)
        byte_offset = len(self.bin)
        self.bin.extend(data)
        view = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(data),
        }
        if target is not None:
            view["target"] = target
        index = len(self.doc["bufferViews"])
        self.doc["bufferViews"].append(view)
        return index

    def _add_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        minmax: bool = False,
    ) -> int:
        array = np.ascontiguousarray(array)
        data = array.tobytes(order="C")
        view = self._add_buffer_view(data, target)
        accessor = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": accessor_type,
        }
        if minmax:
            accessor["min"] = [float(x) for x in np.min(array, axis=0)]
            accessor["max"] = [float(x) for x in np.max(array, axis=0)]
        index = len(self.doc["accessors"])
        self.doc["accessors"].append(accessor)
        return index

    def add_object(self, obj: MeshObject) -> int:
        pivot = np.asarray(obj.pivot, dtype=np.float64)
        primitives = []
        for part in obj.parts:
            verts_world, normals_world, uvs = part.arrays()
            if not len(verts_world):
                continue
            local = verts_world - pivot
            # triangulate the clean quad-dominant topology for glTF
            indices = []
            for face in part.faces:
                for tri in triangulate_face(face):
                    indices.extend(tri)
            indices = np.asarray(indices, dtype=np.uint32)
            pos_acc = self._add_accessor(local.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER, minmax=True)
            nrm_acc = self._add_accessor(normals_world.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER)
            uv_acc = self._add_accessor(uvs.astype(np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
            idx_acc = self._add_accessor(indices, UINT, "SCALAR", ELEMENT_ARRAY_BUFFER)
            material_spec = MATERIALS[part.material]
            mat_index = self.add_material(material_spec)
            primitives.append(
                {
                    "attributes": {
                        "POSITION": pos_acc,
                        "NORMAL": nrm_acc,
                        "TEXCOORD_0": uv_acc,
                    },
                    "indices": idx_acc,
                    "material": mat_index,
                    "mode": 4,
                }
            )
        if not primitives:
            raise ValueError(f"object {obj.name} has no geometry")
        mesh_index = len(self.doc["meshes"])
        mesh = {"name": obj.name, "primitives": primitives}
        if obj.extras:
            mesh["extras"] = obj.extras
        self.doc["meshes"].append(mesh)
        node_index = len(self.doc["nodes"])
        node = {
            "name": obj.name,
            "mesh": mesh_index,
            "translation": [round(float(v), 6) for v in pivot],
        }
        extras = dict(obj.extras)
        extras.update(
            {
                "category": obj.category,
                "triangles": obj.triangle_count,
                "pivot_m": [round(float(v), 5) for v in pivot],
            }
        )
        node["extras"] = extras
        self.doc["nodes"].append(node)
        return node_index

    def finalize(self, objects: Sequence[MeshObject], model_meta: dict):
        object_node_indices: dict[str, int] = {}
        for obj in objects:
            object_node_indices[obj.name] = self.add_object(obj)

        def node_children(predicate):
            return [
                object_node_indices[o.name]
                for o in objects
                if predicate(o)
            ]

        groups = {
            "BOUQUET_FLOWERS": node_children(lambda o: o.category == "flower"),
            "BOUQUET_STEMS": node_children(lambda o: o.category == "stem"),
            "BOUQUET_BUDS": node_children(lambda o: o.category == "bud"),
            "BOUQUET_FOLIAGE": node_children(lambda o: o.category == "leaf"),
            "VESSEL_AND_WATER": node_children(lambda o: o.category in ("vessel", "water")),
        }
        group_indices = {}
        for name, children in groups.items():
            index = len(self.doc["nodes"])
            self.doc["nodes"].append({"name": name, "children": children})
            group_indices[name] = index
        root_index = len(self.doc["nodes"])
        root_meta = dict(model_meta)
        root_meta["groups"] = {
            "flowers": len(groups["BOUQUET_FLOWERS"]),
            "stems": len(groups["BOUQUET_STEMS"]),
            "buds": len(groups["BOUQUET_BUDS"]),
            "leaves": len(groups["BOUQUET_FOLIAGE"]),
            "vessel_and_water": len(groups["VESSEL_AND_WATER"]),
        }
        self.doc["nodes"].append(
            {
                "name": "Rapeseed_Vase_Root",
                "children": list(group_indices.values()),
                "extras": root_meta,
            }
        )
        self.doc["scenes"][0]["nodes"] = [root_index]
        if self.used_extensions:
            self.doc["extensionsUsed"] = sorted(self.used_extensions)
        self.doc["buffers"][0]["byteLength"] = len(self.bin)
        return self.doc

    def write(self, path: Path):
        # pad JSON with spaces and binary with zeroes as required by glTF
        _align4(self.bin)
        self.doc["buffers"][0]["byteLength"] = len(self.bin)
        if self.used_extensions:
            self.doc["extensionsUsed"] = sorted(self.used_extensions)
        json_bytes = json.dumps(self.doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4:
            json_bytes += b" "
        bin_bytes = bytes(self.bin)
        while len(bin_bytes) % 4:
            bin_bytes += b"\x00"
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
            fh.write(struct.pack("<II", len(json_bytes), CHUNK_JSON))
            fh.write(json_bytes)
            fh.write(struct.pack("<II", len(bin_bytes), CHUNK_BIN))
            fh.write(bin_bytes)
        return path


def _obj_number(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def write_obj(
    objects: Sequence[MeshObject],
    obj_path: Path,
    mtl_path: Path,
    used_materials: Sequence[str],
):
    obj_path = Path(obj_path)
    mtl_path = Path(mtl_path)
    with open(obj_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Rapeseed bouquet in a straight clear glass vase\n")
        fh.write("# Units: metres. Y up. Origin: centre of the vase base, floor at y=0.\n")
        fh.write("# High-detail quad-dominant topology. Glass/water transmission is GLB-only.\n")
        fh.write(f"mtllib {mtl_path.name}\n")
        fh.write(f"# {len(objects)} objects\n")
        v_total = 0
        vt_total = 0
        vn_total = 0
        for obj in objects:
            fh.write(f"\no {obj.name}\n")
            for pi, part in enumerate(obj.parts):
                verts, normals, uvs = part.arrays()
                if not len(verts):
                    continue
                fh.write(f"g {obj.name}__{part.material}\n")
                fh.write(f"usemtl {part.material}\n")
                fh.write("s 1\n")
                for v in verts:
                    fh.write(f"v {_obj_number(v)}\n")
                for uv in uvs:
                    fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                for n in normals:
                    fh.write(f"vn {_obj_number(n)}\n")
                for face in part.faces:
                    parts = []
                    for idx in face:
                        vi = v_total + int(idx) + 1
                        vti = vt_total + int(idx) + 1
                        vni = vn_total + int(idx) + 1
                        parts.append(f"{vi}/{vti}/{vni}")
                    fh.write("f " + " ".join(parts) + "\n")
                v_total += len(verts)
                vt_total += len(uvs)
                vn_total += len(normals)
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# PBR materials for rapeseed_vase.obj\n")
        fh.write("# The GLB is the physically based deliverable; MTL carries diffuse/alpha hints only.\n")
        for name in used_materials:
            spec = MATERIALS[name]
            r, g, b = spec.base
            fh.write(f"\nnewmtl {name}\n")
            fh.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            fh.write(f"Ka {r * 0.12:.6f} {g * 0.12:.6f} {b * 0.12:.6f}\n")
            if spec.is_transmissive:
                fh.write("Ks 0.92 0.96 0.96\n")
                fh.write("Ns 900\n")
                fh.write(f"Ni {spec.ior:.6f}\n")
                alpha = 0.12 if name == "Glass_Clear" else (0.36 if name == "Water_Volume" else 0.52)
                fh.write(f"d {alpha:.3f}\n")
                fh.write("illum 7\n")
            elif spec.clearcoat > 0:
                fh.write("Ks 0.25 0.25 0.25\n")
                fh.write("Ns 400\n")
                fh.write("Ni 1.5\n")
                fh.write(f"d {spec.opacity:.3f}\n")
                fh.write("illum 2\n")
            else:
                fh.write("Ks 0.04 0.04 0.04\n")
                fh.write(f"Ns {max(1.0, 2.0 + (1.0 - spec.roughness) * 120.0):.1f}\n")
                fh.write("Ni 1.0\n")
                fh.write(f"d {spec.opacity:.3f}\n")
                fh.write("illum 2\n")
        fh.write("\n# End MTL. For glass reflections, use rapeseed_vase.glb in a PBR viewer.\n")


def used_material_names(objects: Sequence[MeshObject]) -> list[str]:
    seen = []
    for obj in objects:
        for part in obj.parts:
            if part.material not in seen:
                seen.append(part.material)
    return seen


def export_glb(objects: Sequence[MeshObject], path: Path, model_meta: dict):
    builder = GlbBuilder()
    builder.finalize(objects, model_meta)
    builder.write(path)
    return builder
