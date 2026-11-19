#!/usr/bin/env python3
"""Build the deliverable GLB / OBJ and a machine-readable asset manifest."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bouquet import HIGH, WIRE, VASE_TOP, WATER_TOP, build_all
from exporters import export_glb, used_material_names, write_obj
from materials import MATERIALS

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def inspect_objects(objects):
    names = [o.name for o in objects]
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"duplicate object names: {duplicates}")
    bad = []
    nan = []
    for obj in objects:
        for part in obj.parts:
            info = part.check_manifold()
            if info["boundary_edges"] or info["nonmanifold_edges"]:
                bad.append({"object": obj.name, "material": part.material, **info})
            verts, normals, uvs = part.arrays()
            if not np.isfinite(verts).all() or not np.isfinite(normals).all() or not np.isfinite(uvs).all():
                nan.append(obj.name)
    if bad:
        raise RuntimeError(f"topology errors: {bad[:3]}")
    if nan:
        raise RuntimeError(f"non-finite values in {sorted(set(nan))[:5]}")
    return {
        "objects": len(objects),
        "triangles": sum(o.triangle_count for o in objects),
        "vertices": sum(sum(len(p.verts) for p in o.parts) for o in objects),
        "materials": len(used_material_names(objects)),
        "objects_watertight_parts": sum(len(o.parts) for o in objects),
    }


def bounds(objects):
    mins = np.min([o.bbox[0] for o in objects], axis=0)
    maxs = np.max([o.bbox[1] for o in objects], axis=0)
    return mins, maxs


def write_manifest(high, wire, stats_high, stats_wire, files):
    mn, mx = bounds(high)
    dims = mx - mn
    manifest = {
        "asset": "Rapeseed bouquet in a straight glass vase",
        "version": "1.0",
        "scope": "high-detail PBR 3D model; presentation background not included",
        "units": "metres",
        "coordinate_system": {
            "up": "Y",
            "origin": "centre of vase base on the floor plane",
            "forward": "toward the viewer at +Z (flowers lean and face outward)",
        },
        "dimensions_m": {
            "width_x": round(float(dims[0]), 5),
            "height_y": round(float(dims[1]), 5),
            "depth_z": round(float(dims[2]), 5),
            "vase_height": VASE_TOP,
            "vase_outer_diameter": 0.072,
            "water_level": WATER_TOP,
        },
        "counts": {
            "high": stats_high,
            "wire": stats_wire,
        },
        "structure": {
            "flowers": sum(1 for o in high if o.category == "flower"),
            "buds": sum(1 for o in high if o.category == "bud"),
            "stems": sum(1 for o in high if o.category == "stem"),
            "leaves": sum(1 for o in high if o.category == "leaf"),
            "vessel_and_water": sum(1 for o in high if o.category in ("vessel", "water")),
            "naming": "Flower_<cluster>_<branch>_<index>, Stem_<cluster>_pedicel_..., Bud_..., Leaf_...",
            "independence": "Every flower head, bud, stem/pedicel and leaf is a separate glTF node with its own mesh and pivot.",
        },
        "materials": {
            name: {
                "base_srgb": spec.base,
                "roughness": spec.roughness,
                "metallic": spec.metallic,
                "transmission": spec.transmission,
                "ior": spec.ior,
                "thickness_m": spec.thickness,
                "clearcoat": spec.clearcoat,
            }
            for name, spec in MATERIALS.items()
        },
        "glb_extensions": [
            "KHR_materials_transmission",
            "KHR_materials_ior",
            "KHR_materials_volume",
            "KHR_materials_clearcoat",
        ],
        "files": files,
        "reference": "source/reference.jpg (pose and silhouette reference only; background excluded)",
    }
    path = MODEL_DIR / "rapeseed_vase_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    t0 = time.time()
    print("building high-detail model…")
    high = build_all(HIGH, seed=47)
    stats_high = inspect_objects(high)
    print("high:", stats_high)
    high_glb = MODEL_DIR / "rapeseed_vase_high.glb"
    export_glb(
        high,
        high_glb,
        {
            "asset_name": "Rapeseed bouquet in a straight clear glass vase",
            "variant": "high",
            "units": "metres",
            "reference": "source/reference.jpg",
            "materials": "PBR; petals matte pale yellow, stems tender green, glass transmission, water film",
        },
    )
    print("wrote", high_glb, high_glb.stat().st_size, "bytes")

    print("writing OBJ / MTL…")
    used = used_material_names(high)
    obj_path = MODEL_DIR / "rapeseed_vase_high.obj"
    mtl_path = MODEL_DIR / "rapeseed_vase_high.mtl"
    write_obj(high, obj_path, mtl_path, used)
    print("wrote", obj_path, obj_path.stat().st_size, "bytes")

    print("building wireframe LOD…")
    wire = build_all(WIRE, seed=47)
    stats_wire = inspect_objects(wire)
    print("wire:", stats_wire)
    wire_glb = MODEL_DIR / "rapeseed_vase_wireframe.glb"
    export_glb(
        wire,
        wire_glb,
        {
            "asset_name": "Rapeseed bouquet in a straight clear glass vase",
            "variant": "wireframe / hologram LOD",
            "units": "metres",
            "reference": "source/reference.jpg",
            "note": "Reduced ring and petal density for clean hologram wireframe rotation; same layout and separate nodes.",
        },
    )
    print("wrote", wire_glb, wire_glb.stat().st_size, "bytes")
    # The OBJ only needs the high variant; provide the wireframe topology as a
    # second, lightweight OBJ too so it can be used directly by line renderers.
    wire_obj = MODEL_DIR / "rapeseed_vase_wireframe.obj"
    wire_mtl = MODEL_DIR / "rapeseed_vase_wireframe.mtl"
    write_obj(wire, wire_obj, wire_mtl, used_material_names(wire))
    print("wrote", wire_obj, wire_obj.stat().st_size, "bytes")

    files = {
        high_glb.name: high_glb.stat().st_size,
        obj_path.name: obj_path.stat().st_size,
        mtl_path.name: mtl_path.stat().st_size,
        wire_glb.name: wire_glb.stat().st_size,
        wire_obj.name: wire_obj.stat().st_size,
        wire_mtl.name: wire_mtl.stat().st_size,
    }
    manifest = write_manifest(high, wire, stats_high, stats_wire, files)
    print("wrote", manifest)
    print("done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
