"""PBR material definitions (colours are authored in sRGB)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialSpec:
    name: str
    base: tuple[float, float, float]
    roughness: float = 0.7
    metallic: float = 0.0
    transmission: float = 0.0
    ior: float = 1.5
    thickness: float = 0.0
    attenuation: tuple[float, float, float] = (1.0, 1.0, 1.0)
    attenuation_distance: float = 1.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.03
    opacity: float = 1.0
    extras: dict = field(default_factory=dict)

    @property
    def is_transmissive(self) -> bool:
        return self.transmission > 0.001


MATERIALS: dict[str, MaterialSpec] = {
    # true rapeseed yellow: matte lemon-gold, not cream ----------------------
    "Petals_Golden": MaterialSpec("Petals_Golden", (0.925, 0.645, 0.016), roughness=0.76),
    "Petals_Sun": MaterialSpec("Petals_Sun", (0.956, 0.735, 0.035), roughness=0.78),
    "Petals_Deep": MaterialSpec("Petals_Deep", (0.855, 0.535, 0.010), roughness=0.74),
    "Petals_Light": MaterialSpec("Petals_Light", (0.973, 0.800, 0.110), roughness=0.80),
    # flower eye / anthers ---------------------------------------------------
    "Flower_Center": MaterialSpec("Flower_Center", (0.780, 0.670, 0.200), roughness=0.82),
    "Flower_Anther": MaterialSpec("Flower_Anther", (0.990, 0.830, 0.180), roughness=0.70),
    # stems and foliage ------------------------------------------------------
    "Stem_Fresh": MaterialSpec("Stem_Fresh", (0.388, 0.545, 0.224), roughness=0.58),
    "Stem_Mid": MaterialSpec("Stem_Mid", (0.306, 0.451, 0.176), roughness=0.62),
    "Stem_Deep": MaterialSpec("Stem_Deep", (0.224, 0.337, 0.129), roughness=0.66),
    "Calyx_Green": MaterialSpec("Calyx_Green", (0.306, 0.443, 0.169), roughness=0.64),
    "Leaf_Green": MaterialSpec("Leaf_Green", (0.275, 0.416, 0.169), roughness=0.55),
    "Leaf_Vein": MaterialSpec("Leaf_Vein", (0.416, 0.565, 0.259), roughness=0.60),
    "Bud_Green": MaterialSpec("Bud_Green", (0.612, 0.573, 0.169), roughness=0.68),
    "Bud_Tip": MaterialSpec("Bud_Tip", (0.816, 0.678, 0.129), roughness=0.72),
    # vessel ----------------------------------------------------------------
    "Glass_Clear": MaterialSpec(
        "Glass_Clear",
        (0.970, 0.993, 0.988),
        roughness=0.012,
        transmission=0.995,
        ior=1.47,
        thickness=0.0028,
        attenuation=(0.93, 0.99, 0.97),
        attenuation_distance=0.80,
        clearcoat=0.08,
        clearcoat_roughness=0.018,
        extras={"physical_note": "Soda-lime glass shell. Use a transmission-capable viewer."},
    ),
    "Water_Volume": MaterialSpec(
        "Water_Volume",
        (0.861, 0.943, 0.918),
        roughness=0.022,
        transmission=0.975,
        ior=1.333,
        thickness=0.045,
        attenuation=(0.78, 0.92, 0.86),
        attenuation_distance=0.60,
        extras={"physical_note": "Cut-stem water volume inside the glass."},
    ),
    "Water_Line": MaterialSpec(
        "Water_Line",
        (0.700, 0.800, 0.755),
        roughness=0.20,
        transmission=0.0,
        ior=1.333,
        thickness=0.0,
        clearcoat=0.35,
        clearcoat_roughness=0.04,
        opacity=0.34,
        extras={"physical_note": "Waterline / meniscus edge. Alpha-blended so it reads through the glass."},
    ),
    "Water_Surface": MaterialSpec(
        "Water_Surface",
        (0.805, 0.883, 0.845),
        roughness=0.105,
        transmission=0.0,
        ior=1.333,
        thickness=0.0,
        clearcoat=0.60,
        clearcoat_roughness=0.038,
        opacity=0.26,
        extras={"physical_note": "Alpha-blended water surface with a clearcoat sheen; actual water volume is separate."},
    ),
}


PETAL_MATERIALS = ["Petals_Golden", "Petals_Sun", "Petals_Golden", "Petals_Light", "Petals_Sun", "Petals_Golden"]
STEM_MATERIALS = ["Stem_Fresh", "Stem_Fresh", "Stem_Mid", "Stem_Fresh", "Stem_Mid", "Stem_Deep"]
