"""Assemble the rapeseed bouquet, glass vessel and water.

The reference photograph drives the silhouette: a dense central cloud of
flowers, a lower-left shoulder, a looser right side, a small upper-middle
cluster and two tall, solitary stems.  Most flowers are between 1.2 and 2.3 cm
across, mounted on individual pedicels that fork from 13 main stems.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geo import (
    EPS,
    MeshObject,
    Part,
    as_vec,
    bezier,
    frame_from_z,
    leaf_solid,
    normalize,
    petal_solid,
    revolve_profile,
    rapeseed_petal_solid,
    rotate_about,
    sphere_solid,
    tube_solid,
)
from materials import MATERIALS, PETAL_MATERIALS, STEM_MATERIALS

VASE_TOP = 0.176
WATER_TOP = 0.1082
VASE_OUTER_R = 0.0360
VASE_INNER_R = 0.03245
RIM_INNER_R = 0.03255
STRETCH = 1.18


# ---------------------------------------------------------------------------
# quality levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quality:
    name: str
    petal_nu: int
    petal_nv: int
    sepal_nu: int
    sepal_nv: int
    center_segments: int
    center_rings: int
    anther_segments: int
    anther_rings: int
    bud_segments: int
    bud_rings: int
    stem_radial: int
    stem_samples_min: int
    stem_samples_max: int
    vase_segments: int
    water_segments: int
    leaf_nu: int
    leaf_nv: int


HIGH = Quality(
    "high",
    petal_nu=12,
    petal_nv=10,
    sepal_nu=6,
    sepal_nv=4,
    center_segments=16,
    center_rings=10,
    anther_segments=8,
    anther_rings=5,
    bud_segments=14,
    bud_rings=10,
    stem_radial=8,
    stem_samples_min=9,
    stem_samples_max=26,
    vase_segments=128,
    water_segments=96,
    leaf_nu=13,
    leaf_nv=8,
)

WIRE = Quality(
    "wire",
    petal_nu=4,
    petal_nv=3,
    sepal_nu=3,
    sepal_nv=2,
    center_segments=9,
    center_rings=6,
    anther_segments=5,
    anther_rings=3,
    bud_segments=8,
    bud_rings=6,
    stem_radial=5,
    stem_samples_min=6,
    stem_samples_max=18,
    vase_segments=64,
    water_segments=48,
    leaf_nu=7,
    leaf_nv=4,
)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def P(x: float, y: float, z: float) -> np.ndarray:
    """Stretch the flower region vertically while keeping the vase fixed."""
    yy = y if y <= VASE_TOP else VASE_TOP + (y - VASE_TOP) * STRETCH
    return np.array([x, yy, z], dtype=np.float64)


def F(x, y, z, d, fx, fy, fz):
    return dict(p=(x, y, z), d=d, facing=(fx, fy, fz))


def B(x, y, z, d, fx, fy, fz):
    return dict(p=(x, y, z), d=d, facing=(fx, fy, fz))


CLUSTERS = [
    {
        "name": "A_core",
        "base": (0.000, 0.004),
        "hub": (0.004, 0.298, 0.004),
        "branches": [
            {
                "fork": 0.60,
                "anchor": (-0.015, 0.240, 0.010),
                "flowers": [
                    F(-0.024, 0.237, 0.020, 0.0160, -0.4, 0.10, 0.91),
                    F(-0.005, 0.244, 0.026, 0.0140, 0.10, 0.20, 0.97),
                ],
            },
            {
                "fork": 0.66,
                "anchor": (0.020, 0.252, 0.038),
                "flowers": [
                    F(0.018, 0.248, 0.048, 0.0205, 0.22, 0.12, 0.97),
                    F(0.032, 0.257, 0.042, 0.0165, 0.60, 0.22, 0.77),
                    F(0.003, 0.250, 0.044, 0.0170, -0.08, 0.20, 0.97),
                ],
            },
            {
                "fork": 0.73,
                "anchor": (-0.036, 0.281, 0.027),
                "flowers": [
                    F(-0.048, 0.273, 0.035, 0.0215, -0.50, 0.35, 0.79),
                    F(-0.036, 0.286, 0.041, 0.0180, -0.22, 0.45, 0.87),
                    F(-0.024, 0.278, 0.031, 0.0160, 0.10, 0.30, 0.95),
                ],
            },
            {
                "fork": 0.81,
                "anchor": (-0.010, 0.268, -0.030),
                "flowers": [
                    F(-0.024, 0.263, -0.038, 0.0190, -0.50, 0.20, -0.84),
                    F(0.002, 0.276, -0.045, 0.0170, 0.10, 0.40, -0.91),
                    F(0.012, 0.263, -0.030, 0.0150, 0.50, 0.20, -0.84),
                ],
            },
            {
                "fork": 0.88,
                "anchor": (0.036, 0.281, 0.020),
                "flowers": [
                    F(0.052, 0.278, 0.027, 0.0200, 0.70, 0.30, 0.65),
                    F(0.042, 0.289, 0.033, 0.0170, 0.40, 0.50, 0.77),
                    F(0.028, 0.272, 0.023, 0.0150, 0.10, 0.25, 0.96),
                ],
            },
            {
                "fork": 0.94,
                "anchor": (0.010, 0.310, 0.026),
                "flowers": [
                    F(0.000, 0.307, 0.036, 0.0210, -0.10, 0.50, 0.86),
                    F(0.018, 0.313, 0.034, 0.0180, 0.40, 0.45, 0.80),
                    F(0.008, 0.323, 0.024, 0.0160, 0.10, 0.70, 0.70),
                    F(-0.010, 0.315, 0.020, 0.0150, -0.40, 0.60, 0.69),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.004, 0.298, 0.004),
                "flowers": [
                    F(0.006, 0.350, 0.010, 0.0230, 0.05, 0.60, 0.80),
                    F(-0.002, 0.338, 0.001, 0.0170, -0.20, 0.70, 0.68),
                    F(0.016, 0.342, -0.004, 0.0160, 0.30, 0.60, -0.20),
                ],
            },
        ],
        "buds": [
            B(-0.015, 0.342, 0.018, 0.0080, -0.2, 0.7, 0.3),
            B(0.022, 0.360, -0.006, 0.0070, 0.3, 0.7, -0.2),
            B(-0.035, 0.330, 0.030, 0.0065, -0.5, 0.5, 0.4),
        ],
        "stem": "Stem_Mid",
    },
    {
        "name": "B_left_low",
        "base": (-0.010, -0.002),
        "hub": (-0.058, 0.242, 0.030),
        "branches": [
            {
                "fork": 0.75,
                "anchor": (-0.090, 0.238, 0.046),
                "flowers": [
                    F(-0.100, 0.231, 0.053, 0.0200, -0.70, 0.20, 0.68),
                    F(-0.086, 0.245, 0.058, 0.0170, -0.40, 0.40, 0.72),
                    F(-0.106, 0.246, 0.037, 0.0170, -0.80, 0.30, 0.31),
                ],
            },
            {
                "fork": 0.88,
                "anchor": (-0.068, 0.252, 0.012),
                "flowers": [
                    F(-0.066, 0.257, 0.004, 0.0180, -0.20, 0.50, 0.84),
                    F(-0.057, 0.247, -0.005, 0.0150, 0.10, 0.35, 0.93),
                    F(-0.079, 0.259, 0.024, 0.0140, -0.60, 0.45, 0.55),
                ],
            },
            {
                "fork": 0.95,
                "anchor": (-0.082, 0.266, -0.010),
                "flowers": [
                    F(-0.090, 0.272, -0.018, 0.0160, -0.80, 0.40, -0.33),
                    F(-0.072, 0.270, -0.004, 0.0140, -0.20, 0.60, 0.40),
                    F(-0.099, 0.263, 0.003, 0.0130, -0.90, 0.30, 0.22),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.058, 0.242, 0.030),
                "flowers": [
                    F(-0.045, 0.250, 0.040, 0.0180, 0.10, 0.35, 0.90),
                    F(-0.061, 0.260, 0.055, 0.0140, -0.40, 0.50, 0.77),
                ],
            },
        ],
        "buds": [
            B(-0.102, 0.272, 0.052, 0.0070, -0.7, 0.4, 0.4),
            B(-0.062, 0.268, -0.010, 0.0060, -0.2, 0.6, -0.2),
        ],
        "stem": "Stem_Fresh",
    },
    {
        "name": "C_right_mid",
        "base": (0.010, -0.006),
        "hub": (0.066, 0.258, -0.006),
        "branches": [
            {
                "fork": 0.55,
                "anchor": (0.074, 0.220, 0.000),
                "flowers": [
                    F(0.086, 0.216, 0.005, 0.0150, 0.70, 0.10, 0.50),
                    F(0.060, 0.222, 0.012, 0.0130, 0.10, 0.15, 0.95),
                ],
            },
            {
                "fork": 0.72,
                "anchor": (0.083, 0.266, 0.024),
                "flowers": [
                    F(0.096, 0.263, 0.029, 0.0200, 0.80, 0.25, 0.55),
                    F(0.083, 0.276, 0.034, 0.0170, 0.50, 0.50, 0.62),
                    F(0.072, 0.272, 0.030, 0.0150, 0.10, 0.40, 0.90),
                ],
            },
            {
                "fork": 0.84,
                "anchor": (0.085, 0.292, -0.014),
                "flowers": [
                    F(0.099, 0.286, -0.022, 0.0190, 0.85, 0.30, -0.40),
                    F(0.089, 0.301, -0.007, 0.0160, 0.50, 0.60, 0.20),
                    F(0.074, 0.292, -0.021, 0.0140, 0.20, 0.40, -0.85),
                ],
            },
            {
                "fork": 0.92,
                "anchor": (0.052, 0.247, 0.022),
                "flowers": [
                    F(0.050, 0.242, 0.035, 0.0180, 0.10, 0.20, 0.95),
                    F(0.063, 0.251, 0.030, 0.0140, 0.40, 0.30, 0.85),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.066, 0.258, -0.006),
                "flowers": [
                    F(0.078, 0.265, 0.000, 0.0170, 0.80, 0.30, 0.10),
                    F(0.060, 0.275, -0.012, 0.0140, 0.30, 0.60, -0.60),
                ],
            },
        ],
        "buds": [
            B(0.092, 0.306, 0.020, 0.0060, 0.5, 0.6, 0.3),
            B(0.108, 0.280, -0.010, 0.0070, 0.8, 0.4, -0.2),
        ],
        "stem": "Stem_Fresh",
    },
    {
        "name": "D_upper_mid",
        "base": (0.015, 0.004),
        "hub": (-0.002, 0.338, 0.020),
        "branches": [
            {
                "fork": 0.80,
                "anchor": (-0.022, 0.352, 0.024),
                "flowers": [
                    F(-0.028, 0.361, 0.029, 0.0180, -0.30, 0.65, 0.50),
                    F(-0.014, 0.356, 0.034, 0.0150, 0.10, 0.70, 0.60),
                    F(-0.034, 0.348, 0.011, 0.0130, -0.60, 0.45, -0.20),
                ],
            },
            {
                "fork": 0.92,
                "anchor": (0.018, 0.343, -0.018),
                "flowers": [
                    F(0.024, 0.351, -0.026, 0.0170, 0.40, 0.70, -0.50),
                    F(0.031, 0.336, -0.016, 0.0140, 0.70, 0.40, -0.10),
                    F(0.013, 0.349, -0.032, 0.0120, 0.10, 0.60, -0.75),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.002, 0.338, 0.020),
                "flowers": [
                    F(0.000, 0.347, 0.024, 0.0170, 0.00, 0.70, 0.55),
                    F(-0.009, 0.338, 0.032, 0.0130, -0.30, 0.50, 0.80),
                ],
            },
        ],
        "buds": [
            B(0.010, 0.363, 0.010, 0.0070, 0.2, 0.7, 0.2),
            B(-0.025, 0.370, 0.015, 0.0060, -0.4, 0.7, 0.2),
        ],
        "stem": "Stem_Mid",
    },
    {
        "name": "E_tall_left",
        "base": (-0.013, 0.006),
        "hub": (-0.094, 0.372, 0.032),
        "branches": [
            {
                "fork": 0.55,
                "anchor": (-0.078, 0.322, 0.023),
                "flowers": [
                    F(-0.085, 0.333, 0.030, 0.0140, -0.30, 0.60, 0.40),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.094, 0.372, 0.032),
                "flowers": [
                    F(-0.100, 0.379, 0.036, 0.0220, -0.35, 0.75, 0.35),
                ],
            },
        ],
        "buds": [
            B(-0.085, 0.354, 0.026, 0.0060, -0.3, 0.7, 0.2),
        ],
        "stem": "Stem_Fresh",
    },
    {
        "name": "F_tall_right",
        "base": (0.005, -0.008),
        "hub": (0.046, 0.386, -0.010),
        "branches": [
            {
                "fork": 1.0,
                "anchor": (0.046, 0.386, -0.010),
                "flowers": [
                    F(0.043, 0.391, -0.006, 0.0220, 0.10, 0.80, 0.20),
                ],
            },
        ],
        "buds": [
            B(0.054, 0.370, 0.002, 0.0070, 0.3, 0.6, 0.3),
        ],
        "stem": "Stem_Mid",
    },
    {
        "name": "G_far_left",
        "base": (-0.012, -0.004),
        "hub": (-0.102, 0.272, 0.042),
        "branches": [
            {
                "fork": 0.65,
                "anchor": (-0.095, 0.250, 0.030),
                "flowers": [
                    F(-0.088, 0.242, 0.032, 0.0150, -0.10, 0.20, 0.95),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.102, 0.272, 0.042),
                "flowers": [
                    F(-0.114, 0.269, 0.045, 0.0180, -0.80, 0.25, 0.30),
                    F(-0.105, 0.281, 0.052, 0.0150, -0.50, 0.45, 0.65),
                    F(-0.096, 0.268, 0.055, 0.0130, -0.20, 0.25, 0.90),
                ],
            },
        ],
        "buds": [
            B(-0.118, 0.286, 0.050, 0.0060, -0.7, 0.4, 0.4),
        ],
        "stem": "Stem_Fresh",
    },
    {
        "name": "H_far_right",
        "base": (0.012, 0.004),
        "hub": (0.108, 0.294, 0.030),
        "branches": [
            {
                "fork": 0.72,
                "anchor": (0.104, 0.276, 0.026),
                "flowers": [
                    F(0.115, 0.283, 0.028, 0.0150, 0.90, 0.30, 0.10),
                    F(0.096, 0.278, 0.035, 0.0130, 0.30, 0.30, 0.85),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.108, 0.294, 0.030),
                "flowers": [
                    F(0.116, 0.301, 0.034, 0.0180, 0.80, 0.40, 0.20),
                    F(0.104, 0.305, 0.042, 0.0140, 0.40, 0.60, 0.50),
                ],
            },
        ],
        "buds": [
            B(0.120, 0.310, 0.040, 0.0060, 0.8, 0.4, 0.3),
        ],
        "stem": "Stem_Mid",
    },
    {
        "name": "I_back",
        "base": (0.006, -0.014),
        "hub": (0.032, 0.304, -0.052),
        "branches": [
            {
                "fork": 0.70,
                "anchor": (0.048, 0.284, -0.040),
                "flowers": [
                    F(0.056, 0.278, -0.048, 0.0140, 0.70, 0.20, -0.60),
                    F(0.038, 0.286, -0.055, 0.0130, 0.20, 0.30, -0.90),
                ],
            },
            {
                "fork": 0.85,
                "anchor": (0.018, 0.320, -0.040),
                "flowers": [
                    F(0.015, 0.328, -0.046, 0.0130, 0.00, 0.60, -0.75),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.032, 0.304, -0.052),
                "flowers": [
                    F(0.043, 0.311, -0.058, 0.0160, 0.50, 0.40, -0.70),
                    F(0.028, 0.292, -0.062, 0.0180, 0.10, 0.30, -0.91),
                ],
            },
        ],
        "buds": [
            B(0.030, 0.326, -0.060, 0.0060, 0.2, 0.5, -0.7),
        ],
        "stem": "Stem_Deep",
    },
    {
        "name": "J_left_back",
        "base": (-0.008, -0.012),
        "hub": (-0.040, 0.308, -0.040),
        "branches": [
            {
                "fork": 0.80,
                "anchor": (-0.052, 0.288, -0.034),
                "flowers": [
                    F(-0.062, 0.284, -0.042, 0.0140, -0.70, 0.30, -0.60),
                    F(-0.044, 0.292, -0.049, 0.0130, -0.20, 0.30, -0.90),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.040, 0.308, -0.040),
                "flowers": [
                    F(-0.050, 0.314, -0.048, 0.0160, -0.50, 0.40, -0.70),
                    F(-0.030, 0.299, -0.052, 0.0170, 0.00, 0.30, -0.91),
                ],
            },
        ],
        "buds": [
            B(-0.055, 0.320, -0.045, 0.0060, -0.5, 0.4, -0.6),
        ],
        "stem": "Stem_Deep",
    },
    {
        "name": "K_low_right",
        "base": (0.010, 0.012),
        "hub": (0.074, 0.230, 0.048),
        "branches": [
            {
                "fork": 0.80,
                "anchor": (0.064, 0.218, 0.036),
                "flowers": [
                    F(0.058, 0.214, 0.044, 0.0140, 0.10, 0.10, 0.95),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.074, 0.230, 0.048),
                "flowers": [
                    F(0.086, 0.224, 0.052, 0.0170, 0.70, 0.10, 0.60),
                    F(0.068, 0.238, 0.060, 0.0140, 0.20, 0.20, 0.95),
                ],
            },
        ],
        "buds": [
            B(0.090, 0.240, 0.055, 0.0060, 0.6, 0.3, 0.5),
        ],
        "stem": "Stem_Fresh",
    },
    {
        "name": "L_left_mid",
        "base": (-0.011, 0.001),
        "hub": (-0.085, 0.292, 0.008),
        "branches": [
            {
                "fork": 1.0,
                "anchor": (-0.085, 0.292, 0.008),
                "flowers": [
                    F(-0.092, 0.300, 0.014, 0.0150, -0.60, 0.45, 0.35),
                    F(-0.078, 0.287, 0.020, 0.0130, -0.10, 0.30, 0.85),
                ],
            },
        ],
        "buds": [
            B(-0.095, 0.308, 0.005, 0.0060, -0.6, 0.5, 0.2),
        ],
        "stem": "Stem_Mid",
    },
    {
        "name": "M_low_back_right",
        "base": (0.009, 0.010),
        "hub": (0.055, 0.241, 0.055),
        "branches": [
            {
                "fork": 1.0,
                "anchor": (0.055, 0.241, 0.055),
                "flowers": [
                    F(0.060, 0.247, 0.061, 0.0140, 0.40, 0.25, 0.75),
                    F(0.047, 0.235, 0.050, 0.0120, 0.05, 0.20, 0.90),
                ],
            },
        ],
        "buds": [
            B(0.050, 0.252, 0.060, 0.0060, 0.2, 0.4, 0.7),
        ],
        "stem": "Stem_Deep",
    },
]


# Dense, small front-layer flowers close the gaps between the main sprays.
CLUSTERS.extend([
    {
        "name": "N_center_fill",
        "base": (0.002, 0.006),
        "hub": (0.000, 0.293, 0.050),
        "branches": [
            {
                "fork": 0.55,
                "anchor": (-0.018, 0.262, 0.060),
                "flowers": [
                    F(-0.030, 0.256, 0.065, 0.0160, -0.35, 0.20, 0.91),
                    F(-0.010, 0.264, 0.068, 0.0150, 0.10, 0.30, 0.95),
                    F(-0.024, 0.274, 0.058, 0.0140, -0.20, 0.40, 0.89),
                ],
            },
            {
                "fork": 0.72,
                "anchor": (0.020, 0.286, 0.055),
                "flowers": [
                    F(0.030, 0.282, 0.064, 0.0160, 0.50, 0.25, 0.80),
                    F(0.010, 0.291, 0.060, 0.0140, 0.15, 0.40, 0.90),
                    F(0.032, 0.300, 0.050, 0.0130, 0.50, 0.40, 0.75),
                ],
            },
            {
                "fork": 0.86,
                "anchor": (-0.006, 0.305, 0.055),
                "flowers": [
                    F(-0.016, 0.310, 0.061, 0.0140, -0.30, 0.40, 0.86),
                    F(0.006, 0.316, 0.058, 0.0130, 0.20, 0.50, 0.85),
                    F(0.014, 0.305, 0.066, 0.0120, 0.30, 0.30, 0.90),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.000, 0.293, 0.050),
                "flowers": [
                    F(-0.002, 0.330, 0.050, 0.0150, 0.00, 0.60, 0.75),
                    F(0.018, 0.330, 0.045, 0.0130, 0.40, 0.60, 0.65),
                    F(-0.024, 0.320, 0.048, 0.0120, -0.50, 0.50, 0.70),
                ],
            },
        ],
        "buds": [],
        "stem": "Stem_Fresh",
    },
    {
        "name": "O_left_fill",
        "base": (-0.005, -0.002),
        "hub": (-0.045, 0.235, 0.045),
        "branches": [
            {
                "fork": 0.80,
                "anchor": (-0.050, 0.238, 0.055),
                "flowers": [
                    F(-0.056, 0.232, 0.060, 0.0150, -0.35, 0.15, 0.92),
                    F(-0.040, 0.242, 0.063, 0.0130, 0.10, 0.25, 0.95),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (-0.045, 0.235, 0.045),
                "flowers": [
                    F(-0.034, 0.251, 0.052, 0.0140, 0.30, 0.35, 0.86),
                    F(-0.060, 0.246, 0.048, 0.0120, -0.60, 0.30, 0.70),
                ],
            },
        ],
        "buds": [],
        "stem": "Stem_Mid",
    },
    {
        "name": "P_right_fill",
        "base": (0.005, 0.008),
        "hub": (0.050, 0.245, 0.040),
        "branches": [
            {
                "fork": 0.82,
                "anchor": (0.056, 0.250, 0.050),
                "flowers": [
                    F(0.062, 0.246, 0.057, 0.0140, 0.50, 0.25, 0.82),
                    F(0.047, 0.255, 0.060, 0.0130, 0.10, 0.35, 0.93),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.050, 0.245, 0.040),
                "flowers": [
                    F(0.052, 0.265, 0.050, 0.0140, 0.20, 0.40, 0.88),
                ],
            },
        ],
        "buds": [],
        "stem": "Stem_Fresh",
    },
    {
        "name": "Q_back_fill",
        "base": (0.004, -0.010),
        "hub": (0.000, 0.310, -0.045),
        "branches": [
            {
                "fork": 0.82,
                "anchor": (0.014, 0.292, -0.050),
                "flowers": [
                    F(0.022, 0.286, -0.057, 0.0140, 0.35, 0.30, -0.88),
                    F(0.004, 0.298, -0.060, 0.0150, 0.00, 0.40, -0.91),
                ],
            },
            {
                "fork": 1.0,
                "anchor": (0.000, 0.310, -0.045),
                "flowers": [
                    F(-0.010, 0.318, -0.052, 0.0130, -0.25, 0.50, -0.82),
                    F(0.018, 0.314, -0.048, 0.0120, 0.35, 0.45, -0.80),
                ],
            },
        ],
        "buds": [],
        "stem": "Stem_Deep",
    },
])

LEAVES = []  # the reference has a clean stem-only silhouette; no broad leaves


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _target_attach(center, facing, size, bud=False) -> np.ndarray:
    return center - facing * (0.45 * size if bud else 0.16 * size)


def _flower_petal_set(size, frame, center, rng, q: Quality, material_outer, material_inner=None):
    """Four broad, overlapping golden petals: the rapeseed flower.

    A rapeseed flower has four petals in a cross.  Scaling the petals up and
    keeping the centre small and yellow is what makes it read correctly; an
    eight-petal rosette reads as a different flower.
    """
    part = Part(material_outer)
    d = size
    head_roll = rng.uniform(0.0, 2.0 * np.pi)
    thickness = 0.00020 + 0.00004 * rng.random()
    for k in range(4):
        theta = head_roll + k * np.pi / 2.0 + rng.normal(0.0, 0.028)
        length = d * 0.705 * rng.uniform(0.93, 1.07)
        width = d * 0.360 * rng.uniform(0.92, 1.08)
        tilt = np.deg2rad(rng.uniform(3.5, 11.0))
        cup = d * rng.uniform(0.035, 0.070)
        local_v, local_f, local_uv = rapeseed_petal_solid(
            length,
            width,
            cup,
            thickness,
            q.petal_nu,
            q.petal_nv,
            asym=rng.uniform(-0.08, 0.08),
            bend=d * rng.uniform(-0.020, 0.020),
            ridge=d * rng.uniform(0.002, 0.008),
            tip_curl=d * rng.uniform(0.012, 0.034),
            tip_round=0.11 + rng.uniform(0.0, 0.035),
        )
        R = rotate_about([0, 0, 1], theta - np.pi / 2.0) @ rotate_about([1, 0, 0], tilt)
        base = np.array(
            [np.cos(theta) * d * 0.018, np.sin(theta) * d * 0.018, 0.00015],
            dtype=np.float64,
        )
        verts = (local_v @ R.T) + base
        verts = (verts @ frame.T) + center
        part.add_solid(verts, local_f, local_uv)
    return part, None


def _flower_center(size, frame, center, rng, q: Quality):
    d = size
    part = Part("Flower_Center")
    vecs, faces, uvs = sphere_solid(
        center + frame @ np.array([0.0, 0.0, d * 0.014]),
        (d * 0.120, d * 0.120, d * 0.046),
        frame=frame,
        segments=q.center_segments,
        rings=q.center_rings,
    )
    part.add_solid(vecs, faces, uvs)
    anther = Part("Flower_Anther")
    for k in range(7):
        a = 2.0 * np.pi * k / 7.0 + rng.uniform(-0.08, 0.08)
        local = np.array([np.cos(a) * d * 0.072, np.sin(a) * d * 0.072, d * 0.038])
        pos = center + frame @ local
        r = d * rng.uniform(0.014, 0.020)
        vecs, faces, uvs = sphere_solid(
            pos,
            (r, r, r * 0.82),
            frame=frame,
            segments=q.anther_segments,
            rings=q.anther_rings,
        )
        anther.add_solid(vecs, faces, uvs)
    return [part, anther]


def _flower_calyx(size, frame, center, rng, q: Quality):
    d = size
    part = Part("Calyx_Green")
    # receptacle
    vecs, faces, uvs = sphere_solid(
        center + frame @ np.array([0.0, 0.0, -d * 0.06]),
        (d * 0.15, d * 0.15, d * 0.10),
        frame=frame,
        segments=q.center_segments,
        rings=max(4, q.center_rings // 2),
    )
    part.add_solid(vecs, faces, uvs)
    for k in range(4):
        theta = 2.0 * np.pi * k / 4.0 + np.pi / 4.0 + rng.normal(0.0, 0.06)
        length = d * rng.uniform(0.24, 0.31)
        width = d * rng.uniform(0.070, 0.095)
        local_v, local_f, local_uv = petal_solid(
            length,
            width,
            -d * 0.018,
            0.00010,
            q.sepal_nu,
            q.sepal_nv,
            asym=0.0,
            bend=0.0,
            ridge=0.0,
            tip_curl=0.0,
        )
        R = rotate_about([0, 0, 1], theta - np.pi / 2.0) @ rotate_about(
            [1, 0, 0], np.deg2rad(-58.0 + rng.uniform(-5, 5))
        )
        base = np.array([np.cos(theta) * d * 0.06, np.sin(theta) * d * 0.06, -d * 0.055])
        verts = (local_v @ R.T) + base
        verts = (verts @ frame.T) + center
        part.add_solid(verts, local_f, local_uv)
    return part


def build_flower(name, spec, rng, q: Quality, variant_index: int):
    center = P(*spec["p"])
    facing = normalize(spec["facing"])
    # small natural wobble around the authored direction
    facing = normalize(facing + as_vec([rng.normal(0, 0.035), rng.normal(0, 0.03), rng.normal(0, 0.035)]))
    frame = frame_from_z(facing)
    size = float(spec["d"]) * 1.34 * rng.uniform(0.95, 1.05)
    material = PETAL_MATERIALS[variant_index % len(PETAL_MATERIALS)]
    outer, _inner = _flower_petal_set(size, frame, center, rng, q, material)
    obj = MeshObject(name=name, pivot=center, category="flower")
    obj.parts.append(outer)
    for part in _flower_center(size, frame, center, rng, q):
        obj.parts.append(part)
    obj.parts.append(_flower_calyx(size, frame, center, rng, q))
    obj.extras = {
        "type": "flower",
        "species": "Brassica napus (double-flowered rapeseed)",
        "diameter_m": round(size, 4),
        "petal_material": material,
        "petals": 4,
    }
    attach = _target_attach(center, facing, size)
    return obj, attach, facing


def build_bud(name, spec, rng, q: Quality):
    center = P(*spec["p"])
    facing = normalize(spec["facing"])
    facing = normalize(facing + as_vec([rng.normal(0, 0.03), rng.normal(0, 0.03), rng.normal(0, 0.03)]))
    frame = frame_from_z(facing)
    s = float(spec["d"])
    obj = MeshObject(name=name, pivot=center, category="bud")
    body = Part("Bud_Green")
    vecs, faces, uvs = sphere_solid(
        center,
        (s * 0.30, s * 0.30, s * 0.54),
        frame=frame,
        segments=q.bud_segments,
        rings=q.bud_rings,
    )
    body.add_solid(vecs, faces, uvs)
    obj.parts.append(body)
    tip = Part("Bud_Tip")
    vecs, faces, uvs = sphere_solid(
        center + frame @ np.array([0.0, 0.0, s * 0.40]),
        (s * 0.16, s * 0.16, s * 0.20),
        frame=frame,
        segments=max(6, q.bud_segments // 2),
        rings=max(4, q.bud_rings // 2),
    )
    tip.add_solid(vecs, faces, uvs)
    obj.parts.append(tip)
    sepals = Part("Calyx_Green")
    for k in range(3):
        theta = 2.0 * np.pi * k / 3.0 + rng.uniform(-0.08, 0.08)
        local_v, local_f, local_uv = petal_solid(
            s * 0.55,
            s * 0.14,
            -s * 0.035,
            0.00008,
            q.sepal_nu,
            q.sepal_nv,
            asym=0.0,
            bend=0.0,
            ridge=0.0,
            tip_curl=0.0,
        )
        R = rotate_about([0, 0, 1], theta - np.pi / 2.0) @ rotate_about(
            [1, 0, 0], np.deg2rad(-55.0)
        )
        base = np.array([np.cos(theta) * s * 0.055, np.sin(theta) * s * 0.055, -s * 0.28])
        verts = (local_v @ R.T) + base
        verts = (verts @ frame.T) + center
        sepals.add_solid(verts, local_f, local_uv)
    obj.parts.append(sepals)
    obj.extras = {"type": "bud", "length_m": round(s, 4)}
    return obj, center - facing * s * 0.44, facing


def build_leaf(name, spec, rng, q: Quality):
    (x, y, z), direction, normal_hint, length, width = spec
    base_world = P(x, y, z)
    d = normalize(direction)
    normal = as_vec(normal_hint)
    normal = normalize(normal - d * np.dot(normal, d))
    side = normalize(np.cross(d, normal))
    normal = normalize(np.cross(side, d))
    frame = np.column_stack((side, d, normal))
    petiole = 0.34 * length
    blade_origin = base_world + d * petiole
    # blade
    local_v, local_f, local_uv = leaf_solid(
        length,
        width,
        0.00026,
        q.leaf_nu,
        q.leaf_nv,
        lobes=0.12,
        ridge=width * 0.11,
        bend=width * 0.35,
    )
    verts = (local_v @ frame.T) + blade_origin
    obj = MeshObject(name=name, pivot=base_world, category="leaf")
    blade = Part("Leaf_Green")
    blade.add_solid(verts, local_f, local_uv)
    obj.parts.append(blade)
    # midrib
    pts = []
    radii = []
    for i in range(9):
        t = i / 8.0
        local = np.array([0.0, t * length, width * 0.12 * np.sin(np.pi * t) ** 1.1])
        pts.append((local @ frame.T) + blade_origin)
        radii.append(width * (0.018 - 0.010 * t))
    midrib = Part("Leaf_Vein")
    rv, rf, ru = tube_solid(np.asarray(pts), np.asarray(radii), radial=q.stem_radial, caps=True)
    midrib.add_solid(rv, rf, ru)
    obj.parts.append(midrib)
    # petiole
    p0 = base_world
    p1 = base_world + d * (petiole * 0.55) + normal * 0.002
    p2 = blade_origin
    path = bezier(p0, p1, p1, p2, samples=8)
    radii = np.linspace(0.0009, 0.00065, len(path))
    pv, pf, pu = tube_solid(path, radii, radial=q.stem_radial, caps=True)
    pet = Part("Leaf_Green")
    pet.add_solid(pv, pf, pu)
    obj.parts.append(pet)
    obj.extras = {"type": "leaf", "length_m": round(length, 4)}
    return obj


# ---------------------------------------------------------------------------
# stems
# ---------------------------------------------------------------------------


def _trunk_path(base_xy, hub, rng):
    bx, bz = base_xy
    neck_radius = 0.0175
    nx = float(np.clip(hub[0] * 0.18 + rng.uniform(-0.0015, 0.0015), -neck_radius, neck_radius))
    nz = float(np.clip(hub[2] * 0.18 + rng.uniform(-0.0015, 0.0015), -neck_radius, neck_radius))
    if np.hypot(nx, nz) > neck_radius:
        scale = neck_radius / np.hypot(nx, nz)
        nx, nz = nx * scale, nz * scale
    neck = np.array([nx, VASE_TOP + 0.0015, nz])
    p0 = np.array([bx, 0.016, bz])
    p1 = np.array([bx * 0.82 + rng.uniform(-0.001, 0.001), 0.060, bz * 0.82 + rng.uniform(-0.001, 0.001)])
    p2 = np.array([bx * 0.45, 0.130, bz * 0.45])
    lower = bezier(p0, p1, p2, neck, samples=14)
    q1 = neck + (hub - neck) * 0.28 + np.array([rng.uniform(-0.002, 0.002), 0.020, rng.uniform(-0.002, 0.002)])
    q2 = neck + (hub - neck) * 0.72 + np.array([rng.uniform(-0.001, 0.001), 0.012, rng.uniform(-0.001, 0.001)])
    upper = bezier(neck, q1, q2, hub, samples=14)
    return np.concatenate((lower, upper[1:]), axis=0)


def _taper_radii(points, r_start, r_end, rng, wobble=0.05):
    n = len(points)
    t = np.linspace(0.0, 1.0, n)
    base = r_start + (r_end - r_start) * t**0.8
    mod = 1.0 + wobble * np.sin(t * np.pi * 4.0 + rng.uniform(0, 2 * np.pi)) * t * (1 - t) * 4
    return np.maximum(base * mod, 0.00045)


def _make_stem_object(name, path, radii, material, q: Quality, junction=None):
    part = Part(material)
    rv, rf, ru = tube_solid(path, radii, radial=q.stem_radial, caps=True)
    part.add_solid(rv, rf, ru)
    if junction is not None:
        center, radius = junction
        jv, jf, ju = sphere_solid(
            center,
            (radius * 1.25, radius * 1.10, radius * 1.25),
            frame=np.eye(3),
            segments=max(6, q.stem_radial),
            rings=max(4, q.stem_radial // 2 + 2),
        )
        part.add_solid(jv, jf, ju)
    obj = MeshObject(name=name, pivot=path[0].copy(), category="stem")
    obj.add_part(part)
    obj.extras = {"type": "stem", "radius_start_mm": round(float(radii[0]) * 1000, 2)}
    return obj


def _branch_path(start, anchor, rng, samples):
    delta = anchor - start
    lateral = normalize(np.cross(np.array([0.0, 1.0, 0.0]), delta)) if np.linalg.norm(np.cross(np.array([0.0, 1.0, 0.0]), delta)) > 1e-6 else np.array([1.0, 0.0, 0.0])
    off = lateral * rng.uniform(-0.006, 0.006) + np.array([0.0, rng.uniform(-0.003, 0.006), 0.0])
    p1 = start + delta * 0.33 + off
    p2 = start + delta * 0.69 - off * 0.35
    return bezier(start, p1, p2, anchor, samples=samples)


def _pedicel_path(start, end, rng, samples):
    delta = end - start
    lateral = normalize(np.cross(np.array([0.0, 1.0, 0.0]), delta)) if np.linalg.norm(np.cross(np.array([0.0, 1.0, 0.0]), delta)) > 1e-6 else np.array([1.0, 0.0, 0.0])
    off = lateral * rng.uniform(-0.004, 0.004) + np.array([0.0, rng.uniform(-0.001, 0.005), 0.0])
    p1 = start + delta * 0.28 + off
    p2 = start + delta * 0.70 + off * 0.4
    return bezier(start, p1, p2, end, samples=samples)


def build_all(quality: Quality = HIGH, seed: int = 47):
    rng = np.random.default_rng(seed)
    objects: list[MeshObject] = []
    flower_attach: list[tuple[str, np.ndarray]] = []
    all_flower_objects = []
    all_stems = []
    flower_counter = 0
    for ci, cluster in enumerate(CLUSTERS):
        name = cluster["name"]
        hub = P(*cluster["hub"])
        trunk_path = _trunk_path(cluster["base"], hub, rng)
        trunk_radii = _taper_radii(trunk_path, 0.00195, 0.00112, rng)
        stem_material = cluster.get("stem", STEM_MATERIALS[ci % len(STEM_MATERIALS)])
        trunk = _make_stem_object(f"Stem_{name}_trunk", trunk_path, trunk_radii, stem_material, quality)
        all_stems.append(trunk)
        # index of the first trunk sample above the vase rim
        above = np.where(trunk_path[:, 1] > VASE_TOP + 0.0005)[0]
        start_index = int(above[0]) if len(above) else len(trunk_path) - 1
        top_span = np.arange(start_index, len(trunk_path))
        branch_anchors = []
        for bi, branch in enumerate(cluster["branches"]):
            anchor = P(*branch["anchor"])
            # fork point lives on the visible, above-rim portion of the trunk
            frac = float(np.clip(branch["fork"], 0.0, 1.0))
            idx = top_span[min(len(top_span) - 1, int(round(frac * (len(top_span) - 1))))]
            fork = trunk_path[idx]
            branch_anchors.append(anchor)
            if np.linalg.norm(anchor - fork) > 0.0055:
                samples = int(np.clip(np.linalg.norm(anchor - fork) * 430, quality.stem_samples_min, quality.stem_samples_max))
                bpath = _branch_path(fork, anchor, rng, samples)
                bradii = _taper_radii(bpath, 0.00112, 0.00082, rng, wobble=0.03)
                branch_obj = _make_stem_object(
                    f"Stem_{name}_branch{bi + 1:02d}",
                    bpath,
                    bradii,
                    stem_material,
                    quality,
                    junction=(anchor, float(bradii[-1])),
                )
                all_stems.append(branch_obj)
            for fi, flower in enumerate(branch["flowers"]):
                flower_counter += 1
                obj_name = f"Flower_{name}_{fi + 1:02d}"
                # unique per branch numbering is not necessary for JSON but keep names unique
                obj_name = f"Flower_{name}_{bi + 1:02d}_{fi + 1:02d}"
                if any(o.name == obj_name for o in all_flower_objects):
                    obj_name += f"_{flower_counter}"
                variant = (flower_counter * 7 + ci * 3) % len(PETAL_MATERIALS)
                flower_obj, attach, facing = build_flower(obj_name, flower, rng, quality, variant)
                all_flower_objects.append(flower_obj)
                flower_attach.append((obj_name, attach))
                # each flower gets its own fine pedicel
                start = anchor + as_vec(
                    [rng.uniform(-0.0025, 0.0025), rng.uniform(0.0, 0.002), rng.uniform(-0.0025, 0.0025)]
                ) * 0.65
                delta = attach - start
                samples = int(np.clip(np.linalg.norm(delta) * 520, 6, quality.stem_samples_max))
                path = _pedicel_path(start, attach, rng, samples)
                pradii = _taper_radii(path, 0.00080, 0.00053, rng, wobble=0.02)
                all_stems.append(
                    _make_stem_object(
                        f"Stem_{name}_pedicel_{bi + 1:02d}_{fi + 1:02d}",
                        path,
                        pradii,
                        "Stem_Fresh" if rng.random() < 0.7 else "Stem_Mid",
                        quality,
                    )
                )
        # buds get very fine, separate stems
        anchors = branch_anchors or [hub]
        for bi, bud in enumerate(cluster.get("buds", [])):
            b_obj, b_attach, _ = build_bud(f"Bud_{name}_{bi + 1:02d}", bud, rng, quality)
            # attach to nearest branch anchor
            anchor = min(anchors, key=lambda a: float(np.linalg.norm(a - b_attach)))
            start = anchor + as_vec(
                [rng.uniform(-0.002, 0.002), rng.uniform(0.0, 0.0015), rng.uniform(-0.002, 0.002)]
            )
            delta = b_attach - start
            samples = int(np.clip(np.linalg.norm(delta) * 600, 6, quality.stem_samples_max))
            path = _pedicel_path(start, b_attach, rng, samples)
            pradii = _taper_radii(path, 0.00054, 0.00041, rng, wobble=0.02)
            all_stems.append(
                _make_stem_object(
                    f"Stem_Bud_{name}_{bi + 1:02d}",
                    path,
                    pradii,
                    "Stem_Fresh",
                    quality,
                )
            )
            objects.append(b_obj)

    objects.extend(all_stems)
    objects.extend(all_flower_objects)
    for li, leaf in enumerate(LEAVES):
        objects.append(build_leaf(f"Leaf_{li + 1:02d}", leaf, rng, quality))

    vase = _build_vase(quality)
    water_volume = _build_water_volume(quality)
    water_surface = _build_water_surface(quality)
    water_meniscus = _build_water_meniscus(quality)
    objects.extend([vase, water_volume, water_surface, water_meniscus])
    return objects


# ---------------------------------------------------------------------------
# small meniscus ring
# ---------------------------------------------------------------------------


def _torus_solid(center, major_radius, minor_radius, major_segments=96, minor_segments=8):
    """Closed torus around the Y axis, used as the visible water meniscus."""
    cx, cy, cz = center
    verts = []
    uvs = []
    for i in range(major_segments):
        phi = 2.0 * np.pi * i / major_segments
        cp, sp = np.cos(phi), np.sin(phi)
        for j in range(minor_segments):
            psi = 2.0 * np.pi * j / minor_segments
            rr = major_radius + minor_radius * np.cos(psi)
            verts.append(np.array([cx + rr * cp, cy + minor_radius * np.sin(psi), cz + rr * sp]))
            uvs.append((i / major_segments, j / minor_segments))
    faces = []
    for i in range(major_segments):
        ni = (i + 1) % major_segments
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            faces.append(
                (
                    i * minor_segments + j,
                    ni * minor_segments + j,
                    ni * minor_segments + nj,
                    i * minor_segments + nj,
                )
            )
    return np.asarray(verts), faces, np.asarray(uvs)


# ---------------------------------------------------------------------------
# glass and water
# ---------------------------------------------------------------------------


def _build_vase(quality: Quality):
    profile = [
        (0.0000, 0.0000),
        (0.0240, 0.0000),
        (0.0310, 0.0008),
        (0.0348, 0.0030),
        (0.0359, 0.0060),
        (0.0360, 0.0120),
        (0.0360, 0.1630),
        (0.0359, 0.1688),
        (0.0355, 0.1725),
        (0.0346, 0.1752),
        (0.0332, 0.1760),
        (0.0327, 0.1748),
        (0.03245, 0.1720),
        (0.03245, 0.0180),
        (0.0318, 0.0142),
        (0.0280, 0.0128),
        (0.0000, 0.0128),
    ]
    verts, faces, uvs = revolve_profile(profile, segments=quality.vase_segments)
    part = Part("Glass_Clear")
    part.add_solid(verts, faces, uvs)
    obj = MeshObject(name="Vase_Glass", pivot=np.zeros(3), category="vessel")
    obj.add_part(part)
    obj.extras = {
        "type": "vessel",
        "outer_radius_m": VASE_OUTER_R,
        "inner_radius_m": VASE_INNER_R,
        "height_m": VASE_TOP,
        "wall_mm": round((VASE_OUTER_R - VASE_INNER_R) * 1000, 2),
        "water_level_m": WATER_TOP,
        "note": "Closed glass shell; open rim; transmission + IOR material in GLB.",
    }
    return obj


def _build_water_volume(quality: Quality):
    profile = [
        (0.0000, 0.0134),
        (VASE_INNER_R - 0.0009, 0.0134),
        (VASE_INNER_R - 0.0009, WATER_TOP - 0.0003),
        (0.0000, WATER_TOP - 0.0003),
    ]
    verts, faces, uvs = revolve_profile(profile, segments=quality.water_segments)
    part = Part("Water_Volume")
    part.add_solid(verts, faces, uvs)
    obj = MeshObject(name="Water_Volume", pivot=np.zeros(3), category="water")
    obj.add_part(part)
    obj.extras = {"type": "water_volume", "level_m": WATER_TOP, "note": "Closed water volume inside the glass."}
    return obj


def _build_water_meniscus(quality: Quality):
    """A fine waterline ring that reads through clear glass."""
    part = Part("Water_Line")
    r = VASE_INNER_R - 0.0010
    verts, faces, uvs = _torus_solid(
        (0.0, WATER_TOP + 0.00018, 0.0),
        r,
        0.00055,
        major_segments=max(48, quality.water_segments),
        minor_segments=8 if quality is HIGH else 6,
    )
    part.add_solid(verts, faces, uvs)
    obj = MeshObject(name="Water_Meniscus", pivot=np.zeros(3), category="water")
    obj.add_part(part)
    obj.extras = {
        "type": "water_meniscus",
        "level_m": WATER_TOP,
        "note": "Thin meniscus rim at the waterline; separate object for compositing.",
    }
    return obj


def _build_water_surface(quality: Quality):
    r = VASE_INNER_R - 0.0008
    bottom = 0.10806
    top_center = 0.10822
    top_edge = 0.10892
    profile = [
        (0.0000, bottom),
        (r, bottom),
        (r, top_edge - 0.00008),
        (r * 0.985, top_edge),
    ]
    # concave meniscus returning to the centre
    for j in range(1, 8):
        rr = r * (1.0 - j / 8.0)
        yy = top_center + (top_edge - top_center) * (rr / r) ** 8.0
        profile.append((rr, yy))
    profile.append((0.0000, top_center))
    verts, faces, uvs = revolve_profile(profile, segments=quality.water_segments)
    part = Part("Water_Surface")
    part.add_solid(verts, faces, uvs)
    obj = MeshObject(name="Water_Surface", pivot=np.zeros(3), category="water")
    obj.add_part(part)
    obj.extras = {
        "type": "water_surface",
        "level_m": WATER_TOP,
        "note": "Thin meniscus film. Slightly cooler and glossier than the water volume.",
    }
    return obj
