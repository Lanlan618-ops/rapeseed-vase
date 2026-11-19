"""Procedural geometry helpers for the rapeseed bouquet asset.

Everything is built in metres, Y up.  A Part owns one material and can contain
several disconnected closed solids (for example: every petal of one flower).
Faces are kept as polygons so the OBJ can stay quad-dominant; the GLB exporter
triangulates them with a stable fan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# vectors / curves
# ---------------------------------------------------------------------------


def as_vec(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def normalize(value: Sequence[float]) -> np.ndarray:
    v = as_vec(value)
    length = float(np.linalg.norm(v))
    if length < EPS:
        raise ValueError("zero-length vector")
    return v / length


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def catmull_rom(points: Sequence[np.ndarray], samples_per_segment: int = 14) -> np.ndarray:
    """Centripetal-ish Catmull-Rom through the provided points."""
    p = [as_vec(v) for v in points]
    if len(p) == 2:
        return np.linspace(p[0], p[1], samples_per_segment + 1)
    out: list[np.ndarray] = []
    padded = [p[0] + (p[0] - p[1])] + p + [p[-1] + (p[-1] - p[-2])]
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1 : i + 3]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2, t3 = t * t, t * t * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(point)
    out.append(p[-1])
    return np.asarray(out)


def bezier(p0, p1, p2, p3, samples: int = 18) -> np.ndarray:
    p0, p1, p2, p3 = map(as_vec, (p0, p1, p2, p3))
    t = np.linspace(0.0, 1.0, samples + 1)[:, None]
    u = 1.0 - t
    return u**3 * p0 + 3.0 * u**2 * t * p1 + 3.0 * u * t**2 * p2 + t**3 * p3


def frame_from_z(direction, hint=None) -> np.ndarray:
    """3x3 frame whose columns are local x, y, z in world space."""
    z = normalize(direction)
    if hint is None:
        hint = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(z, hint))) > 0.94:
            hint = np.array([1.0, 0.0, 0.0])
    hint = normalize(hint)
    x = np.cross(hint, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x = normalize(x)
    y = normalize(np.cross(z, x))
    return np.column_stack((x, y, z))


def rotate_about(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def signed_volume(vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> float:
    volume = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        a = vertices[face[0]]
        for j in range(1, len(face) - 1):
            b = vertices[face[j]]
            c = vertices[face[j + 1]]
            volume += float(np.dot(a, np.cross(b, c)))
    return volume / 6.0


def triangulate_face(face: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    for j in range(1, len(face) - 1):
        yield int(face[0]), int(face[j]), int(face[j + 1])


def polygon_area(vertices: np.ndarray, face: Sequence[int]) -> float:
    if len(face) < 3:
        return 0.0
    area = np.zeros(3)
    for j in range(1, len(face) - 1):
        area += np.cross(vertices[face[j]] - vertices[face[0]], vertices[face[j + 1]] - vertices[face[0]])
    return 0.5 * float(np.linalg.norm(area))


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass
class Part:
    material: str
    verts: list[np.ndarray] = field(default_factory=list)
    uvs: list[np.ndarray] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    # useful provenance for the report only
    solids: int = 0

    def add_solid(self, vertices, faces, uvs=None, orient=True) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = [tuple(int(i) for i in f) for f in faces]
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must be Nx3")
        if uvs is None:
            uvs_arr = np.zeros((len(vertices), 2), dtype=np.float64)
        else:
            uvs_arr = np.asarray(uvs, dtype=np.float64)
            if uvs_arr.shape != (len(vertices), 2):
                raise ValueError("uvs must be Nx2 and match vertices")
        if orient:
            volume = signed_volume(vertices, faces)
            if volume < 0:
                faces = [tuple(reversed(f)) for f in faces]
        # Drop exact duplicate or degenerate polygons defensively.
        clean = []
        for face in faces:
            if len(set(face)) != len(face):
                continue
            if polygon_area(vertices, face) < 1e-14:
                continue
            clean.append(face)
        offset = len(self.verts)
        self.verts.extend(vertices.tolist())
        self.uvs.extend(uvs_arr.tolist())
        self.faces.extend(tuple(i + offset for i in f) for f in clean)
        self.solids += 1

    def arrays(self):
        verts = np.asarray(self.verts, dtype=np.float64)
        uvs = np.asarray(self.uvs, dtype=np.float64)
        normals = self.compute_normals(verts)
        return verts, normals, uvs

    def compute_normals(self, verts=None) -> np.ndarray:
        if verts is None:
            verts = np.asarray(self.verts, dtype=np.float64)
        normals = np.zeros_like(verts)
        for face in self.faces:
            a, b, c = (verts[face[0]], verts[face[1]], verts[face[2]])
            # accumulate area-weighted face normal across the whole polygon
            for tri in triangulate_face(face):
                a, b, c = (verts[tri[0]], verts[tri[1]], verts[tri[2]])
                n = np.cross(b - a, c - a)
                normals[tri[0]] += n
                normals[tri[1]] += n
                normals[tri[2]] += n
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-16
        normals[valid] /= lengths[valid, None]
        if np.any(~valid):
            normals[~valid] = np.array([0.0, 0.0, 1.0])
        return normals

    def check_manifold(self) -> dict:
        edges: dict[tuple[int, int], int] = {}
        bad_faces = 0
        for face in self.faces:
            for j in range(len(face)):
                a, b = face[j], face[(j + 1) % len(face)]
                edge = (a, b) if a < b else (b, a)
                edges[edge] = edges.get(edge, 0) + 1
        counts = np.asarray(list(edges.values()), dtype=np.int64)
        return {
            "solids": self.solids,
            "vertices": len(self.verts),
            "faces": len(self.faces),
            "triangles": sum(len(f) - 2 for f in self.faces),
            "edges": len(edges),
            "boundary_edges": int(np.sum(counts == 1)),
            "nonmanifold_edges": int(np.sum(counts > 2)),
            "bad_faces": bad_faces,
        }


@dataclass
class MeshObject:
    name: str
    pivot: np.ndarray
    category: str
    parts: list[Part] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def add_part(self, part: Part) -> None:
        if part.faces:
            self.parts.append(part)

    @property
    def triangle_count(self) -> int:
        return sum(sum(len(f) - 2 for f in p.faces) for p in self.parts)

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        allv = np.concatenate([np.asarray(p.verts, dtype=np.float64) for p in self.parts], axis=0)
        return allv.min(axis=0), allv.max(axis=0)

    def validate(self) -> list[dict]:
        out = []
        for part in self.parts:
            info = part.check_manifold()
            info["object"] = self.name
            info["material"] = part.material
            out.append(info)
        return out


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------


def _grid_faces(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = i * cols + j
            b = (i + 1) * cols + j
            c = (i + 1) * cols + j + 1
            d = i * cols + j + 1
            faces.append((a, b, c, d))
    return faces


def petal_solid(
    length: float,
    width: float,
    cup: float,
    thickness: float,
    nu: int = 8,
    nv: int = 6,
    *,
    asym: float = 0.0,
    bend: float = 0.0,
    ridge: float = 0.0,
    tip_curl: float = 0.0,
    width_power: float = 0.62,
    base_lift: float = 0.00025,
):
    """Closed obovate petal, base at origin, tip along +Y, face toward +Z."""
    t = np.linspace(0.0, 1.0, nu + 1)
    v = np.linspace(-1.0, 1.0, nv + 1)
    T, V = np.meshgrid(t, v, indexing="ij")
    silhouette = np.sin(np.pi * np.power(T, 1.04)) ** width_power
    half = width * (0.075 + 0.925 * silhouette)
    half = half * (1.0 + asym * 0.12 * V)
    X = half * V + bend * (T**2)
    Y = length * T
    Z = (
        base_lift
        + cup * (V**2) * (0.18 + 0.82 * T**1.25)
        + ridge * np.exp(-((V / 0.27) ** 2)) * np.sin(np.pi * T) ** 1.25
        + tip_curl * (T**2.2)
    )
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    Xb, Yb, Zb = Xf.copy(), Yf.copy(), Zf - thickness
    n = len(Xf)
    verts = np.column_stack(
        (
            np.concatenate((Xf, Xb)),
            np.concatenate((Yf, Yb)),
            np.concatenate((Zf, Zb)),
        )
    )
    uvs = np.column_stack(
        (
            np.concatenate((T.ravel(), T.ravel())),
            np.concatenate(((V.ravel() + 1) / 2, (V.ravel() + 1) / 2)),
        )
    )
    faces = []
    grid_faces = _grid_faces(nu + 1, nv + 1)
    faces.extend(grid_faces)
    faces.extend([tuple(i + n for i in reversed(f)) for f in grid_faces])
    cols = nv + 1

    def f(i, j):
        return i * cols + j

    # Boundary strips close the front and back surfaces.
    for j in range(nv):
        faces.append((f(0, j), f(0, j + 1), f(0, j + 1) + n, f(0, j) + n))
    for j in range(nv):
        faces.append((f(nu, j), f(nu, j) + n, f(nu, j + 1) + n, f(nu, j + 1)))
    for i in range(nu):
        faces.append((f(i, 0), f(i, 0) + n, f(i + 1, 0) + n, f(i + 1, 0)))
    for i in range(nu):
        faces.append((f(i, nv), f(i + 1, nv), f(i + 1, nv) + n, f(i, nv) + n))
    return verts, faces, uvs


def rapeseed_petal_solid(
    length: float,
    width: float,
    cup: float,
    thickness: float,
    nu: int = 8,
    nv: int = 8,
    *,
    asym: float = 0.0,
    bend: float = 0.0,
    ridge: float = 0.0,
    tip_curl: float = 0.0,
    tip_round: float = 0.13,
    base_lift: float = 0.0003,
):
    """Broad, four-petal rapeseed petal with a rounded, slightly notched tip.

    Local frame: base at origin, tip toward +Y, upper face toward +Z.
    The blade is deliberately broad and blunt; that is what reads as
    *Brassica napus* rather than a small eight-petal rosette.
    """
    t = np.linspace(0.0, 1.0, nu + 1)
    v = np.linspace(-1.0, 1.0, nv + 1)
    T, V = np.meshgrid(t, v, indexing="ij")
    # obovate rapeseed blade: widest just past the middle, then a tapered,
    # genuinely rounded tip instead of the square "chopped paper" silhouette.
    ellipse = np.sqrt(np.clip(1.0 - ((T - 0.56) / 0.44) ** 2, 0.0, 1.0))
    claw = 0.55 + 0.45 * np.clip(T / 0.18, 0.0, 1.0)
    half = width * (0.03 + 0.97 * ellipse) * claw * (1.0 + asym * 0.10 * V)
    # taper the very last millimetres rather than leaving a flat chopped tip
    tip_taper = 1.0 - 0.55 * np.clip((T - 0.82) / 0.18, 0.0, 1.0)
    half = half * tip_taper
    X = half * V + bend * T**2
    round_s = np.clip((T - 0.62) / 0.38, 0.0, 1.0)
    round_s = round_s * round_s * (3.0 - 2.0 * round_s)
    Y = length * T - tip_round * length * (V**4) * round_s
    Z = (
        base_lift
        + cup * (V**2) * (0.28 + 0.72 * T**1.15)
        + ridge * np.exp(-((V / 0.26) ** 2)) * np.sin(np.pi * T) ** 1.2
        + tip_curl * (T**2)
    )
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    Xb, Yb, Zb = Xf.copy(), Yf.copy(), Zf - thickness
    n = len(Xf)
    verts = np.column_stack(
        (
            np.concatenate((Xf, Xb)),
            np.concatenate((Yf, Yb)),
            np.concatenate((Zf, Zb)),
        )
    )
    uvs = np.column_stack(
        (
            np.concatenate((T.ravel(), T.ravel())),
            np.concatenate(((V.ravel() + 1) / 2, (V.ravel() + 1) / 2)),
        )
    )
    faces = []
    grid_faces = _grid_faces(nu + 1, nv + 1)
    faces.extend(grid_faces)
    faces.extend([tuple(i + n for i in reversed(f)) for f in grid_faces])
    cols = nv + 1

    def f(i, j):
        return i * cols + j

    for j in range(nv):
        faces.append((f(0, j), f(0, j + 1), f(0, j + 1) + n, f(0, j) + n))
    for j in range(nv):
        faces.append((f(nu, j), f(nu, j) + n, f(nu, j + 1) + n, f(nu, j + 1)))
    for i in range(nu):
        faces.append((f(i, 0), f(i, 0) + n, f(i + 1, 0) + n, f(i + 1, 0)))
    for i in range(nu):
        faces.append((f(i, nv), f(i + 1, nv), f(i + 1, nv) + n, f(i, nv) + n))
    return verts, faces, uvs


def leaf_solid(
    length: float,
    width: float,
    thickness: float,
    nu: int = 12,
    nv: int = 8,
    *,
    lobes: float = 0.10,
    ridge: float = 0.0009,
    bend: float = 0.0,
):
    """Closed lanceolate, faintly lobed leaf blade. Base at origin, +Y tip."""
    t = np.linspace(0.0, 1.0, nu + 1)
    v = np.linspace(-1.0, 1.0, nv + 1)
    T, V = np.meshgrid(t, v, indexing="ij")
    sil = np.sin(np.pi * np.power(T, 0.92)) ** 0.58
    wave = 1.0 + lobes * np.sin(2.0 * np.pi * 3.2 * T) * T
    half = width * (0.055 + 0.945 * sil) * wave * (1.0 + 0.04 * V)
    X = half * V + bend * T**2
    Y = length * T
    Z = (
        ridge * np.exp(-((V / 0.22) ** 2)) * np.sin(np.pi * T) ** 1.1
        + 0.25 * width * (V**2) * np.sin(np.pi * T) ** 1.6
    )
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    Xb, Yb, Zb = Xf.copy(), Yf.copy(), Zf - thickness
    n = len(Xf)
    verts = np.column_stack(
        (
            np.concatenate((Xf, Xb)),
            np.concatenate((Yf, Yb)),
            np.concatenate((Zf, Zb)),
        )
    )
    uvs = np.column_stack(
        (
            np.concatenate((T.ravel(), T.ravel())),
            np.concatenate(((V.ravel() + 1) / 2, (V.ravel() + 1) / 2)),
        )
    )
    faces = []
    grid_faces = _grid_faces(nu + 1, nv + 1)
    faces.extend(grid_faces)
    faces.extend([tuple(i + n for i in reversed(f)) for f in grid_faces])
    cols = nv + 1

    def f(i, j):
        return i * cols + j

    for j in range(nv):
        faces.append((f(0, j), f(0, j + 1), f(0, j + 1) + n, f(0, j) + n))
    for j in range(nv):
        faces.append((f(nu, j), f(nu, j) + n, f(nu, j + 1) + n, f(nu, j + 1)))
    for i in range(nu):
        faces.append((f(i, 0), f(i, 0) + n, f(i + 1, 0) + n, f(i + 1, 0)))
    for i in range(nu):
        faces.append((f(i, nv), f(i + 1, nv), f(i + 1, nv) + n, f(i, nv) + n))
    return verts, faces, uvs


def tube_solid(
    points: np.ndarray,
    radii: np.ndarray,
    radial: int = 8,
    caps: bool = True,
    uv_scale_v: float = 1.0,
):
    """Closed tube following points with rotation-minimising frames."""
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if len(points) < 2:
        raise ValueError("tube needs at least two points")
    if len(radii) == 1:
        radii = np.full(len(points), radii[0])
    if len(radii) != len(points):
        raise ValueError("radii must match points")
    # remove exactly duplicated consecutive points
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-7:
            keep.append(i)
    points = points[keep]
    radii = radii[keep]
    if len(points) < 2:
        points = np.stack((points[0], points[0] + np.array([0.0, 1e-4, 0.0])))
        radii = np.array([radii[0], radii[0]])
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    # initial side frame
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, tangents[0]))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    side = normalize(np.cross(helper, tangents[0]))
    sides = [side]
    for i in range(1, len(points)):
        prev_t, t = tangents[i - 1], tangents[i]
        axis = np.cross(prev_t, t)
        if np.linalg.norm(axis) > 1e-9:
            angle = np.arctan2(np.linalg.norm(axis), float(np.dot(prev_t, t)))
            rot = rotate_about(axis, angle)
            side = rot @ sides[-1]
        else:
            side = sides[-1].copy()
        side -= t * float(np.dot(side, t))
        side = normalize(side)
        sides.append(side)
    sides = np.asarray(sides)
    normals = np.cross(tangents, sides)
    verts: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    for i, (p, r) in enumerate(zip(points, radii)):
        for j in range(radial):
            a = 2.0 * np.pi * j / radial
            verts.append(p + r * (np.cos(a) * sides[i] + np.sin(a) * normals[i]))
            uvs.append((j / radial, i / max(1, len(points) - 1) * uv_scale_v))
    faces: list[tuple[int, ...]] = []
    for i in range(len(points) - 1):
        for j in range(radial):
            j2 = (j + 1) % radial
            a = i * radial + j
            b = (i + 1) * radial + j
            c = (i + 1) * radial + j2
            d = i * radial + j2
            faces.append((a, b, c, d))
    if caps:
        start_center = len(verts)
        verts.append(points[0])
        uvs.append((0.5, 0.0))
        end_center = len(verts)
        verts.append(points[-1])
        uvs.append((0.5, uv_scale_v))
        for j in range(radial):
            j2 = (j + 1) % radial
            faces.append((start_center, j2, j))
            last = (len(points) - 1) * radial
            faces.append((end_center, last + j, last + j2))
    return np.asarray(verts), faces, np.asarray(uvs)


def revolve_profile(profile: Sequence[Sequence[float]], segments: int = 96):
    """Closed solid of revolution around Y. Profile must start/end on the axis."""
    p = [as_vec(v) for v in profile]
    if abs(p[0][0]) > 1e-9 or abs(p[-1][0]) > 1e-9:
        raise ValueError("revolved profile must start and end on the Y axis")
    verts: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    rows: list[list[int] | int] = []
    total_len = sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1)) or 1.0
    acc = 0.0
    for i, (r, y) in enumerate(p):
        if i > 0:
            acc += float(np.linalg.norm(p[i] - p[i - 1]))
        if r < 1e-9:
            rows.append(len(verts))
            verts.append(np.array([0.0, y, 0.0]))
            uvs.append((0.5, acc / total_len))
        else:
            ring = []
            for j in range(segments):
                a = 2.0 * np.pi * j / segments
                ring.append(len(verts))
                verts.append(np.array([r * np.cos(a), y, r * np.sin(a)]))
                uvs.append((j / segments, acc / total_len))
            rows.append(ring)
    faces: list[tuple[int, ...]] = []
    for i in range(len(p) - 1):
        a, b = rows[i], rows[i + 1]
        if isinstance(a, int) and isinstance(b, int):
            continue
        if isinstance(a, int):
            for j in range(segments):
                j2 = (j + 1) % segments
                faces.append((a, b[j], b[j2]))
        elif isinstance(b, int):
            for j in range(segments):
                j2 = (j + 1) % segments
                faces.append((a[j], b, a[j2]))
        else:
            for j in range(segments):
                j2 = (j + 1) % segments
                faces.append((a[j], b[j], b[j2], a[j2]))
    return np.asarray(verts), faces, np.asarray(uvs)


def sphere_solid(
    center,
    radii,
    frame: np.ndarray | None = None,
    segments: int = 18,
    rings: int = 12,
):
    """Ellipsoid closed mesh. Local long axis is Z when a frame is given."""
    center = as_vec(center)
    rx, ry, rz = (float(v) for v in radii)
    if frame is None:
        frame = np.eye(3)
    verts: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    # north pole
    verts.append(center + frame @ np.array([0.0, 0.0, rz]))
    uvs.append((0.5, 0.0))
    for i in range(1, rings):
        phi = np.pi * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            local = np.array([rx * sp * np.cos(theta), ry * sp * np.sin(theta), rz * cp])
            verts.append(center + frame @ local)
            uvs.append((j / segments, i / rings))
    south = len(verts)
    verts.append(center + frame @ np.array([0.0, 0.0, -rz]))
    uvs.append((0.5, 1.0))
    faces: list[tuple[int, ...]] = []
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append((0, 1 + j, 1 + j2))
    for i in range(rings - 2):
        base = 1 + i * segments
        nxt = 1 + (i + 1) * segments
        for j in range(segments):
            j2 = (j + 1) % segments
            faces.append((base + j, nxt + j, nxt + j2, base + j2))
    base = 1 + (rings - 2) * segments
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append((south, base + j2, base + j))
    return np.asarray(verts), faces, np.asarray(uvs)


# ---------------------------------------------------------------------------
# placement helpers
# ---------------------------------------------------------------------------


def transform_solid(verts, normals_frame: np.ndarray | None, origin, faces, uvs=None):
    """Rotate a local solid and place its base/center at origin."""
    out = np.asarray(verts, dtype=np.float64)
    if normals_frame is not None:
        out = out @ normals_frame.T
    out = out + as_vec(origin)
    return out, faces, uvs


def angle_frame(azimuth: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """Frame for a petal-like blade. Local +Z is the face normal, +Y the blade direction."""
    z = np.array([np.sin(tilt) * np.cos(azimuth), np.cos(tilt), np.sin(tilt) * np.sin(azimuth)])
    z = normalize(z)
    # desired blade direction projected into plane perpendicular to z
    y_hint = np.array([np.cos(azimuth), 0.0, np.sin(azimuth)])
    y = y_hint - z * np.dot(y_hint, z)
    if np.linalg.norm(y) < 1e-7:
        y = np.array([0.0, 1.0, 0.0])
    y = normalize(y)
    x = normalize(np.cross(y, z))
    R = np.column_stack((x, y, z))
    if roll:
        R = R @ rotate_about([0.0, 0.0, 1.0], roll)
    return R
