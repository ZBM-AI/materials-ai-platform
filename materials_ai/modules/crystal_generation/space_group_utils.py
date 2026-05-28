"""空间群约束 — Wyckoff位置生成 / 对称性操作 / 空间群验证"""

import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    from pyxtal import pyxtal as _pyxtal_cls
    HAS_PYXTAL = True
except ImportError:
    HAS_PYXTAL = False


SPACE_GROUP_NAMES = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P2_1", 5: "C2",
    14: "P2_1/c", 15: "C2/c", 19: "P2_1 2_1 2_1",
    33: "Pna2_1", 61: "Pbca", 62: "Pnma",
    129: "P4/nmm", 139: "I4/mmm",
    166: "R-3m", 167: "R-3c",
    191: "P6/mmm", 194: "P6_3/mmc",
    221: "Pm-3m", 225: "Fm-3m", 227: "Fd-3m", 229: "Im-3m",
}

CRYSTAL_SYSTEMS = {
    "triclinic": list(range(1, 3)),
    "monoclinic": list(range(3, 16)),
    "orthorhombic": list(range(16, 75)),
    "tetragonal": list(range(75, 143)),
    "trigonal": list(range(143, 168)),
    "hexagonal": list(range(168, 195)),
    "cubic": list(range(195, 231)),
}


def get_crystal_system(sg: int) -> str:
    for system, sgs in CRYSTAL_SYSTEMS.items():
        if sg in sgs:
            return system
    return "unknown"


def get_lattice_constraints(sg: int) -> Dict[str, str]:
    """根据空间群返回晶格参数约束."""
    system = get_crystal_system(sg)
    if system == "triclinic":
        return {"a": "free", "b": "free", "c": "free",
                "alpha": "free", "beta": "free", "gamma": "free"}
    elif system == "monoclinic":
        return {"alpha": "90", "gamma": "90",
                "a": "free", "b": "free", "c": "free", "beta": "free"}
    elif system == "orthorhombic":
        return {"alpha": "90", "beta": "90", "gamma": "90",
                "a": "free", "b": "free", "c": "free"}
    elif system == "tetragonal":
        return {"alpha": "90", "beta": "90", "gamma": "90",
                "a": "free", "b": "=a", "c": "free"}
    elif system in ("trigonal", "hexagonal"):
        return {"alpha": "90", "beta": "90", "gamma": "120",
                "a": "free", "b": "=a", "c": "free"}
    elif system == "cubic":
        return {"alpha": "90", "beta": "90", "gamma": "90",
                "a": "free", "b": "=a", "c": "=a"}
    return {}


def apply_lattice_constraints(lattice: np.ndarray, sg: int) -> np.ndarray:
    """按空间群约束修正晶格矩阵."""
    constraints = get_lattice_constraints(sg)
    lengths = np.sqrt(np.sum(lattice ** 2, axis=1))
    constrained = lattice.copy()

    if constraints.get("b") == "=a":
        factor_b = lengths[0] / max(lengths[1], 1e-10)
        constrained[1] = constrained[1] * factor_b / np.linalg.norm(constrained[1])
        constrained[1] = constrained[1] * lengths[0] / (np.linalg.norm(constrained[1]) + 1e-10)
    if constraints.get("c") == "=a":
        constrained[2] = constrained[2] * lengths[0] / (np.linalg.norm(constrained[2]) + 1e-10)

    alpha_target = float(constraints.get("alpha", "free").replace("free", "-1"))
    beta_target = float(constraints.get("beta", "free").replace("free", "-1"))
    gamma_target = float(constraints.get("gamma", "free").replace("free", "-1"))

    for name, target, idx0, idx1 in [
        ("alpha", alpha_target, 1, 2),
        ("beta", beta_target, 0, 2),
        ("gamma", gamma_target, 0, 1),
    ]:
        if target > 0:
            v0 = constrained[idx0]
            v1 = constrained[idx1]
            l0 = np.linalg.norm(v0)
            l1 = np.linalg.norm(v1)
            current_angle = np.arccos(
                np.clip(np.dot(v0, v1) / (l0 * l1 + 1e-10), -1.0, 1.0)
            )
            target_rad = np.radians(target)
            rot_axis = np.cross(v0, v1)
            if np.linalg.norm(rot_axis) > 1e-10:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                cos_a = np.cos(target_rad)
                sin_a = np.sin(target_rad)
                K = np.array([
                    [0, -rot_axis[2], rot_axis[1]],
                    [rot_axis[2], 0, -rot_axis[0]],
                    [-rot_axis[1], rot_axis[0], 0],
                ])
                R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
                constrained[idx1] = R @ v1 * (l1 / (np.linalg.norm(R @ v1) + 1e-10))

    return constrained


def generate_random_structure(sg: int, elements: List[str],
                              num_atoms: int = None,
                              volume_per_atom: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用PyXtal生成给定空间群的随机初始结构.

    Returns:
        lattice (3,3), frac_coords (N,3), atom_types (N,)
    """
    if not HAS_PYXTAL:
        return _random_structure_fallback(sg, elements, num_atoms, volume_per_atom)

    from pyxtal import pyxtal

    num_atoms = num_atoms or len(elements) * 4
    atom_counts = [max(1, num_atoms // len(elements)) for _ in elements]
    for i in range(num_atoms - sum(atom_counts)):
        atom_counts[i % len(elements)] += 1

    species = []
    for el, count in zip(elements, atom_counts):
        species.extend([el] * count)

    xtal = pyxtal()
    xtal.from_random(3, sg, species, volume_per_atom * len(species))

    atom_types = np.array([ELEM_TO_ATOMIC_NUM.get(el, 6) for el in species])
    return xtal.lattice.matrix, xtal.frac_coords, atom_types


def _random_structure_fallback(sg: int, elements: List[str],
                                num_atoms: int, volume_per_atom: float):
    """无PyXtal时的随机结构生成降级方案."""
    from .crystal_representation import ATOMIC_NUM_TO_ELEM, ELEM_TO_ATOMIC_NUM

    num_atoms = num_atoms or max(len(elements) * 4, 8)
    atom_types_list = []
    for i in range(num_atoms):
        el = elements[i % len(elements)]
        atom_types_list.append(ELEM_TO_ATOMIC_NUM.get(el, 6))

    total_vol = num_atoms * volume_per_atom

    system = get_crystal_system(sg)
    constraints = get_lattice_constraints(sg)

    if system == "cubic":
        a = total_vol ** (1 / 3)
        lattice = np.diag([a, a, a])
    elif system in ("tetragonal", "hexagonal", "trigonal"):
        c_over_a = 1.5 + np.random.uniform(-0.3, 0.3)
        a = (total_vol / c_over_a) ** (1 / 3)
        c = a * c_over_a
        if system == "hexagonal":
            lattice = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        else:
            lattice = np.diag([a, a, c])
    elif system == "orthorhombic":
        a = (total_vol * np.random.uniform(0.8, 1.2)) ** (1 / 3)
        b = (total_vol / a * np.random.uniform(0.8, 1.2)) ** (1 / 2)
        c = total_vol / (a * b)
        lattice = np.diag([a, b, c])
    else:
        a = (total_vol * 1.0) ** (1 / 3)
        b = (total_vol * 1.3) ** (1 / 3)
        c = total_vol / (a * b)
        beta = np.radians(90 + np.random.uniform(-5, 5))
        lattice = np.array([
            [a, 0, 0],
            [0, b, 0],
            [c * np.cos(beta), 0, c * np.sin(beta)],
        ])

    lattice = apply_lattice_constraints(lattice, sg)

    frac_coords = np.random.uniform(0.05, 0.95, (num_atoms, 3))
    min_dist = 0.15 / (num_atoms ** (1 / 3))
    for _ in range(20):
        too_close = False
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                diff = frac_coords[i] - frac_coords[j]
                diff = diff - np.round(diff)
                cart_diff = lattice.T @ diff
                dist = np.linalg.norm(cart_diff)
                if dist < min_dist:
                    frac_coords[j] = np.random.uniform(0.05, 0.95, 3)
                    too_close = True
        if not too_close:
            break

    return (lattice.astype(np.float32),
            frac_coords.astype(np.float32),
            np.array(atom_types_list, dtype=np.int32))


def get_wyckoff_multiplicity(sg: int, wyckoff_letter: str = "a") -> int:
    """获取Wyckoff位置多重度."""
    if not HAS_PYXTAL:
        return 1
    try:
        from pyxtal.symmetry import Group
        g = Group(sg)
        for wp in g:
            if wp.letter == wyckoff_letter:
                return wp.multiplicity
    except Exception:
        pass
    return 1


def get_symmetry_equivalent_positions(sg: int) -> List[np.ndarray]:
    """获取空间群对称操作矩阵列表 (3x4 affine)."""
    ops = [np.eye(4)]
    if not HAS_PYXTAL:
        return ops
    try:
        from pyxtal.symmetry import Group
        g = Group(sg)
        for op in g:
            if op is not None:
                affine = np.eye(4)
                affine[:3, :3] = op.rotation_matrix
                affine[:3, 3] = op.translation_vector
                ops.append(affine)
    except Exception:
        pass
    return ops


def reduce_to_asymmetric_unit(frac_coords: np.ndarray, sg: int) -> np.ndarray:
    """将分数坐标折叠到非对称单元 (简化)."""
    coords = frac_coords % 1.0
    return coords
