"""晶体结构表示 — 晶格参数/分数坐标/原子类型 + CIF I/O"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pymatgen.core import Structure, Lattice, Element, Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from pyxtal import pyxtal
    HAS_PYXTAL = True
except ImportError:
    HAS_PYXTAL = False


@dataclass
class CrystalStructure:
    """晶体结构统一表示.

    Attributes:
        lattice: (3, 3) 晶格矩阵 [a1, a2, a3] (Angstrom)
        frac_coords: (N, 3) 分数坐标 [0, 1)
        atom_types: (N,) 原子序数列表
        space_group: 空间群编号 (1-230)
        properties: 可选的物性字典
    """

    lattice: np.ndarray
    frac_coords: np.ndarray
    atom_types: np.ndarray
    space_group: int = 1
    properties: Dict = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def lattice_lengths(self) -> np.ndarray:
        """晶格常数 a, b, c (Angstrom)."""
        return np.sqrt(np.sum(self.lattice ** 2, axis=1))

    @property
    def lattice_angles(self) -> np.ndarray:
        """晶格角度 alpha, beta, gamma (度)."""
        a, b, c = self.lattice_lengths
        angles = np.zeros(3)
        angles[0] = np.arccos(np.dot(self.lattice[1], self.lattice[2]) / (b * c + 1e-10))
        angles[1] = np.arccos(np.dot(self.lattice[0], self.lattice[2]) / (a * c + 1e-10))
        angles[2] = np.arccos(np.dot(self.lattice[0], self.lattice[1]) / (a * b + 1e-10))
        return np.degrees(angles)

    @property
    def volume(self) -> float:
        return abs(np.linalg.det(self.lattice))

    @property
    def density(self) -> float:
        """理论密度 (g/cm³)."""
        total_mass = sum(self._atomic_mass(z) for z in self.atom_types)
        vol_cm3 = self.volume * 1e-24
        return total_mass / (vol_cm3 * 6.022e23 + 1e-10)

    def to_pymatgen(self) -> "Structure":
        if not HAS_PYMATGEN:
            raise ImportError("pymatgen not installed")
        species = [Element.from_Z(int(z)) for z in self.atom_types]
        return Structure(
            Lattice(self.lattice),
            species,
            self.frac_coords,
            coords_are_cartesian=False,
        )

    @classmethod
    def from_pymatgen(cls, struct: "Structure", space_group: int = 1) -> "CrystalStructure":
        return cls(
            lattice=struct.lattice.matrix,
            frac_coords=struct.frac_coords,
            atom_types=np.array([s.specie.Z for s in struct.species]),
            space_group=space_group,
        )

    def to_cif_string(self) -> str:
        """导出为 CIF 格式字符串."""
        lengths = self.lattice_lengths
        angles = self.lattice_angles

        species_map = {}
        for z in self.atom_types:
            if z not in species_map:
                sym = Element.from_Z(int(z)).symbol if HAS_PYMATGEN else f"Z{z}"
                species_map[z] = sym

        lines = [
            "data_generated",
            f"_chemical_formula_sum '{self._formula_string()}'",
            "_cell_length_a " + f"{lengths[0]:.6f}",
            "_cell_length_b " + f"{lengths[1]:.6f}",
            "_cell_length_c " + f"{lengths[2]:.6f}",
            "_cell_angle_alpha " + f"{angles[0]:.4f}",
            "_cell_angle_beta " + f"{angles[1]:.4f}",
            "_cell_angle_gamma " + f"{angles[2]:.4f}",
            "_symmetry_space_group_name_H-M '" + f"{self.space_group}'",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        counts = {}
        for i, (z, coord) in enumerate(zip(self.atom_types, self.frac_coords)):
            sym = species_map.get(z, f"Z{z}")
            counts[sym] = counts.get(sym, 0) + 1
            label = f"{sym}{counts[sym]}"
            lines.append(f"  {label}  {sym}  {coord[0]:.8f}  {coord[1]:.8f}  {coord[2]:.8f}")
        return "\n".join(lines)

    @classmethod
    def from_cif_string(cls, cif_str: str) -> "CrystalStructure":
        """从CIF字符串解析晶体结构 (简化解析)."""
        if not HAS_PYMATGEN:
            raise ImportError("pymatgen required for CIF parsing")
        from pymatgen.io.cif import CifParser
        import io
        parser = CifParser(io.StringIO(cif_str))
        struct = parser.get_structures()[0]
        sg = parser.get_symops().get(0, {}).get("space_group", 1)
        if isinstance(sg, str):
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sg = SpacegroupAnalyzer(struct).get_space_group_number()
            except Exception:
                sg = 1
        return cls.from_pymatgen(struct, space_group=int(sg) if sg else 1)

    def to_tensor_dict(self) -> Dict[str, "torch.Tensor"]:
        """转为扩散模型训练用张量字典."""
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        return {
            "lattice": torch.tensor(self.lattice, dtype=torch.float32),
            "frac_coords": torch.tensor(self.frac_coords, dtype=torch.float32),
            "atom_types": torch.tensor(self.atom_types, dtype=torch.long),
            "num_atoms": self.num_atoms,
            "space_group": self.space_group,
        }

    def to_pyg_data(self):
        """转为PyG Data (用于CGCNN)."""
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        from torch_geometric.data import Data

        cutoff = 8.0
        struct = self.to_pymatgen()
        all_nbrs = struct.get_all_neighbors(cutoff, include_index=True)

        edge_src, edge_dst, edge_vecs = [], [], []
        for i, nbrs in enumerate(all_nbrs):
            for nbr in nbrs:
                edge_src.append(i)
                edge_dst.append(nbr.index)
                edge_vecs.append(nbr.coords - struct[i].coords)

        if len(edge_src) == 0:
            edge_src, edge_dst = [0, 0], [1, 1]
            edge_vecs = [[1.0, 0, 0], [-1.0, 0, 0]]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_vec = torch.tensor(edge_vecs, dtype=torch.float32)
        edge_dist = edge_vec.norm(dim=1)

        atom_features = []
        for z in self.atom_types:
            atom_features.append(_get_atom_feature(int(z)))
        x = torch.tensor(atom_features, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_dist.unsqueeze(-1),
                    pos=torch.tensor(self.frac_coords, dtype=torch.float32),
                    batch=torch.zeros(self.num_atoms, dtype=torch.long))

    def _formula_string(self) -> str:
        counts = {}
        for z in self.atom_types:
            sym = Element.from_Z(int(z)).symbol if HAS_PYMATGEN else f"Z{z}"
            counts[sym] = counts.get(sym, 0) + 1
        formula = "".join(f"{k}{v}" if v > 1 else k for k, v in sorted(counts.items()))
        return formula

    @staticmethod
    def _atomic_mass(z: int) -> float:
        masses = {1: 1.008, 3: 6.94, 4: 9.01, 5: 10.81, 6: 12.01, 7: 14.01,
                  8: 16.00, 9: 19.00, 11: 22.99, 12: 24.31, 13: 26.98, 14: 28.09,
                  15: 30.97, 16: 32.07, 17: 35.45, 19: 39.10, 20: 40.08,
                  21: 44.96, 22: 47.87, 23: 50.94, 24: 52.00, 25: 54.94,
                  26: 55.85, 27: 58.93, 28: 58.69, 29: 63.55, 30: 65.38,
                  31: 69.72, 32: 72.63, 33: 74.92, 34: 78.97, 35: 79.90,
                  37: 85.47, 38: 87.62, 39: 88.91, 40: 91.22, 41: 92.91,
                  42: 95.95, 44: 101.07, 45: 102.91, 46: 106.42, 47: 107.87,
                  48: 112.41, 49: 114.82, 50: 118.71, 51: 121.76, 52: 127.60,
                  53: 126.90, 55: 132.91, 56: 137.33, 57: 138.91, 58: 140.12,
                  59: 140.91, 60: 144.24, 62: 150.36, 64: 157.25, 66: 162.50,
                  68: 167.26, 71: 174.97, 72: 178.49, 73: 180.95, 74: 183.84,
                  75: 186.21, 76: 190.23, 77: 192.22, 78: 195.08, 79: 196.97,
                  80: 200.59, 81: 204.38, 82: 207.2, 83: 208.98}
        return masses.get(z, 100.0)

    def __repr__(self):
        return (f"CrystalStructure({self._formula_string()}, "
                f"SG={self.space_group}, atoms={self.num_atoms}, "
                f"V={self.volume:.1f} Å³)")


ATOMIC_NUM_TO_ELEM = {
    1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
    33: "As", 34: "Se", 35: "Br", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
    49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 55: "Cs", 56: "Ba",
    57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 62: "Sm", 64: "Gd", 66: "Dy",
    68: "Er", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os",
    77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi",
}
ELEM_TO_ATOMIC_NUM = {v: k for k, v in ATOMIC_NUM_TO_ELEM.items()}
MAX_ATOMIC_NUMBER = max(ATOMIC_NUM_TO_ELEM.keys())


def _get_atom_feature(z: int, dim: int = 92) -> np.ndarray:
    """原子特征: one-hot(原子序数) + 电负性/半径/价电子."""
    f = np.zeros(dim + 4, dtype=np.float32)
    if 1 <= z <= dim:
        f[int(z) - 1] = 1.0
    feats = {
        1: (2.20, 25, 1), 3: (0.98, 145, 1), 4: (1.57, 105, 2),
        5: (2.04, 85, 3), 6: (2.55, 70, 4), 7: (3.04, 65, 5),
        8: (3.44, 60, 6), 9: (3.98, 50, 7), 11: (0.93, 180, 1),
        12: (1.31, 150, 2), 13: (1.61, 125, 3), 14: (1.90, 110, 4),
        15: (2.19, 100, 5), 16: (2.58, 100, 6), 17: (3.16, 100, 7),
        19: (0.82, 220, 1), 20: (1.00, 180, 2), 21: (1.36, 160, 3),
        22: (1.54, 140, 4), 23: (1.63, 135, 5), 24: (1.66, 140, 6),
        25: (1.55, 140, 7), 26: (1.83, 140, 8), 27: (1.88, 135, 9),
        28: (1.91, 135, 10), 29: (1.90, 135, 11), 30: (1.65, 135, 12),
        31: (1.81, 130, 3), 32: (2.01, 125, 4), 33: (2.18, 115, 5),
        34: (2.55, 115, 6), 35: (2.96, 115, 7), 37: (0.82, 235, 1),
        38: (0.95, 200, 2), 39: (1.22, 180, 3), 40: (1.33, 155, 4),
        41: (1.60, 145, 5), 42: (2.16, 145, 6), 44: (2.20, 130, 8),
        45: (2.28, 135, 9), 46: (2.20, 140, 10), 47: (1.93, 160, 11),
        48: (1.69, 155, 12), 49: (1.78, 155, 3), 50: (1.96, 145, 4),
        51: (2.05, 145, 5), 52: (2.10, 140, 6), 53: (2.66, 140, 7),
        55: (0.79, 260, 1), 56: (0.89, 215, 2), 57: (1.10, 195, 3),
        72: (1.30, 155, 4), 73: (1.50, 145, 5), 74: (2.36, 135, 6),
        75: (1.90, 135, 7), 76: (2.20, 130, 8), 77: (2.20, 135, 9),
        78: (2.28, 135, 10), 79: (2.54, 135, 11), 80: (2.00, 150, 12),
        81: (1.62, 190, 3), 82: (2.33, 180, 4), 83: (2.02, 160, 5),
    }
    if z in feats:
        en, rad, ve = feats[z]
        f[dim] = en / 4.0
        f[dim + 1] = rad / 300.0
        f[dim + 2] = ve / 12.0
        f[dim + 3] = float(z) / 100.0
    return f
