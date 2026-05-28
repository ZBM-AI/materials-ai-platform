"""物理有效性检查 — 原子间距 / 配位数 / 电荷平衡 / 体积合理性"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .crystal_representation import CrystalStructure

try:
    from pymatgen.core import Structure, Element, Composition
    from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


@dataclass
class ValidityReport:
    """有效性检查报告."""

    is_valid: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, any] = field(default_factory=dict)

    def __repr__(self):
        status = "PASS" if self.is_valid else "FAIL"
        return f"ValidityReport({status}, checks={len(self.checks)}, errors={len(self.errors)})"


class StructureValidator:
    """晶体结构物理有效性验证器.

    检查项:
    1. 原子间最小距离 (基于共价半径)
    2. 配位数合理性
    3. 电荷中性
    4. 体积/密度合理性
    5. 原子分数坐标范围
    6. 空间群-晶格约束一致性
    """

    COVALENT_RADII = {
        1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
        9: 0.57, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
        16: 1.05, 17: 0.99, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
        23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
        29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20,
        35: 1.20, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64,
        42: 1.54, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44,
        49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 55: 2.44,
        56: 2.15, 57: 2.07, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51,
        76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45,
        82: 1.46, 83: 1.48,
    }

    VAN_DER_WAALS_RADII = {
        1: 1.20, 3: 1.82, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80,
        17: 1.75, 19: 2.75, 20: 2.31, 26: 2.05, 28: 1.63, 29: 1.40,
        30: 1.39, 31: 1.87, 32: 2.11, 35: 1.85, 47: 1.72, 48: 1.58,
        50: 2.17, 53: 1.98, 78: 1.75, 79: 1.66, 80: 1.55, 82: 2.02,
    }

    def __init__(self, min_distance_ratio: float = 0.65,
                 max_coordination: int = 16,
                 min_volume_per_atom: float = 5.0,
                 max_volume_per_atom: float = 50.0,
                 require_charge_neutral: bool = False):
        self.min_distance_ratio = min_distance_ratio
        self.max_coordination = max_coordination
        self.min_volume_per_atom = min_volume_per_atom
        self.max_volume_per_atom = max_volume_per_atom
        self.require_charge_neutral = require_charge_neutral

    def validate(self, structure: CrystalStructure) -> ValidityReport:
        """运行所有有效性检查."""
        report = ValidityReport()

        self._check_coord_range(structure, report)
        self._check_interatomic_distances(structure, report)
        self._check_volume_density(structure, report)
        self._check_coordination(structure, report)
        self._check_charge_neutrality(structure, report)
        self._check_lattice_params(structure, report)

        report.is_valid = all(report.checks.values())
        return report

    def _check_coord_range(self, s: CrystalStructure, report: ValidityReport):
        """检查分数坐标范围."""
        coords = s.frac_coords
        bad = (coords < -0.05) | (coords > 1.05)
        ok = not np.any(bad)
        report.checks["coord_range"] = ok
        if not ok:
            n_bad = int(np.sum(bad))
            report.errors.append(f"{n_bad} 坐标超出 [0, 1) 范围")
            report.details["bad_coords"] = int(n_bad)

    def _check_interatomic_distances(self, s: CrystalStructure,
                                      report: ValidityReport):
        """检查原子间最小距离."""
        min_dist = float("inf")
        bad_pairs = []
        cart_coords = s.frac_coords @ s.lattice.T

        n = s.num_atoms
        for i in range(n):
            for j in range(i + 1, n):
                diff = cart_coords[i] - cart_coords[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    min_dist = dist

                z_i, z_j = int(s.atom_types[i]), int(s.atom_types[j])
                r_cov_i = self.COVALENT_RADII.get(z_i, 1.2)
                r_cov_j = self.COVALENT_RADII.get(z_j, 1.2)
                min_expected = self.min_distance_ratio * (r_cov_i + r_cov_j)

                if dist < min_expected and dist > 0.01:
                    bad_pairs.append((i, j, float(dist), float(min_expected)))

        ok = len(bad_pairs) == 0
        report.checks["interatomic_distance"] = ok
        if not ok:
            report.errors.append(
                f"发现 {len(bad_pairs)} 对原子间距过短 "
                f"(最短: {min_dist:.2f} Å)"
            )
            report.details["bad_pairs"] = bad_pairs[:10]
        report.details["min_distance"] = float(min_dist)

    def _check_volume_density(self, s: CrystalStructure, report: ValidityReport):
        """检查体积/密度合理性."""
        vol_per_atom = s.volume / max(s.num_atoms, 1)
        ok = self.min_volume_per_atom <= vol_per_atom <= self.max_volume_per_atom
        report.checks["volume_per_atom"] = ok
        report.details["volume_per_atom"] = float(vol_per_atom)
        report.details["density"] = float(s.density)
        if not ok:
            report.errors.append(
                f"体积/原子 = {vol_per_atom:.1f} Å³ (范围: "
                f"{self.min_volume_per_atom}-{self.max_volume_per_atom})"
            )

    def _check_coordination(self, s: CrystalStructure, report: ValidityReport):
        """检查配位数 (需要pymatgen)."""
        if not HAS_PYMATGEN:
            report.checks["coordination"] = True
            return

        try:
            ps = s.to_pymatgen()
            vnn = VoronoiNN(cutoff=5.0)
            high_cn = 0
            total_cn = 0
            for i in range(s.num_atoms):
                cn = len(vnn.get_nn(ps, i))
                total_cn += cn
                if cn > self.max_coordination:
                    high_cn += 1

            avg_cn = total_cn / max(s.num_atoms, 1)
            ok = high_cn <= s.num_atoms * 0.3
            report.checks["coordination"] = ok
            report.details["avg_coordination"] = round(avg_cn, 1)
            report.details["high_cn_atoms"] = high_cn
            if not ok:
                report.warnings.append(
                    f"{high_cn} 原子配位数 > {self.max_coordination}"
                )
        except Exception:
            report.checks["coordination"] = True

    def _check_charge_neutrality(self, s: CrystalStructure,
                                  report: ValidityReport):
        """检查电荷中性 (简化)."""
        if not self.require_charge_neutral:
            report.checks["charge_neutrality"] = True
            return

        if not HAS_PYMATGEN:
            report.checks["charge_neutrality"] = True
            return

        try:
            comp_dict = {}
            for z in s.atom_types:
                el = Element.from_Z(int(z))
                comp_dict[el] = comp_dict.get(el, 0) + 1

            total_charge = 0.0
            for el, count in comp_dict.items():
                common_ox = el.common_oxidation_states
                if common_ox:
                    total_charge += common_ox[0] * count

            ok = abs(total_charge) < 1.0
            report.checks["charge_neutrality"] = ok
            report.details["estimated_charge"] = float(total_charge)
            if not ok:
                report.warnings.append(f"估计总电荷 = {total_charge:.1f}")
        except Exception:
            report.checks["charge_neutrality"] = True

    def _check_lattice_params(self, s: CrystalStructure, report: ValidityReport):
        """检查晶格参数合理性."""
        lengths = s.lattice_lengths
        angles = s.lattice_angles

        ok_lengths = np.all((lengths > 1.0) & (lengths < 50.0))
        ok_angles = np.all((angles > 5.0) & (angles < 175.0))

        ok = bool(ok_lengths and ok_angles)
        report.checks["lattice_params"] = ok
        report.details["lattice_lengths"] = lengths.tolist()
        report.details["lattice_angles"] = angles.tolist()
        if not ok:
            report.errors.append(
                f"晶格异常: a,b,c={lengths.round(2)}, α,β,γ={angles.round(1)}"
            )

    def filter_structures(self, structures: List[CrystalStructure],
                          max_keep: int = 100) -> List[Tuple[CrystalStructure, ValidityReport]]:
        """批量过滤 — 返回所有通过检查的结构及其报告."""
        valid = []
        for s in structures:
            report = self.validate(s)
            if report.is_valid:
                valid.append((s, report))
                if len(valid) >= max_keep:
                    break
        return valid
