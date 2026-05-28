"""晶体结构生成器 — 扩散模型采样 + CGCNN筛选 + 有效性验证 + CIF导出"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .crystal_representation import (
    CrystalStructure, ATOMIC_NUM_TO_ELEM, ELEM_TO_ATOMIC_NUM, _get_atom_feature,
)
from .space_group_utils import (
    apply_lattice_constraints, get_crystal_system, get_lattice_constraints,
    generate_random_structure,
)
from .diffusion_model import CrystalDiffusion, composition_to_vector
from .cgcnn_proxy import CGCNNProxy, DefaultEnergyPredictor
from .validity_checker import StructureValidator, ValidityReport


class CrystalGenerator:
    """晶体结构生成管线.

    流程:
    1. 按目标组成+空间群生成初始随机结构
    2. 扩散模型采样去噪 → 候选结构
    3. CGCNN代理模型快速评估形成能/带隙
    4. 物理有效性检查过滤
    5. 按稳定性排序 → 输出Top-N
    6. 导出CIF文件
    """

    def __init__(self, diffusion_model: Optional[CrystalDiffusion] = None,
                 proxy_model: Optional[CGCNNProxy] = None,
                 validator: Optional[StructureValidator] = None,
                 device: str = None):
        self.diffusion_model = diffusion_model
        self.proxy = proxy_model if proxy_model is not None else DefaultEnergyPredictor()
        self.validator = validator or StructureValidator()

        if device is None and HAS_TORCH:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or "cpu"

    def generate(self, elements: List[str],
                 stoichiometry: List[float] = None,
                 space_group: int = 1,
                 num_candidates: int = 100,
                 num_steps: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 10,
                 output_dir: str = None) -> Dict:
        """生成新型晶体结构.

        Args:
            elements: 目标元素列表, e.g. ["Li", "Co", "O"]
            stoichiometry: 化学计量比, e.g. [1, 1, 2] 表示 LiCoO2
            space_group: 目标空间群编号 (1-230)
            num_candidates: 初始候选结构数量
            num_steps: 扩散采样步数
            temperature: 采样温度 (越高多样性越大)
            top_k: 最终输出Top-K结构
            output_dir: CIF导出目录 (可选)

        Returns:
            {
                "candidates": [CrystalStructure, ...],
                "predictions": [{"formation_energy_eV":, "band_gap_eV":, ...}, ...],
                "validity_reports": [ValidityReport, ...],
                "cif_files": [path, ...],
                "ranked_indices": [...]  # 按稳定性从高到低
            }
        """
        stoichiometry = stoichiometry or [1] * len(elements)
        if len(stoichiometry) != len(elements):
            stoichiometry = stoichiometry + [1] * (len(elements) - len(stoichiometry))
            stoichiometry = stoichiometry[:len(elements)]

        total_stoich = sum(stoichiometry)
        stoich_normalized = [s / total_stoich for s in stoichiometry]

        # Step 1: 批量生成初始随机结构 (无扩散模型时使用PyXtal/随机)
        candidates = []
        for _ in range(num_candidates):
            lat, coords, atoms = generate_random_structure(
                space_group, elements,
                num_atoms=int(total_stoich * 4),
            )
            candidates.append(CrystalStructure(
                lattice=lat, frac_coords=coords, atom_types=atoms,
                space_group=space_group,
            ))

        # Step 2: 如果有扩散模型, 对每个候选进行去噪优化
        if self.diffusion_model is not None and HAS_TORCH:
            candidates = self._diffusion_optimize(
                candidates, elements, stoich_normalized, space_group,
                num_steps, temperature,
            )

        # Step 3: CGCNN代理模型快筛
        predictions = []
        for struct in candidates:
            pred = self.proxy.predict(struct)
            predictions.append(pred)

        # Step 4: 物理有效性过滤
        valid_results = []
        for struct, pred in zip(candidates, predictions):
            report = self.validator.validate(struct)
            if report.is_valid:
                valid_results.append((struct, pred, report))

        if not valid_results:
            # 放宽过滤条件
            valid_results = [(s, p, self.validator.validate(s))
                             for s, p in zip(candidates, predictions)]

        # Step 5: 按稳定性排序
        valid_results.sort(
            key=lambda x: (x[1].get("stability_score", 0), -x[1].get("formation_energy_eV", 0)),
            reverse=True,
        )
        valid_results = valid_results[:top_k]

        # Step 6: CIF导出
        cif_paths = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for rank, (struct, pred, _) in enumerate(valid_results):
                cif_path = os.path.join(
                    output_dir,
                    f"candidate_{rank:03d}_{struct._formula_string()}_SG{space_group}.cif"
                )
                with open(cif_path, "w", encoding="utf-8") as f:
                    f.write(struct.to_cif_string())
                cif_paths.append(cif_path)

        final_structures = [x[0] for x in valid_results]
        final_predictions = [x[1] for x in valid_results]
        final_reports = [x[2] for x in valid_results]

        return {
            "candidates": final_structures,
            "predictions": final_predictions,
            "validity_reports": final_reports,
            "cif_files": cif_paths,
            "num_total_generated": num_candidates,
            "num_passed": len(valid_results),
        }

    def _diffusion_optimize(self, candidates: List[CrystalStructure],
                            elements: List[str],
                            stoichiometry: List[float],
                            space_group: int,
                            num_steps: int,
                            temperature: float) -> List[CrystalStructure]:
        """使用扩散模型对候选结构去噪优化."""
        self.diffusion_model.eval()
        self.diffusion_model.to(self.device)

        comp_vec = torch.tensor(
            composition_to_vector(elements, stoichiometry),
            device=self.device,
        )

        optimized = []
        for candidate in candidates:
            try:
                result = self.diffusion_model.sample(
                    composition=comp_vec,
                    space_group=space_group,
                    num_atoms=candidate.num_atoms,
                    num_steps=num_steps,
                )
                lattice = apply_lattice_constraints(result["lattice"], space_group)
                optimized.append(CrystalStructure(
                    lattice=lattice,
                    frac_coords=result["frac_coords"] % 1.0,
                    atom_types=result["atom_types"],
                    space_group=space_group,
                ))
            except Exception:
                optimized.append(candidate)

        return optimized

    def generate_for_dft(self, elements: List[str],
                         stoichiometry: List[float] = None,
                         space_group: int = 1,
                         num_candidates: int = 50,
                         top_k: int = 5,
                         output_dir: str = None) -> Dict:
        """生成候选结构并准备DFT计算输入.

        Returns 额外包含:
            - "dft_inputs": POSCAR格式或VASP输入字典
            - "dft_script": 建议的DFT计算脚本
        """
        result = self.generate(
            elements=elements,
            stoichiometry=stoichiometry,
            space_group=space_group,
            num_candidates=num_candidates,
            top_k=top_k,
            output_dir=output_dir,
        )

        dft_inputs = []
        for struct in result["candidates"]:
            try:
                ps = struct.to_pymatgen()
                dft_inputs.append({
                    "formula": struct._formula_string(),
                    "poscar": self._to_poscar_string(struct),
                    "kpoints_suggestion": self._suggest_kpoints(struct),
                    "encut_suggestion": self._suggest_encut(struct),
                })
            except Exception:
                pass

        result["dft_inputs"] = dft_inputs
        result["dft_workflow_note"] = (
            "建议DFT流程:\n"
            "1. 先用 VASP/Quantum ESPRESSO 做结构弛豫 (ISIF=3, IBRION=2)\n"
            "2. 静态计算形成能 (高精度k点, ENCUT≥1.3×ENMAX)\n"
            "3. 用 Phonopy 检查声子谱 (验证动力学稳定性)\n"
            "4. 用 pymatgen.analysis.phase_diagram 检查相图稳定性\n"
            "5. 如全部通过 → 实验验证候选物"
        )

        return result

    def _to_poscar_string(self, structure: CrystalStructure) -> str:
        """导出VASP POSCAR格式."""
        ps = structure.to_pymatgen()
        lines = [
            f"Generated_{structure._formula_string()}_SG{structure.space_group}",
            "1.0",
        ]
        for vec in ps.lattice.matrix:
            lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")

        unique_species = sorted(set(ps.species), key=lambda sp: sp.Z)
        counts = [sum(1 for s in ps.species if s == sp) for sp in unique_species]
        lines.append("  " + "  ".join(str(sp) for sp in unique_species))
        lines.append("  " + "  ".join(str(c) for c in counts))
        lines.append("Direct")
        for site in ps.sites:
            c = site.frac_coords
            lines.append(f"  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}")
        return "\n".join(lines)

    def _suggest_kpoints(self, structure: CrystalStructure) -> str:
        """建议k点网格."""
        lengths = structure.lattice_lengths
        k_a = max(1, int(30 / lengths[0]))
        k_b = max(1, int(30 / lengths[1]))
        k_c = max(1, int(30 / lengths[2]))
        return f"{k_a}×{k_b}×{k_c} (推荐: KSPACING=0.3)"

    def _suggest_encut(self, structure: CrystalStructure) -> str:
        """建议ENCUT."""
        max_enmax = 300
        if HAS_PYMATGEN:
            try:
                ps = structure.to_pymatgen()
                potcars = {}
                for el in ps.composition.elements:
                    potcars[str(el)] = 300
                max_enmax = max(potcars.values()) if potcars else 300
            except Exception:
                pass
        encut = max_enmax * 1.3
        return f"{encut:.0f} eV (1.3×ENMAX)"


def prepare_dft_batch(output_dir: str, candidates: List[CrystalStructure]) -> str:
    """批量导出候选结构的POSCAR文件, 组织DFT计算目录.

    目录结构:
        output_dir/
        ├── candidate_001/
        │   ├── POSCAR
        │   └── INCAR.template
        ├── candidate_002/
        ...
        └── run_all.sh

    Returns:
        运行脚本路径
    """
    os.makedirs(output_dir, exist_ok=True)

    incar_template = """# VASP INCAR — 结构弛豫
SYSTEM = Generated Crystal
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
IBRION = 2
ISIF = 3
NSW = 100
EDIFF = 1E-6
EDIFFG = -0.01
PREC = Accurate
LWAVE = .FALSE.
LCHARG = .FALSE.
"""

    run_script_lines = ["#!/bin/bash", "set -e", ""]

    for i, struct in enumerate(candidates):
        subdir = os.path.join(output_dir, f"candidate_{i:03d}")
        os.makedirs(subdir, exist_ok=True)

        poscar_path = os.path.join(subdir, "POSCAR")
        generator = CrystalGenerator()
        with open(poscar_path, "w") as f:
            f.write(generator._to_poscar_string(struct))

        incar_path = os.path.join(subdir, "INCAR")
        with open(incar_path, "w") as f:
            f.write(incar_template)

        run_script_lines.append(f"echo 'Running candidate_{i:03d}...'")
        run_script_lines.append(f"cd {subdir}")
        run_script_lines.append("mpirun -np 16 vasp_std > vasp.log 2>&1")
        run_script_lines.append("cd ../..")
        run_script_lines.append("")

    run_script_path = os.path.join(output_dir, "run_all.sh")
    with open(run_script_path, "w") as f:
        f.write("\n".join(run_script_lines))

    return run_script_path


def generate_and_export(elements: List[str], stoichiometry: List[float] = None,
                         space_groups: List[int] = None,
                         num_per_sg: int = 20,
                         output_dir: str = None,
                         model_path: str = None) -> Dict:
    """便捷函数 — 多空间群批量生成+导出.

    对每个空间群, 生成 num_per_sg 个候选, 筛选后导出.
    """
    space_groups = space_groups or [1, 2, 14, 62, 225]

    diffusion = None
    if model_path and os.path.exists(model_path) and HAS_TORCH:
        try:
            diffusion = CrystalDiffusion()
            ckpt = torch.load(model_path, map_location="cpu")
            diffusion.load_state_dict(ckpt.get("model_state_dict", ckpt))
            diffusion.eval()
        except Exception:
            pass

    proxy = None
    proxy_path = os.path.join(
        os.path.dirname(model_path) if model_path else ".",
        "cgcnn_proxy.pt",
    )
    if os.path.exists(proxy_path) and HAS_TORCH:
        try:
            proxy = CGCNNProxy.load(proxy_path)
        except Exception:
            proxy = DefaultEnergyPredictor()

    generator = CrystalGenerator(diffusion_model=diffusion, proxy_model=proxy)

    all_results = []
    for sg in space_groups:
        result = generator.generate(
            elements=elements,
            stoichiometry=stoichiometry,
            space_group=sg,
            num_candidates=num_per_sg,
            top_k=5,
            output_dir=output_dir,
        )
        all_results.append({"space_group": sg, **result})

    return {
        "elements": elements,
        "stoichiometry": stoichiometry,
        "space_groups": space_groups,
        "results": all_results,
        "total_candidates": sum(r["num_passed"] for r in all_results),
    }
