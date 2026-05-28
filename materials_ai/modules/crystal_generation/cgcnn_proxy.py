"""CGCNN 代理模型 — 快速预测形成能 / 带隙 / 稳定性

作为生成结构的高通量筛选层, 在DFT验证之前快速过滤不稳定的候选结构.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import CGConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

from .crystal_representation import CrystalStructure, _get_atom_feature


class CGCNNProxy(nn.Module):
    """轻量CGCNN — 晶体图卷积网络代理模型.

    输入: 晶体图 (原子特征 + 边距离)
    输出: 形成能 (eV/atom) + 带隙 (eV)
    """

    def __init__(self, node_dim: int = 96, edge_dim: int = 1,
                 hidden_dim: int = 128, num_layers: int = 3,
                 num_targets: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList([
            CGConv(hidden_dim, edge_dim, aggr="mean", batch_norm=True)
            for _ in range(num_layers)
        ])

        self.pool = global_mean_pool
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_targets),
        )

        self.num_targets = num_targets
        self._norm_mean = None
        self._norm_std = None

    def forward(self, data):
        x = self.node_embed(data.x)
        for conv in self.convs:
            x = x + conv(x, data.edge_index, data.edge_attr)
        pooled = self.pool(x, data.batch)
        out = self.head(pooled)
        return out

    def predict(self, structure: CrystalStructure) -> Dict[str, float]:
        """预测单个晶体结构的性质.

        Returns:
            {"formation_energy": eV/atom, "band_gap": eV,
             "stability_score": 0-1, "is_stable": bool}
        """
        self.eval()
        with torch.no_grad():
            data = structure.to_pyg_data()
            if data.x is None or data.edge_index is None:
                return self._fallback_predict()

            logits = self.forward(data).squeeze(0)

            if self._norm_mean is not None:
                logits = logits * self._norm_std + self._norm_mean

            e_form = float(logits[0].item())
            band_gap = float(logits[1].item()) if logits.shape[0] > 1 else 0.0

        # 稳定性评分: 形成能越负越稳定, 带隙>0更可能稳定
        stability = 1.0 / (1.0 + np.exp(e_form + 1.0))
        if band_gap < 0:
            stability *= 0.5
        stability = max(0.0, min(1.0, stability))

        return {
            "formation_energy_eV": round(e_form, 4),
            "band_gap_eV": round(max(0.0, band_gap), 4),
            "stability_score": round(stability, 4),
            "is_stable": e_form < 0 and stability > 0.3,
        }

    def batch_predict(self, structures: List[CrystalStructure]) -> List[Dict]:
        """批量预测."""
        self.eval()
        results = []
        for struct in structures:
            results.append(self.predict(struct))
        return results

    def _fallback_predict(self) -> Dict:
        """无PyG时的规则基线预测."""
        return {
            "formation_energy_eV": 0.0,
            "band_gap_eV": 0.0,
            "stability_score": 0.5,
            "is_stable": False,
        }

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """设置输出归一化参数."""
        self._norm_mean = torch.tensor(mean, dtype=torch.float32)
        self._norm_std = torch.tensor(std, dtype=torch.float32)

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "norm_mean": self._norm_mean,
            "norm_std": self._norm_std,
            "node_dim": self.node_dim,
            "hidden_dim": self.hidden_dim,
            "num_targets": self.num_targets,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CGCNNProxy":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(
            node_dim=ckpt.get("node_dim", 96),
            hidden_dim=ckpt.get("hidden_dim", 128),
            num_targets=ckpt.get("num_targets", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("norm_mean") is not None:
            model._norm_mean = ckpt["norm_mean"]
            model._norm_std = ckpt["norm_std"]
        return model


class DefaultEnergyPredictor:
    """无PyG时的替代方案 — 基于组成和体积的经验预测.

    使用pymatgen的formation_energy预测器或简单规则.
    """

    def __init__(self):
        self._use_pymatgen = HAS_PYMATGEN

    def predict(self, structure: CrystalStructure) -> Dict[str, float]:
        if not self._use_pymatgen:
            return self._rule_based_predict(structure)

        try:
            from pymatgen.analysis.energy_models import EwaldElectrostaticModel
            s = structure.to_pymatgen()

            # 基于体积-电负性的简单形成能估计
            en_sum = 0.0
            for site in s.sites:
                el = site.specie
                en_sum += el.X if hasattr(el, 'X') else 1.5
            avg_en = en_sum / max(len(s.sites), 1)

            vol_per_atom = structure.volume / max(structure.num_atoms, 1)

            e_form_est = -0.5 * avg_en + 0.02 * (vol_per_atom - 20)
            band_gap_est = max(0.0, 0.3 * avg_en - 0.5)

            stability = 1.0 / (1.0 + np.exp(e_form_est + 1.0))

            return {
                "formation_energy_eV": round(float(e_form_est), 4),
                "band_gap_eV": round(float(band_gap_est), 4),
                "stability_score": round(float(stability), 4),
                "is_stable": e_form_est < 0 and stability > 0.3,
            }
        except Exception:
            return self._rule_based_predict(structure)

    def _rule_based_predict(self, structure: CrystalStructure) -> Dict:
        vol_per_atom = structure.volume / max(structure.num_atoms, 1)
        e_form = -1.5 + 0.03 * (vol_per_atom - 18)
        band_gap = max(0.0, 1.5 - 0.05 * vol_per_atom)
        stability = 1.0 / (1.0 + np.exp(e_form + 1.0))

        return {
            "formation_energy_eV": round(float(e_form), 4),
            "band_gap_eV": round(float(band_gap), 4),
            "stability_score": round(float(stability), 4),
            "is_stable": e_form < 0 and stability > 0.3,
        }
