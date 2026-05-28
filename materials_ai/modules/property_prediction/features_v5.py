"""特征工程 v5 — One-Hot成分编码 + 元素属性矩阵 + 工艺参数
提供三种特征表示:
  1. OneHotCompositionFeaturizer: 118维 one-hot 编码 (元素原子分数)
  2. ElementPropertyMatrix: (n_elements, n_properties) 2D矩阵, 用于GNN/Attention
  3. CombinedFeaturizer: 拼接 Magpie + OneHot + 工艺参数, 完整特征集
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pymatgen.core import Composition
from .elemental_data import ELEMENTAL_PROPERTIES, FEATURE_PROPERTY_NAMES

MAX_ATOMIC_NUMBER = 95  # 覆盖 Am (Z=95)

ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am",
]

_Z_TO_IDX = {sym: i for i, sym in enumerate(ELEMENT_SYMBOLS)}

ELEMENT_PROPERTY_LIST = [
    "atomic_number", "atomic_mass", "electronegativity", "atomic_radius",
    "ionization_energy", "electron_affinity", "group", "period",
    "valence_electrons", "melting_point", "density",
]

_DEFAULT_PROPS = {
    "atomic_number": 0, "atomic_mass": 0, "electronegativity": 0.0,
    "atomic_radius": 0.0, "ionization_energy": 0.0, "electron_affinity": 0.0,
    "group": 0, "period": 0, "valence_electrons": 0,
    "melting_point": 300, "density": 0.0,
}


class OneHotCompositionFeaturizer:
    """One-Hot成分编码: 固定95维向量, 每个位置=元素, 值=该元素在化学式中的原子分数."""

    def __init__(self, max_atomic_num: int = MAX_ATOMIC_NUMBER):
        self.max_atomic_num = max_atomic_num
        self._feature_names = None

    def featurize(self, formula: str) -> np.ndarray:
        try:
            comp = Composition(formula)
        except Exception:
            return np.zeros(self.max_atomic_num, dtype=np.float64)
        vec = np.zeros(self.max_atomic_num, dtype=np.float64)
        total = 0.0
        for el, amt in comp.get_el_amt_dict().items():
            idx = _Z_TO_IDX.get(str(el), -1)
            if 0 <= idx < self.max_atomic_num:
                vec[idx] = amt
                total += amt
        if total > 0:
            vec /= total
        return vec

    def featurize_batch(self, formulas: List[str]) -> np.ndarray:
        return np.array([self.featurize(f) for f in formulas], dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        if self._feature_names:
            return self._feature_names
        self._feature_names = [f"onehot_{sym}" for sym in ELEMENT_SYMBOLS[:self.max_atomic_num]]
        return self._feature_names

    def feature_dim(self) -> int:
        return self.max_atomic_num


class ElementPropertyMatrix:
    """元素属性矩阵: 将化学式表示为 (max_elements, n_properties) 的2D矩阵.

    每行对应一个元素, 列为其属性. 超出实际元素数的行填0.
    适用于CNN/Attention/GNN等需要空间结构的模型.
    """

    def __init__(self, max_elements: int = 8, property_list: List[str] = None):
        self.max_elements = max_elements
        self.property_list = property_list or ELEMENT_PROPERTY_LIST
        self.n_properties = len(self.property_list)

    def featurize(self, formula: str) -> np.ndarray:
        """返回 (max_elements, n_properties) 矩阵."""
        try:
            comp = Composition(formula)
        except Exception:
            return np.zeros((self.max_elements, self.n_properties), dtype=np.float64)
        elements = list(comp.elements)
        fractions = list(comp.fractional_composition.values())
        matrix = np.zeros((self.max_elements, self.n_properties), dtype=np.float64)
        for i, (el, frac) in enumerate(zip(elements, fractions)):
            if i >= self.max_elements:
                break
            props = ELEMENTAL_PROPERTIES.get(str(el), _DEFAULT_PROPS)
            for j, pname in enumerate(self.property_list):
                matrix[i, j] = props.get(pname, 0.0) * frac
        return matrix

    def featurize_batch(self, formulas: List[str]) -> np.ndarray:
        """返回 (n_samples, max_elements, n_properties) 张量."""
        return np.array([self.featurize(f) for f in formulas], dtype=np.float64)

    def feature_dim(self) -> Tuple[int, int]:
        return (self.max_elements, self.n_properties)

    def get_property_names(self) -> List[str]:
        return list(self.property_list)


class CombinedFeaturizer:
    """组合特征: Magpie(71) + OneHot(95) + 工艺参数(26) = 192维 (不含Magpie扩展)"""

    def __init__(self):
        from .features_v4 import MagpieFeaturizer
        self.magpie = MagpieFeaturizer()
        self.onehot = OneHotCompositionFeaturizer()
        self._feature_names = None

    def featurize(self, formula: str, process_params: Optional[Dict] = None) -> np.ndarray:
        mag = self.magpie.featurize(formula, process_params)
        oh = self.onehot.featurize(formula)
        return np.concatenate([mag, oh], dtype=np.float64)

    def featurize_batch(self, formulas: List[str],
                        process_params_list: Optional[List[Optional[Dict]]] = None
                        ) -> np.ndarray:
        if process_params_list is None:
            process_params_list = [None] * len(formulas)
        return np.array([self.featurize(f, p) for f, p in zip(formulas, process_params_list)])

    def get_feature_names(self) -> List[str]:
        if self._feature_names:
            return self._feature_names
        self._feature_names = (
            self.magpie.get_feature_names() + self.onehot.get_feature_names()
        )
        return self._feature_names

    def feature_dim(self) -> int:
        return self.magpie.feature_dim() + self.onehot.feature_dim()
