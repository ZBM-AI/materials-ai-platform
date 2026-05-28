"""特征工程 v4 — Magpie特征、元素属性、工艺参数编码"""

import numpy as np
from typing import List, Dict, Optional
from pymatgen.core import Composition
from .elemental_data import ELEMENTAL_PROPERTIES, FEATURE_PROPERTY_NAMES, BLOCK_MAP

PROCESSING_METHODS = [
    "solid_state", "sol_gel", "hydrothermal", "co_precipitation",
    "ball_milling", "CVD", "PVD", "spray_pyrolysis", "molten_salt",
    "microwave", "spark_plasma_sintering", "hot_press", "cold_press",
]

ATMOSPHERE_TYPES = ["air", "N2", "Ar", "O2", "H2", "vacuum", "NH3", "H2S"]


class ProcessParameterEncoder:
    """将材料制备工艺参数编码为数值特征向量 (16维)"""

    def __init__(self):
        self.method_names = PROCESSING_METHODS
        self.atmosphere_names = ATMOSPHERE_TYPES

    def encode(self, params: Optional[Dict] = None) -> np.ndarray:
        """编码工艺参数. params可包含: temperature, pressure, time, method, atmosphere, annealing_temp, annealing_time"""
        if params is None:
            params = {}
        features = []
        temp = float(params.get("temperature", 300))
        features.append(temp / 2000.0)
        features.append(float(params.get("pressure", 1.0)) / 100.0)
        features.append(float(params.get("time", 24.0)) / 168.0)
        features.append(float(params.get("annealing_temp", temp * 0.6)) / 2000.0)
        features.append(float(params.get("annealing_time", 12.0)) / 168.0)
        method = str(params.get("method", "solid_state")).lower().replace("-", "_").replace(" ", "_")
        for m in self.method_names:
            features.append(1.0 if m == method else 0.0)
        atmosphere = str(params.get("atmosphere", "air"))
        for a in self.atmosphere_names:
            features.append(1.0 if a == atmosphere else 0.0)
        return np.array(features, dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        names = ["temperature_norm", "pressure_norm", "time_norm",
                 "annealing_temp_norm", "annealing_time_norm"]
        names.extend([f"method_{m}" for m in self.method_names])
        names.extend([f"atm_{a}" for a in self.atmosphere_names])
        return names

    def dim(self) -> int:
        return 5 + len(self.method_names) + len(self.atmosphere_names)


class MagpieFeaturizer:
    """Magpie风格特征: 元素属性统计 + 工艺参数 = 87维"""

    def __init__(self):
        self.property_names = FEATURE_PROPERTY_NAMES
        self.process_encoder = ProcessParameterEncoder()
        self._feature_names = None

    def featurize(self, formula: str, process_params: Optional[Dict] = None) -> np.ndarray:
        try:
            comp = Composition(formula)
        except Exception:
            comp = Composition("H")
        features = []
        features.extend(self._stoichiometric_features(comp))
        for prop in self.property_names:
            features.extend(self._property_stats(comp, prop))
        features.extend(self._block_features(comp))
        features.extend(self._compound_descriptors(comp))
        features.extend(self.process_encoder.encode(process_params))
        return np.array(features, dtype=np.float64)

    def featurize_batch(self, formulas: List[str],
                        process_params_list: Optional[List[Optional[Dict]]] = None) -> np.ndarray:
        if process_params_list is None:
            process_params_list = [None] * len(formulas)
        return np.array([self.featurize(f, p) for f, p in zip(formulas, process_params_list)])

    def get_feature_names(self) -> List[str]:
        if self._feature_names:
            return self._feature_names
        names = []
        names.extend(["num_elements", "min_fraction", "max_fraction",
                      "comp_entropy", "l2_norm", "mean_atomic_num"])
        for prop in self.property_names:
            for stat in ["mean", "max", "min", "std", "weighted_avg"]:
                names.append(f"{prop}_{stat}")
        names.extend(["s_block_frac", "p_block_frac", "d_block_frac", "f_block_frac",
                      "d_filling_ratio"])
        names.extend(["en_diff", "en_range_ratio", "radius_range", "radius_mismatch",
                      "avg_valence", "total_valence", "ionic_char",
                      "delta_h_mix", "packing_diff", "density_estimate"])
        names.extend(self.process_encoder.get_feature_names())
        self._feature_names = names
        return names

    def feature_dim(self) -> int:
        return len(self.get_feature_names())

    def _stoichiometric_features(self, comp: Composition) -> List[float]:
        fractions = list(comp.fractional_composition.values())
        atomic_nums = [ELEMENTAL_PROPERTIES.get(str(el), {}).get("atomic_number", 0)
                       for el in comp.elements]
        n_elements = len(fractions)
        min_frac = min(fractions) if fractions else 0
        max_frac = max(fractions) if fractions else 0
        entropy = -sum(f * np.log(f + 1e-12) for f in fractions)
        l2_norm = np.sqrt(sum(f * f for f in fractions))
        mean_an = np.mean(atomic_nums) if atomic_nums else 0
        return [n_elements, min_frac, max_frac, entropy, l2_norm, mean_an]

    def _property_stats(self, comp: Composition, prop: str) -> List[float]:
        elements = list(comp.elements)
        fractions = list(comp.fractional_composition.values())
        values = []
        for el in elements:
            props = ELEMENTAL_PROPERTIES.get(str(el), {})
            values.append(props.get(prop, 0.0))
        values = np.array(values, dtype=np.float64)
        fractions = np.array(fractions, dtype=np.float64)
        return [
            float(np.mean(values)),
            float(np.max(values)),
            float(np.min(values)),
            float(np.std(values)) if len(values) > 1 else 0.0,
            float(np.average(values, weights=fractions)),
        ]

    def _block_features(self, comp: Composition) -> List[float]:
        elements = list(comp.elements)
        fractions = list(comp.fractional_composition.values())
        s_frac, p_frac, d_frac, f_frac = 0.0, 0.0, 0.0, 0.0
        d_electrons = 0.0
        for el, frac in zip(elements, fractions):
            block = ELEMENTAL_PROPERTIES.get(str(el), {}).get("block", "s")
            ve = ELEMENTAL_PROPERTIES.get(str(el), {}).get("valence_electrons", 0)
            if block == "s":
                s_frac += frac
            elif block == "p":
                p_frac += frac
            elif block == "d":
                d_frac += frac
                d_electrons += frac * max(ve - 2, 0)
            elif block == "f":
                f_frac += frac
        d_filling = d_electrons / max(d_frac, 0.01) / 10.0
        return [s_frac, p_frac, d_frac, f_frac, d_filling]

    def _compound_descriptors(self, comp: Composition) -> List[float]:
        elements = list(comp.elements)
        fractions = np.array(list(comp.fractional_composition.values()))
        en_values = np.array([ELEMENTAL_PROPERTIES.get(str(el), {}).get("electronegativity", 0.0)
                              for el in elements])
        radius_values = np.array([ELEMENTAL_PROPERTIES.get(str(el), {}).get("atomic_radius", 0.0)
                                  for el in elements])
        valence_values = np.array([ELEMENTAL_PROPERTIES.get(str(el), {}).get("valence_electrons", 0)
                                   for el in elements])
        density_values = np.array([ELEMENTAL_PROPERTIES.get(str(el), {}).get("density", 0.0)
                                   for el in elements])
        en_diff = float(np.max(en_values) - np.min(en_values))
        en_range_ratio = float(en_diff / max(np.mean(en_values), 0.01))
        radius_range = float(np.max(radius_values) - np.min(radius_values))
        mean_radius = float(np.average(radius_values, weights=fractions))
        radius_mismatch = float(np.sqrt(np.average(
            (radius_values - mean_radius) ** 2, weights=fractions)) / max(mean_radius, 0.01))
        avg_valence = float(np.average(valence_values, weights=fractions))
        total_valence = float(np.sum(valence_values * fractions))
        n_elements = len(elements)
        ionic_char = 1.0 - np.exp(-0.25 * en_diff ** 2)
        delta_h_mix = float(0.0)
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                ri, rj = radius_values[i], radius_values[j]
                ci, cj = fractions[i], fractions[j]
                r_avg = (ri + rj) / 2.0
                delta_h_mix += 4.0 * ci * cj * ((ri - rj) / max(r_avg, 0.01)) ** 2
        packing_diff = float(np.std(radius_values) / max(np.mean(radius_values), 0.01))
        density_estimate = float(np.average(density_values, weights=fractions))
        return [en_diff, en_range_ratio, radius_range, radius_mismatch,
                avg_valence, total_valence, ionic_char,
                delta_h_mix, packing_diff, density_estimate]
