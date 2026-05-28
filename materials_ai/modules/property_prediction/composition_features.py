"""化学成分特征工程 v3 — 71维，修复vec冗余"""

import numpy as np
from typing import List
from pymatgen.core import Composition
from .elemental_data import ELEMENTAL_PROPERTIES, FEATURE_PROPERTY_NAMES, BLOCK_MAP


class CompositionFeaturizer:
    """将化学式转换为固定长度特征向量 (71维)"""

    def __init__(self):
        self.property_names = FEATURE_PROPERTY_NAMES
        self._feature_names = None

    def featurize(self, formula: str) -> np.ndarray:
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
        return np.array(features, dtype=np.float64)

    def featurize_batch(self, formulas: List[str]) -> np.ndarray:
        return np.array([self.featurize(f) for f in formulas])

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
            val = props.get(prop, 0.0)
            values.append(val)
        values = np.array(values, dtype=np.float64)
        fractions = np.array(fractions, dtype=np.float64)
        mean_val = float(np.mean(values))
        max_val = float(np.max(values))
        min_val = float(np.min(values))
        std_val = float(np.std(values)) if len(values) > 1 else 0.0
        weighted_avg = float(np.average(values, weights=fractions))
        return [mean_val, max_val, min_val, std_val, weighted_avg]

    def _block_features(self, comp: Composition) -> List[float]:
        elements = list(comp.elements)
        fractions = list(comp.fractional_composition.values())
        s_frac, p_frac, d_frac, f_frac = 0.0, 0.0, 0.0, 0.0
        d_electrons = 0.0
        total_electrons = 0.0
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
            total_electrons += frac * ve
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
                ri = radius_values[i]
                rj = radius_values[j]
                ci = fractions[i]
                cj = fractions[j]
                r_avg = (ri + rj) / 2.0
                delta_h_mix += 4.0 * ci * cj * ((ri - rj) / max(r_avg, 0.01)) ** 2
        packing_diff = float(np.std(radius_values) / max(np.mean(radius_values), 0.01))
        density_estimate = float(np.average(density_values, weights=fractions))
        return [en_diff, en_range_ratio, radius_range, radius_mismatch,
                avg_valence, total_valence, ionic_char,
                delta_h_mix, packing_diff, density_estimate]
