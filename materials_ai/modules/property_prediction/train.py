"""模型训练流程 v3 — 数据清洗、修复CV泄漏、多次分割评估"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.property_prediction.models import PropertyPredictor
from modules.property_prediction.composition_features import CompositionFeaturizer
from pymatgen.core import Composition
import config

warnings.filterwarnings("ignore", category=UserWarning)


def validate_formulas(formulas, targets, name=""):
    """Remove unparseable formulas, return cleaned lists."""
    valid_f, valid_t = [], []
    skipped = []
    for f, t in zip(formulas, targets):
        try:
            Composition(f)
            valid_f.append(f)
            valid_t.append(t)
        except Exception:
            skipped.append(f)
    if skipped:
        print(f"  [WARNING] {name}: Skipped {len(skipped)} unparseable formulas: {skipped}")
    return valid_f, np.array(valid_t)


def count_samples(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as fh:
        return sum(1 for _ in fh) - 1


def print_residual_analysis(y_true, y_pred, name=""):
    residuals = y_true - y_pred
    print(f"  [{name}] Residual analysis:")
    print(f"    Mean residual: {np.mean(residuals):.4f}")
    print(f"    Std residual:  {np.std(residuals):.4f}")
    print(f"    Max overestimate: {np.max(residuals):.4f}")
    print(f"    Max underestimate: {np.min(residuals):.4f}")


def train_band_gap():
    print("=" * 60)
    print("Training Band Gap model (RandomForest + tuning, v3)")
    n = count_samples(config.BAND_GAP_DATA)
    print(f"  Dataset: {n} samples")
    df = pd.read_csv(config.BAND_GAP_DATA)
    formulas = df["formula"].tolist()
    targets = df["band_gap_eV"].values

    formulas, targets = validate_formulas(formulas, targets, "band_gap")
    print(f"  Valid samples after cleaning: {len(formulas)}")
    print(f"  Band gap range: {targets.min():.2f} - {targets.max():.2f} eV")

    predictor = PropertyPredictor("random_forest", tune=True, n_iter=30)
    results = predictor.train_with_tuning(formulas, targets, "band_gap_eV", test_size=0.2, n_splits=3)
    print(f"  Test R^2: {results['test_r2_mean']:.4f} +/- {results['test_r2_std']:.4f}")
    print(f"  Test MAE: {results['test_mae_mean']:.4f} +/- {results['test_mae_std']:.4f} eV")
    print(f"  CV R^2:   {results['cv_r2_mean']:.4f} +/- {results['cv_r2_std']:.4f}")
    print(f"  CV MAE:   {results['cv_mae_mean']:.4f} +/- {results['cv_mae_std']:.4f} eV")

    all_preds = predictor.predict_batch(formulas)
    print_residual_analysis(targets, all_preds, "band_gap")
    predictor.save(config.BAND_GAP_MODEL)
    print(f"  Model saved to: {config.BAND_GAP_MODEL}")
    return results


def train_formation_energy():
    print("=" * 60)
    print("Training Formation Energy model (GradientBoosting + tuning, v3)")
    n = count_samples(config.FORMATION_ENERGY_DATA)
    print(f"  Dataset: {n} samples")
    df = pd.read_csv(config.FORMATION_ENERGY_DATA)
    formulas = df["formula"].tolist()
    targets = df["formation_energy_eV_per_atom"].values

    formulas, targets = validate_formulas(formulas, targets, "formation_energy")
    print(f"  Valid samples after cleaning: {len(formulas)}")
    print(f"  Formation energy range: {targets.min():.2f} - {targets.max():.2f} eV/atom")

    predictor = PropertyPredictor("gradient_boosting", tune=True, n_iter=30)
    results = predictor.train_with_tuning(formulas, targets, "formation_energy_eV_per_atom", test_size=0.2, n_splits=3)
    print(f"  Test R^2: {results['test_r2_mean']:.4f} +/- {results['test_r2_std']:.4f}")
    print(f"  Test MAE: {results['test_mae_mean']:.4f} +/- {results['test_mae_std']:.4f} eV/atom")
    print(f"  CV R^2:   {results['cv_r2_mean']:.4f} +/- {results['cv_r2_std']:.4f}")
    print(f"  CV MAE:   {results['cv_mae_mean']:.4f} +/- {results['cv_mae_std']:.4f} eV/atom")

    all_preds = predictor.predict_batch(formulas)
    print_residual_analysis(targets, all_preds, "formation_energy")
    predictor.save(config.FORMATION_ENERGY_MODEL)
    print(f"  Model saved to: {config.FORMATION_ENERGY_MODEL}")
    return results


def train_mechanical():
    print("=" * 60)
    print("Training Mechanical Properties model (RandomForest + tuning, v3)")
    n = count_samples(config.MECHANICAL_DATA)
    print(f"  Dataset: {n} samples")
    df = pd.read_csv(config.MECHANICAL_DATA)
    formulas = df["formula"].tolist()

    targets_bulk = df["bulk_modulus_GPa"].values
    formulas_bulk, targets_bulk = validate_formulas(formulas, targets_bulk, "bulk_modulus")
    print(f"  Valid samples after cleaning: {len(formulas_bulk)}")
    print(f"  Bulk modulus range: {targets_bulk.min():.0f} - {targets_bulk.max():.0f} GPa")
    predictor_bulk = PropertyPredictor("random_forest", tune=True, n_iter=30)
    results_bulk = predictor_bulk.train_with_tuning(formulas_bulk, targets_bulk, "bulk_modulus_GPa", test_size=0.2, n_splits=3)
    print(f"  Bulk Modulus - Test R^2: {results_bulk['test_r2_mean']:.4f} +/- {results_bulk['test_r2_std']:.4f}")
    print(f"  Bulk Modulus - Test MAE: {results_bulk['test_mae_mean']:.2f} +/- {results_bulk['test_mae_std']:.2f} GPa")
    print(f"  Bulk Modulus - CV R^2:   {results_bulk['cv_r2_mean']:.4f} +/- {results_bulk['cv_r2_std']:.4f}")

    bulk_preds = predictor_bulk.predict_batch(formulas_bulk)
    print_residual_analysis(targets_bulk, bulk_preds, "bulk_modulus")
    predictor_bulk.save(config.MECHANICAL_MODEL)

    targets_shear = df["shear_modulus_GPa"].values
    if len(targets_shear) > 0 and not pd.isna(targets_shear[0]):
        formulas_shear, targets_shear = validate_formulas(formulas, targets_shear, "shear_modulus")
        print(f"  Shear modulus range: {targets_shear.min():.0f} - {targets_shear.max():.0f} GPa")
        predictor_shear = PropertyPredictor("random_forest", tune=True, n_iter=30)
        results_shear = predictor_shear.train_with_tuning(formulas_shear, targets_shear, "shear_modulus_GPa", test_size=0.2, n_splits=3)
        print(f"  Shear Modulus - Test R^2: {results_shear['test_r2_mean']:.4f} +/- {results_shear['test_r2_std']:.4f}")
        print(f"  Shear Modulus - Test MAE: {results_shear['test_mae_mean']:.2f} +/- {results_shear['test_mae_std']:.2f} GPa")
        print(f"  Shear Modulus - CV R^2:   {results_shear['cv_r2_mean']:.4f} +/- {results_shear['cv_r2_std']:.4f}")

        shear_preds = predictor_shear.predict_batch(formulas_shear)
        print_residual_analysis(targets_shear, shear_preds, "shear_modulus")
        shear_path = config.MECHANICAL_MODEL.replace("mechanical", "shear_modulus")
        predictor_shear.save(shear_path)
        results_bulk["shear"] = results_shear

    return results_bulk


def train_thermal_conductivity():
    print("=" * 60)
    print("Training Thermal Conductivity model (RandomForest + tuning, v5)")
    n = count_samples(config.THERMAL_CONDUCTIVITY_DATA)
    print(f"  Dataset: {n} samples")
    df = pd.read_csv(config.THERMAL_CONDUCTIVITY_DATA)
    formulas = df["formula"].tolist()
    targets = df["thermal_conductivity_W_per_mK"].values

    formulas, targets = validate_formulas(formulas, targets, "thermal_conductivity")
    print(f"  Valid samples after cleaning: {len(formulas)}")
    print(f"  Thermal conductivity range: {targets.min():.2f} - {targets.max():.2f} W/(m*K)")

    predictor = PropertyPredictor("random_forest", tune=True, n_iter=30)
    results = predictor.train_with_tuning(
        formulas, targets, "thermal_conductivity_W_per_mK", test_size=0.2, n_splits=3
    )
    print(f"  Test R^2: {results['test_r2_mean']:.4f} +/- {results['test_r2_std']:.4f}")
    print(f"  Test MAE: {results['test_mae_mean']:.4f} +/- {results['test_mae_std']:.4f} W/(m*K)")
    print(f"  CV R^2:   {results['cv_r2_mean']:.4f} +/- {results['cv_r2_std']:.4f}")
    print(f"  CV MAE:   {results['cv_mae_mean']:.4f} +/- {results['cv_mae_std']:.4f} W/(m*K)")

    all_preds = predictor.predict_batch(formulas)
    print_residual_analysis(targets, all_preds, "thermal_conductivity")
    predictor.save(config.THERMAL_CONDUCTIVITY_MODEL)
    print(f"  Model saved to: {config.THERMAL_CONDUCTIVITY_MODEL}")
    return results


def train_yield_strength():
    print("=" * 60)
    print("Training Yield Strength model (RandomForest + tuning, v5)")
    n = count_samples(config.YIELD_STRENGTH_DATA)
    print(f"  Dataset: {n} samples")
    df = pd.read_csv(config.YIELD_STRENGTH_DATA)
    formulas = df["formula"].tolist()
    targets = df["yield_strength_MPa"].values

    valid_f, valid_t = [], []
    skipped = []
    for f, t in zip(formulas, targets):
        if f in ("Al", "Fe", "Cu", "Ti", "Ni", "W", "Mo", "Ta", "Nb", "Co",
                 "Zn", "Mg", "Sn", "Pb", "Au", "Ag", "Pt", "Pd", "Rh"):
            valid_f.append(f)
            valid_t.append(t)
            continue
        try:
            Composition(f)
            valid_f.append(f)
            valid_t.append(t)
        except Exception:
            if '_' in f:
                base = f.split('_')[0]
            else:
                base = f
            try:
                Composition(base)
                valid_f.append(base)
                valid_t.append(t)
            except Exception:
                skipped.append(f)
    if skipped:
        print(f"  [WARNING] yield_strength: {len(skipped)} alloy names simplified")
    formulas, targets = valid_f, np.array(valid_t)
    print(f"  Valid samples after cleaning: {len(formulas)}")
    print(f"  Yield strength range: {targets.min():.0f} - {targets.max():.0f} MPa")

    predictor = PropertyPredictor("random_forest", tune=True, n_iter=30)
    results = predictor.train_with_tuning(
        formulas, targets, "yield_strength_MPa", test_size=0.2, n_splits=3
    )
    print(f"  Test R^2: {results['test_r2_mean']:.4f} +/- {results['test_r2_std']:.4f}")
    print(f"  Test MAE: {results['test_mae_mean']:.2f} +/- {results['test_mae_std']:.2f} MPa")
    print(f"  CV R^2:   {results['cv_r2_mean']:.4f} +/- {results['cv_r2_std']:.4f}")
    print(f"  CV MAE:   {results['cv_mae_mean']:.2f} +/- {results['cv_mae_std']:.2f} MPa")

    all_preds = predictor.predict_batch(formulas)
    print_residual_analysis(targets, all_preds, "yield_strength")
    predictor.save(config.YIELD_STRENGTH_MODEL)
    print(f"  Model saved to: {config.YIELD_STRENGTH_MODEL}")
    return results


def train_all():
    print("\n" + "=" * 60)
    print("Training All Materials Property Prediction Models (v5)")
    print("=" * 60 + "\n")

    all_results = {}
    for name, fn in [
        ("band_gap", train_band_gap),
        ("formation_energy", train_formation_energy),
        ("mechanical", train_mechanical),
        ("thermal_conductivity", train_thermal_conductivity),
        ("yield_strength", train_yield_strength),
    ]:
        try:
            all_results[name] = fn()
        except Exception as e:
            import traceback
            print(f"  [FAILED] {name}: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("Training Complete! Summary (v5):")
    print("=" * 60)
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: FAILED - {res['error']}")
        else:
            tr2 = res.get('test_r2_mean', 'N/A')
            cv2 = res.get('cv_r2_mean', 'N/A')
            cvs = res.get('cv_r2_std', 'N/A')
            print(f"  {name}: Test R^2={tr2:.4f}" if isinstance(tr2, float) else f"  {name}: Test R^2={tr2}"
                  f", CV R^2={cv2:.4f}+/-{cvs:.4f}" if isinstance(cv2, float) else f", CV R^2={cv2}+/-{cvs}")

    print("\n  Target thresholds (v5):")
    print("    Band Gap:           CV R^2 > 0.4, Test R^2 > 0.5")
    print("    Formation Energy:   CV R^2 > 0.5, Test R^2 > 0.8")
    print("    Bulk Modulus:       CV R^2 > 0.3, Test R^2 > 0.5")
    print("    Thermal Conductivity: CV R^2 > 0.5, Test R^2 > 0.6")
    print("    Yield Strength:     CV R^2 > 0.5, Test R^2 > 0.6")
    return all_results


if __name__ == "__main__":
    train_all()
