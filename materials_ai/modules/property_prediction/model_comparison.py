"""模型对比 v5 — 随机森林 vs XGBoost vs GPR vs CGCNN vs MEGNet + MatBench"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

from .features_v4 import MagpieFeaturizer

XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

CGCNN_AVAILABLE = False
try:
    from .cgcnn import train_cgcnn
    CGCNN_AVAILABLE = True
except ImportError:
    pass

MEGNET_AVAILABLE = False
try:
    from .megnet import train_megnet, MEGNET_AVAILABLE as _MG
    MEGNET_AVAILABLE = _MG
except ImportError:
    pass

MATBENCH_AVAILABLE = False
try:
    from .matbench_benchmark import MATBENCH_AVAILABLE as _MB
    MATBENCH_AVAILABLE = _MB
except ImportError:
    pass


def _prepare_data(formulas: List[str], targets: np.ndarray,
                  featurizer: MagpieFeaturizer) -> tuple:
    X = featurizer.featurize_batch(formulas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_random_forest(formulas: List[str], targets: np.ndarray,
                        featurizer: Optional[MagpieFeaturizer] = None,
                        n_estimators: int = 300) -> dict:
    """训练随机森林并返回指标"""
    if featurizer is None:
        featurizer = MagpieFeaturizer()
    start = time.time()
    X_train, X_test, y_train, y_test = _prepare_data(formulas, targets, featurizer)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    cv_results = cross_validate(
        model, X_train, y_train, cv=5,
        scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
    )
    elapsed = time.time() - start
    return {
        "model": "RandomForest",
        "train_r2": float(r2_score(y_train, train_preds)),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "cv_r2_mean": float(cv_results["test_r2"].mean()),
        "cv_r2_std": float(cv_results["test_r2"].std()),
        "cv_mae_mean": float(-cv_results["test_neg_mean_absolute_error"].mean()),
        "time_seconds": elapsed,
        "trained_model": model,
    }


def train_xgboost(formulas: List[str], targets: np.ndarray,
                  featurizer: Optional[MagpieFeaturizer] = None) -> dict:
    """训练XGBoost并返回指标"""
    if not XGB_AVAILABLE:
        return {"model": "XGBoost", "error": "xgboost not installed"}
    if featurizer is None:
        featurizer = MagpieFeaturizer()
    start = time.time()
    X_train, X_test, y_train, y_test = _prepare_data(formulas, targets, featurizer)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    cv_results = cross_validate(
        model, X_train, y_train, cv=5,
        scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
    )
    elapsed = time.time() - start
    return {
        "model": "XGBoost",
        "train_r2": float(r2_score(y_train, train_preds)),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "cv_r2_mean": float(cv_results["test_r2"].mean()),
        "cv_r2_std": float(cv_results["test_r2"].std()),
        "cv_mae_mean": float(-cv_results["test_neg_mean_absolute_error"].mean()),
        "time_seconds": elapsed,
        "trained_model": model,
    }


def train_gaussian_process(formulas: List[str], targets: np.ndarray,
                           featurizer: Optional[MagpieFeaturizer] = None,
                           n_restarts: int = 3) -> dict:
    """训练高斯过程回归并返回指标 (含原生不确定性)"""
    if featurizer is None:
        featurizer = MagpieFeaturizer()
    start = time.time()
    X_train, X_test, y_train, y_test = _prepare_data(formulas, targets, featurizer)
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1))
    n_restarts_actual = min(n_restarts, 3) if len(y_train) > 200 else n_restarts
    model = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-4, normalize_y=True,
        n_restarts_optimizer=n_restarts_actual, random_state=42,
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds, test_std = model.predict(X_test, return_std=True)
    cv_results = cross_validate(
        model, X_train, y_train, cv=min(5, len(y_train) // 20),
        scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
    )
    elapsed = time.time() - start
    return {
        "model": "GaussianProcess",
        "train_r2": float(r2_score(y_train, train_preds)),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "cv_r2_mean": float(cv_results["test_r2"].mean()),
        "cv_r2_std": float(cv_results["test_r2"].std()),
        "cv_mae_mean": float(-cv_results["test_neg_mean_absolute_error"].mean()),
        "mean_uncertainty": float(np.mean(test_std)),
        "calibration_ratio": float(np.mean(np.abs(test_preds - y_test) < test_std)),
        "time_seconds": elapsed,
        "trained_model": model,
    }


def run_model_comparison(formulas: List[str], targets: np.ndarray,
                         target_name: str = "band_gap_eV",
                         include_gpr: bool = True,
                         include_megnet: bool = False) -> pd.DataFrame:
    """运行完整模型对比, 返回结果DataFrame"""
    featurizer = MagpieFeaturizer()
    print(f"\n{'='*60}")
    print(f"Model Comparison: {target_name} (Magpie features, {featurizer.feature_dim()}D)")
    print(f"  Samples: {len(formulas)}")
    print(f"{'='*60}")

    total_steps = 3 + (1 if include_gpr else 0) + (1 if include_megnet else 0)
    step = 0
    results = []

    step += 1
    print(f"\n[{step}/{total_steps}] Training Random Forest...")
    rf = train_random_forest(formulas, targets, featurizer)
    print(f"  RF:  Test R^2={rf['test_r2']:.4f}, CV R^2={rf['cv_r2_mean']:.4f}, "
          f"MAE={rf['test_mae']:.4f}, Time={rf['time_seconds']:.1f}s")
    results.append(rf)

    step += 1
    print(f"\n[{step}/{total_steps}] Training XGBoost...")
    xgb_result = train_xgboost(formulas, targets, featurizer)
    if "error" not in xgb_result:
        print(f"  XGB: Test R^2={xgb_result['test_r2']:.4f}, CV R^2={xgb_result['cv_r2_mean']:.4f}, "
              f"MAE={xgb_result['test_mae']:.4f}, Time={xgb_result['time_seconds']:.1f}s")
    else:
        print(f"  XGB: SKIPPED ({xgb_result['error']})")
    results.append(xgb_result)

    if include_gpr:
        step += 1
        print(f"\n[{step}/{total_steps}] Training Gaussian Process...")
        gpr_result = train_gaussian_process(formulas, targets, featurizer)
        print(f"  GPR: Test R^2={gpr_result['test_r2']:.4f}, CV R^2={gpr_result['cv_r2_mean']:.4f}, "
              f"MAE={gpr_result['test_mae']:.4f}, "
              f"Uncertainty={gpr_result['mean_uncertainty']:.4f}, "
              f"Calibration={gpr_result['calibration_ratio']:.2%}, "
              f"Time={gpr_result['time_seconds']:.1f}s")
        results.append(gpr_result)

    step += 1
    print(f"\n[{step}/{total_steps}] Training CGCNN (PyTorch Geometric)...")
    if CGCNN_AVAILABLE and len(formulas) >= 30:
        cgcnn = train_cgcnn(formulas, targets, epochs=50)
        cgcnn["model"] = "CGCNN"
        cgcnn["time_seconds"] = 0
        print(f"  CGCNN: Test R^2={cgcnn['test_r2']:.4f}, MAE={cgcnn['test_mae']:.4f}")
        results.append(cgcnn)
    else:
        print("  CGCNN: SKIPPED (not available or too few samples)")

    if include_megnet:
        step += 1
        print(f"\n[{step}/{total_steps}] Training MEGNet (PyTorch Geometric)...")
        if MEGNET_AVAILABLE and len(formulas) >= 30:
            megnet = train_megnet(formulas, targets, epochs=60)
            megnet["model"] = "MEGNet"
            megnet["time_seconds"] = 0
            print(f"  MEGNet: Test R^2={megnet['test_r2']:.4f}, MAE={megnet['test_mae']:.4f}")
            results.append(megnet)
        else:
            print("  MEGNet: SKIPPED (not available or too few samples)")

    df = pd.DataFrame(results)
    metric_cols = [c for c in df.columns if c not in ("trained_model",)]
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'='*60}")
    print(df[metric_cols].to_string(index=False))
    return df, featurizer
