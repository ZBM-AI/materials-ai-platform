"""MatBench 基准集成 — 直接下载标准数据集, 对比模型与已发表SOTA

数据来源: MatBench v0.1 (https://matbench.materialsproject.org/)
通过 Materials Project 公共API下载JSON.gz文件, 无需安装matbench包.
"""

import os
import sys
import json
import gzip
import time
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from urllib.request import urlopen
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

MATBENCH_AVAILABLE = True  # 直接下载, 不依赖matbench包

MATBENCH_URLS = {
    "matbench_expt_gap": "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz",
    "matbench_mp_gap": "https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz",
    "matbench_mp_e_form": "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz",
    "matbench_log_gvrh": "https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz",
    "matbench_log_kvrh": "https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "matbench_cache")

SOTA_COMPARISON = {
    "matbench_expt_gap": {
        "description": "Experimental Band Gap (eV)",
        "n_samples": 4604,
        "sota_model": "MODNet (v0.1)",
        "sota_mae": 0.30,
        "baseline_mae": 0.55,
    },
    "matbench_mp_gap": {
        "description": "DFT PBE Band Gap (eV)",
        "n_samples": 106113,
        "sota_model": "ALIGNN",
        "sota_mae": 0.21,
        "baseline_mae": 0.34,
    },
    "matbench_mp_e_form": {
        "description": "DFT Formation Energy (eV/atom)",
        "n_samples": 132752,
        "sota_model": "ALIGNN",
        "sota_mae": 0.022,
        "baseline_mae": 0.071,
    },
    "matbench_log_gvrh": {
        "description": "log10 Shear Modulus (GPa)",
        "n_samples": 10987,
        "sota_model": "coGN+CGCNN",
        "sota_mae": 0.07,
        "baseline_mae": 0.13,
    },
    "matbench_log_kvrh": {
        "description": "log10 Bulk Modulus (GPa)",
        "n_samples": 10987,
        "sota_model": "coGN+CGCNN",
        "sota_mae": 0.05,
        "baseline_mae": 0.09,
    },
}

AVAILABLE_TASKS = {
    "expt_gap": "matbench_expt_gap",
    "mp_gap": "matbench_mp_gap",
    "mp_e_form": "matbench_mp_e_form",
    "log_gvrh": "matbench_log_gvrh",
    "log_kvrh": "matbench_log_kvrh",
}


def _load_or_download(task_name: str, verbose: bool = True) -> dict:
    """加载MatBench数据集: 优先本地JSON, 其次下载JSON.gz. 返回原始字典."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    local_json = os.path.join(CACHE_DIR, f"{task_name}.json")
    local_gz = os.path.join(CACHE_DIR, f"{task_name}.json.gz")

    data = None
    for path in [local_json, local_gz]:
        if os.path.exists(path):
            try:
                if path.endswith('.gz'):
                    with gzip.open(path, 'rb') as gz:
                        data = json.loads(gz.read().decode('utf-8'))
                else:
                    with open(path, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                if verbose:
                    print(f"  Loaded local: {path}")
                break
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Failed to load {path}: {e}")

    if data is None:
        url = MATBENCH_URLS.get(task_name)
        if not url:
            raise ValueError(f"No URL for task: {task_name}")
        if verbose:
            print(f"  Downloading {url} ...")
        try:
            resp = urlopen(url, timeout=30)
            raw = resp.read()
            with gzip.GzipFile(fileobj=BytesIO(raw)) as gz:
                data = json.loads(gz.read().decode('utf-8'))
            with open(local_gz, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
            if verbose:
                print(f"  Cached to: {local_gz}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot download {task_name}: {e}. "
                f"Download manually from {url} and place in {CACHE_DIR}"
            )
    return data


def _extract_formula_from_row(row, has_structure_col: bool) -> str:
    """从MatBench数据行中提取化学式."""
    if has_structure_col:
        struct_dict = row[0]
        if isinstance(struct_dict, dict) and "@module" in struct_dict:
            try:
                from pymatgen.core import Structure
                struct = Structure.from_dict(struct_dict)
                return struct.composition.reduced_formula
            except Exception:
                pass
        return "H"
    else:
        return row[0].strip() if isinstance(row, list) else str(row)


def _extract_target_from_row(row, has_structure_col: bool) -> float:
    """从MatBench数据行中提取目标值."""
    if has_structure_col:
        return float(row[1]) if len(row) > 1 else 0.0
    else:
        return float(row[1]) if isinstance(row, list) else 0.0


def list_available_local_datasets() -> List[str]:
    """列出CACHE_DIR中已下载的MatBench数据集"""
    if not os.path.exists(CACHE_DIR):
        return []
    available = []
    for task_name in MATBENCH_URLS:
        for ext in ['.json', '.json.gz']:
            path = os.path.join(CACHE_DIR, f"{task_name}{ext}")
            if os.path.exists(path):
                available.append(task_name)
                break
    return available


def load_matbench_dataset(task_name: str, n_samples: Optional[int] = None,
                          verbose: bool = True):
    """加载单个MatBench数据集.

    支持两种数据格式:
      1. 本地JSON (matbench v0.1格式): {"columns": ["structure", "gap pbe"], "data": [[Structure, value], ...]}
         从pymatgen Structure对象中提取化学式 → Magpie特征
      2. 标准JSON.gz格式: {"data": [[formula, value], ...]} — 直接用化学式

    Args:
        task_name: 如 'matbench_mp_gap'
        n_samples: 限制样本数, None则全部加载

    Returns:
        formulas: 化学式列表
        targets: 目标值 (numpy array)
        metadata: 元信息字典
    """
    data = _load_or_download(task_name, verbose=verbose)

    has_structure_col = False
    if "columns" in data and len(data["columns"]) > 0:
        first_col = data["columns"][0]
        has_structure_col = (first_col == "structure")

    if "data" in data:
        samples = data["data"]
        if isinstance(samples, list) and len(samples) > 0:
            formulas = [_extract_formula_from_row(s, has_structure_col) for s in samples]
            targets = np.array([_extract_target_from_row(s, has_structure_col) for s in samples],
                               dtype=np.float64)
        elif isinstance(samples, dict):
            formulas = list(samples.keys())
            targets = np.array(list(samples.values()), dtype=np.float64)
        else:
            formulas = ["H"]
            targets = np.array([0.0])
    else:
        formulas = ["H"]
        targets = np.array([0.0])

    metadata = {
        "description": SOTA_COMPARISON.get(task_name, {}).get("description", task_name),
        "n_total": len(formulas),
        "has_structure": has_structure_col,
    }

    if n_samples and n_samples < len(formulas):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(formulas), n_samples, replace=False)
        formulas = [formulas[i] for i in idx]
        targets = targets[idx]

    if verbose:
        print(f"  Formulas: {len(formulas)}, range [{targets.min():.2f}, {targets.max():.2f}]")

    return formulas, targets, metadata


def benchmark_on_matbench(
    task_name: str,
    model_factory,
    n_samples: int = 500,
    test_size: float = 0.2,
    verbose: bool = True,
) -> dict:
    """在MatBench任务上评测一个模型."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    if verbose:
        print(f"  Loading {task_name}...")
    formulas, targets, metadata = load_matbench_dataset(task_name, n_samples, verbose=verbose)
    if verbose:
        print(f"  Loaded {len(formulas)} samples, range [{targets.min():.2f}, {targets.max():.2f}]")

    from .features_v4 import MagpieFeaturizer
    featurizer = MagpieFeaturizer()
    X = featurizer.featurize_batch(formulas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if verbose:
        print(f"  Training on {len(X_train)} samples, testing on {len(X_test)}...")
    start = time.time()
    model = model_factory()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    preds = model.predict(X_test)
    result = {
        "task": task_name,
        "description": metadata.get("description", task_name),
        "n_total": len(formulas),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_r2": float(r2_score(y_test, preds)),
        "training_time_s": round(training_time, 2),
        "feature_dim": featurizer.feature_dim(),
    }
    sota_info = SOTA_COMPARISON.get(task_name, {})
    if sota_info:
        result["sota_model"] = sota_info.get("sota_model", "Unknown")
        result["sota_mae"] = sota_info.get("sota_mae")
        result["baseline_mae"] = sota_info.get("baseline_mae")
        result["mae_gap_to_sota"] = round(result["test_mae"] - sota_info.get("sota_mae", result["test_mae"]), 4)
    if verbose:
        print(f"  Test MAE={result['test_mae']:.4f}, R^2={result['test_r2']:.4f} "
              f"[SOTA: {result.get('sota_model','?')}={result.get('sota_mae','?')}] "
              f"Time={training_time:.1f}s")
    return result


def run_matbench_suite(task_names: Optional[List[str]] = None,
                       n_samples: int = 500,
                       model_type: str = "random_forest",
                       verbose: bool = True) -> pd.DataFrame:
    """在一组MatBench任务上运行完整评测套件."""
    if task_names is None:
        task_names = list(AVAILABLE_TASKS.values())
    def make_model():
        if model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    results = []
    print(f"\n{'='*60}")
    print(f"MatBench Benchmark Suite — {model_type.upper()}")
    print(f"  Samples per task: {n_samples}")
    print(f"{'='*60}")
    for task_name in task_names:
        try:
            res = benchmark_on_matbench(task_name, make_model, n_samples=n_samples, verbose=verbose)
            results.append(res)
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {task_name}: {e}")
            results.append({"task": task_name, "error": str(e)})
    df = pd.DataFrame(results)
    if verbose and len(df) > 0:
        print(f"\n{'='*60}")
        print("MatBench Summary:")
        print(f"{'='*60}")
        cols = ["task", "test_mae", "test_r2", "sota_mae", "mae_gap_to_sota", "training_time_s"]
        available = [c for c in cols if c in df.columns]
        print(df[available].to_string(index=False))
    return df
