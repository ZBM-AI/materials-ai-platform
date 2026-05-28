"""材料性能预测模型 v5 — GPR不确定性 + MatBench基准 + 诊断图"""

import os
import warnings
import numpy as np
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from .composition_features import CompositionFeaturizer

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": None,  # lazy import
    "svr": SVR,
    "ridge": Ridge,
    "elastic_net": ElasticNet,
    "gaussian_process": GaussianProcessRegressor,
}

PARAM_GRIDS = {
    "random_forest": {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__max_depth": [5, 8, 10, 15, 20, None],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", "log2", 0.5, 0.8],
    },
    "gradient_boosting": {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__max_depth": [3, 5, 8],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__subsample": [0.7, 0.8, 1.0],
        "regressor__min_samples_split": [2, 5, 10],
    },
    "svr": {
        "regressor__C": [0.1, 1, 10, 100],
        "regressor__gamma": ["scale", "auto", 0.01, 0.1],
        "regressor__kernel": ["rbf", "poly"],
    },
    "ridge": {
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "elastic_net": {
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0],
        "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "gaussian_process": {
        "regressor__kernel": [
            RBF(length_scale=0.5) + WhiteKernel(noise_level=0.01),
            Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.01),
            ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.01),
            ConstantKernel(1.0) * Matern(length_scale=0.5, nu=1.5) + WhiteKernel(noise_level=0.01),
        ],
        "regressor__alpha": [1e-6, 1e-4, 1e-2],
        "regressor__normalize_y": [True, False],
    },
    "xgboost": {
        "regressor__n_estimators": [100, 200, 300, 500],
        "regressor__max_depth": [3, 5, 8, 10],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__subsample": [0.7, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__reg_alpha": [0, 0.1, 1.0],
        "regressor__reg_lambda": [0.1, 1.0, 10.0],
    },
}


class PropertyPredictor:
    def __init__(self, model_type: str = "random_forest", tune: bool = True, n_iter: int = 30):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Options: {list(MODEL_REGISTRY)}")
        self.model_type = model_type
        self.tune = tune
        self.n_iter = n_iter
        self.model = self._build_model(model_type)
        self.featurizer = CompositionFeaturizer()
        self.is_trained = False
        self._feature_names: Optional[List[str]] = None
        self._target_name: str = ""
        self._cv_results: dict = {}
        self._test_results: dict = {}

    def _build_model(self, model_type: str) -> Pipeline:
        regressor_class = MODEL_REGISTRY[model_type]
        if model_type == "xgboost":
            import xgboost as xgb
            regressor_class = xgb.XGBRegressor
        default_kwargs = {"random_state": 42} if model_type in ("random_forest", "gradient_boosting", "xgboost") else {}
        if model_type == "svr":
            default_kwargs["kernel"] = "rbf"
        if model_type == "gaussian_process":
            default_kwargs["kernel"] = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
            default_kwargs["alpha"] = 1e-4
            default_kwargs["normalize_y"] = True
            default_kwargs["random_state"] = 42
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", regressor_class(**default_kwargs))
        ])

    def train(self, formulas: List[str], targets: np.ndarray, target_name: str = ""):
        X = self.featurizer.featurize_batch(formulas)
        self._feature_names = self.featurizer.get_feature_names()
        self._target_name = target_name
        self.model.fit(X, targets)
        self.is_trained = True
        self._feature_names = self.featurizer.get_feature_names()

    def train_with_tuning(self, formulas: List[str], targets: np.ndarray,
                           target_name: str = "", test_size: float = 0.2,
                           n_splits: int = 3) -> dict:
        target_name = target_name or self._target_name
        X = self.featurizer.featurize_batch(formulas)
        self._feature_names = self.featurizer.get_feature_names()

        split_test_r2, split_test_mae, split_test_rmse = [], [], []
        split_cv_r2, split_cv_mae, split_cv_rmse = [], [], []
        best_model = None
        best_score = -float("inf")

        for split_idx in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, targets, test_size=test_size,
                random_state=42 + split_idx * 17
            )
            model = self._build_model(self.model_type)
            if self.tune and self.model_type in PARAM_GRIDS and len(y_train) >= 30:
                param_grid = PARAM_GRIDS[self.model_type]
                n_iter = min(self.n_iter, self._count_combinations(param_grid))
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iter,
                    cv=min(5, len(y_train)),
                    scoring="r2", random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
            else:
                model.fit(X_train, y_train)

            test_preds = model.predict(X_test)
            test_r2 = float(r2_score(y_test, test_preds))
            test_mae = float(mean_absolute_error(y_test, test_preds))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

            split_test_r2.append(test_r2)
            split_test_mae.append(test_mae)
            split_test_rmse.append(test_rmse)

            if test_r2 > best_score:
                best_score = test_r2
                best_model = model

            cv_result = cross_validate(
                model, X_train, y_train,
                cv=min(5, len(y_train)),
                scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
            )
            split_cv_r2.append(float(cv_result["test_r2"].mean()))
            split_cv_mae.append(float(-cv_result["test_neg_mean_absolute_error"].mean()))
            split_cv_rmse.append(float(np.sqrt(
                -cv_result["test_neg_mean_squared_error"].mean()
            )))

        self.model = best_model
        self.is_trained = True

        total = len(targets)
        test_n = int(total * test_size)
        train_n = total - test_n
        self._test_results = {
            "train_size": train_n,
            "test_size": test_n,
            "test_r2_mean": float(np.mean(split_test_r2)),
            "test_r2_std": float(np.std(split_test_r2)),
            "test_mae_mean": float(np.mean(split_test_mae)),
            "test_mae_std": float(np.std(split_test_mae)),
            "test_rmse_mean": float(np.mean(split_test_rmse)),
            "test_rmse_std": float(np.std(split_test_rmse)),
        }
        self._cv_results = {
            "cv_r2_mean": float(np.mean(split_cv_r2)),
            "cv_r2_std": float(np.std(split_cv_r2)),
            "cv_mae_mean": float(np.mean(split_cv_mae)),
            "cv_mae_std": float(np.std(split_cv_mae)),
            "cv_rmse_mean": float(np.mean(split_cv_rmse)),
            "cv_rmse_std": float(np.std(split_cv_rmse)),
        }
        return {**self._test_results, **self._cv_results}

    def _count_combinations(self, param_grid: dict) -> int:
        total = 1
        for v in param_grid.values():
            total *= len(v)
        return total

    def predict(self, formula: str) -> float:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = self.featurizer.featurize(formula).reshape(1, -1)
        return float(self.model.predict(X)[0])

    def predict_batch(self, formulas: List[str]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X = self.featurizer.featurize_batch(formulas)
        return self.model.predict(X)

    def predict_with_std(self, formula: str) -> Tuple[float, float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X = self.featurizer.featurize(formula).reshape(1, -1)
        regressor = self.model.named_steps["regressor"]
        if self.model_type == "random_forest" and hasattr(regressor, "estimators_"):
            trees = regressor.estimators_
            preds = np.array([t.predict(X)[0] for t in trees])
            return float(preds.mean()), float(preds.std())
        if self.model_type == "gaussian_process":
            X_scaled = self.model.named_steps["scaler"].transform(X)
            pred, std = regressor.predict(X_scaled, return_std=True)
            return float(pred[0]), float(std[0])
        return self.predict(formula), 0.0

    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        if not self.is_trained or not self._feature_names:
            return None
        regressor = self.model.named_steps.get("regressor")
        if hasattr(regressor, "feature_importances_"):
            paired = list(zip(self._feature_names, regressor.feature_importances_))
            paired.sort(key=lambda x: x[1], reverse=True)
            return paired
        elif hasattr(regressor, "coef_"):
            coef = regressor.coef_
            if coef.ndim > 1:
                coef = coef[0]
            paired = list(zip(self._feature_names, np.abs(coef)))
            paired.sort(key=lambda x: x[1], reverse=True)
            return paired
        return None

    @property
    def cv_results(self) -> dict:
        return self._cv_results

    @property
    def test_results(self) -> dict:
        return self._test_results

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self._feature_names,
            "target_name": self._target_name,
            "is_trained": self.is_trained,
            "cv_results": self._cv_results,
            "test_results": self._test_results,
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PropertyPredictor":
        data = joblib.load(filepath)
        predictor = cls(data["model_type"], tune=False)
        predictor.model = data["model"]
        predictor._feature_names = data.get("feature_names")
        predictor._target_name = data.get("target_name", "")
        predictor.is_trained = data.get("is_trained", True)
        predictor._cv_results = data.get("cv_results", {})
        predictor._test_results = data.get("test_results", {})
        return predictor


class EnsemblePredictor:
    """多模型集成预测器"""

    def __init__(self, model_types: List[str] = None):
        if model_types is None:
            model_types = ["random_forest", "gradient_boosting", "ridge"]
        self.model_types = model_types
        self.models: List[PropertyPredictor] = []
        self.featurizer = CompositionFeaturizer()
        self.is_trained = False
        self._target_name: str = ""

    def train(self, formulas: List[str], targets: np.ndarray, target_name: str = "",
              test_size: float = 0.2, tune: bool = True):
        self._target_name = target_name
        results = {}
        for mtype in self.model_types:
            predictor = PropertyPredictor(mtype, tune=tune, n_iter=20)
            res = predictor.train_with_tuning(formulas, targets, target_name, test_size)
            self.models.append(predictor)
            results[mtype] = res
        self.is_trained = True
        return results

    def predict(self, formula: str) -> float:
        preds = [m.predict(formula) for m in self.models]
        return float(np.mean(preds))

    def predict_batch(self, formulas: List[str]) -> np.ndarray:
        all_preds = np.array([m.predict_batch(formulas) for m in self.models])
        return np.mean(all_preds, axis=0)

    def predict_with_std(self, formula: str) -> Tuple[float, float]:
        preds = np.array([m.predict(formula) for m in self.models])
        return float(preds.mean()), float(preds.std())

    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        for m in self.models:
            imp = m.get_feature_importance()
            if imp:
                return imp
        return None

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save(os.path.join(directory, f"ensemble_{m.model_type}.pkl"))

    @classmethod
    def load(cls, directory: str) -> "EnsemblePredictor":
        import glob
        files = glob.glob(os.path.join(directory, "ensemble_*.pkl"))
        predictor = cls(model_types=[])
        for f in sorted(files):
            m = PropertyPredictor.load(f)
            predictor.models.append(m)
            predictor.model_types.append(m.model_type)
        predictor.is_trained = len(predictor.models) > 0
        return predictor
