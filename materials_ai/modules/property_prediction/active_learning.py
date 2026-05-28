"""主动学习策略 — 基于不确定性采样推荐下一个实验点

策略:
  1. 用现有标注数据训练初始模型
  2. 对未标注候选集进行预测, 计算不确定性 (RF标准差 / 委员会分歧)
  3. 选择不确定性最高的样本推荐实验合成
  4. 实验测得真实值后, 加入训练集, 重新训练
  5. 重复直到预算耗尽或模型收敛

不确定性度量:
  - rf_std:      随机森林各树预测的标准差
  - committee:   多个模型预测的方差 (EnsemblePredictor)
  - gaussian:    高斯过程预测方差 (如有GP模型)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from sklearn.ensemble import RandomForestRegressor


# ─── 伪代码 ───────────────────────────────────────────────
ACTIVE_LEARNING_PSEUDOCODE = r'''
╔══════════════════════════════════════════════════════════════╗
║          Active Learning Loop for Materials Discovery        ║
╚══════════════════════════════════════════════════════════════╝

# Input:
#   L = {(x_i, y_i)}           # labeled dataset (已知材料)
#   U = {x_j}                  # unlabeled candidate pool (候选材料)
#   budget B                   # 最大实验次数
#   model M                    # 初始模型

for iteration in range(B):

    # ── Step 1: Train ──
    M.fit(L.features, L.targets)

    # ── Step 2: Predict & Estimate Uncertainty ──
    predictions = []
    uncertainties = []
    for x in U:
        mu = M.predict(x)
        sigma = M.predict_uncertainty(x)   # RF std / committee / GP variance
        predictions.append(mu)
        uncertainties.append(sigma)

    # ── Step 3: Acquisition ──
    # Option A: Pure uncertainty (exploration)
    best_idx = argmax(uncertainties)

    # Option B: Upper Confidence Bound (UCB) for maximization
    # best_idx = argmax(predictions + kappa * uncertainties)

    # Option C: Expected Improvement (EI) for maximization
    # best_idx = argmax(EI(predictions, uncertainties, best_so_far))

    # ── Step 4: Experiment ──
    x_next = U[best_idx]
    y_next = run_experiment(x_next)        # 实验室合成 + 测量

    # ── Step 5: Update ──
    L = L + {(x_next, y_next)}
    U = U - {x_next}

    # ── Step 6: Check Convergence ──
    if max(uncertainties) < epsilon:
        break

return L, M
'''


class ActiveLearner:
    """基于不确定性采样的主动学习器"""

    def __init__(self, model_type: str = "random_forest",
                 strategy: str = "uncertainty",
                 n_estimators: int = 200):
        self.model_type = model_type
        self.strategy = strategy  # "uncertainty", "ucb", "ei"
        self.n_estimators = n_estimators
        self.model = None
        self.labeled_X: Optional[np.ndarray] = None
        self.labeled_y: Optional[np.ndarray] = None
        self.history: List[dict] = []

    def initialize(self, X: np.ndarray, y: np.ndarray):
        """用初始标注数据初始化学习器"""
        self.labeled_X = X.copy()
        self.labeled_y = y.copy()
        self._refit()
        self.history = []

    def _refit(self):
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=15,
                min_samples_split=5, random_state=42, n_jobs=-1
            )
        else:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=8,
                random_state=42, n_jobs=-1
            )
        self.model.fit(self.labeled_X, self.labeled_y)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """计算预测不确定性 (RF树级标准差)"""
        if self.model_type == "random_forest" and hasattr(self.model, "estimators_"):
            trees = self.model.estimators_
            all_preds = np.array([t.predict(X) for t in trees])
            return np.std(all_preds, axis=0)
        elif hasattr(self.model, "predict"):
            preds = self.model.predict(X)
            return np.abs(preds) * 0.1  # fallback 10% relative
        return np.ones(len(X))

    def query(self, candidate_X: np.ndarray,
              candidate_formulas: Optional[List[str]] = None,
              n_query: int = 1, kappa: float = 1.0) -> List[int]:
        """从未标注池中选择下一个要标注的样本.

        Args:
            candidate_X: 候选特征矩阵
            candidate_formulas: 候选化学式 (可选, 用于日志)
            n_query: 每次查询的样本数
            kappa: UCB exploration-exploitation 权衡参数

        Returns:
            selected_indices: 被选中样本在candidate_X中的索引
        """
        predictions = self.model.predict(candidate_X)
        uncertainties = self.predict_uncertainty(candidate_X)

        if self.strategy == "uncertainty":
            scores = uncertainties.copy()
        elif self.strategy == "ucb":
            scores = predictions + kappa * uncertainties
        elif self.strategy == "ei":
            best_so_far = np.max(self.labeled_y) if self.labeled_y is not None else np.max(predictions)
            improvement = predictions - best_so_far
            z = improvement / np.maximum(uncertainties, 1e-10)
            from scipy.stats import norm as scipy_norm
            scores = improvement * scipy_norm.cdf(z) + uncertainties * scipy_norm.pdf(z)
        else:
            scores = uncertainties.copy()

        top_indices = np.argsort(scores)[-n_query:][::-1]
        selected = top_indices.tolist()

        record = {
            "iteration": len(self.history) + 1,
            "selected_indices": selected,
            "predicted_values": predictions[selected].tolist(),
            "uncertainties": uncertainties[selected].tolist(),
            "strategy": self.strategy,
        }
        if candidate_formulas:
            record["selected_formulas"] = [candidate_formulas[i] for i in selected]
        self.history.append(record)
        return selected

    def add_measurement(self, X_new: np.ndarray, y_new: np.ndarray):
        """添加新实验数据并重新训练"""
        self.labeled_X = np.vstack([self.labeled_X, X_new])
        self.labeled_y = np.concatenate([self.labeled_y, y_new])
        self._refit()

    def get_next_experiment(self, candidate_X: np.ndarray,
                            candidate_formulas: List[str],
                            kappa: float = 1.0) -> dict:
        """推荐下一个实验: 返回推荐化学式、预测值和不确定性"""
        idx = self.query(candidate_X, candidate_formulas, n_query=1, kappa=kappa)[0]
        pred = self.model.predict(candidate_X[idx:idx+1])[0]
        unc = self.predict_uncertainty(candidate_X[idx:idx+1])[0]
        return {
            "formula": candidate_formulas[idx],
            "predicted_value": float(pred),
            "uncertainty": float(unc),
            "confidence_68": [float(pred - unc), float(pred + unc)],
            "confidence_95": [float(pred - 2 * unc), float(pred + 2 * unc)],
            "strategy": self.strategy,
            "iteration": len(self.history),
        }

    @property
    def convergence_curve(self) -> dict:
        """返回学习曲线数据"""
        if not self.history:
            return {}
        iters = [h["iteration"] for h in self.history]
        max_unc = [max(h["uncertainties"]) for h in self.history]
        return {"iterations": iters, "max_uncertainty": max_unc}


def run_active_learning_demo(
    labeled_formulas: List[str],
    labeled_targets: np.ndarray,
    candidate_formulas: List[str],
    candidate_X: np.ndarray,
    featurizer,
    n_rounds: int = 5,
) -> dict:
    """主动学习演示: 多轮推荐实验点.

    返回每轮推荐结果和收敛曲线.
    """
    from .features_v4 import MagpieFeaturizer
    if featurizer is None:
        featurizer = MagpieFeaturizer()

    X_labeled = featurizer.featurize_batch(labeled_formulas)

    learner = ActiveLearner(model_type="random_forest", strategy="uncertainty")
    learner.initialize(X_labeled, np.array(labeled_targets))

    print(f"\n{'='*60}")
    print(f"Active Learning Demo — {len(candidate_formulas)} candidates, {n_rounds} rounds")
    print(f"{'='*60}")

    recommendations = []
    for rnd in range(n_rounds):
        rec = learner.get_next_experiment(candidate_X, candidate_formulas)
        print(f"  Round {rnd+1}: Recommend '{rec['formula']}' "
              f"(pred={rec['predicted_value']:.4f}, unc={rec['uncertainty']:.4f})")
        recommendations.append(rec)

        if rnd < n_rounds - 1:
            fake_measurement = rec["predicted_value"] + np.random.normal(0, rec["uncertainty"] * 0.3)
            idx = candidate_formulas.index(rec["formula"])
            learner.add_measurement(
                candidate_X[idx:idx+1],
                np.array([fake_measurement])
            )
            candidate_formulas = [f for i, f in enumerate(candidate_formulas) if i != idx]
            candidate_X = np.delete(candidate_X, idx, axis=0)

    convergence = learner.convergence_curve
    print(f"\n  Final: {len(learner.labeled_y)} labeled samples")
    return {"recommendations": recommendations, "convergence": convergence}
