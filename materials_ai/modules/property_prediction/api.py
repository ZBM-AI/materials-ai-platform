"""FastAPI模型服务 — /predict端点, JSON输入输出, 置信区间

启动: uvicorn modules.property_prediction.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from typing import List, Optional, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import config

app = FastAPI(
    title="Materials Property Prediction API v4",
    description="输入化学成分和工艺参数, 预测材料性能 (带隙/形成能/体模量/剪切模量)",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}

SYNTHESIS_METHODS = [
    "solid_state", "sol_gel", "hydrothermal", "co_precipitation",
    "ball_milling", "CVD", "PVD", "spray_pyrolysis", "molten_salt",
    "microwave", "spark_plasma_sintering", "hot_press", "cold_press",
]


class ProcessParams(BaseModel):
    temperature: Optional[float] = Field(default=300, description="合成温度 (K)")
    pressure: Optional[float] = Field(default=1.0, description="合成压力 (atm)")
    time: Optional[float] = Field(default=24.0, description="合成时间 (h)")
    method: Optional[str] = Field(default="solid_state", description="合成方法")
    atmosphere: Optional[str] = Field(default="air", description="气氛")
    annealing_temp: Optional[float] = Field(default=None, description="退火温度 (K)")
    annealing_time: Optional[float] = Field(default=12.0, description="退火时间 (h)")


class PredictionRequest(BaseModel):
    compositions: List[str] = Field(..., description="化学式列表, 如 ['TiO2', 'Fe2O3']")
    target: str = Field(
        default="band_gap",
        description="预测目标: band_gap, formation_energy, bulk_modulus, shear_modulus"
    )
    process: Optional[ProcessParams] = Field(
        default=None, description="合成工艺参数 (可选)"
    )
    return_std: bool = Field(default=True, description="是否返回预测标准差")


class PredictionItem(BaseModel):
    composition: str
    predicted_value: float
    unit: str
    std_deviation: Optional[float] = None
    confidence_68: Optional[List[float]] = None
    confidence_95: Optional[List[float]] = None


class PredictionResponse(BaseModel):
    target: str
    unit: str
    model_type: str
    predictions: List[PredictionItem]
    metadata: Dict = {}


def load_models():
    global MODELS
    if MODELS:
        return MODELS
    from modules.property_prediction.models import PropertyPredictor
    model_configs = {
        "band_gap": (config.BAND_GAP_MODEL, "RandomForest"),
        "formation_energy": (config.FORMATION_ENERGY_MODEL, "GradientBoosting"),
        "bulk_modulus": (config.MECHANICAL_MODEL, "RandomForest"),
        "shear_modulus": (config.MECHANICAL_MODEL.replace("mechanical", "shear_modulus"), "RandomForest"),
    }
    for key, (path, algo) in model_configs.items():
        if os.path.exists(path):
            MODELS[key] = {
                "predictor": PropertyPredictor.load(path),
                "algorithm": algo,
            }
    return MODELS


UNITS = {
    "band_gap": "eV",
    "formation_energy": "eV/atom",
    "bulk_modulus": "GPa",
    "shear_modulus": "GPa",
}


@app.on_event("startup")
def startup():
    load_models()
    print(f"[API] Loaded {len(MODELS)} models: {list(MODELS.keys())}")


@app.get("/")
def root():
    return {
        "service": "Materials Property Prediction API v4",
        "endpoints": ["/predict", "/health", "/models", "/docs"],
        "available_targets": list(UNITS.keys()),
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(load_models())}


@app.get("/models")
def list_models():
    models_info = {}
    for key, info in load_models().items():
        m = info["predictor"]
        models_info[key] = {
            "algorithm": info["algorithm"],
            "cv_r2_mean": m.cv_results.get("cv_r2_mean", None),
            "cv_mae_mean": m.cv_results.get("cv_mae_mean", None),
            "feature_dim": len(m._feature_names) if m._feature_names else None,
        }
    return models_info


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    models = load_models()
    if req.target not in models:
        target_options = list(models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target '{req.target}'. Options: {target_options}"
        )
    if req.target not in UNITS:
        raise HTTPException(status_code=400, detail=f"No unit defined for {req.target}")

    model_info = models[req.target]
    predictor = model_info["predictor"]
    unit = UNITS[req.target]
    results = []

    for comp in req.compositions:
        try:
            if req.return_std:
                val, std = predictor.predict_with_std(comp)
            else:
                val = predictor.predict(comp)
                std = 0.0

            item = PredictionItem(
                composition=comp,
                predicted_value=round(float(val), 6),
                unit=unit,
                std_deviation=round(float(std), 6) if std > 0 else None,
                confidence_68=[round(val - std, 6), round(val + std, 6)] if std > 0 else None,
                confidence_95=[round(val - 2 * std, 6), round(val + 2 * std, 6)] if std > 0 else None,
            )
            results.append(item)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction failed for '{comp}': {str(e)}"
            )

    return PredictionResponse(
        target=req.target,
        unit=unit,
        model_type=model_info["algorithm"],
        predictions=results,
        metadata={
            "cv_r2_mean": model_info["predictor"].cv_results.get("cv_r2_mean"),
            "n_predictions": len(results),
        },
    )


@app.get("/synthesis_methods")
def get_synthesis_methods():
    return {"methods": SYNTHESIS_METHODS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
