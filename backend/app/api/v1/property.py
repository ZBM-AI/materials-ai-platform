"""Module 3: 性能预测 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends
from ...core.dependencies import get_current_user
from ...schemas.common import PropertyPrediction, APIResponse

router = APIRouter(prefix="/property", tags=["性能预测"])


@router.post("/predict")
async def predict_property(body: PropertyPrediction, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.property_prediction.models import PropertyPredictor
        import config
        results = {}
        target_to_path = {
            "band_gap": config.BAND_GAP_MODEL,
            "formation_energy": config.FORMATION_ENERGY_MODEL,
            "bulk_modulus": config.MECHANICAL_MODEL,
            "thermal_conductivity": config.THERMAL_CONDUCTIVITY_MODEL,
            "yield_strength": config.YIELD_STRENGTH_MODEL,
        }
        for target in body.targets:
            path = target_to_path.get(target)
            if path and os.path.exists(path):
                model = PropertyPredictor.load(path)
                pred = model.predict(body.formula)
                results[target] = float(pred) if hasattr(pred, 'item') else pred
        return APIResponse(success=True, data={"formula": body.formula, "predictions": results})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/models")
async def list_models(user=Depends(get_current_user)):
    import config, os
    models = {}
    for name, path in [
        ("band_gap", config.BAND_GAP_MODEL),
        ("formation_energy", config.FORMATION_ENERGY_MODEL),
        ("bulk_modulus", config.MECHANICAL_MODEL),
        ("thermal_conductivity", config.THERMAL_CONDUCTIVITY_MODEL),
        ("yield_strength", config.YIELD_STRENGTH_MODEL),
    ]:
        models[name] = {"available": os.path.exists(path), "path": path}
    return APIResponse(success=True, data=models)
