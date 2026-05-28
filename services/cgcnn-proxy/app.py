"""CGCNN Proxy Microservice — 形成能/带隙预测."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI
from pydantic import BaseModel
from materials_ai.modules.crystal_generation.cgcnn_proxy import CGCNNProxy, DefaultEnergyPredictor
from materials_ai.modules.crystal_generation.crystal_representation import CrystalStructure

app = FastAPI(title="CGCNN Proxy Service", version="1.0.0")

proxy = DefaultEnergyPredictor()

model_path = os.environ.get("CGCNN_MODEL_PATH", "/models/cgcnn_proxy.pt")
if os.path.exists(model_path):
    try:
        proxy = CGCNNProxy.load(model_path)
    except Exception:
        pass


class PredictRequest(BaseModel):
    lattice: list  # (3, 3) matrix
    frac_coords: list  # (N, 3)
    atom_types: list  # (N,) atomic numbers
    space_group: int = 1


class PredictResponse(BaseModel):
    formation_energy_eV: float
    band_gap_eV: float
    stability_score: float
    is_stable: bool


@app.post("/predict")
async def predict(body: PredictRequest):
    import numpy as np
    struct = CrystalStructure(
        lattice=np.array(body.lattice),
        frac_coords=np.array(body.frac_coords),
        atom_types=np.array(body.atom_types),
        space_group=body.space_group,
    )
    result = proxy.predict(struct)
    return PredictResponse(**result)


@app.get("/health")
async def health():
    return {"status": "healthy"}
