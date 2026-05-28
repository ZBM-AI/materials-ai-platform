"""晶体扩散模型 Microservice."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI
from pydantic import BaseModel
from materials_ai.modules.crystal_generation.structure_generator import CrystalGenerator
import torch

app = FastAPI(title="Crystal Diffusion Service", version="1.0.0")

generator = CrystalGenerator()
model_path = os.environ.get("DIFFUSION_MODEL_PATH", "/models/crystal_diffusion.pt")
if os.path.exists(model_path):
    from materials_ai.modules.crystal_generation.diffusion_model import CrystalDiffusion
    diffusion = CrystalDiffusion()
    diffusion.load_state_dict(torch.load(model_path, map_location="cpu").get("model_state_dict", {}), strict=False)
    diffusion.eval()
    generator.diffusion_model = diffusion


class GenerateRequest(BaseModel):
    elements: list
    stoichiometry: list = None
    space_group: int = 1
    num_candidates: int = 100
    num_steps: int = 100
    temperature: float = 1.0
    top_k: int = 10


class GenerateResponse(BaseModel):
    num_generated: int
    num_passed: int
    candidates: list
    predictions: list


@app.post("/generate")
async def generate(body: GenerateRequest):
    result = generator.generate(
        elements=body.elements,
        stoichiometry=body.stoichiometry,
        space_group=body.space_group,
        num_candidates=body.num_candidates,
        num_steps=body.num_steps,
        temperature=body.temperature,
        top_k=body.top_k,
    )
    return GenerateResponse(
        num_generated=result["num_total_generated"],
        num_passed=result["num_passed"],
        candidates=[{"formula": s._formula_string(), "cif": s.to_cif_string(),
                      "num_atoms": s.num_atoms, "volume": round(s.volume, 2)}
                     for s in result["candidates"]],
        predictions=result["predictions"],
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "has_model": generator.diffusion_model is not None}
