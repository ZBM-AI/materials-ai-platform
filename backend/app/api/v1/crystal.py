"""Module 5: 晶体结构生成 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends
from ...core.dependencies import get_current_user
from ...schemas.common import CrystalGenerationRequest, APIResponse

router = APIRouter(prefix="/crystal", tags=["晶体生成"])


@router.post("/generate")
async def generate_crystals(body: CrystalGenerationRequest, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.crystal_generation.structure_generator import CrystalGenerator
        generator = CrystalGenerator()
        result = generator.generate(
            elements=body.elements,
            stoichiometry=body.stoichiometry,
            space_group=body.space_group,
            num_candidates=body.num_candidates,
            num_steps=body.num_steps,
            temperature=body.temperature,
            top_k=body.top_k,
        )
        return APIResponse(success=True, data={
            "num_generated": result["num_total_generated"],
            "num_passed": result["num_passed"],
            "candidates": [
                {
                    "formula": s._formula_string(),
                    "num_atoms": s.num_atoms,
                    "volume": round(s.volume, 2),
                    "density": round(s.density, 2),
                    "cif": s.to_cif_string(),
                }
                for s in result["candidates"]
            ],
            "predictions": [
                {
                    "formation_energy_eV": p["formation_energy_eV"],
                    "band_gap_eV": p["band_gap_eV"],
                    "stability_score": p["stability_score"],
                    "is_stable": p["is_stable"],
                }
                for p in result["predictions"]
            ],
        })
    except Exception as e:
        raise HTTPException(500, str(e))
