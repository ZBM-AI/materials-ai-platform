"""Module 4: 显微图像分析 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import numpy as np
import tempfile
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from ...core.dependencies import get_current_user
from ...schemas.common import MicroscopyRequest, APIResponse

router = APIRouter(prefix="/microscopy", tags=["显微分析"])


@router.post("/analyze")
async def analyze_micrograph(
    params: MicroscopyRequest = Depends(MicroscopyRequest),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    try:
        import cv2
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        from materials_ai.modules.microscopy_analysis.phase_segmenter import PhaseSegmenter
        from materials_ai.modules.microscopy_analysis.grain_detector import GrainDetector
        from materials_ai.modules.microscopy_analysis.defect_classifier import DefectAnalyzer
        from materials_ai.modules.microscopy_analysis.structure_classifier import MicrostructureClassifier

        phase = PhaseSegmenter().segment(image)
        grain = GrainDetector().detect_grains(image, pixel_scale_um=params.pixel_scale_um,
                                               min_grain_area_px=params.min_grain_area)
        defect = DefectAnalyzer().analyze(image, min_defect_area_px=params.min_defect_area)
        structure = MicrostructureClassifier().classify(image)

        return APIResponse(success=True, data={
            "phase_fractions": phase.get("phase_fractions", {}),
            "num_grains": grain.get("num_grains", 0),
            "avg_grain_size_um": grain.get("avg_grain_size_um", 0),
            "grain_size_astm": grain.get("grain_size_astm", 0),
            "total_defects": defect.get("total_defects", 0),
            "defect_fraction": defect.get("defect_fraction", 0),
            "predicted_microstructure": structure.get("predicted_class", "unknown"),
            "structure_probabilities": structure.get("probabilities", {}),
        })
    except Exception as e:
        raise HTTPException(500, str(e))
