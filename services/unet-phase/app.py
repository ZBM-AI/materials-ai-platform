"""U-Net 物相分割 Microservice."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI, UploadFile, File
from materials_ai.modules.microscopy_analysis.phase_segmenter import PhaseSegmenter
import numpy as np
import cv2

app = FastAPI(title="U-Net Phase Segmentation Service", version="1.0.0")
segmenter = PhaseSegmenter(model_path=os.environ.get("UNET_MODEL_PATH"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = segmenter.segment(image)
    return {
        "phase_fractions": {k: round(float(v), 4) for k, v in result.get("phase_fractions", {}).items()},
        "method": result.get("method", ""),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
