"""YOLOv8 晶粒检测 Microservice."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI, UploadFile, File
from materials_ai.modules.microscopy_analysis.grain_detector import GrainDetector
import numpy as np
import cv2

app = FastAPI(title="YOLO Grain Detection Service", version="1.0.0")
detector = GrainDetector(model_path=os.environ.get("YOLO_MODEL_PATH"))


@app.post("/predict")
async def predict(file: UploadFile = File(...), pixel_scale_um: float = 0.1):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = detector.detect_grains(image, pixel_scale_um=pixel_scale_um)
    return {
        "num_grains": result.get("num_grains", 0),
        "avg_grain_size_um": result.get("avg_grain_size_um", 0),
        "grain_size_astm": result.get("grain_size_astm", 0),
        "intercept_length_um": result.get("intercept_length_um", 0),
        "method": result.get("method", ""),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
