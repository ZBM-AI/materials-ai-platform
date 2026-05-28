"""通用 API Schemas."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class APIResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Any] = None


class LiteratureQuery(BaseModel):
    query: str
    max_results: int = 20
    source: str = "arxiv"


class LiteratureAnalysis(BaseModel):
    text: str
    model: str = "scibert"


class NEREntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class TripletData(BaseModel):
    material: str
    property: Optional[str] = None
    value: Optional[str] = None
    evidence: str = ""
    confidence: float = 1.0


class RAGQuery(BaseModel):
    question: str
    k: int = 5
    use_llm: bool = False


class KGQuery(BaseModel):
    cypher: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None


class PropertyPrediction(BaseModel):
    formula: str
    targets: List[str] = ["formation_energy", "band_gap"]


class MicroscopyRequest(BaseModel):
    pixel_scale_um: float = 0.1
    min_grain_area: int = 100
    min_defect_area: int = 20
    denoise: bool = True
    equalize: bool = True


class CrystalGenerationRequest(BaseModel):
    elements: List[str]
    stoichiometry: List[float] = None
    space_group: int = 1
    num_candidates: int = 100
    num_steps: int = 100
    temperature: float = 1.0
    top_k: int = 10


class QuizRequest(BaseModel):
    topic: str
    num_mcq: int = 8
    num_calc: int = 2


class ExperimentRequest(BaseModel):
    composition: str  # e.g. "Fe-0.45%C"
    temperature: float = 25.0


class HeatTreatmentRequest(BaseModel):
    carbon_content: float = 0.45
    target_property: str = "balanced"


class CodeGenerationRequest(BaseModel):
    query: str
    execute: bool = False
