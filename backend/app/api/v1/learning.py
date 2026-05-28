"""Module 6: 智能学习助手 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends
from ...core.dependencies import get_current_user
from ...schemas.common import (
    RAGQuery, QuizRequest, CodeGenerationRequest,
    ExperimentRequest, HeatTreatmentRequest, APIResponse,
)

router = APIRouter(prefix="/learning", tags=["学习助手"])


@router.post("/qa")
async def ask_question(body: RAGQuery, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.learning_assistant.knowledge_base import KnowledgeBase
        from materials_ai.modules.learning_assistant.qa_engine import MaterialsQA
        kb = KnowledgeBase()
        if not kb.load_index():
            return APIResponse(success=False, message="知识库未构建")
        qa = MaterialsQA(kb)
        result = qa.ask(body.question, k=body.k, use_llm=body.use_llm)
        return APIResponse(success=True, data={
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "follow_ups": result.follow_up_suggestions,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/quiz")
async def generate_quiz(body: QuizRequest, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.learning_assistant.knowledge_base import KnowledgeBase
        from materials_ai.modules.learning_assistant.quiz_generator import QuizGenerator
        kb = KnowledgeBase()
        kb.load_index()
        qg = QuizGenerator(kb)
        quiz = qg.generate_quiz(body.topic, body.num_mcq, body.num_calc)
        return APIResponse(success=True, data={
            "title": quiz.title,
            "total_points": quiz.total_points,
            "duration_minutes": quiz.duration_minutes,
            "questions": [q.to_dict() for q in quiz.questions],
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/code")
async def generate_code(body: CodeGenerationRequest, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.learning_assistant.code_assistant import CodeAssistant
        ca = CodeAssistant()
        result = ca.generate_code(body.query)
        if body.execute and result.success:
            exec_result = ca.execute_code(result.code)
            return APIResponse(success=True, data={
                "code": result.code, "description": result.description,
                "output": exec_result.output, "error": exec_result.error,
            })
        return APIResponse(success=True, data={"code": result.code, "description": result.description})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/predict-phase")
async def predict_phase(body: ExperimentRequest, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.learning_assistant.experiment_advisor import ExperimentAdvisor
        from materials_ai.modules.learning_assistant.crystal_representation import ELEM_TO_ATOMIC_NUM
        advisor = ExperimentAdvisor()
        parts = body.composition.replace("%", "").split("-")
        elements = {}
        for part in parts:
            part = part.strip()
            for el in ["Fe", "C", "Al", "Cu", "Ti", "V", "Ni", "Cr", "Co"]:
                if part.startswith(el):
                    try:
                        elements[el] = float(part.replace(el, ""))
                    except ValueError:
                        elements[el] = 50.0
                    break
        prediction = advisor.predict_phases(elements, body.temperature)
        return APIResponse(success=True, data={
            "phases": [{"name": p["phase"], "fraction": p.get("fraction", 0)}
                       for p in prediction.predicted_phases],
            "confidence": prediction.confidence,
            "temperature": body.temperature,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/heat-treatment")
async def suggest_heat_treatment(body: HeatTreatmentRequest, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.learning_assistant.experiment_advisor import ExperimentAdvisor
        advisor = ExperimentAdvisor()
        advice = advisor.suggest_heat_treatment(body.carbon_content, body.target_property)
        return APIResponse(success=True, data={
            "title": advice.title,
            "process": advice.suggested_process,
            "parameters": advice.parameters,
            "expected_results": advice.expected_results,
            "precautions": advice.precautions,
        })
    except Exception as e:
        raise HTTPException(500, str(e))
