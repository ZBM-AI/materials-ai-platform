"""智能学习与问答助手模块 — RAG教材知识库 + 自动出题 + 代码辅助 + 实验建议"""

from .knowledge_base import TextbookLoader, KnowledgeBase, TextChunk
from .qa_engine import MaterialsQA, QAAnswer
from .quiz_generator import QuizGenerator, QuizQuestion, Quiz
from .code_assistant import CodeAssistant, CodeResult
from .experiment_advisor import ExperimentAdvisor, PhasePrediction, ExperimentAdvice
