"""材料文献挖掘模块 — PDF解析 → NER → 关系抽取 → 三元组存储 → RAG检索"""

from .pdf_parser import PDFParser
from .materials_ner import MaterialsNER, Entity
from .relation_extractor import RelationExtractor, Relation, Triplet
from .search_engine import LiteratureSearchEngine, SearchResult
from .downloader import PaperDownloader, PaperMetadata
from .database import TripletStore, PaperStore
from .bio_schema import (
    EntitySpan, tokens_to_spans, spans_to_bio,
    align_labels_with_wordpieces, validate_bio_consistency,
)
from .ner_trainer import MaterialsNERDataset, SciBERTNERTrainer
from .scibert_ner import SciBERTNER
from .rag_pipeline import MaterialsRAG
from .report_generator import ReportGenerator
from .paper_analyzer import (
    PaperAnalyzer, DeepAnalysisResult, PaperDiscovery, InnovationPoint, PaperShortcoming,
)
from .paper_comparator import PaperComparator, ComparisonMatrix
