"""教材知识库 — PDF加载 / 智能分块 / Embedding / FAISS向量存储"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# 国内用户: 优先使用HF镜像站下载模型
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class TextChunk:
    """知识库文本块."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    chapter: str = ""
    section: str = ""
    metadata: Dict = field(default_factory=dict)


class TextbookLoader:
    """教材PDF加载与解析.

    使用PyMuPDF提取文本, 自动识别章节标题和页码.
    """

    CHAPTER_PATTERNS = [
        re.compile(r'^第[一二三四五六七八九十\d]+章\s*[\.\s、]?\s*(.+)'),
        re.compile(r'^Chapter\s+\d+[\.\s:：]?\s*(.+)', re.IGNORECASE),
        re.compile(r'^\d+[\.\s、]\s*([^\d].{2,})'),
    ]

    SECTION_PATTERNS = [
        re.compile(r'^\d+\.\d+[\.\s、]?\s*(.+)'),
        re.compile(r'^[一二三四五六七八九十]、\s*(.+)'),
    ]

    def __init__(self, textbooks_dir: str = None):
        from config import DATA_DIR
        self.textbooks_dir = textbooks_dir or os.path.join(DATA_DIR, "textbooks")
        os.makedirs(self.textbooks_dir, exist_ok=True)

    def load_textbook(self, filepath: str) -> List[Dict]:
        """加载单个PDF教材, 返回每页的文本和元数据.

        Returns:
            [{"text": str, "page": int, "chapter": str, "section": str}, ...]
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) not installed. Run: pip install PyMuPDF")

        doc = fitz.open(filepath)
        pages = []
        current_chapter = ""
        current_section = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if not text.strip():
                continue

            # 检测章节标题 (通常在前几行)
            lines = text.strip().split("\n")
            for line in lines[:5]:
                line = line.strip()
                for pat in self.CHAPTER_PATTERNS:
                    m = pat.match(line)
                    if m:
                        current_chapter = line
                        current_section = ""
                        break
                for pat in self.SECTION_PATTERNS:
                    m = pat.match(line)
                    if m:
                        current_section = line
                        break

            pages.append({
                "text": text.strip(),
                "page": page_num + 1,  # 1-indexed
                "chapter": current_chapter,
                "section": current_section,
                "source": os.path.basename(filepath),
            })

        doc.close()
        return pages

    def load_all_textbooks(self, directory: str = None) -> List[Dict]:
        """加载目录下所有PDF."""
        directory = directory or self.textbooks_dir
        all_pages = []
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(directory, fname)
                try:
                    pages = self.load_textbook(fpath)
                    all_pages.extend(pages)
                except Exception as e:
                    print(f"Warning: Failed to load {fname}: {e}")
        return all_pages

    def get_available_textbooks(self) -> List[str]:
        """列出可用教材."""
        if not os.path.exists(self.textbooks_dir):
            return []
        return sorted(
            f for f in os.listdir(self.textbooks_dir) if f.lower().endswith(".pdf")
        )


class KnowledgeBase:
    """向量知识库 — 分块 + Embedding + FAISS 检索."""

    def __init__(self, embedding_model_name: str = None,
                 index_path: str = None,
                 device: str = None):
        from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        self.index_path = index_path or FAISS_INDEX_PATH
        self.chunks: List[TextChunk] = []
        self.index = None
        self.embedder = None

        os.makedirs(self.index_path, exist_ok=True)

        if HAS_SBERT:
            self._init_embedder(device or "cpu")

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0

    def _init_embedder(self, device: str = "cpu"):
        """初始化 embedding 模型, 处理网络和缓存问题.

        加载优先级:
        1. 本地缓存 (最快)
        2. HF镜像站 (hf-mirror.com, 国内可用)
        3. HF官方站 (huggingface.co)
        4. 失败则回退到关键词匹配
        """
        import os as _os

        cache_dir = _os.path.join(
            _os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        model_cache = _os.path.join(
            cache_dir, "models--" + self.embedding_model_name.replace("/", "--")
        )

        # 尝试1: 本地缓存
        if _os.path.exists(model_cache):
            try:
                self.embedder = SentenceTransformer(
                    self.embedding_model_name,
                    device=device,
                    local_files_only=True,
                )
                return
            except Exception:
                pass

        # 尝试2: 国内HF镜像站
        mirrors = [
            "https://hf-mirror.com",
            "https://hf.xeduapi.com",
        ]
        for mirror in mirrors:
            try:
                _os.environ["HF_ENDPOINT"] = mirror
                _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
                self.embedder = SentenceTransformer(
                    self.embedding_model_name,
                    device=device,
                )
                print(f"  → 通过镜像站 {mirror} 加载 embedding 模型成功")
                return
            except Exception:
                continue
            finally:
                _os.environ.pop("HF_ENDPOINT", None)

        # 尝试3: HF官方站
        try:
            _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            self.embedder = SentenceTransformer(
                self.embedding_model_name,
                device=device,
            )
        except Exception as e:
            self.embedder = None
            print(f"Warning: Failed to load embedding model '{self.embedding_model_name}': {e}")
            print("  → 将使用关键词匹配回退模式 (不影响基本问答功能)。")
            print("  → 如需向量语义检索, 请手动下载模型到本地缓存。")

    def build_from_pages(self, pages: List[Dict],
                         chunk_size: int = 500,
                         chunk_overlap: int = 100) -> int:
        """从已解析的页面构建知识库.

        Args:
            pages: TextbookLoader.load_all_textbooks() 的输出
            chunk_size: 每块最大字符数
            chunk_overlap: 块间重叠字符数
        Returns:
            创建的块数量
        """
        self.chunks = []
        for page in pages:
            text = page["text"]
            source = page.get("source", "unknown.pdf")
            page_num = page.get("page", 1)
            chapter = page.get("chapter", "")
            section = page.get("section", "")

            if len(text) < 50:
                continue

            # 滑动窗口分块
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) < 50:
                    continue
                self.chunks.append(TextChunk(
                    text=chunk_text,
                    source_file=source,
                    page_number=page_num,
                    chunk_index=i // (chunk_size - chunk_overlap),
                    chapter=chapter,
                    section=section,
                    metadata={"char_start": i, "char_end": i + len(chunk_text)},
                ))

        # 构建FAISS索引
        if HAS_FAISS and HAS_SBERT and self.embedder is not None:
            self._build_faiss_index()
        else:
            print("Warning: FAISS or sentence-transformers not available. Using brute-force search.")

        # 始终保存chunks到磁盘 (即使没有FAISS也能用关键词搜索)
        self._save_index()

        return len(self.chunks)

    def _build_faiss_index(self):
        """构建FAISS向量索引."""
        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(
            texts, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度 (已归一化)
        self.index.add(embeddings.astype(np.float32))
        self._save_index()

    def search(self, query: str, k: int = 5,
               filter_source: str = None) -> List[Tuple[TextChunk, float]]:
        """语义检索 — 返回最相关的k个文本块及相似度分数.

        Args:
            query: 查询文本
            k: 返回数量
            filter_source: 可选, 限定教材来源
        Returns:
            [(TextChunk, score), ...]
        """
        if not self.chunks:
            return []

        if self.index is not None and self.embedder is not None:
            return self._faiss_search(query, k, filter_source)
        else:
            return self._brute_force_search(query, k, filter_source)

    def _faiss_search(self, query: str, k: int,
                      filter_source: str = None) -> List[Tuple[TextChunk, float]]:
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        search_k = k * 3 if filter_source else k
        scores, indices = self.index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_source and chunk.source_file != filter_source:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results

    def _brute_force_search(self, query: str, k: int,
                            filter_source: str = None) -> List[Tuple[TextChunk, float]]:
        """无FAISS时的关键词回退检索 (BM25简化版)."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored = []
        for chunk in self.chunks:
            if filter_source and chunk.source_file != filter_source:
                continue
            text_lower = chunk.text.lower()
            tf = sum(text_lower.count(t) for t in query_terms)
            idf_boost = 1.0 / (1.0 + np.log1p(len(chunk.text)))
            score = tf * idf_boost
            if tf > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _save_index(self):
        """保存FAISS索引和文本块元数据."""
        os.makedirs(self.index_path, exist_ok=True)
        if self.index is not None and HAS_FAISS:
            idx_file = os.path.join(self.index_path, "faiss.index")
            faiss.write_index(self.index, idx_file)

        chunks_data = []
        for c in self.chunks:
            chunks_data.append({
                "text": c.text,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chapter": c.chapter,
                "section": c.section,
                "metadata": c.metadata,
            })

        chunks_file = os.path.join(self.index_path, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    def load_index(self) -> bool:
        """从磁盘加载已有的索引 (chunks + FAISS)."""
        idx_file = os.path.join(self.index_path, "faiss.index")
        chunks_file = os.path.join(self.index_path, "chunks.json")

        if not os.path.exists(chunks_file):
            return False

        if os.path.exists(idx_file) and HAS_FAISS:
            self.index = faiss.read_index(idx_file)

        if os.path.exists(chunks_file):
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = []
            for cd in chunks_data:
                self.chunks.append(TextChunk(
                    text=cd["text"],
                    source_file=cd.get("source_file", ""),
                    page_number=cd.get("page_number", 1),
                    chunk_index=cd.get("chunk_index", 0),
                    chapter=cd.get("chapter", ""),
                    section=cd.get("section", ""),
                    metadata=cd.get("metadata", {}),
                ))

        if HAS_SBERT and not self.embedder:
            try:
                self._init_embedder("cpu")
            except Exception:
                pass

        return self.is_ready

    def get_statistics(self) -> Dict:
        """知识库统计."""
        sources = {}
        chapters = set()
        total_chars = 0
        for c in self.chunks:
            sources[c.source_file] = sources.get(c.source_file, 0) + 1
            if c.chapter:
                chapters.add(c.chapter)
            total_chars += len(c.text)
        return {
            "total_chunks": len(self.chunks),
            "total_chars": total_chars,
            "sources": sources,
            "chapters": sorted(chapters),
            "avg_chunk_size": total_chars // max(len(self.chunks), 1),
        }
