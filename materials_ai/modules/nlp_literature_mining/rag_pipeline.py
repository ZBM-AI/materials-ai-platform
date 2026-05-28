"""RAG文献问答 — LangChain + Chroma向量数据库 + Embeddings"""

import os
from typing import List, Optional, Dict

import config

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    from langchain_classic.chains import RetrievalQA
    from langchain_community.llms import OpenAI as LangChainOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_community.llms import OpenAI as LangChainOpenAI
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False


class MaterialsRAG:
    """材料科学文献RAG问答系统.

    使用LangChain构建: 文本分割 → Embedding → Chroma索引 → 检索问答.
    支持 OpenAI embeddings 和本地 HuggingFace embeddings.
    """

    def __init__(self, embedding_model: str = None, chroma_dir: str = None,
                 collection_name: str = None, use_openai: bool = False,
                 openai_api_key: str = None, chunk_size: int = None,
                 chunk_overlap: int = None):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "langchain not installed. "
                "Run: pip install langchain langchain-community chromadb sentence-transformers"
            )
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL_NAME
        self.chroma_dir = chroma_dir or config.CHROMA_DIR
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self._embeddings = None
        self._vectorstore = None
        self._qa_chain = None
        self._llm = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            if self.use_openai and self.openai_api_key:
                self._embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model="text-embedding-ada-002",
                )
            else:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
        return self._embeddings

    def _get_llm(self):
        if self._llm is None:
            if self.openai_api_key:
                self._llm = LangChainOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                )
            else:
                raise RuntimeError(
                    "OpenAI API key required for Q&A generation. "
                    "Set OPENAI_API_KEY env var or pass openai_api_key."
                )
        return self._llm

    def index_papers(self, papers: List[dict], source_field: str = "raw_text"):
        """将论文全文索引到Chroma向量数据库.

        Args:
            papers: [{"paper_id": ..., "raw_text": ..., "title": ..., ...}, ...]
            source_field: 用于索引的文本字段名
        """
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for paper in papers:
            text = paper.get(source_field, "") or paper.get("abstract", "")
            if not text:
                continue
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "")
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "page_content": chunk,
                    "metadata": {
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_index": i,
                        "source": paper_id,
                    },
                })

        if not documents:
            raise ValueError("No documents to index. Provide papers with text content.")

        texts = [d["page_content"] for d in documents]
        metadatas = [d["metadata"] for d in documents]

        os.makedirs(self.chroma_dir, exist_ok=True)

        self._vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
        )
        return len(documents)

    def load_index(self):
        """加载已有Chroma索引."""
        if not os.path.exists(self.chroma_dir):
            raise FileNotFoundError(f"Chroma index not found at {self.chroma_dir}")
        self._vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
        )
        return self

    def search(self, query: str, k: int = None) -> List[dict]:
        """语义检索, 返回top-k相关文档片段."""
        if self._vectorstore is None:
            raise RuntimeError("No index loaded. Call index_papers() or load_index() first.")
        k = k or config.RAG_TOP_K
        docs = self._vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "paper_id": doc.metadata.get("paper_id", ""),
                "title": doc.metadata.get("title", ""),
                "score": doc.metadata.get("score", 0),
            }
            for doc in docs
        ]

    def ask(self, question: str, k: int = None) -> dict:
        """问答: 检索 + LLM生成答案 + 来源."""
        if self._vectorstore is None:
            raise RuntimeError("No index loaded. Call index_papers() or load_index() first.")

        k = k or config.RAG_TOP_K
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        llm = self._get_llm()

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        result = self._qa_chain({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "paper_id": doc.metadata.get("paper_id", ""),
                "title": doc.metadata.get("title", ""),
                "excerpt": doc.page_content[:200],
            })

        return {
            "question": question,
            "answer": result.get("result", ""),
            "sources": sources,
        }

    @property
    def is_indexed(self) -> bool:
        return self._vectorstore is not None

    def get_stats(self) -> dict:
        """获取索引统计."""
        if self._vectorstore is None:
            return {"indexed": False, "document_count": 0}
        try:
            count = self._vectorstore._collection.count()
        except Exception:
            count = 0
        return {"indexed": True, "document_count": count}
