"""智能问答引擎 — RAG概念问答 + 来源追溯 (页码/章节)"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .knowledge_base import KnowledgeBase, TextChunk


@dataclass
class QAAnswer:
    """问答结果."""

    question: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    follow_up_suggestions: List[str] = field(default_factory=list)

    def format_answer(self) -> str:
        """格式化输出 (带引用)."""
        lines = [self.answer, "", "---", "*参考来源:*"]
        for i, src in enumerate(self.sources, 1):
            lines.append(
                f"  [{i}] {src.get('source', 'N/A')}, "
                f"第 {src.get('page', '?')} 页"
                f"{' (' + src.get('chapter', '') + ')' if src.get('chapter') else ''}"
            )
        return "\n".join(lines)


class MaterialsQA:
    """材料科学智能问答.

    基于RAG: 检索教材相关段落 → 构建prompt → 生成回答.
    支持两种模式:
    - Local: 用检索结果拼接context, 无LLM直接返回原文
    - LLM: 调用OpenAI/本地LLM生成结构化回答
    """

    SYSTEM_PROMPT = """你是一位材料科学教授, 正在为本科生解答问题。

要求:
1. 基于提供的教材内容回答, 不要使用外部知识。
2. 用中文回答, 专业术语保留英文原名 (如 dislocation, grain boundary)。
3. 先给出简洁定义, 再用1-2个具体例子说明。
4. 如果教材内容不足以回答, 请明确说明。
5. 结尾标注信息来源 (页码和章节)。

教材内容:
{context}

问题: {question}

请回答:"""

    CONCEPT_SYSTEM_PROMPT = """你是一位材料科学教授。请基于教材内容用中文解释以下概念。

格式要求:
1. **定义**: 一句话概括
2. **原理**: 2-3句说明物理本质
3. **影响因素**: 列出关键影响因素
4. **实际意义**: 在材料工程中的应用
5. **来源**: 教材页码

教材内容:
{context}"""

    def __init__(self, knowledge_base: KnowledgeBase,
                 api_key: str = None,
                 model_name: str = None,
                 base_url: str = None):
        self.kb = knowledge_base
        self.api_key = api_key
        self.model_name = model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    def ask(self, question: str, k: int = 5,
            use_llm: bool = False) -> QAAnswer:
        """回答概念问题.

        Args:
            question: 学生提问
            k: 检索文档数
            use_llm: 是否使用LLM生成回答 (False则返回原文摘要)
        Returns:
            QAAnswer
        """
        if not self.kb or not self.kb.is_ready:
            return QAAnswer(
                question=question,
                answer="知识库尚未构建。请先上传教材PDF并构建索引。",
                confidence=0.0,
            )

        results = self.kb.search(question, k=k)

        if not results:
            return QAAnswer(
                question=question,
                answer="未在教材中找到相关内容。请尝试换个问法。",
                confidence=0.0,
                follow_up_suggestions=self._suggest_followups(question),
            )

        context = self._build_context(results)
        sources = self._extract_sources(results)

        if use_llm and self.api_key:
            answer, confidence = self._llm_answer(question, context)
        else:
            answer = self._local_answer(question, context, results)
            best_score = max(s for _, s in results)
            confidence = min(1.0, best_score)

        return QAAnswer(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            follow_up_suggestions=self._suggest_followups(question),
        )

    def explain_concept(self, concept: str, use_llm: bool = False) -> QAAnswer:
        """详细解释一个概念 (定义+原理+影响因素+应用)."""
        question = f"请详细解释: {concept}"
        k = 8  # 概念解释需要更多上下文

        results = self.kb.search(concept, k=k)
        if not results:
            return QAAnswer(
                question=question,
                answer=f"未找到关于 '{concept}' 的教材内容。",
                confidence=0.0,
            )

        context = self._build_context(results)
        sources = self._extract_sources(results)

        if use_llm and self.api_key:
            answer, confidence = self._llm_answer(
                concept, context,
                system_template=self.CONCEPT_SYSTEM_PROMPT,
            )
        else:
            answer = self._local_concept_explanation(concept, results)
            best_score = max(s for _, s in results)
            confidence = min(1.0, best_score)

        return QAAnswer(
            question=f"解释概念: {concept}",
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    def _build_context(self, results: List[Tuple[TextChunk, float]],
                       max_tokens: int = 3000) -> str:
        """拼接检索结果作为上下文."""
        parts = []
        total_chars = 0
        for chunk, score in results:
            source_info = (
                f"[来源: {chunk.source_file}, "
                f"第{chunk.page_number}页"
                f"{', ' + chunk.chapter if chunk.chapter else ''}]"
            )
            text = f"{source_info}\n{chunk.text}"
            if total_chars + len(text) > max_tokens * 4:
                break
            parts.append(text)
            total_chars += len(text)
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, results: List[Tuple[TextChunk, float]]) -> List[Dict]:
        """提取去重的引用来源."""
        seen = set()
        sources = []
        for chunk, score in results:
            key = (chunk.source_file, chunk.page_number)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": chunk.source_file,
                    "page": chunk.page_number,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "relevance": round(score, 3),
                })
        return sources[:5]

    def _local_answer(self, question: str, context: str,
                      results: List[Tuple[TextChunk, float]]) -> str:
        """无LLM时的回答 — 拼接最相关段落."""
        lines = [f"**问题**: {question}\n"]
        lines.append("**检索结果** (基于教材内容):\n")

        for i, (chunk, score) in enumerate(results[:3], 1):
            preview = chunk.text[:300].replace("\n", " ")
            lines.append(
                f"**[{i}] {chunk.source_file}, 第{chunk.page_number}页** "
                f"(相关度: {score:.2f})\n"
                f"> {preview}...\n"
            )

        lines.append("\n💡 *提示: 配置OpenAI API Key后可使用LLM生成更完整的回答。*")
        return "\n".join(lines)

    def _local_concept_explanation(self, concept: str,
                                    results: List[Tuple[TextChunk, float]]) -> str:
        """本地概念解释 (无LLM)."""
        lines = [f"## {concept}\n"]
        lines.append("**教材内容摘录**:\n")

        for i, (chunk, score) in enumerate(results[:4], 1):
            preview = chunk.text[:400].replace("\n", " ")
            lines.append(
                f"### 参考 [{i}] — {chunk.source_file}, "
                f"第{chunk.page_number}页\n{preview}\n"
            )

        return "\n".join(lines)

    def _llm_answer(self, question: str, context: str,
                    system_template: str = None) -> Tuple[str, float]:
        """调用LLM生成回答."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            template = system_template or self.SYSTEM_PROMPT
            prompt = template.format(context=context, question=question)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是材料科学教授。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            answer = response.choices[0].message.content
            return answer, 0.85
        except Exception as e:
            return f"(LLM调用失败: {e})\n\n请使用本地模式或检查API Key。", 0.3

    def _suggest_followups(self, question: str) -> List[str]:
        """根据问题生成相关追问建议."""
        keywords_map = {
            "位错": ["位错滑移和攀移有什么区别?", "刃型位错和螺型位错如何区分?",
                     "位错密度对强度的影响?"],
            "相图": ["Fe-C相图的关键相变点有哪些?", "杠杆定律如何应用?",
                     "共晶和共析反应有什么区别?"],
            "扩散": ["菲克第一定律和第二定律的区别?", "扩散系数受哪些因素影响?",
                     "柯肯达尔效应的机理?"],
            "强化": ["细晶强化和固溶强化的区别?", "Orowan绕过机制是什么?",
                     "加工硬化的微观机理?"],
            "晶体": ["FCC和BCC的致密度如何计算?", "四面体和八面体间隙是什么?",
                     "密勒指数如何标定?"],
            "相变": ["马氏体相变的特点?", "TTT曲线和CCT曲线的区别?",
                     "固态相变的分类?"],
            "力学": ["应力-应变曲线的各个阶段?", "韧脆转变温度的影响因素?",
                     "疲劳断裂的三个阶段?"],
        }

        for keyword, suggestions in keywords_map.items():
            if keyword in question:
                return suggestions[:3]

        return [
            f"请详细解释{question}的物理本质",
            f"{question}在工程中有什么应用?",
            f"{question}受哪些因素影响?",
        ]
