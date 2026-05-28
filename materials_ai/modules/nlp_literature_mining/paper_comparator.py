"""多论文对比分析 — 横向比较发现/创新点/不足/方法/材料"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter

from .paper_analyzer import (
    DeepAnalysisResult, PaperDiscovery, InnovationPoint, PaperShortcoming,
)


@dataclass
class ComparisonMatrix:
    """多论文对比矩阵."""
    paper_titles: List[str] = field(default_factory=list)
    common_discoveries: List[str] = field(default_factory=list)
    conflicting_findings: List[Dict] = field(default_factory=list)
    innovation_heatmap: Dict[str, List[str]] = field(default_factory=dict)
    shortcoming_summary: Dict[str, List[str]] = field(default_factory=dict)
    methodology_comparison: Dict[str, str] = field(default_factory=dict)
    material_overlap: List[str] = field(default_factory=list)
    property_overlap: List[str] = field(default_factory=list)
    consensus_points: List[str] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    trend_analysis: str = ""
    cross_paper_summary: str = ""


class PaperComparator:
    """多论文横向对比分析器.

    功能:
    - 交叉对比多篇论文的发现、创新点和不足
    - 识别共识和分歧
    - 发现研究空白和趋势
    - 生成对比矩阵和可视化数据
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com"
        self.model = model or "deepseek-chat"

    def compare(self, analyses: List[Dict]) -> ComparisonMatrix:
        """对多篇论文的分析结果进行横向对比.

        Args:
            analyses: [
                {"title": "...", "analysis": DeepAnalysisResult, "entities": [...], "triplets": [...]},
                ...
            ]

        Returns:
            ComparisonMatrix
        """
        if len(analyses) < 2:
            return self._single_paper_matrix(analyses)

        matrix = ComparisonMatrix()
        matrix.paper_titles = [a.get("title", f"Paper {i+1}") for i, a in enumerate(analyses)]

        # 1. 发现交叉对比
        matrix.common_discoveries, matrix.conflicting_findings = \
            self._cross_compare_discoveries(analyses)

        # 2. 创新点热力图数据
        matrix.innovation_heatmap = self._build_innovation_heatmap(analyses)

        # 3. 不足汇总
        matrix.shortcoming_summary = self._build_shortcoming_summary(analyses)

        # 4. 方法论对比
        matrix.methodology_comparison = self._build_methodology_comparison(analyses)

        # 5. 材料/性能重叠
        matrix.material_overlap, matrix.property_overlap = \
            self._find_overlaps(analyses)

        # 6. 共识点和研究空白
        matrix.consensus_points, matrix.research_gaps = \
            self._identify_consensus_and_gaps(analyses, matrix)

        # 7. 趋势分析
        matrix.trend_analysis = self._analyze_trends(analyses)

        # 8. 跨论文综合总结
        matrix.cross_paper_summary = self._generate_cross_summary(matrix, analyses)

        return matrix

    def _cross_compare_discoveries(self, analyses: List[Dict]) -> tuple:
        """交叉对比各论文的发现."""
        all_discoveries_by_type = {}
        for a in analyses:
            analysis = a.get("analysis")
            if not analysis:
                continue
            for disc in analysis.discoveries:
                key = disc.type
                all_discoveries_by_type.setdefault(key, []).append({
                    "paper": a.get("title", ""),
                    "description": disc.description,
                    "evidence": disc.evidence,
                })

        common = []
        conflicts = []

        for dtype, items in all_discoveries_by_type.items():
            if len(items) >= 2:
                descriptions = [it["description"].lower()[:80] for it in items]
                unique_descs = set(descriptions)
                if len(unique_descs) < len(descriptions):
                    common.append(f"[{dtype}] 多篇论文关注相似的{dtype}方面")
                else:
                    conflicts.append({
                        "type": dtype,
                        "papers": [it["paper"] for it in items],
                        "descriptions": [it["description"] for it in items],
                    })

        return common, conflicts

    def _build_innovation_heatmap(self, analyses: List[Dict]) -> Dict[str, List[str]]:
        """构建创新点热力图数据."""
        heatmap = {}
        categories = ["method", "material", "theory", "application", "performance"]

        for cat in categories:
            row = []
            for a in analyses:
                analysis = a.get("analysis")
                if not analysis:
                    row.append("-")
                    continue
                cat_innovations = [i for i in analysis.innovations if i.category == cat]
                if cat_innovations:
                    row.append(f"{len(cat_innovations)}项: {cat_innovations[0].description[:60]}")
                else:
                    row.append("-")
            heatmap[f"创新维度: {cat}"] = row

        return heatmap

    def _build_shortcoming_summary(self, analyses: List[Dict]) -> Dict[str, List[str]]:
        """汇总所有论文的不足."""
        summary = {}
        for a in analyses:
            analysis = a.get("analysis")
            title = a.get("title", "")
            if not analysis:
                continue
            severities = {"major": [], "moderate": [], "minor": []}
            for s in analysis.shortcomings:
                severities.setdefault(s.severity, []).append(s.description[:100])
            summary[title] = severities.get("major", [])[:3] + \
                             severities.get("moderate", [])[:3] + \
                             severities.get("minor", [])[:2]
        return summary

    def _build_methodology_comparison(self, analyses: List[Dict]) -> Dict[str, str]:
        """方法论对比."""
        comparison = {}
        for a in analyses:
            analysis = a.get("analysis")
            title = a.get("title", "")
            if analysis and analysis.methodology:
                comparison[title] = analysis.methodology[:200]
            else:
                comparison[title] = "未提取到方法学信息"
        return comparison

    def _find_overlaps(self, analyses: List[Dict]) -> tuple:
        """找到论文间重叠的材料和性能."""
        all_materials = []
        all_properties = []

        for a in analyses:
            entities = a.get("entities", [])
            for e in entities:
                etype = getattr(e, "entity_type", "")
                text = getattr(e, "text", str(e))
                if etype == "material":
                    all_materials.append(text)
                elif etype == "property":
                    all_properties.append(text)

        mat_counts = Counter(all_materials)
        prop_counts = Counter(all_properties)

        material_overlap = [m for m, c in mat_counts.most_common(10) if c >= 2]
        property_overlap = [p for p, c in prop_counts.most_common(10) if c >= 2]

        return material_overlap, property_overlap

    def _identify_consensus_and_gaps(self, analyses: List[Dict],
                                     matrix: ComparisonMatrix) -> tuple:
        """识别共识点和研究空白."""
        consensus = []
        gaps = []

        # 共识: 多篇论文共同确认的发现
        if matrix.common_discoveries:
            consensus.append(f"多篇论文在 {len(matrix.common_discoveries)} 个发现维度上有一致结论")

        if matrix.material_overlap:
            consensus.append(f"共同关注的材料: {', '.join(matrix.material_overlap[:5])}")

        if matrix.property_overlap:
            consensus.append(f"共同关注的性能: {', '.join(matrix.property_overlap[:3])}")

        # 空白: 未被覆盖的领域
        all_cats = set()
        for a in analyses:
            analysis = a.get("analysis")
            if analysis:
                for s in analysis.shortcomings:
                    all_cats.add(s.category)

        missing_categories = ["validation", "comparison", "scope", "method", "gap"]
        for cat in missing_categories:
            if cat not in all_cats:
                gaps.append(f"未充分讨论 {cat} 方面的局限 (包括但不限于: 实验验证不足、对比基准缺失等)")

        # 创新维度空白
        innovation_cats_present = set()
        for a in analyses:
            analysis = a.get("analysis")
            if analysis:
                for inn in analysis.innovations:
                    innovation_cats_present.add(inn.category)

        all_innovation_cats = {"method", "material", "theory", "application", "performance"}
        missing_innov = all_innovation_cats - innovation_cats_present
        for cat in missing_innov:
            gaps.append(f"创新维度 [{cat}] 在各论文中均未涉及")

        if not consensus:
            consensus.append("论文间结论差异较大, 未形成明确共识")
        if not gaps:
            gaps.append("各论文覆盖了较全面的研究维度")

        return consensus, gaps

    def _analyze_trends(self, analyses: List[Dict]) -> str:
        """分析研究趋势."""
        years = []
        methods = []
        materials = []

        for a in analyses:
            entities = a.get("entities", [])
            for e in entities:
                etype = getattr(e, "entity_type", "")
                text = getattr(e, "text", str(e))
                if etype == "synthesis_method":
                    methods.append(text)
                elif etype == "material":
                    materials.append(text)

        parts = []

        method_counts = Counter(methods)
        if method_counts:
            top_method = method_counts.most_common(3)
            parts.append(f"关注方法: {', '.join(m + f'({c}篇)' for m, c in top_method)}")

        mat_counts = Counter(materials)
        if mat_counts:
            top_mat = mat_counts.most_common(3)
            parts.append(f"热门材料: {', '.join(m + f'({c}篇)' for m, c in top_mat)}")

        if not parts:
            parts.append("数据不足以分析趋势。建议添加更多论文。")

        parts.append(f"基于 {len(analyses)} 篇论文的横向对比")
        return "\n".join(parts)

    def _generate_cross_summary(self, matrix: ComparisonMatrix,
                                analyses: List[Dict]) -> str:
        """生成跨论文综合总结."""
        n = len(analyses)
        parts = [f"对 {n} 篇论文进行了横向对比分析。"]

        if matrix.consensus_points:
            parts.append("**共识**: " + "; ".join(matrix.consensus_points[:3]))

        if matrix.conflicting_findings:
            parts.append(f"**分歧**: 发现 {len(matrix.conflicting_findings)} 处差异点")

        if matrix.research_gaps:
            parts.append(f"**研究空白**: {len(matrix.research_gaps)} 处未被覆盖的研究维度")

        if matrix.material_overlap:
            parts.append(f"**交叉材料**: {', '.join(matrix.material_overlap[:5])}")

        if self.api_key:
            parts.append(self._llm_cross_summary(matrix, analyses))

        return "\n\n".join(parts)

    def _llm_cross_summary(self, matrix: ComparisonMatrix,
                           analyses: List[Dict]) -> str:
        """LLM生成跨论文总结."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            titles = [a.get("title", "") for a in analyses]
            summaries = []
            for a in analyses:
                analysis = a.get("analysis")
                if analysis:
                    summaries.append(analysis.summary[:500])

            prompt = f"""请对以下 {len(titles)} 篇材料科学论文进行横向对比总结 (用中文):

论文列表:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

各论文摘要:
{chr(10).join(f'{i+1}. {s[:300]}' for i, s in enumerate(summaries))}

共识点: {matrix.consensus_points}
研究空白: {matrix.research_gaps}
材料重叠: {matrix.material_overlap}

请用200字左右总结这些论文之间的关联、异同和互补性, 侧重分析:
1. 这些研究在什么方向上是互补的?
2. 有什么共同的局限性?
3. 未来研究的切入点在哪里?"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800,
            )
            return "(AI分析): " + response.choices[0].message.content
        except Exception:
            return ""

    def _single_paper_matrix(self, analyses: List[Dict]) -> ComparisonMatrix:
        """只有一篇论文时的简化矩阵."""
        matrix = ComparisonMatrix()
        if analyses:
            a = analyses[0]
            matrix.paper_titles = [a.get("title", "Paper 1")]
            matrix.cross_paper_summary = "仅有一篇论文, 无法进行横向对比。请上传至少2篇论文进行对比分析。"
            analysis = a.get("analysis")
            if analysis:
                matrix.methodology_comparison = {
                    a.get("title", ""): analysis.methodology or "未提取"
                }
                matrix.shortcoming_summary = {
                    a.get("title", ""): [s.description for s in analysis.shortcomings]
                }
        return matrix

    def to_comparison_table(self, matrix: ComparisonMatrix) -> Dict:
        """将对比矩阵转为前端可用的表格数据."""
        rows = []
        n = len(matrix.paper_titles)

        for i in range(n):
            row = {"论文": matrix.paper_titles[i]}
            for j, cat in enumerate(
                ["method", "material", "theory", "application", "performance"]
            ):
                key = f"创新维度: {cat}"
                if key in matrix.innovation_heatmap:
                    row[cat] = matrix.innovation_heatmap[key][i] if i < len(
                        matrix.innovation_heatmap[key]
                    ) else "-"
                else:
                    row[cat] = "-"
            rows.append(row)

        return {
            "columns": ["论文", "method", "material", "theory", "application", "performance"],
            "rows": rows,
        }
