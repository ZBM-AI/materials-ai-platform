"""文献综述生成器 — 模板综述 + Plotly趋势分析图"""

import os
import json
from typing import List, Dict, Optional
from collections import Counter
from datetime import datetime

import config

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class ReportGenerator:
    """从三元组和论文数据自动生成文献综述草稿和趋势分析图."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_review(self, triplets: List[dict], papers: List[dict] = None,
                        topic: str = "Materials Science") -> str:
        """从三元组数据生成文献综述Markdown草稿.

        Args:
            triplets: [{material, property, value, evidence, paper_id, ...}, ...]
            papers: [{paper_id, title, year, ...}, ...] (可选, 用于引用)
            topic: 综述主题

        Returns:
            Markdown格式的综述文本
        """
        if not triplets:
            return f"# {topic} — Literature Review\n\n*No data available.*\n"

        paper_map = {}
        if papers:
            paper_map = {p.get("paper_id", ""): p for p in papers}

        # 按材料分组
        by_material = {}
        for t in triplets:
            mat = t.get("material", "unknown")
            by_material.setdefault(mat, []).append(t)

        # 统计
        materials = list(by_material.keys())
        properties = list(set(t.get("property", "") for t in triplets))
        paper_ids = list(set(t.get("paper_id", "") for t in triplets))

        lines = []
        lines.append(f"# {topic} — Literature Review (Auto-generated)")
        lines.append(f"")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append(f"")
        lines.append(f"## Overview")
        lines.append(f"")
        lines.append(f"- **Materials studied**: {len(materials)}")
        lines.append(f"- **Properties reported**: {len(properties)}")
        lines.append(f"- **Data points**: {len(triplets)}")
        lines.append(f"- **Papers analyzed**: {len(paper_ids)}")
        lines.append(f"")

        # 材料概览
        lines.append(f"## Materials Summary")
        lines.append(f"")
        mat_counts = Counter(t.get("material", "unknown") for t in triplets)
        for mat, count in mat_counts.most_common(20):
            lines.append(f"- **{mat}**: {count} data points")

        lines.append(f"")

        # 每个材料的详细性能
        for mat in sorted(materials, key=lambda m: len(by_material.get(m, [])), reverse=True)[:15]:
            mat_triplets = by_material[mat]
            lines.append(f"### {mat}")
            lines.append(f"")
            prop_data = {}
            for t in mat_triplets:
                prop = t.get("property", "unknown")
                prop_data.setdefault(prop, []).append(t)

            for prop, entries in sorted(prop_data.items()):
                vals = []
                for e in entries:
                    vn = e.get("value_numeric")
                    if vn is not None:
                        try:
                            vals.append(float(vn))
                        except (ValueError, TypeError):
                            vals.append(e.get("value", ""))
                    else:
                        vals.append(e.get("value", "N/A"))

                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    avg = sum(numeric_vals) / len(numeric_vals)
                    lines.append(f"- **{prop}**: {min(numeric_vals):.3g} ~ {max(numeric_vals):.3g} "
                                 f"(avg {avg:.3g}, n={len(numeric_vals)})")
                else:
                    lines.append(f"- **{prop}**: {vals[:5]}")

            lines.append(f"")

        # 方法学
        synthesis_methods = set()
        microstructures = set()
        for t in triplets:
            ev = t.get("evidence", "")
            # 简单关键词提取合成方法和微观结构
            for kw in ["sol-gel", "CVD", "PVD", "hydrothermal", "ball milling",
                       "electrodeposition", "sputtering", "calcination", "annealing",
                       "quenching", "precipitation", "casting", "forging", "additive manufacturing"]:
                if kw.lower() in ev.lower():
                    synthesis_methods.add(kw)
            for kw in ["grain boundary", "perovskite", "spinel", "amorphous",
                       "crystalline", "nanocrystalline", "polycrystalline", "dendrite",
                       "precipitate", "dislocation", "twin", "phase boundary"]:
                if kw.lower() in ev.lower():
                    microstructures.add(kw)

        if synthesis_methods:
            lines.append(f"## Synthesis Methods")
            lines.append(f"")
            for m in sorted(synthesis_methods):
                lines.append(f"- {m}")
            lines.append(f"")

        if microstructures:
            lines.append(f"## Microstructures Observed")
            lines.append(f"")
            for m in sorted(microstructures):
                lines.append(f"- {m}")
            lines.append(f"")

        # 参考文献
        if paper_map:
            lines.append(f"## References")
            lines.append(f"")
            for i, pid in enumerate(paper_ids[:50], 1):
                p = paper_map.get(pid, {})
                title = p.get("title", pid)
                year = p.get("year", "")
                yr_str = f" ({year})" if year else ""
                lines.append(f"{i}. {title}{yr_str}")

        return "\n".join(lines)

    def generate_trend_chart(self, triplets: List[dict],
                             papers: List[dict] = None) -> Optional[dict]:
        """生成年度趋势图: 各材料研究热度随年份变化.

        Returns:
            Plotly figure JSON dict, or None if plotly unavailable.
        """
        if not HAS_PLOTLY:
            return None

        if not papers:
            return self._figure_to_dict(self._empty_chart("No paper metadata for trend analysis"))

        # 构建 paper_id → year 映射
        pid_to_year = {}
        for p in papers:
            pid = p.get("paper_id", "")
            year = p.get("year", 0)
            if year:
                pid_to_year[pid] = year

        if not pid_to_year:
            return self._figure_to_dict(self._empty_chart("No year data in papers"))

        # 聚合: material × year → count
        mat_year_counts = {}
        for t in triplets:
            mat = t.get("material", "unknown")
            pid = t.get("paper_id", "")
            year = pid_to_year.get(pid)
            if year is None:
                continue
            key = (mat, year)
            mat_year_counts[key] = mat_year_counts.get(key, 0) + 1

        if not mat_year_counts:
            return self._figure_to_dict(self._empty_chart("No year-material matches"))

        # 取top-8材料
        mat_totals = Counter()
        for (mat, year), cnt in mat_year_counts.items():
            mat_totals[mat] += cnt
        top_mats = [m for m, _ in mat_totals.most_common(8)]

        years = sorted(set(y for (_, y) in mat_year_counts))

        fig = go.Figure()
        for mat in top_mats:
            counts = [mat_year_counts.get((mat, y), 0) for y in years]
            fig.add_trace(go.Scatter(
                x=years, y=counts, mode="lines+markers", name=mat,
                hovertemplate=f"{mat}<br>Year: %{{x}}<br>Data points: %{{y}}<extra></extra>",
            ))

        fig.update_layout(
            title="Material Research Trends by Year",
            xaxis_title="Year",
            yaxis_title="Number of Data Points",
            template="plotly_white",
            hovermode="x unified",
        )
        return self._figure_to_dict(fig)

    def generate_material_distribution(self, triplets: List[dict]) -> Optional[dict]:
        """生成材料分布饼图/柱状图."""
        if not HAS_PLOTLY:
            return None

        mat_counts = Counter(t.get("material", "unknown") for t in triplets)
        top = mat_counts.most_common(15)

        fig = go.Figure(data=[
            go.Bar(
                x=[m for m, _ in top],
                y=[c for _, c in top],
                text=[c for _, c in top],
                textposition="outside",
                marker_color="#4ECDC4",
            )
        ])
        fig.update_layout(
            title="Top Materials by Data Points",
            xaxis_title="Material",
            yaxis_title="Count",
            template="plotly_white",
        )
        return self._figure_to_dict(fig)

    def generate_property_distribution(self, triplets: List[dict]) -> Optional[dict]:
        """生成性能参数分布图."""
        if not HAS_PLOTLY:
            return None

        prop_counts = Counter(t.get("property", "unknown") for t in triplets)

        fig = go.Figure(data=[
            go.Pie(
                labels=list(prop_counts.keys()),
                values=list(prop_counts.values()),
                hole=0.4,
                textinfo="label+percent",
            )
        ])
        fig.update_layout(
            title="Property Distribution",
            template="plotly_white",
        )
        return self._figure_to_dict(fig)

    def generate_full_report(self, triplets: List[dict], papers: List[dict] = None,
                             topic: str = "Materials Science") -> dict:
        """生成完整报告: Markdown综述 + 3张图表.

        Returns:
            {"review": str, "trend_chart": dict|None, "mat_dist": dict|None, "prop_dist": dict|None}
        """
        return {
            "review": self.generate_review(triplets, papers, topic),
            "trend_chart": self.generate_trend_chart(triplets, papers),
            "material_distribution": self.generate_material_distribution(triplets),
            "property_distribution": self.generate_property_distribution(triplets),
        }

    def save_report(self, report: dict, filename: str = None) -> str:
        """保存报告到文件.

        Args:
            report: generate_full_report()返回的dict
            filename: 文件名 (不含路径)

        Returns:
            保存路径
        """
        if filename is None:
            filename = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 保存综述Markdown
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report["review"])

        # 保存图表为HTML
        charts_html = os.path.join(self.output_dir, f"{filename}_charts.html")
        chart_keys = [
            ("trend_chart", "Material Research Trends"),
            ("material_distribution", "Material Distribution"),
            ("property_distribution", "Property Distribution"),
        ]
        html_parts = [
            "<html><head><meta charset='utf-8'>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}",
            ".chart{margin:30px 0}</style></head><body>",
            f"<h1>{filename}</h1>",
        ]
        for key, title in chart_keys:
            fig_dict = report.get(key)
            if fig_dict:
                fig_json = json.dumps(fig_dict)
                html_parts.append(
                    f"<div class='chart'><h2>{title}</h2>"
                    f"<div id='{key}'></div></div>"
                    f"<script>Plotly.newPlot('{key}', {fig_json}.data, {fig_json}.layout);</script>"
                )
        html_parts.append("</body></html>")

        with open(charts_html, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        return md_path

    @staticmethod
    def _figure_to_dict(fig) -> dict:
        """将Plotly Figure转为JSON-serializable dict."""
        try:
            return json.loads(fig.to_json())
        except Exception:
            return {}

    @staticmethod
    def _empty_chart(message: str):
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#999"))
        fig.update_layout(template="plotly_white")
        return fig
