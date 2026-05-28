"""知识图谱可视化 v2 — pyvis交互图 + Plotly统计图"""

import os
import tempfile
import networkx as nx

from .ontology import ENTITY_COLORS, RELATION_LABELS_ZH, ENTITY_LABELS_ZH


class GraphVisualizer:
    """图谱可视化, 支持pyvis交互式和Plotly统计图."""

    # ---- pyvis ----

    @staticmethod
    def to_pyvis(graph: nx.MultiDiGraph, height: str = "600px",
                  physics: bool = True) -> "Network":
        from pyvis.network import Network
        net = Network(height=height, width="100%", directed=True,
                      bgcolor="#ffffff", font_color="#333333")
        if physics:
            net.set_options("""
            {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -2000,
                  "springLength": 150,
                  "springConstant": 0.04
                },
                "minVelocity": 0.75
              }
            }
            """)

        for node_id, attrs in graph.nodes(data=True):
            etype = attrs.get("entity_type", "")
            color = ENTITY_COLORS.get(etype, "#999999")
            name = attrs.get("name", node_id)
            degree = graph.degree(node_id)
            size = min(max(degree * 5 + 15, 15), 60)
            zh_label = ENTITY_LABELS_ZH.get(etype, etype)
            title = (f"<b>{name}</b><br>"
                     f"Type: {zh_label} ({etype})<br>"
                     f"Degree: {degree}")
            net.add_node(node_id, label=name, color=color, size=size, title=title,
                         font={"size": 12})

        for u, v, attrs in graph.edges(data=True):
            predicate = attrs.get("predicate", "")
            label = RELATION_LABELS_ZH.get(predicate, predicate)
            evidence = attrs.get("evidence", "")
            title = evidence[:200] if evidence else label
            net.add_edge(u, v, label=label, title=title, arrows="to",
                         color={"color": "#888", "opacity": 0.7})

        return net

    @staticmethod
    def to_pyvis_html(graph: nx.MultiDiGraph, height: str = "600px") -> str:
        net = GraphVisualizer.to_pyvis(graph, height)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w+",
                                          encoding="utf-8") as f:
            net.save_graph(f.name)
            html_path = f.name
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        try:
            os.unlink(html_path)
        except Exception:
            pass
        return html_content

    @staticmethod
    def to_pyvis_file(graph: nx.MultiDiGraph, output_path: str, height: str = "600px"):
        net = GraphVisualizer.to_pyvis(graph, height)
        net.save_graph(output_path)
        return output_path

    # ---- Plotly 统计图 ----

    @staticmethod
    def entity_pie_chart(entity_counts: dict) -> "go.Figure":
        """实体类型饼图."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        labels = []
        values = []
        colors = []
        for etype, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            labels.append(ENTITY_LABELS_ZH.get(etype, etype))
            values.append(count)
            colors.append(ENTITY_COLORS.get(etype, "#999"))

        fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                      marker=dict(colors=colors),
                                      hole=0.4, textinfo="label+value")])
        fig.update_layout(title="Entity Type Distribution", template="plotly_white")
        return fig

    @staticmethod
    def relation_bar_chart(relation_counts: dict) -> "go.Figure":
        """关系类型柱状图."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        labels = []
        values = []
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
            labels.append(RELATION_LABELS_ZH.get(rel, rel))
            values.append(count)

        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#45B7D1",
                                      text=values, textposition="outside")])
        fig.update_layout(title="Relationship Type Distribution",
                          xaxis_title="Relation", yaxis_title="Count",
                          template="plotly_white")
        return fig

    @staticmethod
    def degree_histogram(graph: nx.MultiDiGraph) -> "go.Figure":
        """节点度数直方图."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        degrees = [d for _, d in graph.degree()]
        fig = go.Figure(data=[go.Histogram(x=degrees, nbinsx=20,
                                            marker_color="#4ECDC4")])
        fig.update_layout(title="Node Degree Distribution",
                          xaxis_title="Degree", yaxis_title="Count",
                          template="plotly_white")
        return fig
