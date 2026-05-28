"""知识图谱存储 — JSON序列化/反序列化"""

import json
import os
import networkx as nx
from networkx.readwrite import json_graph


class GraphStore:
    @staticmethod
    def save(graph: nx.MultiDiGraph, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = json_graph.node_link_data(graph)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(filepath: str) -> nx.MultiDiGraph:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)
