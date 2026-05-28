"""图神经网络链接预测 — RGCN + 评分函数, 推荐潜在材料-性能关联"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.nn import RGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


class MaterialKnowledgeGraph:
    """将NetworkX图谱转为PyG HeteroData格式供RGCN训练."""

    def __init__(self):
        self.node_types = ["Material", "Process", "Property", "Microstructure",
                           "Application", "Composition", "Phase"]
        self.edge_types = [
            ("Material", "hasProperty", "Property"),
            ("Material", "usesProcess", "Process"),
            ("Material", "hasMicrostructure", "Microstructure"),
            ("Material", "usedIn", "Application"),
            ("Process", "affectsProperty", "Property"),
            ("Process", "resultsIn", "Microstructure"),
        ]
        self._node_id_to_idx = {}
        self._idx_to_node_id = {}
        self._node_type_map = {}

    def from_networkx(self, graph: nx.MultiDiGraph) -> dict:
        """从NetworkX图构建PyG格式数据.

        Returns:
            {
                "x_dict": {node_type: feature_tensor, ...},  # 各类型节点特征
                "edge_index_dict": {edge_type: (src→dst tensor), ...},
                "node_ids": [str],  # idx→node_id映射
                "num_nodes": int,
            }
        """
        if not HAS_NX:
            raise ImportError("networkx not installed")

        # 收集所有节点并分配索引
        node_list = []
        for nid, attrs in graph.nodes(data=True):
            node_list.append((nid, attrs.get("entity_type", "Material")))
            self._node_type_map[nid] = attrs.get("entity_type", "Material")

        # 按类型分组建索引
        type_to_indices = {t: [] for t in self.node_types}
        for idx, (nid, ntype) in enumerate(node_list):
            mapped_type = self._map_type(ntype)
            if mapped_type in type_to_indices:
                type_to_indices[mapped_type].append((idx, nid))
            self._node_id_to_idx[nid] = idx
            self._idx_to_node_id[idx] = nid

        # 节点特征 (one-hot类型的embedding)
        x_dict = {}
        for ntype, indices in type_to_indices.items():
            if not indices:
                continue
            idxs = [i for i, _ in indices]
            # 简单特征: 每个节点使用one-hot + degree
            features = []
            for idx, nid in indices:
                deg = graph.degree(nid) if nid in graph else 1
                features.append(self._node_feature(nid, ntype, deg, graph))
            x_dict[ntype] = torch.tensor(np.array(features), dtype=torch.float)

        # 边索引
        edge_index_dict = {}
        for src_nid, dst_nid, attrs in graph.edges(data=True):
            if src_nid not in self._node_id_to_idx or dst_nid not in self._node_id_to_idx:
                continue
            src_type = self._map_type(self._node_type_map.get(src_nid, "Material"))
            dst_type = self._map_type(self._node_type_map.get(dst_nid, "Property"))
            pred = attrs.get("predicate", "hasProperty")
            edge_key = (src_type, pred, dst_type)
            if edge_key not in edge_index_dict:
                edge_index_dict[edge_key] = ([], [])
            src_idx = self._node_id_to_idx[src_nid]
            dst_idx = self._node_id_to_idx[dst_nid]
            edge_index_dict[edge_key][0].append(src_idx)
            edge_index_dict[edge_key][1].append(dst_idx)

        # 转为tensor
        edge_index_dict = {
            k: torch.tensor([v[0], v[1]], dtype=torch.long)
            for k, v in edge_index_dict.items() if v[0]
        }

        return {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "node_ids": [self._idx_to_node_id[i] for i in range(len(node_list))],
            "num_nodes": len(node_list),
            "material_indices": [idx for idx, (_, ntype) in enumerate(node_list)
                                 if self._map_type(ntype) == "Material"],
            "property_indices": [idx for idx, (_, ntype) in enumerate(node_list)
                                  if self._map_type(ntype) == "Property"],
        }

    def _node_feature(self, nid: str, ntype: str, degree: int,
                      graph: nx.MultiDiGraph) -> np.ndarray:
        """生成节点初始特征向量 (16维)."""
        type_vec = np.zeros(len(self.node_types))
        if ntype in self.node_types:
            type_vec[self.node_types.index(ntype)] = 1.0
        name_hash = int(hashlib.md5(nid.encode()).hexdigest()[:8], 16) / (16**8)
        feat = np.concatenate([
            type_vec,
            [np.log1p(degree) / 10.0, name_hash, 1.0],
            np.zeros(16 - len(self.node_types) - 3),
        ])
        return feat[:16]

    def _map_type(self, etype: str) -> str:
        mapping = {
            "material": "Material", "property": "Property",
            "processing_method": "Process", "synthesis_method": "Process",
            "crystal_structure": "Microstructure", "microstructure": "Microstructure",
            "application": "Application", "property_value": "Property",
            "composition": "Composition", "phase": "Phase",
            "Process": "Process", "Material": "Material", "Property": "Property",
            "Microstructure": "Microstructure", "Application": "Application",
            "Composition": "Composition", "Phase": "Phase",
        }
        return mapping.get(etype, "Material")


class RGCNLinkPredictor(nn.Module):
    """RGCN编码器 + DistMult评分函数, 用于材料-性能链接预测."""

    def __init__(self, num_nodes_dict: Dict[str, int], hidden_dim: int = 64,
                 num_relations: int = 6, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        if not HAS_TORCH or not HAS_PYG:
            raise ImportError("torch/torch_geometric not installed")

        self.hidden_dim = hidden_dim
        self.embeddings = nn.ModuleDict()

        # 可学习节点Embedding (各类型独立)
        for ntype, num in num_nodes_dict.items():
            self.embeddings[ntype] = nn.Embedding(num, hidden_dim)

        # RGCN卷积层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.dropout = nn.Dropout(dropout)
        self.relation_embed = nn.Embedding(num_relations, hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                edge_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播, 返回各类型节点的最终embedding."""
        # 初始节点特征
        node_emb = {}
        for ntype, emb in self.embeddings.items():
            if ntype in x_dict:
                node_emb[ntype] = self.dropout(emb.weight[:x_dict[ntype].size(0)])

        # RGCN消息传递
        for conv in self.convs:
            # 简化: 对所有边做消息传递
            new_emb = {}
            for (src_t, rel, dst_t), ei in edge_index_dict.items():
                if (src_t in node_emb and dst_t in node_emb and
                    ei.size(0) > 0 and ei.size(1) > 0):
                    # 构建同构图子集
                    all_emb = {t: e for t, e in node_emb.items()}
                    try:
                        updated = conv(all_emb, ei)
                        for t, e in updated.items():
                            if t not in new_emb:
                                new_emb[t] = e
                    except Exception:
                        pass
            for t, e in new_emb.items():
                node_emb[t] = F.relu(e)
            node_emb = {t: self.dropout(e) for t, e in node_emb.items()}

        return node_emb

    def score(self, mat_emb: torch.Tensor, prop_emb: torch.Tensor,
              rel_idx: int = 0) -> torch.Tensor:
        """DistMult评分: mat^T * diag(R_r) * prop."""
        r_emb = self.relation_embed(torch.tensor(rel_idx))
        return torch.sum(mat_emb * r_emb * prop_emb, dim=-1)

    def predict_links(self, material_indices: List[int],
                      property_indices: List[int],
                      node_emb: Dict[str, torch.Tensor]) -> List[dict]:
        """预测所有 (材料, 性能) 对的评分, 返回top推荐."""
        mat_emb = node_emb.get("Material")
        prop_emb = node_emb.get("Property")
        if mat_emb is None or prop_emb is None:
            return []

        scores = []
        with torch.no_grad():
            for mi in material_indices:
                if mi >= mat_emb.size(0):
                    continue
                m = mat_emb[mi:mi + 1]
                for pi in property_indices:
                    if pi >= prop_emb.size(0):
                        continue
                    p = prop_emb[pi:pi + 1]
                    s = self.score(m, p).item()
                    scores.append((mi, pi, s))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


def train_link_prediction(graph: nx.MultiDiGraph, epochs: int = 100,
                          lr: float = 0.01, hidden_dim: int = 64,
                          val_ratio: float = 0.1) -> Tuple[RGCNLinkPredictor, dict]:
    """训练RGCN链接预测模型.

    Args:
        graph: NetworkX图谱
        epochs: 训练轮数
        lr: 学习率
        hidden_dim: 隐层维度
        val_ratio: 验证集比例

    Returns:
        (model, metrics_dict)
    """
    if not HAS_TORCH or not HAS_PYG:
        raise ImportError("torch/torch_geometric not installed. Run: pip install torch torch-geometric")

    # 构建图数据
    builder = MaterialKnowledgeGraph()
    data = builder.from_networkx(graph)

    mat_indices = data.get("material_indices", [])
    prop_indices = data.get("property_indices", [])

    if not mat_indices or not prop_indices:
        raise ValueError("Graph does not contain enough Material/Property nodes")

    # 提取正负样本
    existing_pairs = set()
    for (src_t, rel, dst_t), ei in data["edge_index_dict"].items():
        if src_t == "Material" and dst_t == "Property":
            for j in range(ei.size(1)):
                existing_pairs.add((ei[0, j].item(), ei[1, j].item()))

    pos_samples = list(existing_pairs)

    # 生成负样本 (随机不存在的材料-性能对)
    neg_samples = []
    neg_size = max(len(pos_samples), 100)
    while len(neg_samples) < neg_size:
        mi = np.random.choice(mat_indices)
        pi = np.random.choice(prop_indices)
        if (mi, pi) not in existing_pairs:
            neg_samples.append((mi, pi))

    # 划分训练/验证集
    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)
    split = int(len(pos_samples) * (1 - val_ratio))
    train_pos = pos_samples[:split]
    val_pos = pos_samples[split:]
    val_neg = neg_samples[len(pos_samples):len(pos_samples) + len(val_pos)]

    # 模型初始化
    num_nodes_dict = {t: e.size(0) for t, e in data["x_dict"].items()}
    model = RGCNLinkPredictor(
        num_nodes_dict=num_nodes_dict,
        hidden_dim=hidden_dim,
        num_relations=len(builder.edge_types),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    history = {"train_loss": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        node_emb = model(data["x_dict"], data["edge_index_dict"],
                         torch.tensor([0]))

        # BPR loss: 正样本评分应高于负样本
        if train_pos and neg_samples:
            batch_pos = train_pos[:len(neg_samples)]
            pos_scores = []
            neg_scores = []
            for (mi, pi), (mj, pj) in zip(batch_pos, neg_samples):
                m_emb = node_emb["Material"][mi:mi + 1]
                p_emb = node_emb["Property"][pi:pi + 1]
                n_m_emb = node_emb["Material"][mj:mj + 1]
                n_p_emb = node_emb["Property"][pj:pj + 1]
                pos_scores.append(model.score(m_emb, p_emb))
                neg_scores.append(model.score(n_m_emb, n_p_emb))

            pos_tensor = torch.stack(pos_scores).squeeze()
            neg_tensor = torch.stack(neg_scores).squeeze()
            loss = -torch.mean(F.logsigmoid(pos_tensor - neg_tensor))
            loss.backward()
            optimizer.step()
            history["train_loss"].append(loss.item())

        # 验证
        if (epoch + 1) % 20 == 0 and val_pos and val_neg:
            model.eval()
            with torch.no_grad():
                node_emb = model(data["x_dict"], data["edge_index_dict"],
                                 torch.tensor([0]))
                val_pos_scores = []
                val_neg_scores = []
                for mi, pi in val_pos:
                    s = model.score(node_emb["Material"][mi:mi + 1],
                                    node_emb["Property"][pi:pi + 1]).item()
                    val_pos_scores.append(s)
                for mi, pi in val_neg:
                    s = model.score(node_emb["Material"][mi:mi + 1],
                                    node_emb["Property"][pi:pi + 1]).item()
                    val_neg_scores.append(s)

                auc = _roc_auc(val_pos_scores, val_neg_scores)
                history["val_auc"].append(auc)

    # 最终预测
    model.eval()
    with torch.no_grad():
        final_emb = model(data["x_dict"], data["edge_index_dict"],
                          torch.tensor([0]))
        all_preds = model.predict_links(mat_indices, prop_indices, final_emb)

    # 筛除已有链接
    new_predictions = [(mi, pi, s) for mi, pi, s in all_preds
                       if (mi, pi) not in existing_pairs]

    results = {
        "model": model,
        "history": history,
        "predictions": new_predictions[:50],  # top-50
        "node_ids": data["node_ids"],
        "material_count": len(mat_indices),
        "property_count": len(prop_indices),
        "existing_links": len(existing_pairs),
    }
    return model, results


def _roc_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """简单AUC计算."""
    scores = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    scores.sort(key=lambda x: x[0], reverse=True)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = 0.0
    tp = 0
    for _, label in scores:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)
