"""MEGNet — Materials Graph Network (Chen et al. 2019) 简化实现

MEGNet 核心: 节点更新 ← 边聚合; 边更新 ← 节点+全局状态; 全局状态 ← 节点+边聚合
与CGCNN的区别: MEGNet有全局状态向量u, 边特征也参与消息传递更新

参考: Chen, C. et al. "Graph Networks as a Universal ML Framework for Molecules and Crystals"
      arXiv:1812.05055 (2018)
"""

import numpy as np
from typing import List, Optional, Tuple

MEGNET_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.data import Data, DataLoader
    MEGNET_AVAILABLE = True
except ImportError:
    pass

if MEGNET_AVAILABLE:

    class MEGNetLayer(nn.Module):
        """单层MEGNet更新: 边→节点→全局状态 三向更新."""

        def __init__(self, node_dim: int, edge_dim: int, global_dim: int,
                     hidden_dim: int = 64):
            super().__init__()
            # Edge update: e_ij = MLP(e_ij, v_i, v_j, u)
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, edge_dim),
            )
            # Node update: v_i = MLP(v_i, sum_j(e_ij), u)
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, node_dim),
            )
            # Global update: u = MLP(u, mean(v), mean(e))
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim + node_dim + edge_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, global_dim),
            )
            self.node_dim = node_dim
            self.edge_dim = edge_dim

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            # Edge update
            edge_input = torch.cat([edge_attr, x[row], x[col], u[batch[row]]], dim=-1)
            edge_attr_new = edge_attr + self.edge_mlp(edge_input)
            # Node update
            from torch_geometric.nn import scatter
            edge_aggr = scatter(edge_attr_new, row, dim=0, reduce='mean')
            node_input = torch.cat([x, edge_aggr, u[batch]], dim=-1)
            x_new = x + self.node_mlp(node_input)
            # Global update
            node_mean = global_mean_pool(x_new, batch)
            edge_mean = global_mean_pool(edge_attr_new, batch[edge_index[0]])
            u_input = torch.cat([u, node_mean, edge_mean], dim=-1)
            u_new = u + self.global_mlp(u_input)
            return x_new, edge_attr_new, u_new


    class MEGNet(nn.Module):
        """MEGNet模型: 3层消息传递 + 全局池化 → 回归预测."""

        def __init__(self, node_feat_dim: int = 11, edge_feat_dim: int = 20,
                     global_feat_dim: int = 32, n_layers: int = 3,
                     hidden_dim: int = 64, output_dim: int = 1):
            super().__init__()
            self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
            self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)
            self.global_embed = nn.Linear(global_feat_dim, hidden_dim)
            self.layers = nn.ModuleList([
                MEGNetLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
                for _ in range(n_layers)
            ])
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            u = data.u if hasattr(data, 'u') and data.u is not None else torch.zeros(
                int(batch.max().item() + 1), self.global_embed.in_features,
                device=x.device
            )
            x = F.softplus(self.node_embed(x))
            edge_attr = F.softplus(self.edge_embed(edge_attr))
            u = F.softplus(self.global_embed(u))
            for layer in self.layers:
                x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
            node_pooled = global_mean_pool(x, batch)
            edge_pooled = global_mean_pool(edge_attr, batch[edge_index[0]])
            combined = torch.cat([node_pooled, edge_pooled, u], dim=-1)
            return self.output_mlp(combined).squeeze(-1)


def _gaussian_expand(values: np.ndarray, centers: np.ndarray, width: float = 0.5) -> np.ndarray:
    """高斯展开: 将标量值扩展到RBF空间."""
    diff = values[:, np.newaxis] - centers[np.newaxis, :]
    return np.exp(-0.5 * (diff / width) ** 2)


def build_megnet_graph(formula: str, radius_centers: np.ndarray = None,
                       n_radius_basis: int = 20, cutoff: float = 8.0) -> "Data":
    """从化学式构建MEGNet图.

    节点特征: 元素属性 (11维)
    边特征: 原子间距离的高斯展开 (20维)
    全局特征: 化学式级别的描述符 (32维, 可学习)
    """
    if not MEGNET_AVAILABLE:
        raise ImportError("torch_geometric not installed")

    from pymatgen.core import Composition
    from .elemental_data import ELEMENTAL_PROPERTIES
    from .features_v4 import MagpieFeaturizer

    ELEMENT_PROPERTY_LIST = [
        "atomic_number", "atomic_mass", "electronegativity", "atomic_radius",
        "ionization_energy", "electron_affinity", "group", "period",
        "valence_electrons", "melting_point", "density",
    ]
    _DEFAULT_PROPS = {
        "atomic_number": 0, "atomic_mass": 0, "electronegativity": 0.0,
        "atomic_radius": 0.0, "ionization_energy": 0.0, "electron_affinity": 0.0,
        "group": 0, "period": 0, "valence_electrons": 0,
        "melting_point": 300, "density": 0.0,
    }

    try:
        comp = Composition(formula)
    except Exception:
        comp = Composition("H")

    elements = list(comp.elements)
    n_atoms_per_unit = max(1, int(np.ceil(sum(comp.get_el_amt_dict().values()))))

    nodes = []
    for el in elements:
        props = ELEMENTAL_PROPERTIES.get(str(el), _DEFAULT_PROPS)
        node_feat = np.array([props.get(p, 0.0) for p in ELEMENT_PROPERTY_LIST], dtype=np.float32)
        multi = max(1, int(round(comp.get_el_amt_dict().get(el, 1))))
        for _ in range(multi):
            nodes.append(node_feat)
    if not nodes:
        nodes = [np.zeros(len(ELEMENT_PROPERTY_LIST), dtype=np.float32)]
        n_atoms_per_unit = 1
    nodes = np.array(nodes, dtype=np.float32)
    actual_n = len(nodes)

    if radius_centers is None:
        radius_centers = np.linspace(0, cutoff, n_radius_basis, dtype=np.float32)

    node_radii = np.array([
        ELEMENTAL_PROPERTIES.get(str(el), _DEFAULT_PROPS).get("atomic_radius", 1.0)
        for el in elements for _ in range(max(1, int(round(comp.get_el_amt_dict().get(el, 1)))))
    ], dtype=np.float32)
    if len(node_radii) < actual_n:
        node_radii = np.pad(node_radii, (0, actual_n - len(node_radii)), constant_values=1.0)

    edges_src, edges_dst, edge_attrs = [], [], []
    for i in range(actual_n):
        for j in range(actual_n):
            dist = node_radii[i] + node_radii[j]
            if dist > cutoff:
                continue
            edge_attrs.append(_gaussian_expand(np.array([dist], dtype=np.float32),
                                               radius_centers).flatten())
            edges_src.append(i)
            edges_dst.append(j)

    if not edges_src:
        edges_src, edges_dst = [0, 0], [0, 0]
        edge_attrs = [np.zeros(n_radius_basis, dtype=np.float32) for _ in range(2)]

    edge_attr = np.array(edge_attrs, dtype=np.float32)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    featurizer = MagpieFeaturizer()
    global_vec = featurizer.featurize(formula)[:32].astype(np.float32)
    if len(global_vec) < 32:
        global_vec = np.pad(global_vec, (0, 32 - len(global_vec)))

    data = Data(
        x=torch.tensor(nodes, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        u=torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
        y=torch.tensor([0.0], dtype=torch.float32),
    )
    return data


def build_megnet_batch(formulas: List[str], targets: Optional[np.ndarray] = None
                       ) -> List["Data"]:
    """构建一批MEGNet图."""
    graphs = []
    for i, formula in enumerate(formulas):
        g = build_megnet_graph(formula)
        if targets is not None:
            g.y = torch.tensor([float(targets[i])], dtype=torch.float32)
        graphs.append(g)
    return graphs


def train_megnet(formulas: List[str], targets: np.ndarray,
                 epochs: int = 100, lr: float = 0.001, batch_size: int = 32,
                 test_size: float = 0.2, verbose: bool = True) -> dict:
    """训练MEGNet模型."""
    if not MEGNET_AVAILABLE:
        return {"model": "MEGNet", "error": "torch_geometric not installed"}

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        formulas, targets, test_size=test_size, random_state=42
    )

    train_graphs = build_megnet_batch(X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train), y_train)
    test_graphs = build_megnet_batch(X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test), y_test)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MEGNet(node_feat_dim=11, edge_feat_dim=20, global_feat_dim=32,
                   n_layers=3, hidden_dim=64, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  MEGNet Epoch {epoch+1:3d}/{epochs} - Loss: {avg_loss:.6f}")

    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_preds.extend(pred.cpu().numpy().tolist())
            test_true.extend(batch.y.cpu().numpy().squeeze().tolist())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)

    return {
        "model": "MEGNet",
        "test_r2": float(r2_score(test_true, test_preds)),
        "test_mae": float(mean_absolute_error(test_true, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(test_true, test_preds))),
        "epochs": epochs,
        "trained_model": model,
    }
