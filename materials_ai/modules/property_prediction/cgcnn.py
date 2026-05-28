"""简化CGCNN — 基于PyTorch Geometric的晶体图卷积网络, 用于带隙预测

参考: Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018)
简化: 无需CIF文件, 从化学式自动构建代理晶体图
"""

import numpy as np
from typing import List, Tuple, Optional
from pymatgen.core import Composition, Structure
from pymatgen.core.lattice import Lattice
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from .elemental_data import ELEMENTAL_PROPERTIES

ELEM_LIST = sorted(
    [k for k in ELEMENTAL_PROPERTIES if len(k) <= 2 and k[0].isupper()],
    key=lambda x: ELEMENTAL_PROPERTIES[x].get("atomic_number", 0)
)
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEM_LIST)}
MAX_ATOMIC_NUM = max(ELEMENTAL_PROPERTIES[e].get("atomic_number", 0) for e in ELEM_LIST)

PROPERTY_KEYS = [
    "electronegativity", "atomic_radius", "ionization_energy",
    "electron_affinity", "valence_electrons", "atomic_mass",
]
PROPERTY_KEYS_NORMALIZED = {  # rough max values for normalization
    "electronegativity": 4.0,
    "atomic_radius": 300.0,
    "ionization_energy": 25.0,
    "electron_affinity": 4.0,
    "valence_electrons": 12.0,
    "atomic_mass": 300.0,
}


def get_node_features(atomic_number: int) -> np.ndarray:
    """从原子序数构建节点特征: one-hot(前10) + 归一化属性"""
    features = []
    one_hot = np.zeros(min(len(ELEM_LIST), 100))
    el_str = None
    for sym, idx in ELEM_TO_IDX.items():
        if ELEMENTAL_PROPERTIES.get(sym, {}).get("atomic_number", 0) == atomic_number:
            one_hot[idx] = 1.0
            el_str = sym
            break
    features.extend(one_hot.tolist())
    if el_str:
        props = ELEMENTAL_PROPERTIES.get(el_str, {})
        for key in PROPERTY_KEYS:
            val = props.get(key, 0.0)
            norm = PROPERTY_KEYS_NORMALIZED.get(key, 1.0)
            features.append(val / max(norm, 0.01))
    else:
        features.extend([0.0] * len(PROPERTY_KEYS))
    return np.array(features, dtype=np.float32)


def composition_to_structure(formula: str, scaling_factor: float = 2.5) -> Structure:
    """从化学式构建代理晶体结构 (简单立方近似).

    根据加权平均原子半径估算晶格参数, 按化学计量比填充原子.
    """
    comp = Composition(formula)
    elements = list(comp.elements)
    fractions = list(comp.fractional_composition.values())
    total_atoms = int(np.ceil(sum(fractions) * 8))
    min_atoms = max(4, total_atoms)
    radii = []
    for el in elements:
        r = ELEMENTAL_PROPERTIES.get(str(el), {}).get("atomic_radius", 150)
        radii.append(r)
    avg_radius = float(np.average(radii, weights=fractions)) if fractions else 150
    volume_per_atom = (4.0 / 3.0) * np.pi * (avg_radius * 1e-12) ** 3
    lattice_const = (min_atoms * volume_per_atom) ** (1.0 / 3.0) * 1e12 * scaling_factor
    lattice = Lattice.cubic(lattice_const)
    atom_counts = []
    for frac in fractions:
        count = max(1, int(round(frac * min_atoms / sum(fractions))))
        atom_counts.append(count)
    while sum(atom_counts) > 100:
        atom_counts = [max(1, c - 1) for c in atom_counts]
    while sum(atom_counts) < 4:
        for i in range(len(atom_counts)):
            atom_counts[i] += 1
            if sum(atom_counts) >= 4:
                break
    species = []
    for el, count in zip(elements, atom_counts):
        species.extend([str(el)] * count)
    n_atoms = len(species)
    frac_coords = np.random.RandomState(42).uniform(0, 1, (n_atoms, 3))
    return Structure(lattice, species, frac_coords)


def build_crystal_graph(structure: Structure, radius: float = 8.0,
                        max_neighbors: int = 12) -> Tuple[torch.Tensor, ...]:
    """从pymatgen Structure构建PyG图数据.

    节点: 原子 (one-hot + 属性)
    边: 距离 < radius的原子对, 边特征用高斯展开
    """
    atomic_numbers = [site.specie.number for site in structure.sites]
    node_features = np.array([get_node_features(an) for an in atomic_numbers])
    coords = np.array([site.coords for site in structure.sites])
    lattice_matrix = structure.lattice.matrix
    n_atoms = len(atomic_numbers)
    edge_src, edge_dst, edge_vectors = [], [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            diff = coords[j] - coords[i]
            for a in range(-1, 2):
                for b in range(-1, 2):
                    for c in range(-1, 2):
                        offset = (a * lattice_matrix[0] + b * lattice_matrix[1] +
                                  c * lattice_matrix[2])
                        dist = np.linalg.norm(diff + offset)
                        if dist < radius:
                            edge_src.append(i)
                            edge_dst.append(j)
                            edge_vectors.append(diff + offset)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    if len(edge_src) > max_neighbors * n_atoms:
        indices = list(range(len(edge_src)))
        np.random.RandomState(42).shuffle(indices)
        keep = indices[:max_neighbors * n_atoms]
        edge_src = [edge_src[k] for k in keep]
        edge_dst = [edge_dst[k] for k in keep]
        edge_vectors = [edge_vectors[k] for k in keep]
    if len(edge_src) == 0:
        edge_src, edge_dst = [0, 1], [1, 0]
        edge_vectors = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    distances = np.array([np.linalg.norm(v) for v in edge_vectors])
    edge_attr = gaussian_expansion(distances, dmin=0.0, dmax=radius, steps=40)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return x, edge_index, edge_attr


def gaussian_expansion(distances: np.ndarray, dmin: float = 0.0,
                       dmax: float = 8.0, steps: int = 40) -> np.ndarray:
    """将距离用高斯函数展开为边特征"""
    centers = np.linspace(dmin, dmax, steps)
    gamma = 1.0 / ((centers[1] - centers[0]) * 0.5) ** 2 if steps > 1 else 1.0
    expanded = np.exp(-gamma * (distances[:, np.newaxis] - centers[np.newaxis, :]) ** 2)
    return expanded.astype(np.float32)


class CGCNNConv(nn.Module):
    """CGCNN卷积层: 节点更新 = sigma(W1*node + W2*sum(node_j * edge_attr_ij))"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.edge_fc = nn.Linear(edge_dim, node_dim)
        self.node_fc = nn.Linear(node_dim, hidden_dim)
        self.neighbor_fc = nn.Linear(node_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        edge_weight = torch.sigmoid(self.edge_fc(edge_attr))
        neighbor_msg = x[src] * edge_weight
        agg = torch.zeros_like(x)
        agg = agg.index_add(0, dst, neighbor_msg)
        out = F.softplus(self.bn1(self.node_fc(x)) + self.bn2(self.neighbor_fc(agg)))
        return out


class CGCNN(nn.Module):
    """简化CGCNN模型: 3层卷积 + 全局平均池化 + MLP输出"""

    def __init__(self, node_dim: int, edge_dim: int = 40,
                 hidden_dim: int = 128, num_layers: int = 3,
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGCNNConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        x = F.softplus(self.node_embed(x))
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = x + residual
            x = self.dropout(x)
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        return self.output(x).squeeze(-1)


class CrystalGraphDataset(Dataset):
    """从化学式列表构建晶体图数据集.

    支持三种模式:
      1. structures_dict: 提供 {formula: pymatgen Structure} 字典, 使用真实结构
      2. structures_list: 提供与formulas一一对应的pymatgen Structure列表
      3. 默认: 从composition_to_structure()构建代理结构 (原有行为)
    """

    def __init__(self, formulas: List[str], targets: np.ndarray,
                 radius: float = 8.0,
                 structures_dict: Optional[dict] = None,
                 structures_list: Optional[list] = None):
        self.formulas = formulas
        self.targets = targets
        self.radius = radius
        self.structures_dict = structures_dict
        self.structures_list = structures_list
        self._graphs = None

    def _get_structure(self, formula: str, idx: int) -> Structure:
        if self.structures_dict and formula in self.structures_dict:
            return self.structures_dict[formula]
        if self.structures_list and idx < len(self.structures_list):
            s = self.structures_list[idx]
            if s is not None:
                return s
        return composition_to_structure(formula)

    def _build_all(self):
        if self._graphs is not None:
            return
        self._graphs = []
        for i, formula in enumerate(self.formulas):
            try:
                struct = self._get_structure(formula, i)
                x, ei, ea = build_crystal_graph(struct, radius=self.radius)
                self._graphs.append((x, ei, ea))
            except Exception:
                x = torch.zeros((1, get_node_features(1).shape[0]))
                ei = torch.zeros((2, 0), dtype=torch.long)
                ea = torch.zeros((0, 40))
                self._graphs.append((x, ei, ea))

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if self._graphs is None:
            self._build_all()
        x, ei, ea = self._graphs[idx]
        from torch_geometric.data import Data
        return Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor(self.targets[idx], dtype=torch.float))


def load_structures_from_matbench(matbench_json_path: str, n_samples: int = None
                                  ) -> Tuple[List[str], np.ndarray, list]:
    """从MatBench JSON文件加载真实晶体结构.

    返回: (formulas, targets, structures) 其中structures是pymatgen Structure列表.
    """
    import json
    import gzip

    if matbench_json_path.endswith('.gz'):
        with gzip.open(matbench_json_path, 'rb') as fh:
            data = json.loads(fh.read().decode('utf-8'))
    else:
        with open(matbench_json_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

    formulas, targets, structures = [], [], []
    samples = data.get("data", []) if isinstance(data, dict) else data
    if n_samples:
        import random
        random.Random(42).shuffle(samples)
        samples = samples[:n_samples]

    for row in samples:
        try:
            struct_dict = row[0]
            target = float(row[1])
            if isinstance(struct_dict, dict) and "@module" in struct_dict:
                struct = Structure.from_dict(struct_dict)
                formula = struct.composition.reduced_formula
                formulas.append(formula)
                targets.append(target)
                structures.append(struct)
        except Exception:
            continue

    return formulas, np.array(targets), structures


def collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def train_cgcnn(formulas: List[str], targets: np.ndarray,
                epochs: int = 100, lr: float = 0.001,
                test_size: float = 0.2, radius: float = 8.0,
                structures_dict: Optional[dict] = None,
                structures_list: Optional[list] = None) -> dict:
    """训练CGCNN模型并返回评估指标.

    structures_dict: {formula: Structure} 字典, 提供真实晶体结构 (优先使用)
    structures_list: 与formulas一一对应的Structure列表
    """
    from torch_geometric.loader import DataLoader as GDataLoader

    if structures_dict or structures_list:
        str_list = [None] * len(formulas)
        if structures_dict:
            for i, f in enumerate(formulas):
                str_list[i] = structures_dict.get(f)
        elif structures_list:
            str_list = structures_list
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            formulas, targets, str_list, test_size=test_size, random_state=42
        )
        train_dataset = CrystalGraphDataset(X_train, y_train, radius=radius, structures_list=s_train)
        test_dataset = CrystalGraphDataset(X_test, y_test, radius=radius, structures_list=s_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            formulas, targets, test_size=test_size, random_state=42
        )
        train_dataset = CrystalGraphDataset(X_train, y_train, radius=radius)
        test_dataset = CrystalGraphDataset(X_test, y_test, radius=radius)
    train_loader = GDataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    sample_graph = train_dataset[0]
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1) if sample_graph.edge_attr.numel() > 0 else 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGCNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step(total_loss / len(train_dataset))

    model.eval()
    with torch.no_grad():
        test_loader = GDataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
        all_preds, all_y = [], []
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_y.extend(batch.y.cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_y = np.array(all_y)

    return {
        "test_r2": float(r2_score(all_y, all_preds)),
        "test_mae": float(mean_absolute_error(all_y, all_preds)),
        "test_rmse": float(np.sqrt(np.mean((all_y - all_preds) ** 2))),
    }
