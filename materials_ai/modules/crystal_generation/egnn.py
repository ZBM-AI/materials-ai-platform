"""E(n) Equivariant Graph Neural Network — 扩散模型去噪骨干网络

参考: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)
用于CDVAE/DiffCSP中坐标+特征的等变消息传递.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EGNNLayer(nn.Module):
    """单层E(n) Equivariant GNN.

    消息传递: m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)
    坐标更新: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
    特征更新: h_i' = phi_h(h_i, sum_j m_ij)
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0, residual: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual

        input_dim = hidden_dim * 2 + 1 + edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, hidden_dim) 节点特征
            x: (N, 3) 坐标 (分数坐标或笛卡尔坐标)
            edge_index: (2, E) 边索引
            edge_attr: (E, edge_dim) 可选的边特征
        Returns:
            h_new: (N, hidden_dim)
            x_new: (N, 3)
        """
        row, col = edge_index[0], edge_index[1]

        # 相对位置 + 平方距离
        rel_pos = x[row] - x[col]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        # 边消息
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        # 坐标更新 (等变)
        coord_weight = self.coord_mlp(m_ij)
        x_aggr = torch.zeros_like(x)
        x_aggr.index_add_(0, row, rel_pos * coord_weight)
        x_new = x + x_aggr

        # 特征更新
        aggr = torch.zeros(x.shape[0], self.hidden_dim, device=h.device)
        aggr.index_add_(0, row, m_ij)
        h_new = self.node_mlp(torch.cat([h, aggr], dim=-1))
        h_new = self.layer_norm(h_new)

        if self.residual:
            h_new = h_new + h

        return h_new, x_new


class EGNN(nn.Module):
    """多层EGNN — 用于扩散模型去噪网络.

    对晶格参数和原子种类使用单独的MLP分支 (标量更新),
    对分数坐标使用EGNN等变层 (向量更新).
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 num_atom_types: int = 100, atom_embed_dim: int = 64,
                 lattice_dim: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.atom_embed = nn.Embedding(num_atom_types + 1, atom_embed_dim)

        self.node_in = nn.Sequential(
            nn.Linear(atom_embed_dim + hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, atom_embed_dim),
        )

        # 晶格去噪网络
        self.lattice_encoder = nn.Sequential(
            nn.Linear(lattice_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lattice_decoder = nn.Sequential(
            nn.Linear(hidden_dim + atom_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, lattice_dim),
        )

        # 原子类型预测
        self.atom_classifier = nn.Sequential(
            nn.Linear(atom_embed_dim, atom_embed_dim),
            nn.SiLU(),
            nn.Linear(atom_embed_dim, num_atom_types + 1),
        )

    def forward(self, atom_types: torch.Tensor, frac_coords: torch.Tensor,
                lattice_params: torch.Tensor, t: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                condition: Optional[torch.Tensor] = None):
        """
        Args:
            atom_types: (batch_atoms,) 原子类型(加噪版本 — 连续嵌入)
            frac_coords: (batch_atoms, 3) 分数坐标 (加噪)
            lattice_params: (batch, 6) 晶格参数 [a,b,c,alpha,beta,gamma]
            t: (batch,) or (batch_atoms,) 扩散时间步嵌入
            edge_index: (2, E) 图边
            edge_attr: (E, edge_dim) 边特征 (距离)
            condition: (batch, cond_dim) 可选的组成/空间群条件
        Returns:
            eps_coords: (batch_atoms, 3) 预测坐标噪声
            eps_lattice: (batch, 6) 预测晶格噪声
            atom_logits: (batch_atoms, num_types+1) 原子类型预测
        """
        batch_size = lattice_params.shape[0]
        num_atoms = atom_types.shape[0]
        atoms_per_batch = num_atoms // max(batch_size, 1)

        # 原子嵌入 (连续版本 — soft embedding from classifier logits)
        if atom_types.dim() == 1 and atom_types.dtype == torch.long:
            h_atom = self.atom_embed(atom_types)
        else:
            h_atom = atom_types

        # 时间嵌入
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1 and t.shape[0] == batch_size:
            t_per_atom = t.repeat_interleave(atoms_per_batch).unsqueeze(-1)
        else:
            t_per_atom = t.unsqueeze(-1)

        t_embed = self._time_embedding(t_per_atom, self.hidden_dim)

        # 初始节点特征
        h = torch.cat([h_atom, t_embed, t_per_atom], dim=-1)
        h = self.node_in(h)

        # EGNN 消息传递
        x = frac_coords.clone()
        for layer in self.egnn_layers:
            h, x = layer(h, x, edge_index, edge_attr)

        # 输出
        atom_embed_out = self.node_out(h)
        eps_coords = x - frac_coords
        atom_logits = self.atom_classifier(atom_embed_out)

        # 晶格去噪
        lattice_h = self.lattice_encoder(lattice_params)
        atom_pool = atom_embed_out.view(batch_size, atoms_per_batch, -1).mean(dim=1)
        lattice_h = torch.cat([lattice_h, atom_pool], dim=-1)
        eps_lattice = self.lattice_decoder(lattice_h)

        return eps_coords, eps_lattice, atom_logits

    @staticmethod
    def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """正弦时间步嵌入."""
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device) * (
                np.log(10000) / half
            )
        )
        args = t.float() * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def build_crystal_graph(frac_coords: torch.Tensor, lattice: torch.Tensor,
                        cutoff: float = 8.0, max_neighbors: int = 24):
    """构建晶体图 — 基于截止距离的多边连接.

    Args:
        frac_coords: (N, 3) 分数坐标
        lattice: (3, 3) 晶格矩阵
        cutoff: 截止半径 (Angstrom)
        max_neighbors: 每个原子的最大邻居数
    Returns:
        edge_index: (2, E) 边
        edge_vec: (E, 3) 边向量 (笛卡尔)
        edge_dist: (E,) 边距离
    """
    N = frac_coords.shape[0]
    cart_coords = frac_coords @ lattice.T if lattice.dim() == 2 else frac_coords

    if N <= 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=frac_coords.device)
        edge_vec = torch.zeros(1, 3, device=frac_coords.device)
        edge_dist = torch.zeros(1, device=frac_coords.device)
        return edge_index, edge_vec, edge_dist

    rel = cart_coords.unsqueeze(0) - cart_coords.unsqueeze(1)  # (N, N, 3)
    dist = rel.norm(dim=-1)  # (N, N)

    mask = (dist < cutoff) & (dist > 0.01)
    src, dst = torch.where(mask)

    if src.shape[0] == 0:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=frac_coords.device)
        edge_vec = torch.tensor([[0., 0., 0.], [0., 0., 0.]], device=frac_coords.device)
        edge_dist = torch.tensor([0., 0.], device=frac_coords.device)
        return edge_index, edge_vec, edge_dist

    edge_vec = rel[src, dst]
    edge_dist = dist[src, dst]
    edge_index = torch.stack([src, dst], dim=0)

    if edge_index.shape[1] > N * max_neighbors:
        _, topk_idx = torch.topk(-edge_dist, k=N * max_neighbors)
        edge_index = edge_index[:, topk_idx]
        edge_vec = edge_vec[topk_idx]
        edge_dist = edge_dist[topk_idx]

    return edge_index, edge_vec, edge_dist
