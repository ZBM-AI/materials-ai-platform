"""扩散模型 — 前向加噪 + 反向去噪 (CDVAE/DiffCSP 风格)

晶体扩散:
- 晶格参数: 连续高斯扩散 (6维: a,b,c, α,β,γ)
- 分数坐标: 连续高斯扩散 (N×3, 考虑周期性边界)
- 原子类型: 离散扩散 (D3PM / 掩码扩散)

条件: 目标组成(元素及其化学计量比) + 空间群
"""

import numpy as np
from typing import Dict, Optional, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .egnn import EGNN, build_crystal_graph


class NoiseScheduler:
    """余弦噪声调度 (Nichol & Dhariwal, 2021)."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "cosine"):
        self.num_timesteps = num_timesteps

        if schedule == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            t = np.linspace(0, num_timesteps, steps)
            alphas_cumprod = np.cos((t / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向加噪: x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε."""
        t_idx = t.long()
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].to(x.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].to(x.device)

        noise = torch.randn_like(x)
        while sqrt_alpha.dim() < x.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x + sqrt_one_minus * noise
        return x_t, noise

    def get_coeff(self, t: torch.Tensor):
        """获取去噪损失系数."""
        t_idx = t.long()
        alpha = self.alphas[t_idx]
        alpha_cumprod = self.alphas_cumprod[t_idx]
        beta = self.betas[t_idx]
        sigma = torch.sqrt(beta * (1.0 - alpha_cumprod) / (1.0 - alpha + 1e-10))
        return alpha, sigma


class CrystalDiffusion(nn.Module):
    """晶体结构扩散模型.

    对晶格(连续) + 坐标(连续周期) + 原子类型(离散) 联合扩散.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 num_atom_types: int = 84, atom_embed_dim: int = 64,
                 num_timesteps: int = 1000, lattice_dim: int = 6,
                 coord_noise_scale: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.atom_embed_dim = atom_embed_dim
        self.num_timesteps = num_timesteps
        self.coord_noise_scale = coord_noise_scale

        self.noise_scheduler = NoiseScheduler(num_timesteps)

        self.atom_embed = nn.Embedding(num_atom_types + 2, atom_embed_dim)

        self.denoiser = EGNN(
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_atom_types=num_atom_types + 1,
            atom_embed_dim=atom_embed_dim,
            lattice_dim=lattice_dim,
        )

        # 组成编码器 (将目标元素比例转为条件)
        self.composition_encoder = nn.Sequential(
            nn.Linear(num_atom_types, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 空间群嵌入
        self.space_group_embed = nn.Embedding(231, hidden_dim // 2)

    def forward(self, batch: Dict) -> Dict:
        """单步去噪训练.

        Args:
            batch: {
                "lattice": (B, 3, 3),
                "frac_coords": (total_atoms, 3),
                "atom_types": (total_atoms,),
                "batch_idx": (total_atoms,) batch归属,
                "composition": (B, num_atom_types) 目标组成向量,
                "space_group": (B,) 空间群编号,
            }
        Returns:
            {"loss": total_loss, "loss_lattice": ..., "loss_coord": ..., "loss_atom": ...}
        """
        device = batch["lattice"].device
        B = batch["lattice"].shape[0]
        total_atoms = batch["frac_coords"].shape[0]

        lattice_params = self._lattice_to_params(batch["lattice"])  # (B, 6)

        # 采样时间步
        t_coord = torch.randint(0, self.num_timesteps, (total_atoms,), device=device)
        t_lattice = torch.randint(0, self.num_timesteps, (B,), device=device)

        # 1) 分数坐标加噪
        x0 = batch["frac_coords"]
        x_t, eps_coord = self.noise_scheduler.add_noise(x0, t_coord)
        x_t = x_t % 1.0

        # 2) 晶格加噪
        l0 = lattice_params
        l_t, eps_lattice = self.noise_scheduler.add_noise(l0, t_lattice)

        # 3) 原子类型离散扩散: 随机掩码
        atom_mask = torch.rand(total_atoms, device=device) < 0.15
        atom_types_noisy = batch["atom_types"].clone()
        atom_types_noisy[atom_mask] = self.num_atom_types  # [MASK] token

        # 构建条件
        condition = self._build_condition(batch["composition"], batch["space_group"])

        # 构建图
        atoms_per = total_atoms // B
        edge_indices, edge_attrs = [], []
        for b in range(B):
            mask_b = batch["batch_idx"] == b
            coords_b = x_t[mask_b]
            lat_b = batch["lattice"][b]
            ei, ev, ed = build_crystal_graph(coords_b, lat_b)
            edge_indices.append(ei + b * atoms_per)
            edge_attrs.append(ed.unsqueeze(-1))

        edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 1), dtype=torch.long, device=device)
        edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else None

        # 去噪预测
        pred_eps_coord, pred_eps_lattice, atom_logits = self.denoiser(
            atom_types_noisy, x_t, l_t, t_coord, edge_index, edge_attr, condition,
        )

        # 损失
        loss_coord = F.mse_loss(pred_eps_coord, eps_coord)
        loss_lattice = F.mse_loss(pred_eps_lattice, eps_lattice)

        atom_target = batch["atom_types"].clone()
        atom_target[~atom_mask] = -100  # ignore unmasked
        loss_atom = F.cross_entropy(atom_logits, atom_target, ignore_index=-100)

        total_loss = loss_lattice + loss_coord * 0.5 + loss_atom * 0.1

        return {
            "loss": total_loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_atom": loss_atom,
        }

    @torch.no_grad()
    def sample(self, composition: torch.Tensor, space_group: int,
               num_atoms: int, num_steps: int = None,
               return_trajectory: bool = False) -> Dict:
        """反向扩散采样 — 从噪声生成晶体结构.

        Args:
            composition: (num_atom_types,) 组成向量
            space_group: 空间群编号
            num_atoms: 目标原子数
            num_steps: 采样步数 (默认全部)
            return_trajectory: 是否返回中间轨迹
        Returns:
            {"lattice": (3,3), "frac_coords": (N,3), "atom_types": (N,)}
        """
        device = composition.device
        num_steps = num_steps or self.num_timesteps
        step_size = self.num_timesteps // num_steps

        # 初始化 — 纯噪声
        x_t = torch.rand(num_atoms, 3, device=device)
        l_t = torch.randn(1, 6, device=device)
        l_t[:, 3:] = l_t[:, 3:] * 30  # smaller noise for angles

        atom_types = torch.randint(0, self.num_atom_types, (num_atoms,), device=device)

        condition = self._build_condition(
            composition.unsqueeze(0),
            torch.tensor([space_group], device=device),
        )

        batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)

        trajectory = [] if return_trajectory else None

        for step in range(num_steps):
            t_val = self.num_timesteps - 1 - step * step_size
            t = torch.full((num_atoms,), t_val, device=device, dtype=torch.long)
            t_batch = torch.tensor([t_val], device=device, dtype=torch.long)

            lat_tensor = self._params_to_lattice(l_t.squeeze(0))
            ei, ev, ed = build_crystal_graph(x_t, lat_tensor)
            edge_attr = ed.unsqueeze(-1) if ed is not None else None

            pred_eps_coord, pred_eps_lattice, atom_logits = self.denoiser(
                atom_types, x_t, l_t, t, ei, edge_attr, condition,
            )

            # 更新
            alpha_bar_t = self.noise_scheduler.alphas_cumprod[t_val].to(device)
            beta_t = self.noise_scheduler.betas[t_val].to(device)
            alpha_t = self.noise_scheduler.alphas[t_val].to(device)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_t = torch.sqrt(alpha_t)

            # 预测 x0
            pred_x0 = (x_t - sqrt_one_minus_alpha_bar * pred_eps_coord) / (
                torch.sqrt(alpha_bar_t) + 1e-10)
            pred_x0 = pred_x0 % 1.0

            pred_l0 = (l_t - sqrt_one_minus_alpha_bar * pred_eps_lattice) / (
                torch.sqrt(alpha_bar_t) + 1e-10)

            # 噪声
            if step < num_steps - 1:
                z_coord = torch.randn_like(x_t)
                z_lattice = torch.randn_like(l_t)
                z_lattice[:, 3:] = z_lattice[:, 3:] * 0.1

                x_t = (sqrt_alpha_t * (1.0 - self.noise_scheduler.alphas_cumprod[t_val - step_size])) / (
                    1.0 - alpha_bar_t + 1e-10) * pred_x0 + beta_t * torch.sqrt(
                    self.noise_scheduler.alphas_cumprod[t_val - step_size]) / (
                    1.0 - alpha_bar_t + 1e-10) * x_t + torch.sqrt(beta_t) * z_coord
                x_t = x_t % 1.0

                l_t = ((sqrt_alpha_t * (1.0 - self.noise_scheduler.alphas_cumprod[t_val - step_size])) / (
                    1.0 - alpha_bar_t + 1e-10)) * pred_l0 + beta_t * torch.sqrt(
                    self.noise_scheduler.alphas_cumprod[t_val - step_size]) / (
                    1.0 - alpha_bar_t + 1e-10) * l_t + torch.sqrt(beta_t) * z_lattice
            else:
                x_t = pred_x0
                l_t = pred_l0

            atom_types = atom_logits.argmax(dim=-1)

            if return_trajectory:
                trajectory.append({
                    "frac_coords": x_t.cpu().numpy().copy(),
                    "lattice_params": l_t.cpu().numpy().copy(),
                    "atom_types": atom_types.cpu().numpy().copy(),
                })

        lattice = self._params_to_lattice(l_t.squeeze(0))

        result = {
            "lattice": lattice.cpu().numpy(),
            "frac_coords": x_t.cpu().numpy(),
            "atom_types": atom_types.cpu().numpy(),
            "space_group": space_group,
        }
        if return_trajectory:
            result["trajectory"] = trajectory

        return result

    def _lattice_to_params(self, lattice: torch.Tensor) -> torch.Tensor:
        """(B, 3, 3) -> (B, 6) [a, b, c, α, β, γ]."""
        a = torch.norm(lattice[:, 0], dim=1)
        b = torch.norm(lattice[:, 1], dim=1)
        c = torch.norm(lattice[:, 2], dim=1)

        alpha = torch.acos(
            torch.clamp(
                torch.sum(lattice[:, 1] * lattice[:, 2], dim=1) / (b * c + 1e-10),
                -1.0, 1.0
            )
        )
        beta = torch.acos(
            torch.clamp(
                torch.sum(lattice[:, 0] * lattice[:, 2], dim=1) / (a * c + 1e-10),
                -1.0, 1.0
            )
        )
        gamma = torch.acos(
            torch.clamp(
                torch.sum(lattice[:, 0] * lattice[:, 1], dim=1) / (a * b + 1e-10),
                -1.0, 1.0
            )
        )

        return torch.stack([a, b, c, alpha, beta, gamma], dim=1)

    def _params_to_lattice(self, params: torch.Tensor) -> torch.Tensor:
        """(6,) -> (3, 3) 晶格矩阵."""
        a, b, c, alpha, beta, gamma = params
        lattice = torch.zeros(3, 3, device=params.device)
        lattice[0, 0] = a
        lattice[1, 0] = b * torch.cos(gamma)
        lattice[1, 1] = b * torch.sin(gamma)
        lattice[2, 0] = c * torch.cos(beta)
        lattice[2, 1] = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / (
            torch.sin(gamma) + 1e-10
        )
        lattice[2, 2] = torch.sqrt(
            torch.clamp(c ** 2 - lattice[2, 0] ** 2 - lattice[2, 1] ** 2, min=0)
        )
        return lattice

    def _build_condition(self, composition: torch.Tensor,
                         space_group: torch.Tensor) -> torch.Tensor:
        """构建条件向量: 组成编码 + 空间群嵌入."""
        comp_emb = self.composition_encoder(composition.float())
        sg_emb = self.space_group_embed(space_group)
        return torch.cat([comp_emb, sg_emb], dim=-1)


def composition_to_vector(elements: List[str], stoichiometry: List[float],
                          max_atomic_num: int = 84) -> np.ndarray:
    """组成 → 归一化向量."""
    from .crystal_representation import ELEM_TO_ATOMIC_NUM

    vec = np.zeros(max_atomic_num, dtype=np.float32)
    total = sum(stoichiometry)
    for el, stoich in zip(elements, stoichiometry):
        z = ELEM_TO_ATOMIC_NUM.get(el, 0)
        if 1 <= z <= max_atomic_num:
            vec[int(z) - 1] = stoich / total
    return vec
