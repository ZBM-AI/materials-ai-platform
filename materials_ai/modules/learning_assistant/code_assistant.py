"""Python代码辅助 — 材料科学计算代码生成与安全执行"""

import os
import re
import sys
import textwrap
from io import StringIO
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodeResult:
    """代码执行结果."""

    code: str
    output: str = ""
    error: str = ""
    success: bool = False
    description: str = ""


class CodeAssistant:
    """材料科学Python代码生成器.

    支持的代码类别:
    - 相图绘制 (二元/三元相图的关键区域)
    - 扩散曲线 (菲克定律数值解)
    - 晶体结构可视化 (FCC/BCC/HCP)
    - 力学性能曲线 (应力-应变, 疲劳S-N)
    - 热力学计算 (Gibbs自由能)
    - TTT/CCT曲线示意
    """

    CODE_TEMPLATES = {
        "fe_c_phase_diagram": {
            "description": "Fe-C 相图关键区域",
            "keywords": ["fe-c相图", "铁碳相图", "phase diagram fe-c", "fe-c phase"],
            "code": r'''
import numpy as np
import matplotlib.pyplot as plt

# Fe-C 二元相图关键数据点 (亚稳系: Fe-Fe3C)
# 温度 (C) 和对应的碳含量 (wt%)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# --- 特征温度线 ---
# A3线 (γ → α 转变开始)
T_a3 = np.array([912, 850, 800, 750, 727])
C_a3 = np.array([0.0, 0.1, 0.25, 0.45, 0.77])
ax.plot(C_a3, T_a3, 'b-', linewidth=2, label='A$_3$ (γ→α start)')

# Acm线 (γ → Fe3C 开始)
T_acm = np.array([1148, 1000, 900, 800, 727])
C_acm = np.array([2.11, 1.5, 0.95, 0.8, 0.77])
ax.plot(C_acm, T_acm, 'r-', linewidth=2, label='A$_{cm}$ (γ→Fe$_3$C start)')

# A1线 (共析等温线)
ax.axhline(y=727, color='green', linestyle='--', linewidth=1.5, label='A$_1$ (727°C)')
# 共晶等温线
ax.axhline(y=1148, color='purple', linestyle='--', linewidth=1.5, label='Eutectic (1148°C)')

# --- 关键点标注 ---
points = {
    'Eutectoid\n(0.77%, 727°C)': (0.77, 727),
    'Eutectic\n(4.30%, 1148°C)': (4.30, 1148),
    r'$\alpha$-Fe max C\n(0.022%)': (0.022, 727),
    r'$\gamma$-Fe max C\n(2.11%)': (2.11, 1148),
}

for label, (c, t) in points.items():
    ax.plot(c, t, 'ko', markersize=6)
    ax.annotate(label, (c, t), textcoords="offset points",
                xytext=(10, 10), fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.8))

# --- 相区标注 ---
ax.text(0.05, 800, r'$\mathbf{\alpha + \gamma}$', fontsize=12, color='blue')
ax.text(0.4, 900, r'$\mathbf{\gamma}$ (Austenite)', fontsize=14, color='darkred')
ax.text(0.3, 500, r'$\mathbf{\alpha + Fe_3C}$', fontsize=12, color='darkgreen')
ax.text(0.3, 600, r'$\mathbf{Pearlite}$', fontsize=10, style='italic')
ax.text(1.5, 1000, r'$\mathbf{\gamma + Fe_3C}$', fontsize=12, color='orange')
ax.text(3.5, 800, r'$\mathbf{Fe_3C + L}$', fontsize=10)
ax.text(3.0, 1200, r'$\mathbf{L}$ (Liquid)', fontsize=14, color='red')

# --- 钢/铸铁分界线 ---
ax.axvline(x=2.11, color='gray', linestyle=':', linewidth=1)
ax.text(1.0, 1300, 'Steel', fontsize=12, ha='center')
ax.text(3.5, 1300, 'Cast Iron', fontsize=12, ha='center')

ax.set_xlim(0, 6.7)
ax.set_ylim(400, 1600)
ax.set_xlabel('Carbon Content (wt%)', fontsize=13)
ax.set_ylabel('Temperature (°C)', fontsize=13)
ax.set_title('Fe-C Phase Diagram (Metastable: Fe-Fe$_3$C)', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("关键相变反应:")
print("  共析 (Eutectoid): γ(0.77%C) → α(0.022%C) + Fe₃C(6.67%C)  at 727°C")
print("  共晶 (Eutectic):  L(4.30%C) → γ(2.11%C) + Fe₃C(6.67%C)  at 1148°C")
print("  包晶 (Peritectic): δ(0.09%C) + L(0.53%C) → γ(0.17%C)  at 1495°C")
''',
        },
        "diffusion_curve": {
            "description": "菲克第二定律扩散曲线",
            "keywords": ["扩散曲线", "菲克", "扩散方程", "diffusion", "fick"],
            "code": r'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 无限长扩散偶 — 菲克第二定律解析解
# C(x,t) = (C1+C2)/2 + (C2-C1)/2 * erf(x / (2*sqrt(D*t)))

D = 1e-11  # 扩散系数 (m^2/s)
C1 = 0.0   # 左侧初始浓度
C2 = 1.0   # 右侧初始浓度

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

times = [3600, 14400, 86400, 172800]  # 1h, 4h, 24h, 48h (秒)
x = np.linspace(-0.005, 0.005, 500)  # ±5 mm

for t in times:
    C = (C1 + C2) / 2 + (C2 - C1) / 2 * erf(x / (2 * np.sqrt(D * t)))
    hours = t / 3600
    ax.plot(x * 1000, C, linewidth=2, label=f't = {hours:.0f} h')

ax.set_xlabel('Distance x (mm)', fontsize=12)
ax.set_ylabel('Concentration C (a.u.)', fontsize=12)
ax.set_title('Diffusion Profiles — Fick\'s 2nd Law\n'
             f'D = {D:.1e} m$^2$/s', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 标注关键时间对应的扩散距离 x ≈ sqrt(Dt)
for t in [3600, 86400]:
    x_rms = np.sqrt(D * t) * 1000
    ax.axvline(x=x_rms, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-x_rms, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'$\\sqrt{{Dt}}$ = {x_rms:.2f} mm\n({t/3600:.0f}h)',
                xy=(x_rms, 0.6), fontsize=8, ha='left')

plt.tight_layout()
plt.show()

print("扩散距离估算 (x ≈ sqrt(Dt)):")
for t in times:
    x_rms = np.sqrt(D * t) * 1e6
    print(f"  t = {t/3600:6.1f} h  →  x_rms ≈ {x_rms:.1f} μm")
''',
        },
        "crystal_structure_3d": {
            "description": "晶体结构3D可视化 (FCC/BCC/HCP)",
            "keywords": ["晶体结构", "fcc", "bcc", "hcp", "crystal", "晶胞"],
            "code": r'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_unit_cell(ax, structure='FCC', a=1.0):
    """绘制FCC/BCC/HCP晶胞的3D原子排布"""

    # 顶点
    corners = np.array([[x, y, z] for x in [0, a] for y in [0, a] for z in [0, a]])
    ax.scatter(*corners.T, c='blue', s=100, alpha=0.6, label='Corner atoms')

    if structure == 'FCC':
        # 面心原子
        face_centers = np.array([
            [a/2, a/2, 0], [a/2, a/2, a],
            [a/2, 0, a/2], [a/2, a, a/2],
            [0, a/2, a/2], [a, a/2, a/2],
        ])
        ax.scatter(*face_centers.T, c='red', s=150, alpha=0.8, label='Face-center atoms')
        title = 'FCC (Face-Centered Cubic)'
        cn = 12
        apf = 0.74

    elif structure == 'BCC':
        # 体心原子
        ax.scatter(a/2, a/2, a/2, c='red', s=200, alpha=0.8, label='Body-center atom')
        title = 'BCC (Body-Centered Cubic)'
        cn = 8
        apf = 0.68

    elif structure == 'HCP':
        a_hcp, c = a, a * np.sqrt(8/3)
        corners_hcp = np.array([
            [0, 0, 0], [a, 0, 0], [a/2, a*np.sqrt(3)/2, 0],
            [0, 0, c], [a, 0, c], [a/2, a*np.sqrt(3)/2, c],
        ])
        mid_atom = np.array([[a/2, a/(2*np.sqrt(3)), c/2]])
        ax.scatter(*corners_hcp.T, c='blue', s=100, alpha=0.6)
        ax.scatter(*mid_atom.T, c='red', s=150, alpha=0.8)
        title = 'HCP (Hexagonal Close-Packed)'
        cn = 12
        apf = 0.74

    # 晶胞边框
    edges = [
        (0, 0, 0), (a, 0, 0), (a, a, 0), (0, a, 0), (0, 0, 0),
        (0, 0, a), (a, 0, a), (a, a, a), (0, a, a), (0, 0, a),
    ]
    for i in range(len(edges) - 1):
        ax.plot3D(
            *zip(*[(edges[i][0], edges[i+1][0]),
                    (edges[i][1], edges[i+1][1]),
                    (edges[i][2], edges[i+1][2])]),
            'gray', linewidth=1, alpha=0.5
        )
    # vertical edges
    for x, y in [(0, 0), (a, 0), (a, a), (0, a)]:
        ax.plot3D(*zip(*[(x, y, 0), (x, y, a)]), 'gray', linewidth=1, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\nCN = {cn}, APF = {apf:.2f}', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_box_aspect([1, 1, 1])

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

plot_unit_cell(ax1, 'FCC')
plot_unit_cell(ax2, 'BCC')
plot_unit_cell(ax3, 'HCP')

plt.tight_layout()
plt.show()
''',
        },
        "stress_strain": {
            "description": "应力-应变曲线",
            "keywords": ["应力应变", "stress strain", "拉伸", "力学曲线"],
            "code": r'''
import numpy as np
import matplotlib.pyplot as plt

# 模拟典型工程应力-应变曲线
strain = np.linspace(0, 0.25, 500)

# --- 模型参数 ---
E = 200e3       # 弹性模量 (MPa) — 钢
sigma_y = 350   # 屈服强度 (MPa)
sigma_uts = 550 # 抗拉强度 (MPa)
n = 0.15        # 加工硬化指数
elongation = 0.20  # 延伸率

stress = np.zeros_like(strain)
for i, eps in enumerate(strain):
    if eps <= sigma_y / E:
        stress[i] = E * eps
    else:
        plastic_strain = eps - sigma_y / E
        stress[i] = min(sigma_y + 400 * plastic_strain**n, sigma_uts)

# necking region
neck_start = np.argmax(stress)
stress[neck_start:] = sigma_uts * np.exp(-3.0 * (strain[neck_start:] - strain[neck_start]))

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(strain * 100, stress, 'b-', linewidth=2.5)

# 标注关键点
ax.axhline(y=sigma_y, color='orange', linestyle='--', linewidth=1)
ax.annotate(f'Yield Strength\n$\\sigma_y$ = {sigma_y} MPa',
            xy=(sigma_y/E*100, sigma_y), fontsize=10, color='orange')

ax.axhline(y=sigma_uts, color='red', linestyle='--', linewidth=1)
ax.annotate(f'UTS = {sigma_uts} MPa',
            xy=(strain[neck_start]*100, sigma_uts), fontsize=10, color='red')

# 弹性模量斜率线
eps_elastic = np.array([0, sigma_y/E]) * 100
sigma_elastic = np.array([0, sigma_y])
ax.plot(eps_elastic, sigma_elastic, 'gray', linewidth=3, alpha=0.5,
        label=f'E = {E/1000:.0f} GPa')

# 标注区域
ax.axvspan(0, sigma_y/E*100, alpha=0.1, color='green', label='Elastic Region')
ax.axvspan(sigma_y/E*100, strain[neck_start]*100, alpha=0.1, color='yellow', label='Plastic (Uniform)')
ax.axvspan(strain[neck_start]*100, max(strain)*100, alpha=0.1, color='red', label='Necking')

ax.set_xlabel('Engineering Strain (%)', fontsize=13)
ax.set_ylabel('Engineering Stress (MPa)', fontsize=13)
ax.set_title('Typical Stress-Strain Curve (Ductile Metal)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(strain) * 100)

# 韧性 (曲线下面积)
toughness = np.trapz(stress, strain)
ax.text(0.65, 0.3, f'Toughness ≈ {toughness:.0f} MJ/m$^3$\n'
        f'Elongation ≈ {elongation*100:.0f}%',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()
''',
        },
        "lever_rule": {
            "description": "杠杆定律计算相比例",
            "keywords": ["杠杆定律", "lever rule", "相比例", "phase fraction"],
            "code": r'''
import numpy as np

def lever_rule(C0: float, Ca: float, Cb: float) -> tuple:
    """杠杆定律: 计算两相质量分数.

    Args:
        C0: 合金平均成分 (wt%)
        Ca: α 相的成分 (wt%)
        Cb: β 相的成分 (wt%)
    Returns:
        (W_alpha, W_beta): α和β的质量分数
    """
    if Ca >= Cb:
        raise ValueError("Ca must be < Cb")
    if not (Ca <= C0 <= Cb):
        raise ValueError(f"C0 ({C0}) 必须在 [{Ca}, {Cb}] 范围内")

    W_alpha = (Cb - C0) / (Cb - Ca)
    W_beta = (C0 - Ca) / (Cb - Ca)

    return W_alpha, W_beta

# ===== Fe-C 合金例子 =====
print("=" * 60)
print("Fe-C 合金相比例计算 (杠杆定律)")
print("=" * 60)

# 共析温度稍下: α (0.022% C) + Fe3C (6.67% C)
C_alpha = 0.022  # 铁素体含碳量
C_fe3c = 6.67    # 渗碳体含碳量

alloys = [0.2, 0.45, 0.77, 1.0, 1.5]  # 常见钢的含碳量

print(f"\n{'C0 (wt% C)':<14} {'W_α (ferrite)':<16} {'W_Fe3C (cementite)':<20} {'组织'}")
print("-" * 60)

for C0 in alloys:
    W_alpha, W_fe3c = lever_rule(C0, C_alpha, C_fe3c)
    if C0 < 0.77:
        microstructure = "亚共析钢 (Hypoeutectoid)"
    elif C0 == 0.77:
        microstructure = "共析钢 (Eutectoid) — Pearlite"
    else:
        microstructure = "过共析钢 (Hypereutectoid)"
    print(f"{C0:<14.2f} {W_alpha:<16.3f} {W_fe3c:<20.3f} {microstructure}")

# ===== 相图单变量计算 =====
print(f"\n{'='*60}")
print("计算 Fe-0.45%C 钢在共析温度稍下的显微组织:")
print(f"{'='*60}")

# 先共析铁素体
C0 = 0.45
W_proeutectoid_alpha = lever_rule(C0, C_alpha, 0.77)[0]
W_pearlite = 1.0 - W_proeutectoid_alpha
W_total_alpha = lever_rule(C0, C_alpha, C_fe3c)[0]
W_total_fe3c = 1.0 - W_total_alpha

print(f"  先共析铁素体 (Proeutectoid α): {W_proeutectoid_alpha*100:.1f}%")
print(f"  珠光体 (Pearlite):              {W_pearlite*100:.1f}%")
print(f"  总铁素体 (Total α):             {W_total_alpha*100:.1f}%")
print(f"  总渗碳体 (Total Fe₃C):          {W_total_fe3c*100:.1f}%")
''',
        },
    }

    def __init__(self, api_key: str = None,
                 base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def match_template(self, query: str) -> Optional[str]:
        """根据用户查询匹配代码模板."""
        query_lower = query.lower()
        for template_id, info in self.CODE_TEMPLATES.items():
            for kw in info["keywords"]:
                if kw.lower() in query_lower:
                    return template_id
        return None

    def generate_code(self, query: str, use_llm: bool = False) -> CodeResult:
        """根据用户查询生成Python代码.

        Args:
            query: 用户的代码需求 (如 "画Fe-C相图")
            use_llm: 是否使用LLM生成新代码
        Returns:
            CodeResult
        """
        # 先尝试匹配模板
        template_id = self.match_template(query)

        if template_id:
            info = self.CODE_TEMPLATES[template_id]
            code = textwrap.dedent(info["code"]).strip()
            return CodeResult(
                code=code,
                success=True,
                description=info["description"],
            )

        # 尝试LLM生成
        if use_llm and self.api_key:
            return self._llm_generate_code(query)

        # 通用回退
        return CodeResult(
            code=self._fallback_code(query),
            success=True,
            description="通用材料科学计算模板",
        )

    def execute_code(self, code: str, timeout: int = 30) -> CodeResult:
        """在隔离环境中执行Python代码 (安全沙箱)."""
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        success = True
        error = ""

        safe_globals = {
            "np": __import__("numpy"),
            "plt": None,
            "print": print,
            "__builtins__": {
                k: v for k, v in __builtins__.__dict__.items()
                if k not in ("__import__", "eval", "exec", "compile", "open")
            },
        }

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            safe_globals["plt"] = plt
            safe_globals["matplotlib"] = matplotlib

            from scipy.special import erf
            safe_globals["erf"] = erf
            safe_globals["scipy"] = __import__("scipy")
        except ImportError:
            pass

        try:
            exec(code, safe_globals)
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {e}"

        sys.stdout = old_stdout
        output = captured.getvalue()

        return CodeResult(
            code=code,
            output=output,
            error=error,
            success=success,
            description="Code execution result",
        )

    def _llm_generate_code(self, query: str) -> CodeResult:
        """LLM生成材料科学Python代码."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            system_prompt = """你是一位材料科学计算专家。请生成完整、可运行的Python代码。
要求:
1. 使用 numpy, matplotlib, scipy 等标准库
2. 代码包含详细注释 (中英文皆可)
3. 生成图表时用 plt.show()
4. 关键物理量正确标注单位
5. 代码完整可直接运行"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请生成Python代码: {query}"},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content
            code = self._extract_code_block(raw)
            return CodeResult(code=code, success=True, description="LLM生成")
        except Exception as e:
            return CodeResult(code="", error=str(e), success=False)

    def _extract_code_block(self, text: str) -> str:
        """从LLM输出中提取Python代码块."""
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(matches)
        # 如果没有 markdown 代码块, 尝试直接返回
        if "import " in text and "plt." in text:
            return text.strip()
        return text.strip()

    def _fallback_code(self, query: str) -> str:
        """通用材料科学计算模板."""
        return textwrap.dedent(f'''
"""
材料科学计算: {query}
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO: 根据具体问题调整参数
print("材料科学计算模板")
print("=" * 40)
print("请根据具体需求修改以下参数:")

# 常用材料参数示例
E = 200e9      # 弹性模量 (Pa)
nu = 0.3       # 泊松比
rho = 7800     # 密度 (kg/m³)
Tm = 1811      # 熔点 (K) — 纯铁

print(f"  弹性模量 E = {{E/1e9:.1f}} GPa")
print(f"  泊松比 ν = {{nu}}")
print(f"  密度 ρ = {{rho}} kg/m³")
print(f"  熔点 Tm = {{Tm}} K")
''').strip()

    def list_available_templates(self) -> Dict[str, str]:
        """列出所有可用模板."""
        return {tid: info["description"] for tid, info in self.CODE_TEMPLATES.items()}
