"""实验参数建议 — 基于成分预测平衡相 / 推荐热处理工艺参数"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PhasePrediction:
    """相预测结果."""

    composition: Dict[str, float]
    temperature: float
    predicted_phases: List[Dict[str, any]]
    confidence: str  # high | medium | low
    basis: str  # 预测依据


@dataclass
class ExperimentAdvice:
    """实验建议."""

    title: str
    composition: Dict[str, float]
    suggested_process: str
    parameters: Dict[str, any]
    expected_results: str
    precautions: List[str]
    references: List[str]


class ExperimentAdvisor:
    """实验参数建议与虚拟相预测.

    基于经验规则和简化热力学模型:
    - Fe-C合金相预测
    - 热处理工艺窗口 (退火/正火/淬火/回火)
    - 常见材料体系的实验参数建议
    """

    # Fe-C合金关键温度 (单位: °C)
    FE_C_CRITICAL_TEMPS = {
        "A1": 727,      # 共析温度
        "A3": {           # γ→α 开始温度 (取决于含碳量)
            0.0: 912, 0.1: 870, 0.2: 830, 0.3: 795,
            0.4: 765, 0.5: 740, 0.6: 730, 0.77: 727,
        },
        "Acm": {          # γ→Fe₃C 开始温度
            0.77: 727, 0.9: 780, 1.0: 830, 1.2: 920,
            1.5: 1030, 2.0: 1120, 2.11: 1148,
        },
    }

    HEAT_TREATMENT_GUIDE = {
        "full_annealing": {
            "description": "完全退火 — 加热至 Ac3+30~50°C, 炉冷",
            "temperature_range": "Ac3 + (30~50)°C",
            "cooling": "炉冷 (Furnace cooling, ~100°C/h)",
            "target": "细化晶粒, 消除内应力, 降低硬度",
            "applicable": "亚共析钢 (C < 0.77%)",
        },
        "normalizing": {
            "description": "正火 — 加热至 Ac3/Acm+30~50°C, 空冷",
            "temperature_range": "Ac3 或 Acm + (30~50)°C",
            "cooling": "空冷 (Air cooling, ~10°C/s)",
            "target": "均匀化组织, 细化晶粒, 改善切削加工性",
            "applicable": "所有碳钢",
        },
        "quenching": {
            "description": "淬火 — 加热至 Ac3+30~50°C (亚共析) 或 Ac1+30~50°C (过共析), 快冷",
            "temperature_range": "Ac3 + (30~50)°C (亚共析) / Ac1 + (30~50)°C (过共析)",
            "cooling": "水冷/油冷 (>100°C/s)",
            "target": "获得马氏体组织, 显著提高硬度",
            "applicable": "C > 0.25% 的钢",
        },
        "tempering": {
            "description": "回火 — 淬火后加热至 150~650°C, 保温后空冷",
            "temperature_range": "150~650°C",
            "cooling": "空冷",
            "target": "消除淬火应力, 调整硬度-韧性平衡",
            "applicable": "淬火钢",
        },
        "spheroidizing": {
            "description": "球化退火 — 加热至 Ac1±20°C, 长时间保温",
            "temperature_range": "Ac1 - 20°C ~ Ac1 + 20°C",
            "cooling": "缓慢冷却",
            "target": "获得球状渗碳体, 改善切削加工性",
            "applicable": "过共析钢 (C > 0.77%)",
        },
    }

    def __init__(self):
        pass

    def predict_phases(self, elements: Dict[str, float],
                       temperature: float = 25) -> PhasePrediction:
        """根据成分和温度预测平衡相组成.

        Args:
            elements: {元素: 质量分数%} 如 {"Fe": 99.55, "C": 0.45}
            temperature: 温度 (°C)
        Returns:
            PhasePrediction
        """
        if "Fe" in elements and "C" in elements:
            return self._predict_fe_c(elements, temperature)

        # 通用组成预测
        return self._predict_general(elements, temperature)

    def _predict_fe_c(self, elements: Dict[str, float],
                      temp: float) -> PhasePrediction:
        """Fe-C 合金相预测."""
        C = elements.get("C", 0) * 100  # 转为 wt% (输入可能是小数)

        # 如果C已经是wt%形式 (如 C=0.45 表示 0.45wt%)
        if C < 0.01 and elements.get("Fe", 99) > 99:
            C = C * 100  # 从小数转为百分比

        # 归一化: 确保C是wt% (0-6.67)
        if C > 6.67:
            C = C / 100.0 if C <= 100 else 6.67

        phases = []

        if temp > 1538:
            phases.append({"phase": "液相 L", "fraction": 1.0, "composition": f"Fe-{C:.3f}%C"})
        elif temp > 1394:
            phases.append({"phase": "δ-Fe (BCC)", "fraction": 1.0})
        elif temp > 1148:
            if C <= 2.11:
                phases.append({"phase": "γ-Fe (奥氏体, FCC)", "fraction": 1.0})
            elif C <= 4.30:
                liq_f = (C - 2.11) / (4.30 - 2.11)
                phases.append({"phase": "液相 L", "fraction": min(1.0, liq_f)})
                phases.append({"phase": "γ-Fe (奥氏体)", "fraction": max(0.0, 1.0 - liq_f)})
            else:
                phases.append({"phase": "液相 L + Fe₃C", "fraction": 1.0})
        elif temp > 727:
            if C <= 0.022:
                phases.append({"phase": "α-Fe (铁素体, BCC)", "fraction": 1.0})
            elif C <= 0.77:
                # α + γ 两相区
                C_alpha = max(0.0001, -0.003 + 0.00004 * temp)
                C_gamma = max(C_alpha, 0.76 - 0.0005 * (temp - 727))
                if C_gamma > C_alpha + 0.001:
                    alpha_f = (C_gamma - C) / (C_gamma - C_alpha + 1e-10)
                    gamma_f = 1.0 - alpha_f
                    alpha_f = max(0.0, min(1.0, alpha_f))
                    gamma_f = max(0.0, min(1.0, gamma_f))
                    phases.append({"phase": "α-Fe (铁素体)", "fraction": round(alpha_f, 3),
                                   "composition": f"~{C_alpha:.3f}%C"})
                    phases.append({"phase": "γ-Fe (奥氏体)", "fraction": round(gamma_f, 3),
                                   "composition": f"~{C_gamma:.3f}%C"})
            elif C <= 2.11:
                phases.append({"phase": "γ-Fe (奥氏体)", "fraction": 1.0})
            else:
                gamma_f = (6.67 - C) / (6.67 - 2.11)
                phases.append({"phase": "γ-Fe (奥氏体)", "fraction": round(gamma_f, 3)})
                phases.append({"phase": "Fe₃C (渗碳体)", "fraction": round(1.0 - gamma_f, 3)})
        else:  # T < 727
            if C <= 0.022:
                phases.append({"phase": "α-Fe (铁素体)", "fraction": 1.0})
            elif C <= 0.77:
                alpha_f = (0.77 - C) / (0.77 - 0.022)
                phases.append({"phase": "α-Fe (铁素体, 先共析)", "fraction": round(alpha_f, 3)})
                phases.append({"phase": "珠光体 (α+Fe₃C)", "fraction": round(1.0 - alpha_f, 3)})
            elif C == 0.77:
                phases.append({"phase": "珠光体 (α+Fe₃C) — 全共析", "fraction": 1.0})
            else:
                pearlite_f = (6.67 - C) / (6.67 - 0.77)
                phases.append({"phase": "珠光体", "fraction": round(pearlite_f, 3)})
                phases.append({"phase": "Fe₃C (渗碳体, 先共析)", "fraction": round(1.0 - pearlite_f, 3)})

        return PhasePrediction(
            composition=elements,
            temperature=temp,
            predicted_phases=phases,
            confidence="high" if phases else "low",
            basis="Fe-C 平衡相图",
        )

    def _predict_general(self, elements: Dict[str, float],
                         temp: float) -> PhasePrediction:
        """通用组成预测 (简化)."""
        phases = []
        total = sum(elements.values())
        norm_comp = {k: v / total for k, v in elements.items()}

        if temp > 1000:
            phases.append({"phase": "可能为单相固溶体或液相", "fraction": 1.0})
        else:
            phases.append({"phase": "建议查阅具体相图", "fraction": 1.0})

        return PhasePrediction(
            composition=norm_comp,
            temperature=temp,
            predicted_phases=phases,
            confidence="low",
            basis="通用规则 (建议查阅实验相图)",
        )

    def suggest_heat_treatment(self, carbon_content: float,
                               target_property: str = "balanced") -> ExperimentAdvice:
        """根据含碳量推荐热处理工艺.

        Args:
            carbon_content: 含碳量 (wt%)
            target_property: "hardness" | "strength" | "ductility" | "balanced"
        """
        C = carbon_content

        if C < 0.25:
            steel_type = "低碳钢 (Low-carbon steel)"
        elif C < 0.60:
            steel_type = "中碳钢 (Medium-carbon steel)"
        elif C <= 2.11:
            steel_type = "高碳钢 (High-carbon steel)"
        else:
            steel_type = "铸铁 (Cast iron)"

        if target_property == "hardness":
            process = "quenching"
            advice = self.HEAT_TREATMENT_GUIDE["quenching"]
            temp = 780 if C < 0.77 else 760
            params = {
                "austenitizing_temp_C": temp,
                "holding_time_min": "30 + 1 min/mm 壁厚",
                "quench_medium": "水 (C<0.4%) 或 油 (C>0.4%)",
                "tempering_temp_C": 200 if C > 0.5 else 300,
                "expected_hardness": "45-65 HRC",
            }
        elif target_property == "ductility":
            process = "full_annealing"
            advice = self.HEAT_TREATMENT_GUIDE["full_annealing"]
            params = {
                "annealing_temp_C": self._estimate_ac3(C) + 40,
                "holding_time_min": "60 + 2 min/mm 壁厚",
                "cooling_method": "炉冷至 500°C 后空冷",
                "expected_hardness": "10-25 HRC",
            }
        else:  # balanced
            process = "normalizing + tempering"
            advice = self.HEAT_TREATMENT_GUIDE["normalizing"]
            params = {
                "normalizing_temp_C": self._estimate_ac3(C) + 50,
                "holding_time_min": "30 + 1 min/mm 壁厚",
                "cooling_method": "空冷",
                "tempering_temp_C": 550 if C > 0.4 else 400,
                "expected_hardness": "25-40 HRC",
            }

        return ExperimentAdvice(
            title=f"{steel_type} 热处理建议 ({C:.2f}%C)",
            composition={"Fe": 100 - C, "C": C},
            suggested_process=process,
            parameters=params,
            expected_results=advice["target"],
            precautions=[
                "防止脱碳: 在保护气氛或真空炉中加热",
                "控制加热速度: 避免热应力导致变形/开裂",
                f"奥氏体化温度不宜过高, 以防晶粒粗大",
                "淬火后应及时回火, 防止延迟开裂",
                "大截面零件需考虑淬透性 (选择合金钢)",
            ],
            references=[
                "《材料科学基础》(第4版), 第8章 固态相变",
                "ASM Handbook Vol.4: Heat Treating",
            ],
        )

    def _estimate_ac3(self, carbon_content: float) -> float:
        """估算Ac3温度."""
        C = carbon_content
        ac3_data = self.FE_C_CRITICAL_TEMPS["A3"]
        if C <= 0:
            return 912
        cs = sorted(ac3_data.keys())
        for i in range(len(cs) - 1):
            if cs[i] <= C <= cs[i + 1]:
                t1, t2 = ac3_data[cs[i]], ac3_data[cs[i + 1]]
                frac = (C - cs[i]) / (cs[i + 1] - cs[i])
                return t1 + frac * (t2 - t1)
        return ac3_data[cs[-1]] if C > cs[-1] else ac3_data[cs[0]]

    def suggest_synthesis(self, target_compound: str) -> ExperimentAdvice:
        """根据目标化合物推荐合成方法."""
        synthesis_db = {
            "BaTiO3": {
                "method": "固相反应法 (Solid-state reaction)",
                "precursors": "BaCO₃ + TiO₂",
                "calcination": "1100-1200°C, 2-4h",
                "sintering": "1300-1350°C, 2h",
                "atmosphere": "空气",
            },
            "LiCoO2": {
                "method": "固相反应法",
                "precursors": "Li₂CO₃ + Co₃O₄",
                "calcination": "800-900°C, 12h",
                "sintering": "900°C, 12h",
                "atmosphere": "空气/氧气",
            },
            "YBa2Cu3O7": {
                "method": "固相反应法",
                "precursors": "Y₂O₃ + BaCO₃ + CuO",
                "calcination": "900-950°C, 12-24h",
                "sintering": "950°C, 12h + O₂退火 450°C",
                "atmosphere": "氧气流",
            },
            "ZnO": {
                "method": "水热法 (Hydrothermal)",
                "precursors": "Zn(NO₃)₂·6H₂O + NaOH",
                "temperature": "150-200°C",
                "duration": "12-24h",
                "atmosphere": "高压釜 (Autoclave)",
            },
        }

        info = synthesis_db.get(target_compound)
        if info:
            return ExperimentAdvice(
                title=f"{target_compound} 合成方案",
                composition={"target": target_compound},
                suggested_process=info["method"],
                parameters={
                    "前驱体": info["precursors"],
                    "煅烧": info.get("calcination", ""),
                    "烧结": info.get("sintering", ""),
                    "气氛": info.get("atmosphere", ""),
                },
                expected_results=f"获得 {target_compound} 粉末/块体",
                precautions=[
                    "称量前将前驱体充分干燥",
                    "中间研磨保证均匀混合",
                    "控制升降温速率 (2-5°C/min)",
                    "使用XRD验证物相纯度",
                ],
                references=[f"{target_compound} 合成文献 (建议检索最新研究)"],
            )

        return ExperimentAdvice(
            title=f"{target_compound} 合成建议 (通用)",
            composition={"target": target_compound},
            suggested_process="建议检索文献",
            parameters={"note": "该化合物暂无预设方案, 请查阅最新研究论文"},
            expected_results="待文献支持",
            precautions=["查阅至少3篇独立文献确认方案"],
            references=["Web of Science / Google Scholar"],
        )

    def predict_cooling_curve(self, carbon_content: float,
                              cooling_rate: float = 1.0) -> Dict:
        """基于简化TTT/CCT概念, 预测连续冷却后的组织.

        Args:
            carbon_content: 含碳量 (wt%)
            cooling_rate: 冷却速率 (°C/s)
        Returns:
            {predicted_microstructure, hardness_estimate, ...}
        """
        C = carbon_content

        if cooling_rate > 100:
            ms = 561 - 474 * C - 33 * 1.0  # Ms温度 (Andrews公式简化)
            microstructure = "马氏体 (Martensite)"
            hardness = 30 + 40 * C  # HRC
        elif cooling_rate > 10:
            microstructure = "贝氏体 + 少量马氏体"
            hardness = 25 + 30 * C
        elif cooling_rate > 1:
            microstructure = "细珠光体 (索氏体/Sorbite)"
            hardness = 15 + 15 * C
        elif cooling_rate > 0.1:
            microstructure = "珠光体 + 铁素体 (亚共析) 或 珠光体 + 渗碳体 (过共析)"
            hardness = 10 + 10 * C
        else:
            microstructure = "粗珠光体 (平衡组织)"
            hardness = 5 + 8 * C

        return {
            "carbon_content": C,
            "cooling_rate_C_per_s": cooling_rate,
            "predicted_microstructure": microstructure,
            "estimated_hardness_HRC": round(min(68, hardness), 1),
            "Ms_temperature_C": round(561 - 474 * C - 33, 0) if C < 1.2 else 25,
            "note": "简化CCT预测, 实际组织受合金元素影响",
        }
