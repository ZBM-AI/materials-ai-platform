"""显微图像分析报告生成器"""

import os
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from io import BytesIO

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MicrographReport:
    """生成显微图像分析综合报告 (HTML + JSON)."""

    def __init__(self, output_dir: str = None):
        from config import DATA_DIR
        self.output_dir = output_dir or os.path.join(DATA_DIR, "reports", "microscopy")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, original_image: np.ndarray,
                 phase_result: Dict = None,
                 grain_result: Dict = None,
                 defect_result: Dict = None,
                 structure_result: Dict = None,
                 metadata: Dict = None) -> str:
        """生成完整HTML分析报告.

        Returns:
            HTML报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"micrograph_report_{timestamp}.html")

        sections = []

        # 头部
        sections.append(f"""
        <html><head><meta charset="utf-8">
        <title>Micrograph Analysis Report — {timestamp}</title>
        <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: auto; padding: 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;
                        border-left: 4px solid #3498db; }}
        .metric-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .metric-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; }}
        .image-container {{ margin: 15px 0; text-align: center; }}
        .image-container img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
        .image-row {{ display: flex; gap: 15px; flex-wrap: wrap; }}
        .image-row .img-box {{ flex: 1; min-width: 300px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f1f3f4; font-weight: 600; }}
        .bar {{ background: #3498db; height: 20px; border-radius: 3px; transition: width 0.3s; }}
        .bar-label {{ font-size: 12px; margin: 2px 0; }}
        </style></head><body>
        <h1>Microscopy Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """)

        # 元数据
        if metadata:
            sections.append("<h2>Sample Information</h2><table>")
            for k, v in metadata.items():
                sections.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
            sections.append("</table>")

        # 原始图像
        orig_b64 = self._img_to_base64(original_image)
        sections.append(f"<h2>Original Micrograph</h2>"
                         f"<div class='image-container'><img src='data:image/png;base64,{orig_b64}'></div>")

        # 物相分割
        if phase_result:
            sections.append("<h2>Phase Segmentation</h2>")
            sections.append(f"<p>Method: <b>{phase_result.get('method', 'N/A')}</b></p>")

            fractions = phase_result.get("phase_fractions", {})
            sections.append("<div class='metric-grid'>")
            for phase, frac in fractions.items():
                pct = frac * 100
                sections.append(
                    f"<div class='metric-card'>"
                    f"<div class='value'>{pct:.1f}%</div>"
                    f"<div class='label'>{phase}</div>"
                    f"</div>"
                )
            sections.append("</div>")

            # 分割可视化
            seg_img = phase_result.get("segmented_image")
            if seg_img is not None:
                seg_b64 = self._img_to_base64(seg_img)
                sections.append("<div class='image-row'>"
                                 "<div class='img-box'><h4>Segmentation Map</h4>"
                                 f"<img src='data:image/png;base64,{seg_b64}'></div>"
                                 "</div>")

        # 晶粒分析
        if grain_result:
            sections.append("<h2>Grain Analysis</h2>")
            sections.append(f"<p>Method: <b>{grain_result.get('method', 'N/A')}</b></p>")

            sections.append("<div class='metric-grid'>")
            sections.append(f"<div class='metric-card'><div class='value'>{grain_result.get('num_grains', 0)}</div><div class='label'>Grains Detected</div></div>")
            sections.append(f"<div class='metric-card'><div class='value'>{grain_result.get('avg_grain_size_um', 0):.2f}</div><div class='label'>Avg Grain Size (μm)</div></div>")
            sections.append(f"<div class='metric-card'><div class='value'>{grain_result.get('grain_size_astm', 0)}</div><div class='label'>ASTM Grain Size</div></div>")
            sections.append(f"<div class='metric-card'><div class='value'>{grain_result.get('intercept_length_um', 0):.2f}</div><div class='label'>Intercept Length (μm)</div></div>")
            sections.append("</div>")

            grain_img = grain_result.get("annotated_image")
            if grain_img is not None:
                gb64 = self._img_to_base64(grain_img)
                sections.append(f"<div class='image-container'><h4>Grain Boundaries</h4>"
                                 f"<img src='data:image/png;base64,{gb64}'></div>")

            # 尺寸分布表
            areas = grain_result.get("grain_areas_um2", [])
            diams = grain_result.get("grain_diameters_um", [])
            if areas and diams:
                sections.append("<h4>Grain Size Distribution</h4><table>"
                                 "<tr><th>Statistic</th><th>Area (μm²)</th><th>Diameter (μm)</th></tr>")
                sections.append(f"<tr><td>Mean</td><td>{np.mean(areas):.2f}</td><td>{np.mean(diams):.2f}</td></tr>")
                sections.append(f"<tr><td>Std</td><td>{np.std(areas):.2f}</td><td>{np.std(diams):.2f}</td></tr>")
                sections.append(f"<tr><td>Min</td><td>{np.min(areas):.2f}</td><td>{np.min(diams):.2f}</td></tr>")
                sections.append(f"<tr><td>Max</td><td>{np.max(areas):.2f}</td><td>{np.max(diams):.2f}</td></tr>")
                sections.append(f"<tr><td>Median</td><td>{np.median(areas):.2f}</td><td>{np.median(diams):.2f}</td></tr>")
                sections.append("</table>")

        # 缺陷分析
        if defect_result:
            sections.append("<h2>Defect Analysis</h2>")
            sections.append("<div class='metric-grid'>")
            sections.append(f"<div class='metric-card'><div class='value'>{defect_result.get('total_defects', 0)}</div><div class='label'>Total Defects</div></div>")
            sections.append(f"<div class='metric-card'><div class='value'>{defect_result.get('defect_fraction', 0)*100:.2f}%</div><div class='label'>Area Fraction</div></div>")
            sections.append("</div>")

            for dtype in ["pores", "cracks", "inclusions"]:
                items = defect_result.get(dtype, [])
                if items:
                    sections.append(f"<h4>{dtype.capitalize()} ({len(items)})</h4>")

            def_img = defect_result.get("annotated_image")
            if def_img is not None:
                db64 = self._img_to_base64(def_img)
                sections.append(f"<div class='image-container'>"
                                 f"<img src='data:image/png;base64,{db64}'></div>")

        # 组织分类
        if structure_result:
            sections.append("<h2>Microstructure Classification</h2>")
            sections.append(f"<div class='metric-card' style='margin:15px 0;'><div class='value' style='font-size:36px;'>{structure_result.get('predicted_class', 'N/A')}</div><div class='label'>Predicted Microstructure ({structure_result.get('method', '')})</div></div>")

            probs = structure_result.get("probabilities", {})
            if probs:
                sections.append("<h4>Class Probabilities</h4>")
                for cls_, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    pct = prob * 100
                    sections.append(f"<div class='bar-label'>{cls_}: {pct:.1f}%</div>"
                                     f"<div style='background:#eee;border-radius:3px;'>"
                                     f"<div class='bar' style='width:{pct}%'></div></div>")

        sections.append("</body></html>")

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sections))

        return html_path

    def export_json(self, phase_result: Dict = None,
                    grain_result: Dict = None,
                    defect_result: Dict = None,
                    structure_result: Dict = None,
                    metadata: Dict = None) -> str:
        """导出JSON格式分析数据."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "phase_analysis": self._serialize(phase_result),
            "grain_analysis": self._serialize(grain_result),
            "defect_analysis": self._serialize(defect_result),
            "structure_classification": self._serialize(structure_result),
        }

        json_path = os.path.join(
            self.output_dir,
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return json_path

    @staticmethod
    def _img_to_base64(img: np.ndarray) -> str:
        if img is None:
            return ""
        _, buf = cv2.imencode('.png', img)
        return base64.b64encode(buf).decode('utf-8')

    @staticmethod
    def _serialize(result: Dict) -> dict:
        """去除numpy类型, 准备序列化."""
        if result is None:
            return {}
        clean = {}
        for k, v in result.items():
            if k.endswith("_image") or k.endswith("_map"):
                continue
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (list, tuple)) and len(v) > 0:
                if isinstance(v[0], np.ndarray):
                    continue
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, dict):
                clean[k] = MicrographReport._serialize(v)
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    clean[k] = [MicrographReport._serialize(item) for item in v]
                else:
                    clean[k] = v
            else:
                clean[k] = v
        return clean
