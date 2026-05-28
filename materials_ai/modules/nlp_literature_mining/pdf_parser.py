"""PDF解析器 v3 — PyMuPDF(主) + Tesseract OCR(扫描回退) + pdfplumber/pypdf(兜底) + 图片提取"""

import os
import re
import io
import base64
from typing import Optional, List, Dict


class PDFParser:
    def __init__(self):
        self.sections = []

    def parse(self, filepath: str) -> dict:
        """解析PDF/TXT文件, 返回结构化文本 + 元数据 + 关键图片"""
        filename = os.path.basename(filepath)
        text = self._extract_text(filepath)
        text = self._clean_text(text)
        sections = self._split_sections(text)
        metadata = self.extract_metadata(filepath) if filepath.lower().endswith('.pdf') else {}
        images = self.extract_images(filepath) if filepath.lower().endswith('.pdf') else []
        return {
            "paper_id": self._generate_paper_id(filename),
            "filename": filename,
            "raw_text": text,
            "sections": sections,
            "abstract": self._extract_abstract(text),
            "word_count": len(text.split()),
            "metadata": metadata,
            "is_scanned": metadata.get("is_scanned", False),
            "images": images,
        }

    def parse_text(self, text: str, filename: str = "unknown.txt") -> dict:
        """直接解析文本 (用于TXT文件或外部文本)"""
        text = self._clean_text(text)
        sections = self._split_sections(text)
        return {
            "paper_id": self._generate_paper_id(filename),
            "filename": filename,
            "raw_text": text,
            "sections": sections,
            "abstract": self._extract_abstract(text),
            "word_count": len(text.split()),
            "metadata": {},
            "is_scanned": False,
        }

    def extract_images(self, filepath: str, min_width: int = 150,
                       min_height: int = 150, max_images: int = 20) -> List[Dict]:
        """从PDF中提取关键图片 (过滤掉小图标/装饰元素).

        Args:
            filepath: PDF文件路径
            min_width: 最小图片宽度 (px), 过滤小图标
            min_height: 最小图片高度 (px)
            max_images: 最多返回图片数

        Returns:
            [{"image_base64": str, "width": int, "height": int,
              "page_number": int, "caption": str, "format": str,
              "nearby_text": str}, ...]
        """
        images = []
        try:
            import fitz
            doc = fitz.open(filepath)

            for page_num in range(len(doc)):
                if len(images) >= max_images * 2:
                    break

                page = doc[page_num]
                page_text = page.get_text("text")
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    if len(images) >= max_images * 2:
                        break

                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image.get("ext", "png")
                        w = base_image.get("width", 0)
                        h = base_image.get("height", 0)

                        # 过滤小图标和装饰元素
                        if w < min_width and h < min_height:
                            continue
                        if w < 50 or h < 50:
                            continue
                        if w > 5000 or h > 5000:
                            continue

                        # 转换为 base64
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                        # 生成标注: 从页面文本中查找图片说明
                        caption = self._find_figure_caption(page_text, img_idx, image_list)
                        nearby_text = self._get_nearby_text_for_image(page_text)

                        images.append({
                            "image_base64": img_b64,
                            "width": w,
                            "height": h,
                            "page_number": page_num + 1,
                            "caption": caption,
                            "format": img_ext,
                            "nearby_text": nearby_text[:300],
                            "size_kb": len(img_bytes) // 1024,
                        })
                    except Exception:
                        continue

            doc.close()
        except ImportError:
            pass
        except Exception as e:
            print(f"  [Image extraction error] {e}")

        # 按页数排序, 取前 max_images 张
        images.sort(key=lambda x: (x["page_number"], -x["width"] * x["height"]))
        return images[:max_images]

    def _find_figure_caption(self, page_text: str, img_idx: int,
                             all_images: list) -> str:
        """查找图片对应的Figure说明文字."""
        # 常见Figure标注模式
        caption_patterns = [
            rf'(?:Fig\.?|Figure|FIG\.?|FIGURE)\s*\d+\s*[\.\:\-–—]?\s*(.+?)(?:\n\n|\n[A-Z]|\Z)',
            rf'(?:图\s*\d+)\s*[\.\:\-–—]?\s*(.+?)(?:\n\n|\n[A-Z]|\Z)',
            rf'(?:Fig\.?|Figure)\s*\d+\s*\.?\s*(.+?)(?:\n|\Z)',
        ]

        for pat in caption_patterns:
            matches = list(re.finditer(pat, page_text, re.IGNORECASE | re.DOTALL))
            if matches:
                best = matches[min(img_idx, len(matches) - 1)]
                caption = best.group(1).strip() if best.lastindex else best.group(0).strip()
                return caption[:200]

        return ""

    def _get_nearby_text_for_image(self, page_text: str) -> str:
        """获取图片周围的文字上下文."""
        # 返回页面开头的前几行作为上下文 (简化实现)
        lines = page_text.strip().split("\n")
        meaningful = [l.strip() for l in lines if len(l.strip()) > 20]
        if meaningful:
            return "\n".join(meaningful[:5])
        return page_text[:300]

    def extract_metadata(self, filepath: str) -> dict:
        """使用PyMuPDF提取PDF元数据 (标题/作者/页数/是否扫描件)"""
        info = {}
        try:
            import fitz
            with fitz.open(filepath) as doc:
                meta = doc.metadata
                info = {
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "page_count": doc.page_count,
                    "is_scanned": self._is_scanned(filepath),
                }
        except Exception:
            info = {"title": "", "author": "", "page_count": 0, "is_scanned": False}
        return info

    def _is_scanned(self, filepath: str) -> bool:
        """启发式判断: 前三页可提取字符数 < 200 则认为扫描件"""
        try:
            import fitz
            with fitz.open(filepath) as doc:
                pages = min(3, doc.page_count)
                total = sum(len(doc[i].get_text("text").strip()) for i in range(pages))
            return total < 200
        except Exception:
            return False

    def _extract_text(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf(filepath)
        elif ext in ('.txt', '.text'):
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_pdf(self, filepath: str) -> str:
        """多策略PDF提取: PyMuPDF → OCR(扫描件) → pdfplumber → pypdf"""
        # 策略1: PyMuPDF (fitz)
        text = self._extract_pdf_fitz(filepath)
        if text and len(text.strip()) > 200:
            return text
        # 策略2: OCR (扫描件回退)
        text = self._extract_pdf_ocr(filepath)
        if text and len(text.strip()) > 100:
            return text
        # 策略3: pdfplumber
        return self._extract_pdf_plumber(filepath)

    def _extract_pdf_fitz(self, filepath: str) -> str:
        """PyMuPDF提取: 按块提取保持阅读顺序"""
        try:
            import fitz
            text_parts = []
            with fitz.open(filepath) as doc:
                for page in doc:
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        block_text = block[4].strip() if len(block) > 4 else ""
                        if block_text:
                            text_parts.append(block_text)
            return "\n".join(text_parts)
        except Exception:
            return ""

    def _extract_pdf_ocr(self, filepath: str) -> str:
        """OCR扫描PDF — PaddleOCR(中文优先) → Tesseract(英文回退)"""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            return ""

        try:
            images = convert_from_path(filepath, dpi=200, first_page=1, last_page=50)
        except Exception:
            return ""

        text_parts = []

        # 策略2a: PaddleOCR (中文+英文, 纯Python, 无需系统依赖)
        for img in images:
            page_text = self._ocr_paddle(img)
            if page_text and len(page_text.strip()) > 20:
                text_parts.append(page_text)

        if text_parts and sum(len(t) for t in text_parts) > 100:
            return "\n\n".join(text_parts)

        # 策略2b: Tesseract (英文回退, 需要系统安装)
        text_parts = []
        try:
            import pytesseract
            for img in images:
                page_text = pytesseract.image_to_string(img, lang='eng+chi_sim',
                                                        config='--oem 1 --psm 6')
                if page_text.strip():
                    text_parts.append(page_text)
            if text_parts:
                return "\n\n".join(text_parts)
        except Exception:
            pass

        return ""

    @staticmethod
    def _ocr_paddle(img) -> str:
        """使用PaddleOCR识别单张图片中的文字 (中英文混合)."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return ""
        try:
            _paddle = getattr(PDFParser, '_paddle_ocr_instance', None)
            if _paddle is None:
                PDFParser._paddle_ocr_instance = PaddleOCR(
                    use_angle_cls=True, lang='ch', use_gpu=False,
                    show_log=False,
                )
                _paddle = PDFParser._paddle_ocr_instance
            import numpy as np
            arr = np.array(img)
            result = _paddle.ocr(arr, cls=True)
            if not result or not result[0]:
                return ""
            lines = []
            for line in result[0]:
                text = line[1][0] if len(line) > 1 else ""
                if text:
                    lines.append(text)
            return "\n".join(lines)
        except Exception:
            return ""

    def _extract_pdf_plumber(self, filepath: str) -> str:
        """pdfplumber提取 (原方法重命名)"""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            return self._extract_pdf_pypdf(filepath)

    def _extract_pdf_pypdf(self, filepath: str) -> str:
        """pypdf提取 (终极回退)"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            raise RuntimeError(f"Failed to extract text from PDF: {filepath}")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n\x80-\xFFĀ-ɏͰ-ϿΑ-ω一-鿿]', '', text)
        return text.strip()

    def _split_sections(self, text: str) -> list:
        patterns = [
            r'\b(?:Abstract|ABSTRACT)\b',
            r'\b(?:Introduction|INTRODUCTION)\b',
            r'\b(?:Experimental|EXPERIMENTAL|Methods|METHODS|Materials and Methods)\b',
            r'\b(?:Results(?:\s+and\s+Discussion)?|RESULTS(?:\s+AND\s+DISCUSSION)?)\b',
            r'\b(?:Discussion|DISCUSSION)\b',
            r'\b(?:Conclusions?|CONCLUSIONS?|Summary|SUMMARY)\b',
            r'\b(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b',
        ]
        sections = []
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, text):
                matches.append((m.start(), m.group()))
        matches.sort()
        for i, (start, title) in enumerate(matches):
            end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append({"title": title, "content": content})
        if not sections:
            sections.append({"title": "Full Text", "content": text})
        return sections

    def _extract_abstract(self, text: str) -> str:
        patterns = [
            r'(?:Abstract|ABSTRACT)[\.\:\-–—]*\s*(.+?)(?:\b(?:Introduction|INTRODUCTION)\b)',
            r'(?:Abstract|ABSTRACT)[\.\:\-–—]*\s*(.+?)(?:\n\n)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                abstract = m.group(1).strip()
                if len(abstract.split()) > 20:
                    return abstract[:2000]
        first_para = text.split('\n\n')[0] if '\n\n' in text else text[:1000]
        return first_para[:2000]

    def _generate_paper_id(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return name[:50]
