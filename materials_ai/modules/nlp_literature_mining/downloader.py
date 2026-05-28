"""论文批量下载 — arXiv + Semantic Scholar + Crossref + CNKI + PubMed"""

import os
import re
import ssl
import time
import json
import hashlib
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class PaperMetadata:
    title: str
    authors: List[str] = field(default_factory=list)
    year: int = 0
    doi: str = ""
    abstract: str = ""
    source: str = ""
    pdf_url: str = ""
    local_path: str = ""


class PaperDownloader:
    """批量搜索并下载材料科学论文.

    支持 arXiv API 和 Semantic Scholar API.
    优先使用 requests 库 (SSL兼容性更好), 自动回退 urllib.
    """

    def __init__(self, download_dir: str = None, max_papers: int = 50):
        if download_dir is None:
            from config import DOWNLOAD_DIR
            download_dir = DOWNLOAD_DIR
        self.download_dir = download_dir
        self.max_papers = max_papers
        os.makedirs(self.download_dir, exist_ok=True)

    # ---- HTTP helpers ----

    def _get_ssl_context(self):
        """创建宽松的SSL上下文 (用于Windows证书问题)."""
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        return ctx

    def _http_get(self, url: str, timeout: int = 30) -> bytes:
        """HTTP GET, 优先 requests → urllib+SSL fallback."""
        headers = {"User-Agent": "MaterialsAI/2.0"}

        if HAS_REQUESTS:
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.SSLError:
                resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.RequestException:
                pass  # fall through to urllib

        # urllib fallback
        req = urllib.request.Request(url, headers=headers)
        try:
            ctx = self._get_ssl_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return resp.read()
        except Exception:
            # 最后手段: 跳过SSL验证
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return resp.read()

    def _http_get_json(self, url: str, timeout: int = 30) -> dict:
        return json.loads(self._http_get(url, timeout=timeout))

    def _http_download_file(self, url: str, local_path: str, timeout: int = 60) -> bool:
        """下载文件到本地路径."""
        headers = {"User-Agent": "MaterialsAI/2.0"}

        if HAS_REQUESTS:
            try:
                resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except requests.exceptions.SSLError:
                resp = requests.get(url, headers=headers, timeout=timeout, stream=True, verify=False)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except requests.exceptions.RequestException:
                pass

        # urllib fallback
        req = urllib.request.Request(url, headers=headers)
        try:
            ctx = self._get_ssl_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                with open(local_path, 'wb') as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                return True
        except Exception:
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    with open(local_path, 'wb') as f:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                    return True
            except Exception:
                return False

    def _retry_request(self, url: str, timeout: int = 30, max_retries: int = 3,
                       is_json: bool = False) -> bytes:
        """带重试和退避的HTTP请求."""
        last_error = None
        for attempt in range(max_retries):
            try:
                if is_json:
                    return self._http_get_json(url, timeout=timeout)
                return self._http_get(url, timeout=timeout)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    time.sleep(wait)
                    continue
                last_error = e
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        if last_error:
            if is_json:
                return {}
            return b""

    # ---- Search APIs ----

    def search_arxiv(self, query: str, year_from: int = None,
                     year_to: int = None) -> List[PaperMetadata]:
        """通过 arXiv API 搜索论文.

        API: https://export.arxiv.org/api/query?search_query=all:{query}&max_results=N
        """
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(self.max_papers, 100),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        papers = []
        try:
            content = self._retry_request(url, timeout=30)
            if not content:
                return papers
            if isinstance(content, bytes):
                root = ET.fromstring(content)
            else:
                return papers

            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                title_text = title.text.strip().replace("\n", " ") if title is not None else ""
                summary = entry.find("atom:summary", ns)
                abstract = summary.text.strip().replace("\n", " ") if summary is not None else ""
                published = entry.find("atom:published", ns)
                year = int(published.text[:4]) if published is not None else 0
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text.strip())
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href", "")
                        break
                if not pdf_url:
                    arxiv_id = entry.find("atom:id", ns)
                    if arxiv_id is not None:
                        aid = arxiv_id.text.strip().split("/abs/")[-1]
                        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
                papers.append(PaperMetadata(
                    title=title_text, authors=authors, year=year,
                    abstract=abstract, source="arxiv", pdf_url=pdf_url,
                ))
        except Exception as e:
            print(f"  [arXiv search error] {e}")
        return papers

    def search_semantic_scholar(self, query: str, year_from: int = None,
                                year_to: int = None,
                                api_key: str = None) -> List[PaperMetadata]:
        """通过 Semantic Scholar API 搜索论文.

        API: https://api.semanticscholar.org/graph/v1/paper/search?query={query}
        速率限制: 无API Key 1 req/s, 有API Key 10 req/s
        """
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(self.max_papers, 100),
            "offset": 0,
            "fields": "title,authors,year,abstract,externalIds,openAccessPdf,publicationDate",
        }
        papers = []

        # Semantic Scholar 分页获取 (每页最多100)
        total_needed = min(self.max_papers, 500)
        headers = {"User-Agent": "MaterialsAI/2.0"}
        if api_key:
            headers["x-api-key"] = api_key

        while len(papers) < total_needed:
            params["offset"] = len(papers)
            params["limit"] = min(100, total_needed - len(papers))
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            try:
                if HAS_REQUESTS:
                    resp = requests.get(url, headers=headers, timeout=30)
                    if resp.status_code == 429:
                        wait = 30
                        print(f"  [S2 rate limited, waiting {wait}s...]")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    data = self._http_get_json(url, timeout=30)

                items = data.get("data", [])
                if not items:
                    break

                for item in items:
                    year = item.get("year") or 0
                    if year_from and year < year_from:
                        continue
                    if year_to and year > year_to:
                        continue
                    authors = [a.get("name", "") for a in item.get("authors", [])]
                    oa = item.get("openAccessPdf") or {}
                    pdf_url = oa.get("url", "")
                    doi = (item.get("externalIds") or {}).get("DOI", "")
                    papers.append(PaperMetadata(
                        title=item.get("title", ""),
                        authors=authors, year=year, doi=doi,
                        abstract=item.get("abstract", "") or "",
                        source="semantic_scholar", pdf_url=pdf_url,
                    ))

                # 检查是否还有更多页
                if len(items) < params["limit"]:
                    break

                time.sleep(1.1)  # 遵守 1 req/s 速率限制

            except Exception as e:
                if "429" in str(e):
                    time.sleep(30)
                    continue
                print(f"  [Semantic Scholar search error] {e}")
                break

        return papers

    def search_crossref(self, query: str, year_from: int = None,
                        year_to: int = None, max_results: int = 50) -> List[PaperMetadata]:
        """通过 Crossref API 搜索论文 (免费, 无需API Key).

        API: https://api.crossref.org/works?query={query}&rows=N
        速率限制: ~50 req/s (公共API)
        """
        base_url = "https://api.crossref.org/works"
        papers = []
        rows = min(max_results, 100)
        offset = 0

        while len(papers) < max_results:
            params = {
                "query": query,
                "rows": rows,
                "offset": offset,
                "filter": "type:journal-article",
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            try:
                data = self._retry_request(url, timeout=30, is_json=True)
                if not data:
                    break

                items = data.get("message", {}).get("items", [])
                if not items:
                    break

                for item in items:
                    year = item.get("created", {}).get("date-parts", [[0]])[0][0]
                    if year_from and year < year_from:
                        continue
                    if year_to and year > year_to:
                        continue

                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else ""

                    authors = []
                    for author in item.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        full = f"{given} {family}".strip()
                        if full:
                            authors.append(full)

                    abstract = item.get("abstract", "")
                    if isinstance(abstract, str) and len(abstract) > 5000:
                        abstract = abstract[:5000] + "..."

                    doi = item.get("DOI", "")
                    link_list = item.get("link", [])
                    pdf_url = ""
                    for link in link_list:
                        if link.get("content-type") == "application/pdf":
                            pdf_url = link.get("URL", "")
                            break

                    papers.append(PaperMetadata(
                        title=title, authors=authors, year=year, doi=doi,
                        abstract=abstract, source="crossref", pdf_url=pdf_url,
                    ))

                offset += rows
                total = data.get("message", {}).get("total-results", 0)
                if offset >= total or offset >= 1000:
                    break

                time.sleep(0.2)
            except Exception as e:
                print(f"  [Crossref search error] {e}")
                break

        return papers

    def search_cnki(self, query: str, year_from: int = None,
                    year_to: int = None, max_results: int = 30) -> List[PaperMetadata]:
        """通过知网(CNKI)搜索中文学术论文.

        知网没有公开API, 通过搜索结果页抓取元数据 (标题/作者/摘要/DOI).
        完整PDF需要用户自行从知网下载后上传到平台分析。

        搜索策略:
        1. 主题搜索 (SU=主题)
        2. 解析页面中的文献条目
        """
        papers = []
        query_encoded = urllib.parse.quote(query)

        # CNKI 新版搜索接口
        search_urls = [
            f"https://kns.cnki.net/kns8s/search?classid=YSTT4HG0&kw={query_encoded}&korder=SU",
            f"https://kns.cnki.net/kns8/defaultresult/index?kwd={query_encoded}",
        ]

        # 尝试多种解析策略
        for base_url in search_urls:
            try:
                content = self._retry_request(base_url, timeout=20)
                if not content or not isinstance(content, bytes):
                    continue
                html = content.decode("utf-8", errors="replace")

                # 从页面提取文献条目
                papers = self._parse_cnki_html(html, year_from, year_to, max_results)
                if papers:
                    break
            except Exception as e:
                print(f"  [CNKI search attempt error] {e}")
                continue

        # 如果网页抓取失败, 尝试知网旧版导出接口
        if not papers:
            papers = self._search_cnki_legacy(query, year_from, year_to, max_results)

        # 去重
        seen = set()
        unique = []
        for p in papers:
            key = p.title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:max_results]

    def _parse_cnki_html(self, html: str, year_from: int, year_to: int,
                         max_results: int) -> List[PaperMetadata]:
        """解析知网搜索结果HTML, 提取文献元数据."""
        papers = []
        import re as _re

        # 模式1: 查找结果列表项 (新版本页面结构)
        # 标题模式: <a class="fz14" ...>Title</a>
        title_pattern = _re.compile(
            r'<a[^>]*?(?:class="[^"]*?(?:title|fz14|fz)[^"]*?")[^>]*?>\s*(.+?)\s*</a>',
            _re.IGNORECASE | _re.DOTALL,
        )
        # 作者模式: <span class="author">...</span> 或 <p class="author">
        author_pattern = _re.compile(
            r'<(?P<tag>span|p)[^>]*?(?:class="[^"]*?author[^"]*?")[^>]*?>\s*(.+?)\s*</(?P=tag)>',
            _re.IGNORECASE | _re.DOTALL,
        )
        # 摘要模式
        abstract_pattern = _re.compile(
            r'<(?P<tag>div|p)[^>]*?(?:class="[^"]*?(?:abstract|summary)[^"]*?")[^>]*?>\s*(.+?)\s*</(?P=tag)>',
            _re.IGNORECASE | _re.DOTALL,
        )
        # 来源/期刊/年份
        source_pattern = _re.compile(
            r'<(?P<tag>span|p)[^>]*?(?:class="[^"]*?(?:source|journal|year)[^"]*?")[^>]*?>\s*(.+?)\s*</(?P=tag)>',
            _re.IGNORECASE | _re.DOTALL,
        )

        titles = title_pattern.findall(html)
        authors_list = author_pattern.findall(html)
        abstracts = abstract_pattern.findall(html)
        sources = source_pattern.findall(html)

        for i, title in enumerate(titles[:max_results]):
            title = title.strip()
            if len(title) < 5:
                continue

            authors = []
            if i < len(authors_list):
                author_text = authors_list[i]
                if isinstance(author_text, tuple):
                    author_text = author_text[1] if len(author_text) > 1 else author_text[0]
                authors = [a.strip() for a in _re.split(r'[;；,]', str(author_text)) if a.strip()]

            abstract = ""
            if i < len(abstracts):
                a = abstracts[i]
                if isinstance(a, tuple):
                    a = a[1] if len(a) > 1 else a[0]
                abstract = str(a).strip()[:2000]

            source_info = ""
            if i < len(sources):
                s = sources[i]
                if isinstance(s, tuple):
                    s = s[1] if len(s) > 1 else s[0]
                source_info = str(s).strip()

            year = 0
            year_match = _re.search(r'(\d{4})', source_info)
            if year_match:
                year = int(year_match.group(1))

            if year_from and year and year < year_from:
                continue
            if year_to and year and year > year_to:
                continue

            papers.append(PaperMetadata(
                title=title, authors=authors, year=year,
                abstract=abstract, source="cnki",
            ))

        return papers

    def _search_cnki_legacy(self, query: str, year_from: int, year_to: int,
                            max_results: int) -> List[PaperMetadata]:
        """知网旧版API回退 (可能已失效)."""
        papers = []
        try:
            legacy_url = (
                f"https://oversea.cnki.net/kcms/detail/detail.aspx?"
                f"dbcode=CJFD&filename=&query={urllib.parse.quote(query)}"
            )
            content = self._retry_request(legacy_url, timeout=15)
            if content:
                # 尽力解析
                pass
        except Exception:
            pass
        return papers

    def import_cnki_export(self, export_text: str) -> List[PaperMetadata]:
        """从知网导出的文献数据批量导入.

        支持格式:
        - EndNote格式 (知网导出 → EndNote)
        - RefWorks格式
        - 知网NoteExpress格式
        - 简单文本列表 (每行一篇, 标题格式)

        用户可从知网批量勾选 → 导出 → 选择EndNote格式 → 粘贴到此处.
        """
        papers = []
        if not export_text.strip():
            return papers

        # 检测格式
        if export_text.strip().startswith("%0 "):
            # EndNote tagged format
            papers = self._parse_endnote_format(export_text)
        elif export_text.strip().startswith("TY  - "):
            # RIS format (RefWorks)
            papers = self._parse_ris_format(export_text)
        elif export_text.strip().startswith("<?xml"):
            # XML format
            papers = self._parse_cnki_xml(export_text)
        elif "RT " in export_text and "SR " in export_text:
            # NoteExpress format
            papers = self._parse_noteexpress_format(export_text)
        else:
            # 简单文本行回退
            papers = self._parse_simple_text_list(export_text)

        for p in papers:
            p.source = "cnki_import"
        return papers

    def _parse_endnote_format(self, text: str) -> List[PaperMetadata]:
        """解析EndNote标记格式."""
        import re as _re
        papers = []
        current = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                if current.get("title"):
                    papers.append(self._dict_to_metadata(current))
                current = {}
                continue
            if line.startswith("%T "):
                current["title"] = line[3:].strip()
            elif line.startswith("%A "):
                authors = current.setdefault("authors", [])
                authors.append(line[3:].strip())
            elif line.startswith("%D "):
                try:
                    current["year"] = int(line[3:].strip())
                except ValueError:
                    pass
            elif line.startswith("%X ") or line.startswith("%! "):
                abstract = current.get("abstract", "")
                if abstract:
                    abstract += " "
                current["abstract"] = abstract + line[3:].strip()
            elif line.startswith("%R "):
                current["doi"] = line[3:].strip()
            elif line.startswith("%J "):
                current["journal"] = line[3:].strip()
        if current.get("title"):
            papers.append(self._dict_to_metadata(current))
        return papers

    def _parse_ris_format(self, text: str) -> List[PaperMetadata]:
        """解析RIS/RefWorks格式."""
        import re as _re
        papers = []
        current = {}
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("TY  - "):
                if current.get("title"):
                    papers.append(self._dict_to_metadata(current))
                current = {"type": line[6:].strip()}
            elif line.startswith("TI  - "):
                current["title"] = line[6:].strip()
            elif line.startswith("AU  - "):
                authors = current.setdefault("authors", [])
                authors.append(line[6:].strip())
            elif line.startswith("PY  - "):
                try:
                    current["year"] = int(line[6:].strip())
                except ValueError:
                    pass
            elif line.startswith("AB  - ") or line.startswith("N2  - "):
                abstract = current.get("abstract", "")
                if abstract:
                    abstract += " "
                current["abstract"] = abstract + line[6:].strip()
            elif line.startswith("DO  - "):
                current["doi"] = line[6:].strip()
        if current.get("title"):
            papers.append(self._dict_to_metadata(current))
        return papers

    def _parse_cnki_xml(self, text: str) -> List[PaperMetadata]:
        """解析知网XML导出格式."""
        papers = []
        try:
            root = ET.fromstring(text)
            for record in root.findall(".//record"):
                title = ""
                title_el = record.find(".//title") or record.find(".//Title")
                if title_el is not None:
                    title = title_el.text or ""

                authors = []
                for au in record.findall(".//author") or record.findall(".//Author"):
                    if au.text:
                        authors.append(au.text.strip())

                year = 0
                year_el = record.find(".//year") or record.find(".//Year")
                if year_el is not None and year_el.text:
                    try:
                        year = int(year_el.text.strip()[:4])
                    except ValueError:
                        pass

                abstract = ""
                abs_el = record.find(".//abstract") or record.find(".//Abstract")
                if abs_el is not None and abs_el.text:
                    abstract = abs_el.text.strip()[:2000]

                doi = ""
                doi_el = record.find(".//doi") or record.find(".//DOI")
                if doi_el is not None and doi_el.text:
                    doi = doi_el.text.strip()

                if title:
                    papers.append(PaperMetadata(
                        title=title, authors=authors, year=year,
                        abstract=abstract, doi=doi, source="cnki_import",
                    ))

        except ET.ParseError:
            pass
        return papers

    def _parse_noteexpress_format(self, text: str) -> List[PaperMetadata]:
        """解析NoteExpress导出格式."""
        import re as _re
        papers = []
        current = {}
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("RT "):
                if current.get("title"):
                    papers.append(self._dict_to_metadata(current))
                current = {}
            elif line.startswith("T1 "):
                current["title"] = line[3:].strip()
            elif line.startswith("A1 "):
                authors = current.setdefault("authors", [])
                authors.append(line[3:].strip())
            elif line.startswith("YR "):
                try:
                    current["year"] = int(line[3:].strip())
                except ValueError:
                    pass
            elif line.startswith("AB "):
                current["abstract"] = line[3:].strip()
        if current.get("title"):
            papers.append(self._dict_to_metadata(current))
        return papers

    def _parse_simple_text_list(self, text: str) -> List[PaperMetadata]:
        """从简单文本列表解析 (备用)."""
        papers = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 10 and line[0] not in ("#", "-", "*"):
                papers.append(PaperMetadata(title=line, source="manual_import"))
        return papers

    def _dict_to_metadata(self, d: dict) -> PaperMetadata:
        return PaperMetadata(
            title=d.get("title", ""),
            authors=d.get("authors", []),
            year=d.get("year", 0),
            doi=d.get("doi", ""),
            abstract=d.get("abstract", ""),
            source=d.get("source", "imported"),
        )

    def search_pubmed(self, query: str, year_from: int = None,
                      year_to: int = None, max_results: int = 50) -> List[PaperMetadata]:
        """通过 PubMed Entrez API 搜索生物材料/医学相关论文.

        API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
        速率限制: 无API Key 3 req/s, 有API Key 10 req/s
        """
        base_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        base_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # 构建查询 (添加材料科学相关过滤)
        full_query = f'("{query}"[All Fields]) AND (english[Language])'
        if year_from and year_to:
            full_query += f' AND ({year_from}:{year_to}[pdat])'

        # Step 1: 搜索获取PubMed IDs
        search_params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": min(max_results, 100),
            "retmode": "json",
            "sort": "relevance",
        }
        search_url = f"{base_search}?{urllib.parse.urlencode(search_params)}"
        papers = []

        try:
            search_data = self._retry_request(search_url, timeout=30, is_json=True)
            if not search_data:
                return papers

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return papers

            # Step 2: 批量获取论文详情
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract",
            }
            fetch_url = f"{base_fetch}?{urllib.parse.urlencode(fetch_params)}"
            fetch_content = self._retry_request(fetch_url, timeout=30)

            if not fetch_content or not isinstance(fetch_content, bytes):
                return papers

            root = ET.fromstring(fetch_content)
            for article in root.findall(".//PubmedArticle"):
                try:
                    medline = article.find(".//MedlineCitation")
                    if medline is None:
                        continue

                    article_info = medline.find(".//Article")
                    if article_info is None:
                        continue

                    title_el = article_info.find(".//ArticleTitle")
                    title = title_el.text.strip() if title_el is not None and title_el.text else ""

                    authors = []
                    for author in article_info.findall(".//Author"):
                        last = author.find("LastName")
                        fore = author.find("ForeName")
                        name_parts = []
                        if last is not None and last.text:
                            name_parts.append(last.text)
                        if fore is not None and fore.text:
                            name_parts.append(fore.text)
                        if name_parts:
                            authors.append(" ".join(name_parts))

                    abstract_el = article_info.find(".//Abstract/AbstractText")
                    abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""

                    year = 0
                    date_el = article_info.find(".//DateCompleted/Year")
                    if date_el is not None and date_el.text:
                        try:
                            year = int(date_el.text)
                        except ValueError:
                            pass

                    doi = ""
                    for eid in article.findall(".//ELocationID"):
                        if eid.attrib.get("EIdType") == "doi" and eid.text:
                            doi = eid.text
                            break

                    pmid = medline.find("PMID")
                    pmid_text = pmid.text if pmid is not None and pmid.text else ""
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC_{pmid_text}/pdf/" if pmid_text else ""

                    papers.append(PaperMetadata(
                        title=title, authors=authors, year=year, doi=doi,
                        abstract=abstract, source="pubmed", pdf_url=pdf_url,
                    ))
                except Exception:
                    continue

        except Exception as e:
            print(f"  [PubMed search error] {e}")

        return papers

    def download_pdf(self, metadata: PaperMetadata) -> Optional[str]:
        """下载单篇PDF, 返回本地路径或None"""
        if not metadata.pdf_url:
            return None
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', metadata.title)[:80]
        local_path = os.path.join(self.download_dir, f"{safe_title}.pdf")
        if os.path.exists(local_path):
            metadata.local_path = local_path
            return local_path
        try:
            if self._http_download_file(metadata.pdf_url, local_path, timeout=60):
                metadata.local_path = local_path
                return local_path
        except Exception as e:
            print(f"  [Download error] {metadata.title[:50]}: {e}")
        return None

    def batch_download(self, query: str, sources: List[str] = None,
                       year_from: int = None, year_to: int = None,
                       api_key: str = None) -> List[str]:
        """编排: 搜索 → 下载 → 返回本地PDF路径列表"""
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "crossref"]
        all_meta = []
        for src in sources:
            if src == "arxiv":
                all_meta.extend(self.search_arxiv(query, year_from, year_to))
            elif src == "semantic_scholar":
                all_meta.extend(
                    self.search_semantic_scholar(query, year_from, year_to, api_key=api_key)
                )
            elif src == "crossref":
                all_meta.extend(self.search_crossref(query, year_from, year_to))
            elif src == "pubmed":
                all_meta.extend(self.search_pubmed(query, year_from, year_to))
            elif src == "cnki":
                cnki_results = self.search_cnki(query, year_from, year_to)
                all_meta.extend(cnki_results)
                if cnki_results:
                    print(f"  [CNKI] 找到 {len(cnki_results)} 篇论文 (PDF需手动下载上传)")
        # 按标题去重
        seen = set()
        unique = []
        for m in all_meta:
            key = m.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(m)
        paths = []
        for i, meta in enumerate(unique):
            print(f"  [{i+1}/{len(unique)}] {meta.title[:60]}...")
            path = self.download_pdf(meta)
            if path:
                paths.append(path)
            time.sleep(0.5)
        print(f"  Downloaded {len(paths)}/{len(unique)} papers to {self.download_dir}")
        return paths

    def list_downloaded(self) -> List[str]:
        """列出已下载的PDF文件路径"""
        if not os.path.exists(self.download_dir):
            return []
        return sorted([
            os.path.join(self.download_dir, f)
            for f in os.listdir(self.download_dir) if f.endswith('.pdf')
        ])
