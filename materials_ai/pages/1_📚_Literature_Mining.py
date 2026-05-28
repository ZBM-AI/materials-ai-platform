"""模块1: NLP文献挖掘 v3 — Streamlit 页面 (6标签)"""

import streamlit as st
import os
import sys
import re
import json
import tempfile
import pandas as pd
from io import StringIO
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.nlp_literature_mining.pdf_parser import PDFParser
from modules.nlp_literature_mining.materials_ner import MaterialsNER
from modules.nlp_literature_mining.relation_extractor import RelationExtractor, Triplet
from modules.nlp_literature_mining.search_engine import LiteratureSearchEngine
from modules.nlp_literature_mining.downloader import PaperDownloader
from modules.nlp_literature_mining.database import TripletStore, PaperStore
from modules.nlp_literature_mining.scibert_ner import SciBERTNER
from modules.nlp_literature_mining.rag_pipeline import MaterialsRAG, HAS_LANGCHAIN
from modules.nlp_literature_mining.report_generator import ReportGenerator
from modules.nlp_literature_mining.paper_analyzer import (
    PaperAnalyzer, DeepAnalysisResult, PaperDiscovery, InnovationPoint, PaperShortcoming,
)
from modules.nlp_literature_mining.paper_comparator import PaperComparator, ComparisonMatrix
from modules.nlp_literature_mining.ner_trainer import (
    generate_seed_bio_data, generate_synthetic_bio_data, render_bio_preview_html, BIO_COLOR_MAP,
)
from utils.data_loader import DataLoader
import config

st.set_page_config(page_title="文献挖掘", page_icon="📚", layout="wide")

st.title("📚 文献挖掘 v3 — 智能文献深度分析平台")
st.markdown("多数据源下载 → 深度解析 → NER → 关系抽取 → 发现/创新/不足 → 多论文对比 → RAG问答")

# 初始化模块
@st.cache_resource
def load_core_modules():
    return {
        "parser": PDFParser(),
        "ner": MaterialsNER(),
        "re": RelationExtractor(),
        "scibert": SciBERTNER(),
    }

core = load_core_modules()

# 侧边栏: DeepSeek API Key
with st.sidebar:
    st.header("⚙️ 全局设置")
    deepseek_key = st.text_input("DeepSeek V4 Pro API Key", type="password",
                                  value=config.DEEPSEEK_API_KEY if config.DEEPSEEK_API_KEY else "",
                                  help="用于LLM增强分析、深度问答和论文智能解读。\n\n"
                                       "获取方式: platform.deepseek.com → API Keys → 创建\n"
                                       "默认使用 deepseek-chat 模型, 兼容 OpenAI SDK。")
    deepseek_model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"],
                                   index=0, help="deepseek-chat: 通用; deepseek-reasoner: 深度推理")
    st.markdown("---")
    st.caption("模块1 · 文献挖掘 v3")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 论文下载", "📄 上传与深度分析", "🧠 SciBERT NER",
    "🔍 文献检索", "🤖 RAG 问答", "📊 报告生成",
])

# ============================================================
# Tab 1: 论文下载 (多数据源: arXiv · Semantic Scholar · Crossref · CNKI · PubMed)
# ============================================================
with tab1:
    st.subheader("批量下载论文 — 多数据源聚合搜索")
    st.caption("支持 arXiv · Semantic Scholar · Crossref · CNKI (知网) · PubMed。自动去重、PDF下载。")

    col1, col2, col3 = st.columns(3)
    with col1:
        query = st.text_input("搜索关键词", "perovskite solar cell",
                              help="支持中英文关键词、材料名称、合成方法等")
    with col2:
        sources = st.multiselect(
            "数据源",
            ["arxiv", "semantic_scholar", "crossref", "pubmed", "cnki"],
            default=["arxiv", "semantic_scholar", "crossref"],
            format_func=lambda x: {
                "arxiv": "arXiv (预印本)",
                "semantic_scholar": "Semantic Scholar",
                "crossref": "Crossref (综合)",
                "pubmed": "PubMed (生物材料)",
                "cnki": "CNKI 知网 (中文)",
            }[x],
        )
    with col3:
        max_papers = st.slider("每数据源最大论文数", 5, 100, 20)

    col_y1, col_y2 = st.columns(2)
    with col_y1:
        year_from = st.number_input("起始年份", 2000, 2026, 2020)
    with col_y2:
        year_to = st.number_input("截止年份", 2000, 2026, 2026)

    if st.button("🔍 多源搜索并下载", type="primary"):
        downloader = PaperDownloader(max_papers=max_papers)
        all_meta = []

        source_map = {
            "arxiv": lambda: downloader.search_arxiv(query, year_from, year_to),
            "semantic_scholar": lambda: downloader.search_semantic_scholar(query, year_from, year_to),
            "crossref": lambda: downloader.search_crossref(query, year_from, year_to, max_papers),
            "pubmed": lambda: downloader.search_pubmed(query, year_from, year_to, max_papers),
            "cnki": lambda: downloader.search_cnki(query, year_from, year_to, max_papers),
        }

        progress = st.progress(0)
        for i, src in enumerate(sources):
            status = st.status(f"正在搜索 {src}...")
            with status:
                try:
                    papers = source_map[src]()
                    all_meta.extend(papers)
                    st.write(f"找到 {len(papers)} 篇")
                except Exception as e:
                    st.error(f"搜索失败: {e}")
            progress.progress((i + 1) / (len(sources) + 1))

        # 去重
        seen = set()
        unique_meta = []
        for m in all_meta:
            key = m.title.lower().strip()[:100]
            if key not in seen:
                seen.add(key)
                unique_meta.append(m)

        progress.progress(1.0)
        st.info(f"去重后共 {len(unique_meta)} 篇论文 (原始 {len(all_meta)} 篇)")

        # 显示搜索结果
        if unique_meta:
            st.markdown("### 搜索结果")
            meta_df = pd.DataFrame([
                {
                    "标题": m.title[:100],
                    "来源": m.source,
                    "年份": m.year,
                    "作者": ", ".join(m.authors[:3]),
                    "可下载": "✓" if m.pdf_url else "✗",
                }
                for m in unique_meta
            ])
            st.dataframe(meta_df, use_container_width=True, hide_index=True)

            # 下载有PDF链接的论文
            downloadable = [m for m in unique_meta if m.pdf_url]
            if downloadable:
                if st.button(f"📥 下载 {len(downloadable)} 篇有PDF链接的论文"):
                    progress2 = st.progress(0)
                    downloaded = []
                    for i, meta in enumerate(downloadable):
                        path = downloader.download_pdf(meta)
                        if path:
                            downloaded.append(path)
                        progress2.progress((i + 1) / len(downloadable))
                    progress2.empty()
                    st.success(f"成功下载 {len(downloaded)}/{len(downloadable)} 篇到 `{config.DOWNLOAD_DIR}`")
            else:
                st.warning("搜索结果中无可直接下载的PDF链接。CNKI/部分期刊需手动下载PDF后上传。")
        progress.empty()

    # ---- CNKI 批量导入 ----
    st.markdown("---")
    with st.expander("📋 CNKI/知网 批量导入 (从知网导出的文献数据)"):
        st.caption("从知网批量勾选论文 → 导出 → 选择 EndNote 或 RefWorks 格式 → 粘贴到下方 → 即可导入元数据。")
        col_cnki1, col_cnki2 = st.columns([3, 1])
        with col_cnki1:
            cnki_export_text = st.text_area(
                "粘贴知网导出的文献数据 (EndNote/RIS/XML格式)",
                height=200,
                placeholder="%0 Journal Article\n%T 论文标题\n%A 作者\n%D 2024\n...",
            )
        with col_cnki2:
            st.caption("**支持格式:**")
            st.caption("- EndNote (.enw)")
            st.caption("- RefWorks/RIS (.ris)")
            st.caption("- NoteExpress")
            st.caption("- XML")
            st.caption("- 简单标题列表")
            st.caption("")
            st.caption("**操作步骤:**")
            st.caption("1. 在知网勾选论文")
            st.caption("2. 点击「导出与分析」")
            st.caption("3. 选择「EndNote」")
            st.caption("4. 复制全部内容粘贴到左侧")

            if st.button("📤 导入元数据", use_container_width=True) and cnki_export_text.strip():
                downloader = PaperDownloader()
                imported = downloader.import_cnki_export(cnki_export_text)
                if imported:
                    # 保存到临时 session state
                    st.session_state.cnki_imported = [
                        {"title": p.title, "authors": ", ".join(p.authors),
                         "year": p.year, "abstract": p.abstract[:200]}
                        for p in imported
                    ]
                    st.success(f"成功导入 {len(imported)} 篇论文元数据. 请在「上传与分析」中上传对应PDF进行深度分析。")

        # 显示已导入的CNKI元数据
        if "cnki_imported" in st.session_state and st.session_state.cnki_imported:
            st.caption(f"最近导入 {len(st.session_state.cnki_imported)} 篇:")
            st.dataframe(pd.DataFrame(st.session_state.cnki_imported), use_container_width=True, hide_index=True)

    # 已下载列表
    downloaded_files = [
        f for f in os.listdir(config.DOWNLOAD_DIR) if f.endswith('.pdf')
    ] if os.path.exists(config.DOWNLOAD_DIR) else []
    if downloaded_files:
        with st.expander(f"已下载论文 ({len(downloaded_files)} 篇)"):
            for f in sorted(downloaded_files):
                st.write(f"- {f}")

# ============================================================
# Tab 2: 上传与深度分析 (多文件 · 发现/创新/不足 · 对比)
# ============================================================
with tab2:
    st.subheader("上传论文 — 深度分析与多论文对比")
    st.caption("支持多篇PDF/TXT上传 · 提取发现/创新点/不足 · 多论文横向对比")

    # 初始化 session state
    if "paper_analyses" not in st.session_state:
        st.session_state.paper_analyses = {}
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "analyzed_entities" not in st.session_state:
        st.session_state.analyzed_entities = {}
    if "analyzed_triplets" not in st.session_state:
        st.session_state.analyzed_triplets = {}

    # ---- 上传区域 ----
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_files = st.file_uploader(
            "上传论文 (支持多篇PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="可一次选择多个文件。CNKI论文请先下载PDF再上传。",
        )

    with col_sample:
        st.write("**样本论文:**")
        sample_files = [f for f in os.listdir(config.PAPERS_DIR) if f.endswith('.txt')]
        selected_sample = st.selectbox(
            "选择样本", ["(不选择)"] + sample_files,
            label_visibility="collapsed",
            key="sample_select",
        )

        st.markdown("---")
        st.write("**分析选项:**")
        has_api = bool(deepseek_key)
        use_llm = st.checkbox("🤖 使用LLM增强分析", value=has_api,
                               help="需要DeepSeek API Key。勾选后分析更准确。")
        if use_llm and not has_api:
            st.warning("⚠️ 请先在左侧边栏输入 DeepSeek API Key")

    # ---- 准备待处理文件列表 ----
    files_to_process = []

    if uploaded_files:
        for uf in uploaded_files:
            suffix = os.path.splitext(uf.name)[1] or '.txt'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                files_to_process.append({"path": tmp.name, "name": uf.name})
    elif selected_sample and selected_sample != "(不选择)":
        files_to_process.append({
            "path": os.path.join(config.PAPERS_DIR, selected_sample),
            "name": selected_sample,
        })

    # ---- 一键分析按钮 ----
    if files_to_process:
        st.markdown("---")
        do_analyze = st.button(
            f"🔍 开始分析 ({len(files_to_process)} 篇论文)",
            type="primary", use_container_width=True,
        )

        if do_analyze:
            all_analyses = []

            for idx, file_info in enumerate(files_to_process):
                fpath = file_info["path"]
                fname = file_info["name"]

                st.markdown(f"### 📄 论文 {idx+1}: {fname}")
                status_container = st.status(f"处理中: {fname}", expanded=True)

                with status_container:
                    # Step 1: 解析
                    st.write("📖 解析PDF...")
                    try:
                        parsed = core["parser"].parse(fpath)
                        st.write(f"✅ 解析完成: {parsed['word_count']:,} 词, "
                                f"{len(parsed.get('sections', []))} 个章节")
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                        try:
                            raw = open(fpath, 'r', encoding='utf-8', errors='replace').read()
                            parsed = core["parser"].parse_text(raw, fname)
                            st.write(f"⚠️ 已用文本模式解析: {parsed['word_count']:,} 词")
                        except Exception:
                            st.error("无法读取文件, 跳过。")
                            continue

                    # Step 2: NER + 关系
                    st.write("🏷️ 提取实体和关系...")
                    try:
                        entities = core["ner"].extract_entities(parsed["raw_text"])
                        st.write(f"✅ 提取 {len(entities)} 个实体")
                    except Exception as e:
                        st.warning(f"NER失败: {e}, 使用空实体列表继续")
                        entities = []

                    try:
                        relations = core["re"].extract_relations(parsed["raw_text"], entities)
                        triplets = core["re"].extract_triplets(parsed["raw_text"], entities)
                        st.write(f"✅ 提取 {len(relations)} 条关系, {len(triplets)} 条三元组")
                    except Exception as e:
                        st.warning(f"关系提取失败: {e}")
                        relations, triplets = [], []

                    # Step 3: 深度分析 (带缓存: 同文件+同API Key不重复调用)
                    st.write("🧠 深度语义分析...")
                    analysis_cache_key = f"deep_analysis_{fname}_{hash(parsed['raw_text'][:500])}_{bool(use_llm)}"
                    if analysis_cache_key in st.session_state:
                        analysis = st.session_state[analysis_cache_key]
                        st.write(f"✅ (缓存) 发现 {len(analysis.discoveries)} 项 / "
                                f"创新点 {len(analysis.innovations)} 项 / "
                                f"不足 {len(analysis.shortcomings)} 项")
                    else:
                        try:
                            analyzer = PaperAnalyzer(
                                api_key=deepseek_key if use_llm else None,
                                model=deepseek_model,
                            )
                            analysis = analyzer.analyze(parsed, entities, relations, triplets)
                            st.session_state[analysis_cache_key] = analysis
                            st.write(f"✅ 发现 {len(analysis.discoveries)} 项 / "
                                    f"创新点 {len(analysis.innovations)} 项 / "
                                    f"不足 {len(analysis.shortcomings)} 项")
                        except Exception as e:
                            st.error(f"深度分析失败: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            analysis = DeepAnalysisResult(paper_title=fname)
                            st.session_state[analysis_cache_key] = analysis

                status_container.update(label=f"完成: {fname}", state="running",
                                        expanded=False)

                # 保存到 session state
                st.session_state.paper_analyses[fname] = analysis
                st.session_state.analyzed_entities[fname] = entities
                st.session_state.analyzed_triplets[fname] = triplets
                all_analyses.append({
                    "title": fname, "analysis": analysis,
                    "entities": entities, "triplets": triplets,
                })

                # ---- 显示分析结果 ----
                col_count1, col_count2, col_count3, col_count4 = st.columns(4)
                col_count1.metric("🔬 发现", len(analysis.discoveries))
                col_count2.metric("💡 创新点", len(analysis.innovations))
                col_count3.metric("⚠️ 不足", len(analysis.shortcomings))
                col_count4.metric("🏷️ 实体", len(entities))

                analysis_tabs = st.tabs([
                    "🔬 科学发现", "💡 创新点", "⚠️ 不足之处",
                    "🖼️ 图片", "🏷️ 实体/三元组", "📝 摘要",
                ])

                with analysis_tabs[0]:
                    st.caption(f"从全文 {parsed['word_count']:,} 词中分析得到:")
                    if analysis.discoveries:
                        for i, d in enumerate(analysis.discoveries, 1):
                            type_icons = {
                                "material": "🧪", "reaction": "⚗️", "phenomenon": "🔍",
                                "property": "📊", "mechanism": "⚙️",
                            }
                            icon = type_icons.get(d.type, "📌")
                            with st.container():
                                st.markdown(f"**{i}. {icon} [{d.type}]** {d.description}")
                                if d.evidence:
                                    with st.expander("原文证据"):
                                        st.caption(d.evidence[:300])
                    else:
                        st.info("未提取到明确的科学发现。\n\n"
                               "→ 建议在侧边栏输入 **DeepSeek API Key** 并勾选「LLM增强分析」获得更精准的结果。")

                with analysis_tabs[1]:
                    if analysis.innovations:
                        for i, inn in enumerate(analysis.innovations, 1):
                            cat_icons = {
                                "method": "🔧", "material": "🧪", "theory": "📐",
                                "application": "🚀", "performance": "📈",
                            }
                            icon = cat_icons.get(inn.category, "💡")
                            st.markdown(f"**{i}. {icon} [{inn.category}]** {inn.description}")
                            if inn.significance:
                                st.caption(f"▸ 意义: {inn.significance}")
                            if inn.evidence:
                                with st.expander("📄 原文依据"):
                                    st.caption(inn.evidence[:400])
                    else:
                        st.info("未提取到明确的创新点。"
                               "→ 建议使用LLM增强分析。")

                with analysis_tabs[2]:
                    if analysis.shortcomings:
                        for i, s in enumerate(analysis.shortcomings, 1):
                            sev_icon = {"major": "🔴", "moderate": "🟡", "minor": "🟢"}
                            icon = sev_icon.get(s.severity, "⚪")
                            st.markdown(f"**{i}. {icon} [{s.severity}] [{s.category}]** {s.description}")
                            if s.evidence:
                                with st.expander("📄 原文依据"):
                                    st.caption(s.evidence[:400])
                    else:
                        st.info("未提取到明确的不足/局限。")

                with analysis_tabs[3]:
                    images = parsed.get("images", [])
                    if images:
                        st.caption(f"从PDF中提取到 **{len(images)}** 张关键图片")
                        cols = st.columns(2)
                        for i, img in enumerate(images[:10]):
                            with cols[i % 2]:
                                img_html = (
                                    f'<img src="data:image/{img["format"]};base64,{img["image_base64"]}"'
                                    f' style="width:100%; border:1px solid #ddd; border-radius:6px; margin:4px 0;"/>'
                                )
                                st.markdown(img_html, unsafe_allow_html=True)
                                cap = img.get("caption") or img.get("nearby_text", "")[:100]
                                st.caption(f"p.{img['page_number']} | {img['width']}×{img['height']}px"
                                          f" | {cap}")
                    else:
                        st.info("未提取到关键图片。")

                with analysis_tabs[4]:
                    ecol1, ecol2 = st.columns(2)
                    with ecol1:
                        st.caption(f"**实体统计 ({len(entities)} 个)**")
                        if entities:
                            type_counts = {}
                            for e in entities:
                                type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
                            for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"- {t}: {c}")
                            st.dataframe(
                                pd.DataFrame([
                                    {"实体": e.text, "类型": e.entity_type}
                                    for e in entities[:20]
                                ]),
                                use_container_width=True, hide_index=True,
                            )
                        else:
                            st.caption("(未提取到实体)")

                    with ecol2:
                        st.caption(f"**三元组 ({len(triplets)} 条)**")
                        if triplets:
                            st.dataframe(
                                pd.DataFrame([
                                    {"材料": t.material, "性能": t.property, "值": t.value}
                                    for t in triplets[:15]
                                ]),
                                use_container_width=True, hide_index=True,
                            )
                            if st.button(f"💾 保存 {len(triplets)} 条三元组到数据库",
                                        key=f"save_{fname[:15]}"):
                                store = TripletStore()
                                pid = store.insert_paper({
                                    "filename": fname,
                                    "raw_text": parsed.get("raw_text", "")[:1000],
                                    "title": parsed.get("metadata", {}).get("title", fname),
                                })
                                store.insert_triplets(pid, [
                                    {"material": t.material, "property": t.property,
                                     "value": t.value, "value_numeric": t.value_numeric,
                                     "evidence": t.evidence, "confidence": t.confidence}
                                    for t in triplets
                                ])
                                st.success(f"已保存! (paper_id: {pid})")
                        else:
                            st.caption("(未提取到三元组)")

                with analysis_tabs[5]:
                    # 原始摘要/全文
                    st.markdown("### 📋 原文摘要")
                    has_abstract = parsed.get("abstract") and len(parsed.get("abstract", "")) > 20
                    if has_abstract:
                        st.markdown(parsed["abstract"][:2000])
                    else:
                        st.text(parsed.get("raw_text", "")[:2000])
                    if parsed.get("metadata", {}).get("title"):
                        st.caption(f"📌 标题: {parsed['metadata']['title']}")

                    # 自动生成的内容总结
                    st.markdown("---")
                    st.markdown("### 🤖 内容总结")
                    if analysis.summary:
                        st.markdown(analysis.summary[:1500])
                    else:
                        # 基于分析结果生成简要总结
                        summary_parts = []
                        if analysis.discoveries:
                            types = set(d.type for d in analysis.discoveries)
                            summary_parts.append(f"本文涉及 {len(analysis.discoveries)} 项科学发现"
                                               f"（{', '.join(sorted(types))}）")
                        if analysis.innovations:
                            cats = set(i.category for i in analysis.innovations)
                            summary_parts.append(f"提出 {len(analysis.innovations)} 个创新点"
                                               f"（{', '.join(sorted(cats))}）")
                        if analysis.shortcomings:
                            cats = set(s.category for s in analysis.shortcomings)
                            summary_parts.append(f"存在 {len(analysis.shortcomings)} 处不足"
                                               f"（{', '.join(sorted(cats))}）")
                        if analysis.keywords:
                            summary_parts.append(f"关键词: {', '.join(analysis.keywords[:10])}")
                        if summary_parts:
                            for sp in summary_parts:
                                st.markdown(f"- {sp}")
                        else:
                            st.caption("请使用LLM增强分析获得更全面的内容总结。")

                    # 三元组快速摘要
                    if triplets and len(triplets) > 0:
                        st.markdown("---")
                        st.markdown("### 🔗 三元组摘要")
                        key_triplets = [t for t in triplets if t.material and t.property][:5]
                        for t in key_triplets:
                            val_str = f" = {t.value}" if t.value else ""
                            st.markdown(f"- **{t.material}** → {t.property}{val_str}")
                    elif not has_abstract:
                        st.caption("未提取到结构化三元组信息。")

                st.markdown("---")

            # ---- 多论文对比分析 (自动触发) ----
            if len(all_analyses) >= 2:
                st.markdown("## 📊 多论文对比分析")
                st.caption(f"对 {len(all_analyses)} 篇论文进行横向对比")

                comparator = PaperComparator(
                    api_key=deepseek_key if use_llm else None,
                    model=deepseek_model,
                )
                with st.spinner("正在进行交叉对比分析..."):
                    try:
                        matrix = comparator.compare(all_analyses)
                        st.session_state.comparison_result = matrix
                    except Exception as e:
                        st.error(f"对比分析失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        matrix = None

                if matrix:
                    comp_tabs = st.tabs([
                        "📋 总览", "🔬 发现对比", "💡 创新热力图",
                        "⚠️ 不足汇总", "⚗️ 方法论对比",
                    ])

                    with comp_tabs[0]:
                        st.markdown("### 综合对比总结")
                        st.markdown(matrix.cross_paper_summary)

                        if matrix.paper_titles:
                            st.write("**对比论文:**")
                            for i, t in enumerate(matrix.paper_titles, 1):
                                st.write(f"{i}. {t}")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**共识点:**")
                            for p in matrix.consensus_points:
                                st.success(p)
                        with c2:
                            st.write("**研究空白:**")
                            for g in matrix.research_gaps:
                                st.warning(g)

                        st.caption(f"趋势: {matrix.trend_analysis}")

                    with comp_tabs[1]:
                        if matrix.common_discoveries:
                            for d in matrix.common_discoveries:
                                st.info(d)
                        if matrix.conflicting_findings:
                            for cf in matrix.conflicting_findings:
                                st.warning(f"[{cf['type']}] 不同论文有不同结论")
                        if matrix.material_overlap:
                            st.write(f"**共同材料:** {', '.join(matrix.material_overlap)}")
                        if matrix.property_overlap:
                            st.write(f"**共同性能:** {', '.join(matrix.property_overlap)}")
                        if not any([matrix.common_discoveries, matrix.conflicting_findings,
                                   matrix.material_overlap, matrix.property_overlap]):
                            st.info("论文间的发现维度差异较大, 未找到明显重叠。")

                    with comp_tabs[2]:
                        table_data = comparator.to_comparison_table(matrix)
                        if table_data["rows"]:
                            st.dataframe(pd.DataFrame(table_data["rows"]),
                                        use_container_width=True)

                    with comp_tabs[3]:
                        for title, sc_list in matrix.shortcoming_summary.items():
                            with st.expander(f"{title} ({len(sc_list)} 项)"):
                                for s in sc_list:
                                    st.caption(f"- {s}")

                    with comp_tabs[4]:
                        for title, method in matrix.methodology_comparison.items():
                            st.write(f"**{title}:** {method}")

            elif len(all_analyses) == 1:
                st.info("📊 仅分析了一篇论文。**再上传一篇**即可自动触发横向对比分析。")

    elif not uploaded_files and selected_sample == "(不选择)":
        st.info("👆 请上传 PDF/TXT 论文文件，然后点击「开始分析」按钮。")
        st.markdown("""
        **使用步骤:**
        1. 上传一篇或多篇论文PDF/TXT
        2. (可选) 在左侧边栏输入 **DeepSeek API Key** → 勾选「LLM增强分析」
        3. 点击 **「开始分析」** 按钮
        4. 查看每篇论文的 **科学发现/创新点/不足** 分析结果
        5. 上传 ≥2 篇后自动生成 **横向对比报告**

        **DeepSeek API Key 获取方式:**
        → 访问 [platform.deepseek.com](https://platform.deepseek.com)
        → 注册/登录 → API Keys → 创建新Key → 复制粘贴到左侧边栏
        """)

# ============================================================
# Tab 3: SciBERT NER (4 sub-tabs)
# ============================================================
with tab3:
    st.subheader("🧠 SciBERT NER — 模型训练与管理")
    st.caption("BIO训练数据生成 · 一键微调SciBERT · 模型评估 · 双引擎对比")

    ner_tab_a, ner_tab_b, ner_tab_c, ner_tab_d = st.tabs([
        "📊 训练数据工坊", "🚀 一键训练", "📈 模型评估", "⚔️ 双引擎对比",
    ])

    # ================================================================
    # Sub-tab A: 训练数据工坊
    # ================================================================
    with ner_tab_a:
        st.markdown("### 训练数据工坊")
        st.caption("管理BIO标注训练数据 — 生成 / 预览 / 导入 / 导出")

        st.markdown("#### 数据来源")
        data_col1, data_col2 = st.columns(2)

        with data_col1:
            st.write("**A1. 从已解析论文生成种子数据**")
            st.caption("使用规则引擎(MaterialsNER)自动标注已解析论文，生成BIO训练数据。")
            max_papers_a1 = st.slider("处理论文数", 1, 50, 10, key="ner_a1_n")
            if st.button("🔄 生成种子数据", key="ner_gen_seed", use_container_width=True):
                with st.spinner("从论文生成BIO标注数据..."):
                    papers = DataLoader.load_parsed_papers()
                    if not papers:
                        st.warning("未找到已解析论文。请先在「上传与分析」中处理论文。")
                    else:
                        n_records, tag_counts, seed_path = generate_seed_bio_data(
                            core["ner"], papers, max_papers_a1,
                        )
                        st.success(f"已生成 {n_records} 条BIO标注 → `{seed_path}`")
                        tag_df = pd.DataFrame(
                            [{"BIO标签": k, "数量": v} for k, v in sorted(tag_counts.items())]
                        )
                        st.bar_chart(tag_df.set_index("BIO标签"))

        with data_col2:
            st.write("**A2. 生成合成训练数据**")
            st.caption("使用词典模板生成大量合成BIO数据，扩充训练集。")
            n_synth = st.slider("合成样本数", 100, 5000, 1000, 100, key="ner_a2_n")
            if st.button("🎲 生成合成数据", key="ner_gen_synth", use_container_width=True):
                with st.spinner("生成合成BIO标注数据..."):
                    n_records, tag_counts, synth_path = generate_synthetic_bio_data(n_synth)
                    st.success(f"已生成 {n_records} 条合成BIO数据 → `{synth_path}`")

        # 数据文件预览
        st.markdown("---")
        st.markdown("#### 现有训练数据文件")
        os.makedirs(config.NER_TRAINING_DIR, exist_ok=True)
        jsonl_files = [f for f in os.listdir(config.NER_TRAINING_DIR) if f.endswith(".jsonl")]
        if jsonl_files:
            sel_file = st.selectbox("选择文件预览", jsonl_files, key="ner_preview_file")
            if sel_file:
                fpath = os.path.join(config.NER_TRAINING_DIR, sel_file)
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                st.write(f"**{sel_file}**: {len(lines)} 条标注数据")

                # 标签分布
                tag_counts = {}
                for line in lines:
                    rec = json.loads(line)
                    for tag in rec.get("tags", []):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                if tag_counts:
                    st.bar_chart(pd.DataFrame(
                        [{"Tag": k, "Count": v} for k, v in sorted(tag_counts.items()) if k != "O"]
                    ).set_index("Tag"))

                # BIO可视化预览
                st.markdown("**标注预览 (前10条):**")
                for i, line in enumerate(lines[:10]):
                    rec = json.loads(line)
                    st.markdown(
                        render_bio_preview_html(rec.get("tokens", []), rec.get("tags", [])),
                        unsafe_allow_html=True,
                    )
        else:
            st.info(f"训练数据目录为空 (`{config.NER_TRAINING_DIR}`)。请先生成训练数据。")

    # ================================================================
    # Sub-tab B: 一键训练
    # ================================================================
    with ner_tab_b:
        st.markdown("### 一键训练 SciBERT NER")
        st.caption("配置参数 → 一键微调SciBERT进行材料科学实体识别")

        # 检查训练数据
        os.makedirs(config.NER_TRAINING_DIR, exist_ok=True)
        jsonl_files = [f for f in os.listdir(config.NER_TRAINING_DIR) if f.endswith(".jsonl")]
        has_data = len(jsonl_files) > 0

        if not has_data:
            st.warning("请先在「训练数据工坊」中生成或导入训练数据。")
        else:
            total_records = 0
            for jf in jsonl_files:
                with open(os.path.join(config.NER_TRAINING_DIR, jf), "r", encoding="utf-8") as f:
                    total_records += sum(1 for _ in f)
            st.success(f"已就绪: {len(jsonl_files)} 个数据文件, 共 {total_records} 条标注数据")

        st.markdown("---")
        st.markdown("#### 训练参数")

        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            epochs = st.number_input("训练轮数", 1, 20, config.SCIBERT_NUM_EPOCHS, key="ner_epochs")
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="ner_batch")
        with param_col2:
            lr = st.select_slider("Learning Rate",
                                  options=[5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
                                  value=config.SCIBERT_LEARNING_RATE, key="ner_lr")
            max_len = st.selectbox("Max Sequence Length", [128, 256, 512], index=2, key="ner_maxlen")
        with param_col3:
            val_split = st.slider("验证集比例", 0.05, 0.30, 0.15, 0.05, key="ner_val")
            save_model = st.checkbox("训练后保存模型", True, key="ner_save")

        st.markdown("---")
        train_col1, train_col2 = st.columns([1, 2])
        with train_col1:
            train_btn = st.button("🚀 开始训练", type="primary", use_container_width=True,
                                  disabled=not has_data)

        if train_btn and has_data:
            with st.spinner("正在训练 SciBERT NER 模型... (可能需要数分钟到数十分钟)"):
                try:
                    from modules.nlp_literature_mining.ner_trainer import SciBERTNERTrainer

                    # 加载所有JSONL数据
                    all_data = []
                    for jf in jsonl_files:
                        fpath = os.path.join(config.NER_TRAINING_DIR, jf)
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    all_data.append(json.loads(line))

                    st.info(f"加载 {len(all_data)} 条数据，开始训练...")

                    trainer = SciBERTNERTrainer(
                        model_name=config.SCIBERT_MODEL_NAME,
                        output_dir=config.NER_MODEL_DIR if save_model else "saved_models/scibert_ner_temp",
                        max_length=max_len,
                        batch_size=batch_size,
                        learning_rate=lr,
                        num_epochs=epochs,
                        tag_to_id=config.TAG_TO_ID,
                    )

                    # 训练 (在后台执行)
                    trainer.train(train_data=all_data, val_split=val_split, save_best=save_model)

                    st.success(f"训练完成! 模型已保存到 `{config.NER_MODEL_DIR}`")

                    # 评估
                    split_idx = max(1, int(len(all_data) * val_split))
                    test_data = all_data[-split_idx:] if split_idx > 0 else all_data[-max(1, len(all_data)//10):]
                    metrics = trainer.evaluate(test_data)
                    if isinstance(metrics, dict):
                        st.markdown("#### 训练结果 (验证集)")
                        eval_rows = []
                        for label, vals in metrics.items():
                            if isinstance(vals, dict) and vals.get("support", 0) > 0:
                                eval_rows.append({
                                    "实体类型": label,
                                    "Precision": f"{vals['precision']:.3f}",
                                    "Recall": f"{vals['recall']:.3f}",
                                    "F1-Score": f"{vals['f1-score']:.3f}",
                                    "Support": vals["support"],
                                })
                        if eval_rows:
                            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

                    # 重新加载core模块的SciBERT以使用新模型
                    st.cache_resource.clear()
                    st.success("模型已就绪! 切换到「双引擎对比」标签测试新模型。")

                except ImportError as e:
                    st.error(f"缺少依赖: {e}\n\n请安装: `pip install torch transformers datasets seqeval`")
                except Exception as e:
                    st.error(f"训练失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # 训练历史 (session state)
        if "ner_training_history" not in st.session_state:
            st.session_state.ner_training_history = []

    # ================================================================
    # Sub-tab C: 模型评估
    # ================================================================
    with ner_tab_c:
        st.markdown("### 模型评估")
        st.caption("加载已训练模型 → 上传测试集 → 评估P/R/F1")

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.write("**模型状态**")
            model_exists = os.path.exists(config.NER_MODEL_DIR) and os.path.isdir(config.NER_MODEL_DIR)
            if model_exists:
                st.success(f"✅ 模型已就绪: `{config.NER_MODEL_DIR}`")
            else:
                st.warning("⚠️ 未找到已训练模型。请先在「一键训练」中训练模型。")

            st.write("**测试数据**")
            test_data_option = st.radio("选择测试数据来源",
                                       ["使用验证集 (从训练数据中划分)", "上传测试文件 (JSONL)"],
                                       key="ner_test_src")
            test_data = None

            if "使用验证集" in test_data_option:
                jsonl_files = [f for f in os.listdir(config.NER_TRAINING_DIR) if f.endswith(".jsonl")]
                if jsonl_files:
                    st.caption(f"将从 {len(jsonl_files)} 个训练文件中划分 {int(val_split*100)}% 作为测试集")
                    test_data = []
                    for jf in jsonl_files:
                        with open(os.path.join(config.NER_TRAINING_DIR, jf), "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    test_data.append(json.loads(line))
                    split_idx = max(1, int(len(test_data) * 0.15))
                    test_data = test_data[-split_idx:]
                    st.caption(f"测试集: {len(test_data)} 条")
            else:
                uploaded_test = st.file_uploader("上传JSONL测试文件", type=["jsonl"], key="ner_test_upload")
                if uploaded_test:
                    test_data = []
                    for line in uploaded_test.read().decode("utf-8").split("\n"):
                        if line.strip():
                            test_data.append(json.loads(line))
                    st.caption(f"已加载 {len(test_data)} 条测试数据")

        with eval_col2:
            st.write("**评估结果**")
            if st.button("📊 运行评估", type="primary", use_container_width=True,
                        disabled=not model_exists or not test_data):
                with st.spinner("评估中..."):
                    try:
                        from modules.nlp_literature_mining.ner_trainer import SciBERTNERTrainer
                        trainer = SciBERTNERTrainer(
                            model_name=config.SCIBERT_MODEL_NAME,
                            output_dir=config.NER_MODEL_DIR,
                            tag_to_id=config.TAG_TO_ID,
                        )
                        # 手动加载模型
                        from transformers import AutoTokenizer, AutoModelForTokenClassification
                        trainer.tokenizer = AutoTokenizer.from_pretrained(config.NER_MODEL_DIR)
                        trainer.model = AutoModelForTokenClassification.from_pretrained(config.NER_MODEL_DIR)
                        trainer.id_to_tag = {v: k for k, v in trainer.tag_to_id.items()}

                        metrics = trainer.evaluate(test_data)
                        if isinstance(metrics, dict):
                            eval_rows = []
                            for label, vals in metrics.items():
                                if isinstance(vals, dict) and vals.get("support", 0) > 0:
                                    f1 = vals["f1-score"]
                                    color = "🟢" if f1 >= 0.8 else ("🟡" if f1 >= 0.6 else "🔴")
                                    eval_rows.append({
                                        "": color,
                                        "Entity": label,
                                        "Precision": f"{vals['precision']:.3f}",
                                        "Recall": f"{vals['recall']:.3f}",
                                        "F1": f"{f1:.3f}",
                                        "Samples": vals["support"],
                                    })
                            if eval_rows:
                                st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
                                # Overall scores
                                overall = [r for r in eval_rows if r["Entity"] in ("micro avg", "macro avg", "weighted avg")]
                                if not overall:
                                    overall = [r for r in eval_rows if "avg" in r["Entity"].lower()]
                                for o in overall:
                                    st.metric(
                                        label=f"Overall ({o['Entity']})",
                                        value=f"F1={o['F1']}",
                                        delta=f"P={o['Precision']} R={o['Recall']}",
                                    )
                    except Exception as e:
                        st.error(f"评估失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # ================================================================
    # Sub-tab D: 双引擎对比
    # ================================================================
    with ner_tab_d:
        st.markdown("### 双引擎对比: 规则引擎 vs SciBERT")
        st.caption("同一段文本分别用规则引擎和SciBERT模型提取实体，直观对比差异。")

        comp_text = st.text_area(
            "输入测试文本:",
            "TiO2 nanoparticles were synthesized via sol-gel method. "
            "The band gap of the resulting material was measured to be 3.2 eV, "
            "and it exhibited a perovskite crystal structure suitable for solar cell applications. "
            "The FeCoNiCrMn high-entropy alloy showed excellent tensile strength of 500 MPa.",
            height=130,
            key="ner_comp_text",
        )

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("#### 🔧 规则引擎 (MaterialsNER)")
            if st.button("运行规则引擎", key="ner_run_rule", use_container_width=True):
                with st.spinner("规则引擎提取中..."):
                    rule_entities = core["ner"].extract_entities(comp_text)
                    if rule_entities:
                        # Highlight text
                        colored_text = comp_text
                        color_map = {
                            "material": "#FF6B6B", "property": "#45B7D1",
                            "property_value": "#96CEB4", "synthesis_method": "#4ECDC4",
                            "microstructure": "#FFEAA7", "crystal_structure": "#FFEAA7",
                            "application": "#DDA0DD",
                        }
                        for ent in sorted(rule_entities, key=lambda e: e.start_char, reverse=True):
                            bg = color_map.get(ent.entity_type, "#CCC")
                            span = f'<span style="background:{bg};padding:1px 3px;border-radius:3px" title="{ent.entity_type}">{comp_text[ent.start_char:ent.end_char]}</span>'
                            colored_text = colored_text[:ent.start_char] + span + colored_text[ent.end_char:]
                        st.markdown(
                            f'<div style="font-size:14px;line-height:2">{colored_text}</div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            pd.DataFrame([
                                {"Entity": e.text, "Type": e.entity_type}
                                for e in rule_entities
                            ]),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("规则引擎未提取到实体。")

        with comp_col2:
            st.markdown("#### 🤖 SciBERT 模型")
            model_ready = os.path.exists(config.NER_MODEL_DIR)
            if not model_ready:
                st.warning("未找到已训练模型，将回退到规则引擎。")

            if st.button("运行 SciBERT", key="ner_run_scibert",
                        use_container_width=True, disabled=not model_ready):
                with st.spinner("SciBERT推理中..."):
                    scibert = SciBERTNER()
                    scibert_entities = scibert.extract_entities(comp_text)
                    engine_label = "SciBERT 微调模型" if not scibert.use_fallback else "规则引擎 (回退)"
                    st.caption(f"引擎: {engine_label}")
                    if scibert_entities:
                        colored_text = comp_text
                        color_map = {
                            "material": "#FF6B6B", "property": "#45B7D1",
                            "property_value": "#96CEB4", "synthesis_method": "#4ECDC4",
                            "microstructure": "#FFEAA7", "crystal_structure": "#FFEAA7",
                            "application": "#DDA0DD",
                        }
                        for ent in sorted(scibert_entities, key=lambda e: e.start_char, reverse=True):
                            bg = color_map.get(ent.entity_type, "#CCC")
                            span = f'<span style="background:{bg};padding:1px 3px;border-radius:3px" title="{ent.entity_type}">{comp_text[ent.start_char:ent.end_char]}</span>'
                            colored_text = colored_text[:ent.start_char] + span + colored_text[ent.end_char:]
                        st.markdown(
                            f'<div style="font-size:14px;line-height:2">{colored_text}</div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            pd.DataFrame([
                                {"Entity": e.text, "Type": e.entity_type}
                                for e in scibert_entities
                            ]),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("SciBERT未提取到实体。")

        # 对比统计
        st.markdown("---")
        st.markdown("**对比说明:**")
        st.caption("""
        - **规则引擎**: 基于词典+正则匹配，速度快但覆盖有限，无法识别新实体变体
        - **SciBERT**: 基于深度学习语义理解，可识别训练数据中未见过的实体表达
        - 训练SciBERT模型后，左侧规则引擎和右侧SciBERT会给出不同结果，直观展示差异
        - 颜色图例: 🔴材料 🟢合成方法 🔵性能参数 🟢数值 🟡微观结构 🟣应用
        """)

# ============================================================
# Tab 4: 文献检索
# ============================================================
with tab4:
    st.subheader("文献检索")

    parsed_papers = DataLoader.load_parsed_papers()
    if not parsed_papers:
        st.warning("未找到预解析论文数据。请先在「上传与分析」中处理论文。")
    else:
        search_engine = LiteratureSearchEngine(parsed_papers)

        col_kw, col_type = st.columns([2, 1])
        with col_kw:
            keyword = st.text_input("关键词搜索", placeholder="如: TiO2, band gap, perovskite...")
        with col_type:
            search_type = st.selectbox(
                "实体类型", ["全部"] + config.ENTITY_TYPES,
            )

        if keyword:
            etype = None if search_type == "全部" else search_type
            results = search_engine.search_entities(entity_type=etype, keyword=keyword)
            if results:
                st.success(f"找到 {len(results)} 条结果")
                st.dataframe(
                    [{"Material/Entity": r["entity"]["text"],
                      "Type": r["entity"]["entity_type"],
                      "Paper": r["filename"]}
                     for r in results],
                    use_container_width=True,
                )
            else:
                st.info("未找到匹配结果。")

    # 三元组查询
    st.markdown("---")
    st.subheader("三元组查询")
    col_m, col_p, col_v = st.columns(3)
    with col_m:
        q_mat = st.text_input("材料名称", placeholder="TiO2")
    with col_p:
        q_prop = st.text_input("性能参数", placeholder="band gap")
    with col_v:
        q_limit = st.number_input("返回条数", 10, 200, 50)

    if st.button("查询三元组"):
        store = TripletStore()
        results = store.query_triplets(
            material=q_mat or None,
            property=q_prop or None,
            limit=q_limit,
        )
        if results:
            st.success(f"找到 {len(results)} 条三元组")
            st.dataframe([
                {"Material": r.get("material", ""),
                 "Property": r.get("property", ""),
                 "Value": r.get("value", ""),
                 "Confidence": f"{r.get('confidence', 0):.0%}",
                 "Paper": r.get("paper_id", "")}
                for r in results
            ], use_container_width=True)
        else:
            st.info("未找到匹配的三元组。")

# ============================================================
# Tab 5: RAG 问答
# ============================================================
with tab5:
    st.subheader("RAG 文献问答")
    st.caption("基于向量检索 + LLM 的私有文献库问答 (LangChain + Chroma)。")

    if not HAS_LANGCHAIN:
        st.warning("LangChain 未安装。运行: `pip install langchain langchain-community chromadb sentence-transformers`")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**索引状态**")
            if os.path.exists(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
                st.success(f"Chroma索引已存在: `{config.CHROMA_DIR}`")
                try:
                    rag = MaterialsRAG().load_index()
                    stats = rag.get_stats()
                    st.metric("已索引文档数", stats.get("document_count", 0))
                except Exception as e:
                    st.warning(f"加载索引失败: {e}")
                    rag = MaterialsRAG()
            else:
                st.info("尚未建立索引。请先导入论文数据。")
                rag = MaterialsRAG()

        with col2:
            st.write("**建立/更新索引**")
            use_parsed = st.checkbox("使用已解析论文", value=True)
            if st.button("构建索引"):
                papers = []
                if use_parsed:
                    parsed = DataLoader.load_parsed_papers()
                    papers = [{"paper_id": p.get("filename", str(i)),
                               "title": p.get("title", ""),
                               "raw_text": p.get("raw_text", "")}
                              for i, p in enumerate(parsed)]

                if not papers:
                    st.warning("没有可索引的论文数据。")
                else:
                    with st.spinner(f"正在索引 {len(papers)} 篇论文..."):
                        try:
                            count = rag.index_papers(papers)
                            st.success(f"索引完成! {count} 个文本块已存储。")
                        except Exception as e:
                            st.error(f"索引失败: {e}")

        st.markdown("---")
        st.subheader("提问")

        question = st.text_input(
            "输入你的问题:",
            placeholder="如: 2023年后有哪些提升铝合金强度的新工艺?",
        )
        col_q, _ = st.columns([1, 3])
        with col_q:
            ask_button = st.button("提问", type="primary", disabled=not question)

        if ask_button and question:
            if not rag.is_indexed:
                st.error("请先构建索引。")
            else:
                with st.spinner("检索中..."):
                    try:
                        result = rag.ask(question, k=config.RAG_TOP_K)
                        st.markdown("### 回答")
                        st.write(result["answer"])
                        st.markdown("### 来源")
                        for i, src in enumerate(result.get("sources", []), 1):
                            st.caption(f"{i}. [{src.get('title', 'N/A')}] — {src.get('excerpt', '')[:150]}...")
                    except Exception as e:
                        st.error(f"问答失败: {e}")
                        st.info("提示: RAG问答需要 OpenAI API Key。设置环境变量 `OPENAI_API_KEY`。")

# ============================================================
# Tab 6: 报告生成
# ============================================================
with tab6:
    st.subheader("文献综述 & 趋势分析")
    st.caption("从三元组数据自动生成综述草稿和可视化图表。")

    generator = ReportGenerator()

    col_topic, col_gen = st.columns([3, 1])
    with col_topic:
        topic = st.text_input("综述主题", "Advanced Materials for Energy Applications")
    with col_gen:
        st.write("")
        st.write("")
        gen_button = st.button("📝 生成报告", type="primary")

    if gen_button:
        store = TripletStore()
        paper_store = PaperStore()
        triplets = store.query_triplets(limit=500)
        papers = paper_store.list_all()

        if not triplets:
            st.warning("数据库中暂无三元组数据。请先在「上传与分析」中提取并保存三元组。")
        else:
            with st.spinner("正在生成报告..."):
                report = generator.generate_full_report(triplets, papers, topic)

            st.markdown("### 综述草稿")
            st.markdown(report["review"][:5000])
            if len(report["review"]) > 5000:
                st.caption(f"... (全文共 {len(report['review'])} 字符)")

            st.markdown("---")
            st.markdown("### 趋势分析图")

            chart_tabs = st.tabs(["研究趋势", "材料分布", "性能分布"])
            with chart_tabs[0]:
                if report.get("trend_chart"):
                    try:
                        fig = json.loads(json.dumps(report["trend_chart"]))
                        import plotly.graph_objects as go
                        st.plotly_chart(go.Figure(data=fig.get("data", []),
                                                   layout=fig.get("layout", {})),
                                        use_container_width=True)
                    except Exception:
                        st.info("暂无趋势数据。")
                else:
                    st.info("需要论文年份元数据。")

            with chart_tabs[1]:
                if report.get("material_distribution"):
                    try:
                        fig = json.loads(json.dumps(report["material_distribution"]))
                        import plotly.graph_objects as go
                        st.plotly_chart(go.Figure(data=fig.get("data", []),
                                                   layout=fig.get("layout", {})),
                                        use_container_width=True)
                    except Exception:
                        st.info("暂无材料分布数据。")

            with chart_tabs[2]:
                if report.get("property_distribution"):
                    try:
                        fig = json.loads(json.dumps(report["property_distribution"]))
                        import plotly.graph_objects as go
                        st.plotly_chart(go.Figure(data=fig.get("data", []),
                                                   layout=fig.get("layout", {})),
                                        use_container_width=True)
                    except Exception:
                        st.info("暂无性能分布数据。")

            col_save, _ = st.columns([1, 3])
            with col_save:
                if st.button("💾 保存报告到文件"):
                    path = generator.save_report(report)
                    st.success(f"报告已保存至: `{path}`")

    # 数据库统计
    st.markdown("---")
    st.subheader("数据库统计")
    store = TripletStore()
    stats = store.get_statistics()
    c1, c2, c3 = st.columns(3)
    c1.metric("三元组总数", stats.get("total_triplets", 0))
    c2.metric("唯一材料", stats.get("unique_materials", 0))
    c3.metric("唯一性能", stats.get("unique_properties", 0))
