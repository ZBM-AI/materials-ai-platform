"""模块6: 智能学习与问答助手 — Streamlit 页面"""

import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.learning_assistant.knowledge_base import TextbookLoader, KnowledgeBase
from modules.learning_assistant.qa_engine import MaterialsQA
from modules.learning_assistant.quiz_generator import QuizGenerator, Quiz
from modules.learning_assistant.code_assistant import CodeAssistant
from modules.learning_assistant.experiment_advisor import ExperimentAdvisor
import config

st.set_page_config(page_title="智能学习助手", page_icon="🎓", layout="wide")

st.title("🎓 智能学习与问答助手")
st.markdown("RAG教材知识库 · 概念问答 · 自动出题 · Python代码辅助 · 实验参数建议")

# 缓存清理: 限制API结果缓存数量, 防止session_state无限增长
_MAX_CACHED_QA = 30
_qa_cache_keys = sorted([k for k in st.session_state if k.startswith("qa_")])
for _old_key in _qa_cache_keys[:-_MAX_CACHED_QA] if len(_qa_cache_keys) > _MAX_CACHED_QA else []:
    del st.session_state[_old_key]
_quiz_cache_keys = sorted([k for k in st.session_state if k.startswith("quiz_")])
for _old_key in _quiz_cache_keys[:-_MAX_CACHED_QA] if len(_quiz_cache_keys) > _MAX_CACHED_QA else []:
    del st.session_state[_old_key]

# ---- 侧边栏 ----
with st.sidebar:
    st.header("⚙️ 配置")

    st.subheader("API (可选)")
    deepseek_key = st.text_input("DeepSeek API Key", type="password",
                                  value=config.OPENAI_API_KEY if config.OPENAI_API_KEY else "",
                                  help="用于LLM增强问答和出题。\n获取方式: platform.deepseek.com → API Keys → 创建")
    deepseek_model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"],
                                   index=0, help="deepseek-chat: 通用对话; deepseek-reasoner: 深度推理")

    if not deepseek_key:
        st.caption("💡 在 [platform.deepseek.com](https://platform.deepseek.com) 免费注册获取API Key")

    st.subheader("教材管理")
    loader = TextbookLoader()
    available = loader.get_available_textbooks()

    if available:
        st.write(f"已发现 **{len(available)}** 本教材:")
        for tb in available:
            st.caption(f"- {tb}")
    else:
        st.warning("未找到教材PDF")
        st.caption(f"请将PDF放入: `{loader.textbooks_dir}`")

    if st.button("🔄 重新构建知识库", use_container_width=True):
        with st.spinner("正在解析教材并构建向量索引..."):
            pages = loader.load_all_textbooks()
            if pages:
                st.cache_resource.clear()  # 清除所有缓存资源
                st.session_state.pop("kb", None)
                st.session_state.kb = KnowledgeBase()
                st.session_state.kb.build_from_pages(pages)
                stats = st.session_state.kb.get_statistics()
                st.success(f"已索引 {len(pages)} 页, {stats['total_chunks']} 个文本块")
            else:
                st.warning("未能解析任何PDF页面")
        st.rerun()

    st.markdown("---")
    st.caption("模块6 · 材料科学AI学习助手")

# ---- 初始化知识库 ----
@st.cache_resource
def get_cached_kb():
    """延迟初始化: 先不加载 embedding 模型, 等需要时再加载."""
    kb = KnowledgeBase()
    # 只尝试加载索引, 不触发模型下载
    _ = kb.load_index()
    return kb

if "kb" not in st.session_state:
    with st.spinner("正在加载知识库..."):
        st.session_state.kb = get_cached_kb()

kb = st.session_state.kb
kb_ready = kb.is_ready

qa = MaterialsQA(knowledge_base=kb, api_key=deepseek_key if deepseek_key else None)
quiz_gen = QuizGenerator(knowledge_base=kb, api_key=deepseek_key if deepseek_key else None)
code_ast = CodeAssistant(api_key=deepseek_key if deepseek_key else None)
exp_advisor = ExperimentAdvisor()

# ---- 状态栏 ----
if kb_ready:
    stats = kb.get_statistics()
    st.success(f"知识库就绪: {stats['total_chunks']} 个文本块 | {len(stats['sources'])} 本教材 | {len(stats['chapters'])} 个章节")
else:
    st.warning("知识库未构建。请在侧边栏上传教材PDF并点击「重新构建知识库」。")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 概念问答", "📝 自动出题", "🐍 Python代码", "🔬 实验建议", "📊 知识库统计",
])

# ---- Tab 1: 概念问答 ----
with tab1:
    st.subheader("材料科学概念问答")
    st.caption("基于教材知识库的RAG问答 · 支持来源追溯 (页码/章节)")

    q_col1, q_col2, q_col3 = st.columns([5, 2, 1])
    with q_col1:
        question = st.text_input(
            "提问", placeholder="例如: 请解释霍尔-佩奇关系 | 位错攀移机制 | 什么是柯肯达尔效应?",
            label_visibility="collapsed",
        )
    with q_col2:
        q_mode = st.selectbox("回答模式", ["本地检索", "LLM增强 (需API Key)"],
                               label_visibility="collapsed")
    with q_col3:
        ask_btn = st.button("提问", type="primary", use_container_width=True)

    # 快捷提问
    st.markdown("**快捷提问:**")
    quick_questions = [
        "解释霍尔-佩奇 (Hall-Petch) 关系",
        "位错攀移 (climb) 的机制是什么?",
        "什么是柯肯达尔 (Kirkendall) 效应?",
        "FCC和BCC晶体的区别?",
        "Fe-C相图的共析反应",
        "细晶强化的原理",
        "菲克第一定律和第二定律的区别",
        "马氏体相变的特点",
    ]
    qcols = st.columns(4)
    for i, qq in enumerate(quick_questions):
        with qcols[i % 4]:
            if st.button(qq, key=f"qq_{i}"):
                question = qq
                ask_btn = True
                st.session_state._question = qq

    if ask_btn and question:
        use_llm = "LLM" in q_mode and deepseek_key
        cache_key = f"qa_{question}_{use_llm}_{deepseek_key[:8] if deepseek_key else 'nokey'}"

        # 检查缓存: 相同问题+相同模式不重复调用API
        if cache_key not in st.session_state:
            with st.spinner("检索教材内容中..."):
                st.session_state[cache_key] = qa.ask(question, k=5, use_llm=use_llm)

        result = st.session_state[cache_key]

        st.markdown("---")
        st.markdown(result.answer)

        if result.sources:
            st.markdown("---")
            st.markdown("**参考来源:**")
            src_data = []
            for i, src in enumerate(result.sources, 1):
                src_data.append({
                    "#": i,
                    "教材": src.get("source", "?"),
                    "页码": src.get("page", "?"),
                    "章节": src.get("chapter", ""),
                    "相关度": src.get("relevance", 0),
                })
            st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

        if result.follow_up_suggestions:
            st.markdown("**💡 你可能还想问:**")
            for sug in result.follow_up_suggestions:
                st.caption(f"- {sug}")

# ---- Tab 2: 自动出题 ----
with tab2:
    st.subheader("自动生成练习题")
    st.caption("基于教材章节自动出题 · 选择题 + 计算题 + 详细解答")

    quiz_col1, quiz_col2, quiz_col3 = st.columns([3, 1, 1])
    with quiz_col1:
        quiz_topic = st.text_input("考察知识点", placeholder="位错与塑性变形 / 相图 / 扩散 / 晶体结构")
    with quiz_col2:
        num_mcq = st.slider("选择题数", 3, 20, 8)
    with quiz_col3:
        num_calc = st.slider("计算题数", 0, 5, 2)

    quiz_mode = st.radio("出题模式", ["内置题库 (即时)", "LLM生成 (需API Key)"],
                          horizontal=True, index=0)

    if st.button("🎲 生成试卷", type="primary"):
        use_llm_quiz = "LLM" in quiz_mode and deepseek_key
        quiz_cache_key = f"quiz_{quiz_topic}_{num_mcq}_{num_calc}_{use_llm_quiz}"

        if quiz_cache_key not in st.session_state:
            with st.spinner("出题中..."):
                st.session_state[quiz_cache_key] = quiz_gen.generate_quiz(
                    topic=quiz_topic, num_mcq=num_mcq, num_calc=num_calc,
                    use_llm=use_llm_quiz,
                )

        quiz = st.session_state[quiz_cache_key]

        if quiz.questions:
            st.success(f"已生成 {len(quiz.questions)} 道题目 (满分 {quiz.total_points} 分, 建议时间 {quiz.duration_minutes} 分钟)")

            # 答题模式
            st.markdown("---")
            st.markdown("### 📋 开始答题")

            user_answers = {}
            for i, q in enumerate(quiz.questions):
                with st.expander(f"**第{i+1}题** [{q.question_type.upper()}] {q.difficulty} ({q.points}分)", expanded=(i < 3)):
                    st.markdown(q.question)

                    if q.question_type in ("mcq", "true_false"):
                        if q.options:
                            user_answers[i] = st.radio(
                                f"选择答案 #{i+1}",
                                q.options,
                                key=f"ans_{i}",
                                index=None,
                            )
                    elif q.question_type == "calculation":
                        user_answers[i] = st.text_area(
                            f"你的计算过程/答案 #{i+1}",
                            key=f"calc_{i}",
                            placeholder="输入计算过程和最终答案...",
                        )

                    # 显示答案按钮
                    if st.button(f"查看答案 #{i+1}", key=f"reveal_{i}"):
                        st.info(f"**正确答案**: {q.correct_answer}")
                        st.success(f"**解析**: {q.explanation}")
                        if q.source_ref:
                            st.caption(f"来源: {q.source_ref}")

            # 交卷
            if st.button("📤 提交试卷", type="primary"):
                correct = 0
                for i, q in enumerate(quiz.questions):
                    if q.question_type == "mcq" and user_answers.get(i):
                        selected_letter = user_answers[i].split(".")[0].strip()
                        if selected_letter == q.correct_answer.strip():
                            correct += 1
                st.balloons()
                st.success(f"选择题得分: {correct}/{len([q for q in quiz.questions if q.question_type=='mcq'])} (计算题需人工批改)")
        else:
            st.warning("未能生成题目。请输入更具体的知识点或尝试更宽泛的主题。")

# ---- Tab 3: Python 代码 ----
with tab3:
    st.subheader("Python 材料科学计算")
    st.caption("相图绘制 · 扩散曲线 · 晶体结构可视化 · 力学性能")

    code_col1, code_col2 = st.columns([3, 1])
    with code_col1:
        code_query = st.text_input(
            "描述你的需求",
            placeholder="例如: 画Fe-C相图 | 计算FCC致密度 | 菲克扩散曲线 | 杠杆定律计算",
            label_visibility="collapsed",
        )
    with code_col2:
        run_code = st.button("生成代码", type="primary", use_container_width=True)

    # 模板列表
    with st.expander("可用模板", expanded=False):
        templates = code_ast.list_available_templates()
        tcols = st.columns(3)
        for i, (tid, desc) in enumerate(templates.items()):
            with tcols[i % 3]:
                st.caption(f"**{desc}**")
                if st.button(f"加载 →", key=f"tpl_{tid}"):
                    code_query = tid
                    run_code = True

    if run_code and code_query:
        with st.spinner("生成代码中..."):
            result = code_ast.generate_code(code_query)

        if result.success and result.code:
            st.code(result.code, language="python")

            if st.button("▶️ 运行代码", type="primary"):
                with st.spinner("执行中..."):
                    exec_result = code_ast.execute_code(result.code)

                if exec_result.output:
                    st.markdown("### 输出结果")
                    st.code(exec_result.output, language="text")
                if exec_result.error:
                    st.error(exec_result.error)
        else:
            st.warning("未能生成匹配的代码。请尝试更具体的描述。")

# ---- Tab 4: 实验建议 ----
with tab4:
    st.subheader("实验参数建议与虚拟相预测")

    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        st.markdown("### 🔬 相预测 (Phase Prediction)")

        comp_input = st.text_input(
            "合金成分", value="Fe-0.45%C",
            placeholder="Fe-0.45%C | Al-4%Cu | Ti-6Al-4V",
        )
        temp_input = st.slider("温度 (°C)", 0, 1600, 800, 10)

        if st.button("预测平衡相", use_container_width=True):
            elements = {}
            parts = comp_input.replace("%", "").split("-")
            for part in parts:
                part = part.strip()
                match = False
                for el, z in [("Fe", 26), ("C", 6), ("Al", 13), ("Cu", 29),
                              ("Ti", 22), ("V", 23), ("Ni", 28), ("Cr", 24)]:
                    if part.startswith(el):
                        try:
                            val = float(part.replace(el, ""))
                            elements[el] = val
                        except ValueError:
                            elements[el] = 100.0
                        match = True
                        break
                if not match:
                    elements[part] = 50.0

            if not elements:
                st.warning("请使用格式: Fe-0.45%C")
            else:
                prediction = exp_advisor.predict_phases(elements, temp_input)
                st.markdown(f"**预测置信度**: {prediction.confidence.upper()}")
                st.markdown(f"**依据**: {prediction.basis}")

                for p in prediction.predicted_phases:
                    frac = p.get("fraction", 0)
                    comp = p.get("composition", "")
                    st.metric(
                        label=p["phase"],
                        value=f"{frac*100:.1f}%" if frac < 1 else "100%",
                        delta=comp if comp else None,
                    )

    with exp_col2:
        st.markdown("### 🔥 热处理建议")

        carbon = st.slider("含碳量 (wt%)", 0.0, 6.67, 0.45, 0.01)
        target = st.selectbox("目标性能", ["balanced", "hardness", "ductility"],
                               format_func=lambda x: {"balanced": "综合性能", "hardness": "高硬度", "ductility": "高韧性"}[x])

        if st.button("获取热处理方案", use_container_width=True):
            advice = exp_advisor.suggest_heat_treatment(carbon, target)
            st.markdown(f"### {advice.title}")
            st.markdown(f"**推荐工艺**: {advice.suggested_process}")

            for k, v in advice.parameters.items():
                st.metric(label=k.replace("_", " ").title(), value=str(v))

            st.markdown("### 注意事项")
            for p in advice.precautions:
                st.warning(p)

        st.markdown("---")
        st.markdown("### 🧪 合成方案查询")

        compound = st.text_input("目标化合物", placeholder="BaTiO3 | LiCoO2 | YBa2Cu3O7 | ZnO")
        if st.button("查询合成方案", use_container_width=True) and compound:
            synth = exp_advisor.suggest_synthesis(compound.strip())
            st.markdown(f"### {synth.title}")
            st.markdown(f"**方法**: {synth.suggested_process}")
            for k, v in synth.parameters.items():
                st.write(f"- **{k}**: {v}")

        # 冷却曲线预测
        st.markdown("---")
        st.markdown("### ❄️ CCT 冷却组织预测")

        cool_carbon = st.slider("含碳量", 0.0, 1.5, 0.45, 0.01, key="cool_c")
        cool_rate = st.select_slider(
            "冷却速率 (°C/s)",
            options=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0],
            value=1.0,
        )
        if st.button("预测冷却组织", use_container_width=True):
            cooling = exp_advisor.predict_cooling_curve(cool_carbon, cool_rate)
            st.metric("预测组织", cooling["predicted_microstructure"])
            st.metric("估计硬度", f"{cooling['estimated_hardness_HRC']} HRC")
            st.metric("Ms温度", f"{cooling['Ms_temperature_C']:.0f}°C")
            st.caption(cooling["note"])

# ---- Tab 5: 知识库统计 ----
with tab5:
    st.subheader("知识库统计信息")

    if kb_ready:
        stats = kb.get_statistics()

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("文本块总数", stats["total_chunks"])
        with metric_cols[1]:
            st.metric("总字符数", f"{stats['total_chars']:,}")
        with metric_cols[2]:
            st.metric("平均块大小", f"{stats['avg_chunk_size']} 字符")
        with metric_cols[3]:
            st.metric("教材数", len(stats["sources"]))

        st.markdown("### 教材来源")
        src_df = pd.DataFrame(
            [{"教材": k, "文本块数": v} for k, v in stats["sources"].items()]
        )
        st.dataframe(src_df, use_container_width=True, hide_index=True)

        st.markdown("### 章节覆盖")
        for ch in stats["chapters"][:20]:
            st.caption(f"- {ch}")
        if len(stats["chapters"]) > 20:
            st.caption(f"... 等 {len(stats['chapters'])} 个章节")

        st.markdown("### 检索测试")
        test_query = st.text_input("测试查询", "位错")
        if test_query:
            test_results = kb.search(test_query, k=3)
            for i, (chunk, score) in enumerate(test_results):
                st.markdown(f"**[{i+1}] 相关度: {score:.4f}** — {chunk.source_file}, p.{chunk.page_number}")
                st.text(chunk.text[:300] + "...")
    else:
        st.warning("知识库未构建。请在侧边栏上传教材PDF。")

st.markdown("---")
st.caption("模块6 · 材料科学本科智能学习助手 · RAG + 自动出题 + 代码辅助 + 实验建议")
