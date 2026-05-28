"""模块2: 材料知识图谱 v2 — Streamlit 页面 (6标签)"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.knowledge_graph.ontology import (
    NODE_LABELS, RELATION_TYPES, NODE_PROPERTIES,
    ENTITY_COLORS, ENTITY_LABELS_ZH, RELATION_LABELS_ZH, CYPHER_TEMPLATES,
)
from modules.knowledge_graph.graph_builder import KnowledgeGraphBuilder, HAS_NEO4J
from modules.knowledge_graph.graph_query import GraphQuery
from modules.knowledge_graph.graph_viz import GraphVisualizer
from modules.knowledge_graph.gnn_link_predictor import (
    HAS_TORCH, HAS_PYG, MaterialKnowledgeGraph, RGCNLinkPredictor, train_link_prediction,
)
from modules.nlp_literature_mining.database import TripletStore
from utils.data_loader import DataLoader
import config

st.set_page_config(page_title="知识图谱", page_icon="🕸️", layout="wide")

st.title("🕸️ 材料知识图谱 v2")
st.markdown("本体驱动 · Neo4j存储 · Cypher查询 · RGCN推理")

# ---- 加载图谱 ----
@st.cache_resource
def load_kg():
    seed_kg = DataLoader.load_seed_kg()
    parsed_papers = DataLoader.load_parsed_papers()
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_extractions(parsed_papers, seed_kg)

    # 尝试加载三元组
    try:
        store = TripletStore()
        triplets = store.query_triplets(limit=500)
        graph = builder.build_from_triplets(triplets)
    except Exception:
        pass

    query = GraphQuery(graph)
    return graph, query, builder

graph, query_engine, builder = load_kg()

# ---- 6 Tabs ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ 图谱可视化", "🔍 实体查询", "⚡ Cypher 查询",
    "🧠 RGCN 推理", "📊 统计 & Schema", "💾 Neo4j 导入",
])

# ============================================================
# Tab 1: 图谱可视化
# ============================================================
with tab1:
    st.subheader("交互式知识图谱")
    st.caption("节点按类型着色 | 拖拽/缩放/悬停查看详情 | 箭头表示关系方向")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.write("**图例**")
        legend_html = ""
        for etype, color in ENTITY_COLORS.items():
            label = ENTITY_LABELS_ZH.get(etype, etype)
            legend_html += (
                f'<div style="display:flex;align-items:center;margin:4px 0;">'
                f'<div style="width:14px;height:14px;background:{color};'
                f'border-radius:3px;margin-right:6px;"></div>'
                f'<span style="font-size:13px;">{label}</span></div>'
            )
        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("---")
        show_types = st.multiselect(
            "显示实体类型:",
            list(set(attrs.get("entity_type", "") for _, attrs in graph.nodes(data=True))),
            default=list(set(attrs.get("entity_type", "") for _, attrs in graph.nodes(data=True))),
            format_func=lambda x: ENTITY_LABELS_ZH.get(x, x),
        )
        physics_enabled = st.checkbox("物理引擎动画", value=True)

    with col1:
        filtered_nodes = [
            n for n, attrs in graph.nodes(data=True)
            if attrs.get("entity_type", "") in show_types
        ]
        subgraph = graph.subgraph(filtered_nodes).copy() if filtered_nodes else graph

        if subgraph.number_of_nodes() > 0:
            html_content = GraphVisualizer.to_pyvis_html(
                subgraph, height="600px"
            ) if not physics_enabled else GraphVisualizer.to_pyvis_html(
                subgraph, height="600px"
            )
            st.components.v1.html(html_content, height=620, scrolling=True)
        else:
            st.warning("没有匹配的节点。")

# ============================================================
# Tab 2: 实体查询
# ============================================================
with tab2:
    st.subheader("实体检索 & 路径发现")

    col_search, col_path = st.columns(2)

    with col_search:
        search_query = st.text_input("搜索实体", placeholder="如: TiO2, graphene, steel...")
        if search_query:
            results = query_engine.search(search_query)
            if results:
                st.success(f"找到 {len(results)} 个实体")
                for r in results[:20]:
                    etype = r.get("entity_type", "")
                    color = ENTITY_COLORS.get(etype, "#999")
                    st.markdown(
                        f'<span style="color:{color};font-weight:bold;">● {r["name"]}</span> '
                        f'({ENTITY_LABELS_ZH.get(etype, etype)})',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("未找到。模糊匹配:")
                fuzzy = query_engine.fuzzy_search(search_query)
                for r in fuzzy[:10]:
                    st.write(f"- {r['name']} ({r.get('entity_type', '')})")

    with col_path:
        st.write("**路径发现**")
        e1 = st.text_input("起始实体", key="path_e1")
        e2 = st.text_input("目标实体", key="path_e2")
        if st.button("查找路径"):
            if e1 and e2:
                path = query_engine.find_path(e1, e2)
                if path:
                    st.success(f"路径长度: {len(path) - 1}")
                    path_str = " → ".join(p["name"] for p in path)
                    st.markdown(path_str)
                else:
                    st.warning("未找到路径")

    st.markdown("---")
    st.subheader("实体详情")
    detail = st.text_input("实体名称", placeholder="如: TiO2, MAPbI3...")
    if detail:
        neighbors = query_engine.get_neighbors(detail)
        entity = neighbors.get("entity")
        if entity:
            etype = entity.get("entity_type", "")
            color = ENTITY_COLORS.get(etype, "#999")
            st.markdown(f'### <span style="color:{color};">● {entity["name"]}</span>',
                       unsafe_allow_html=True)
            st.write(f"**类型**: {ENTITY_LABELS_ZH.get(etype, etype)}")

            if neighbors.get("relations"):
                st.write("**关联关系:**")
                st.dataframe([
                    {
                        "方向": "→" if r["direction"] == "outgoing" else "←",
                        "关系": RELATION_LABELS_ZH.get(r["predicate"], r["predicate"]),
                        "目标": r["target"]["name"],
                        "目标类型": ENTITY_LABELS_ZH.get(
                            r["target"].get("entity_type", ""), ""
                        ),
                    }
                    for r in neighbors["relations"]
                ], use_container_width=True)
        else:
            st.warning(f"未找到: {detail}")

# ============================================================
# Tab 3: Cypher 查询 (Neo4j)
# ============================================================
with tab3:
    st.subheader("Cypher 高级查询")
    st.caption("需要 Neo4j 连接。支持材料工艺推荐、性能影响分析等。")

    if not HAS_NEO4J:
        st.warning("py2neo 未安装。运行: `pip install py2neo`")
    elif not builder.neo4j_available:
        st.info("Neo4j 未连接。启动 Neo4j 数据库后刷新页面。")

    query_type = st.selectbox("查询模板:", list(CYPHER_TEMPLATES.keys()), format_func=lambda x: {
        "improve_strength_ductility": "查找提高强度不降塑性的工艺",
        "recommend_high_performance": "推荐潜在高性能材料",
        "compare_materials": "材料对比",
        "process_effects": "工艺影响分析",
        "all_material_property_pairs": "材料-性能对列表",
        "export_subgraph": "导出子图",
        "microstructure_mediated_improvement": "微观结构调控制性能",
    }.get(x, x))

    col1, col2 = st.columns(2)
    params = {}

    if query_type == "improve_strength_ductility":
        with col1:
            params["material_name"] = st.text_input("材料名称", "steel")
    elif query_type == "recommend_high_performance":
        with col1:
            params["property_name"] = st.text_input("性能名称", "strength")
        with col2:
            params["min_processes"] = st.slider("最少工艺数", 1, 10, 2)
            params["limit"] = st.slider("返回条数", 5, 50, 10)
    elif query_type == "compare_materials":
        with col1:
            params["material1"] = st.text_input("材料1", "TiO2")
        with col2:
            params["material2"] = st.text_input("材料2", "ZnO")
    elif query_type == "process_effects":
        with col1:
            params["process_name"] = st.text_input("工艺名称", "annealing")
    elif query_type == "microstructure_mediated_improvement":
        with col1:
            params["microstructure"] = st.text_input("微观结构", "grain boundary")
        with col2:
            params["property"] = st.text_input("性能", "strength")
            params["limit"] = st.slider("返回条数", 5, 50, 20)

    st.code(CYPHER_TEMPLATES[query_type], language="cypher")

    if st.button("执行查询", type="primary"):
        if not builder.neo4j_available:
            st.error("Neo4j 不可用。请在「Neo4j 导入」标签中先导入数据。")
        else:
            try:
                result = query_engine.cypher_query(
                    CYPHER_TEMPLATES[query_type], params
                )
                if result:
                    st.success(f"返回 {len(result)} 条结果")
                    st.dataframe(result, use_container_width=True)
                else:
                    st.info("无结果。确认数据已导入且参数匹配。")
            except Exception as e:
                st.error(f"查询失败: {e}")

# ============================================================
# Tab 4: RGCN 链接预测
# ============================================================
with tab4:
    st.subheader("RGCN 链接预测 — 推荐潜在材料-性能关联")
    st.caption("基于关系图卷积网络预测材料可能具备的性能。")

    can_train = HAS_TORCH and HAS_PYG

    if not can_train:
        st.warning("PyTorch / PyTorch Geometric 未安装。\n\n"
                   "训练需要: `pip install torch torch-geometric`")

    col1, col2 = st.columns([1, 1])
    with col1:
        epochs = st.slider("训练轮数", 20, 300, config.RGCN_EPOCHS)
        hidden_dim = st.select_slider("隐层维度", [32, 64, 128, 256], 64)
        lr = st.select_slider("学习率", [0.001, 0.005, 0.01, 0.02, 0.05], 0.01,
                              format_func=lambda x: f"{x:.3f}")

    with col2:
        st.write("**训练数据统计**")
        num_mats = sum(1 for _, a in graph.nodes(data=True)
                       if a.get("entity_type") in ("material", "Material"))
        num_props = sum(1 for _, a in graph.nodes(data=True)
                        if a.get("entity_type") in ("property", "Property"))
        num_edges = graph.number_of_edges()
        st.metric("材料节点", num_mats)
        st.metric("性能节点", num_props)
        st.metric("总边数", num_edges)

    if st.button("训练 RGCN 模型", type="primary", disabled=not can_train):
        if num_mats < 3 or num_props < 3:
            st.error("材料和性能节点太少 (需要 ≥3 个)。请先导入更多数据。")
        else:
            with st.spinner("训练中..."):
                try:
                    model, results = train_link_prediction(
                        graph, epochs=epochs, hidden_dim=hidden_dim, lr=lr
                    )

                    st.success("训练完成!")

                    # 显示loss曲线
                    if results["history"]["train_loss"]:
                        st.line_chart({
                            "Training Loss": results["history"]["train_loss"],
                        })

                    if results["history"]["val_auc"]:
                        st.metric("验证AUC", f"{results['history']['val_auc'][-1]:.3f}")

                    st.markdown("---")
                    st.subheader("Top 推荐链接 (潜在新材料-性能关联)")

                    predictions = results["predictions"]
                    node_ids = results["node_ids"]

                    if predictions:
                        pred_data = []
                        for mi, pi, score in predictions[:20]:
                            pred_data.append({
                                "Material": node_ids[mi] if mi < len(node_ids) else f"idx_{mi}",
                                "Property": node_ids[pi] if pi < len(node_ids) else f"idx_{pi}",
                                "Score": f"{score:.4f}",
                            })
                        st.dataframe(pred_data, use_container_width=True)
                    else:
                        st.info("暂无高置信度预测。")

                except Exception as e:
                    st.error(f"训练失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================
# Tab 5: 统计 & Schema
# ============================================================
with tab5:
    st.subheader("图谱统计")

    stats = query_engine.get_statistics()

    c1, c2, c3 = st.columns(3)
    c1.metric("节点总数", stats["num_nodes"])
    c2.metric("边总数", stats["num_edges"])
    c3.metric("关系类型数", len(stats.get("relation_types", {})))

    col1, col2 = st.columns(2)
    with col1:
        st.write("**实体类型分布**")
        if stats.get("entity_types"):
            st.bar_chart(stats["entity_types"])
            fig = GraphVisualizer.entity_pie_chart(stats["entity_types"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**关系类型分布**")
        if stats.get("relation_types"):
            st.bar_chart(stats["relation_types"])

    st.markdown("---")
    st.subheader("知识图谱 Schema (本体)")

    st.write("**节点标签:**")
    for label, zh_name in NODE_LABELS.items():
        color = ENTITY_COLORS.get(label, "#999")
        st.markdown(f'- <span style="color:{color};">● {zh_name}</span> ({label})',
                   unsafe_allow_html=True)

    st.write("")
    st.write("**关系类型:**")
    for rel_type, defn in RELATION_TYPES.items():
        label_zh = defn["label_zh"]
        from_l, to_l = defn["from"], defn["to"]
        st.write(f"- **{label_zh}** ({rel_type}): {from_l} → {to_l}")

    st.markdown("---")
    st.info("""
    **种子知识图谱**: 包含 16 种材料、12 项性能、10 种加工方法、12 种晶体结构、12 个应用领域。
    数据涵盖钙钛矿光伏、锂电池正极、热电材料、光催化、高熵合金、宽带隙半导体、二维材料等。
    """)

# ============================================================
# Tab 6: Neo4j 导入
# ============================================================
with tab6:
    st.subheader("Neo4j 数据导入")
    st.caption("批量导入三元组和解析结果到 Neo4j 图数据库。")

    if not HAS_NEO4J:
        st.warning("py2neo 未安装。运行: `pip install py2neo`")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_input("Neo4j URI", value=config.NEO4J_URI, key="neo4j_uri",
                          disabled=True)
        with col2:
            st.text_input("用户名", value=config.NEO4J_USER, key="neo4j_user",
                          disabled=True)
        with col3:
            st.text_input("密码", value="****", key="neo4j_pwd", disabled=True)

        if builder.neo4j_available:
            st.success("✅ Neo4j 已连接")
        else:
            st.warning("⚠️ Neo4j 未连接。确保数据库已启动。")

        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("初始化 Schema (约束+索引)"):
                try:
                    builder.init_neo4j_schema()
                    st.success("Schema 初始化完成")
                except Exception as e:
                    st.error(f"失败: {e}")

        with col_act2:
            if st.button("导入三元组 → Neo4j"):
                try:
                    store = TripletStore()
                    triplets = store.query_triplets(limit=500)
                    if triplets:
                        count = builder.import_to_neo4j(triplets=triplets)
                        st.success(f"导入完成: {count} 个节点+关系")
                    else:
                        st.warning("没有三元组数据。请先在模块1中提取并保存。")
                except Exception as e:
                    st.error(f"导入失败: {e}")

        with col_act3:
            if st.button("清空 Neo4j", type="secondary"):
                confirm = st.checkbox("确认清空所有数据? 此操作不可逆。")
                if confirm:
                    try:
                        builder.clear_neo4j()
                        st.success("已清空")
                    except Exception as e:
                        st.error(f"失败: {e}")
