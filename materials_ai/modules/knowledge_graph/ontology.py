"""材料科学知识图谱本体 — Neo4j节点标签/关系类型/属性模式"""

# ============================================================
# Neo4j 节点标签 (Node Labels)
# ============================================================
NODE_LABELS = {
    "Material":       "材料",
    "Composition":    "成分/元素",
    "Phase":          "相",
    "Process":        "工艺",
    "Property":       "性能",
    "Application":    "应用",
    "Microstructure": "微观结构",
    "Paper":          "论文来源",
}

# ============================================================
# 节点属性模式 (Node Property Schemas)
# ============================================================
NODE_PROPERTIES = {
    "Material":       ["name", "formula", "material_class", "source", "paper_ids"],
    "Composition":    ["name", "element", "fraction", "unit"],
    "Phase":          ["name", "crystal_system", "space_group", "temperature_range"],
    "Process":        ["name", "process_type", "temperature", "time", "atmosphere"],
    "Property":       ["name", "property_type", "unit", "description"],
    "Application":    ["name", "application_field", "description"],
    "Microstructure": ["name", "microstructure_type", "length_scale", "description"],
    "Paper":          ["paper_id", "title", "year", "doi", "source"],
}

# ============================================================
# 关系类型定义 (Relationship Types)
# ============================================================
RELATION_TYPES = {
    # 材料→性能 (Material has Property with Value)
    "hasProperty": {
        "from": "Material", "to": "Property",
        "label_zh": "具有性能",
        "properties": ["value", "value_numeric", "unit", "confidence", "paper_id", "evidence"],
    },
    # 材料→工艺 (Material is processed by Process)
    "usesProcess": {
        "from": "Material", "to": "Process",
        "label_zh": "采用工艺",
        "properties": ["confidence", "paper_id", "evidence"],
    },
    # 材料→微观结构
    "hasMicrostructure": {
        "from": "Material", "to": "Microstructure",
        "label_zh": "呈现微观结构",
        "properties": ["confidence", "paper_id", "evidence"],
    },
    # 材料→应用
    "usedIn": {
        "from": "Material", "to": "Application",
        "label_zh": "应用于",
        "properties": ["confidence", "paper_id", "evidence"],
    },
    # 材料→成分 (Material consists of Composition elements)
    "hasComposition": {
        "from": "Material", "to": "Composition",
        "label_zh": "含元素",
        "properties": ["fraction", "paper_id"],
    },
    # 材料→相
    "hasPhase": {
        "from": "Material", "to": "Phase",
        "label_zh": "具有相",
        "properties": ["temperature", "pressure", "paper_id"],
    },
    # 工艺→性能 (Process improves/degrades Property)
    "affectsProperty": {
        "from": "Process", "to": "Property",
        "label_zh": "影响性能",
        "properties": ["effect", "magnitude", "direction", "paper_id", "evidence"],
    },
    # 工艺→微观结构 (Process results in Microstructure)
    "resultsIn": {
        "from": "Process", "to": "Microstructure",
        "label_zh": "形成结构",
        "properties": ["confidence", "paper_id", "evidence"],
    },
    # 论文→节点 (Paper reports findings about any node)
    "reports": {
        "from": "Paper", "to": "Material",
        "label_zh": "报道了",
        "properties": ["paper_id"],
    },
}

# ============================================================
# 颜色映射 (用于可视化)
# ============================================================
ENTITY_COLORS = {
    "Material":       "#FF6B6B",
    "Composition":    "#FFD93D",
    "Phase":          "#C9B1FF",
    "Process":        "#45B7D1",
    "Property":       "#4ECDC4",
    "Application":    "#FFEAA7",
    "Microstructure": "#96CEB4",
    "Paper":          "#DDA0DD",
    # 兼容旧版key
    "material":           "#FF6B6B",
    "property":           "#4ECDC4",
    "processing_method":  "#45B7D1",
    "crystal_structure":  "#96CEB4",
    "application":        "#FFEAA7",
    "property_value":     "#DDA0DD",
}

ENTITY_LABELS_ZH = {k: v for k, v in NODE_LABELS.items()}
ENTITY_LABELS_ZH.update({
    "material":           "材料",
    "property":           "性能",
    "processing_method":  "加工方法",
    "synthesis_method":   "合成方法",
    "crystal_structure":  "晶体结构",
    "application":        "应用领域",
    "microstructure":     "微观结构",
    "property_value":     "性能值",
})

RELATION_LABELS_ZH = {
    k: v["label_zh"] for k, v in RELATION_TYPES.items()
}
RELATION_LABELS_ZH.update({
    "hasProperty":       "具有性能",
    "usesProcess":       "采用工艺",
    "hasMicrostructure": "呈现微观结构",
    "usedIn":            "应用于",
    "hasComposition":    "含元素",
    "hasPhase":          "具有相",
    "affectsProperty":   "影响性能",
    "resultsIn":         "形成结构",
    "reports":           "报道了",
    # 兼容
    "processedBy":  "加工方式",
    "hasStructure": "具有结构",
    "hasValue":     "数值为",
    "relatedTo":    "关联于",
    "synthesizedBy": "合成方式",
})

# ============================================================
# Cypher 查询模板
# ============================================================
CYPHER_TEMPLATES = {
    # 查找可提高钢的屈服强度且不显著降低延伸率的工艺
    "improve_strength_ductility": """
        MATCH (m:Material)
        WHERE toLower(m.name) CONTAINS toLower($material_name)
        MATCH (m)-[:usesProcess]->(p:Process)
        MATCH (p)-[r1:affectsProperty]->(prop1:Property)
        WHERE toLower(prop1.name) CONTAINS 'yield strength'
          AND r1.direction = 'increase'
        OPTIONAL MATCH (p)-[r2:affectsProperty]->(prop2:Property)
        WHERE toLower(prop2.name) CONTAINS 'elongation'
        WITH p, prop1, r1, prop2, r2
        WHERE r2 IS NULL OR r2.direction <> 'decrease'
           OR (r2.direction = 'decrease' AND abs(coalesce(r2.magnitude, 0)) < 0.1)
        RETURN DISTINCT p.name AS process,
               prop1.name AS improved_property,
               r1.magnitude AS improvement_magnitude,
               coalesce(prop2.name, 'N/A') AS ductility_property,
               coalesce(r2.magnitude, 0) AS ductility_change
        ORDER BY r1.magnitude DESC
    """,

    # 推荐潜在高性能材料
    "recommend_high_performance": """
        MATCH (m:Material)-[:hasProperty]->(prop:Property)
        MATCH (prop)<-[r:affectsProperty]-(p:Process)
        WHERE toLower(prop.name) CONTAINS toLower($property_name)
          AND coalesce(r.direction, 'increase') = 'increase'
        WITH m, collect(DISTINCT p.name) AS processes,
             avg(coalesce(r.magnitude, 0.5)) AS avg_improvement
        WHERE size(processes) >= $min_processes
        RETURN m.name AS material,
               m.material_class AS material_class,
               processes,
               avg_improvement
        ORDER BY avg_improvement DESC
        LIMIT $limit
    """,

    # 材料对比查询
    "compare_materials": """
        MATCH (m1:Material {name: $material1})
        MATCH (m2:Material {name: $material2})
        OPTIONAL MATCH (m1)-[r1]->(n)
        WHERE type(r1) IN ['hasProperty', 'usesProcess', 'hasMicrostructure', 'usedIn']
        OPTIONAL MATCH (m2)-[r2]->(n2)
        WHERE type(r2) = type(r1)
        RETURN m1.name AS material1, type(r1) AS relation_type,
               collect(DISTINCT n.name)[0..5] AS material1_values,
               m2.name AS material2,
               collect(DISTINCT n2.name)[0..5] AS material2_values
        LIMIT 20
    """,

    # 查找某工艺影响的所有性能
    "process_effects": """
        MATCH (p:Process)
        WHERE toLower(p.name) CONTAINS toLower($process_name)
        MATCH (p)-[r:affectsProperty]->(prop:Property)
        OPTIONAL MATCH (p)-[r2:resultsIn]->(ms:Microstructure)
        RETURN p.name AS process,
               collect(DISTINCT {property: prop.name,
                                 direction: r.direction,
                                 magnitude: r.magnitude}) AS affected_properties,
               collect(DISTINCT ms.name) AS resulting_microstructures
    """,

    # 材料-性能缺失链接预测用 (供RGCN训练)
    "all_material_property_pairs": """
        MATCH (m:Material)
        MATCH (prop:Property)
        OPTIONAL MATCH (m)-[r:hasProperty]->(prop)
        RETURN m.name AS material, prop.name AS property,
               CASE WHEN r IS NOT NULL THEN 1 ELSE 0 END AS exists,
               coalesce(r.value_numeric, 0) AS value,
               coalesce(r.confidence, 0) AS confidence
    """,

    # 子图导出
    "export_subgraph": """
        MATCH (m:Material)
        WHERE toLower(m.name) CONTAINS toLower($keyword)
        MATCH (m)-[r1]->(n1)
        OPTIONAL MATCH (n1)-[r2]->(n2)
        RETURN m, r1, n1, r2, n2
        LIMIT 100
    """,

    # 查找通过特定微观结构提升性能的工艺
    "microstructure_mediated_improvement": """
        MATCH (p:Process)-[:resultsIn]->(ms:Microstructure)
        MATCH (ms)<-[:hasMicrostructure]-(m:Material)
        MATCH (m)-[:hasProperty]->(prop:Property)
        WHERE toLower(ms.name) CONTAINS toLower($microstructure)
          AND toLower(prop.name) CONTAINS toLower($property)
        RETURN m.name AS material, p.name AS process,
               ms.name AS microstructure, prop.name AS property
        LIMIT $limit
    """,
}
