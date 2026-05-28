"""知识图谱 Schema — 实体/关系类型和颜色定义"""

ENTITY_TYPES = ["material", "property", "processing_method", "crystal_structure", "application", "property_value"]

RELATION_DEFS = {
    "hasProperty":       ("material", "property"),
    "processedBy":       ("material", "processing_method"),
    "hasStructure":      ("material", "crystal_structure"),
    "usedIn":            ("material", "application"),
    "hasValue":          ("property", "property_value"),
    "relatedTo":         ("material", "material"),
}

ENTITY_COLORS = {
    "material":           "#FF6B6B",
    "property":           "#4ECDC4",
    "processing_method":  "#45B7D1",
    "crystal_structure":  "#96CEB4",
    "application":        "#FFEAA7",
    "property_value":     "#DDA0DD",
}

ENTITY_LABELS_ZH = {
    "material":           "材料",
    "property":           "性能",
    "processing_method":  "加工方法",
    "crystal_structure":  "晶体结构",
    "application":        "应用领域",
    "property_value":     "性能值",
}

RELATION_LABELS_ZH = {
    "hasProperty":   "具有性能",
    "processedBy":   "加工方式",
    "hasStructure":  "具有结构",
    "usedIn":        "应用于",
    "hasValue":      "数值为",
    "relatedTo":     "关联于",
}
