"""材料科学实体/关系模式定义 — 词典 + 正则表达式"""

# ============================================================
# 材料名称词典
# ============================================================
MATERIAL_NAMES = [
    # 氧化物
    "titanium dioxide", "zinc oxide", "iron oxide", "aluminum oxide", "alumina",
    "silicon dioxide", "silica", "zirconia", "ceria", "magnesium oxide",
    "copper oxide", "nickel oxide", "cobalt oxide", "manganese oxide",
    "tin oxide", "indium tin oxide", "ITO", "barium titanate", "strontium titanate",
    "bismuth ferrite", "lanthanum manganite", "yttria-stabilized zirconia", "YSZ",
    # 钙钛矿
    "perovskite", "methylammonium lead iodide", "MAPbI3", "formamidinium",
    "CsPbI3", "CsPbBr3", "FAPbI3", "MASnI3", "double perovskite",
    "halide perovskite", "oxide perovskite",
    # 电池材料
    "lithium cobalt oxide", "LCO", "lithium iron phosphate", "LFP",
    "lithium nickel manganese cobalt oxide", "NMC", "lithium nickel cobalt aluminum oxide", "NCA",
    "lithium manganese oxide", "LMO", "lithium titanate", "LTO",
    "graphite", "silicon anode", "solid electrolyte",
    "LLZO", "LGPS", "NASICON", "garnet electrolyte",
    # 热电材料
    "bismuth telluride", "lead telluride", "skutterudite", "half-Heusler",
    "silicon germanium", "SiGe", "tin selenide", "copper selenide",
    "magnesium silicide", "cobalt triantimonide",
    # 催化材料
    "titanium dioxide", "anatase", "rutile", "zinc oxide", "cadmium sulfide",
    "bismuth vanadate", "graphitic carbon nitride", "g-C3N4",
    "metal-organic framework", "MOF", "zeolite", "ZSM-5",
    "platinum", "palladium", "ruthenium", "iridium",
    # 金属与合金
    "steel", "stainless steel", "carbon steel", "titanium alloy",
    "aluminum alloy", "magnesium alloy", "nickel superalloy",
    "high-entropy alloy", "HEA", "FeCoNiCrMn", "Cantor alloy",
    "shape memory alloy", "NiTi", "Nitinol", "bulk metallic glass",
    # 半导体
    "silicon", "germanium", "gallium arsenide", "GaAs", "gallium nitride", "GaN",
    "silicon carbide", "SiC", "indium phosphide", "InP", "zinc sulfide", "ZnS",
    "cadmium telluride", "CdTe", "molybdenum disulfide", "MoS2",
    "hexagonal boron nitride", "h-BN", "black phosphorus",
    # 碳材料
    "graphene", "graphene oxide", "reduced graphene oxide", "carbon nanotube",
    "CNT", "fullerene", "C60", "carbon fiber", "diamond", "graphite",
    # 陶瓷
    "silicon nitride", "boron carbide", "tungsten carbide", "Sialon",
    "hydroxyapatite", "bioactive glass", "aluminum nitride",
]

# ============================================================
# 性能名称词典
# ============================================================
PROPERTY_NAMES = [
    # 电子性能
    "band gap", "bandgap", "band-gap",
    "formation energy", "cohesive energy", "total energy",
    "dielectric constant", "relative permittivity", "refractive index",
    "electrical conductivity", "electrical resistivity", "sheet resistance",
    "carrier mobility", "electron mobility", "hole mobility",
    "doping concentration", "Fermi level", "work function",
    "Seebeck coefficient", "thermoelectric figure of merit", "ZT",
    "power factor", "Hall coefficient",
    # 力学性能
    "bulk modulus", "shear modulus", "Young's modulus", "elastic modulus",
    "yield strength", "tensile strength", "ultimate tensile strength",
    "compressive strength", "flexural strength", "fracture toughness",
    "hardness", "Vickers hardness", "nanoindentation hardness",
    "Poisson's ratio", "elongation", "ductility", "brittleness",
    # 热性能
    "thermal conductivity", "thermal expansion coefficient",
    "specific heat capacity", "heat capacity", "thermal diffusivity",
    "Debye temperature", "melting point", "glass transition temperature",
    "thermal stability", "lattice thermal conductivity",
    # 光学性能
    "absorption coefficient", "photoluminescence", "quantum yield",
    "transmittance", "reflectance", "extinction coefficient",
    "plasmon resonance", "upconversion", "photocatalytic activity",
    # 电化学性能
    "specific capacity", "discharge capacity", "capacity retention",
    "rate capability", "Coulombic efficiency", "cycling stability",
    "ionic conductivity", "lithium ion conductivity", "electrochemical window",
    "exchange current density", "Tafel slope", "overpotential",
    "hydrogen evolution reaction", "HER", "oxygen evolution reaction", "OER",
    "oxygen reduction reaction", "ORR",
    # 磁性能
    "magnetic moment", "saturation magnetization", "coercivity",
    "Curie temperature", "magnetic anisotropy", "spin polarization",
    "magnetoresistance", "exchange bias",
]

# ============================================================
# 加工方法词典
# ============================================================
PROCESSING_KEYWORDS = [
    # 化学方法
    "sol-gel", "solvothermal", "hydrothermal", "co-precipitation",
    "chemical vapor deposition", "CVD", "plasma-enhanced CVD", "PECVD",
    "atomic layer deposition", "ALD", "molecular beam epitaxy", "MBE",
    "electrodeposition", "electroplating", "electrospinning",
    "spray pyrolysis", "solution combustion", "auto-combustion",
    "pechini method", "citrate gel", "reverse micelle", "microemulsion",
    # 物理方法
    "ball milling", "high-energy ball milling", "mechanical alloying",
    "spark plasma sintering", "SPS", "hot pressing", "hot isostatic pressing",
    "cold pressing", "sintering", "solid-state reaction",
    "thermal evaporation", "electron beam evaporation", "e-beam evaporation",
    "sputtering", "magnetron sputtering", "pulsed laser deposition", "PLD",
    "arc melting", "induction melting", "vacuum melting",
    "melt spinning", "rapid solidification", "quenching", "annealing",
    "zone melting", "Czochralski method", "Bridgman method", "float zone",
    # 新型方法
    "3D printing", "additive manufacturing", "selective laser melting", "SLM",
    "inkjet printing", "screen printing", "doctor blade", "tape casting",
    "spin coating", "dip coating", "drop casting", "Langmuir-Blodgett",
    "template synthesis", "self-assembly", "exfoliation", "liquid-phase exfoliation",
]

# ============================================================
# 晶体结构词典
# ============================================================
CRYSTAL_STRUCTURES = [
    "cubic", "tetragonal", "orthorhombic", "hexagonal", "monoclinic",
    "triclinic", "rhombohedral",
    "perovskite", "spinel", "inverse spinel", "fluorite", "rock salt",
    "wurtzite", "zinc blende", "sphalerite", "garnet", "NASICON",
    "layered structure", "layered oxide", "tunnel structure",
    "olivine", "NASICON-type", "rutile", "anatase", "brookite",
    "body-centered cubic", "bcc", "face-centered cubic", "fcc",
    "hexagonal close-packed", "hcp", "diamond cubic",
    "A-site", "B-site", "octahedral site", "tetrahedral site",
    "vacancy", "interstitial", "substitutional",
    "space group", "point group", "lattice parameter",
    "polycrystalline", "single crystal", "amorphous", "nanocrystalline",
]

# ============================================================
# 应用领域词典
# ============================================================
APPLICATIONS = [
    # 能源
    "solar cell", "photovoltaic", "perovskite solar cell", "tandem solar cell",
    "dye-sensitized solar cell", "organic solar cell", "quantum dot solar cell",
    "battery", "cathode", "anode", "electrolyte", "separator", "current collector",
    "lithium-ion battery", "sodium-ion battery", "solid-state battery",
    "lithium-sulfur battery", "lithium-air battery", "redox flow battery",
    "fuel cell", "solid oxide fuel cell", "SOFC", "proton exchange membrane", "PEM",
    "supercapacitor", "pseudocapacitor", "electric double layer capacitor",
    "thermoelectric generator", "thermoelectric cooler",
    "water splitting", "photocatalytic water splitting",
    "electrocatalysis", "photocatalysis", "photoelectrochemical cell",
    # 电子
    "transistor", "field-effect transistor", "FET", "thin-film transistor",
    "LED", "light-emitting diode", "OLED", "quantum dot LED", "QLED",
    "sensor", "gas sensor", "biosensor", "chemical sensor", "pressure sensor",
    "memory device", "RRAM", "memristor", "phase change memory",
    "integrated circuit", "semiconductor", "dielectric", "interconnect",
    "flexible electronics", "wearable electronics", "transparent electrode",
    "photodetector", "photodiode", "X-ray detector", "infrared detector",
    # 结构材料
    "structural material", "lightweight alloy", "high-temperature alloy",
    "coating", "thermal barrier coating", "corrosion-resistant coating",
    "hard coating", "wear-resistant", "cutting tool", "abrasive",
    "armor", "ballistic protection", "aerospace", "automotive",
    # 生物医学
    "biomedical implant", "bone regeneration", "drug delivery",
    "biosensor", "bioimaging", "photothermal therapy", "MRI contrast agent",
    "dental material", "stent", "suture",
    # 环境
    "water purification", "air purification", "CO2 capture", "CO2 reduction",
    "wastewater treatment", "desalination", "heavy metal removal",
    "pollutant degradation", "volatile organic compound", "VOC removal",
    # 其他
    "magnet", "permanent magnet", "soft magnet", "magnetic refrigeration",
    "superconductor", "high-temperature superconductor", "topological insulator",
    "ferroelectric", "piezoelectric", "multiferroic", "metamaterial",
]

# ============================================================
# 正则表达式
# ============================================================
# 化学式: 匹配 "TiO2", "SrTiO3", "LiFePO4", "MAPbI3" 等
CHEMICAL_FORMULA_PATTERN = r'\b(?:[A-Z][a-z]?\d*)+(?:\([^)]*\)\d*)*\b'

# 属性值模式: "3.2 eV", "500 MPa", "2.5 W/mK", "1.5 × 10^-3 S/cm"
NUMBER_PATTERN = r'(\d+\.?\d*)\s*(?:[×xX]\s*10\^?[-−]?\d+\s*)?'
UNIT_PATTERN = r'(?:eV|meV|J|kJ|MPa|GPa|W/(?:m[·⋅]?K)|W/mK|S/cm|Ω[·⋅]?cm|g/cm[³3]|W[·⋅]?m[−–]?[¹1][·⋅]?K[−–]?[¹1]|μV/K|cm[²2]/Vs|emu/g|Oe|T|K|°C|at%|wt%|mol%|W)'
PROPERTY_VALUE_PATTERN = NUMBER_PATTERN + r'\s*' + UNIT_PATTERN

# 材料名称关键词 (用于过滤，至少匹配到化学式或已知材料)
MATERIAL_KEYWORD_PATTERNS = [
    r'\b(?:oxide|alloy|perovskite|spinel|garnet|ceramic|polymer|composite|metal|semiconductor|electrolyte|electrode|catalyst)\b',
]
