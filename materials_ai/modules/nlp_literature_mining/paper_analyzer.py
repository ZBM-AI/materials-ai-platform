"""论文深度分析引擎 — 发现提取 / 创新点 / 不足分析 (LLM + 规则双引擎)"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PaperDiscovery:
    """论文中的科学发现."""
    type: str  # material, reaction, phenomenon, mechanism, property, method
    description: str
    evidence: str = ""
    confidence: float = 0.5


@dataclass
class InnovationPoint:
    """论文创新点."""
    category: str  # method, material, theory, application, performance
    description: str
    significance: str = ""
    evidence: str = ""


@dataclass
class PaperShortcoming:
    """论文不足之处."""
    category: str  # method, scope, validation, comparison, gap
    description: str
    severity: str = "moderate"
    evidence: str = ""


@dataclass
class DeepAnalysisResult:
    """论文深度分析综合结果."""
    paper_title: str = ""
    discoveries: List[PaperDiscovery] = field(default_factory=list)
    innovations: List[InnovationPoint] = field(default_factory=list)
    shortcomings: List[PaperShortcoming] = field(default_factory=list)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    methodology: str = ""
    main_finding: str = ""
    raw_text_snippet: str = ""


class PaperAnalyzer:
    """论文深度分析器 — 提取论文的核心发现、创新点和不足.

    双引擎模式:
    - LLM引擎 (需OpenAI API Key): 语义理解精准分析
    - 规则引擎 (默认): 基于正则+关键词的模式匹配
    """

    # 发现相关的触发短语 (英文 + 中文)
    DISCOVERY_PATTERNS = {
        "material": [
            # 英文
            r'(?:(?:novel|new|novel)\s+)?(?:material|compound|alloy|ceramic|polymer|composite)\s+(?:called|named|denoted|designated)\s+[""]?(\w[\w\s\-]+?)[""]?',
            r'(?:synthesi[sz]ed|prepared|fabricated|developed)\s+(?:a\s+)?(?:novel|new|original)?\s*(?:material|compound|alloy|ceramic)\s*,?\s*[""]?(\w[\w\s\-]+?)[""]?',
            r'(?:discovered|found|identified)\s+(?:a\s+)?(?:new|novel)?\s*(?:phase|compound|material|alloy)\s*(?:of|,)?\s*[""]?(\w[\w\s\-]+?)[""]?',
            # 通用/灵活英文
            r'(?:exhibit(?:s|ed)?|show(?:s|ed|n)?|display(?:s|ed)?|has|have|with|adopts?)\s+(?:a\s+)?(?:layered|perovskite|spinel|garnet|rock[-\s]?salt|olivine|cubic|tetragonal|hexagonal|orthorhombic|monoclinic|triclinic|face[-\s]centered\s+cubic|body[-\s]centered\s+cubic|FCC|BCC|HCP)\s+(?:crystal\s+)?(?:structure|phase)',
            r'(?:is|are|was|were)\s+(?:typically|commonly|usually|generally|widely)\s+(?:synthesi[sz]ed|prepared|fabricated|produced)\s+(?:by|via|through|using)\s+([\w\s\-\+/\(\)]+)',
            r'(?:emerg(?:ing|ed)|promising|potential|attractive|candidate)\s+(?:as\s+(?:a\s+)?)?(?:cathode|anode|electrode|electrolyte|catalyst|material|compound)',
            r'(?:alternative|replacement|substitute)\s+(?:cathode|anode|electrode|electrolyte)\s+(?:material|compound)s?',
            # 中文
            r'(?:合成|制备|开发|设计|构建)(?:了|出|了一种|了新型|了新的)?(?:新型|新|新颖)?[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|化合物|合金|陶瓷|聚合物|复合材料|催化剂|纳米材料|薄膜|涂层)',
            r'(?:发现|找到|鉴定出|筛选出)(?:了|了一种|了新型|了新的)?(?:新型|新|新颖)?[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|化合物|合金|相|物质|结构)',
            r'(?:首次|第一次|率先)(?:合成|制备|开发|报道|发现)(?:了|出)?(?:的)?(?:新型|新)?[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|化合物|合金|物质)',
            r'研制(?:了|出|了一种)(?:新型|新)?[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|化合物|合金|陶瓷|涂层|薄膜)',
        ],
        "reaction": [
            # 英文
            r'(?:novel|new|unreported)\s+(?:reaction|transformation|decomposition|oxidation|reduction|phase\s+transition)',
            r'(?:discovered|observed|found)\s+(?:a\s+)?(?:novel|new)\s+(?:reaction|mechanism|pathway)',
            r'(?:reacts?\s+with\s+\w+\s+to\s+form|undergoes?\s+(?:a\s+)?\w+\s+(?:transformation|transition|change))',
            # 通用/灵活英文
            r'(?:exhibit(?:s|ed)?|show(?:s|ed|n)?|undergo(?:es|ne)?)\s+(?:a\s+)?(?:reversible|irreversible|structural)\s+(?:phase\s+)?(?:transition|transformation|change|reaction)',
            r'(?:electrochemical|redox|intercalation|deintercalation|conversion|alloying)\s+(?:react(?:ion|ivity)|mechanism|process|behavior)',
            # 中文
            r'(?:发现|观察到|监测到|捕获到)(?:了|了一种|了新型|了新的)?(?:新型|新|前所未有)?[\w\s/\-+()（）%.×·一-鿿]*?(?:反应|相变|转变|转化|分解|氧化|还原|相转变)',
            r'(?:揭示了|阐明了|明确了|提出了)(?:的|新的)?(?:反应机理|反应路径|反应机制|相变机制|转化机制)',
            r'(?:发生了|产生了|出现了)(?:新型|新|特殊的)?(?:反应|相变|马氏体相变|共析反应|包晶反应)',
        ],
        "phenomenon": [
            # 英文
            r'(?:novel|new|unusual|unexpected|surprising)\s+(?:phenomenon|behavior|effect|property)',
            r'(?:discovered|observed|found|noticed)\s+(?:that\s+)?(?:an?\s+)?(?:unusual|unexpected|surprising|remarkable)',
            r'(?:for\s+the\s+first\s+time[,:]?\s*)(?:we\s+)?(?:report|demonstrate|show|observe|present)',
            # 通用/灵活英文
            r'(?:exhibit(?:s|ed)?|show(?:s|ed|n)?|display(?:s|ed)?)\s+(?:an?\s+)?(?:unusual|unexpected|interesting|remarkable|peculiar|unique)\s+(?:phenomenon|behavior|effect|feature|characteristic)',
            r'(?:interesting(?:ly)?|notably|remarkably|surprisingly)[,\.\s]+(?:the|this|we)\s+',
            # 中文
            r'(?:发现|观察到|注意到|探测到)(?:了|了一种)?(?:新型|新的|异常的|特殊的|有趣的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:现象|效应|行为|规律|趋势)',
            r'(?:首次|第一次)(?:发现|观察到|报道|证实|证明)(?:了|的)?(?:现象|效应|规律|行为)',
            r'(?:表现出|展现出|呈现出|显示出)(?:了)?(?:异常|特殊|独特|优异|卓越|出色)的[\w\s/\-+()（）%.×·一-鿿]*?(?:性能|特性|行为|能力)',
        ],
        "property": [
            # 英文
            r'(?:exhibited|showed|displayed|possessed)\s+(?:exceptional|outstanding|superior|enhanced|improved|high|ultrahigh)\s+(\w[\w\s\-]+?(?:property|strength|conductivity|stability|resistance|performance))',
            r'(?:measured|determined|calculated)\s+(?:the\s+)?(\w+)\s+(?:to\s+be\s+)?([\d.]+)\s*(eV|MPa|GPa|W/(?:m·K|mK)|S/cm|%|nm)',
            r'(?:band\s+gap|formation\s+energy|tensile\s+strength|yield\s+strength|hardness|conductivity|resistivity)\s+(?:of|is|was|:)',
            # 通用/灵活英文
            r'(?:exhibit(?:s|ed)?|show(?:s|ed|n)?|display(?:s|ed)?|has|have|with)\s+(?:a\s+)?(?:high|exceptional|outstanding|superior|enhanced|improved|excellent|remarkable|good)\s+([\w\s\-]+?(?:capacity|conductivity|stability|performance|efficiency|strength|hardness|resistance|structure))',
            r'(?:achieve(?:s|d)?|reach(?:es|ed)?|deliver(?:s|ed)?)\s+(?:a\s+)?(?:high|specific|theoretical)\s+(?:capacity|performance|efficiency|conductivity)\s+(?:of\s+)?([\d.]+)',
            r'(?:demonstrate(?:s|d)?|show(?:s|ed|n)?|exhibit(?:s|ed)?)\s+(?:an?\s+)?(?:excellent|outstanding|remarkable|exceptional|superb|great)\s+(?:combination\s+of\s+)?(\w[\w\s]+?)\s+(?:exceeding|above|over|of|at)\s+([\d.]+)\s*(?:eV|MPa|GPa|K|%|mAh)',
            r'(?:theoretical|specific)\s+(?:capacity|conductivity|energy|power)\s+(?:of|is|was|:)\s*([\d.]+)\s*(?:mAh|eV|MPa|GPa|W|S)',
            # 中文
            r'(?:表现出|展现出|具有|拥有)(?:了)?(?:优异|出色|卓越|超高|极高|良好|优秀|突出)的[\w\s/\-+()（）%.×·一-鿿]*?(?:性能|强度|硬度|韧性|导电|导热|耐腐蚀|抗氧化|稳定性)',
            r'(?:测得|测定|测试得到|计算得到)(?:的)?(?:带隙|形成能|抗拉强度|屈服强度|硬度|电导率|热导率|弹性模量)(?:为|是|达到|高达)([\d.]+)',
            r'(?:性能|强度|硬度|韧性|导电率|热导率|带隙|模量)(?:达到|高达|提升至|提高到了|增强至)([\d.]+)',
        ],
        "mechanism": [
            # 英文
            r'(?:proposed|suggested|put\s+forward|presented)\s+(?:a\s+)?(?:new|novel|alternative)\s+(?:mechanism|model|theory|explanation)',
            r'(?:revealed|elucidated|uncovered|clarified)\s+(?:the\s+)?(?:mechanism|origin|cause|reason)',
            r'(?:underlying\s+mechanism|rate-limiting\s+step|dominates?\s+the)',
            # 中文
            r'(?:提出|建议|给出|建立)(?:了|了一种|了一个)?(?:新型|新|新的|改进的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:机理|机制|模型|理论|解释|假说)',
            r'(?:揭示|阐明|澄清|解释|说明)(?:了|出)?(?:的)?(?:机理|机制|原因|起源|本质|规律)',
            r'(?:强化机制|韧化机制|变形机制|断裂机制|扩散机制|形核机制|长大机制)',
        ],
    }

    # 创新点识别规则 (英文 + 中文)
    INNOVATION_PATTERNS = {
        "method": [
            # 英文
            r'(?:novel|new|innovative|improved|advanced)\s+(?:method|technique|approach|strategy|procedure|protocol|route|synthesis)',
            r'(?:first\s+time|first\s+report|first\s+demonstration|first\s+application)\s+(?:of|to|in|using)',
            r'(?:developed|designed|engineered|constructed|fabricated)\s+(?:a\s+)?(?:novel|new|unique|innovative)',
            # 通用/灵活英文
            r'(?:co[\-]?precipitation|sol[\-]?gel|hydrothermal|solid[\-]?state|electrospinning|ball[\-]?milling|CVD|PVD|ALD|MBE)\s+(?:method|technique|synthesis|reaction|route|approach|process)',
            r'(?:doping|substitut(?:ing|ion)|decorating|coating|modifying)\s+(?:with|by|using)\s+([\w\s]+?)(?:to\s+(?:improve|enhance|increase|boost|achieve))?',
            # 中文
            r'(?:首次|第一次|率先)(?:提出|使用|采用|应用|实现|开发|建立)(?:了|的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:方法|技术|工艺|策略|方案|路线|途径|流程)',
            r'(?:新型|新颖|创新|改进|先进)(?:的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:方法|技术|工艺|策略|方案|路线|途径|流程)',
            r'(?:原位|实时|高通量|多尺度|跨尺度|多场耦合)(?:的)?(?:表征|测试|分析|观测|监测|检测)',
            r'(?:自主研发|自主设计|原创)(?:了|的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:方法|技术|设备|装置|系统|平台)',
        ],
        "material": [
            # 英文
            r'(?:novel|new|innovative)\s+(?:composition|formula|stoichiometry|doping|architecture|structure|design)',
            r'(?:never\s+before\s+(?:synthesized|reported|studied|investigated|explored))',
            r'(?:high-entropy|multi-principal|compositionally\s+complex|gradient|hierarchical|bio-inspired)',
            # 通用/灵活英文
            r'(?:high[\-]?entropy|multi[\-]?principal|compositionally[\-]?complex|nano[\-]?structured|hierarchical|porous|core[\-]?shell|gradient)\s+(?:alloy|material|structure|coating)',
            r'(?:alternative|cheaper|safer|sustainable|eco[\-]?friendly)\s+(?:to|for|than)\s+(?:conventional|traditional|commercial)\s+([\w\s]+)',
            # 中文
            r'(?:新型|新颖|创新)(?:的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:成分|配方|化学计量比|掺杂|结构|设计|体系|组合)',
            r'(?:高熵|多主元|梯度|多级|仿生|纳米)[\w\s/\-+()（）%.×·一-鿿]*?(?:合金|材料|结构|涂层|薄膜|复合材料)',
            r'(?:首次|从未)(?:被)?(?:合成|报道|研究|探索|制备|开发)',
            r'(?:新型|新)?(?:多孔|核壳|中空|层状|纤维|颗粒)[\w\s/\-+()（）%.×·一-鿿]*?(?:结构|形貌|形态)',
        ],
        "theory": [
            # 英文
            r'(?:new|novel|modified|extended|generalized)\s+(?:theory|model|equation|framework|formalism|simulation)',
            r'(?:first-principles|ab\s+initio|DFT|molecular\s+dynamics|finite\s+element|phase-field)\s+(?:calculation|simulation|modeling|study)',
            # 中文
            r'(?:新|新型|改进|扩展)(?:的)?[\w\s/\-+()（）%.×·一-鿿]*?(?:理论|模型|方程|框架|公式|模拟|计算方法)',
            r'(?:第一性原理|密度泛函|分子动力学|有限元|相场)(?:计算|模拟|研究|建模)',
            r'(?:推导|建立|提出|构建)(?:了|出)?(?:的)?(?:新|理论|解析|预测)(?:公式|方程|模型|表达式)',
        ],
        "application": [
            # 英文
            r'(?:demonstrated|showcased|validated|proved)\s+(?:the\s+)?(?:potential|feasibility|applicability|utility)\s+(?:of|for|in|as)',
            r'(?:for\s+the\s+first\s+time\s+(?:applied|used|employed|utilized|demonstrated)\s+(?:in|for|as|to))',
            # 中文
            r'(?:首次|率先|第一次)(?:将|在|应用于|用于|作为)[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|器件|电池|催化|传感|涂层|防护|储能)',
            r'(?:验证|证实|证明|展示)(?:了)(?:该)?(?:材料|方法|技术)(?:在|用于|作为)(?:方面的|领域)(?:应用潜力|可行性|前景)',
        ],
        "performance": [
            # 英文
            r'(?:outperform|surpass|exceed|beat|outclass)(?:s|ed|ing)?\s+(?:existing|conventional|commercial|state-of-the-art|current)',
            r'(?:record(?:-| )?(?:high|breaking))\s+(?:performance|efficiency|strength|conductivity|capacity)',
            r'(?:significantly|substantially|dramatically|remarkably|notably)\s+(?:enhanced|improved|increased|boosted|augmented)',
            # 中文
            r'(?:远超|超过|优于|胜过|大幅超越)(?:现有|传统|商业|已知|此前)[\w\s/\-+()（）%.×·一-鿿]*?(?:材料|方法|性能|指标|纪录)',
            r'(?:创纪录|破纪录)(?:的)?(?:性能|效率|强度|硬度|韧性|导电率|容量)',
            r'(?:显著|大幅|明显|极大|有效)(?:提高|提升|增强|改善|优化|增加)(?:了)?[\w\s/\-+()（）%.×·一-鿿]*?(?:性能|强度|硬度|韧性|效率|稳定性)',
        ],
    }

    # 不足/局限识别规则 (英文 + 中文)
    SHORTCOMING_PATTERNS = {
        "method": [
            # 英文
            r'(?:limitation|limitations|drawback|drawbacks|shortcoming|shortcomings|disadvantage|weakness)\s+(?:of|in|with)?\s*(?:the\s+)?(?:method|technique|approach|procedure)',
            r'(?:however|although|nevertheless|nonetheless|despite|while)\b[^.]{0,100}?\b(?:limitation|limited|limiting|restricted|constrained)',
            # 通用/灵活英文
            r'(?:method|technique|approach|procedure|protocol|route)\s+(?:is|was|remains|suffers\s+from|has)\s+(?:limitation|drawback|shortcoming|disadvantage|weakness|issue)',
            r'(?:time[\-]?consuming|expensive|complex|complicated|difficult|challenging|tedious)\s+(?:method|technique|approach|process|procedure|synthesis)',
            r'(?:despite|although|while)\b[^.]{0,200}?\b(?:still|remain(?:s|ing)?|yet|continue(?:s|d)?)\s+(?:a\s+)?(?:challenge|problem|issue|obstacle|concern|drawback|limitation)',
            # 中文
            r'(?:方法|技术|工艺|手段|途径)(?:的|方面|上)?(?:局限|不足|缺点|缺陷|问题|弊端)',
            r'(?:但|但是|然而|不过|尽管如此)[^。]{0,80}(?:局限|不足|缺陷|缺点|受限|限制)',
            r'(?:不)(?:适用于|适合|能用于|可用于|够|足以|具备|具有)',
        ],
        "scope": [
            # 英文
            r'(?:only|solely|exclusively|merely|just)\s+(?:valid|applicable|relevant|tested|studied|investigated)\s+(?:for|to|in|under|at)',
            r'(?:small|limited|narrow|restricted)\s+(?:sample|dataset|range|scope|temperature|pressure|concentration)',
            r'(?:further|more|additional|extensive|comprehensive)\s+(?:stud|investigation|research|work|experiment|validation|verification)\s+(?:is|are|would\s+be)\s+(?:needed|required|necessary|essential)',
            # 通用/灵活英文
            r'(?:only|solely|exclusively|merely|just)\s+(?:tested|studied|investigated|demonstrated|valid|applicable)\s+(?:for|to|in|under|at|on)\s+(?:a\s+)?(?:single|one|specific|certain|limited)',
            r'(?:low|poor|inferior|insufficient|inadequate|unsatisfactory)\s+(?:thermal\s+)?(?:stability|cycling|retention|performance|capacity|efficiency|conductivity|strength)',
            r'(?:despite|although|while)\b[^.]{0,150}?\b(?:are|is|were|was)\s+(?:being\s+)?(?:develop(?:ed|ing)|investigat(?:ed|ing)|explor(?:ed|ing)|sought|need(?:ed|ing)|requir(?:ed|ing))\s+(?:for|to|as)\s+(?:achieve\s+)?(?:improved|better|enhanced|higher|lower|cheaper|more\s+(?:stable|abundant|sustainable))',
            r'(?:limiting|limited|restricts?|restricted|confined)\s+(?:its|the|their|to)\s+(?:application|use|performance|efficiency|activity|functionality|operation|photoactivity|absorption)\s+(?:to|in|at|under)',
            r'(?:various|several|multiple)\s+(?:strategies|approaches|methods|attempts|efforts)\s+(?:including|such\s+as|like)\s+.+?\s+(?:have\s+been|are\s+being|were)\s+(?:explored|investigated|developed|pursued|studied)\s+(?:to|in\s+order\s+to)\s+(?:overcome|address|mitigate|solve|improve|extend|enhance)',
            # 中文
            r'(?:仅|只|仅仅|仅限)(?:适用于|限于|针对|在.*?条件下|在.*?范围内)',
            r'(?:样本|数据|范围|温度|压力|浓度|组分)(?:有限|较小|较窄|不足|偏少|偏窄)',
            r'(?:仍需|还需要|有待|尚需|需要)(?:进一步|更多|深入|大量|系统)(?:的)?(?:研究|探索|验证|实验|调查|工作)',
            r'(?:无法|不能|难以)(?:解释|说明|描述|预测|涵盖|概括|推广)',
        ],
        "validation": [
            # 英文
            r'(?:lacking|lacks|lack|without|absent|missing)\s+(?:experimental\s+)?(?:validation|verification|confirmation|proof|evidence|demonstration)',
            r'(?:discrepancy|discrepancies|deviation|difference|mismatch|gap)\s+(?:between|of)\s+(?:experimental|theoretical|calculated|predicted|modeled)',
            # 中文
            r'(?:缺少|缺乏|缺失|没有|未)(?:进行|做|开展|实施)(?:实验)?(?:验证|检验|确认|证明|佐证|校准)',
            r'(?:实验|理论|计算|模拟|预测)(?:与|和)(?:理论|实验|计算|模拟|预测)(?:之?间?)(?:存在|有|出现)(?:偏差|差异|差距|偏离|出入)',
            r'(?:误差|不确定度|偏差)(?:在|约为|达到|高达|不超过)',
            r'(?:未)(?:验证|检验|确认|测试|校准|标定)',
        ],
        "comparison": [
            # 英文
            r'(?:did\s+not|failed\s+to|was\s+not|were\s+not)\s+(?:compare|compared|benchmark|benchmarked)(?:\s+(?:with|against|to))?',
            r'(?:absence|lack)\s+(?:of|in)?\s*(?:comparison|benchmark|baseline|reference|control|standard)',
            r'(?:lower[\-]?cost|cheaper|more\s+abundant|more\s+sustainable|more\s+stable|less\s+expensive|less\s+toxic)\s+(?:alternative|option|replacement|substitute|candidate)',
            # 中文
            r'(?:未与|没有与|未和|没有和|缺乏与)(?:现有|传统|商业|文献|报道)(?:进行)?(?:对比|比较|对标)',
            r'(?:缺少|缺乏|没有)(?:对比|对照|比较|基准|参考|标准)(?:实验|样品|数据|组)',
        ],
        "gap": [
            # 英文
            r'(?:remains?\s+(?:unclear|unknown|unresolved|unexplained|unanswered|elusive|open|ambiguous))',
            r'(?:future\s+(?:work|research|study|investigation|effort))\s+(?:should|must|needs?\s+to|could|would)\s+(?:focus|address|explore|investigate|examine|clarify)',
            r'(?:open\s+(?:question|problem|issue|challenge)|knowledge\s+gap|research\s+gap)',
            # 通用/灵活英文
            r'(?:remain(?:s|ing)?)\s+(?:a\s+)?(?:challenge|problem|issue|obstacle|bottleneck|open\s+question)',
            r'(?:further|more|additional|extensive)\s+(?:research|study|investigation|work|effort)\s+(?:is|are|would\s+be|should\s+be)\s+(?:needed|required|necessary|essential|warranted)',
            r'(?:yet|still|however|nevertheless)\s+(?:to\s+be|remains?\s+to\s+be)\s+(?:fully\s+)?(?:understood|elucidated|clarified|explored|investigated|determined|established|demonstrated)',
            # 中文
            r'(?:仍)(?:不清楚|未知|未解决|未解释|未回答|悬而未决|存在争议|有待商榷)',
            r'(?:未来|今后|下一步)(?:的)?(?:工作|研究|探索|方向)(?:应|应该|需要|可以|将)(?:关注|聚焦|开展|深入|着重|尝试)',
            r'(?:尚)(?:不清楚|需研究|待探索|待解决|待阐明|有待深入研究)',
        ],
    }

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com"
        self.model = model or "deepseek-chat"
        self.use_llm = bool(api_key)

    def analyze(self, parsed_doc: dict, entities: list = None,
                relations: list = None, triplets: list = None) -> DeepAnalysisResult:
        """对单篇论文进行深度分析.

        Args:
            parsed_doc: PDFParser.parse() 的输出
            entities: 提取的实体列表
            relations: 关系列表
            triplets: 三元组列表

        Returns:
            DeepAnalysisResult with discoveries, innovations, shortcomings
        """
        text = parsed_doc.get("raw_text", "")
        title = parsed_doc.get("filename", "")
        if parsed_doc.get("metadata", {}).get("title"):
            title = parsed_doc["metadata"]["title"]

        result = DeepAnalysisResult(
            paper_title=title,
            raw_text_snippet=text[:5000],
        )

        if self.use_llm:
            result = self._llm_analyze(text, title, result)
        else:
            result = self._rule_based_analyze(text, parsed_doc, entities, triplets, result)

        return result

    def _rule_based_analyze(self, text: str, parsed_doc: dict,
                            entities: list, triplets: list,
                            result: DeepAnalysisResult) -> DeepAnalysisResult:
        """基于规则的分析管道."""
        # 提取发现
        result.discoveries = self._extract_discoveries(text)

        # 提取创新点
        result.innovations = self._extract_innovations(text)

        # 提取不足
        result.shortcomings = self._extract_shortcomings(text)

        # 提取摘要 (基于已提取的发现/创新/不足)
        result.summary = self._generate_summary(parsed_doc, entities, triplets, result)

        # 提取关键词
        result.keywords = self._extract_keywords(text, entities)

        # 提取方法学
        result.methodology = self._extract_methodology(text)

        # 主要发现
        result.main_finding = self._extract_main_finding(text, triplets)

        return result

    def _extract_discoveries(self, text: str) -> List[PaperDiscovery]:
        """从论文中提取科学发现."""
        discoveries = []
        for disc_type, patterns in self.DISCOVERY_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    raw = m.group(0).strip()
                    if len(raw) < 4:
                        continue
                    desc = self._expand_to_clause(text, m.start(), m.end(), max_len=300)
                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    context = text[start:end].strip()
                    discoveries.append(PaperDiscovery(
                        type=disc_type,
                        description=desc,
                        evidence=context,
                        confidence=0.6,
                    ))
        return self._deduplicate_discoveries(discoveries)

    def _extract_innovations(self, text: str) -> List[InnovationPoint]:
        """提取论文创新点."""
        innovations = []
        for cat, patterns in self.INNOVATION_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    raw = m.group(0).strip()
                    if len(raw) < 4:
                        continue
                    desc = self._expand_to_clause(text, m.start(), m.end(), max_len=300)
                    context_start = max(0, m.start() - 80)
                    context_end = min(len(text), m.end() + 80)
                    evidence = text[context_start:context_end].strip()

                    significance = self._infer_significance(desc, cat)

                    innovations.append(InnovationPoint(
                        category=cat,
                        description=desc,
                        significance=significance,
                        evidence=evidence,
                    ))
        return self._deduplicate_innovations(innovations)

    def _extract_shortcomings(self, text: str) -> List[PaperShortcoming]:
        """提取论文不足之处."""
        shortcomings = []
        severity_keywords = {
            "major": ["critical", "significant", "substantial", "fundamental",
                      "severely", "serious", "major"],
            "moderate": ["limitation", "limited", "however", "although", "despite",
                        "drawback", "shortcoming"],
            "minor": ["minor", "slight", "marginal", "potential", "may", "might"],
        }

        for cat, patterns in self.SHORTCOMING_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    raw = m.group(0).strip()
                    if len(raw) < 4:
                        continue
                    desc = self._expand_to_clause(text, m.start(), m.end(), max_len=400)

                    context_start = max(0, m.start() - 80)
                    context_end = min(len(text), m.end() + 80)
                    evidence = text[context_start:context_end].strip()

                    severity = "moderate"
                    desc_lower = desc.lower()
                    if any(kw in desc_lower for kw in severity_keywords["major"]):
                        severity = "major"
                    elif any(kw in desc_lower for kw in severity_keywords["minor"]):
                        severity = "minor"

                    shortcomings.append(PaperShortcoming(
                        category=cat,
                        description=desc,
                        severity=severity,
                        evidence=evidence,
                    ))
        return self._deduplicate_shortcomings(shortcomings)

    @staticmethod
    def _expand_to_clause(text: str, start: int, end: int, max_len: int = 300) -> str:
        """Extend a regex match to cover the surrounding clause/sentence."""
        # Expand left to previous sentence delimiter or max 40 chars
        left = max(0, start - 40)
        left = max(left, start - 40)
        # Find nearest sentence delimiter to the left
        for i in range(start - 1, left - 1, -1):
            if text[i] in '。！？\n.?!':
                left = i + 1
                break
        # Expand right to next sentence delimiter or max 80 chars
        right = min(len(text), end + 80)
        for i in range(end, right):
            if text[i] in '。！？\n.?!':
                right = i + 1
                break
        clause = text[left:right].strip()
        if len(clause) > max_len:
            clause = clause[:max_len] + '...'
        return clause

    def _generate_summary(self, parsed_doc: dict, entities: list = None,
                          triplets: list = None, result: 'DeepAnalysisResult' = None) -> str:
        """生成论文内容总结.

        优先返回原文摘要，在此基础上补充分析结果。
        """
        text = parsed_doc.get("raw_text", "")
        abstract = parsed_doc.get("abstract", "")

        parts = []

        # 1. 原文摘要作为基础
        if abstract and len(abstract) > 50:
            parts.append(abstract[:1500])
        else:
            # 取首段作为原文概述
            first_para = text.split("\n\n")[0] if "\n\n" in text else text[:800]
            parts.append(first_para[:1200])

        # 2. 基于分析结果的补充总结
        if result:
            analysis_parts = []

            if result.discoveries:
                d_items = []
                for d in result.discoveries[:5]:
                    d_items.append(f"· [{d.type}] {d.description[:100]}")
                analysis_parts.append("**主要发现:**\n" + "\n".join(d_items))

            if result.innovations:
                i_items = []
                for i in result.innovations[:3]:
                    i_items.append(f"· [{i.category}] {i.description[:100]}")
                analysis_parts.append("**创新点:**\n" + "\n".join(i_items))

            if result.shortcomings:
                s_items = []
                for s in result.shortcomings[:3]:
                    s_items.append(f"· [{s.category}] {s.description[:100]}")
                analysis_parts.append("**不足之处:**\n" + "\n".join(s_items))

            if analysis_parts:
                parts.append("\n---\n" + "\n\n".join(analysis_parts))

        # 3. 材料与性能补充
        if entities:
            mat_entities = [e.text for e in entities if hasattr(e, 'entity_type')
                           and e.entity_type == "material"]
            if mat_entities:
                parts.append(f"\n**涉及材料**: {', '.join(set(mat_entities[:8]))}")

        if triplets:
            key_triplets = [t for t in triplets if t.material and t.property][:5]
            if key_triplets:
                parts.append("**关键数据**: " + "; ".join(
                    f"{t.material} → {t.property} = {t.value}" for t in key_triplets
                ))

        return "\n".join(parts) if len("\n".join(parts)) < 3000 else "\n".join(parts)[:3000]

    def _extract_keywords(self, text: str, entities: list = None) -> List[str]:
        """提取关键词."""
        keywords = []

        # 从实体中提取
        if entities:
            for e in entities:
                text_attr = getattr(e, 'text', str(e))
                type_attr = getattr(e, 'entity_type', '')
                if type_attr in ("material", "property", "synthesis_method", "microstructure"):
                    if text_attr.lower() not in [k.lower() for k in keywords]:
                        keywords.append(text_attr)

        # 从致谢/关键词段提取
        kw_section = re.search(
            r'(?:Keywords?|KEYWORDS?|Key\s+words?)[\.:\-–—]*\s*(.+?)(?:\n\n|\n\w)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if kw_section:
            kw_text = kw_section.group(1).strip()
            for sep in [",", ";", "·", "•"]:
                if sep in kw_text:
                    keywords.extend(k.strip() for k in kw_text.split(sep) if k.strip())
                    break
            else:
                keywords.extend(kw_text.split())

        return list(dict.fromkeys(keywords))[:30]

    def _extract_methodology(self, text: str) -> str:
        """提取论文的研究方法."""
        method_section = re.search(
            r'(?:Experimental|Methods?|Methodology|EXPERIMENTAL|METHODS?|METHODOLOGY)\b[^\n]*\n(.+?)(?:\n(?:Results?|Discussion|RESULTS?|DISCUSSION)\b|\Z)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if method_section:
            return method_section.group(1).strip()[:2000]

        method_keywords = [
            "XRD", "SEM", "TEM", "XPS", "FTIR", "Raman", "AFM", "DSC", "TGA",
            "sol-gel", "CVD", "PVD", "hydrothermal", "solvothermal", "co-precipitation",
            "ball milling", "spark plasma sintering", "hot pressing", "annealing",
            "DFT", "molecular dynamics", "finite element", "machine learning",
        ]
        found = []
        for kw in method_keywords:
            if kw.lower() in text.lower():
                found.append(kw)
        return ", ".join(found) if found else ""

    def _extract_main_finding(self, text: str, triplets: list = None) -> str:
        """提取论文的主要发现."""
        patterns = [
            r'(?:main|key|principal|major|central|important|significant)\s+(?:finding|result|discovery|conclusion|outcome|observation)\b[^.]*?[.!](?:\s|$)',
            r'(?:we\s+(?:found|discovered|demonstrate|show|reveal|report|conclude)\s+that\b[^.]*?[.!](?:\s|$))',
            r'(?:in\s+summary|in\s+conclusion|to\s+conclude|overall)\b[^.]*?[.!](?:\s|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(0).strip()[:500]

        if triplets and len(triplets) > 0:
            best = triplets[0]
            return f"发现 {best.material} 的 {best.property} = {best.value}"

        return ""

    def _infer_significance(self, description: str, category: str) -> str:
        """推断创新点的科学意义."""
        desc_lower = description.lower()
        templates = {
            "method": "提供了一种新的实验/计算方法, 可推广至类似体系的研究",
            "material": "拓展了材料设计空间, 为开发新型功能材料提供候选",
            "theory": "深化了对机理/规律的理论理解, 为预测和优化提供依据",
            "application": "验证了材料在实际器件/工程中的应用潜力",
            "performance": "实现了性能的显著提升, 接近或超越现有最佳水平",
        }
        base = templates.get(category, "推动了材料科学领域的发展")

        if "first" in desc_lower or "never" in desc_lower:
            base = "首次实现/报道, 具有开创性意义。" + base
        if "record" in desc_lower or "outperform" in desc_lower:
            base = "突破了现有性能瓶颈。" + base
        return base

    def _llm_analyze(self, text: str, title: str,
                     result: DeepAnalysisResult) -> DeepAnalysisResult:
        """使用LLM进行深度语义分析."""
        if not self.api_key:
            return self._rule_based_analyze(text, {}, None, None, result)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            prompt = self._build_llm_prompt(title, text[:8000])
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            output = response.choices[0].message.content
            result = self._parse_llm_output(output, result)
        except Exception as e:
            print(f"  [LLM analysis fallback to rules] {e}")
            result = self._rule_based_analyze(text, {}, None, None, result)

        return result

    def _build_llm_prompt(self, title: str, text: str) -> str:
        return f"""请分析以下材料科学论文, 提取结构化信息。用中文回答。

论文标题: {title}

论文内容 (前8000字符):
{text}

请按以下JSON格式输出 (不要加额外解释):

{{
  "discoveries": [
    {{"type": "material/reaction/phenomenon/property/mechanism", "description": "...", "evidence": "原文摘录"}}
  ],
  "innovations": [
    {{"category": "method/material/theory/application/performance", "description": "...", "significance": "为什么重要"}}
  ],
  "shortcomings": [
    {{"category": "method/scope/validation/comparison/gap", "description": "...", "severity": "minor/moderate/major"}}
  ],
  "summary": "200字中文综述",
  "main_finding": "一句话核心发现",
  "methodology": "论文使用的研究方法",
  "keywords": ["关键词1", "关键词2"]
}}"""

    def _parse_llm_output(self, output: str, result: DeepAnalysisResult) -> DeepAnalysisResult:
        """解析LLM的JSON输出."""
        try:
            import json
            json_match = re.search(r'\{[\s\S]*\}', output)
            if not json_match:
                return result
            data = json.loads(json_match.group())

            for d in data.get("discoveries", []):
                result.discoveries.append(PaperDiscovery(
                    type=d.get("type", ""),
                    description=d.get("description", ""),
                    evidence=d.get("evidence", ""),
                    confidence=0.85,
                ))

            for inn in data.get("innovations", []):
                result.innovations.append(InnovationPoint(
                    category=inn.get("category", ""),
                    description=inn.get("description", ""),
                    significance=inn.get("significance", ""),
                ))

            for s in data.get("shortcomings", []):
                result.shortcomings.append(PaperShortcoming(
                    category=s.get("category", ""),
                    description=s.get("description", ""),
                    severity=s.get("severity", "moderate"),
                ))

            result.summary = data.get("summary", result.summary)
            result.main_finding = data.get("main_finding", result.main_finding)
            result.methodology = data.get("methodology", result.methodology)
            result.keywords = data.get("keywords", result.keywords)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [LLM parse error] {e}")

        return result

    def _deduplicate_discoveries(self, items: List[PaperDiscovery]) -> List[PaperDiscovery]:
        seen = set()
        unique = []
        for item in items:
            key = item.description.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:10]

    def _deduplicate_innovations(self, items: List[InnovationPoint]) -> List[InnovationPoint]:
        seen = set()
        unique = []
        for item in items:
            key = item.description.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:8]

    def _deduplicate_shortcomings(self, items: List[PaperShortcoming]) -> List[PaperShortcoming]:
        seen = set()
        unique = []
        for item in items:
            key = item.description.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:8]


PROMPT_SYSTEM = """你是一位材料科学领域的资深审稿人, 擅长分析学术论文的创新点和局限性。
你需要仔细阅读论文内容, 提取以下信息:

1. **科学发现** (discoveries): 论文发现了什么? 新物质? 新反应? 新现象? 新性能? 新机理?
2. **创新点** (innovations): 论文的创新在哪里? 方法创新? 材料创新? 理论创新? 性能突破?
3. **不足与局限** (shortcomings): 论文的不足之处? 方法局限? 验证不充分? 适用范围窄?
4. **方法论** (methodology): 论文用了哪些研究方法?
5. **核心发现** (main_finding): 一句话总结最重要的发现

请基于论文实际内容做客观分析, 不要编造信息。如果某方面没有足够信息, 可以标注"未提及"。
用中文回答, 保持学术性和专业性。"""
