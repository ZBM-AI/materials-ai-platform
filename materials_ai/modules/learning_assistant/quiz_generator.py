"""自动出题系统 — 选择题 + 计算题 + 详细解答"""

import os
import re
import random
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .knowledge_base import KnowledgeBase, TextChunk


@dataclass
class QuizQuestion:
    """题目."""

    question_type: str  # "mcq" | "calculation" | "true_false" | "fill_blank"
    question: str
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"  # easy | medium | hard
    source_ref: str = ""
    points: int = 5

    def to_dict(self) -> Dict:
        return {
            "type": self.question_type,
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "source": self.source_ref,
            "points": self.points,
        }


@dataclass
class Quiz:
    """一套试卷."""

    title: str
    topic: str
    questions: List[QuizQuestion]
    total_points: int = 0
    duration_minutes: int = 30

    def __post_init__(self):
        self.total_points = sum(q.points for q in self.questions)
        self.duration_minutes = max(10, len(self.questions) * 3)


class QuizGenerator:
    """基于知识库自动出题.

    - 选择题 (MCQ): 从教材段落提取关键概念, 构造4选1
    - 计算题: 根据章节生成定量计算题
    - 判断题: 正误辨析
    """

    GENERATE_MCQ_PROMPT = """基于以下教材内容, 生成 {num} 道材料科学单项选择题 (4选1)。

教材内容:
{context}

要求:
1. 题目考察核心概念理解, 不考死记硬背的数字
2. 干扰项要有迷惑性 (常见的错误概念)
3. 提供正确答案和解析
4. 难度递进: easy(概念识记) → medium(原理理解) → hard(综合应用)

请按以下JSON格式输出:
```json
[
  {{
    "question": "题目?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct": "A",
    "explanation": "解析...",
    "difficulty": "medium"
  }}
]
```"""

    GENERATE_CALC_PROMPT = """基于以下教材内容, 生成 {num} 道材料科学计算题。

教材内容:
{context}

要求:
1. 包含具体数值, 考察公式应用
2. 提供详细的分步解答过程
3. 答案带单位

请按以下JSON格式输出:
```json
[
  {{
    "question": "计算题描述?",
    "answer": "最终答案 (含单位)",
    "steps": ["步骤1: ...", "步骤2: ...", "步骤3: ..."],
    "difficulty": "medium"
  }}
]
```"""

    def __init__(self, knowledge_base: KnowledgeBase,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None):
        self.kb = knowledge_base
        self.api_key = api_key
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def generate_quiz(self, topic: str, num_mcq: int = 8,
                      num_calc: int = 2,
                      use_llm: bool = False) -> Quiz:
        """根据主题生成一套完整试卷.

        Args:
            topic: 考察的知识点 (如 "位错与塑性变形")
            num_mcq: 选择题数量
            num_calc: 计算题数量
            use_llm: 是否使用LLM生成
        Returns:
            Quiz
        """
        results = self.kb.search(topic, k=8)
        if not results:
            return Quiz(
                title=f"关于 {topic} 的练习",
                topic=topic,
                questions=[],
            )

        questions = []

        if use_llm and self.api_key:
            context = self._build_quiz_context(results)
            mcqs = self._llm_generate_mcq(context, num_mcq)
            calcs = self._llm_generate_calc(context, num_calc)
            questions.extend(mcqs)
            questions.extend(calcs)
        else:
            questions = self._rule_based_quiz(topic, results, num_mcq, num_calc)

        return Quiz(
            title=f"《材料科学基础》— {topic} 练习题",
            topic=topic,
            questions=questions,
        )

    def generate_mcq_batch(self, topic: str, num: int = 10,
                           difficulty: str = None) -> List[QuizQuestion]:
        """批量生成选择题."""
        results = self.kb.search(topic, k=5)
        if not results:
            return []
        return self._rule_based_mcq(topic, results, num)

    def generate_calculation(self, topic: str) -> List[QuizQuestion]:
        """生成计算题."""
        results = self.kb.search(topic, k=5)
        if not results:
            return []
        return self._rule_based_calc(topic, results)

    def _build_quiz_context(self, results: List[Tuple[TextChunk, float]],
                            max_chars: int = 4000) -> str:
        parts = []
        total = 0
        for chunk, _ in results:
            text = chunk.text
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    def _rule_based_quiz(self, topic: str,
                         results: List[Tuple[TextChunk, float]],
                         num_mcq: int, num_calc: int) -> List[QuizQuestion]:
        """基于规则模板的本地出题 (无LLM)."""
        questions = []

        mcqs = self._rule_based_mcq(topic, results, num_mcq)
        questions.extend(mcqs)

        calcs = self._rule_based_calc(topic, results)
        questions.extend(calcs[:num_calc])

        return questions

    def _rule_based_mcq(self, topic: str,
                        results: List[Tuple[TextChunk, float]],
                        num: int) -> List[QuizQuestion]:
        """从教材段落自动构造选择题.

        策略: 提取关键句子 → 挖空关键词 → 生成干扰项
        """
        mcq_templates = {
            "位错": [
                {
                    "question": "刃型位错的柏氏矢量与位错线方向的关系是?",
                    "options": ["A. 平行", "B. 垂直", "C. 成45°角", "D. 无关"],
                    "correct": "B",
                    "explanation": "刃型位错的柏氏矢量垂直于位错线方向, 而螺型位错的柏氏矢量平行于位错线方向。这是区分两种位错的关键特征。",
                    "difficulty": "easy",
                },
                {
                    "question": "位错攀移 (climb) 需要哪种扩散机制?",
                    "options": ["A. 间隙扩散", "B. 空位扩散", "C. 晶界扩散", "D. 表面扩散"],
                    "correct": "B",
                    "explanation": "位错攀移依靠空位向位错核心扩散 (正攀移) 或空位从位错核心扩散出去 (负攀移), 因此是空位扩散控制的非保守运动。",
                    "difficulty": "medium",
                },
                {
                    "question": "根据泰勒公式 τ = αGb√ρ, 流变应力与位错密度的关系是?",
                    "options": ["A. τ ∝ ρ", "B. τ ∝ √ρ", "C. τ ∝ ρ²", "D. τ 与 ρ 无关"],
                    "correct": "B",
                    "explanation": "泰勒硬化公式为 τ = τ₀ + αGb√ρ, 流变应力正比于位错密度的平方根。",
                    "difficulty": "medium",
                },
                {
                    "question": "FCC晶体中全位错的柏氏矢量通常为?",
                    "options": ["A. a/2<111>", "B. a/2<110>", "C. a<100>", "D. a/3<112>"],
                    "correct": "B",
                    "explanation": "FCC晶体中全位错的柏氏矢量为 a/2<110> 型, 因其是FCC的最短平移矢量。a/2<111> 是BCC的全位错柏氏矢量。",
                    "difficulty": "hard",
                },
            ],
            "相图": [
                {
                    "question": "Fe-C相图中, 共析点的含碳量是多少?",
                    "options": ["A. 0.022%", "B. 0.77%", "C. 2.11%", "D. 4.30%"],
                    "correct": "B",
                    "explanation": "Fe-C相图中, 共析点含碳量为0.77wt%, 温度为727°C, 反应为 γ → α + Fe₃C (珠光体转变)。",
                    "difficulty": "easy",
                },
                {
                    "question": "杠杆定律计算: 含碳0.45%的钢在共析温度稍下, 珠光体的质量分数是?",
                    "options": ["A. 约33%", "B. 约57%", "C. 约75%", "D. 约100%"],
                    "correct": "B",
                    "explanation": "根据杠杆定律: W(珠光体) = (0.45-0.022)/(0.77-0.022) ≈ 57.2%。先共析铁素体约占42.8%。",
                    "difficulty": "medium",
                },
                {
                    "question": "三元相图的等温截面图有几个自由度 (恒压)?",
                    "options": ["A. 0", "B. 1", "C. 2", "D. 3"],
                    "correct": "C",
                    "explanation": "恒压下, F = C - P + 1。三元系(C=3)在等温截面上, 单相区(P=1)自由度F=3-1+1-1(恒温)=2。",
                    "difficulty": "hard",
                },
            ],
            "晶体结构": [
                {
                    "question": "FCC晶体的致密度 (APF) 是多少?",
                    "options": ["A. 0.68", "B. 0.74", "C. 0.52", "D. 0.80"],
                    "correct": "B",
                    "explanation": "FCC致密度 = 4 × (4/3)π(√2a/4)³ / a³ = π√2/6 ≈ 0.74。BCC的致密度为0.68。",
                    "difficulty": "easy",
                },
                {
                    "question": "NaCl晶体中, 每个Na⁺的最近邻Cl⁻配位数是?",
                    "options": ["A. 4", "B. 6", "C. 8", "D. 12"],
                    "correct": "B",
                    "explanation": "NaCl型结构 (岩盐结构) 中, 阴阳离子均为六配位, 配位八面体。属于Fm-3m空间群。",
                    "difficulty": "easy",
                },
                {
                    "question": "BCC晶体中四面体间隙的位置在?",
                    "options": [
                        "A. 面心位置 {1/2, 1/2, 0}",
                        "B. {1/2, 1/4, 0} 及其等效位置",
                        "C. 体心位置 {1/2, 1/2, 1/2}",
                        "D. 角顶位置 {0, 0, 0}",
                    ],
                    "correct": "B",
                    "explanation": "BCC的四面体间隙位于{1/2, 1/4, 0}及其等效位置, 每个晶胞含12个四面体间隙, 半径约为0.29R。",
                    "difficulty": "hard",
                },
            ],
            "扩散": [
                {
                    "question": "菲克第一定律描述的是哪种扩散?",
                    "options": ["A. 非稳态扩散", "B. 稳态扩散", "C. 晶界扩散", "D. 表面扩散"],
                    "correct": "B",
                    "explanation": "菲克第一定律 J = -D(dC/dx) 描述稳态扩散 (浓度不随时间变化)。菲克第二定律 ∂C/∂t = D∂²C/∂x² 描述非稳态扩散。",
                    "difficulty": "easy",
                },
                {
                    "question": "扩散系数 D 的温度依赖性遵循什么关系?",
                    "options": [
                        "A. D ∝ T",
                        "B. D = D₀ exp(-Q/RT)",
                        "C. D = D₀ exp(Q/RT)",
                        "D. D 与温度无关",
                    ],
                    "correct": "B",
                    "explanation": "扩散系数遵循阿伦尼乌斯关系 D = D₀ exp(-Q/RT), ln D 与 1/T 呈线性关系, 斜率 = -Q/R。",
                    "difficulty": "easy",
                },
                {
                    "question": "柯肯达尔 (Kirkendall) 效应证明了什么?",
                    "options": [
                        "A. 间隙扩散比空位扩散快",
                        "B. 不同组元的扩散速率不同 (空位机制)",
                        "C. 晶界是快速扩散通道",
                        "D. 位错可以加速扩散",
                    ],
                    "correct": "B",
                    "explanation": "柯肯达尔效应 (惰性标志物移动) 证明了在置换固溶体中, 不同组元的本征扩散系数不同, 扩散主要由空位交换机制实现。",
                    "difficulty": "medium",
                },
            ],
            "力学性能": [
                {
                    "question": "屈服强度 σ_y 与晶粒直径 d 的 Hall-Petch 关系式是?",
                    "options": [
                        "A. σ_y = σ₀ + kd",
                        "B. σ_y = σ₀ + kd⁻¹ᐟ²",
                        "C. σ_y = σ₀ + kd²",
                        "D. σ_y = σ₀ exp(-kd)",
                    ],
                    "correct": "B",
                    "explanation": "Hall-Petch关系: σ_y = σ₀ + k_y·d⁻¹ᐟ², 晶粒越细, 屈服强度越高。这是细晶强化的理论基础。",
                    "difficulty": "easy",
                },
                {
                    "question": "疲劳断裂的典型断口特征是什么?",
                    "options": [
                        "A. 河流花样", "B. 韧窝", "C. 海滩条纹 (Beach marks)", "D. 沿晶断裂面",
                    ],
                    "correct": "C",
                    "explanation": "疲劳断口的典型特征包括: 疲劳源区、海滩条纹(宏观)、疲劳辉纹(微观)、最终瞬断区。河流花样是解理断裂的特征。",
                    "difficulty": "medium",
                },
            ],
            "强化机制": [
                {"question": "以下哪种强化机制不改变位错密度?", "options": ["A. 加工硬化", "B. 细晶强化", "C. 固溶强化", "D. 第二相强化"], "correct": "B", "explanation": "细晶强化通过增加晶界面积阻碍位错运动, 不改变晶粒内部位错密度。加工硬化直接增加位错密度。", "difficulty": "medium"},
                {"question": "Orowan绕过机制描述的是哪种强化?", "options": ["A. 固溶强化", "B. 细晶强化", "C. 沉淀强化(第二相)", "D. 相变强化"], "correct": "C", "explanation": "Orowan机制: 位错绕过不可变形第二相粒子, 留下位错环, 所需应力 τ = Gb/L。适用于沉淀强化/弥散强化。", "difficulty": "medium"},
                {"question": "固溶强化的主要机理是?", "options": ["A. 晶界阻碍位错", "B. 溶质原子引起晶格畸变阻碍位错", "C. 第二相粒子钉扎位错", "D. 增加位错密度"], "correct": "B", "explanation": "固溶强化通过溶质原子与基体原子尺寸差异引起晶格畸变, 与位错产生弹性交互作用, 阻碍位错滑移。", "difficulty": "easy"},
            ],
            "凝固": [
                {"question": "临界形核功 ΔG* 与过冷度 ΔT 的关系是?", "options": ["A. ΔG* ∝ ΔT", "B. ΔG* ∝ 1/ΔT", "C. ΔG* ∝ 1/ΔT²", "D. ΔG* ∝ ΔT²"], "correct": "C", "explanation": "临界形核功 ΔG* ∝ 1/ΔT², 过冷度越大, 形核功越小, 形核越容易。", "difficulty": "medium"},
                {"question": "均匀形核和非均匀形核的主要区别是?", "options": ["A. 形核温度不同", "B. 是否依赖外来表面降低形核功", "C. 晶体结构不同", "D. 冷却速度不同"], "correct": "B", "explanation": "非均匀形核利用模具壁或杂质表面降低形核功, 因此形核率远高于均匀形核。", "difficulty": "easy"},
            ],
            "固态相变": [
                {"question": "马氏体相变的主要特征不包括?", "options": ["A. 无扩散型相变", "B. 切变共格", "C. 依赖原子长程扩散", "D. 表面浮凸效应"], "correct": "C", "explanation": "马氏体相变是无扩散型相变(原子协同运动), 不依赖原子长程扩散。特征: 切变共格、表面浮凸、惯习面、位向关系。", "difficulty": "medium"},
                {"question": "TTT曲线中, C形曲线的鼻子温度对应什么?", "options": ["A. 马氏体开始转变温度", "B. 相变速度最快的温度", "C. 共析温度", "D. 再结晶温度"], "correct": "B", "explanation": "TTT曲线的鼻子处对应相变孕育期最短(相变最快), 是过冷度驱动力和扩散系数竞争的结果。", "difficulty": "medium"},
            ],
            "塑性变形": [
                {"question": "Schmid定律描述的是什么关系?", "options": ["A. 应力与应变", "B. 临界分切应力与屈服应力", "C. 位错密度与强度", "D. 晶粒尺寸与硬度"], "correct": "B", "explanation": "Schmid定律: τ_c = σ_y·cosφ·cosλ, 当分切应力达到临界值τ_c时晶体开始滑移屈服。cosφ·cosλ 称为Schmid因子。", "difficulty": "medium"},
                {"question": "多晶体塑性变形的特点是什么?", "options": ["A. 各晶粒独立变形", "B. 需要至少5个独立滑移系开动", "C. 只需1个滑移系", "D. 晶界不参与变形"], "correct": "B", "explanation": "Von Mises准则: 多晶体任意形状变化需要至少5个独立滑移系。FCC有12个{111}<110>滑移系, 塑性最好。", "difficulty": "hard"},
            ],
            "断裂": [
                {"question": "解理断裂属于哪种断裂类型?", "options": ["A. 韧性断裂", "B. 脆性断裂(穿晶)", "C. 沿晶断裂", "D. 疲劳断裂"], "correct": "B", "explanation": "解理断裂是脆性穿晶断裂, 沿特定晶体学面(解理面)扩展, 断口特征为河流花样。BCC金属在低温下常见。", "difficulty": "easy"},
                {"question": "韧脆转变温度(DBTT)通常通过什么实验测定?", "options": ["A. 拉伸实验", "B. 硬度测试", "C. 系列温度冲击实验(Charpy)", "D. 疲劳实验"], "correct": "C", "explanation": "通过不同温度下的Charpy冲击实验, 测定冲击功随温度的变化, 确定韧脆转变温度。体心立方金属有明显DBTT。", "difficulty": "easy"},
            ],
            "高分子材料": [
                {"question": "热塑性塑料和热固性塑料的根本区别是?", "options": ["A. 密度不同", "B. 加热后是否可反复软化", "C. 颜色不同", "D. 硬度不同"], "correct": "B", "explanation": "热塑性塑料(线型/支化结构)加热可逆软化; 热固性塑料(交联网络结构)加热固化不可逆。", "difficulty": "easy"},
                {"question": "聚合物的玻璃化转变温度T_g是?", "options": ["A. 熔点", "B. 从玻璃态变为高弹态的温度", "C. 分解温度", "D. 结晶温度"], "correct": "B", "explanation": "T_g是非晶态聚合物或结晶聚合物非晶区从硬而脆的玻璃态转变为柔软高弹态的温度, 是链段运动被激活的温度。", "difficulty": "easy"},
            ],
            "陶瓷": [
                {"question": "陶瓷材料的主要缺点是什么?", "options": ["A. 硬度低", "B. 脆性大(韧性低)", "C. 耐高温性能差", "D. 导电性好"], "correct": "B", "explanation": "陶瓷材料的离子键/共价键使其硬度高、耐高温, 但位错难以运动, 脆性大是其主要缺点。", "difficulty": "easy"},
            ],
            "电性能": [
                {"question": "半导体Si的导电性随温度升高而?", "options": ["A. 降低(电阻增大)", "B. 增强(电阻减小)", "C. 不变", "D. 先升后降"], "correct": "B", "explanation": "半导体导电性随温度升高而增强(电阻减小), 因为更多电子被热激发到导带。金属相反, 温度升高电阻增大。", "difficulty": "easy"},
            ],
        }

        questions = []
        topic_lower = topic.lower()

        # 模糊匹配: 计算topic与每个模板key的重叠度
        best_key, best_score = None, 0
        for key in mcq_templates:
            key_lower = key.lower()
            # 双向包含得分最高
            if key in topic or topic in key:
                score = len(key)
            else:
                # 字符重叠度
                overlap = len(set(key_lower) & set(topic_lower))
                score = overlap / max(len(key_lower), 1)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score > 0:
            templates = mcq_templates[best_key]
            for t in templates:
                ref = ""
                for chunk, _ in results:
                    if best_key in chunk.text or best_key in chunk.chapter:
                        ref = f"{chunk.source_file}, p.{chunk.page_number}"
                        break
                if not ref and results:
                    ref = f"{results[0][0].source_file}, p.{results[0][0].page_number}"
                questions.append(QuizQuestion(
                    question_type="mcq",
                    question=t["question"],
                    options=t["options"],
                    correct_answer=t["correct"],
                    explanation=t["explanation"],
                    difficulty=t.get("difficulty", "medium"),
                    source_ref=ref,
                ))

        # 如果匹配到的题目不够, 用通用关键词补充
        if len(questions) < num:
            fallback = self._keyword_based_mcq(results, num - len(questions))
            questions.extend(fallback)

        return questions[:(num or len(questions))]

    def _keyword_based_mcq(self, results: List[Tuple[TextChunk, float]],
                           num: int) -> List[QuizQuestion]:
        """从检索结果提取关键句构造选择题/判断题 (支持中英文)."""
        questions = []
        for chunk, _ in results[:num * 2]:
            text = chunk.text
            # 中英文分别分句
            if any('一' <= c <= '鿿' for c in text[:50]):
                sentences = re.split(r'[。；？?！!]', text)
            else:
                sentences = re.split(r'[.;?!]', text)
            for sent in sentences:
                sent = sent.strip()
                # 中文放宽长度限制
                is_cn = any('一' <= c <= '鿿' for c in sent)
                min_len, max_len = (10, 80) if is_cn else (30, 150)
                if len(sent) < min_len or len(sent) > max_len:
                    continue
                # 跳过纯数字/符号句
                if re.match(r'^[\d\s\.\,\;\:\-\+\=\(\)\[\]\{\}…%°℃]+$', sent):
                    continue
                questions.append(QuizQuestion(
                    question_type="true_false",
                    question=f"判断正误: {sent}",
                    options=["A. 正确", "B. 错误"],
                    correct_answer="A",
                    explanation=f"该陈述来自教材《{chunk.source_file}》第{chunk.page_number}页。",
                    difficulty="easy",
                    source_ref=f"{chunk.source_file}, p.{chunk.page_number}",
                ))
                break
            if len(questions) >= num:
                break
        return questions

    def _rule_based_calc(self, topic: str,
                         results: List[Tuple[TextChunk, float]]) -> List[QuizQuestion]:
        """生成计算题."""
        calc_templates = [
            QuizQuestion(
                question_type="calculation",
                question=(
                    "计算FCC晶体的致密度 (Atomic Packing Factor)。"
                    "已知: FCC晶胞常数 a, 原子半径 r = √2a/4, "
                    "每个晶胞含4个原子。请写出计算过程。"
                ),
                correct_answer="π√2/6 ≈ 0.74 (74%)",
                explanation=(
                    "解: 1) 每个刚球体积 V_atom = (4/3)πr³\n"
                    "2) r = √2a/4 → r³ = (2√2/64)a³\n"
                    "3) 4个原子总体积 = 4 × (4/3)π × (2√2/64)a³ = (π√2/6)a³\n"
                    "4) 晶胞体积 = a³\n"
                    "5) APF = (π√2/6)a³ / a³ = π√2/6 ≈ 0.74"
                ),
                difficulty="medium",
            ),
            QuizQuestion(
                question_type="calculation",
                question=(
                    "一根直径为10mm的铜单晶沿[001]方向拉伸, "
                    "屈服强度为25MPa。如果滑移系为{111}<110>, "
                    "请计算临界分切应力 (CRSS)。"
                ),
                correct_answer="τ_crss ≈ 10.2 MPa",
                explanation=(
                    "解: 1) Schmid定律: τ_crss = σ_y · cosφ · cosλ\n"
                    "2) 对BCC/FCC: 选最大Schmid因子\n"
                    "3) [001]拉伸方向, 滑移面{111}, 方向<110>\n"
                    "4) cosφ = [001]·[111]/|[001]||[111]| = 1/√3\n"
                    "5) cosλ = [001]·[110]/|[001]||[110]| = 0 (最不利取向)\n"
                    "6) 实际取最大Schmid因子 ≈ 0.408\n"
                    "7) τ_crss = 25 × 0.408 ≈ 10.2 MPa"
                ),
                difficulty="hard",
            ),
            QuizQuestion(
                question_type="calculation",
                question=(
                    "已知扩散系数 D₀ = 2.0×10⁻⁵ m²/s, 激活能 Q = 142 kJ/mol。"
                    "求 900°C 时的扩散系数 D。\n"
                    "(R = 8.314 J/(mol·K))"
                ),
                correct_answer="D ≈ 9.67×10⁻¹² m²/s",
                explanation=(
                    "解: 1) 阿伦尼乌斯公式: D = D₀·exp(-Q/RT)\n"
                    "2) T = 900 + 273 = 1173 K\n"
                    "3) Q/RT = 142000/(8.314×1173) = 14.56\n"
                    "4) D = 2.0×10⁻⁵ × exp(-14.56)\n"
                    "5) D = 2.0×10⁻⁵ × 4.84×10⁻⁷\n"
                    "6) D ≈ 9.67×10⁻¹² m²/s"
                ),
                difficulty="medium",
            ),
        ]

        keyword_map = {
            "晶体": [0], "致密度": [0], "FCC": [0],
            "位错": [1], "滑移": [1], "临界分切应力": [1], "Schmid": [1],
            "扩散": [2], "阿伦尼乌斯": [2],
        }

        selected = []
        for keyword, indices in keyword_map.items():
            if keyword in topic and indices[0] not in selected:
                selected.append(indices[0])

        if not selected:
            selected = list(range(len(calc_templates)))

        return [calc_templates[i] for i in selected]

    def _llm_generate_mcq(self, context: str, num: int) -> List[QuizQuestion]:
        """LLM生成选择题."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            prompt = self.GENERATE_MCQ_PROMPT.format(context=context[:3000], num=num)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            questions = []
            for item in data.get("questions", data if isinstance(data, list) else []):
                questions.append(QuizQuestion(
                    question_type="mcq",
                    question=item["question"],
                    options=item.get("options", []),
                    correct_answer=item.get("correct", ""),
                    explanation=item.get("explanation", ""),
                    difficulty=item.get("difficulty", "medium"),
                ))
            return questions
        except Exception:
            return []

    def _llm_generate_calc(self, context: str, num: int) -> List[QuizQuestion]:
        """LLM生成计算题."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            prompt = self.GENERATE_CALC_PROMPT.format(context=context[:3000], num=num)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            questions = []
            for item in data.get("questions", data if isinstance(data, list) else []):
                steps = item.get("steps", [])
                explanation = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
                questions.append(QuizQuestion(
                    question_type="calculation",
                    question=item["question"],
                    correct_answer=item.get("answer", ""),
                    explanation=explanation,
                    difficulty=item.get("difficulty", "medium"),
                    points=10,
                ))
            return questions
        except Exception:
            return []
