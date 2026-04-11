"""
流星运行时 — 主题回指消解器 (TopicAnaphoraResolver)

零形回指 (Zero Anaphora) 是中文最重要的语法特性——主语可以省略，
由上下文中的主题自动推断。

TopicAnaphoraResolver 跟踪 R63 (主题寄存器) 中的当前主题，
实现句子间的零形回指消解：

  "张三来了。他走了。"
    → 张三进入 R63 (主题)
    → 他 resolves to 张三 from R63
    → 主题继承跨句子边界

设计哲学:
  - R63 始终持有当前主题实体
  - 零形回指 = 从 R63 读取主题
  - 主题转移 = 用新实体替换 R63
  - 主题继承 = 保持 R63 不变 (同一实体继续操作)

用法:
    resolver = TopicAnaphoraResolver()
    resolver.process("张三来了")      # 张三 → R63
    resolver.process("他走了")          # 他 → resolves to 张三
    resolver.current_topic  # "张三"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════
# 回指类型
# ══════════════════════════════════════════════════════════════════════

class AnaphoraType(Enum):
    """回指类型"""
    ZERO_ANAPHORA = "零形回指"     # 省略主语 (张三来了。[他]走了)
    PRONOUN_ANAPHORA = "代词回指"   # 他/她/它/其
    REFLEXIVE = "反身回指"          # 自己
    NO_ANAPHORA = "无回指"          # 有明确主语


# ══════════════════════════════════════════════════════════════════════
# 中文代词表
# ══════════════════════════════════════════════════════════════════════

PRONOUN_MAP: dict[str, str] = {
    "他": "third_person_male",
    "她": "third_person_female",
    "它": "third_person_neuter",
    "其": "possessive",
    "自己": "reflexive",
    "这": "demonstrative_near",
    "那": "demonstrative_far",
    "该": "demonstrative_formal",
}

# 指示主题转移的关键词
TOPIC_SHIFT_MARKERS = [
    "但是", "不过", "然而", "可是", "虽然", "另外",
    "此外", "同时", "然后", "接着", "于是",
    "因此", "所以", "如果", "假设",
]

# 人称代词 → 第三人称 (需要回指消解)
THIRD_PERSON_PRONOUNS = {"他", "她", "它"}

# 反身代词
REFLEXIVE_PRONOUNS = {"自己", "自身", "本人"}

# 所有需要回指消解的代词
ALL_ANAPHORA_PRONOUNS = THIRD_PERSON_PRONOUNS | REFLEXIVE_PRONOUNS


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TopicEntry:
    """主题条目 — R63 寄存器中保存的主题信息"""
    entity: str               # 主题实体名 (如 "张三")
    entity_type: str = ""      # 实体类型 (如 "人", "物")
    classifier: str = ""       # 关联的量词 (如 "位", "台")
    confidence: float = 1.0    # 置信度
    sentence_index: int = 0    # 出现的句子序号
    context: str = ""          # 上下文/场景

    @property
    def is_person(self) -> bool:
        """是否为人"""
        return self.entity_type in ("人", "人员", "人员(敬称)")

    def __repr__(self) -> str:
        ctx = f" [{self.context}]" if self.context else ""
        return (
            f"TopicEntry({self.entity}"
            f"{f'({self.entity_type})' if self.entity_type else ''}"
            f", conf={self.confidence:.1f}, sent={self.sentence_index}{ctx})"
        )


@dataclass
class AnaphoraResolution:
    """回指消解结果"""
    pronoun: str                  # 代词文本
    resolved_entity: str          # 消解到的实体
    anaphora_type: AnaphoraType   # 回指类型
    source_sentence: int          # 回指出现的句子
    source_topic_sentence: int     # 主题被设置的句子
    confidence: float = 1.0        # 消解置信度

    @property
    def type_name(self) -> str:
        return self.anaphora_type.value

    def __repr__(self) -> str:
        return (
            f"AnaphoraResolution({self.pronoun} → {self.resolved_entity}"
            f" [{self.type_name}], conf={self.confidence:.1f})"
        )


@dataclass
class SentenceAnalysis:
    """句子分析结果"""
    text: str
    index: int
    subject: Optional[str] = None
    topic: Optional[str] = None
    has_zero_anaphora: bool = False
    anaphora_pronouns: list[str] = field(default_factory=list)
    has_topic_shift: bool = False
    topic_shift_marker: str = ""


# ══════════════════════════════════════════════════════════════════════
# 句子分析器
# ══════════════════════════════════════════════════════════════════════

# 匹配中文句子的主语/谓语
# 简化: 识别句子开头的实体或代词
_SUBJECT_PATTERN = re.compile(
    r"^([^\s,，。！？；：、]+?)(?:\s+(.+))?$"
)

# 零形回指检测: 句子以动词开头 (没有主语)
_ZERO_ANAPHORA_PATTERN = re.compile(
    r"^(?:来|去|走|跑|坐|站|看|听|说|吃|喝|做|是|有|没|不|在|到|从"
    r"|给|被|让|把|用|按|向|对|把|将|会|能|要|想|应|该|已|正|曾|刚|才|还|又|再"
    r"|加|减|乘|除|计算|打印|加载|存储|增加|减少|告诉|询问|广播|委托|返回)"
    r")"
)


class _SentenceAnalyzer:
    """句子分析器 — 识别中文句子的主语、回指和主题转移"""

    @staticmethod
    def analyze(text: str, index: int) -> SentenceAnalysis:
        """分析单个中文句子"""
        text = text.strip()
        result = SentenceAnalysis(text=text, index=index)

        if not text:
            return result

        # 检测主题转移标记
        for marker in TOPIC_SHIFT_MARKERS:
            if text.startswith(marker):
                result.has_topic_shift = True
                result.topic_shift_marker = marker
                break

        # 提取主语: 句子开头的名词/人名/代词
        match = _SUBJECT_PATTERN.match(text)
        if match:
            potential_subject = match.group(1)
            # 检查是否为代词
            pronouns = [p for p in ALL_ANAPHORA_PRONOUNS if p in potential_subject]
            if pronouns:
                result.anaphora_pronouns = pronouns
                result.has_zero_anaphora = len(pronouns) > 0
                # 如果只有代词, 可能是零形回指
                if potential_subject in THIRD_PERSON_PRONOUNS and len(potential_subject) <= 2:
                    # "他走了" — 只有代词, 可能是回指
                    remaining = match.group(2) if match.group(2) else ""
                    if remaining:
                        result.has_zero_anaphora = True
            else:
                result.subject = potential_subject
        else:
            # 检查是否以动词开头 (零形回指)
            if _ZERO_ANAPHORA_PATTERN.match(text):
                result.has_zero_anaphora = True

        # 提取主题关键词
        # 主题 = 句子最关注的对象
        result.topic = result.subject  # 默认主题就是主语

        return result


# ══════════════════════════════════════════════════════════════════════
# TopicAnaphoraResolver — 主题回指消解器
# ══════════════════════════════════════════════════════════════════════

class TopicAnaphoraResolver:
    """
    主题回指消解器 — 基于 R63 (主题寄存器) 的零形回指消解。

    核心机制:
      1. 当出现明确主语时, 将其写入 R63 (主题设置)
      2. 当出现零形回指时, 从 R63 读取主题 (回指消解)
      3. 当出现代词时, 将代词映射到 R63 中的实体 (代词消解)
      4. 主题继承: 除非明确转移, R63 在句子间保持

    示例:
        >>> resolver = TopicAnaphoraResolver()
        >>> resolver.process("张三来了")
        TopicEntry(张三(人), conf=1.0, sent=0)
        >>> resolver.process("他走了")
        AnaphoraResolution(他 → 张三 [代词回指], conf=1.0)
        >>> resolver.process("吃了饭")
        AnaphoraResolution(∅ → 张三 [零形回指], conf=0.9)
        >>> resolver.current_topic
        '张三'
    """

    # 主题寄存器编号
    TOPIC_REGISTER = 63

    # 置信度衰减系数 (跨句子时)
    CONFIDENCE_DECAY = 0.05

    def __init__(self):
        self._current_topic: Optional[TopicEntry] = None
        self._topic_history: list[TopicEntry] = []
        self._sentence_count: int = 0
        _analyzer = _SentenceAnalyzer()  # noqa: F841
        self._analyzer = _SentenceAnalyzer()
        self._resolution_log: list[AnaphoraResolution] = []
        self._sentence_log: list[SentenceAnalysis] = []
        self._topic_register: int = 0  # R63 的模拟值

    @property
    def current_topic(self) -> str | None:
        """获取当前主题实体名"""
        if self._current_topic:
            return self._current_topic.entity
        return None

    @property
    def current_topic_entry(self) -> Optional[TopicEntry]:
        """获取当前主题完整条目"""
        return self._current_topic

    @property
    def topic_register(self) -> int:
        """获取主题寄存器 R63 的模拟值"""
        return self._topic_register

    @property
    def topic_history(self) -> list[TopicEntry]:
        """获取主题历史记录"""
        return list(self._topic_history)

    @property
    def sentence_count(self) -> int:
        """已处理的句子数"""
        return self._sentence_count

    def process(self, sentence: str) -> AnaphoraResolution | None:
        """
        处理一个中文句子, 进行主题设置/回指消解。

        Args:
            sentence: 中文句子

        Returns:
            如果有回指消解则返回 AnaphoraResolution, 否则返回 None
        """
        self._sentence_count += 1
        sentence = sentence.strip().rstrip("。！？；：、")
        if not sentence:
            return None

        analysis = self._analyzer.analyze(sentence, self._sentence_count - 1)
        self._sentence_log.append(analysis)

        # 衰减当前主题的置信度
        if self._current_topic:
            self._current_topic.confidence -= self.CONFIDENCE_DECAY
            if self._current_topic.confidence < 0.1:
                self._current_topic = None

        # 优先处理代词回指
        if analysis.anaphora_pronouns:
            return self._resolve_pronoun(analysis)

        # 处理零形回指
        if analysis.has_zero_anaphora and self._current_topic:
            return self._resolve_zero_anaphora(analysis)

        # 明确主语 → 设置主题
        if analysis.subject and not analysis.has_zero_anaphora:
            self._set_topic(analysis)

        return None

    def process_program(self, text: str) -> list[AnaphoraResolution]:
        """
        处理多行中文程序 (用句号、感叹号等分句)。

        Args:
            text: 多行中文文本

        Returns:
            所有的回指消解结果列表
        """
        results: list[AnaphoraResolution] = []
        # 按中英文标点分句
        sentences = re.split(r'[。！？\n]+', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            resolution = self.process(sent)
            if resolution:
                results.append(resolution)
        return results

    def _set_topic(self, analysis: SentenceAnalysis) -> TopicEntry:
        """设置当前主题"""
        entry = TopicEntry(
            entity=analysis.subject,
            sentence_index=analysis.index,
            confidence=1.0,
        )
        self._current_topic = entry
        self._topic_history.append(entry)
        self._topic_register = hash(entry.entity) % 0xFFFFFFFF
        return entry

    def _resolve_pronoun(self, analysis: SentenceAnalysis) -> AnaphoraResolution:
        """消解代词回指"""
        if not self._current_topic:
            # 无法消解 — 没有已知主题
            return AnaphoraResolution(
                pronoun=analysis.anaphora_pronouns[0],
                resolved_entity="",
                anaphora_type=AnaphoraType.PRONOUN_ANAPHORA,
                source_sentence=analysis.index,
                source_topic_sentence=0,
                confidence=0.0,
            )

        resolved = self._current_topic
        return AnaphoraResolution(
            pronoun=analysis.anaphora_pronouns[0],
            resolved_entity=resolved.entity,
            anaphora_type=AnaphoraType.PRONOUN_ANAPHORA,
            source_sentence=analysis.index,
            source_topic_sentence=resolved.sentence_index,
            confidence=resolved.confidence,
        )

    def _resolve_zero_anaphora(self, analysis: SentenceAnalysis) -> AnaphoraResolution:
        """消解零形回指"""
        resolved = self._current_topic
        return AnaphoraResolution(
            pronoun="∅",
            resolved_entity=resolved.entity if resolved else "",
            anaphora_type=AnaphoraType.ZERO_ANAPHORA,
            source_sentence=analysis.index,
            source_topic_sentence=resolved.sentence_index if resolved else 0,
            confidence=resolved.confidence * 0.9 if resolved else 0.0,
        )

    def resolution_log(self) -> list[AnaphoraResolution]:
        """获取所有回指消解记录"""
        return list(self._resolution_log)

    def sentence_log(self) -> list[SentenceAnalysis]:
        """获取所有句子分析记录"""
        return list(self._sentence_log)

    def reset(self) -> None:
        """重置主题状态"""
        self._current_topic = None
        self._topic_history.clear()
        self._resolution_log.clear()
        self._sentence_log.clear()
        self._sentence_count = 0
        self._topic_register = 0

    def __repr__(self) -> str:
        topic = self.current_topic or "(无)"
        return (
            f"TopicAnaphoraResolver(topic={topic}, "
            f"sentences={self._sentence_count}, "
            f"history_len={len(self._topic_history)})"
        )
