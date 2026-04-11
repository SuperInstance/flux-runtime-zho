"""
流星词汇系统 — 中文词汇瓦片 (Flux Vocabulary Tiling System)

多层词汇架构:
  Level 0 — 原始词 (原始词): 计算/加载/存储/跳转/增加/减少/打印/停机
  Level 1 — 组合词 (组合词): 累加/递归/比较/排序/搜索
  Level 2 — 领域词 (领域词): 航海/数学/智能体 等领域专用词汇
  Level 3 — 复合词 (复合词): 多步骤计算瓦片

设计哲学:
  - 词汇即类型系统 — 每个词汇条目携带量词类型标注
  - 主题-评论映射 — 词汇模式天然匹配中文语法结构
  - 可组合瓦片 — 高层词汇可分解为低层词汇组合
  - .fluxvocab-zho 文件格式 — 支持外部词汇文件加载

使用方法:
    from flux_zho.vocabulary import VocabularyRegistry

    registry = VocabularyRegistry()
    registry.load_defaults()
    assembly = registry.compile("计算 三加四")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════
# 词汇层级枚举
# ══════════════════════════════════════════════════════════════════════

class VocabLevel(IntEnum):
    """词汇瓦片层级"""
    PRIMITIVE = 0     # 原始词 — 直接映射到单条 FLUX 指令
    COMPOSITE = 1     # 组合词 — 多条 FLUX 指令组合
    DOMAIN = 2        # 领域词 — 特定领域的复杂指令序列
    COMPOUND = 3      # 复合词 — 多步骤计算瓦片


# 层级中文名
LEVEL_NAMES: dict[VocabLevel, str] = {
    VocabLevel.PRIMITIVE: "原始词",
    VocabLevel.COMPOSITE: "组合词",
    VocabLevel.DOMAIN: "领域词",
    VocabLevel.COMPOUND: "复合词",
}


# ══════════════════════════════════════════════════════════════════════
# 词汇条目 — 单个词汇瓦片
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VocabEntry:
    """
    词汇条目: 中文词语模式 → FLUX 字节码汇编模板。

    属性:
        word:          中文关键词 (如 "计算", "加载", "航行")
        level:         词汇层级 (0=原始, 1=组合, 2=领域, 3=复合)
        pattern:       正则表达式模式 (用于识别源文本)
        template:      FLUX 汇编模板 (用 {name} 表示占位符替换)
        placeholders:  占位符名 → 正则捕获组编号
        classifier:    关联量词 (用于类型检查)
        description:   中文描述
        domain:        所属领域 (如 "航海", "数学", "智能体")
        tags:          标签列表
        examples:      使用示例列表
    """
    word: str
    level: VocabLevel
    pattern: re.Pattern
    template: str
    placeholders: dict[str, int] = field(default_factory=dict)
    classifier: str = ""
    description: str = ""
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def recognize(self, text: str) -> dict[str, str] | None:
        """
        识别文本是否匹配此词汇模式。

        返回占位符字典或 None。
        """
        match = self.pattern.fullmatch(text.strip())
        if not match:
            return None
        result = {}
        for name, group_idx in self.placeholders.items():
            if group_idx <= len(match.groups()):
                result[name] = match.group(group_idx).strip()
        return result

    def compile(self, text: str) -> str | None:
        """
        编译文本为此词汇的汇编输出。

        返回 FLUX 汇编字符串或 None (如果模式不匹配)。
        """
        captures = self.recognize(text)
        if captures is None:
            return None
        assembly = self.template
        for name, value in captures.items():
            assembly = assembly.replace(f"{{{name}}}", value.strip())
        return assembly

    def __repr__(self) -> str:
        clf = f"[{self.classifier}]" if self.classifier else ""
        dom = f" ({self.domain})" if self.domain else ""
        return (
            f"VocabEntry('{self.word}', L{self.level.value}"
            f"{clf}{dom}, '{self.description}')"
        )


# ══════════════════════════════════════════════════════════════════════
# .fluxvocab-zho 文件解析器
# ══════════════════════════════════════════════════════════════════════

class VocabFileParser:
    """
    .fluxvocab-zho 文件解析器。

    文件格式:
        # 注释行以 # 开头
        @version 1.0
        @lang zho

        # Level 0 — 原始词
        计算  PRIM  "计算 $a 加 $b"  "MOVI R0, $a\\nMOVI R1, $b\\nIADD R0, R0, R1\\nHALT"  a=1 b=2  量词=个
        加载  PRIM  "加载 $reg 为 $val"  "MOVI R{reg}, $val\\nHALT"  reg=1 val=2  量词=个
    """

    @staticmethod
    def parse(content: str) -> list[VocabEntry]:
        """
        解析 .fluxvocab-zho 文件内容。

        Returns:
            词汇条目列表
        """
        entries: list[VocabEntry] = []
        metadata: dict[str, str] = {}

        for line in content.splitlines():
            line = line.strip()

            # 空行和注释
            if not line or line.startswith("#"):
                continue

            # 元数据
            if line.startswith("@"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
                continue

            entry = VocabFileParser._parse_line(line)
            if entry:
                entries.append(entry)

        return entries

    @staticmethod
    def _parse_line(line: str) -> VocabEntry | None:
        """解析单行词汇条目"""
        # 提取量词标注
        classifier = ""
        clf_match = re.search(r"量词=(\S+)", line)
        if clf_match:
            classifier = clf_match.group(1)
            line = line[:clf_match.start()].strip()

        # 提取领域标注
        domain = ""
        dom_match = re.search(r"领域=(\S+)", line)
        if dom_match:
            domain = dom_match.group(1)
            line = line[:dom_match.start()].strip()

        # 提取标签
        tags: list[str] = []
        tag_match = re.search(r"标签=(\S+)", line)
        if tag_match:
            tags = tag_match.group(1).split(",")
            line = line[:tag_match.start()].strip()

        # 分割字段 (至少两个空格分隔)
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 3:
            return None

        word = parts[0]
        level_str = parts[1].upper()
        pattern_str = parts[2]

        # 层级解析
        level_map = {
            "PRIM": VocabLevel.PRIMITIVE,
            "COMPOSITE": VocabLevel.COMPOSITE,
            "COMP": VocabLevel.COMPOSITE,
            "DOM": VocabLevel.DOMAIN,
            "DOMAIN": VocabLevel.DOMAIN,
            "COMPOUND": VocabLevel.COMPOUND,
        }
        level = level_map.get(level_str, VocabLevel.PRIMITIVE)

        # 提取模板
        template = ""
        if len(parts) > 3:
            tpl_part = parts[3]
            tpl_match = re.match(r'"(.+?)"', tpl_part, re.DOTALL)
            if tpl_match:
                template = tpl_match.group(1)
            else:
                template = tpl_part

        # 提取占位符
        placeholders: dict[str, int] = {}
        for part in parts[4:]:
            if "=" in part:
                name, idx = part.split("=", 1)
                try:
                    placeholders[name.strip()] = int(idx.strip())
                except ValueError:
                    pass

        # 将模式字符串中的 $name 转为正则捕获组
        regex_str = re.escape(pattern_str)
        regex_str = re.sub(
            r"\\\$(\w+)",
            lambda m: f"(?P<{m.group(1)}>\\S+)",
            regex_str,
        )

        try:
            pattern = re.compile(regex_str)
        except re.error:
            pattern = re.compile(re.escape(pattern_str))

        return VocabEntry(
            word=word,
            level=level,
            pattern=pattern,
            template=template,
            placeholders=placeholders,
            classifier=classifier,
            domain=domain,
            tags=tags,
            description=pattern_str,
        )


# ══════════════════════════════════════════════════════════════════════
# 词汇注册表 — 中央词汇目录
# ══════════════════════════════════════════════════════════════════════

class VocabularyRegistry:
    """
    中文词汇注册表 — 管理所有词汇瓦片。

    功能:
      - 按层级组织词汇
      - 按领域筛选词汇
      - 按量词类型过滤
      - 编译中文文本为 FLUX 汇编
      - 加载外部 .fluxvocab-zho 文件
    """

    def __init__(self) -> None:
        self._entries: list[VocabEntry] = []
        self._by_word: dict[str, VocabEntry] = {}
        self._by_level: dict[VocabLevel, list[VocabEntry]] = {
            lvl: [] for lvl in VocabLevel
        }
        self._by_domain: dict[str, list[VocabEntry]] = {}
        self._by_classifier: dict[str, list[VocabEntry]] = {}

    # ── 注册管理 ───────────────────────────────────────────────

    def register(self, entry: VocabEntry) -> None:
        """注册一个词汇条目"""
        self._entries.append(entry)
        self._by_word[entry.word] = entry
        self._by_level[entry.level].append(entry)

        if entry.domain:
            if entry.domain not in self._by_domain:
                self._by_domain[entry.domain] = []
            self._by_domain[entry.domain].append(entry)

        if entry.classifier:
            if entry.classifier not in self._by_classifier:
                self._by_classifier[entry.classifier] = []
            self._by_classifier[entry.classifier].append(entry)

    def unregister(self, word: str) -> bool:
        """注销一个词汇条目"""
        entry = self._by_word.pop(word, None)
        if entry is None:
            return False
        self._entries.remove(entry)
        self._by_level[entry.level].remove(entry)
        if entry.domain and entry.domain in self._by_domain:
            self._by_domain[entry.domain].remove(entry)
        if entry.classifier and entry.classifier in self._by_classifier:
            self._by_classifier[entry.classifier].remove(entry)
        return True

    def load_file(self, path: str | Path) -> int:
        """
        加载 .fluxvocab-zho 文件。

        Returns:
            成功加载的条目数
        """
        path = Path(path)
        if not path.exists():
            return 0
        content = path.read_text(encoding="utf-8")
        entries = VocabFileParser.parse(content)
        for entry in entries:
            self.register(entry)
        return len(entries)

    def load_defaults(self) -> None:
        """加载内置的标准词汇表 (所有层级)"""
        for entry in _create_builtin_vocabulary():
            self.register(entry)

    # ── 查询接口 ───────────────────────────────────────────────

    def search(
        self,
        text: str,
        level: VocabLevel | None = None,
        domain: str = "",
    ) -> list[VocabEntry]:
        """
        搜索匹配文本的词汇条目。

        Args:
            text:   要搜索的文本
            level:  限定层级 (None = 所有层级)
            domain: 限定领域

        Returns:
            匹配的词汇条目列表 (按层级排序)
        """
        results: list[VocabEntry] = []
        sources = self._entries if level is None else self._by_level.get(level, [])

        for entry in sources:
            if domain and entry.domain != domain:
                continue
            if entry.recognize(text) is not None:
                results.append(entry)

        # 按层级排序 (低层级优先)
        results.sort(key=lambda e: e.level)
        return results

    def compile(self, text: str) -> str | None:
        """
        编译中文文本为 FLUX 汇编。

        按层级顺序搜索 (0→1→2→3)，返回第一个匹配结果。

        Returns:
            FLUX 汇编字符串或 None
        """
        for level in VocabLevel:
            for entry in self._by_level[level]:
                assembly = entry.compile(text)
                if assembly is not None:
                    return assembly
        return None

    def get_entry(self, word: str) -> VocabEntry | None:
        """按关键词获取词汇条目"""
        return self._by_word.get(word)

    def get_level(self, level: VocabLevel) -> list[VocabEntry]:
        """获取指定层级的所有词汇"""
        return list(self._by_level.get(level, []))

    def get_domain(self, domain: str) -> list[VocabEntry]:
        """获取指定领域的所有词汇"""
        return list(self._by_domain.get(domain, []))

    def get_classifier(self, classifier: str) -> list[VocabEntry]:
        """获取指定量词类型的所有词汇"""
        return list(self._by_classifier.get(classifier, []))

    @property
    def count(self) -> int:
        """注册的词汇总数"""
        return len(self._entries)

    def all_entries(self) -> list[VocabEntry]:
        """获取所有词汇条目"""
        return list(self._entries)

    def domains(self) -> list[str]:
        """获取所有领域名称"""
        return list(self._by_domain.keys())


# ══════════════════════════════════════════════════════════════════════
# 内置标准词汇 — Level 0 到 Level 3
# ══════════════════════════════════════════════════════════════════════

def _create_builtin_vocabulary() -> list[VocabEntry]:
    """
    创建内置标准词汇表。

    Level 0 — 原始词 (单条 FLUX 指令):
        计算, 加载, 存储, 跳转, 增加, 减少, 打印, 停机

    Level 1 — 组合词 (多条 FLUX 指令):
        累加, 递归, 比较, 排序, 搜索, 阶乘, 求和

    Level 2 — 领域词:
        航海: 航行/锚定/探测/信号/编队
        数学: 求和/求积/阶乘/斐波那契
        智能体: 告知/询问/委托/广播/信任验证

    Level 3 — 复合词:
        多步骤计算瓦片
    """
    entries: list[VocabEntry] = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 0: 原始词 — 直接映射到 FLUX 指令
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 计算...加... — 整数加法
    entries.append(VocabEntry(
        word="加法",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"计算\s+(.+?)\s*加\s+(.+)"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nIADD R0, R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="计算 a 加 b",
        tags=["算术", "加"],
        examples=["计算 三加四", "计算 十加二十"],
    ))

    # 计算...减... — 整数减法
    entries.append(VocabEntry(
        word="减法",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"计算\s+(.+?)\s*减\s+(.+)"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nISUB R0, R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="计算 a 减 b",
        tags=["算术", "减"],
        examples=["计算 十减三", "计算 五减二"],
    ))

    # 计算...乘... — 整数乘法
    entries.append(VocabEntry(
        word="乘法",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"计算\s+(.+?)\s*乘\s+(.+)"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nIMUL R0, R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="计算 a 乘 b",
        tags=["算术", "乘"],
        examples=["计算 五乘六", "计算 三乘七"],
    ))

    # 计算...除以... — 整数除法
    entries.append(VocabEntry(
        word="除法",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"计算\s+(.+?)\s*除以?\s*(.+)"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nIDIV R0, R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="计算 a 除以 b",
        tags=["算术", "除"],
        examples=["计算 十二除以 四", "计算 二十除以 五"],
    ))

    # 加载...为... — 寄存器赋值
    entries.append(VocabEntry(
        word="加载",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"加载\s+(.+?)\s+为\s+(.+)"),
        template="MOVI {reg}, {val}\nHALT",
        placeholders={"reg": 1, "val": 2},
        classifier="个",
        description="加载寄存器为值",
        tags=["寄存器", "赋值"],
        examples=["加载 寄存器零 为 四十二", "加载 甲 为 十"],
    ))

    # 增加... — 自增
    entries.append(VocabEntry(
        word="增加",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"增加\s+(.+)"),
        template="INC {reg}\nHALT",
        placeholders={"reg": 1},
        classifier="个",
        description="寄存器值加一",
        tags=["控制流", "递增"],
        examples=["增加 甲", "增加 寄存器零"],
    ))

    # 减少... — 自减
    entries.append(VocabEntry(
        word="减少",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"减少\s+(.+)"),
        template="DEC {reg}\nHALT",
        placeholders={"reg": 1},
        classifier="次",
        description="寄存器值减一",
        tags=["控制流", "递减"],
        examples=["减少 甲", "减少 寄存器零"],
    ))

    # 打印... — 输出
    entries.append(VocabEntry(
        word="打印",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"打印\s*(.*)"),
        template="PRINT {reg}\nHALT",
        placeholders={"reg": 1},
        classifier="个",
        description="打印寄存器值",
        tags=["系统", "打印"],
        examples=["打印 甲", "打印 寄存器零"],
    ))

    # 停机 — 程序终止
    entries.append(VocabEntry(
        word="停机",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"停机"),
        template="HALT",
        description="程序停机",
        tags=["系统"],
        examples=["停机"],
    ))

    # 返回... — 返回值
    entries.append(VocabEntry(
        word="返回",
        level=VocabLevel.PRIMITIVE,
        pattern=re.compile(r"返回\s*(.*)"),
        template="MOVI R0, {val}\nRET",
        placeholders={"val": 1},
        classifier="个",
        description="返回值",
        tags=["控制流", "返回"],
        examples=["返回 四十二"],
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 1: 组合词 — 多条 FLUX 指令组合
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 阶乘 — N!
    entries.append(VocabEntry(
        word="阶乘",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"(.+?)\s*的阶乘"),
        template=(
            "MOVI R0, {n}\n"
            "MOVI R1, 1\n"
            "MOV R2, R0\n"
            "dec_loop:\n"
            "IMUL R1, R1, R2\n"
            "DEC R2\n"
            "JNZ R2, dec_loop\n"
            "MOV R0, R1\n"
            "HALT"
        ),
        placeholders={"n": 1},
        classifier="个",
        description="N 的阶乘",
        tags=["算术", "阶乘"],
        examples=["五的阶乘", "十的阶乘"],
    ))

    # 从...到...的和 — 区间求和
    entries.append(VocabEntry(
        word="区间求和",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"从\s+(.+?)\s+到\s+(.+?)\s+的和"),
        template=(
            "MOVI R0, {a}\n"
            "MOVI R1, {b}\n"
            "MOV R2, 0\n"
            "sum_loop:\n"
            "IADD R2, R2, R0\n"
            "INC R0\n"
            "CMP R0, R1\n"
            "JL sum_loop\n"
            "IADD R2, R2, R1\n"
            "MOV R0, R2\n"
            "HALT"
        ),
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="从 a 到 b 的区间求和",
        tags=["算术", "求和"],
        examples=["从 一 到 十 的和", "从 一 到 一百 的和"],
    ))

    # 比较...和... — 相等比较
    entries.append(VocabEntry(
        word="比较",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"比较\s+(.+?)\s+和\s+(.+?)\s+是否相等"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nCMP R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="比较两个值是否相等",
        tags=["比较"],
        examples=["比较 三 和 三 是否相等"],
    ))

    # 计算...的余数... — 取模
    entries.append(VocabEntry(
        word="取模",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"计算\s+(.+?)\s*的余数\s*(.+)"),
        template="MOVI R0, {a}\nMOVI R1, {b}\nIMOD R0, R0, R1\nHALT",
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="计算 a 的余数 b",
        tags=["算术", "模"],
        examples=["计算 十的余数 三"],
    ))

    # ...的...次方 — 幂运算
    entries.append(VocabEntry(
        word="幂运算",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"(.+?)\s*的\s*(.+?)\s*次方"),
        template=(
            "MOVI R0, 1\n"
            "MOVI R1, {base}\n"
            "MOV R2, {exp}\n"
            "pow_loop:\n"
            "IMUL R0, R0, R1\n"
            "DEC R2\n"
            "JNZ R2, pow_loop\n"
            "HALT"
        ),
        placeholders={"base": 1, "exp": 2},
        classifier="个",
        description="base 的 exp 次方",
        tags=["算术", "幂"],
        examples=["二 的 十 次方"],
    ))

    # 累加...次 — 循环累加
    entries.append(VocabEntry(
        word="累加",
        level=VocabLevel.COMPOSITE,
        pattern=re.compile(r"累加\s+(.+?)\s*次"),
        template=(
            "MOVI R0, 0\n"
            "MOVI R1, 1\n"
            "MOV R2, {n}\n"
            "acc_loop:\n"
            "IADD R0, R0, R1\n"
            "INC R1\n"
            "DEC R2\n"
            "JNZ R2, acc_loop\n"
            "HALT"
        ),
        placeholders={"n": 1},
        classifier="次",
        description="累加 n 次",
        tags=["算术", "累加"],
        examples=["累加 十 次"],
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 2: 领域词 — 航海领域
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 以...节速度航行...小时 — 航程计算
    entries.append(VocabEntry(
        word="航行",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"以\s+(.+?)\s*节速度航行\s+(.+?)\s*小时"),
        template="MOVI R0, {speed}\nMOVI R1, {hours}\nIMUL R0, R0, R1\nHALT",
        placeholders={"speed": 1, "hours": 2},
        classifier="海里",
        description="航海: 航程 = 速度 × 时间",
        domain="航海",
        tags=["航海", "速度", "距离"],
        examples=["以 十二 节速度航行 五 小时"],
    ))

    # 创建...艘船只 — 创建船队
    entries.append(VocabEntry(
        word="创建船队",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"创建\s+(.+?)\s*艘船只"),
        template="MOVI R0, {n}\nREGION_CREATE R0\nHALT",
        placeholders={"n": 1},
        classifier="艘",
        description="航海: 创建船队",
        domain="航海",
        tags=["航海", "船只", "创建"],
        examples=["创建 三 艘船只"],
    ))

    # 在...号锚地抛锚 — 抛锚
    entries.append(VocabEntry(
        word="抛锚",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"在\s+(.+?)\s*号锚地抛锚"),
        template="MOVI R0, {n}\nMOVI R1, 1\nHALT",
        placeholders={"n": 1},
        classifier="锚",
        description="航海: 抛锚",
        domain="航海",
        tags=["航海", "锚"],
        examples=["在 三 号锚地抛锚"],
    ))

    # 探测...海里 — 深度探测
    entries.append(VocabEntry(
        word="探测",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"探测\s+(.+?)\s*海里"),
        template="MOVI R0, {depth}\nHALT",
        placeholders={"depth": 1},
        classifier="海里",
        description="航海: 深度探测",
        domain="航海",
        tags=["航海", "探测"],
        examples=["探测 两百 海里"],
    ))

    # 信号... — 发送信号
    entries.append(VocabEntry(
        word="信号",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"发送信号\s*(.+)"),
        template="MOVI R0, 1\nMOVI R1, 0\nBROADCAST R0, R1\nHALT",
        placeholders={},
        classifier="条",
        description="航海: 发送信号",
        domain="航海",
        tags=["航海", "信号"],
        examples=["发送信号"],
    ))

    # 编队...艘船 — 编队航行
    entries.append(VocabEntry(
        word="编队",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"编队\s+(.+?)\s*艘船"),
        template=(
            "MOVI R0, {n}\n"
            "MOVI R1, 1\n"
            "MOV R2, R0\n"
            "fleet_loop:\n"
            "DEC R2\n"
            "JZ R2, fleet_done\n"
            "JMP fleet_loop\n"
            "fleet_done:\n"
            "HALT"
        ),
        placeholders={"n": 1},
        classifier="艘",
        description="航海: 编队航行",
        domain="航海",
        tags=["航海", "编队"],
        examples=["编队 五 艘船"],
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 2: 领域词 — 数学领域
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 斐波那契 — 斐波那契数列第N项
    entries.append(VocabEntry(
        word="斐波那契",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"斐波那契\s*第?\s*(.+?)\s*项"),
        template=(
            "MOVI R0, 0\n"
            "MOVI R1, 1\n"
            "MOVI R2, {n}\n"
            "DEC R2\n"
            "fib_loop:\n"
            "JZ R2, fib_done\n"
            "MOV R3, R1\n"
            "IADD R1, R0, R1\n"
            "MOV R0, R3\n"
            "DEC R2\n"
            "JMP fib_loop\n"
            "fib_done:\n"
            "MOV R0, R1\n"
            "HALT"
        ),
        placeholders={"n": 1},
        classifier="个",
        description="数学: 斐波那契数列第 N 项",
        domain="数学",
        tags=["数学", "斐波那契"],
        examples=["斐波那契 第 十 项"],
    ))

    # 求积 — 从1到N的乘积
    entries.append(VocabEntry(
        word="求积",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"从\s+(.+?)\s+到\s+(.+?)\s+的积"),
        template=(
            "MOVI R0, 1\n"
            "MOVI R1, {a}\n"
            "MOVI R2, {b}\n"
            "prod_loop:\n"
            "IMUL R0, R0, R1\n"
            "INC R1\n"
            "CMP R1, R2\n"
            "JL prod_loop\n"
            "IMUL R0, R0, R2\n"
            "HALT"
        ),
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="数学: 从 a 到 b 的乘积",
        domain="数学",
        tags=["数学", "乘积"],
        examples=["从 一 到 五 的积"],
    ))

    # 取反... — 取负
    entries.append(VocabEntry(
        word="取反",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"取反\s+(.+)"),
        template="MOVI R0, {a}\nINEG R0\nHALT",
        placeholders={"a": 1},
        classifier="个",
        description="数学: 取负",
        domain="数学",
        tags=["数学", "负"],
        examples=["取反 五"],
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 2: 领域词 — 智能体领域
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 告知... — 智能体告知
    entries.append(VocabEntry(
        word="告知",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"告诉\s+(.+?)\s+(.+)"),
        template="MOVI R0, {agent}\nMOVI R1, {msg}\nTELL R0, R1\nHALT",
        placeholders={"agent": 1, "msg": 2},
        classifier="位",
        description="智能体: 告知消息",
        domain="智能体",
        tags=["智能体", "告知", "A2A"],
        examples=["告诉 甲 报告船位"],
    ))

    # 询问... — 智能体询问
    entries.append(VocabEntry(
        word="询问",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"询问\s+(.+?)\s+(.+)"),
        template="MOVI R0, {agent}\nMOVI R1, {topic}\nASK R0, R1\nHALT",
        placeholders={"agent": 1, "topic": 2},
        classifier="位",
        description="智能体: 询问信息",
        domain="智能体",
        tags=["智能体", "询问", "A2A"],
        examples=["询问 甲 当前船速"],
    ))

    # 委托... — 智能体委托
    entries.append(VocabEntry(
        word="委托",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"委托\s+(.+?)\s+(.+)"),
        template="MOVI R0, {agent}\nMOVI R1, {task}\nDELEGATE R0, R1\nHALT",
        placeholders={"agent": 1, "task": 2},
        classifier="位",
        description="智能体: 委托任务",
        domain="智能体",
        tags=["智能体", "委托", "A2A"],
        examples=["委托 甲 检查引擎"],
    ))

    # 广播... — 智能体广播
    entries.append(VocabEntry(
        word="广播",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"广播\s+(.+)"),
        template="MOVI R0, 0\nMOVI R1, {msg}\nBROADCAST R0, R1\nHALT",
        placeholders={"msg": 1},
        classifier="条",
        description="智能体: 广播消息",
        domain="智能体",
        tags=["智能体", "广播", "A2A"],
        examples=["广播 全体注意"],
    ))

    # 验证...的权限 — 信任验证
    entries.append(VocabEntry(
        word="信任验证",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"验证\s+(.+?)\s*的权限"),
        template="MOVI R0, {agent}\nMOVI R1, 1\nTRUST_CHECK R0, R1\nHALT",
        placeholders={"agent": 1},
        classifier="位",
        description="智能体: 信任验证",
        domain="智能体",
        tags=["智能体", "信任", "A2A"],
        examples=["验证 甲 的权限"],
    ))

    # 请求报告并等待 — 组合 A2A 模式
    entries.append(VocabEntry(
        word="请求并等待",
        level=VocabLevel.DOMAIN,
        pattern=re.compile(r"请求\s+(.+?)\s+报告\s+(.+?)\s+然后\s+等待"),
        template="MOVI R0, {agent}\nMOVI R1, {topic}\nASK R0, R1\nMOVI R2, 1\nHALT",
        placeholders={"agent": 1, "topic": 2},
        classifier="位",
        description="智能体: 请求报告并等待",
        domain="智能体",
        tags=["智能体", "组合", "A2A"],
        examples=["请求 甲 报告 船速 然后 等待"],
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Level 3: 复合词 — 多步骤计算瓦片
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 航海完整任务: 创建船队 → 设定航速 → 航行 → 计算燃油
    entries.append(VocabEntry(
        word="航海任务",
        level=VocabLevel.COMPOUND,
        pattern=re.compile(
            r"出发:\s*(.+?)\s*艘船,\s*航速\s*(.+?)\s*节,\s*航行\s*(.+?)\s*小时"
        ),
        template=(
            "-- 航海复合瓦片\n"
            "MOVI R0, {vessels}\n"
            "REGION_CREATE R0\n"
            "MOVI R10, {speed}\n"
            "MOVI R11, {hours}\n"
            "-- 计算航程\n"
            "IMUL R12, R10, R11\n"
            "-- 计算燃油 (假设每海里消耗 1 单位)\n"
            "MOV R0, R12\n"
            "HALT"
        ),
        placeholders={"vessels": 1, "speed": 2, "hours": 3},
        classifier="艘",
        description="复合: 完整航海任务",
        domain="航海",
        tags=["复合", "航海", "任务"],
        examples=["出发: 三 艘船, 航速 十二 节, 航行 五 小时"],
    ))

    # 数学复合: 计算平均值
    entries.append(VocabEntry(
        word="平均值",
        level=VocabLevel.COMPOUND,
        pattern=re.compile(r"计算\s+(.+?)\s*和\s+(.+?)\s*的平均值"),
        template=(
            "-- 平均值 = (a + b) / 2\n"
            "MOVI R0, {a}\n"
            "MOVI R1, {b}\n"
            "IADD R2, R0, R1\n"
            "MOVI R3, 2\n"
            "IDIV R0, R2, R3\n"
            "HALT"
        ),
        placeholders={"a": 1, "b": 2},
        classifier="个",
        description="复合: 计算两数平均值",
        domain="数学",
        tags=["复合", "数学"],
        examples=["计算 三 和 七 的平均值"],
    ))

    # 智能体复合: 多步骤协调
    entries.append(VocabEntry(
        word="协调任务",
        level=VocabLevel.COMPOUND,
        pattern=re.compile(r"协调:\s*(.+?)\s*告知\s*(.+?),\s*(.+?)\s*委托\s*(.+)"),
        template=(
            "-- 智能体协调复合瓦片\n"
            "MOVI R0, {sender}\n"
            "MOVI R1, {msg}\n"
            "TELL R0, R1\n"
            "MOVI R0, {delegator}\n"
            "MOVI R1, {task}\n"
            "DELEGATE R0, R1\n"
            "HALT"
        ),
        placeholders={"sender": 1, "msg": 2, "delegator": 3, "task": 4},
        classifier="位",
        description="复合: 智能体协调任务",
        domain="智能体",
        tags=["复合", "智能体", "协调"],
        examples=["协调: 甲 告知 乙, 丙 委托 丁"],
    ))

    return entries


# ══════════════════════════════════════════════════════════════════════
# 全局实例和公开 API
# ══════════════════════════════════════════════════════════════════════

_global_registry: VocabularyRegistry | None = None


def get_registry() -> VocabularyRegistry:
    """
    获取全局词汇注册表 (延迟初始化)。

    首次调用时加载所有内置标准词汇。
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = VocabularyRegistry()
        _global_registry.load_defaults()
    return _global_registry


def compile_text(text: str) -> str | None:
    """
    编译中文文本为 FLUX 汇编。

    搜索所有层级的词汇，返回第一个匹配。

    Args:
        text: 中文文本

    Returns:
        FLUX 汇编字符串或 None
    """
    registry = get_registry()
    return registry.compile(text)


def search_vocabulary(text: str) -> list[VocabEntry]:
    """搜索匹配的词汇条目"""
    registry = get_registry()
    return registry.search(text)
