"""
流星 FIR — 流体中间表示 (Flux Intermediate Representation)

中文优先的 SSA (静态单赋值) 构建器。

设计哲学:
  - 主题-评论 (Topic-Comment) 结构 → 延续树 (Continuation Tree) 变换
  - 量词 (Classifier) 系统 → 类型推断基础
  - 零形回指 (Zero Anaphora) → 主题寄存器 R63 跟踪
  - 中文变量命名: 支持 甲乙丙丁、寄存器零 等中文标识符
  - 基本块以中文终止符标记: 开始/结束/跳转/循环

FIR 架构:
  1. FIRValue — SSA 值 (每次赋值产生新版本)
  2. FIRBlock — 基本块 (含中文标签终止符)
  3. FIRPhi  — Φ 节点 (量词作用域合并)
  4. FIRBuilder — 构建器 (主题追踪 + 类型推断)
  5. FIRProgram — 完整 FIR 程序

FIR → FLUX 字节码 的流程:
  FIR 程序 → 序列化基本块 → 生成汇编 → 编码为字节码
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════
# 量词类型系统 — 量词 → 类型映射
# ══════════════════════════════════════════════════════════════════════

class FirType(IntEnum):
    """FIR 类型系统 — 基于中文量词的类型层次"""
    UNKNOWN = 0         # 未知类型
    INTEGER = 1         # 整数 (通用)
    SEQUENCE = 2        # 序列 (条)
    DOCUMENT = 3        # 文档 (本)
    MACHINE = 4         # 机器 (台)
    AGENT = 5           # 智能体 (位)
    COUNTER = 6         # 计数器 (次)
    DISTANCE = 7        # 距离 (海里)
    SPEED = 8           # 速度 (节)
    VESSEL = 9          # 船只 (艘)
    MESSAGE = 10        # 消息 (封/条信息)
    STEP = 11           # 步骤 (步)
    KIND = 12           # 种类 (种)
    ANIMAL = 13         # 动物 (只)
    PERSON = 14         # 人员 (名)
    HONORED = 15        # 敬称人员 (位)
    ITEM = 16           # 事务 (件)
    WEIGHT = 17         # 重量 (吨)
    TIME = 18           # 时间 (秒/小时)
    REPORT = 19         # 报告 (份)
    ANCHOR = 20         # 锚 (锚)


# 量词 → FIR 类型映射
CLASSIFIER_TO_FIR_TYPE: dict[str, FirType] = {
    # 通用量词
    "个": FirType.INTEGER,
    "只": FirType.ANIMAL,
    "条": FirType.SEQUENCE,
    "本": FirType.DOCUMENT,
    "台": FirType.MACHINE,
    "位": FirType.HONORED,
    "名": FirType.PERSON,
    "件": FirType.ITEM,
    "种": FirType.KIND,
    "套": FirType.MACHINE,
    # 数字量词
    "次": FirType.COUNTER,
    "遍": FirType.COUNTER,
    "趟": FirType.COUNTER,
    "步": FirType.STEP,
    "轮": FirType.COUNTER,
    # 航海量词
    "艘": FirType.VESSEL,
    "节": FirType.SPEED,
    "海里": FirType.DISTANCE,
    "锚": FirType.ANCHOR,
    "帆": FirType.VESSEL,
    "吨": FirType.WEIGHT,
    # 信息量词
    "条信息": FirType.MESSAGE,
    "则": FirType.MESSAGE,
    "封": FirType.MESSAGE,
    "份": FirType.REPORT,
    "项": FirType.ITEM,
    # 时间量词
    "秒": FirType.TIME,
    "小时": FirType.TIME,
}

# FIR 类型中文名
FIR_TYPE_NAMES: dict[FirType, str] = {
    FirType.UNKNOWN: "未知",
    FirType.INTEGER: "整数",
    FirType.SEQUENCE: "序列",
    FirType.DOCUMENT: "文档",
    FirType.MACHINE: "机器",
    FirType.AGENT: "智能体",
    FirType.COUNTER: "计数器",
    FirType.DISTANCE: "距离",
    FirType.SPEED: "速度",
    FirType.VESSEL: "船只",
    FirType.MESSAGE: "消息",
    FirType.STEP: "步骤",
    FirType.KIND: "种类",
    FirType.ANIMAL: "动物",
    FirType.PERSON: "人员",
    FirType.HONORED: "敬称人员",
    FirType.ITEM: "事务",
    FirType.WEIGHT: "重量",
    FirType.TIME: "时间",
    FirType.REPORT: "报告",
    FirType.ANCHOR: "锚",
}

# FIR 类型 → 默认量词
FIR_TYPE_TO_CLASSIFIER: dict[FirType, str] = {
    FirType.INTEGER: "个",
    FirType.SEQUENCE: "条",
    FirType.DOCUMENT: "本",
    FirType.MACHINE: "台",
    FirType.AGENT: "位",
    FirType.COUNTER: "次",
    FirType.DISTANCE: "海里",
    FirType.SPEED: "节",
    FirType.VESSEL: "艘",
    FirType.MESSAGE: "条",
    FirType.STEP: "步",
    FirType.KIND: "种",
    FirType.ANIMAL: "只",
    FirType.PERSON: "名",
    FirType.HONORED: "位",
    FirType.ITEM: "件",
    FirType.WEIGHT: "吨",
    FirType.TIME: "秒",
    FirType.REPORT: "份",
    FirType.ANCHOR: "锚",
}


# ══════════════════════════════════════════════════════════════════════
# 中文寄存器名映射
# ══════════════════════════════════════════════════════════════════════

# 天干寄存器名: 甲乙丙丁戊己庚辛壬癸 → R0-R9
TIANGAN_REGISTERS: dict[str, int] = {
    "甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4,
    "己": 5, "庚": 6, "辛": 7, "壬": 8, "癸": 9,
}

# 完整中文寄存器名
ZH_REGISTERS_FULL: dict[str, int] = {
    "寄存器零": 0, "寄存器一": 1, "寄存器二": 2, "寄存器三": 3,
    "寄存器四": 4, "寄存器五": 5, "寄存器六": 6, "寄存器七": 7,
    "寄存器八": 8, "寄存器九": 9, "寄存器十": 10,
    **TIANGAN_REGISTERS,
}

# 中文数字映射 (简化版，用于 FIR 内部)
ZH_SIMPLE_DIGITS: dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}


# ══════════════════════════════════════════════════════════════════════
# FIR 操作码 — 中间表示层指令集
# ══════════════════════════════════════════════════════════════════════

class FirOpcode(IntEnum):
    """FIR 中间表示操作码"""
    # 通用
    NOP = auto()         # 空操作

    # 算术运算
    CONST = auto()       # 常量定义
    ADD = auto()         # 加法
    SUB = auto()         # 减法
    MUL = auto()         # 乘法
    DIV = auto()         # 除法
    MOD = auto()         # 取模
    NEG = auto()         # 取反
    INC = auto()         # 自增
    DEC = auto()         # 自减

    # 位运算
    AND = auto()         # 按位与
    OR = auto()          # 按位或
    XOR = auto()         # 按位异或
    NOT = auto()         # 按位取反
    SHL = auto()         # 左移
    SHR = auto()         # 右移

    # 比较
    CMP_EQ = auto()      # 等于
    CMP_LT = auto()      # 小于
    CMP_LE = auto()      # 小于等于
    CMP_GT = auto()      # 大于
    CMP_GE = auto()      # 大于等于

    # 数据移动
    COPY = auto()        # 复制
    LOAD = auto()        # 加载
    STORE = auto()       # 存储

    # 控制流
    JUMP = auto()        # 无条件跳转
    BRANCH = auto()      # 条件分支
    RETURN = auto()      # 返回
    CALL = auto()        # 函数调用

    # 特殊 (中文特定)
    TOPIC_SET = auto()   # 设置主题 (MOV → R63)
    TOPIC_GET = auto()   # 获取主题 (零形回指)
    CLASSIFY = auto()    # 量词类型标注
    HONORIFY = auto()    # 敬称提升

    # 终止符
    HALT = auto()        # 停机
    UNREACHABLE = auto() # 不可达


# FIR 操作码中文名
FIR_OPCODE_NAMES: dict[FirOpcode, str] = {
    FirOpcode.NOP: "空操作",
    FirOpcode.CONST: "常量",
    FirOpcode.ADD: "加法",
    FirOpcode.SUB: "减法",
    FirOpcode.MUL: "乘法",
    FirOpcode.DIV: "除法",
    FirOpcode.MOD: "取模",
    FirOpcode.NEG: "取反",
    FirOpcode.INC: "自增",
    FirOpcode.DEC: "自减",
    FirOpcode.AND: "按位与",
    FirOpcode.OR: "按位或",
    FirOpcode.XOR: "按位异或",
    FirOpcode.NOT: "按位取反",
    FirOpcode.SHL: "左移",
    FirOpcode.SHR: "右移",
    FirOpcode.CMP_EQ: "等于",
    FirOpcode.CMP_LT: "小于",
    FirOpcode.CMP_LE: "小于等于",
    FirOpcode.CMP_GT: "大于",
    FirOpcode.CMP_GE: "大于等于",
    FirOpcode.COPY: "复制",
    FirOpcode.LOAD: "加载",
    FirOpcode.STORE: "存储",
    FirOpcode.JUMP: "跳转",
    FirOpcode.BRANCH: "分支",
    FirOpcode.RETURN: "返回",
    FirOpcode.CALL: "调用",
    FirOpcode.TOPIC_SET: "设定主题",
    FirOpcode.TOPIC_GET: "获取主题",
    FirOpcode.CLASSIFY: "量词标注",
    FirOpcode.HONORIFY: "敬称提升",
    FirOpcode.HALT: "停机",
    FirOpcode.UNREACHABLE: "不可达",
}


# ══════════════════════════════════════════════════════════════════════
# FIR 值系统 — SSA 静态单赋值
# ══════════════════════════════════════════════════════════════════════

class FIRValue:
    """
    FIR SSA 值 — 每次赋值产生唯一的版本化值。

    SSA 保证: 每个变量只被赋值一次。如果需要重新赋值，
    则创建新版本 (如 船速_0, 船速_1, 船速_2)。

    中文特点:
      - 变量名使用中文 (如 船速、航程、燃料)
      - 版本号用阿拉伯数字后缀
      - 类型通过量词自动推断
    """

    _counter: int = 0

    def __init__(
        self,
        name: str,
        version: int | None = None,
        fir_type: FirType = FirType.UNKNOWN,
        classifier: str = "",
        register: int = -1,
        is_topic: bool = False,
        context: str = "",
    ) -> None:
        """
        初始化 FIR 值。

        Args:
            name:       中文变量名 (如 "船速", "航程", "甲")
            version:    SSA 版本号 (自动递增如果为 None)
            fir_type:   FIR 类型 (从量词推断)
            classifier: 原始量词 (如 "艘", "节", "海里")
            register:   分配的寄存器编号 (-1 表示未分配)
            is_topic:   是否为主题寄存器值 (R63)
            context:    敬称/语境标记 (如 "尊敬", "正式")
        """
        self.name = name
        if version is None:
            FIRValue._counter += 1
            self.version = FIRValue._counter
        else:
            self.version = version
        self.fir_type = fir_type
        self.classifier = classifier
        self.register = register
        self.is_topic = is_topic
        self.context = context

    @property
    def full_name(self) -> str:
        """完整 SSA 名称: 变量名_版本号"""
        return f"{self.name}_{self.version}"

    @property
    def type_name(self) -> str:
        """类型的中文名称"""
        return FIR_TYPE_NAMES.get(self.fir_type, "未知")

    @property
    def type_classifier(self) -> str:
        """类型对应的默认量词"""
        return FIR_TYPE_TO_CLASSIFIER.get(self.fir_type, "个")

    def __repr__(self) -> str:
        ctx = f" [{self.context}]" if self.context else ""
        reg = f" → R{self.register}" if self.register >= 0 else ""
        topic = " ★" if self.is_topic else ""
        return (
            f"FIRValue({self.full_name}: {self.type_name}"
            f"('{self.classifier}'){ctx}{reg}{topic})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FIRValue):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    @classmethod
    def reset_counter(cls) -> None:
        """重置全局版本计数器"""
        cls._counter = 0


# ══════════════════════════════════════════════════════════════════════
# FIR 指令 — 单条中间表示指令
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FIRInstruction:
    """
    FIR 指令 — 单条中间表示指令。

    每条指令包含: 操作码、操作数列表、可选的结果值。
    """
    opcode: FirOpcode
    operands: list[FIRValue | int | str | None] = field(default_factory=list)
    result: FIRValue | None = None
    comment: str = ""
    source_line: str = ""  # 原始中文源代码行

    @property
    def opcode_name(self) -> str:
        """操作码中文名"""
        return FIR_OPCODE_NAMES.get(self.opcode, "未知")

    def __repr__(self) -> str:
        ops_str = ", ".join(
            repr(op) if isinstance(op, FIRValue) else str(op)
            for op in self.operands
        )
        res = f" → {self.result}" if self.result else ""
        cmt = f"  # {self.comment}" if self.comment else ""
        src = f"  /* {self.source_line} */" if self.source_line else ""
        return f"  {self.opcode_name}({ops_str}){res}{cmt}{src}"


# ══════════════════════════════════════════════════════════════════════
# FIR Phi 节点 — 量词作用域合并
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FIRPhi:
    """
    FIR Φ 节点 — SSA 合并点。

    在中文语法中，当量词作用域交叉时需要 Φ 节点。
    例如: "这艘船(艘→船只)，那条河(条→序列)" 在分支合并时
    需要一个 Φ 节点来确定变量的最终类型。

    Phi 节点的语义:
      - 合并来自不同基本块的同一变量
      - 保持量词类型的一致性
      - 如果类型冲突，提升到最近的公共父类型
    """
    result: FIRValue
    sources: list[tuple[FIRValue, str]] = field(default_factory=list)
    # sources: [(value, block_label), ...]

    @property
    def merged_type(self) -> FirType:
        """合并后的类型 — 取最具体的公共类型"""
        types = set(
            src[0].fir_type for src in self.sources
            if src[0].fir_type != FirType.UNKNOWN
        )
        if len(types) == 0:
            return FirType.UNKNOWN
        if len(types) == 1:
            return types.pop()
        # 多类型冲突 → 提升为通用类型
        return FirType.INTEGER

    @property
    def merged_classifier(self) -> str:
        """合并后的量词 — 取第一个已知量词"""
        for src in self.sources:
            if src[0].classifier:
                return src[0].classifier
        return "个"

    def __repr__(self) -> str:
        srcs = ", ".join(
            f"{val.full_name}@{lbl}" for val, lbl in self.sources
        )
        return f"  Φ({self.result.full_name}) ← {srcs}"


# ══════════════════════════════════════════════════════════════════════
# FIR 基本块 — 带中文终止符的代码块
# ══════════════════════════════════════════════════════════════════════

# 中文终止符名称
TERMINATOR_NAMES = {
    "开始": "开始",
    "结束": "结束",
    "跳转": "跳转",
    "循环": "循环",
    "分支": "分支",
    "返回": "返回",
    "停机": "停机",
    "不可达": "不可达",
}


class FIRBlock:
    """
    FIR 基本块 — 一段直线代码，以单个终止符结尾。

    中文终止符:
      开始  — 程序/函数入口
      结束  — 程序正常结束
      跳转  — 无条件跳转到目标块
      循环  — 循环回跳
      分支  — 条件分支 (两个目标)
      返回  — 函数返回
      停机  — 虚拟机停机
      不可达 — 不可达代码

    每个基本块包含:
      - 指令列表 (按顺序执行)
      - Phi 节点列表 (在块开头，用于 SSA 合并)
      - 终止符 (块的最后一条指令)
    """

    def __init__(
        self,
        label: str,
        terminator: str = "结束",
        predecessors: list[str] | None = None,
    ) -> None:
        """
        初始化 FIR 基本块。

        Args:
            label:         中文标签 (如 "主程序", "循环体", "检查条件")
            terminator:    终止符类型 (开始/结束/跳转/循环/分支/返回/停机)
            predecessors:  前驱块标签列表
        """
        self.label = label
        self.terminator = terminator
        self.predecessors = predecessors or []
        self.instructions: list[FIRInstruction] = []
        self.phi_nodes: list[FIRPhi] = []
        self.successors: list[str] = []

        # 主题上下文 — 该块的主题寄存器值
        self.topic_value: FIRValue | None = None

    def add_instruction(self, instr: FIRInstruction) -> None:
        """添加一条指令到基本块"""
        self.instructions.append(instr)

    def add_phi(self, phi: FIRPhi) -> None:
        """添加一个 Phi 节点到基本块"""
        self.phi_nodes.append(phi)

    def set_terminator_jump(self, target: str) -> None:
        """设置终止符为跳转到目标块"""
        self.terminator = "跳转"
        self.successors = [target]

    def set_terminator_loop(self, target: str) -> None:
        """设置终止符为循环回跳"""
        self.terminator = "循环"
        self.successors = [target]

    def set_terminator_branch(self, true_target: str, false_target: str) -> None:
        """设置终止符为条件分支"""
        self.terminator = "分支"
        self.successors = [true_target, false_target]

    def set_terminator_return(self) -> None:
        """设置终止符为返回"""
        self.terminator = "返回"
        self.successors = []

    def set_terminator_halt(self) -> None:
        """设置终止符为停机"""
        self.terminator = "停机"
        self.successors = []

    @property
    def is_entry(self) -> bool:
        """是否为入口块"""
        return self.terminator == "开始"

    @property
    def has_terminator(self) -> bool:
        """是否已设置终止指令"""
        if not self.instructions:
            return False
        last = self.instructions[-1]
        return last.opcode in (
            FirOpcode.JUMP, FirOpcode.BRANCH,
            FirOpcode.RETURN, FirOpcode.HALT,
            FirOpcode.UNREACHABLE,
        )

    def __repr__(self) -> str:
        pred_str = f" ← [{', '.join(self.predecessors)}]" if self.predecessors else ""
        succ_str = f" → [{', '.join(self.successors)}]" if self.successors else ""
        return f"FIRBlock({self.label}: {self.terminator}{pred_str}{succ_str})"


# ══════════════════════════════════════════════════════════════════════
# 主题-评论 → 延续树 变换
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TopicCommentNode:
    """
    主题-评论语法树节点。

    中文语法特点: 主题-评论 (Topic-Comment) 结构。
    "这个值，加三" 中:
      - "这个值" 是主题 (Topic)
      - "加三" 是评论 (Comment)

    零形回指 (Zero Anaphora):
      "再加三" — 主题为空 (零形)，隐式指向上一次的主题。
    """
    topic: str                         # 主题文本 (空字符串表示零形回指)
    comment: str                       # 评论文本
    classifier: str = ""               # 关联的量词
    topic_value: FIRValue | None = None  # 主题对应的 FIR 值
    children: list[TopicCommentNode] = field(default_factory=list)
    source_line: str = ""

    @property
    def has_zero_anaphora(self) -> bool:
        """是否存在零形回指"""
        return not self.topic and not self.comment.startswith("加载")

    def __repr__(self) -> str:
        topic_disp = f"'{self.topic}'" if self.topic else "∅(零形回指)"
        clf = f"[{self.classifier}]" if self.classifier else ""
        return f"TC({topic_disp}, '{self.comment}'{clf})"


class ContinuationTree:
    """
    延续树 — 主题-评论结构变换后的控制流表示。

    将中文主题-评论序列变换为基于延续的树结构:
    1. 识别每个句子的主题和评论
    2. 零形回指链接到上一个主题
    3. 构建延续链 (每个评论的输出是下一个评论的输入)
    4. 生成基本块

    示例:
      "加载甲为三。甲加二。打印。" →
        节点1: topic="甲", comment="加载为三"
        节点2: topic="甲", comment="加二"
        节点3: topic=""  , comment="打印" (零形回指 → 甲)
    """

    def __init__(self) -> None:
        self.nodes: list[TopicCommentNode] = []
        self._last_topic: str = ""

    def add_node(self, node: TopicCommentNode) -> None:
        """添加节点到延续树"""
        # 处理零形回指
        if node.has_zero_anaphora and self._last_topic:
            node.topic = self._last_topic
        elif node.topic:
            self._last_topic = node.topic

        self.nodes.append(node)

    def get_chain(self) -> list[TopicCommentNode]:
        """获取延续链 (按顺序的节点列表)"""
        return list(self.nodes)

    def to_basic_blocks(self) -> list[FIRBlock]:
        """
        将延续树转换为 FIR 基本块。

        规则:
          1. 每个连续的非分支节点序列形成一个基本块
          2. 遇到循环/条件时分割块
          3. 最后一个块以停机结束
        """
        blocks: list[FIRBlock] = []

        if not self.nodes:
            # 空程序 — 创建一个空的基本块
            block = FIRBlock("空程序", terminator="停机")
            blocks.append(block)
            return blocks

        # 创建入口块
        entry = FIRBlock("开始", terminator="开始")
        blocks.append(entry)

        # 当前工作块
        current_block = FIRBlock("主体")
        blocks.append(current_block)

        for node in self.nodes:
            # 简单启发式: 检测是否需要新块
            needs_new_block = any(
                keyword in node.comment
                for keyword in ("循环", "如果", "判断", "跳转", "返回", "停机")
            )

            if needs_new_block:
                current_block.set_terminator_halt()
                current_block = FIRBlock(f"块_{len(blocks)}")
                blocks.append(current_block)

        # 结束块
        current_block.set_terminator_halt()

        return blocks


# ══════════════════════════════════════════════════════════════════════
# 敬称/语境感知系统
# ══════════════════════════════════════════════════════════════════════

# 敬称标记
HONORIFIC_MARKERS = {
    "阁下": "尊敬",
    "殿下": "尊敬",
    "尊": "尊敬",
    "贵": "尊敬",
    "贵方": "尊敬",
    "令": "尊敬",
    "先生": "礼貌",
    "女士": "礼貌",
    "船长": "专业",
    "舰长": "专业",
}

# 语境标记
CONTEXT_MARKERS = {
    "文言": "文言",
    "古语": "文言",
    "白话": "白话",
    "口语": "口语",
    "正式": "正式",
}


def detect_honorific(text: str) -> str:
    """
    检测文本中的敬称标记。

    返回敬称级别: "尊敬", "礼貌", "专业", "" (无敬称)
    """
    for marker, level in HONORIFIC_MARKERS.items():
        if marker in text:
            return level
    return ""


def detect_context(text: str) -> str:
    """
    检测文本的语境风格。

    返回: "文言", "白话", "正式", "口语", "" (未检测到)
    """
    for marker, context in CONTEXT_MARKERS.items():
        if marker in text:
            return context
    return ""


# ══════════════════════════════════════════════════════════════════════
# FIR 构建器 — 主构建接口
# ══════════════════════════════════════════════════════════════════════

class FIRBuilder:
    """
    FIR 构建器 — 构建 SSA 形式的 FIR 程序。

    功能:
      - 创建 SSA 值 (自动版本管理)
      - 构建基本块 (中文标签)
      - 主题寄存器 R63 跟踪 (零形回指)
      - 量词 → 类型推断
      - 敬称/语境感知
      - 主题-评论 → 延续树变换

    使用示例:
        builder = FIRBuilder()
        builder.topic_set("船速", 12, "节")
        val = builder.add(船速_val, builder.const(3))
        builder.halt()
    """

    # 主题寄存器编号 (R63)
    TOPIC_REGISTER = 63

    def __init__(self) -> None:
        """初始化 FIR 构建器"""
        self.blocks: list[FIRBlock] = []
        self.values: dict[str, list[FIRValue]] = {}  # name → [versioned values]
        self.current_block: FIRBlock | None = None
        self.continuation_tree = ContinuationTree()
        self.topic_stack: list[FIRValue] = []  # 主题值栈
        self.context: str = ""
        self.honorific: str = ""

        # 创建入口块
        self._new_block("开始", terminator="开始")

    # ── 基本块管理 ─────────────────────────────────────────────

    def _new_block(self, label: str, terminator: str = "结束") -> FIRBlock:
        """创建新的基本块"""
        block = FIRBlock(label, terminator=terminator)
        if self.current_block is not None:
            block.predecessors = [self.current_block.label]
        self.blocks.append(block)
        self.current_block = block
        return block

    def new_block(self, label: str) -> FIRBlock:
        """
        创建新的基本块 (公开 API)。

        Args:
            label: 中文标签

        Returns:
            新创建的 FIRBlock
        """
        return self._new_block(label)

    def seal_block(self, terminator: str = "结束") -> None:
        """
        封闭当前基本块。

        Args:
            terminator: 终止符名称 (结束/跳转/循环/分支/返回/停机)
        """
        if self.current_block:
            self.current_block.terminator = terminator

    def set_current_block(self, label: str) -> bool:
        """切换到指定标签的基本块"""
        for block in self.blocks:
            if block.label == label:
                self.current_block = block
                return True
        return False

    # ── SSA 值管理 ─────────────────────────────────────────────

    def _make_value(
        self,
        name: str,
        fir_type: FirType = FirType.UNKNOWN,
        classifier: str = "",
        register: int = -1,
        is_topic: bool = False,
    ) -> FIRValue:
        """创建新的 SSA 值"""
        val = FIRValue(
            name=name,
            fir_type=fir_type,
            classifier=classifier,
            register=register,
            is_topic=is_topic,
            context=self.honorific,
        )
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(val)
        return val

    def get_latest(self, name: str) -> FIRValue | None:
        """获取变量的最新版本"""
        versions = self.values.get(name, [])
        return versions[-1] if versions else None

    def const(
        self,
        value: int,
        name: str = "常量",
        classifier: str = "个",
    ) -> FIRValue:
        """
        创建常量值。

        Args:
            value:      整数值
            name:       变量名
            classifier: 量词 (默认 "个")

        Returns:
            新创建的 FIRValue
        """
        fir_type = CLASSIFIER_TO_FIR_TYPE.get(classifier, FirType.INTEGER)
        val = self._make_value(name, fir_type=fir_type, classifier=classifier)
        reg = self._alloc_register()

        instr = FIRInstruction(
            opcode=FirOpcode.CONST,
            operands=[value],
            result=val,
            comment=f"{name} = {value}{classifier}",
        )
        self._emit(instr)
        val.register = reg
        return val

    def copy(self, source: FIRValue, name: str = "") -> FIRValue:
        """
        复制一个值。

        Args:
            source: 源值
            name:   新变量名 (如果为空则使用源名称)

        Returns:
            新创建的 FIRValue (不同版本)
        """
        target_name = name or source.name
        val = self._make_value(
            target_name,
            fir_type=source.fir_type,
            classifier=source.classifier,
        )
        reg = self._alloc_register()

        instr = FIRInstruction(
            opcode=FirOpcode.COPY,
            operands=[source],
            result=val,
            comment=f"复制 {source.full_name}",
        )
        self._emit(instr)
        val.register = reg
        return val

    # ── 算术运算 ───────────────────────────────────────────────

    def add(self, a: FIRValue, b: FIRValue, name: str = "和") -> FIRValue:
        """加法: a + b"""
        val = self._make_value(name, fir_type=FirType.INTEGER, classifier="个")
        instr = FIRInstruction(
            opcode=FirOpcode.ADD,
            operands=[a, b],
            result=val,
            comment=f"{a.full_name} + {b.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def sub(self, a: FIRValue, b: FIRValue, name: str = "差") -> FIRValue:
        """减法: a - b"""
        val = self._make_value(name, fir_type=FirType.INTEGER, classifier="个")
        instr = FIRInstruction(
            opcode=FirOpcode.SUB,
            operands=[a, b],
            result=val,
            comment=f"{a.full_name} - {b.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def mul(self, a: FIRValue, b: FIRValue, name: str = "积") -> FIRValue:
        """乘法: a × b"""
        val = self._make_value(name, fir_type=FirType.INTEGER, classifier="个")
        instr = FIRInstruction(
            opcode=FirOpcode.MUL,
            operands=[a, b],
            result=val,
            comment=f"{a.full_name} × {b.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def div(self, a: FIRValue, b: FIRValue, name: str = "商") -> FIRValue:
        """除法: a ÷ b"""
        val = self._make_value(name, fir_type=FirType.INTEGER, classifier="个")
        instr = FIRInstruction(
            opcode=FirOpcode.DIV,
            operands=[a, b],
            result=val,
            comment=f"{a.full_name} ÷ {b.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def mod(self, a: FIRValue, b: FIRValue, name: str = "余") -> FIRValue:
        """取模: a % b"""
        val = self._make_value(name, fir_type=FirType.INTEGER, classifier="个")
        instr = FIRInstruction(
            opcode=FirOpcode.MOD,
            operands=[a, b],
            result=val,
            comment=f"{a.full_name} % {b.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def neg(self, a: FIRValue, name: str = "反") -> FIRValue:
        """取反: -a"""
        val = self._make_value(name, fir_type=a.fir_type, classifier=a.classifier)
        instr = FIRInstruction(
            opcode=FirOpcode.NEG,
            operands=[a],
            result=val,
            comment=f"-{a.full_name}",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def inc(self, a: FIRValue, name: str = "") -> FIRValue:
        """自增: a + 1"""
        target_name = name or a.name
        val = self._make_value(target_name, fir_type=a.fir_type, classifier=a.classifier)
        instr = FIRInstruction(
            opcode=FirOpcode.INC,
            operands=[a],
            result=val,
            comment=f"{a.full_name}++",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def dec(self, a: FIRValue, name: str = "") -> FIRValue:
        """自减: a - 1"""
        target_name = name or a.name
        val = self._make_value(target_name, fir_type=a.fir_type, classifier=a.classifier)
        instr = FIRInstruction(
            opcode=FirOpcode.DEC,
            operands=[a],
            result=val,
            comment=f"{a.full_name}--",
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    # ── 比较运算 ───────────────────────────────────────────────

    def cmp_eq(self, a: FIRValue, b: FIRValue, name: str = "等") -> FIRValue:
        """相等比较"""
        val = self._make_value(name, fir_type=FirType.INTEGER)
        instr = FIRInstruction(
            opcode=FirOpcode.CMP_EQ,
            operands=[a, b],
            result=val,
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def cmp_lt(self, a: FIRValue, b: FIRValue, name: str = "小于") -> FIRValue:
        """小于比较"""
        val = self._make_value(name, fir_type=FirType.INTEGER)
        instr = FIRInstruction(
            opcode=FirOpcode.CMP_LT,
            operands=[a, b],
            result=val,
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    def cmp_gt(self, a: FIRValue, b: FIRValue, name: str = "大于") -> FIRValue:
        """大于比较"""
        val = self._make_value(name, fir_type=FirType.INTEGER)
        instr = FIRInstruction(
            opcode=FirOpcode.CMP_GT,
            operands=[a, b],
            result=val,
        )
        self._emit(instr)
        val.register = self._alloc_register()
        return val

    # ── 控制流 ─────────────────────────────────────────────────

    def jump(self, target_label: str) -> None:
        """无条件跳转"""
        instr = FIRInstruction(
            opcode=FirOpcode.JUMP,
            operands=[target_label],
            comment=f"跳转至 {target_label}",
        )
        self._emit(instr)
        if self.current_block:
            self.current_block.set_terminator_jump(target_label)

    def branch(
        self,
        condition: FIRValue,
        true_label: str,
        false_label: str,
    ) -> None:
        """条件分支"""
        instr = FIRInstruction(
            opcode=FirOpcode.BRANCH,
            operands=[condition, true_label, false_label],
            comment=f"如果 {condition.full_name} 则 {true_label} 否则 {false_label}",
        )
        self._emit(instr)
        if self.current_block:
            self.current_block.set_terminator_branch(true_label, false_label)

    def ret(self, value: FIRValue | None = None) -> None:
        """返回"""
        instr = FIRInstruction(
            opcode=FirOpcode.RETURN,
            operands=[value] if value else [],
            comment=f"返回 {value.full_name}" if value else "返回",
        )
        self._emit(instr)
        if self.current_block:
            self.current_block.set_terminator_return()

    def halt(self) -> None:
        """停机"""
        instr = FIRInstruction(
            opcode=FirOpcode.HALT,
            operands=[],
            comment="停机",
        )
        self._emit(instr)
        if self.current_block:
            self.current_block.set_terminator_halt()

    # ── 主题寄存器 (R63) 操作 ──────────────────────────────────

    def topic_set(self, name: str, value: int, classifier: str = "个") -> FIRValue:
        """
        设置主题寄存器 R63。

        在中文语法中，主题 (Topic) 是句子的首要关注对象。
        "这艘船，速度是十二节" — "这艘船" 是主题。

        R63 存储当前主题，后续操作可通过零形回指隐式引用。

        Args:
            name:       主题名称 (中文)
            value:      主题值 (整数)
            classifier: 量词

        Returns:
            主题 FIRValue (标记为 is_topic=True)
        """
        fir_type = CLASSIFIER_TO_FIR_TYPE.get(classifier, FirType.INTEGER)
        val = self._make_value(
            name,
            fir_type=fir_type,
            classifier=classifier,
            register=self.TOPIC_REGISTER,
            is_topic=True,
        )
        val.context = self.honorific

        instr = FIRInstruction(
            opcode=FirOpcode.TOPIC_SET,
            operands=[value, classifier],
            result=val,
            comment=f"设定主题: {name}({value}{classifier}) → R{self.TOPIC_REGISTER}",
        )
        self._emit(instr)

        # 推入主题栈
        self.topic_stack.append(val)
        return val

    def topic_get(self) -> FIRValue | None:
        """
        获取当前主题 (零形回指)。

        零形回指 (Zero Anaphora) — 中文的省略主语:
        "再加三" — 隐含主题为 R63 的当前值。

        Returns:
            当前主题 FIRValue，如果没有主题则返回 None
        """
        if not self.topic_stack:
            return None

        topic_val = self.topic_stack[-1]

        # 创建零形回指值
        zero_val = self._make_value(
            topic_val.name,
            fir_type=topic_val.fir_type,
            classifier=topic_val.classifier,
            register=self.TOPIC_REGISTER,
            is_topic=True,
        )

        instr = FIRInstruction(
            opcode=FirOpcode.TOPIC_GET,
            operands=[],
            result=zero_val,
            comment=f"零形回指: {topic_val.full_name} (∅)",
        )
        self._emit(instr)
        return zero_val

    def topic_pop(self) -> FIRValue | None:
        """弹出主题栈"""
        if self.topic_stack:
            return self.topic_stack.pop()
        return None

    @property
    def current_topic(self) -> FIRValue | None:
        """获取当前主题 (不创建新值)"""
        return self.topic_stack[-1] if self.topic_stack else None

    # ── 量词类型标注 ───────────────────────────────────────────

    def classify(self, value: FIRValue, classifier: str) -> FIRValue:
        """
        为值标注量词类型。

        Args:
            value:      原始值
            classifier: 量词

        Returns:
            带类型标注的新值
        """
        fir_type = CLASSIFIER_TO_FIR_TYPE.get(classifier, FirType.UNKNOWN)
        typed_val = self._make_value(
            value.name,
            fir_type=fir_type,
            classifier=classifier,
        )

        instr = FIRInstruction(
            opcode=FirOpcode.CLASSIFY,
            operands=[value, classifier, int(fir_type)],
            result=typed_val,
            comment=f"量词标注: {value.full_name} → {classifier}({FIR_TYPE_NAMES[fir_type]})",
        )
        self._emit(instr)
        return typed_val

    def honorify(self, value: FIRValue, level: str = "尊敬") -> FIRValue:
        """
        敬称提升 — 提升值的语境级别。

        在中文中，对不同对象使用不同敬称:
        "位" > "名" > "个" — 从敬称到中性

        Args:
            value: 原始值
            level: 敬称级别

        Returns:
            敬称提升后的新值
        """
        honored_val = self._make_value(
            value.name,
            fir_type=value.fir_type,
            classifier=classifier_upgrade(value.classifier, level),
            is_topic=value.is_topic,
        )
        honored_val.context = level

        instr = FIRInstruction(
            opcode=FirOpcode.HONORIFY,
            operands=[value, level],
            result=honored_val,
            comment=f"敬称提升: {value.classifier} → {honored_val.classifier} [{level}]",
        )
        self._emit(instr)
        return honored_val

    # ── Phi 节点 ───────────────────────────────────────────────

    def phi(
        self,
        name: str,
        sources: list[tuple[FIRValue, str]],
        classifier: str = "",
    ) -> FIRValue:
        """
        创建 Phi 节点。

        Args:
            name:       变量名
            sources:    [(value, block_label), ...]
            classifier: 量词

        Returns:
            Phi 合并后的值
        """
        # 推断合并类型
        types = set(
            v.fir_type for v, _ in sources if v.fir_type != FirType.UNKNOWN
        )
        merged_type = types.pop() if len(types) == 1 else FirType.INTEGER
        if not classifier:
            classifier = sources[0][0].classifier if sources else "个"

        result_val = self._make_value(
            name,
            fir_type=merged_type,
            classifier=classifier,
        )

        phi_node = FIRPhi(result=result_val, sources=sources)
        if self.current_block:
            self.current_block.add_phi(phi_node)

        return result_val

    # ── 主题-评论 解析 ─────────────────────────────────────────

    def parse_topic_comment(self, line: str) -> TopicCommentNode | None:
        """
        解析中文源代码行为主题-评论结构。

        识别模式:
          "XXX 为 YYY"        → topic=XXX, comment="为 YYY"
          "XXX 加/减/乘 YYY"  → topic=XXX, comment="加/减/乘 YYY"
          "加载 XXX 为 YYY"    → topic=XXX, comment="加载为 YYY"
          "告诉 XXX YYY"       → topic=XXX, comment="告诉 YYY"
          "加/减/乘 YYY"       → topic="", comment="加/减/乘 YYY" (零形回指)

        Args:
            line: 中文源代码行

        Returns:
            TopicCommentNode 或 None
        """
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("//") or line.startswith("#"):
            return None

        # 检测语境/敬称
        honorific = detect_honorific(line)
        if honorific:
            self.honorific = honorific
        context = detect_context(line)
        if context:
            self.context = context

        # 模式: "加载 主题 为 值" (优先匹配加载)
        m = re.match(r"^加载\s*(.+?)\s*为\s*(.+)$", line)
        if m:
            topic = m.group(1).strip()
            value = m.group(2).strip()
            clf = self._extract_classifier(value)
            node = TopicCommentNode(
                topic=topic,
                comment=f"加载为 {value}",
                classifier=clf,
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "计算 主题 加/减/乘/除以 值"
        m = re.match(r"^计算\s*(.+?)\s*(加|减|乘|除以|的余数)\s*(.+)$", line)
        if m:
            topic = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip()
            clf = self._extract_classifier(val)
            node = TopicCommentNode(
                topic=topic,
                comment=f"计算 {op} {val}",
                classifier=clf,
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "主题 加/减/乘/除以 值"
        m = re.match(r"^(.+?)\s*(加|减|乘|除以|的余数)\s*(.+)$", line)
        if m:
            topic = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip()
            clf = self._extract_classifier(val)
            node = TopicCommentNode(
                topic=topic,
                comment=f"{op} {val}",
                classifier=clf,
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "主题 为 值" (通用)
        m = re.match(r"^(.+?)\s*为\s*(.+)$", line)
        if m:
            topic = m.group(1).strip()
            comment = m.group(2).strip()
            # 提取量词
            clf = self._extract_classifier(comment)
            node = TopicCommentNode(
                topic=topic,
                comment=f"为 {comment}",
                classifier=clf,
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "告诉/询问/委托 主题 内容"
        m = re.match(r"^(告诉|询问|委托|广播)\s+(.+?)\s+(.+)$", line)
        if m:
            verb = m.group(1).strip()
            topic = m.group(2).strip()
            content = m.group(3).strip()
            node = TopicCommentNode(
                topic=topic,
                comment=f"{verb} {content}",
                classifier="位",  # 智能体用位
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "增加/减少 主题" (零形回指情况下的显式引用)
        m = re.match(r"^(增加|减少|打印|返回|停机)\s*(.*)$", line)
        if m:
            verb = m.group(1).strip()
            rest = m.group(2).strip()
            node = TopicCommentNode(
                topic="",
                comment=f"{verb} {rest}".strip(),
                classifier="",
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        # 模式: "N 的阶乘"
        m = re.match(r"^(.+?)\s*的阶乘$", line)
        if m:
            topic = m.group(1).strip()
            node = TopicCommentNode(
                topic=topic,
                comment="的阶乘",
                classifier="个",
                source_line=line,
            )
            self.continuation_tree.add_node(node)
            return node

        return None

    def _extract_classifier(self, text: str) -> str:
        """从文本中提取量词"""
        # 常见模式: "三艘船", "十二节速度", "一百海里"
        multi_char = ["海里", "条信息"]
        for clf in multi_char:
            if clf in text:
                return clf
        # 单字量词
        for clf in CLASSIFIER_TO_FIR_TYPE:
            if len(clf) == 1 and clf in text:
                return clf
        return ""

    # ── 内部工具 ───────────────────────────────────────────────

    def _emit(self, instr: FIRInstruction) -> None:
        """发射一条指令到当前基本块"""
        if self.current_block is None:
            return
        self.current_block.add_instruction(instr)

    def _alloc_register(self) -> int:
        """分配寄存器 (简单策略: 递增分配，避免 R63)"""
        # 找出所有已使用的寄存器
        used_regs: set[int] = set()
        for block in self.blocks:
            for instr in block.instructions:
                if instr.result and instr.result.register >= 0:
                    used_regs.add(instr.result.register)

        # 从 R0 开始分配，跳过 R63 (主题寄存器)
        for r in range(63):
            if r not in used_regs:
                return r
        return 0  # 回退

    # ── 序列化 ─────────────────────────────────────────────────

    def to_assembly(self) -> str:
        """
        将 FIR 程序序列化为 FLUX 汇编文本。

        将 FIR 指令映射到 FLUX 字节码指令:
          CONST  → MOVI
          ADD    → IADD
          SUB    → ISUB
          MUL    → IMUL
          DIV    → IDIV
          MOD    → IMOD
          NEG    → INEG
          INC    → INC
          DEC    → DEC
          COPY   → MOV
          TOPIC_SET → MOVI R63
          TOPIC_GET → MOV Rn, R63
          HALT   → HALT
        """
        lines: list[str] = []
        lines.append("-- 流星 FIR → FLUX 汇编")
        lines.append(f"-- 主题寄存器: R{self.TOPIC_REGISTER}")
        lines.append("")

        for block in self.blocks:
            # 块标签
            lines.append(f"{block.label}:")

            # Phi 节点
            for phi in block.phi_nodes:
                srcs = ", ".join(
                    f"R{v.register}" if v.register >= 0 else v.full_name
                    for v, _ in phi.sources
                )
                lines.append(f"  -- Φ: {phi.result.full_name} ← {srcs}")

            # 指令
            for instr in block.instructions:
                asm = self._instruction_to_asm(instr)
                if asm:
                    lines.append(f"  {asm}")

            lines.append("")

        return "\n".join(lines)

    def _instruction_to_asm(self, instr: FIRInstruction) -> str:
        """将单条 FIR 指令转换为 FLUX 汇编"""
        r = instr.result
        rd = f"R{r.register}" if r and r.register >= 0 else "R0"
        comment = f"  -- {instr.comment}" if instr.comment else ""

        if instr.opcode == FirOpcode.CONST:
            imm = instr.operands[0] if instr.operands else 0
            return f"MOVI {rd}, {imm}{comment}"

        elif instr.opcode == FirOpcode.COPY:
            src = instr.operands[0]
            if isinstance(src, FIRValue):
                rs = f"R{src.register}" if src.register >= 0 else "R0"
                return f"MOV {rd}, {rs}{comment}"
            return f"MOV {rd}, R0{comment}"

        elif instr.opcode == FirOpcode.ADD:
            a, b = instr.operands[0], instr.operands[1]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            rb = f"R{b.register}" if isinstance(b, FIRValue) and b.register >= 0 else "R0"
            return f"IADD {rd}, {ra}, {rb}{comment}"

        elif instr.opcode == FirOpcode.SUB:
            a, b = instr.operands[0], instr.operands[1]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            rb = f"R{b.register}" if isinstance(b, FIRValue) and b.register >= 0 else "R0"
            return f"ISUB {rd}, {ra}, {rb}{comment}"

        elif instr.opcode == FirOpcode.MUL:
            a, b = instr.operands[0], instr.operands[1]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            rb = f"R{b.register}" if isinstance(b, FIRValue) and b.register >= 0 else "R0"
            return f"IMUL {rd}, {ra}, {rb}{comment}"

        elif instr.opcode == FirOpcode.DIV:
            a, b = instr.operands[0], instr.operands[1]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            rb = f"R{b.register}" if isinstance(b, FIRValue) and b.register >= 0 else "R0"
            return f"IDIV {rd}, {ra}, {rb}{comment}"

        elif instr.opcode == FirOpcode.MOD:
            a, b = instr.operands[0], instr.operands[1]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            rb = f"R{b.register}" if isinstance(b, FIRValue) and b.register >= 0 else "R0"
            return f"IMOD {rd}, {ra}, {rb}{comment}"

        elif instr.opcode == FirOpcode.NEG:
            a = instr.operands[0]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            return f"INEG {ra}{comment}"

        elif instr.opcode == FirOpcode.INC:
            a = instr.operands[0]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            return f"INC {ra}{comment}"

        elif instr.opcode == FirOpcode.DEC:
            a = instr.operands[0]
            ra = f"R{a.register}" if isinstance(a, FIRValue) and a.register >= 0 else "R0"
            return f"DEC {ra}{comment}"

        elif instr.opcode == FirOpcode.TOPIC_SET:
            imm = instr.operands[0] if instr.operands else 0
            return f"MOVI R{self.TOPIC_REGISTER}, {imm}{comment}"

        elif instr.opcode == FirOpcode.TOPIC_GET:
            return f"MOV {rd}, R{self.TOPIC_REGISTER}{comment}"

        elif instr.opcode == FirOpcode.HALT:
            return f"HALT{comment}"

        elif instr.opcode == FirOpcode.JUMP:
            target = instr.operands[0] if instr.operands else "0"
            return f"JMP {target}{comment}"

        elif instr.opcode == FirOpcode.BRANCH:
            cond = instr.operands[0]
            true_t = instr.operands[1]
            false_t = instr.operands[2]
            cr = f"R{cond.register}" if isinstance(cond, FIRValue) and cond.register >= 0 else "R0"
            return f"JNZ {cr}, {true_t}  -- 否则 {false_t}{comment}"

        elif instr.opcode == FirOpcode.RETURN:
            return f"RET{comment}"

        return f"-- [未转换: {instr.opcode_name}]{comment}"

    # ── 调试输出 ───────────────────────────────────────────────

    def dump(self) -> str:
        """输出 FIR 程序的完整文本表示"""
        lines: list[str] = []
        lines.append("═══ FIR 程序 ═══")
        lines.append(f"基本块数: {len(self.blocks)}")
        lines.append(f"SSA 值数: {sum(len(v) for v in self.values.values())}")
        lines.append(f"主题栈深度: {len(self.topic_stack)}")
        if self.honorific:
            lines.append(f"敬称级别: {self.honorific}")
        if self.context:
            lines.append(f"语境风格: {self.context}")
        lines.append("")

        for block in self.blocks:
            lines.append(f"┌─ {block.label} ({block.terminator})")
            if block.predecessors:
                lines.append(f"│  前驱: {', '.join(block.predecessors)}")
            if block.successors:
                lines.append(f"│  后继: {', '.join(block.successors)}")

            for phi in block.phi_nodes:
                lines.append(f"│  {phi}")

            for instr in block.instructions:
                lines.append(f"│  {repr(instr)}")

            lines.append(f"└─ {block.label}_结束")
            lines.append("")

        # 延续树
        chain = self.continuation_tree.get_chain()
        if chain:
            lines.append("═══ 延续树 ═══")
            for i, node in enumerate(chain):
                lines.append(f"  [{i}] {node}")
            lines.append("")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# FIR 程序 — 完整的 FIR 程序容器
# ══════════════════════════════════════════════════════════════════════

class FIRProgram:
    """
    FIR 程序 — 完整的 FIR 中间表示程序。

    包含:
      - 名称
      - 构建器产生的所有基本块
      - 延续树
      - 序列化方法
    """

    def __init__(
        self,
        name: str = "无名程序",
        builder: FIRBuilder | None = None,
    ) -> None:
        """
        初始化 FIR 程序。

        Args:
            name:    程序名称 (中文)
            builder: 可选的 FIRBuilder (如果为 None 则创建新的)
        """
        self.name = name
        self.builder = builder or FIRBuilder()

    @property
    def blocks(self) -> list[FIRBlock]:
        """获取所有基本块"""
        return self.builder.blocks

    @property
    def values(self) -> dict[str, list[FIRValue]]:
        """获取所有 SSA 值"""
        return self.builder.values

    def to_assembly(self) -> str:
        """转换为 FLUX 汇编文本"""
        return self.builder.to_assembly()

    def dump(self) -> str:
        """输出 FIR 程序的完整文本表示"""
        header = f"═══ FIR 程序: {self.name} ═══\n"
        return header + self.builder.dump()

    def __repr__(self) -> str:
        return f"FIRProgram({self.name}, {len(self.blocks)} 块)"


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def classifier_upgrade(current: str, level: str) -> str:
    """
    量词升级 — 根据敬称级别提升量词。

    层次: 位(尊敬) > 名(中性) > 个(通用)

    Args:
        current: 当前量词
        level:   目标敬称级别

    Returns:
        升级后的量词
    """
    # 已经是最高级
    if current == "位":
        return current

    if level == "尊敬":
        return "位"
    elif level == "礼貌":
        if current == "个":
            return "名"
        return current
    elif level == "专业":
        return current
    return current


def infer_fir_type(text: str) -> tuple[FirType, str]:
    """
    从中文文本推断 FIR 类型和量词。

    扫描文本中的量词，返回对应的 FIR 类型。

    Args:
        text: 中文文本

    Returns:
        (FirType, 量词字符串)
    """
    # 先检查多字量词
    for clf in sorted(CLASSIFIER_TO_FIR_TYPE.keys(), key=len, reverse=True):
        if clf in text:
            return CLASSIFIER_TO_FIR_TYPE[clf], clf
    return FirType.INTEGER, "个"


def build_from_chinese(source: str, program_name: str = "中文程序") -> FIRProgram:
    """
    从中文源代码构建 FIR 程序。

    这是构建 FIR 程序的高级入口。

    Args:
        source:       中文源代码 (多行)
        program_name: 程序名称

    Returns:
        完整的 FIRProgram
    """
    builder = FIRBuilder()
    program = FIRProgram(name=program_name, builder=builder)

    for line in source.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("//") or line.startswith("#"):
            continue

        node = builder.parse_topic_comment(line)
        if node is None:
            # 未能解析的行 — 添加为注释
            if builder.current_block:
                instr = FIRInstruction(
                    opcode=FirOpcode.UNREACHABLE,
                    operands=[],
                    comment=f"未解析: {line}",
                )
                builder.current_block.add_instruction(instr)

    # 确保最后停机
    builder.halt()

    return program
