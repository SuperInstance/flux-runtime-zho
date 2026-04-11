"""
流星运行时 — 量子寄存器系统 (Quantum Registers R60-R63)

量子寄存器是 R60-R63 的扩展实现，支持:
  - 叠加态 (Superposition): 一个寄存器持有多个带置信度权重的值
  - 坍缩 (Collapse): 参与计算时选取最高置信度的值
  - 置信传播 (Confidence Propagation): 跨寄存器传递置信度

设计哲学:
  传统寄存器: R0 = 42 (确定值)
  量子寄存器: R60 = [(0.7, "猫"), (0.3, "狗")]
    → 70% 确定是猫, 30% 确定是狗
  → 用于计算时取最高置信度, 但保留完整分布

量子寄存器映射:
    R60 — 主题叠加寄存器 (主题可能是多个实体之一)
    R61 — 类型叠加寄存器 (类型可能是多种之一)
    R62 — 置信叠加寄存器 (存储置信度分布)
    R63 — 主量子寄存器 (全局量子状态)

用法:
    qr = QuantumRegister()
    qr.set_superposition("R60", [(0.7, "猫"), (0.3, "狗")])
    qr.set_superposition("R61", [(0.8, "AnimalType"), (0.2, "VehicleType")])
    result = qr.collapse("R60")  # → "猫" (最高置信度)
    qr.propagate("R60", "R61")     # R61继承R60的置信度
"""

from __future__ annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════
# 量子值 — 单个带置信度的值
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QuantumValue:
    """
    量子值 — 一个带置信度权重的值。

    不可变数据结构, 每次修改产生新实例。

    属性:
        confidence: 置信度 [0.0, 1.0]
        value: 实际值 (可以是任意 Python 对象)
    """
    confidence: float
    value: Any

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"置信度必须在 [0.0, 1.0] 范围内, 实际: {self.confidence}")

    def __repr__(self) -> str:
        return f"Q({self.value!r}, {self.confidence:.2f})"


# ══════════════════════════════════════════════════════════════════════
# 量子寄存器条目 — 叠加态存储
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SuperpositionState:
    """
    叠加态 — 一个寄存器的量子状态。

    存储多个可能值及其概率分布。
    总置信度自动归一化。
    """
    values: list[QuantumValue] = field(default_factory=list)
    name: str = ""
    register: int = 0

    def __post_init__(self):
        self._normalize()

    @property
    def collapsed(self) -> Any | None:
        """坍缩后的确定值 — 最高置信度的值"""
        if not self.values:
            return None
        return max(self.values, key=lambda qv: qv.confidence).value

    @property
    def collapsed_confidence(self) -> float:
        """坍缩置信度 — 最高置信度值"""
        if not self.values:
            return 0.0
        return max(qv.confidence for qv in self.values)

    @property
    def entropy(self) -> float:
        """香农熵 — 状态的不确定性度量"""
        if not self.values:
            return 0.0
        total = sum(qv.confidence for qv in self.values)
        if total == 0:
            return 0.0
        h = 0.0
        for qv in self.values:
            p = qv.confidence / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def is_deterministic(self) -> bool:
        """是否为确定态 (单一值, 置信度=1.0)"""
        return len(self.values) == 1 and self.values[0].confidence >= 1.0

    @property
    def value_count(self) -> int:
        """可能值的数量"""
        return len(self.values)

    def add(self, value: Any, confidence: float) -> None:
        """添加一个可能的值到叠加态"""
        self.values.append(QuantumValue(confidence=confidence, value=value))
        self._normalize()

    def _normalize(self) -> None:
        """归一化置信度 (总和为 1.0)"""
        total = sum(qv.confidence for qv in self.values)
        if total > 0 and total != 1.0:
            factor = 1.0 / total
            self.values = [
                QuantumValue(confidence=qv.confidence * factor, value=qv.value)
                for qv in self.values
            ]

    def filter(self, min_confidence: float) -> SuperpositionState:
        """过滤低于阈值的值, 返回新叠加态"""
        filtered = [qv for qv in self.values if qv.confidence >= min_confidence]
        new_state = SuperpositionState(
            name=self.name,
            register=self.register,
        )
        new_state.values = filtered
        if new_state.values:
            new_state._normalize()
        return new_state

    def merge(self, other: SuperpositionState) -> SuperpositionState:
        """合并两个叠加态 (贝叶斯更新)"""
        merged = SuperpositionState(
            name=self.name,
            register=self.register,
        )
        # 合并相同值
        value_map: dict[Any, float] = {}
        for qv in self.values:
            value_map[id(qv.value)] = value_map.get(id(qv.value), 0) + qv.confidence
        for qv in other.values:
            value_map[id(qv.value)] = value_map.get(id(qv.value), 0) + qv.confidence
        # 创建合并后的值
        seen_ids: set[int] = set()
        for qv in self.values:
            vid = id(qv.value)
            if vid not in seen_ids:
                merged.add(qv.value, value_map[vid])
                seen_ids.add(vid)
        for qv in other.values:
            vid = id(qv.value)
            if vid not in seen_ids:
                merged.add(qv.value, value_map[vid])
                seen_ids.add(vid)
        return merged

    def __repr__(self) -> str:
        state_str = ", ".join(repr(qv) for qv in self.values[:5])
        if len(self.values) > 5:
            state_str += f", ... ({len(self.values)} values)"
        reg = f" R{self.register}" if self.register >= 0 else ""
        name = f" [{self.name}]" if self.name else ""
        return f"Superposition{reg}{name}: {state_str}"


# ══════════════════════════════════════════════════════════════════════
# 量子寄存器系统
# ══════════════════════════════════════════════════════════════════════

# 特殊量子寄存器编号
QR_TOPIC = 60       # R60 — 主题叠加
QR_TYPE = 61        # R61 — 类型叠加
QR_CONFIDENCE = 62    # R62 — 置信分布
QR_MAIN = 63         # R63 — 主量子寄存器


class QuantumRegisterSystem:
    """
    量子寄存器系统 — 管理 R60-R63 的量子叠加态。

    特殊寄存器用途:
        R60 = 主题叠加: 当前主题可能是多个实体之一
        R61 = 类型叠加: 当前值的类型可能是多种之一
        R62 = 置信叠加: 存储跨寄存器的置信度分布
        R63 = 主量子: 全局量子状态 (确定性值的叠加态)

    用法:
        qrs = QuantumRegisterSystem()
        qrs.set("R60", [(0.7, "张三"), (0.3, "李四")])
        qrs.set("R61", [(0.8, "PersonType"), (0.2, "AnimalType")])
        qrs.collapse("R60")   # → "张三"
        qrs.propagate("R60", "R61")  # R61继承R60的置信度分布
    """

    def __init__(self):
        # 64 个寄存器, 默认为确定性态
        self._registers: dict[int, SuperpositionState] = {}
        # 初始化确定性态
        for i in range(64):
            self._registers[i] = SuperpositionState(register=i)

        # 特殊量子寄存器初始化
        for qr in (QR_TOPIC, QR_TYPE, QR_CONFIDENCE, QR_MAIN):
            self._registers[qr] = SuperpositionState(register=qr)

        self._collapse_log: list[tuple[int, Any, float]] = []
        self._propagation_log: list[tuple[int, int, float]] = []

    def set(self, register: str | int, values: list[tuple[float, Any]],
            name: str = "") -> SuperpositionState:
        """
        设置寄存器的叠加态。

        Args:
            register: 寄存器名 (如 "R60") 或编号
            values: [(confidence, value), ...] 列表
            name: 寄存器描述名

        Returns:
            创建的叠加态
        """
        reg_num = self._parse_register(register)
        state = SuperpositionState(name=name, register=reg_num)
        for confidence, value in values:
            state.add(value, confidence)
        self._registers[reg_num] = state
        return state

    def set_deterministic(self, register: str | int, value: Any,
                           name: str = "") -> None:
        """
        设置寄存器为确定态 (单一值, 置信度=1.0)。

        Args:
            register: 寄存存器名或编号
            value: 确定值
            name: 描述名
        """
        reg_num = self._parse_register(register)
        state = SuperpositionState(name=name, register=reg_num)
        state.add(value, 1.0)
        self._registers[reg_num] = state

    def get(self, register: str | int) -> SuperpositionState:
        """获取寄存器的叠加态"""
        reg_num = self._parse_register(register)
        return self._registers[reg_num]

    def collapse(self, register: str | int) -> Any | None:
        """
        坍缩寄存器 — 选取最高置信度的值。

        注意: 坍缩不会修改叠加态 (无观测效应)。
        如需修改, 使用 collapse_and_update()。

        Returns:
            坍缩后的确定值
        """
        reg_num = self._parse_register(register)
        state = self._registers[reg_num]
        collapsed = state.collapsed
        if collapsed is not None:
            self._collapse_log.append((reg_num, collapsed, state.collapsed_confidence))
        return collapsed

    def collapse_and_update(self, register: str | int) -> Any | None:
        """
        坍缩并更新 — 坍缩后只保留最高置信度的值 (观测效应)。

        Returns:
            坍缩后的确定值
        """
        reg_num = self._parse_register(register)
        state = self._registers[reg_num]
        collapsed = state.collapsed
        if collapsed is not None:
            self._registers[reg_num] = SuperpositionState(
                name=state.name,
                register=reg_num,
            )
            self._registers[reg_num].add(collapsed, 1.0)
            self._collapse_log.append((reg_num, collapsed, 1.0))
        return collapsed

    def propagate(self, source: str | int, target: str | int) -> float:
        """
        置信传播 — 将源寄存器的置信度分布传播到目标寄存器。

        如果目标寄存器已有值, 执行贝叶斯合并 (信任更新)。

        Args:
            source: 源寄存器
            target: 目标寄存器

        Returns:
            合并后的总置信度
        """
        src_num = self._parse_register(source)
        tgt_num = self._parse_register(target)
        src_state = self._registers[src_num]
        tgt_state = self._registers[tgt_num]

        merged = tgt_state.merge(src_state)
        self._registers[tgt_num] = merged

        total_conf = sum(qv.confidence for qv in merged.values)
        self._propagation_log.append((src_num, tgt_num, total_conf))
        return total_conf

    def bayesian_update(self, register: str | int,
                        evidence_value: Any,
                        evidence_confidence: float) -> None:
        """
        贝叶斯更新 — 根据证据更新寄存器的置信度分布。

        如果 evidence_value 已在叠加态中, 提升其置信度。
        否则, 添加新值到叠加态。

        Args:
            register: 寄存器
            evidence_value: 证据值
            evidence_confidence: 证据置信度
        """
        reg_num = self._parse_register(register)
        state = self._registers[reg_num]

        # 检查是否已有相同值
        found = False
        for qv in state.values:
            if qv.value == evidence_value:
                # 贝叶斯更新: P(A|B) = P(B|A)*P(A)/P(B)
                new_conf = min(1.0, qv.confidence + evidence_confidence * (1.0 - qv.confidence))
                state.add(evidence_value, new_confidence)
                found = True
                break

        if not found:
            state.add(evidence_value, evidence_confidence)

    def filter_low_confidence(self, register: str | int,
                              threshold: float = 0.1) -> SuperpositionState:
        """
        过滤低置信度值, 返回过滤后的新叠加态。

        Args:
            register: 寄存器
            threshold: 最低置信度阈值

        Returns:
            过滤后的叠加态
        """
        reg_num = self._parse_register(register)
        old_state = self._registers[reg_num]
        new_state = old_state.filter(threshold)
        self._registers[reg_num] = new_state
        return new_state

    def entropy(self, register: str | int) -> float:
        """计算寄存器的香农熵 (不确定性)"""
        reg_num = self._parse_register(register)
        return self._registers[reg_num].entropy

    def is_deterministic(self, register: str | int) -> bool:
        """检查寄存器是否为确定态"""
        reg_num = self._parse_register(register)
        return self._registers[reg_num].is_deterministic

    def collapse_log(self) -> list[tuple[int, Any, float]]:
        """获取所有坍缩记录"""
        return list(self._collapse_log)

    def propagation_log(self) -> list[tuple[int, int, float]]:
        """获取所有传播记录"""
        return list(self._propagation_log)

    def all_states(self) -> dict[int, SuperpositionState]:
        """获取所有寄存器的状态"""
        return {reg: state for reg, state in self._registers.items()}

    def summary(self) -> str:
        """生成寄存器状态摘要"""
        lines = ["量子寄存器系统状态:", "=" * 40]
        for reg_num in sorted(self._registers.keys()):
            state = self._registers[reg_num]
            if reg_num < 60 or state.is_deterministic:
                if state.collapsed is not None:
                    lines.append(f"  R{reg_num:02d}: {state.collapsed!r} (确定态)")
            else:
                collapsed = state.collapsed
                conf = state.collapsed_confidence
                n = state.value_count
                lines.append(
                    f"  R{reg_num:02d}: {collapsed!r} ({conf:.0%}, "
                    f"{n} values, H={state.entropy:.2f} bits"
                )
        return "\n".join(lines)

    def _parse_register(self, register: str | int) -> int:
        """解析寄存器名"""
        if isinstance(register, int):
            return register
        s = register.strip().upper()
        if s.startswith("R"):
            try:
                num = int(s[1:])
                if 0 <= num < 64:
                    return num
            except ValueError:
                pass
        return 0

    def reset(self) -> None:
        """重置所有寄存器为确定态"""
        for i in range(64):
            self._registers[i] = SuperpositionState(register=i)
        self._collapse_log.clear()
        self._propagation_log.clear()

    def __repr__(self) -> str:
        quantum_count = sum(
            1 for reg, state in self._registers.items()
            if not state.is_deterministic and reg >= 60
        )
        return (
            f"QuantumRegisterSystem("
            f"quantum_regs={quantum_count}/4, "
            f"collapse_ops={len(self._collapse_log)}, "
            f"propagations={len(self._propagation_log)})"
        )
