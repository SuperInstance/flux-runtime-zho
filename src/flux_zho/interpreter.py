"""
流星解释器 — 中文自然语言 → FLUX字节码 → VM执行

核心架构:
  1. FLUX Opcodes — 与主运行时完全一致的指令集（语言无关层）
  2. FLUX VM — 64寄存器栈式虚拟机（语言无关层）
  3. FLUX Encoder — 汇编文本 → 字节数组（语言无关层）
  4. 中文NL解释器 — 中文模式匹配 → 汇编（中文特定层）
  5. 量词系统 — 量词验证与类型推断（中文特定层）
  6. 主题追踪 — 主题寄存器R63实现零形回指（中文特定层）
"""

import re
import struct
from typing import Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 第一层：FLUX 操作码 — 语言无关（与主运行时一致）
# ═══════════════════════════════════════════════════════════

class Op:
    """FLUX 字节码操作码集 — 变长编码(1-8字节)"""
    # 控制流 (0x00-0x07)
    NOP = 0x00
    MOV = 0x01
    LOAD = 0x02
    STORE = 0x03
    JMP = 0x04
    JZ = 0x05
    JNZ = 0x06
    CALL = 0x07

    # 整数算术 (0x08-0x0F)
    IADD = 0x08
    ISUB = 0x09
    IMUL = 0x0A
    IDIV = 0x0B
    IMOD = 0x0C
    INEG = 0x0D
    INC = 0x0E
    DEC = 0x0F

    # 位运算 (0x10-0x17)
    IAND = 0x10
    IOR = 0x11
    IXOR = 0x12
    INOT = 0x13
    ISHL = 0x14
    ISHR = 0x15
    ROTL = 0x16
    ROTR = 0x17

    # 比较 (0x18-0x1F)
    ICMP = 0x18
    IEQ = 0x19
    ILT = 0x1A
    ILE = 0x1B
    IGT = 0x1C
    IGE = 0x1D
    TEST = 0x1E
    SETCC = 0x1F

    # 栈操作 (0x20-0x27)
    PUSH = 0x20
    POP = 0x21
    DUP = 0x22
    SWAP = 0x23
    ROT = 0x24
    ENTER = 0x25
    LEAVE = 0x26
    ALLOCA = 0x27

    # 函数操作 (0x28-0x2F)
    RET = 0x28
    CALL_IND = 0x29
    TAILCALL = 0x2A
    MOVI = 0x2B
    IREM = 0x2C
    CMP = 0x2D
    JE = 0x2E
    JNE = 0x2F

    # 内存管理 (0x30-0x37)
    REGION_CREATE = 0x30
    REGION_DESTROY = 0x31
    REGION_TRANSFER = 0x32
    MEMCOPY = 0x33
    MEMSET = 0x34
    MEMCMP = 0x35
    JL = 0x36
    JGE = 0x37

    # 类型操作 (0x38-0x3F)
    CAST = 0x38
    BOX = 0x39
    UNBOX = 0x3A
    CHECK_TYPE = 0x3B
    CHECK_BOUNDS = 0x3C

    # 浮点算术 (0x40-0x47)
    FADD = 0x40
    FSUB = 0x41
    FMUL = 0x42
    FDIV = 0x43
    FNEG = 0x44
    FABS = 0x45
    FMIN = 0x46
    FMAX = 0x47

    # 浮点比较 (0x48-0x4F)
    FEQ = 0x48
    FLT = 0x49
    FLE = 0x4A
    FGT = 0x4B
    FGE = 0x4C

    # 字符串操作 (0x50-0x57)
    SLEN = 0x50
    SCONCAT = 0x51
    SCHAR = 0x52
    SSUB = 0x53
    SCMP = 0x54

    # 智能体协议 (0x60-0x7F)
    TELL = 0x60
    ASK = 0x61
    DELEGATE = 0x62
    BROADCAST = 0x63
    TRUST_CHECK = 0x64
    CAPABILITY_REQ = 0x65

    # 系统 (0xFE-0xFF)
    PRINT = 0xFE
    HALT = 0xFF


# 操作码反查表
OP_NAMES: dict[int, str] = {}
for _name, _code in list(vars(Op).items()):
    if isinstance(_code, int) and not _name.startswith("_") and _name.isupper():
        OP_NAMES[_code] = _name


# ═══════════════════════════════════════════════════════════
# 第二层：FLUX 字节码编码器 — 汇编文本 → 字节数组
# ═══════════════════════════════════════════════════════════

@dataclass
class EncodedInstruction:
    offset: int
    opcode: int
    operands: list
    size: int
    mnemonic: str


def encode_assembly(assembly: str) -> tuple[bytearray, list[EncodedInstruction], dict[str, int]]:
    """
    将汇编文本编码为字节数组。

    支持: 寄存器操作(IADD R0, R1, R2)、立即数操作(MOVI R0, 42)、
    跳转操作(JZ R0, label / JMP label)。
    """
    lines = assembly.split("\n")
    labels: dict[str, int] = {}
    raw_lines: list[tuple[str, int]] = []
    instructions: list[EncodedInstruction] = []

    # 第一遍：收集标签（使用字节偏移量）
    # 先做一次预编码来计算字节偏移
    _opcode_map_pre: dict[str, int] = {}
    for _n, _c in vars(Op).items():
        if isinstance(_c, int) and not _n.startswith("_") and _n.isupper():
            _opcode_map_pre[_n] = _c

    def _pre_encode_line_size(mnemonic: str, args: list[str]) -> int:
        """估算指令字节数（用于标签偏移计算）"""
        op = _opcode_map_pre.get(mnemonic.upper(), 0)
        if op in (Op.NOP, Op.HALT, Op.RET, Op.LEAVE): return 1
        if op in (Op.INC, Op.DEC, Op.INEG, Op.INOT, Op.PRINT): return 2
        if op in (Op.MOV, Op.LOAD, Op.STORE, Op.CMP, Op.ICMP,
                  Op.FEQ, Op.FLT, Op.FLE, Op.FGT, Op.FGE,
                  Op.SLEN, Op.SCHAR, Op.SSUB, Op.SCMP,
                  Op.CAST, Op.BOX, Op.UNBOX, Op.CHECK_TYPE, Op.CHECK_BOUNDS,
                  Op.FNEG, Op.FABS, Op.FMIN, Op.FMAX,
                  Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
                  Op.TRUST_CHECK, Op.CAPABILITY_REQ,
                  Op.PUSH, Op.POP, Op.REGION_CREATE, Op.REGION_DESTROY): return 3
        if op in (Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD,
                  Op.IAND, Op.IOR, Op.IXOR, Op.ISHL, Op.ISHR,
                  Op.ROTL, Op.ROTR, Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV,
                  Op.SCONCAT): return 4
        if op == Op.MOVI: return 4
        if op in (Op.JZ, Op.JNZ, Op.JE, Op.JNE, Op.JL, Op.JGE): return 4
        if op == Op.JMP: return 3
        if op in (Op.DUP, Op.SWAP, Op.ENTER, Op.ALLOCA): return 2
        if op in (Op.MEMCOPY, Op.MEMSET, Op.MEMCMP): return 4
        return 1  # 默认1字节

    idx = 0  # 字节偏移量
    for i, line in enumerate(lines):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("--") or trimmed.startswith("//"):
            continue
        if trimmed.endswith(":"):
            labels[trimmed[:-1].strip()] = idx
            continue
        raw_lines.append((trimmed, i))
        parts = re.split(r"[\s,]+", trimmed.strip())
        parts = [p for p in parts if p]
        args = [a.strip() for a in ",".join(parts[1:]).split(",") if a.strip()]
        idx += _pre_encode_line_size(parts[0], args)

    # 第二遍：编码
    buf = bytearray()

    # 建立操作码映射
    opcode_map: dict[str, int] = {}
    for name, code in vars(Op).items():
        if isinstance(code, int) and not name.startswith("_") and name.isupper():
            opcode_map[name] = code

    for line_text, _line_num in raw_lines:
        offset = len(buf)
        parts = re.split(r"[\s,]+", line_text.strip())
        parts = [p for p in parts if p]
        mnemonic = parts[0].upper()
        args = [a.strip() for a in ",".join(parts[1:]).split(",") if a.strip()]

        op = opcode_map.get(mnemonic, Op.NOP)
        encoded = _encode_line(op, mnemonic, args, labels)
        buf.extend(encoded["bytes"])
        instructions.append(EncodedInstruction(
            offset=offset,
            opcode=encoded["bytes"][0] if encoded["bytes"] else 0,
            operands=encoded["operands"],
            size=len(encoded["bytes"]),
            mnemonic=mnemonic,
        ))

    return buf, instructions, labels


def _parse_reg(s: str) -> int:
    m = re.match(r"^R(\d+)$", s.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"无效寄存器: {s}")
    return int(m.group(1))


def _parse_imm(s: str) -> int:
    try:
        return int(s.strip())
    except ValueError:
        raise ValueError(f"无效立即数: {s}")


def _encode_u16(val: int) -> bytes:
    return bytes([val & 0xFF, (val >> 8) & 0xFF])


def _encode_line(op: int, mnemonic: str, args: list[str], labels: dict[str, int]) -> dict:
    """编码单条汇编指令"""
    # 零操作数指令
    if op in (Op.NOP, Op.HALT, Op.RET, Op.LEAVE):
        return {"bytes": bytes([op]), "operands": []}

    # 单寄存器指令
    if op in (Op.INC, Op.DEC, Op.INEG, Op.INOT, Op.PRINT):
        r = _parse_reg(args[0]) if args else 0
        return {"bytes": bytes([op, r]), "operands": [r]}

    # 双寄存器指令
    if op in (Op.MOV, Op.LOAD, Op.STORE):
        rd = _parse_reg(args[0])
        rs = _parse_reg(args[1])
        return {"bytes": bytes([op, rd, rs]), "operands": [rd, rs]}

    # 三寄存器指令
    three_reg_ops = {
        Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD,
        Op.IAND, Op.IOR, Op.IXOR, Op.ISHL, Op.ISHR,
        Op.ROTL, Op.ROTR, Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV,
        Op.SCONCAT,
    }
    if op in three_reg_ops:
        rd = _parse_reg(args[0])
        ra = _parse_reg(args[1])
        rb = _parse_reg(args[2]) if len(args) > 2 else 0
        return {"bytes": bytes([op, rd, ra, rb]), "operands": [rd, ra, rb]}

    # 立即数加载
    if op == Op.MOVI:
        r = _parse_reg(args[0])
        imm = _parse_imm(args[1])
        return {"bytes": bytes([op, r]) + _encode_u16(imm), "operands": [r, imm]}

    # 比较
    cmp_ops = {Op.CMP, Op.ICMP, Op.FEQ, Op.FLT, Op.FLE, Op.FGT, Op.FGE,
               Op.SLEN, Op.SCHAR, Op.SSUB, Op.SCMP}
    if op in cmp_ops:
        ra = _parse_reg(args[0])
        rb = _parse_reg(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([op, ra, rb]), "operands": [ra, rb]}

    # 条件跳转
    cond_jumps = {Op.JZ, Op.JNZ, Op.JE, Op.JNE, Op.JL, Op.JGE}
    if op in cond_jumps:
        r = _parse_reg(args[0])
        target = args[1] if len(args) > 1 else "0"
        addr = labels.get(target, _parse_imm(target) if target.lstrip("-").isdigit() else 0)
        return {"bytes": bytes([op, r]) + _encode_u16(addr), "operands": [r, addr]}

    # 无条件跳转
    if op == Op.JMP:
        target = args[0] if args else "0"
        addr = labels.get(target, _parse_imm(target) if target.lstrip("-").isdigit() else 0)
        return {"bytes": bytes([op]) + _encode_u16(addr), "operands": [addr]}

    # 栈操作
    if op in (Op.PUSH, Op.POP):
        r = _parse_reg(args[0]) if args else 0
        return {"bytes": bytes([op, r]), "operands": [r]}

    if op in (Op.DUP, Op.SWAP, Op.ENTER, Op.ALLOCA):
        return {"bytes": bytes([op, 0]), "operands": [0]}

    # 内存操作
    if op in (Op.REGION_CREATE, Op.REGION_DESTROY):
        r = _parse_reg(args[0]) if args else 0
        return {"bytes": bytes([op, r]), "operands": [r]}

    if op in (Op.MEMCOPY, Op.MEMSET, Op.MEMCMP):
        a = _parse_reg(args[0])
        b = _parse_reg(args[1])
        c = _parse_reg(args[2]) if len(args) > 2 else 0
        return {"bytes": bytes([op, a, b, c]), "operands": [a, b, c]}

    # 类型操作
    type_ops = {Op.CAST, Op.BOX, Op.UNBOX, Op.CHECK_TYPE, Op.CHECK_BOUNDS,
                Op.FNEG, Op.FABS, Op.FMIN, Op.FMAX}
    if op in type_ops:
        a = _parse_reg(args[0])
        b = _parse_reg(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([op, a, b]), "operands": [a, b]}

    # 智能体协议
    a2a_ops = {Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
               Op.TRUST_CHECK, Op.CAPABILITY_REQ}
    if op in a2a_ops:
        a = _parse_reg(args[0])
        b = _parse_reg(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([op, a, b]), "operands": [a, b]}

    return {"bytes": bytes([op]), "operands": []}


def quick_encode(assembly: str) -> bytearray:
    """快速编码：汇编字符串 → 字节数组"""
    buf, _, _ = encode_assembly(assembly)
    return buf


# ═══════════════════════════════════════════════════════════
# 第三层：FLUX 字节码反汇编器 — 字节数组 → 可读汇编
# ═══════════════════════════════════════════════════════════

@dataclass
class DecodedInstruction:
    offset: int
    mnemonic: str
    operands: str
    size: int
    opcode: int


def disassemble(bytecode: bytes) -> list[DecodedInstruction]:
    """将字节数组反汇编为可读指令列表"""
    instructions = []
    offset = 0
    data = bytes(bytecode)

    while offset < len(data):
        start = offset
        op = data[offset]
        name = OP_NAMES.get(op, f"UNKNOWN_0x{op:02x}")
        offset += 1
        operands = []

        if op in (Op.NOP, Op.HALT, Op.RET, Op.LEAVE):
            pass
        elif op in (Op.INC, Op.DEC, Op.INEG, Op.INOT, Op.PRINT):
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op in (Op.MOV, Op.LOAD, Op.STORE):
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op in (Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD,
                     Op.IAND, Op.IOR, Op.IXOR, Op.ISHL, Op.ISHR,
                     Op.ROTL, Op.ROTR, Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV,
                     Op.SCONCAT):
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op == Op.MOVI:
            operands.append(f"R{data[offset]}")
            offset += 1
            imm = data[offset] | (data[offset + 1] << 8)
            offset += 2
            operands.append(str(imm if imm < 32768 else imm - 65536))
        elif op in (Op.CMP, Op.ICMP, Op.FEQ, Op.FLT, Op.FLE, Op.FGT, Op.FGE,
                     Op.SLEN, Op.SCHAR, Op.SSUB, Op.SCMP):
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op in (Op.JZ, Op.JNZ, Op.JE, Op.JNE, Op.JL, Op.JGE):
            operands.append(f"R{data[offset]}")
            offset += 1
            addr = data[offset] | (data[offset + 1] << 8)
            offset += 2
            operands.append(str(addr))
        elif op == Op.JMP:
            addr = data[offset] | (data[offset + 1] << 8)
            offset += 2
            operands.append(str(addr))
        elif op in (Op.PUSH, Op.POP, Op.REGION_CREATE, Op.REGION_DESTROY):
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op in (Op.DUP, Op.SWAP, Op.ENTER, Op.ALLOCA):
            offset += 1
        elif op in (Op.CAST, Op.BOX, Op.UNBOX, Op.CHECK_TYPE, Op.CHECK_BOUNDS,
                     Op.FNEG, Op.FABS, Op.FMIN, Op.FMAX,
                     Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
                     Op.TRUST_CHECK, Op.CAPABILITY_REQ):
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
        elif op in (Op.MEMCOPY, Op.MEMSET, Op.MEMCMP):
            operands.append(f"R{data[offset]}")
            offset += 1
            operands.append(f"R{data[offset]}")
            offset += 1
            offset += 1
        else:
            pass  # 未知指令，跳过

        instructions.append(DecodedInstruction(
            offset=start,
            mnemonic=name,
            operands=", ".join(operands),
            size=offset - start,
            opcode=op,
        ))

    return instructions


def format_assembly(instructions: list[DecodedInstruction]) -> str:
    """格式化反汇编输出"""
    return "\n".join(
        f"  {i.offset:04d}:  {i.mnemonic:<12s} {i.operands}"
        for i in instructions
    )


# ═══════════════════════════════════════════════════════════
# 第四层：FLUX 虚拟机 — 64寄存器栈式执行引擎
# ═══════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """VM执行结果"""
    success: bool
    result: int
    registers: list[int]
    cycles: int
    disassembly: str
    error: str | None
    halted: bool
    trace: list[str] = field(default_factory=list)


class FluxVM:
    """
    FLUX 虚拟机 — 64寄存器、栈式执行引擎。

    所有语言编译到这个通用字节码引擎。
    R63 用作主题寄存器（中文特定约定）。
    """

    def __init__(self, bytecode: bytes, *, max_cycles: int = 1_000_000,
                 trace: bool = False):
        self.regs = [0] * 64
        self.flags = {"zero": False, "negative": False, "carry": False, "overflow": False}
        self.stack: list[int] = []
        self.pc = 0
        self.bytecode = bytes(bytecode)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.halted = False
        self.error: str | None = None
        self.trace_enabled = trace
        self.trace_log: list[str] = []

    def read_reg(self, r: int) -> int:
        return self.regs[r] if 0 <= r < 64 else 0

    def write_reg(self, r: int, val: int) -> None:
        if 0 <= r < 64:
            self.regs[r] = val & 0xFFFFFFFF if val >= 0 else (val & 0xFFFFFFFF)

    def _push(self, val: int) -> None:
        self.stack.append(val)

    def _pop(self) -> int:
        return self.stack.pop() if self.stack else 0

    def _update_flags(self, result: int) -> None:
        self.flags["zero"] = result == 0
        self.flags["negative"] = result < 0

    def _read_u16(self, offset: int) -> int:
        return self.bytecode[offset] | (self.bytecode[offset + 1] << 8)

    def _log(self, msg: str) -> None:
        if self.trace_enabled:
            self.trace_log.append(msg)

    def execute(self) -> ExecutionResult:
        """执行字节码直到停机或达到最大周期数"""
        disasm = disassemble(self.bytecode)
        disasm_str = format_assembly(disasm)

        while self.pc < len(self.bytecode) and not self.halted and self.cycles < self.max_cycles:
            op = self.bytecode[self.pc]
            self.cycles += 1

            if op == Op.NOP:
                self.pc += 1

            elif op == Op.MOV:
                rd, rs = self.bytecode[self.pc + 1], self.bytecode[self.pc + 2]
                self.write_reg(rd, self.read_reg(rs))
                self.pc += 3

            elif op == Op.LOAD:
                rd, rs = self.bytecode[self.pc + 1], self.bytecode[self.pc + 2]
                self.write_reg(rd, self.read_reg(rs))
                self.pc += 3

            elif op == Op.STORE:
                rd, rs = self.bytecode[self.pc + 1], self.bytecode[self.pc + 2]
                self.write_reg(rs, self.read_reg(rd))
                self.pc += 3

            elif op == Op.MOVI:
                r = self.bytecode[self.pc + 1]
                imm = self._read_u16(self.pc + 2)
                val = imm if imm < 32768 else imm - 65536
                self.write_reg(r, val)
                self._update_flags(val)
                self.pc += 4

            elif op == Op.IADD:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                result = self.read_reg(ra) + self.read_reg(rb)
                self.write_reg(rd, result)
                self._update_flags(result)
                self.pc += 4

            elif op == Op.ISUB:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                result = self.read_reg(ra) - self.read_reg(rb)
                self.write_reg(rd, result)
                self._update_flags(result)
                self.pc += 4

            elif op == Op.IMUL:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                result = self.read_reg(ra) * self.read_reg(rb)
                self.write_reg(rd, result)
                self._update_flags(result)
                self.pc += 4

            elif op == Op.IDIV:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                divisor = self.read_reg(rb)
                if divisor == 0:
                    self.error = "除零错误"
                    self.halted = True
                else:
                    result = int(self.read_reg(ra) / divisor)
                    self.write_reg(rd, result)
                    self._update_flags(result)
                self.pc += 4

            elif op == Op.IMOD:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                divisor = self.read_reg(rb)
                if divisor == 0:
                    self.error = "模零错误"
                    self.halted = True
                else:
                    result = self.read_reg(ra) % divisor
                    self.write_reg(rd, result)
                    self._update_flags(result)
                self.pc += 4

            elif op == Op.INEG:
                r = self.bytecode[self.pc + 1]
                result = -self.read_reg(r)
                self.write_reg(r, result)
                self._update_flags(result)
                self.pc += 2

            elif op == Op.INC:
                r = self.bytecode[self.pc + 1]
                result = self.read_reg(r) + 1
                self.write_reg(r, result)
                self._update_flags(result)
                self.pc += 2

            elif op == Op.DEC:
                r = self.bytecode[self.pc + 1]
                result = self.read_reg(r) - 1
                self.write_reg(r, result)
                self._update_flags(result)
                self.pc += 2

            elif op == Op.CMP:
                ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2]
                diff = self.read_reg(ra) - self.read_reg(rb)
                self._update_flags(diff)
                self.pc += 3

            elif op == Op.ICMP:
                ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2]
                a, b = self.read_reg(ra), self.read_reg(rb)
                self.flags["zero"] = a == b
                self.flags["negative"] = a < b
                self.pc += 3

            elif op == Op.IAND:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                self.write_reg(rd, self.read_reg(ra) & self.read_reg(rb))
                self.pc += 4

            elif op == Op.IOR:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                self.write_reg(rd, self.read_reg(ra) | self.read_reg(rb))
                self.pc += 4

            elif op == Op.IXOR:
                rd, ra, rb = self.bytecode[self.pc+1], self.bytecode[self.pc+2], self.bytecode[self.pc+3]
                self.write_reg(rd, self.read_reg(ra) ^ self.read_reg(rb))
                self.pc += 4

            elif op == Op.INOT:
                r = self.bytecode[self.pc + 1]
                self.write_reg(r, ~self.read_reg(r))
                self.pc += 2

            elif op == Op.JMP:
                addr = self._read_u16(self.pc + 1)
                self.pc = addr

            elif op == Op.JZ:
                r = self.bytecode[self.pc + 1]
                addr = self._read_u16(self.pc + 2)
                if self.read_reg(r) == 0:
                    self.pc = addr
                else:
                    self.pc += 4

            elif op == Op.JNZ:
                r = self.bytecode[self.pc + 1]
                addr = self._read_u16(self.pc + 2)
                if self.read_reg(r) != 0:
                    self.pc = addr
                else:
                    self.pc += 4

            elif op == Op.JE:
                addr = self._read_u16(self.pc + 2)
                self.pc = addr if self.flags["zero"] else self.pc + 4

            elif op == Op.JNE:
                addr = self._read_u16(self.pc + 2)
                self.pc = addr if not self.flags["zero"] else self.pc + 4

            elif op == Op.JL:
                addr = self._read_u16(self.pc + 2)
                self.pc = addr if self.flags["negative"] else self.pc + 4

            elif op == Op.JGE:
                addr = self._read_u16(self.pc + 2)
                self.pc = addr if not self.flags["negative"] else self.pc + 4

            elif op == Op.PUSH:
                r = self.bytecode[self.pc + 1]
                self._push(self.read_reg(r))
                self.pc += 2

            elif op == Op.POP:
                r = self.bytecode[self.pc + 1]
                self.write_reg(r, self._pop())
                self.pc += 2

            elif op == Op.DUP:
                if self.stack:
                    self._push(self.stack[-1])
                self.pc += 2

            elif op == Op.SWAP:
                if len(self.stack) >= 2:
                    self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
                self.pc += 2

            elif op == Op.RET:
                self.halted = True
                self.pc += 1

            elif op == Op.CALL:
                self.halted = True
                self.pc += 1

            elif op == Op.PRINT:
                r = self.bytecode[self.pc + 1]
                self._log(f"PRINT R{r} = {self.read_reg(r)}")
                self.pc += 2

            # 智能体协议 (A2A) — 3字节指令
            elif op in (Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
                        Op.TRUST_CHECK, Op.CAPABILITY_REQ):
                a = self.bytecode[self.pc + 1]
                b = self.bytecode[self.pc + 2]
                self._log(f"A2A op={op:#x} R{a}, R{b}")
                self.pc += 3

            elif op == Op.HALT:
                self.halted = True
                self.pc += 1

            else:
                self.pc += 1  # 未知指令，跳过

        if self.cycles >= self.max_cycles:
            self.error = "超出最大周期数"

        return ExecutionResult(
            success=self.error is None,
            result=self.regs[0],
            registers=list(self.regs),
            cycles=self.cycles,
            disassembly=disasm_str,
            error=self.error,
            halted=self.halted,
            trace=self.trace_log,
        )


def quick_exec(assembly: str, trace: bool = False) -> ExecutionResult:
    """快速执行：汇编字符串 → 执行结果"""
    bytecode = quick_encode(assembly)
    vm = FluxVM(bytecode, trace=trace)
    return vm.execute()


# ═══════════════════════════════════════════════════════════
# 第五层：中文数字系统 — 中文数字 ↔ 阿拉伯数字
# ═══════════════════════════════════════════════════════════

# 中文数字映射
ZH_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}

# 寄存器中文名
ZH_REGISTERS = {
    "寄存器零": 0, "寄存器一": 1, "寄存器二": 2, "寄存器三": 3,
    "寄存器四": 4, "寄存器五": 5, "寄存器六": 6, "寄存器七": 7,
    "寄存器八": 8, "寄存器九": 9, "寄存器十": 10,
    "甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4,
    "己": 5, "庚": 6, "辛": 7, "壬": 8, "癸": 9,
}

def parse_zh_number(text: str) -> int:
    """
    解析中文数字为整数。
    支持: "三", "十", "十五", "三百二十七", "四十二", "二十", "一千"等。
    也支持直接阿拉伯数字和全角数字。

    算法:
      用 temp 保存最近一个数字，遇到单位时 temp × 单位值 累加到 current。
      遇到万时，current × 10000 累加到 result 并重置。
    """
    text = text.strip()

    # 直接阿拉伯数字
    if re.match(r'^-?\d+$', text):
        return int(text)

    # 全角数字映射
    ZH_FULLWIDTH = "０１２３４５６７８９"

    result = 0      # 最终结果
    current = 0     # 当前区段累积值
    temp = 0        # 最近一个数字（待乘以单位）

    for char in text:
        if char in ZH_FULLWIDTH:
            temp = ZH_FULLWIDTH.index(char)
            current += temp
            continue

        if char not in ZH_DIGITS:
            continue

        val = ZH_DIGITS[char]

        if val < 10:
            # 数字0-9：仅记录到temp，不立即累加
            temp = val
        elif val == 10:    # 十
            multiplier = temp if temp > 0 else 1
            current += multiplier * 10
            temp = 0
        elif val == 100:   # 百
            multiplier = temp if temp > 0 else 1
            current += multiplier * 100
            temp = 0
        elif val == 1000:  # 千
            multiplier = temp if temp > 0 else 1
            current += multiplier * 1000
            temp = 0
        elif val == 10000: # 万
            result += (current + temp) * 10000
            current = 0
            temp = 0

    result += current + temp
    return result


def resolve_value(text: str, topic_reg_val: int = 0) -> int | str:
    """
    解析值：支持中文数字、阿拉伯数字、寄存器名和零形回指。
    零形回指：空字符串或"它"表示使用主题寄存器的值。
    """
    text = text.strip()
    if not text or text == "它" or text == "其":
        return topic_reg_val
    if text in ZH_REGISTERS:
        return f"R{ZH_REGISTERS[text]}"
    try:
        return parse_zh_number(text)
    except (ValueError, TypeError):
        return text  # 返回原始字符串（可能是变量名）


# ═══════════════════════════════════════════════════════════
# 第六层：量词系统 — 量词验证与类型推断
# ═══════════════════════════════════════════════════════════

# 量词 → 类型类别映射
CLASSIFIER_TYPES = {
    # 通用量词
    "个": "通用",
    "只": "动物/小件",
    "条": "长形/河流/消息",
    "本": "书籍/文件",
    "台": "机器/设备",
    "位": "人(敬称)",
    "名": "人",
    "件": "事务/衣物",
    "种": "种类",
    "套": "套装",
    # 数字量词
    "次": "次数",
    "遍": "遍数",
    "趟": "趟数",
    "步": "步骤",
    # 航海量词
    "艘": "船只",
    "节": "航速(节)",
    "海里": "距离",
    "锚": "锚",
    "帆": "帆",
    # 信息量词
    "条信息": "消息",
    "封": "信件",
    "份": "文件/报告",
    "项": "条款",
}

# 量词 → 推荐的名词上下文
CLASSIFIER_NOUN_HINTS = {
    "艘": ["船", "舰", "艇"],
    "台": ["机器", "计算机", "设备"],
    "本": ["书", "文件", "册"],
    "条": ["河", "鱼", "消息", "船"],
    "位": ["人", "先生", "女士"],
    "名": ["人", "学生", "员工"],
    "只": ["猫", "狗", "鸟"],
    "节": ["课", "航速"],
    "次": ["循环", "计算", "尝试"],
}


class ClassifierError(Exception):
    """量词验证失败异常"""
    pass


def validate_classifier(number_text: str, classifier: str) -> bool:
    """
    验证量词使用是否合法。

    规则:
    - 数字必须搭配量词使用（三**只**猫，不能说 三猫）
    - 量词必须在 CLASSIFIER_TYPES 中注册
    """
    if not classifier:
        return False  # 缺少量词
    if classifier in CLASSIFIER_TYPES:
        return True
    return False


def infer_type_from_classifier(classifier: str) -> str:
    """从量词推断类型"""
    return CLASSIFIER_TYPES.get(classifier, "未知")


def suggest_classifiers(noun: str) -> list[str]:
    """根据名词推荐量词"""
    suggestions = []
    for clf, nouns in CLASSIFIER_NOUN_HINTS.items():
        if noun in nouns:
            suggestions.append(clf)
    return suggestions


# ═══════════════════════════════════════════════════════════
# 第七层：中文NL模式定义 — 主题-评论结构
# ═══════════════════════════════════════════════════════════

@dataclass
class ChinesePattern:
    """中文NL模式定义"""
    pattern: str          # 正则表达式模式
    name: str             # 模式名称（中文）
    assembly_tpl: str     # 汇编模板（用{name}表示捕获组）
    result_reg: int       # 结果寄存器
    level: int = 0        # 层级：0=原始, 1=组合
    tags: list[str] = field(default_factory=list)


# 中文数字转换（用于模式中）
def _zh_to_arabic(s: str) -> str:
    """将中文数字部分转换为阿拉伯数字"""
    try:
        return str(parse_zh_number(s))
    except (ValueError, TypeError):
        return s


# 所有中文NL模式
CHINESE_PATTERNS: list[ChinesePattern] = [
    # ─── 算术运算 ───
    ChinesePattern(
        pattern=r"计算\s*(.+?)\s*加\s*(.+)",
        name="加法",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\nIADD R0, R0, R1\nHALT",
        result_reg=0,
        tags=["算术", "加"],
    ),
    ChinesePattern(
        pattern=r"计算\s*(.+?)\s*减\s*(.+)",
        name="减法",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\nISUB R0, R0, R1\nHALT",
        result_reg=0,
        tags=["算术", "减"],
    ),
    ChinesePattern(
        pattern=r"计算\s*(.+?)\s*乘\s*(.+)",
        name="乘法",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\nIMUL R0, R0, R1\nHALT",
        result_reg=0,
        tags=["算术", "乘"],
    ),
    ChinesePattern(
        pattern=r"计算\s*(.+?)\s*除以?\s*(.+)",
        name="除法",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\nIDIV R0, R0, R1\nHALT",
        result_reg=0,
        tags=["算术", "除"],
    ),
    ChinesePattern(
        pattern=r"计算\s*(.+?)\s*的余数\s*(.+)",
        name="取模",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\nIMOD R0, R0, R1\nHALT",
        result_reg=0,
        tags=["算术", "模"],
    ),
    ChinesePattern(
        pattern=r"从\s*(.+?)\s*到\s*(.+?)\s*的和",
        name="区间求和",
        assembly_tpl="MOVI R0, {a}\nMOVI R1, {b}\n-- 从 a 累加到 b\nMOV R2, R0\nloop:\nIADD R2, R2, R1\nDEC R1\nCMP R0, R1\nJL loop\nMOV R0, R2\nHALT",
        result_reg=0,
        level=1,
        tags=["算术", "求和"],
    ),
    ChinesePattern(
        pattern=r"(.+?)\s*的阶乘",
        name="阶乘",
        assembly_tpl="MOVI R0, {n}\nMOVI R1, 1\nMOV R2, R0\ndec_loop:\nIMUL R1, R1, R2\nDEC R2\nJNZ R2, dec_loop\nMOV R0, R1\nHALT",
        result_reg=0,
        level=1,
        tags=["算术", "阶乘"],
    ),

    # ─── 寄存器操作 ───
    ChinesePattern(
        pattern=r"加载\s*(.+?)\s*为\s*(.+)",
        name="寄存器赋值",
        assembly_tpl="MOVI {reg}, {val}\nHALT",
        result_reg=0,
        tags=["寄存器"],
    ),
    ChinesePattern(
        pattern=r"(.+?)\s*加\s*(.+)",
        name="寄存器加法",
        assembly_tpl="IADD R0, {a}, {b}\nHALT",
        result_reg=0,
        level=1,
        tags=["寄存器", "加"],
    ),
    ChinesePattern(
        pattern=r"(.+?)\s*减\s*(.+)",
        name="寄存器减法",
        assembly_tpl="ISUB R0, {a}, {b}\nHALT",
        result_reg=0,
        level=1,
        tags=["寄存器", "减"],
    ),

    # ─── 控制流 ───
    ChinesePattern(
        pattern=r"减少\s*(.+)",
        name="递减",
        assembly_tpl="DEC {reg}\nHALT",
        result_reg=0,
        tags=["控制流", "递减"],
    ),
    ChinesePattern(
        pattern=r"增加\s*(.+)",
        name="递增",
        assembly_tpl="INC {reg}\nHALT",
        result_reg=0,
        tags=["控制流", "递增"],
    ),

    # ─── 智能体通信 ───
    ChinesePattern(
        pattern=r"告诉\s*(.+?)\s*(.+)",
        name="告知",
        assembly_tpl="MOVI R0, {agent}\nMOVI R1, {msg}\nTELL R0, R1\nHALT",
        result_reg=0,
        tags=["智能体", "告知"],
    ),
    ChinesePattern(
        pattern=r"询问\s*(.+?)\s*(.+)",
        name="询问",
        assembly_tpl="MOVI R0, {agent}\nMOVI R1, {topic}\nASK R0, R1\nHALT",
        result_reg=0,
        tags=["智能体", "询问"],
    ),
    ChinesePattern(
        pattern=r"广播\s*(.+)",
        name="广播",
        assembly_tpl="MOVI R0, 0\nMOVI R1, {msg}\nBROADCAST R0, R1\nHALT",
        result_reg=0,
        tags=["智能体", "广播"],
    ),
    ChinesePattern(
        pattern=r"委托\s*(.+?)\s*(.+)",
        name="委托",
        assembly_tpl="MOVI R0, {agent}\nMOVI R1, {task}\nDELEGATE R0, R1\nHALT",
        result_reg=0,
        tags=["智能体", "委托"],
    ),

    # ─── 返回 ───
    ChinesePattern(
        pattern=r"返回\s*(.+)",
        name="返回值",
        assembly_tpl="MOVI R0, {val}\nRET",
        result_reg=0,
        tags=["控制流", "返回"],
    ),

    # ─── 打印 ───
    ChinesePattern(
        pattern=r"打印\s*(.+)",
        name="打印",
        assembly_tpl="PRINT {reg}\nHALT",
        result_reg=0,
        tags=["系统", "打印"],
    ),
]


# ═══════════════════════════════════════════════════════════
# 第八层：中文NL解释器 — 主入口
# ═══════════════════════════════════════════════════════════

@dataclass
class CompilationResult:
    """编译结果"""
    success: bool
    assembly: str
    bytecode: bytearray
    pattern_name: str
    captures: dict[str, str]
    error: str | None = None
    classifier_warnings: list[str] = field(default_factory=list)


def _resolve_register(text: str) -> str:
    """将中文寄存器名解析为 Rn 格式"""
    text = text.strip()
    if text in ZH_REGISTERS:
        return f"R{ZH_REGISTERS[text]}"
    if re.match(r'^R\d+$', text, re.IGNORECASE):
        return text.upper()
    return text


def _resolve_operand(text: str, as_immediate: bool = False) -> str:
    """
    解析操作数：中文数字、寄存器名或阿拉伯数字。

    Args:
        as_immediate: 如果为True，寄存器名解析为数字ID而非寄存器引用。
                        用于智能体通信场景（甲→0, 乙→1）。
    """
    text = text.strip()
    # 检查是否为中文寄存器名
    if text in ZH_REGISTERS:
        if as_immediate:
            return str(ZH_REGISTERS[text])  # 甲 → "0" (立即数)
        return f"R{ZH_REGISTERS[text]}"  # 甲 → "R0" (寄存器引用)
    if re.match(r'^R\d+$', text, re.IGNORECASE):
        if as_immediate:
            return str(int(text[1:]))  # R0 → "0"
        return text.upper()
    # 尝试解析为数字
    try:
        num = parse_zh_number(text)
        return str(num)
    except (ValueError, TypeError):
        # 非数字非寄存器 → 作为消息文本，编码为简单哈希值
        if as_immediate:
            return str(hash(text) & 0xFFFF)
        return text


def compile_chinese(nl_text: str) -> CompilationResult:
    """
    将中文自然语言编译为FLUX字节码。

    编译管道:
      1. 模式匹配 — 找到匹配的中文NL模式
      2. 捕获组解析 — 解析中文数字和寄存器
      3. 汇编展开 — 将值填入汇编模板
      4. 字节码编码 — 汇编 → 字节数组
      5. 量词检查 — 验证量词使用（可选）
    """
    nl_text = nl_text.strip()

    # 跳过注释
    if nl_text.startswith("--") or nl_text.startswith("//") or nl_text.startswith("#"):
        return CompilationResult(
            success=False, assembly="", bytecode=bytearray(),
            pattern_name="", captures={}, error="注释行",
        )

    # 尝试匹配所有模式
    for pattern in CHINESE_PATTERNS:
        match = re.match(pattern.pattern, nl_text, re.DOTALL)
        if match:
            captures = match.groupdict() if match.groupdict() else {}
            # 如果没有命名组，用编号组
            if not captures:
                groups = match.groups()
                # 根据模式名称和模板确定捕获组语义
                captures = _assign_capture_names(pattern.name, groups, pattern.assembly_tpl)

            # 解析所有捕获值
            # 智能体通信模式中的agent/msg需要解析为立即数
            agent_tags = {"智能体", "告知", "询问", "委托", "广播"}
            use_immediate = bool(agent_tags & set(pattern.tags))
            resolved: dict[str, str] = {}
            for key, val in captures.items():
                resolved[key] = _resolve_operand(val, as_immediate=use_immediate)

            # 展开汇编模板
            assembly = pattern.assembly_tpl
            for key, val in resolved.items():
                assembly = assembly.replace(f"{{{key}}}", val)

            # 编码为字节码
            try:
                bytecode = quick_encode(assembly)
            except Exception as e:
                return CompilationResult(
                    success=False, assembly=assembly, bytecode=bytearray(),
                    pattern_name=pattern.name, captures=captures,
                    error=f"编码错误: {e}",
                )

            return CompilationResult(
                success=True,
                assembly=assembly,
                bytecode=bytecode,
                pattern_name=pattern.name,
                captures=captures,
                error=None,
            )

    # 没有匹配的模式 — 尝试直接当作汇编（仅包含有效助记符时才视为汇编）
    valid_mnemonics = set(vars(Op).keys())
    parts = nl_text.split()
    if parts and parts[0].upper() in valid_mnemonics:
        try:
            bytecode = quick_encode(nl_text)
            return CompilationResult(
                success=True,
                assembly=nl_text,
                bytecode=bytecode,
                pattern_name="直接汇编",
                captures={},
                error=None,
            )
        except Exception:
            pass

    return CompilationResult(
        success=False, assembly="", bytecode=bytearray(),
        pattern_name="", captures={},
        error=f"无法理解: {nl_text}",
    )


def _detect_template_vars(template: str) -> list[str]:
    """从汇编模板中提取 {var} 变量名"""
    return re.findall(r'\{(\w+)\}', template)


def _assign_capture_names(pattern_name: str, groups: tuple, assembly_tpl: str = "") -> dict[str, str]:
    """根据模式名称为编号捕获组分配语义名称"""
    if not groups:
        return {}

    name_map = {
        "加法": ("a", "b"),
        "减法": ("a", "b"),
        "乘法": ("a", "b"),
        "除法": ("a", "b"),
        "取模": ("a", "b"),
        "区间求和": ("a", "b"),
        "寄存器加法": ("a", "b"),
        "寄存器减法": ("a", "b"),
        "寄存器赋值": ("reg", "val"),
        "告知": ("agent", "msg"),
        "询问": ("agent", "topic"),
        "委托": ("agent", "task"),
        "递减": ("reg",),
        "递增": ("reg",),
        "返回值": ("val",),
        "打印": ("reg",),
        "广播": ("msg",),
        "阶乘": ("n",),
    }

    names = name_map.get(pattern_name, None)
    if names is None:
        # 回退：从汇编模板中检测变量名
        if assembly_tpl:
            tpl_vars = _detect_template_vars(assembly_tpl)
            if tpl_vars and len(tpl_vars) == len(groups):
                names = tuple(tpl_vars)

    if names is None:
        names = tuple(f"g{i}" for i in range(len(groups)))

    return {names[i]: groups[i] for i in range(min(len(names), len(groups)))}


def compile_and_execute(nl_text: str, trace: bool = False) -> tuple[CompilationResult, ExecutionResult]:
    """编译并执行中文自然语言"""
    comp = compile_chinese(nl_text)
    if not comp.success:
        return comp, ExecutionResult(
            success=False, result=0, registers=[], cycles=0,
            disassembly="", error=comp.error, halted=False,
        )
    vm = FluxVM(comp.bytecode, trace=trace)
    exec_result = vm.execute()
    return comp, exec_result


def compile_program(nl_lines: str, trace: bool = False) -> tuple[CompilationResult, ExecutionResult]:
    """
    编译并执行多行中文程序。

    支持注释(-- 和 #)和空行。
    每行独立编译并顺序执行。
    """
    lines = nl_lines.strip().split("\n")
    all_asm: list[str] = []
    pattern_names: list[str] = []
    last_error = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("//") or line.startswith("#"):
            continue

        comp = compile_chinese(line)
        if comp.success:
            # 移除末尾的HALT（多行程序中只需要一个）
            asm = comp.assembly.rstrip()
            if asm.endswith("HALT") and line != lines[-1].strip():
                asm = asm[:-4].rstrip()
            all_asm.append(asm)
            pattern_names.append(comp.pattern_name)
        else:
            last_error = comp.error

    full_asm = "\n".join(all_asm)
    if not full_asm:
        full_asm = "NOP\nHALT"

    # 确保以HALT结尾
    full_asm = full_asm.rstrip()
    if not full_asm.endswith("HALT"):
        full_asm += "\nHALT"

    bytecode = quick_encode(full_asm)
    vm = FluxVM(bytecode, trace=trace)
    exec_result = vm.execute()

    comp_result = CompilationResult(
        success=last_error is None,
        assembly=full_asm,
        bytecode=bytecode,
        pattern_name="; ".join(pattern_names),
        captures={},
        error=last_error,
    )
    return comp_result, exec_result


def format_bytecode_hex(bytecode: bytes) -> str:
    """将字节码格式化为十六进制字符串"""
    return " ".join(f"{b:02x}" for b in bytecode)
