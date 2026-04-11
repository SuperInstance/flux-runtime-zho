"""
流星编码器 — 独立汇编编码/反汇编模块

从 interpreter.py 提取的编码/反汇编逻辑，作为干净的独立模块。

支持:
  - 中文助记符: 加载/存储/计算/增加/减少/打印/停机 等
  - 英文助记符: MOV/STORE/IADD/INC/DEC/PRINT/HALT 等
  - 中文寄存器名: 甲(0), 乙(1), 丙(2), 丁(3), 戊(4), 己(5), 庚(6), 辛(7), 壬(8), 癸(9)
  - 阿拉伯数字寄存器: R0–R63
  - 中文数字立即数: 三, 十二, 一百零五 等
  - 标签与前向引用
  - 注释: # 或 // 或 -- 开头

导出函数:
  encode_assembly()   — 汇编文本 → (字节数组, 指令列表, 标签表)
  quick_encode()      — 汇编文本 → 字节数组 (快速)
  disassemble()       — 字节数组 → 指令列表
  format_assembly()   — 指令列表 → 可读文本
  format_bytecode_hex() — 字节数组 → 十六进制字符串
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# FLUX 操作码 — 与 interpreter.py Op 类完全一致
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# 中文助记符映射
# ══════════════════════════════════════════════════════════════════════

# 中文 → 操作码
ZH_MNEMONICS: dict[str, int] = {
    # 控制流
    "空操作": Op.NOP,
    "移动": Op.MOV,
    "加载": Op.MOVI,      # 加载对应立即数加载
    "存储": Op.STORE,
    "跳转": Op.JMP,
    "零跳": Op.JZ,
    "非零跳": Op.JNZ,
    "调用": Op.CALL,
    "返回": Op.RET,

    # 算术运算
    "计算": Op.IADD,       # 默认为加法
    "加": Op.IADD,
    "减": Op.ISUB,
    "乘": Op.IMUL,
    "除": Op.IDIV,
    "模": Op.IMOD,
    "取反": Op.INEG,
    "增加": Op.INC,
    "减少": Op.DEC,

    # 比较
    "比较": Op.CMP,
    "等于": Op.JE,
    "不等于": Op.JNE,
    "小于": Op.JL,
    "大于等于": Op.JGE,

    # 栈操作
    "压栈": Op.PUSH,
    "出栈": Op.POP,
    "复制栈": Op.DUP,
    "交换栈": Op.SWAP,

    # 系统
    "打印": Op.PRINT,
    "停机": Op.HALT,
    "停止": Op.HALT,

    # 智能体协议
    "告知": Op.TELL,
    "询问": Op.ASK,
    "委托": Op.DELEGATE,
    "广播": Op.BROADCAST,
    "信任验证": Op.TRUST_CHECK,
    "能力请求": Op.CAPABILITY_REQ,

    # 内存管理
    "创建区域": Op.REGION_CREATE,
    "销毁区域": Op.REGION_DESTROY,
}

# 合并助记符表 (英文优先，中文作为补充)
_MNEMONIC_TO_OPCODE: dict[str, int] = {}
# 先加载英文操作码
for _n, _c in list(vars(Op).items()):
    if isinstance(_c, int) and not _n.startswith("_") and _n.isupper():
        _MNEMONIC_TO_OPCODE[_n] = _c
# 再加载中文操作码 (中文不覆盖英文)
for _zh, _opcode in ZH_MNEMONICS.items():
    if _zh not in _MNEMONIC_TO_OPCODE:
        _MNEMONIC_TO_OPCODE[_zh] = _opcode


# ══════════════════════════════════════════════════════════════════════
# 中文寄存器名映射
# ══════════════════════════════════════════════════════════════════════

# 天干寄存器: 甲乙丙丁戊己庚辛壬癸 → R0-R9
TIANGAN_REGISTERS: dict[str, int] = {
    "甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4,
    "己": 5, "庚": 6, "辛": 7, "壬": 8, "癸": 9,
}

# 完整中文寄存器映射
ZH_REGISTERS: dict[str, int] = {
    "寄存器零": 0, "寄存器一": 1, "寄存器二": 2, "寄存器三": 3,
    "寄存器四": 4, "寄存器五": 5, "寄存器六": 6, "寄存器七": 7,
    "寄存器八": 8, "寄存器九": 9, "寄存器十": 10,
    "主题寄存器": 63,
    **TIANGAN_REGISTERS,
}


# ══════════════════════════════════════════════════════════════════════
# 中文数字解析
# ══════════════════════════════════════════════════════════════════════

ZH_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}


def parse_zh_number(text: str) -> int:
    """
    解析中文数字为整数。

    支持: "三", "十", "十五", "三百二十七", "四十二", "二十", "一千"等。
    也支持直接阿拉伯数字。
    """
    text = text.strip()

    # 直接阿拉伯数字
    if re.match(r'^-?\d+$', text):
        return int(text)

    # 十六进制
    if text.startswith("0x") or text.startswith("0X"):
        return int(text, 16)

    # 全角数字
    ZH_FULLWIDTH = "０１２３４５６７８９"

    result = 0
    current = 0
    temp = 0

    for char in text:
        if char in ZH_FULLWIDTH:
            temp = ZH_FULLWIDTH.index(char)
            current += temp
            continue

        if char not in ZH_DIGITS:
            continue

        val = ZH_DIGITS[char]

        if val < 10:
            temp = val
        elif val == 10:
            multiplier = temp if temp > 0 else 1
            current += multiplier * 10
            temp = 0
        elif val == 100:
            multiplier = temp if temp > 0 else 1
            current += multiplier * 100
            temp = 0
        elif val == 1000:
            multiplier = temp if temp > 0 else 1
            current += multiplier * 1000
            temp = 0
        elif val == 10000:
            result += (current + temp) * 10000
            current = 0
            temp = 0

    result += current + temp
    return result


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EncodedInstruction:
    """已编码的指令"""
    offset: int
    opcode: int
    operands: list = field(default_factory=list)
    size: int = 1
    mnemonic: str = ""

    def __repr__(self) -> str:
        ops = ", ".join(str(o) for o in self.operands)
        return f"[{self.offset:04d}] {self.mnemonic:<16s} {ops}"


@dataclass
class DecodedInstruction:
    """反汇编的指令"""
    offset: int
    mnemonic: str
    operands: str
    size: int
    opcode: int

    def __repr__(self) -> str:
        return f"  {self.offset:04d}:  {self.mnemonic:<12s} {self.operands}"


# ══════════════════════════════════════════════════════════════════════
# 解析辅助函数
# ══════════════════════════════════════════════════════════════════════

def _parse_register(s: str) -> int:
    """
    解析寄存器名。

    支持:
      - R0–R63 (大小写不敏感)
      - 甲乙丙丁戊己庚辛壬癸 (0-9)
      - 寄存器零~寄存器十 (0-10)
      - 纯数字 0-63
    """
    s = s.strip()

    # R 前缀
    m = re.match(r"^R(\d+)$", s, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 0 <= num < 64:
            return num
        raise ValueError(f"寄存器超出范围: R{num} (有效: R0–R63)")

    # 中文寄存器名
    if s in ZH_REGISTERS:
        return ZH_REGISTERS[s]

    # 纯数字
    try:
        num = int(s)
        if 0 <= num < 64:
            return num
        raise ValueError(f"寄存器超出范围: {num} (有效: 0–63)")
    except ValueError:
        pass

    raise ValueError(
        f"无效寄存器: '{s}'。"
        f"支持 R0–R63、甲乙丙丁...、寄存器零~寄存器十"
    )


def _parse_immediate(s: str) -> int:
    """
    解析立即数值。

    支持:
      - 阿拉伯数字: 42, 0xFF, 0b1010
      - 中文数字: 三, 十二, 三百二十七
    """
    s = s.strip()

    # 阿拉伯数字
    try:
        return int(s)
    except ValueError:
        pass

    # 十六进制
    if s.startswith("0x") or s.startswith("0X"):
        try:
            return int(s, 16)
        except ValueError:
            pass

    # 二进制
    if s.startswith("0b") or s.startswith("0B"):
        try:
            return int(s, 2)
        except ValueError:
            pass

    # 中文数字
    try:
        return parse_zh_number(s)
    except (ValueError, TypeError):
        pass

    raise ValueError(f"无效立即数: '{s}'")


def _encode_u16(val: int) -> bytes:
    """编码 16 位无符号整数 (小端序)"""
    val = val & 0xFFFF
    return bytes([val & 0xFF, (val >> 8) & 0xFF])


# ══════════════════════════════════════════════════════════════════════
# 指令大小估算 (用于标签偏移计算)
# ══════════════════════════════════════════════════════════════════════

def _estimate_instruction_size(mnemonic: str, args: list[str]) -> int:
    """估算指令字节数 (必须与实际编码一致)"""
    opcode = _MNEMONIC_TO_OPCODE.get(mnemonic.upper(), Op.NOP)

    # 零操作数
    if opcode in (Op.NOP, Op.HALT, Op.RET, Op.LEAVE):
        return 1

    # 单寄存器
    if opcode in (Op.INC, Op.DEC, Op.INEG, Op.INOT, Op.PRINT):
        return 2

    # 双寄存器
    if opcode in (Op.MOV, Op.LOAD, Op.STORE, Op.CMP, Op.ICMP,
                  Op.FEQ, Op.FLT, Op.FLE, Op.FGT, Op.FGE,
                  Op.SLEN, Op.SCHAR, Op.SSUB, Op.SCMP,
                  Op.CAST, Op.BOX, Op.UNBOX, Op.CHECK_TYPE, Op.CHECK_BOUNDS,
                  Op.FNEG, Op.FABS, Op.FMIN, Op.FMAX,
                  Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
                  Op.TRUST_CHECK, Op.CAPABILITY_REQ,
                  Op.PUSH, Op.POP, Op.REGION_CREATE, Op.REGION_DESTROY):
        return 3

    # 三寄存器
    if opcode in (Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD,
                  Op.IAND, Op.IOR, Op.IXOR, Op.ISHL, Op.ISHR,
                  Op.ROTL, Op.ROTR, Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV,
                  Op.SCONCAT):
        return 4

    # 立即数加载
    if opcode == Op.MOVI:
        return 4

    # 条件跳转
    if opcode in (Op.JZ, Op.JNZ, Op.JE, Op.JNE, Op.JL, Op.JGE):
        return 4

    # 无条件跳转
    if opcode == Op.JMP:
        return 3

    # 栈操作 (单字节操作数)
    if opcode in (Op.DUP, Op.SWAP, Op.ENTER, Op.ALLOCA):
        return 2

    # 内存操作
    if opcode in (Op.MEMCOPY, Op.MEMSET, Op.MEMCMP):
        return 4

    return 1


# ══════════════════════════════════════════════════════════════════════
# 单条指令编码
# ══════════════════════════════════════════════════════════════════════

def _encode_instruction(
    opcode: int,
    mnemonic: str,
    args: list[str],
    labels: dict[str, int],
) -> dict:
    """编码单条汇编指令"""
    # 零操作数
    if opcode in (Op.NOP, Op.HALT, Op.RET, Op.LEAVE):
        return {"bytes": bytes([opcode]), "operands": []}

    # 单寄存器
    if opcode in (Op.INC, Op.DEC, Op.INEG, Op.INOT, Op.PRINT):
        r = _parse_register(args[0]) if args else 0
        return {"bytes": bytes([opcode, r]), "operands": [r]}

    # 双寄存器
    if opcode in (Op.MOV, Op.LOAD, Op.STORE):
        rd = _parse_register(args[0])
        rs = _parse_register(args[1])
        return {"bytes": bytes([opcode, rd, rs]), "operands": [rd, rs]}

    # 三寄存器
    three_reg_ops = {
        Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD,
        Op.IAND, Op.IOR, Op.IXOR, Op.ISHL, Op.ISHR,
        Op.ROTL, Op.ROTR, Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV,
        Op.SCONCAT,
    }
    if opcode in three_reg_ops:
        rd = _parse_register(args[0])
        ra = _parse_register(args[1])
        rb = _parse_register(args[2]) if len(args) > 2 else 0
        return {"bytes": bytes([opcode, rd, ra, rb]), "operands": [rd, ra, rb]}

    # 立即数加载
    if opcode == Op.MOVI:
        r = _parse_register(args[0])
        imm = _parse_immediate(args[1])
        return {"bytes": bytes([opcode, r]) + _encode_u16(imm), "operands": [r, imm]}

    # 比较
    cmp_ops = {Op.CMP, Op.ICMP, Op.FEQ, Op.FLT, Op.FLE, Op.FGT, Op.FGE,
               Op.SLEN, Op.SCHAR, Op.SSUB, Op.SCMP}
    if opcode in cmp_ops:
        ra = _parse_register(args[0])
        rb = _parse_register(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([opcode, ra, rb]), "operands": [ra, rb]}

    # 条件跳转
    cond_jumps = {Op.JZ, Op.JNZ, Op.JE, Op.JNE, Op.JL, Op.JGE}
    if opcode in cond_jumps:
        r = _parse_register(args[0])
        target = args[1] if len(args) > 1 else "0"
        if target in labels:
            addr = labels[target]
        else:
            try:
                addr = _parse_immediate(target)
            except ValueError:
                addr = 0
        return {"bytes": bytes([opcode, r]) + _encode_u16(addr), "operands": [r, addr]}

    # 无条件跳转
    if opcode == Op.JMP:
        target = args[0] if args else "0"
        if target in labels:
            addr = labels[target]
        else:
            try:
                addr = _parse_immediate(target)
            except ValueError:
                addr = 0
        return {"bytes": bytes([opcode]) + _encode_u16(addr), "operands": [addr]}

    # 栈操作
    if opcode in (Op.PUSH, Op.POP):
        r = _parse_register(args[0]) if args else 0
        return {"bytes": bytes([opcode, r]), "operands": [r]}

    if opcode in (Op.DUP, Op.SWAP, Op.ENTER, Op.ALLOCA):
        return {"bytes": bytes([opcode, 0]), "operands": [0]}

    # 内存操作
    if opcode in (Op.REGION_CREATE, Op.REGION_DESTROY):
        r = _parse_register(args[0]) if args else 0
        return {"bytes": bytes([opcode, r]), "operands": [r]}

    if opcode in (Op.MEMCOPY, Op.MEMSET, Op.MEMCMP):
        a = _parse_register(args[0])
        b = _parse_register(args[1])
        c = _parse_register(args[2]) if len(args) > 2 else 0
        return {"bytes": bytes([opcode, a, b, c]), "operands": [a, b, c]}

    # 类型操作
    type_ops = {Op.CAST, Op.BOX, Op.UNBOX, Op.CHECK_TYPE, Op.CHECK_BOUNDS,
                Op.FNEG, Op.FABS, Op.FMIN, Op.FMAX}
    if opcode in type_ops:
        a = _parse_register(args[0])
        b = _parse_register(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([opcode, a, b]), "operands": [a, b]}

    # 智能体协议
    a2a_ops = {Op.TELL, Op.ASK, Op.DELEGATE, Op.BROADCAST,
               Op.TRUST_CHECK, Op.CAPABILITY_REQ}
    if opcode in a2a_ops:
        a = _parse_register(args[0])
        b = _parse_register(args[1]) if len(args) > 1 else 0
        return {"bytes": bytes([opcode, a, b]), "operands": [a, b]}

    return {"bytes": bytes([opcode]), "operands": []}


# ══════════════════════════════════════════════════════════════════════
# 主编码函数
# ══════════════════════════════════════════════════════════════════════

def encode_assembly(
    assembly: str,
) -> tuple[bytearray, list[EncodedInstruction], dict[str, int]]:
    """
    将汇编文本编码为 FLUX 字节码。

    支持:
      - 英文助记符: MOV, IADD, MOVI, JMP, HALT, ...
      - 中文助记符: 加载, 计算, 增加, 减少, 跳转, 停机, ...
      - 寄存器: R0–R63, 甲乙丙丁戊己庚辛壬癸, 寄存器零~寄存器十
      - 立即数: 阿拉伯数字, 中文数字 (三, 十二, 三百二十七)
      - 标签: 名称: (冒号结尾)
      - 前向引用: 跳转到尚未定义的标签
      - 注释: # 或 // 或 -- 开头

    Args:
        assembly: 汇编文本 (支持多行)

    Returns:
        (字节数组, 指令列表, 标签表)

    示例:
        >>> buf, instrs, labels = encode_assembly("MOVI 甲, 四十二\\n停机")
        >>> buf[0]  # MOVI opcode
        43
        >>> buf[1]  # Register 0 (甲)
        0
    """
    lines = assembly.split("\n")
    labels: dict[str, int] = {}
    raw_lines: list[tuple[str, int]] = []
    instructions: list[EncodedInstruction] = []

    # ── 第一遍: 收集标签和计算字节偏移 ──
    byte_offset = 0

    for i, line in enumerate(lines):
        trimmed = line.strip()

        # 空行和注释
        if not trimmed or trimmed.startswith("#") or trimmed.startswith("//") or trimmed.startswith("--"):
            continue

        # 标签 (以冒号结尾)
        if trimmed.endswith(":"):
            label_name = trimmed[:-1].strip()
            labels[label_name] = byte_offset
            continue

        raw_lines.append((trimmed, i))

        # 解析助记符和参数
        parts = re.split(r"[\s,]+", trimmed)
        parts = [p for p in parts if p]
        mnemonic = parts[0]
        args = [a.strip() for a in ",".join(parts[1:]).split(",") if a.strip()]

        byte_offset += _estimate_instruction_size(mnemonic, args)

    # ── 第二遍: 编码 ──
    buf = bytearray()

    for line_text, _line_num in raw_lines:
        offset = len(buf)

        parts = re.split(r"[\s,]+", line_text.strip())
        parts = [p for p in parts if p]
        mnemonic = parts[0]
        args = [a.strip() for a in ",".join(parts[1:]).split(",") if a.strip()]

        # 查找操作码 (中文和英文)
        opcode = _MNEMONIC_TO_OPCODE.get(mnemonic.upper(), Op.NOP)

        encoded = _encode_instruction(opcode, mnemonic, args, labels)
        buf.extend(encoded["bytes"])

        # 确定显示用的助记符名称
        display_mnemonic = OP_NAMES.get(
            encoded["bytes"][0] if encoded["bytes"] else 0,
            mnemonic.upper(),
        )

        instructions.append(EncodedInstruction(
            offset=offset,
            opcode=encoded["bytes"][0] if encoded["bytes"] else 0,
            operands=encoded["operands"],
            size=len(encoded["bytes"]),
            mnemonic=display_mnemonic,
        ))

    return buf, instructions, labels


# ══════════════════════════════════════════════════════════════════════
# 快捷函数
# ══════════════════════════════════════════════════════════════════════

def quick_encode(assembly: str) -> bytearray:
    """快速编码: 汇编文本 → 字节数组"""
    buf, _, _ = encode_assembly(assembly)
    return buf


def format_bytecode_hex(bytecode: bytes | bytearray) -> str:
    """将字节数组格式化为十六进制字符串"""
    return " ".join(f"{b:02x}" for b in bytecode)


# ══════════════════════════════════════════════════════════════════════
# 反汇编器
# ══════════════════════════════════════════════════════════════════════

def disassemble(bytecode: bytes) -> list[DecodedInstruction]:
    """
    将 FLUX 字节数组反汇编为可读指令列表。

    Args:
        bytecode: FLUX 字节数组

    Returns:
        反汇编后的指令列表
    """
    instructions: list[DecodedInstruction] = []
    offset = 0
    data = bytes(bytecode)

    while offset < len(data):
        start = offset
        op = data[offset]
        name = OP_NAMES.get(op, f"UNKNOWN_0x{op:02x}")
        offset += 1
        operands: list[str] = []

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
    """格式化反汇编输出为可读文本"""
    return "\n".join(
        f"  {i.offset:04d}:  {i.mnemonic:<12s} {i.operands}"
        for i in instructions
    )


# ══════════════════════════════════════════════════════════════════════
# 验证函数
# ══════════════════════════════════════════════════════════════════════

def is_valid_mnemonic(mnemonic: str) -> bool:
    """检查助记符是否有效 (中文或英文)"""
    return mnemonic.upper() in _MNEMONIC_TO_OPCODE


def is_valid_register(name: str) -> bool:
    """检查寄存器名是否有效"""
    try:
        _parse_register(name)
        return True
    except ValueError:
        return False


def get_mnemonic_opcode(mnemonic: str) -> int | None:
    """获取助记符对应的操作码"""
    return _MNEMONIC_TO_OPCODE.get(mnemonic.upper())


def get_register_number(name: str) -> int | None:
    """获取寄存器名对应的编号"""
    try:
        return _parse_register(name)
    except ValueError:
        return None
