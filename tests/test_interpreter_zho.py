"""
流星运行时 — 中文NL解释器测试套件

覆盖范围:
  - 中文数字解析
  - 寄存器名解析
  - 算术运算模式（加减乘除模）
  - 阶乘
  - 寄存器操作
  - 智能体通信（告知、询问、广播）
  - 量词验证
  - 零形回指
  - 主题-评论结构
  - VM执行引擎
  - 字节码编码/反汇编
  - 多行程序编译
"""

import pytest
from flux_zho.interpreter import (
    Op,
    OP_NAMES,
    parse_zh_number,
    ZH_REGISTERS,
    resolve_value,
    validate_classifier,
    infer_type_from_classifier,
    suggest_classifiers,
    ClassifierError,
    compile_chinese,
    compile_and_execute,
    compile_program,
    FluxVM,
    quick_encode,
    quick_exec,
    disassemble,
    format_assembly,
    format_bytecode_hex,
    encode_assembly,
    CHINESE_PATTERNS,
)


# ═══════════════════════════════════════════════════════
# 中文数字解析测试
# ═══════════════════════════════════════════════════════

class TestChineseNumbers:
    """中文数字解析"""

    def test_simple_digits(self):
        assert parse_zh_number("三") == 3
        assert parse_zh_number("五") == 5
        assert parse_zh_number("零") == 0
        assert parse_zh_number("九") == 9

    def test_two_alternate(self):
        """两 和 二 都表示 2"""
        assert parse_zh_number("两") == 2
        assert parse_zh_number("二") == 2

    def test_ten(self):
        assert parse_zh_number("十") == 10
        assert parse_zh_number("十五") == 15
        assert parse_zh_number("十二") == 12
        assert parse_zh_number("二十") == 20

    def test_larger_numbers(self):
        assert parse_zh_number("一百") == 100
        assert parse_zh_number("三百二十七") == 327
        assert parse_zh_number("四十二") == 42

    def test_arabic_fallback(self):
        assert parse_zh_number("42") == 42
        assert parse_zh_number("100") == 100
        assert parse_zh_number("0") == 0
        assert parse_zh_number("12345") == 12345


# ═══════════════════════════════════════════════════════
# 操作码测试
# ═══════════════════════════════════════════════════════

class TestOpcodes:
    """FLUX操作码一致性"""

    def test_opcode_values(self):
        assert Op.NOP == 0x00
        assert Op.HALT == 0xFF
        assert Op.IADD == 0x08
        assert Op.PRINT == 0xFE

    def test_op_names_reverse_lookup(self):
        assert OP_NAMES[0x00] == "NOP"
        assert OP_NAMES[0xFF] == "HALT"
        assert OP_NAMES[0x08] == "IADD"

    def test_all_opcodes_have_names(self):
        """确保所有操作码都有名称"""
        opcodes = [v for k, v in vars(Op).items()
                   if isinstance(v, int) and k.isupper() and not k.startswith("_")]
        for op in opcodes:
            assert op in OP_NAMES, f"操作码 {op:#x} 缺少名称"


# ═══════════════════════════════════════════════════════
# VM执行引擎测试
# ═══════════════════════════════════════════════════════

class TestVM:
    """FLUX虚拟机执行"""

    def test_simple_addition(self):
        result = quick_exec("MOVI R0, 3\nMOVI R1, 4\nIADD R0, R0, R1\nHALT")
        assert result.success
        assert result.result == 7
        assert result.halted

    def test_subtraction(self):
        result = quick_exec("MOVI R0, 10\nMOVI R1, 3\nISUB R0, R0, R1\nHALT")
        assert result.success
        assert result.result == 7

    def test_multiplication(self):
        result = quick_exec("MOVI R0, 5\nMOVI R1, 6\nIMUL R0, R0, R1\nHALT")
        assert result.success
        assert result.result == 30

    def test_division(self):
        result = quick_exec("MOVI R0, 20\nMOVI R1, 4\nIDIV R0, R0, R1\nHALT")
        assert result.success
        assert result.result == 5

    def test_division_by_zero(self):
        result = quick_exec("MOVI R0, 10\nMOVI R1, 0\nIDIV R0, R0, R1\nHALT")
        assert not result.success
        assert result.error == "除零错误"

    def test_mov_immediate(self):
        result = quick_exec("MOVI R0, 42\nHALT")
        assert result.success
        assert result.result == 42

    def test_increment(self):
        result = quick_exec("MOVI R0, 5\nINC R0\nHALT")
        assert result.success
        assert result.result == 6

    def test_decrement(self):
        result = quick_exec("MOVI R0, 5\nDEC R0\nHALT")
        assert result.success
        assert result.result == 4

    def test_factorial_five(self):
        """5! = 120"""
        asm = (
            "MOVI R0, 5\n"
            "MOVI R1, 1\n"
            "MOV R2, R0\n"
            "dec_loop:\n"
            "IMUL R1, R1, R2\n"
            "DEC R2\n"
            "JNZ R2, dec_loop\n"
            "MOV R0, R1\n"
            "HALT"
        )
        result = quick_exec(asm)
        assert result.success
        assert result.result == 120

    def test_register_access(self):
        """多寄存器操作"""
        result = quick_exec("MOVI R0, 10\nMOVI R1, 20\nMOVI R2, 30\nIADD R3, R0, R1\nHALT")
        assert result.success
        assert result.registers[0] == 10
        assert result.registers[1] == 20
        assert result.registers[2] == 30
        assert result.registers[3] == 30

    def test_print_trace(self):
        """PRINT指令生成追踪"""
        result = quick_exec("MOVI R0, 42\nPRINT R0\nHALT", trace=True)
        assert result.success
        assert len(result.trace) == 1
        assert "42" in result.trace[0]


# ═══════════════════════════════════════════════════════
# 中文NL编译测试
# ═══════════════════════════════════════════════════════

class TestChineseCompilation:
    """中文自然语言编译"""

    def test_addition(self):
        comp, exec_result = compile_and_execute("计算 三加四")
        assert comp.success
        assert exec_result.result == 7

    def test_multiplication(self):
        comp, exec_result = compile_and_execute("计算 五乘六")
        assert comp.success
        assert exec_result.result == 30

    def test_subtraction(self):
        comp, exec_result = compile_and_execute("计算 十减三")
        assert comp.success
        assert exec_result.result == 7

    def test_division(self):
        comp, exec_result = compile_and_execute("计算 十二除以 四")
        assert comp.success
        assert exec_result.result == 3

    def test_factorial(self):
        comp, exec_result = compile_and_execute("五的阶乘")
        assert comp.success
        assert exec_result.result == 120

    def test_load_register(self):
        comp, exec_result = compile_and_execute("加载 寄存器零 为 四十二")
        assert comp.success
        assert exec_result.result == 42

    def test_decrement_register(self):
        comp, exec_result = compile_and_execute("减少 寄存器零")
        assert comp.success

    def test_tell_agent(self):
        comp, _ = compile_and_execute("告诉 甲 你好")
        assert comp.success
        assert comp.pattern_name == "告知"

    def test_ask_agent(self):
        comp, _ = compile_and_execute("询问 甲 船速")
        assert comp.success
        assert comp.pattern_name == "询问"

    def test_broadcast(self):
        comp, _ = compile_and_execute("广播 全体注意")
        assert comp.success
        assert comp.pattern_name == "广播"

    def test_return_value(self):
        comp, _ = compile_and_execute("返回 四十二")
        assert comp.success

    def test_print_register(self):
        comp, _ = compile_and_execute("打印 寄存器零")
        assert comp.success

    def test_comment_ignored(self):
        comp, _ = compile_and_execute("-- 这是注释")
        assert not comp.success  # 注释行被跳过

    def test_unknown_pattern(self):
        comp, _ = compile_and_execute("这是无法理解的话")
        assert not comp.success
        assert comp.error is not None


# ═══════════════════════════════════════════════════════
# 量词系统测试
# ═══════════════════════════════════════════════════════

class TestClassifierSystem:
    """量词类型系统"""

    def test_valid_classifier(self):
        assert validate_classifier("三", "个") is True
        assert validate_classifier("五", "只") is True
        assert validate_classifier("一", "台") is True

    def test_missing_classifier(self):
        assert validate_classifier("三", "") is False
        assert validate_classifier("三", None) is False

    def test_unknown_classifier(self):
        assert validate_classifier("三", "xyz") is False

    def test_infer_type(self):
        assert infer_type_from_classifier("艘") == "船只"
        assert infer_type_from_classifier("台") == "机器/设备"
        assert infer_type_from_classifier("本") == "书籍/文件"

    def test_suggest_classifiers_for_noun(self):
        suggestions = suggest_classifiers("船")
        assert "艘" in suggestions

        suggestions = suggest_classifiers("计算机")
        assert "台" in suggestions

        suggestions = suggest_classifiers("书")
        assert "本" in suggestions


# ═══════════════════════════════════════════════════════
# 零形回指与主题-评论测试
# ═══════════════════════════════════════════════════════

class TestTopicComment:
    """主题-评论结构与零形回指"""

    def test_register_name_mapping(self):
        """中文寄存器名映射"""
        assert "寄存器零" in ZH_REGISTERS
        assert ZH_REGISTERS["寄存器零"] == 0
        assert ZH_REGISTERS["寄存器一"] == 1
        assert ZH_REGISTERS["甲"] == 0
        assert ZH_REGISTERS["乙"] == 1

    def test_zero_anaphora_empty(self):
        """空字符串触发零形回指"""
        assert resolve_value("", topic_reg_val=42) == 42

    def test_zero_anaphora_pronoun(self):
        """'它'触发零形回指"""
        assert resolve_value("它", topic_reg_val=42) == 42
        assert resolve_value("其", topic_reg_val=99) == 99

    def test_patterns_exist(self):
        """确保定义了足够的中文模式"""
        assert len(CHINESE_PATTERNS) >= 15

    def test_arithmetic_patterns(self):
        """算术模式全部存在"""
        names = [p.name for p in CHINESE_PATTERNS]
        assert "加法" in names
        assert "减法" in names
        assert "乘法" in names
        assert "除法" in names
        assert "阶乘" in names


# ═══════════════════════════════════════════════════════
# 编码器与反汇编测试
# ═══════════════════════════════════════════════════════

class TestEncoderDecoder:
    """字节码编码与反汇编"""

    def test_encode_decode_roundtrip(self):
        asm = "MOVI R0, 42\nHALT"
        bytecode = quick_encode(asm)
        instructions = disassemble(bytecode)
        assert len(instructions) == 2
        assert instructions[0].mnemonic == "MOVI"
        assert instructions[1].mnemonic == "HALT"

    def test_format_assembly(self):
        bytecode = quick_encode("MOVI R0, 42\nHALT")
        instructions = disassemble(bytecode)
        output = format_assembly(instructions)
        assert "MOVI" in output
        assert "HALT" in output

    def test_bytecode_hex_format(self):
        bytecode = quick_encode("NOP\nHALT")
        hex_str = format_bytecode_hex(bytecode)
        assert "00" in hex_str
        assert "ff" in hex_str

    def test_label_encoding(self):
        """标签编码测试"""
        asm = "MOVI R0, 5\nloop:\nDEC R0\nJNZ R0, loop\nHALT"
        buf, instructions, labels = encode_assembly(asm)
        assert "loop" in labels


# ═══════════════════════════════════════════════════════
# 多行程序测试
# ═══════════════════════════════════════════════════════

class TestMultiLinePrograms:
    """多行中文程序编译"""

    def test_multi_line_basic(self):
        program = """
-- 多行程序示例
加载 寄存器零 为 三
加载 寄存器一 为 七
"""
        comp, exec_result = compile_program(program)
        assert comp.success
        assert exec_result.registers[0] == 3
        assert exec_result.registers[1] == 7

    def test_multi_line_with_comments(self):
        program = """
# 这是注释
加载 寄存器零 为 十
加载 寄存器一 为 二十
"""
        comp, exec_result = compile_program(program)
        assert comp.success
