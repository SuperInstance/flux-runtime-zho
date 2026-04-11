"""
流星 FIR 构建器测试套件

覆盖范围:
  - FIRValue: SSA 值创建、版本管理、类型标注
  - FIRBlock: 基本块构建、中文终止符、前后继关系
  - FIRPhi: Phi 节点创建、类型合并
  - FIRBuilder: 主题寄存器 R63 跟踪、量词类型推断
  - TopicCommentNode: 主题-评论解析
  - ContinuationTree: 延续树构建
  - 敬称/语境感知系统
  - FIR 程序序列化
  - 分类器升级逻辑
  - 辅助函数
"""

import pytest

from flux_zho.fir import (
    # 类型系统
    FirType,
    FIR_TYPE_NAMES,
    FIR_TYPE_TO_CLASSIFIER,
    CLASSIFIER_TO_FIR_TYPE,
    classifier_upgrade,
    infer_fir_type,
    # 操作码
    FirOpcode,
    FIR_OPCODE_NAMES,
    # 值系统
    FIRValue,
    # 指令
    FIRInstruction,
    # Phi 节点
    FIRPhi,
    # 基本块
    FIRBlock,
    TERMINATOR_NAMES,
    # 主题-评论
    TopicCommentNode,
    ContinuationTree,
    # 敬称/语境
    detect_honorific,
    detect_context,
    HONORIFIC_MARKERS,
    CONTEXT_MARKERS,
    # 构建器
    FIRBuilder,
    FIRProgram,
    build_from_chinese,
    # 寄存器
    TIANGAN_REGISTERS,
    ZH_REGISTERS_FULL,
)


# ═══════════════════════════════════════════════════════════════
# FIRValue 测试 — SSA 值系统
# ═══════════════════════════════════════════════════════════════

class TestFIRValue:
    """SSA 值创建和版本管理"""

    def test_create_simple_value(self):
        """创建简单的 SSA 值"""
        val = FIRValue(name="测试", version=1)
        assert val.name == "测试"
        assert val.version == 1
        assert val.fir_type == FirType.UNKNOWN
        assert val.classifier == ""

    def test_auto_version(self):
        """自动版本递增"""
        FIRValue.reset_counter()
        v1 = FIRValue(name="甲")
        v2 = FIRValue(name="甲")
        assert v1.version != v2.version
        assert v2.version > v1.version

    def test_full_name(self):
        """完整 SSA 名称格式"""
        val = FIRValue(name="船速", version=3)
        assert val.full_name == "船速_3"

    def test_type_name(self):
        """类型中文名称"""
        val = FIRValue(name="距离", fir_type=FirType.DISTANCE)
        assert val.type_name == "距离"

    def test_type_classifier(self):
        """类型默认量词"""
        val = FIRValue(name="距离", fir_type=FirType.DISTANCE)
        assert val.type_classifier == "海里"

    def test_topic_flag(self):
        """主题寄存器标记"""
        val = FIRValue(name="主题", is_topic=True)
        assert val.is_topic is True

    def test_context(self):
        """语境标记"""
        val = FIRValue(name="阁下", context="尊敬")
        assert val.context == "尊敬"

    def test_equality(self):
        """值相等性比较"""
        v1 = FIRValue(name="测试", version=1)
        v2 = FIRValue(name="测试", version=1)
        v3 = FIRValue(name="测试", version=2)
        assert v1 == v2
        assert v1 != v3

    def test_hash(self):
        """值哈希"""
        v1 = FIRValue(name="测试", version=1)
        v2 = FIRValue(name="测试", version=1)
        assert hash(v1) == hash(v2)

    def test_repr(self):
        """字符串表示包含关键信息"""
        val = FIRValue(name="船速", version=1, classifier="节", fir_type=FirType.SPEED)
        r = repr(val)
        assert "船速_1" in r
        assert "速度" in r
        assert "节" in r


# ═══════════════════════════════════════════════════════════════
# FirType 测试 — 量词类型系统
# ═══════════════════════════════════════════════════════════════

class TestFirType:
    """FIR 类型系统"""

    def test_classifier_to_type(self):
        """量词 → 类型映射"""
        assert CLASSIFIER_TO_FIR_TYPE["艘"] == FirType.VESSEL
        assert CLASSIFIER_TO_FIR_TYPE["节"] == FirType.SPEED
        assert CLASSIFIER_TO_FIR_TYPE["海里"] == FirType.DISTANCE
        assert CLASSIFIER_TO_FIR_TYPE["位"] == FirType.HONORED
        assert CLASSIFIER_TO_FIR_TYPE["台"] == FirType.MACHINE
        assert CLASSIFIER_TO_FIR_TYPE["本"] == FirType.DOCUMENT
        assert CLASSIFIER_TO_FIR_TYPE["次"] == FirType.COUNTER

    def test_type_names_complete(self):
        """所有类型都有中文名"""
        for ft in FirType:
            if ft == FirType.UNKNOWN:
                continue
            assert ft in FIR_TYPE_NAMES, f"类型 {ft} 缺少中文名"

    def test_type_to_classifier_roundtrip(self):
        """类型 → 量词反向映射"""
        for clf, ft in CLASSIFIER_TO_FIR_TYPE.items():
            if ft != FirType.UNKNOWN:
                assert ft in FIR_TYPE_TO_CLASSIFIER

    def test_infer_fir_type(self):
        """从文本推断类型"""
        ft, clf = infer_fir_type("三艘船")
        assert ft == FirType.VESSEL
        assert clf == "艘"

    def test_infer_fir_type_multi_char(self):
        """多字量词推断"""
        ft, clf = infer_fir_type("一百海里")
        assert ft == FirType.DISTANCE
        assert clf == "海里"

    def test_infer_fir_type_default(self):
        """无匹配量词时默认为整数"""
        ft, clf = infer_fir_type("你好世界")
        assert ft == FirType.INTEGER
        assert clf == "个"


# ═══════════════════════════════════════════════════════════════
# FIRBlock 测试 — 基本块构建
# ═══════════════════════════════════════════════════════════════

class TestFIRBlock:
    """基本块构建与终止符"""

    def test_create_block(self):
        """创建基本块"""
        block = FIRBlock("主程序")
        assert block.label == "主程序"
        assert block.terminator == "结束"
        assert len(block.instructions) == 0

    def test_entry_block(self):
        """入口块检测"""
        block = FIRBlock("入口", terminator="开始")
        assert block.is_entry

    def test_add_instruction(self):
        """添加指令"""
        block = FIRBlock("测试")
        instr = FIRInstruction(opcode=FirOpcode.CONST, operands=[42])
        block.add_instruction(instr)
        assert len(block.instructions) == 1

    def test_add_phi(self):
        """添加 Phi 节点"""
        block = FIRBlock("合并块")
        val = FIRValue(name="x", version=1)
        phi = FIRPhi(result=val, sources=[])
        block.add_phi(phi)
        assert len(block.phi_nodes) == 1

    def test_terminator_jump(self):
        """跳转终止符"""
        block = FIRBlock("循环体")
        block.set_terminator_jump("循环开始")
        assert block.terminator == "跳转"
        assert block.successors == ["循环开始"]

    def test_terminator_loop(self):
        """循环终止符"""
        block = FIRBlock("循环检查")
        block.set_terminator_loop("循环开始")
        assert block.terminator == "循环"
        assert block.successors == ["循环开始"]

    def test_terminator_branch(self):
        """分支终止符"""
        block = FIRBlock("条件判断")
        block.set_terminator_branch("真分支", "假分支")
        assert block.terminator == "分支"
        assert block.successors == ["真分支", "假分支"]

    def test_terminator_return(self):
        """返回终止符"""
        block = FIRBlock("函数结束")
        block.set_terminator_return()
        assert block.terminator == "返回"

    def test_terminator_halt(self):
        """停机终止符"""
        block = FIRBlock("程序结束")
        block.set_terminator_halt()
        assert block.terminator == "停机"

    def test_predecessors(self):
        """前驱块关系"""
        block = FIRBlock("块B", predecessors=["块A"])
        assert "块A" in block.predecessors

    def test_repr(self):
        """字符串表示"""
        block = FIRBlock("主程序", terminator="停机")
        block.successors = ["结束"]
        r = repr(block)
        assert "主程序" in r
        assert "停机" in r

    def test_terminator_names(self):
        """中文终止符名称完整性"""
        expected = ["开始", "结束", "跳转", "循环", "分支", "返回", "停机", "不可达"]
        for name in expected:
            assert name in TERMINATOR_NAMES


# ═══════════════════════════════════════════════════════════════
# FIRPhi 测试 — Phi 节点与类型合并
# ═══════════════════════════════════════════════════════════════

class TestFIRPhi:
    """Phi 节点类型合并"""

    def test_single_source_type(self):
        """单一来源类型"""
        val = FIRValue(name="x", version=1, fir_type=FirType.SPEED, classifier="节")
        phi = FIRPhi(
            result=FIRValue(name="x", version=2),
            sources=[(val, "块A")],
        )
        assert phi.merged_type == FirType.SPEED
        assert phi.merged_classifier == "节"

    def test_same_type_merge(self):
        """相同类型合并"""
        v1 = FIRValue(name="x", version=1, fir_type=FirType.SPEED, classifier="节")
        v2 = FIRValue(name="x", version=2, fir_type=FirType.SPEED, classifier="节")
        phi = FIRPhi(
            result=FIRValue(name="x", version=3),
            sources=[(v1, "块A"), (v2, "块B")],
        )
        assert phi.merged_type == FirType.SPEED

    def test_type_conflict_upgrade(self):
        """类型冲突提升为通用类型"""
        v1 = FIRValue(name="x", version=1, fir_type=FirType.VESSEL, classifier="艘")
        v2 = FIRValue(name="x", version=2, fir_type=FirType.SPEED, classifier="节")
        phi = FIRPhi(
            result=FIRValue(name="x", version=3),
            sources=[(v1, "块A"), (v2, "块B")],
        )
        assert phi.merged_type == FirType.INTEGER

    def test_empty_sources(self):
        """空来源默认类型"""
        phi = FIRPhi(
            result=FIRValue(name="x", version=1),
            sources=[],
        )
        assert phi.merged_type == FirType.UNKNOWN

    def test_repr(self):
        """Phi 节点字符串表示"""
        val = FIRValue(name="x", version=1)
        phi = FIRPhi(
            result=FIRValue(name="x", version=2),
            sources=[(val, "块A")],
        )
        r = repr(phi)
        assert "Φ" in r
        assert "块A" in r


# ═══════════════════════════════════════════════════════════════
# TopicCommentNode 测试 — 主题-评论结构
# ═══════════════════════════════════════════════════════════════

class TestTopicCommentNode:
    """主题-评论语法树节点"""

    def test_create_node(self):
        """创建主题-评论节点"""
        node = TopicCommentNode(topic="船速", comment="十二节")
        assert node.topic == "船速"
        assert node.comment == "十二节"
        assert not node.has_zero_anaphora

    def test_zero_anaphora(self):
        """零形回指检测"""
        node = TopicCommentNode(topic="", comment="加三")
        assert node.has_zero_anaphora

    def test_not_zero_anaphora_load(self):
        """加载指令不是零形回指"""
        node = TopicCommentNode(topic="", comment="加载为三")
        assert not node.has_zero_anaphora

    def test_classifier(self):
        """量词标注"""
        node = TopicCommentNode(topic="船队", comment="三艘", classifier="艘")
        assert node.classifier == "艘"

    def test_repr(self):
        """字符串表示"""
        node = TopicCommentNode(topic="船速", comment="十二节", classifier="节")
        r = repr(node)
        assert "船速" in r
        assert "十二节" in r
        assert "节" in r


# ═══════════════════════════════════════════════════════════════
# ContinuationTree 测试 — 延续树
# ═══════════════════════════════════════════════════════════════

class TestContinuationTree:
    """延续树构建"""

    def test_empty_tree(self):
        """空延续树生成空块"""
        tree = ContinuationTree()
        blocks = tree.to_basic_blocks()
        assert len(blocks) == 1  # 至少有一个空块

    def test_add_node(self):
        """添加节点"""
        tree = ContinuationTree()
        node = TopicCommentNode(topic="甲", comment="为三")
        tree.add_node(node)
        chain = tree.get_chain()
        assert len(chain) == 1

    def test_zero_anaphora_resolution(self):
        """零形回指自动链接到上一个主题"""
        tree = ContinuationTree()
        tree.add_node(TopicCommentNode(topic="甲", comment="为三"))
        tree.add_node(TopicCommentNode(topic="", comment="加二"))  # 零形回指
        chain = tree.get_chain()
        assert chain[1].topic == "甲"  # 零形回指已解析

    def test_topic_persistence(self):
        """主题持续到被新主题替换"""
        tree = ContinuationTree()
        tree.add_node(TopicCommentNode(topic="甲", comment="为三"))
        tree.add_node(TopicCommentNode(topic="", comment="加二"))
        tree.add_node(TopicCommentNode(topic="", comment="乘二"))
        chain = tree.get_chain()
        assert chain[1].topic == "甲"
        assert chain[2].topic == "甲"

    def test_topic_change(self):
        """主题切换"""
        tree = ContinuationTree()
        tree.add_node(TopicCommentNode(topic="甲", comment="为三"))
        tree.add_node(TopicCommentNode(topic="乙", comment="为四"))
        tree.add_node(TopicCommentNode(topic="", comment="加一"))
        chain = tree.get_chain()
        assert chain[2].topic == "乙"  # 链接到最新的主题


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 测试 — 核心构建器
# ═══════════════════════════════════════════════════════════════

class TestFIRBuilder:
    """FIR 构建器核心功能"""

    def test_create_builder(self):
        """创建构建器"""
        builder = FIRBuilder()
        assert len(builder.blocks) >= 1  # 至少有入口块
        assert builder.current_block is not None

    def test_const_value(self):
        """创建常量值"""
        builder = FIRBuilder()
        val = builder.const(42, name="测试")
        assert val.name == "测试"
        assert val.fir_type == FirType.INTEGER
        assert val.classifier == "个"

    def test_const_with_classifier(self):
        """带量词的常量"""
        builder = FIRBuilder()
        val = builder.const(12, name="船速", classifier="节")
        assert val.fir_type == FirType.SPEED
        assert val.classifier == "节"

    def test_addition(self):
        """加法运算"""
        builder = FIRBuilder()
        a = builder.const(3, name="甲")
        b = builder.const(4, name="乙")
        result = builder.add(a, b, name="和")
        assert result.name == "和"
        assert result.version > 0

    def test_subtraction(self):
        """减法运算"""
        builder = FIRBuilder()
        a = builder.const(10)
        b = builder.const(3)
        result = builder.sub(a, b, name="差")
        assert result.name == "差"

    def test_multiplication(self):
        """乘法运算"""
        builder = FIRBuilder()
        a = builder.const(5)
        b = builder.const(6)
        result = builder.mul(a, b, name="积")
        assert result.name == "积"

    def test_division(self):
        """除法运算"""
        builder = FIRBuilder()
        a = builder.const(20)
        b = builder.const(4)
        result = builder.div(a, b, name="商")
        assert result.name == "商"

    def test_modulo(self):
        """取模运算"""
        builder = FIRBuilder()
        a = builder.const(10)
        b = builder.const(3)
        result = builder.mod(a, b, name="余")
        assert result.name == "余"

    def test_negation(self):
        """取反运算"""
        builder = FIRBuilder()
        a = builder.const(5)
        result = builder.neg(a, name="反")
        assert result.name == "反"

    def test_increment(self):
        """自增运算"""
        builder = FIRBuilder()
        a = builder.const(5, name="计数")
        result = builder.inc(a)
        assert result.name == "计数"

    def test_decrement(self):
        """自减运算"""
        builder = FIRBuilder()
        a = builder.const(5, name="计数")
        result = builder.dec(a)
        assert result.name == "计数"


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 主题寄存器测试
# ═══════════════════════════════════════════════════════════════

class TestTopicRegister:
    """主题寄存器 R63 跟踪"""

    def test_topic_set(self):
        """设置主题"""
        builder = FIRBuilder()
        val = builder.topic_set("船速", 12, "节")
        assert val.is_topic
        assert val.register == 63
        assert val.classifier == "节"
        assert val.fir_type == FirType.SPEED

    def test_topic_get(self):
        """获取主题 (零形回指)"""
        builder = FIRBuilder()
        builder.topic_set("船速", 12, "节")
        zero = builder.topic_get()
        assert zero is not None
        assert zero.is_topic
        assert zero.register == 63

    def test_topic_get_no_topic(self):
        """无主题时返回 None"""
        builder = FIRBuilder()
        zero = builder.topic_get()
        assert zero is None

    def test_topic_stack(self):
        """主题栈管理"""
        builder = FIRBuilder()
        builder.topic_set("甲", 1, "个")
        builder.topic_set("乙", 2, "个")
        assert builder.current_topic is not None
        assert builder.current_topic.name == "乙"

    def test_topic_pop(self):
        """弹出主题"""
        builder = FIRBuilder()
        builder.topic_set("甲", 1, "个")
        builder.topic_set("乙", 2, "个")
        popped = builder.topic_pop()
        assert popped is not None
        assert popped.name == "乙"
        assert builder.current_topic.name == "甲"

    def test_topic_empty_pop(self):
        """空栈弹出返回 None"""
        builder = FIRBuilder()
        result = builder.topic_pop()
        assert result is None


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 量词类型测试
# ═══════════════════════════════════════════════════════════════

class TestClassifierInference:
    """量词类型推断"""

    def test_classify_value(self):
        """量词类型标注"""
        builder = FIRBuilder()
        val = builder.const(5, name="数据")
        typed = builder.classify(val, "艘")
        assert typed.fir_type == FirType.VESSEL
        assert typed.classifier == "艘"

    def test_classify_keeps_name(self):
        """量词标注保留变量名"""
        builder = FIRBuilder()
        val = builder.const(5, name="船")
        typed = builder.classify(val, "艘")
        assert typed.name == "船"

    def test_honorify_upgrade(self):
        """敬称提升"""
        builder = FIRBuilder()
        val = builder.const(5, name="人", classifier="名")
        honored = builder.honorify(val, "尊敬")
        assert honored.classifier == "位"

    def test_honorify_already_honored(self):
        """已经是最高敬称不再升级"""
        builder = FIRBuilder()
        val = builder.const(5, name="人", classifier="位")
        honored = builder.honorify(val, "尊敬")
        assert honored.classifier == "位"


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 控制流测试
# ═══════════════════════════════════════════════════════════════

class TestControlFlow:
    """控制流指令"""

    def test_halt(self):
        """停机指令"""
        builder = FIRBuilder()
        builder.halt()
        assert builder.current_block.terminator == "停机"

    def test_new_block(self):
        """创建新块"""
        builder = FIRBuilder()
        block = builder.new_block("循环体")
        assert block.label == "循环体"

    def test_seal_block(self):
        """封闭块"""
        builder = FIRBuilder()
        builder.seal_block("结束")
        assert builder.current_block.terminator == "结束"

    def test_jump(self):
        """跳转"""
        builder = FIRBuilder()
        builder.new_block("跳转源")
        builder.jump("跳转目标")
        assert builder.current_block.terminator == "跳转"

    def test_branch(self):
        """条件分支"""
        builder = FIRBuilder()
        cond = builder.const(1)
        builder.branch(cond, "真", "假")
        assert builder.current_block.terminator == "分支"

    def test_ret(self):
        """返回"""
        builder = FIRBuilder()
        val = builder.const(42)
        builder.ret(val)
        assert builder.current_block.terminator == "返回"


# ═══════════════════════════════════════════════════════════════
# FIRBuilder Phi 节点测试
# ═══════════════════════════════════════════════════════════════

class TestPhiNodes:
    """Phi 节点创建"""

    def test_phi_creation(self):
        """Phi 节点创建"""
        builder = FIRBuilder()
        v1 = builder.const(1, name="x", classifier="节")
        v2 = builder.const(2, name="x", classifier="节")
        result = builder.phi("x", [(v1, "块A"), (v2, "块B")], classifier="节")
        assert result.name == "x"
        assert len(builder.current_block.phi_nodes) == 1

    def test_phi_auto_classifier(self):
        """Phi 自动推断量词"""
        builder = FIRBuilder()
        v1 = builder.const(1, name="x", classifier="艘")
        v2 = builder.const(2, name="x", classifier="艘")
        result = builder.phi("x", [(v1, "块A"), (v2, "块B")])
        assert result.classifier == "艘"


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 主题-评论解析测试
# ═══════════════════════════════════════════════════════════════

class TestTopicCommentParsing:
    """主题-评论结构解析"""

    def test_parse_load(self):
        """解析加载指令"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("加载 甲 为 三")
        assert node is not None
        assert node.topic == "甲"
        assert "三" in node.comment

    def test_parse_arithmetic(self):
        """解析算术运算"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("计算 三加四")
        assert node is not None
        assert node is not None  # 模式匹配成功
        assert "加" in node.comment or "四" in node.comment

    def test_parse_agent_tell(self):
        """解析智能体告知"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("告诉 甲 准备出航")
        assert node is not None
        assert node.topic == "甲"
        assert node.classifier == "位"

    def test_parse_factorial(self):
        """解析阶乘"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("五的阶乘")
        assert node is not None
        assert node.topic == "五"
        assert "阶乘" in node.comment

    def test_parse_comment_line(self):
        """注释行返回 None"""
        builder = FIRBuilder()
        assert builder.parse_topic_comment("-- 这是注释") is None
        assert builder.parse_topic_comment("// 这也是注释") is None
        assert builder.parse_topic_comment("# 还是注释") is None

    def test_parse_empty_line(self):
        """空行返回 None"""
        builder = FIRBuilder()
        assert builder.parse_topic_comment("") is None

    def test_parse_increment(self):
        """解析增加指令"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("增加 甲")
        assert node is not None

    def test_parse_decrement(self):
        """解析减少指令"""
        builder = FIRBuilder()
        node = builder.parse_topic_comment("减少 乙")
        assert node is not None


# ═══════════════════════════════════════════════════════════════
# FIRBuilder 序列化测试
# ═══════════════════════════════════════════════════════════════

class TestSerialization:
    """FIR 序列化为 FLUX 汇编"""

    def test_const_to_asm(self):
        """常量序列化"""
        builder = FIRBuilder()
        builder.const(42, name="答案")
        builder.halt()
        asm = builder.to_assembly()
        assert "MOVI" in asm
        assert "42" in asm

    def test_add_to_asm(self):
        """加法序列化"""
        builder = FIRBuilder()
        a = builder.const(3)
        b = builder.const(4)
        builder.add(a, b)
        builder.halt()
        asm = builder.to_assembly()
        assert "IADD" in asm

    def test_topic_set_to_asm(self):
        """主题设置序列化"""
        builder = FIRBuilder()
        builder.topic_set("船速", 12, "节")
        builder.halt()
        asm = builder.to_assembly()
        assert "R63" in asm

    def test_halt_in_asm(self):
        """停机序列化"""
        builder = FIRBuilder()
        builder.halt()
        asm = builder.to_assembly()
        assert "HALT" in asm


# ═══════════════════════════════════════════════════════════════
# FIRBuilder dump 测试
# ═══════════════════════════════════════════════════════════════

class TestDump:
    """FIR 调试输出"""

    def test_dump_not_empty(self):
        """dump 输出非空"""
        builder = FIRBuilder()
        builder.const(42)
        builder.halt()
        dump = builder.dump()
        assert len(dump) > 0
        assert "FIR" in dump

    def test_dump_contains_block_info(self):
        """dump 包含块信息"""
        builder = FIRBuilder()
        builder.const(42)
        builder.halt()
        dump = builder.dump()
        assert "基本块" in dump


# ═══════════════════════════════════════════════════════════════
# FIRProgram 测试
# ═══════════════════════════════════════════════════════════════

class TestFIRProgram:
    """完整 FIR 程序"""

    def test_create_program(self):
        """创建 FIR 程序"""
        program = FIRProgram(name="测试程序")
        assert program.name == "测试程序"
        assert len(program.blocks) >= 1

    def test_program_to_asm(self):
        """程序序列化为汇编"""
        program = FIRProgram(name="加法")
        program.builder.const(3)
        program.builder.const(4)
        program.builder.add(
            program.builder.get_latest("常量") or program.builder.const(0),
            program.builder.const(0),
            name="和",
        )
        program.builder.halt()
        asm = program.to_assembly()
        assert "MOVI" in asm

    def test_program_dump(self):
        """程序调试输出"""
        program = FIRProgram(name="测试")
        program.builder.const(42)
        program.builder.halt()
        dump = program.dump()
        assert "测试" in dump

    def test_program_repr(self):
        """程序字符串表示"""
        program = FIRProgram(name="示例")
        r = repr(program)
        assert "示例" in r


# ═══════════════════════════════════════════════════════════════
# build_from_chinese 测试
# ═══════════════════════════════════════════════════════════════

class TestBuildFromChinese:
    """从中文源代码构建 FIR"""

    def test_simple_program(self):
        """简单中文程序"""
        program = build_from_chinese("加载 甲 为 三\n停机", "简单程序")
        assert program.name == "简单程序"
        assert len(program.blocks) >= 1

    def test_arithmetic_program(self):
        """算术中文程序"""
        program = build_from_chinese("计算 三加四\n停机", "加法")
        assert program is not None
        assert program.name == "加法"
        assert len(program.blocks) >= 1

    def test_multi_line_with_comments(self):
        """带注释的多行程序"""
        source = """
-- 这是注释
加载 甲 为 十
加载 乙 为 二十
# 也是注释
停机
"""
        program = build_from_chinese(source, "注释测试")
        assert len(program.blocks) >= 1


# ═══════════════════════════════════════════════════════════════
# 敬称/语境检测测试
# ═══════════════════════════════════════════════════════════════

class TestHonorificDetection:
    """敬称检测"""

    def test_detect_ge_xia(self):
        """检测'阁下'"""
        assert detect_honorific("阁下请指示") == "尊敬"

    def test_detect_gui_fang(self):
        """检测'贵方'"""
        assert detect_honorific("贵方船速") == "尊敬"

    def test_detect_xian_sheng(self):
        """检测'先生'"""
        assert detect_honorific("张先生") == "礼貌"

    def test_no_honorific(self):
        """无敬称"""
        assert detect_honorific("你好世界") == ""


class TestContextDetection:
    """语境检测"""

    def test_detect_classical(self):
        """检测文言"""
        assert detect_context("文言风格") == "文言"

    def test_detect_ancient(self):
        """检测古语"""
        assert detect_context("古语") == "文言"

    def test_detect_colloquial(self):
        """检测口语"""
        assert detect_context("口语表达") == "口语"

    def test_no_context(self):
        """无语境"""
        assert detect_context("普通文本") == ""


# ═══════════════════════════════════════════════════════════════
# 辅助函数测试
# ═══════════════════════════════════════════════════════════════

class TestHelperFunctions:
    """辅助函数"""

    def test_classifier_upgrade_basic(self):
        """量词基本升级"""
        assert classifier_upgrade("个", "尊敬") == "位"
        assert classifier_upgrade("名", "尊敬") == "位"

    def test_classifier_upgrade_already_highest(self):
        """已是最高级不升级"""
        assert classifier_upgrade("位", "尊敬") == "位"

    def test_classifier_upgrade_polite(self):
        """礼貌级别升级"""
        assert classifier_upgrade("个", "礼貌") == "名"
        assert classifier_upgrade("名", "礼貌") == "名"

    def test_classifier_upgrade_no_change(self):
        """专业级别不改变量词"""
        assert classifier_upgrade("艘", "专业") == "艘"

    def test_tiangan_registers(self):
        """天干寄存器映射"""
        assert TIANGAN_REGISTERS["甲"] == 0
        assert TIANGAN_REGISTERS["乙"] == 1
        assert TIANGAN_REGISTERS["癸"] == 9

    def test_full_registers(self):
        """完整寄存器映射"""
        assert ZH_REGISTERS_FULL["寄存器零"] == 0
        assert ZH_REGISTERS_FULL["甲"] == 0


# ═══════════════════════════════════════════════════════════════
# FIROpcode 测试
# ═══════════════════════════════════════════════════════════════

class TestFIROpcode:
    """FIR 操作码系统"""

    def test_opcode_names_complete(self):
        """所有操作码都有中文名"""
        for op in FirOpcode:
            assert op in FIR_OPCODE_NAMES, f"操作码 {op} 缺少中文名"

    def test_unique_values(self):
        """操作码值唯一"""
        values = [op.value for op in FirOpcode]
        assert len(values) == len(set(values))

    def test_chinese_topic_ops(self):
        """中文特定操作码存在"""
        assert FirOpcode.TOPIC_SET in FirOpcode
        assert FirOpcode.TOPIC_GET in FirOpcode
        assert FirOpcode.CLASSIFY in FirOpcode
        assert FirOpcode.HONORIFY in FirOpcode
