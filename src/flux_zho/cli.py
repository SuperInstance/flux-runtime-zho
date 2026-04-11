"""
流星命令行界面 — 中文优先的CLI体验

命令:
  你好     — 运行演示（计算三加四）
  编译     — 编译中文为字节码
  运行     — 编译并执行
  解释     — 反汇编字节码
  反汇编   — 反汇编十六进制
  开放     — 启动REPL交互模式
"""

import sys
import argparse
from flux_zho import __version__
from flux_zho.interpreter import (
    compile_chinese,
    compile_and_execute,
    compile_program,
    disassemble,
    format_assembly,
    format_bytecode_hex,
    FluxVM,
    quick_encode,
)


BANNER = r"""
╔══════════════════════════════════════════════╗
║  流星 · 流体语言通用执行                      ║
║  Liúxīng · Flux Runtime (中文优先)            ║
║  版本 {version}                              ║
╚══════════════════════════════════════════════╝
""".format(version=__version__)


def _print_result(comp, exec_result, verbose=False):
    """格式化输出编译和执行结果"""
    if not comp.success:
        print(f"  ❌ 编译失败: {comp.error}")
        return

    print(f"  ✅ 模式匹配: {comp.pattern_name}")
    print(f"  📝 汇编代码:")
    for line in comp.assembly.strip().split("\n"):
        print(f"     {line}")
    print(f"  📦 字节码: {format_bytecode_hex(comp.bytecode)}")

    if exec_result:
        print(f"  ⚙️  执行周期: {exec_result.cycles}")
        if exec_result.error:
            print(f"  ❌ 执行错误: {exec_result.error}")
        else:
            print(f"  📊 结果寄存器 R0 = {exec_result.result}")
        if verbose and exec_result.trace:
            print(f"  🔍 执行追踪:")
            for t in exec_result.trace:
                print(f"     {t}")


def cmd_nihao(_args):
    """
    演示命令 — 你好世界！
    展示中文自然语言编程的核心能力。
    """
    print(BANNER)
    print("  你好！欢迎来到流星运行时 ✨\n")
    print("  ── 演示：中文自然语言 → 字节码 → 执行 ──\n")

    demos = [
        ("加法", "计算 三加四"),
        ("乘法", "计算 五乘六"),
        ("减法", "计算 十减三"),
        ("阶乘", "五的阶乘"),
        ("寄存器赋值", "加载 寄存器零 为 四十二"),
    ]

    for name, code in demos:
        print(f"  【{name}】 {code}")
        comp, exec_result = compile_and_execute(code)
        print(f"     → R0 = {exec_result.result}")
        if not comp.success:
            print(f"     → 错误: {comp.error}")
        print()

    print("  ── 多行程序示例 ──\n")
    program = """
加载 寄存器零 为 三
加载 寄存器一 为 七
计算 寄存器零 加 寄存器一
"""
    print(f"  程序代码:{program}")
    comp, exec_result = compile_program(program)
    print(f"  编译结果:")
    print(f"    {comp.assembly.strip()}")
    print(f"  执行结果: R0 = {exec_result.result}")
    print(f"  执行周期: {exec_result.cycles}")

    print("\n  ── 输入 'flux-zho 开放' 进入交互模式 ──")


def cmd_compile(args):
    """编译中文自然语言为字节码"""
    text = args.code
    if not text:
        print("  ❌ 请提供要编译的代码: flux-zho 编译 '计算 三加四'")
        return

    comp, _ = compile_and_execute(text)
    if args.hex:
        print(format_bytecode_hex(comp.bytecode))
    else:
        _print_result(comp, None)


def cmd_run(args):
    """编译并执行中文自然语言"""
    text = args.code
    if not text:
        print("  ❌ 请提供要执行的代码: flux-zho 运行 '计算 三加四'")
        return

    comp, exec_result = compile_and_execute(text, trace=args.trace)
    if args.quiet:
        print(exec_result.result)
    else:
        _print_result(comp, exec_result, verbose=args.verbose)


def cmd_explain(args):
    """反汇编字节码"""
    text = args.code
    if not text:
        print("  ❌ 请提供要解释的代码")
        return

    comp, _ = compile_and_execute(text)
    if not comp.success:
        print(f"  ❌ 编译失败: {comp.error}")
        return

    instructions = disassemble(comp.bytecode)
    print("  ══ 反汇编输出 ══")
    print(format_assembly(instructions))
    print(f"\n  字节码: {format_bytecode_hex(comp.bytecode)}")


def cmd_disasm(args):
    """反汇编十六进制字节码"""
    hex_str = args.hexcode.strip()
    if not hex_str:
        print("  ❌ 请提供十六进制字节码")
        return

    try:
        bytecode = bytes(int(b, 16) for b in hex_str.split())
    except ValueError as e:
        print(f"  ❌ 无效的十六进制: {e}")
        return

    instructions = disassemble(bytecode)
    print("  ══ 反汇编输出 ══")
    print(format_assembly(instructions))


def cmd_open(args):
    """REPL交互模式 — 开放"""
    print(BANNER)
    print("  交互模式已启动。输入中文代码，按回车执行。")
    print("  输入 '退出' 或 'quit' 结束。\n")

    while True:
        try:
            line = input("  流星> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not line:
            continue
        if line in ("退出", "quit", "exit", "再见"):
            print("  再见！")
            break
        if line == "帮助" or line == "help":
            print("  可用示例:")
            print("    计算 三加四          — 加法")
            print("    计算 五乘六          — 乘法")
            print("    五的阶乘             — 阶乘")
            print("    加载 寄存器零 为 十   — 赋值")
            print("    告诉 甲 你好世界      — 智能体通信")
            print("    帮助                  — 显示此帮助")
            print("    退出                  — 结束")
            continue

        comp, exec_result = compile_and_execute(line, trace=args.trace)
        if comp.success:
            print(f"  → R0 = {exec_result.result}  (周期: {exec_result.cycles})")
            if exec_result.error:
                print(f"  → ⚠️  {exec_result.error}")
        else:
            print(f"  ❌ {comp.error}")


def main():
    """流星CLI主入口"""
    parser = argparse.ArgumentParser(
        prog="flux-zho",
        description="流星运行时 — 主题优先的中文自然语言编程",
        epilog="示例: flux-zho 你好 | flux-zho 运行 '计算 三加四'",
    )
    parser.add_argument("-v", "--version", action="version", version=f"流星运行时 v{__version__}")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 你好
    sub_nihao = subparsers.add_parser("你好", help="运行演示")
    sub_nihao.set_defaults(func=cmd_nihao)

    # 编译
    sub_compile = subparsers.add_parser("编译", help="编译中文为字节码")
    sub_compile.add_argument("code", nargs="?", help="要编译的中文代码")
    sub_compile.add_argument("--hex", action="store_true", help="仅输出十六进制字节码")
    sub_compile.set_defaults(func=cmd_compile)

    # 运行
    sub_run = subparsers.add_parser("运行", help="编译并执行中文代码")
    sub_run.add_argument("code", nargs="?", help="要执行的中文代码")
    sub_run.add_argument("--trace", action="store_true", help="启用执行追踪")
    sub_run.add_argument("--verbose", action="store_true", help="详细输出")
    sub_run.add_argument("--quiet", "-q", action="store_true", help="仅输出结果")
    sub_run.set_defaults(func=cmd_run)

    # 解释
    sub_explain = subparsers.add_parser("解释", help="反汇编字节码")
    sub_explain.add_argument("code", nargs="?", help="要反汇编的中文代码")
    sub_explain.set_defaults(func=cmd_explain)

    # 反汇编
    sub_disasm = subparsers.add_parser("反汇编", help="反汇编十六进制字节码")
    sub_disasm.add_argument("hexcode", nargs="?", help="十六进制字节码字符串")
    sub_disasm.set_defaults(func=cmd_disasm)

    # 开放 (REPL)
    sub_open = subparsers.add_parser("开放", help="启动交互模式")
    sub_open.add_argument("--trace", action="store_true", help="启用执行追踪")
    sub_open.set_defaults(func=cmd_open)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        # 无参数时显示帮助
        print(BANNER)
        parser.print_help()
        print("\n  快速开始: flux-zho 你好")


if __name__ == "__main__":
    main()
