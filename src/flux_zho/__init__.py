"""
流星运行时 — 流体语言通用执行 (Flux Runtime — Chinese First)

中文优先的自然语言编程运行时。基于主题-评论结构，
以量词系统为类型基础，零形回指实现上下文传递。

设计哲学:
  - 主题-评论 (Topic-Comment) 是默认结构
  - 量词 (Classifier) 构成类型层次
  - 零形回指 (Zero Anaphora) 实现隐式参数传递
  - 词汇即类型系统

使用方法:
    flux-zho 你好          # 运行演示
    flux-zho 编译 "计算 三加四"   # 编译中文为字节码
    flux-zho 运行 "计算 三加四"   # 执行并显示结果
"""

__version__ = "0.1.0"
__author__ = "流星团队"
