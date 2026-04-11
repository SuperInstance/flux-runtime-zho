"""
流星运行时 — 量词类型求解器 (ClassifierTypeSolver)

量词 (Classifier) 是中文最美丽的类型系统——每个名词必须通过量词声明其类别。
ClassifierTypeSolver 将这一语法特性转化为类型约束系统：

    量词 → 类型约束 → 编译时/运行时验证

设计哲学:
  - 只 → AnimalType: 量词只约束名词为动物类别
  - 本 → BookType: 量词本约束名词为书籍/文档
  - 台 → MachineType: 量词台约束名词为机器/设备
  - 位 → PersonType(respectful): 敬称量词位约束为人(敬称)
  - 个 → AnyType: 通用量词个不约束类型
  - 语境回退: "三个工程师" — 个是通用量词，但工程师暗示PersonType

用法:
    solver = ClassifierTypeSolver()
    solver.resolve("三只猫")  # → ("猫", AnimalType, "只")
    solver.resolve("五位客人")  # → ("客人", PersonType, "位")
    solver.resolve("一台电脑")  # → ("电脑", MachineType, "台")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# 类型层次枚举
# ══════════════════════════════════════════════════════════════════════

class ClassifierType(IntEnum):
    """量词类型层次 — 每个量词对应一个类型约束"""
    ANY = 0             # 个 → 无约束
    ANIMAL = 1          # 只 → 动物/小件
    PERSON = 2          # 名 → 人(中性)
    PERSON_RESPECT = 3  # 位 → 人(敬称)
    BOOK = 4            # 本 → 书籍/文档
    MACHINE = 5         # 台 → 机器/设备
    SET = 6             # 套 → 套装/组合
    VESSEL = 7          # 艘 → 船只
    LONG = 8            # 条 → 长形/序列/河流/消息
    FLAT = 9            # 张 → 平面/纸张/桌面
    PIECE = 10          # 块 → 块状/土地
    STICK = 11          # 根 → 细长/棍子/绳子
    ROUND = 12          # 颗 → 小圆/珠子/星星
    GRAIN = 13          # 粒 → 颗粒/种子
    DROP = 14           # 滴 → 液体
    COUNTER = 15        # 次/遍/趟 → 次数/动作
    STEP = 16           # 步 → 步骤/流程
    KIND = 17           # 种 → 种类/类别
    ITEM = 18           # 件 → 事务/衣物/物品
    PLANE = 19          # 片 → 平面/土地/天空
    SPEED = 20          # 节 → 速度(航速)
    DISTANCE = 21       # 海里 → 距离
    WEIGHT = 22         # 吨 → 重量
    TIME = 23           # 秒/小时 → 时间
    MESSAGE = 24        # 封/条信息 → 消息/信件
    REPORT = 25         # 份 → 报告/文件
    CLAUSE = 26         # 项 → 条款/项目
    CONTAINER = 27      # 杯/瓶/桶/箱 → 容器
    AIRCRAFT = 28       # 架 → 飞机/框架
    ANCHOR = 29         # 锚 → 锚
    SAIL = 30           # 帆 → 帆
    ROUND_TRIP = 31     # 轮 → 轮次/回合
    FRAMEWORK = 32      # 架 → 框架/骨架


# 类型中文名
CLASSIFIER_TYPE_NAMES: dict[ClassifierType, str] = {
    ClassifierType.ANY: "任意",
    ClassifierType.ANIMAL: "动物",
    ClassifierType.PERSON: "人员",
    ClassifierType.PERSON_RESPECT: "人员(敬称)",
    ClassifierType.BOOK: "书籍",
    ClassifierType.MACHINE: "机器",
    ClassifierType.SET: "套装",
    ClassifierType.VESSEL: "船只",
    ClassifierType.LONG: "长形/序列",
    ClassifierType.FLAT: "平面",
    ClassifierType.PIECE: "块状",
    ClassifierType.STICK: "细长",
    ClassifierType.ROUND: "小圆",
    ClassifierType.GRAIN: "颗粒",
    ClassifierType.DROP: "液体",
    ClassifierType.COUNTER: "计数",
    ClassifierType.STEP: "步骤",
    ClassifierType.KIND: "种类",
    ClassifierType.ITEM: "事务",
    ClassifierType.PLANE: "平面",
    ClassifierType.SPEED: "速度",
    ClassifierType.DISTANCE: "距离",
    ClassifierType.WEIGHT: "重量",
    ClassifierType.TIME: "时间",
    ClassifierType.MESSAGE: "消息",
    ClassifierType.REPORT: "报告",
    ClassifierType.CLAUSE: "条款",
    ClassifierType.CONTAINER: "容器",
    ClassifierType.AIRCRAFT: "飞机",
    ClassifierType.ANCHOR: "锚",
    ClassifierType.SAIL: "帆",
    ClassifierType.ROUND_TRIP: "轮次",
    ClassifierType.FRAMEWORK: "框架",
}


# ══════════════════════════════════════════════════════════════════════
# 量词 → 类型映射表 (30+ 量词)
# ══════════════════════════════════════════════════════════════════════

CLASSIFIER_TO_TYPE: dict[str, ClassifierType] = {
    # 通用量词
    "个": ClassifierType.ANY,
    "只": ClassifierType.ANIMAL,
    "条": ClassifierType.LONG,
    "本": ClassifierType.BOOK,
    "台": ClassifierType.MACHINE,
    "位": ClassifierType.PERSON_RESPECT,
    "名": ClassifierType.PERSON,
    "件": ClassifierType.ITEM,
    "种": ClassifierType.KIND,
    "套": ClassifierType.SET,
    # 形状量词
    "张": ClassifierType.FLAT,
    "块": ClassifierType.PIECE,
    "片": ClassifierType.PLANE,
    "颗": ClassifierType.ROUND,
    "根": ClassifierType.STICK,
    "粒": ClassifierType.GRAIN,
    "滴": ClassifierType.DROP,
    # 数字/动作量词
    "次": ClassifierType.COUNTER,
    "遍": ClassifierType.COUNTER,
    "趟": ClassifierType.COUNTER,
    "步": ClassifierType.STEP,
    "轮": ClassifierType.ROUND_TRIP,
    # 航海量词
    "艘": ClassifierType.VESSEL,
    "节": ClassifierType.SPEED,
    "海里": ClassifierType.DISTANCE,
    "锚": ClassifierType.ANCHOR,
    "帆": ClassifierType.SAIL,
    "吨": ClassifierType.WEIGHT,
    # 信息量词
    "封": ClassifierType.MESSAGE,
    "条信息": ClassifierType.MESSAGE,
    "则": ClassifierType.MESSAGE,
    "份": ClassifierType.REPORT,
    "项": ClassifierType.CLAUSE,
    # 时间量词
    "秒": ClassifierType.TIME,
    "小时": ClassifierType.TIME,
    # 容器量词
    "杯": ClassifierType.CONTAINER,
    "瓶": ClassifierType.CONTAINER,
    "桶": ClassifierType.CONTAINER,
    "箱": ClassifierType.CONTAINER,
    # 机器量词
    "架": ClassifierType.AIRCRAFT,
    # 其他
    "群": ClassifierType.GROUP if False else ClassifierType.ANY,  # fallback
}

# 覆盖 fallback
CLASSIFIER_TO_TYPE["群"] = ClassifierType.ANY


# ══════════════════════════════════════════════════════════════════════
# 名词 → 隐含类型表 (语境回退)
# ══════════════════════════════════════════════════════════════════════

NOUN_IMPLIED_TYPE: dict[str, ClassifierType] = {
    # 动物
    "猫": ClassifierType.ANIMAL,
    "狗": ClassifierType.ANIMAL,
    "鸟": ClassifierType.ANIMAL,
    "鱼": ClassifierType.ANIMAL,
    "虎": ClassifierType.ANIMAL,
    "龙": ClassifierType.ANIMAL,
    "马": ClassifierType.ANIMAL,
    "牛": ClassifierType.ANIMAL,
    "羊": ClassifierType.ANIMAL,
    "鸡": ClassifierType.ANIMAL,
    "兔": ClassifierType.ANIMAL,
    "蛇": ClassifierType.ANIMAL,
    "鼠": ClassifierType.ANIMAL,
    "象": ClassifierType.ANIMAL,
    "鹿": ClassifierType.ANIMAL,
    "狼": ClassifierType.ANIMAL,
    # 人
    "人": ClassifierType.PERSON,
    "先生": ClassifierType.PERSON_RESPECT,
    "女士": ClassifierType.PERSON_RESPECT,
    "老师": ClassifierType.PERSON_RESPECT,
    "教授": ClassifierType.PERSON_RESPECT,
    "医生": ClassifierType.PERSON_RESPECT,
    "工程师": ClassifierType.PERSON,
    "学生": ClassifierType.PERSON,
    "员工": ClassifierType.PERSON,
    "水手": ClassifierType.PERSON,
    "船员": ClassifierType.PERSON,
    "客人": ClassifierType.PERSON,
    "朋友": ClassifierType.PERSON,
    "孩子": ClassifierType.PERSON,
    "孩子": ClassifierType.PERSON,
    "男孩": ClassifierType.PERSON,
    "女孩": ClassifierType.PERSON,
    "男人": ClassifierType.PERSON,
    "女人": ClassifierType.PERSON,
    # 机器
    "电脑": ClassifierType.MACHINE,
    "计算机": ClassifierType.MACHINE,
    "机器": ClassifierType.MACHINE,
    "设备": ClassifierType.MACHINE,
    "发动机": ClassifierType.MACHINE,
    "雷达": ClassifierType.MACHINE,
    "汽车": ClassifierType.MACHINE,
    "手机": ClassifierType.MACHINE,
    "相机": ClassifierType.MACHINE,
    "电视": ClassifierType.MACHINE,
    "冰箱": ClassifierType.MACHINE,
    "空调": ClassifierType.MACHINE,
    "洗衣机": ClassifierType.MACHINE,
    # 书籍
    "书": ClassifierType.BOOK,
    "文件": ClassifierType.BOOK,
    "册": ClassifierType.BOOK,
    "手册": ClassifierType.BOOK,
    "日志": ClassifierType.BOOK,
    "杂志": ClassifierType.BOOK,
    "报纸": ClassifierType.BOOK,
    "词典": ClassifierType.BOOK,
    # 船只
    "船": ClassifierType.VESSEL,
    "舰": ClassifierType.VESSEL,
    "艇": ClassifierType.VESSEL,
    "航母": ClassifierType.VESSEL,
    "驱逐舰": ClassifierType.VESSEL,
    "潜艇": ClassifierType.VESSEL,
    "渔船": ClassifierType.VESSEL,
    "货船": ClassifierType.VESSEL,
    # 其他
    "消息": ClassifierType.LONG,
    "指令": ClassifierType.LONG,
    "河": ClassifierType.LONG,
    "路": ClassifierType.LONG,
    "纸": ClassifierType.FLAT,
    "桌子": ClassifierType.FLAT,
    "地": ClassifierType.PIECE,
    "山": ClassifierType.PIECE,
    "钱": ClassifierType.ROUND,
    "星": ClassifierType.ROUND,
    "水": ClassifierType.DROP,
    "时间": ClassifierType.TIME,
    "速度": ClassifierType.SPEED,
    "距离": ClassifierType.DISTANCE,
    "重量": ClassifierType.WEIGHT,
}


# ══════════════════════════════════════════════════════════════════════
# 量词 → 推荐名词表
# ══════════════════════════════════════════════════════════════════════

CLASSIFIER_NOUN_DB: dict[str, list[str]] = {
    "只": ["猫", "狗", "鸟", "虎", "龙", "马", "牛", "羊", "鸡", "兔",
           "蛇", "鼠", "象", "鹿", "狼", "手", "眼睛", "耳朵", "脚"],
    "本": ["书", "文件", "册", "手册", "日志", "杂志", "词典", "笔记"],
    "台": ["机器", "计算机", "电脑", "设备", "发动机", "雷达", "汽车", "手机",
           "相机", "电视", "冰箱", "空调", "洗衣机", "打印机"],
    "位": ["人", "先生", "女士", "老师", "教授", "医生", "工程师", "专家",
           "贵宾", "客人", "领导"],
    "名": ["人", "学生", "员工", "水手", "船员", "工人", "农民", "教师"],
    "条": ["河", "鱼", "消息", "指令", "船", "路", "新闻", "规则", "法律"],
    "艘": ["船", "舰", "艇", "航母", "驱逐舰", "潜艇", "渔船", "货船"],
    "张": ["纸", "桌子", "椅子", "床", "票", "照片", "地图", "脸"],
    "本": ["书", "文件", "册", "手册", "日志"],
    "件": ["衣服", "衬衫", "裤子", "外套", "大衣", "礼物", "事情", "案件"],
    "封": ["信", "邮件", "电报", "贺卡", "推荐信"],
    "份": ["文件", "报告", "清单", "计划", "合同", "菜单", "报纸"],
    "颗": ["星", "珠子", "宝石", "钻石", "种子", "子弹", "糖", "药"],
    "根": ["棍子", "绳子", "针", "筷子", "头发", "草", "竹子", "线"],
    "滴": ["水", "油", "血", "泪", "汗", "雨", "墨水"],
    "杯": ["水", "茶", "咖啡", "酒", "牛奶", "果汁"],
    "瓶": ["水", "酒", "油", "药", "香水", "牛奶", "可乐"],
    "台": ["机器", "电脑", "设备"],
    "套": ["衣服", "家具", "设备", "工具", "书籍"],
    "节": ["课", "火车", "视频", "电池"],
    "次": ["循环", "计算", "尝试", "实验", "考试", "会议", "旅行"],
    "步": ["步骤", "流程", "棋", "距离", "计划"],
    "吨": ["货物", "排水量", "钢材", "粮食", "煤炭"],
    "海里": ["距离", "航程", "航线", "范围"],
    "秒": ["时间", "等待", "持续"],
    "小时": ["时间", "课程", "会议", "旅行", "等待"],
}


# ══════════════════════════════════════════════════════════════════════
# 数字 → 阿拉伯数字映射
# ══════════════════════════════════════════════════════════════════════

_ZH_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}

_ZH_FULLWIDTH = "０１２３４５６７８９"


def _parse_zh_number(text: str) -> int:
    """简化版中文数字解析"""
    text = text.strip()
    if re.match(r'^-?\d+$', text):
        return int(text)
    result = 0
    current = 0
    temp = 0
    for char in text:
        if char in _ZH_FULLWIDTH:
            temp = _ZH_FULLWIDTH.index(char)
            current += temp
            continue
        if char not in _ZH_DIGITS:
            continue
        val = _ZH_DIGITS[char]
        if val < 10:
            temp = val
        elif val == 10:
            current += (temp if temp > 0 else 1) * 10
            temp = 0
        elif val == 100:
            current += (temp if temp > 0 else 1) * 100
            temp = 0
        elif val == 1000:
            current += (temp if temp > 0 else 1) * 1000
            temp = 0
        elif val == 10000:
            result += (current + temp) * 10000
            current = 0
            temp = 0
    result += current + temp
    return result


# ══════════════════════════════════════════════════════════════════════
# 量词解析模式
# ══════════════════════════════════════════════════════════════════════

# 匹配 "数字+量词+名词" 模式
# 例如: "三只猫", "五位客人", "一台电脑"
_CLASSIFIER_PATTERN = re.compile(
    r"(\S+?)(只|条|本|台|位|名|件|种|套|张|块|片|颗|根|粒|滴"
    r"|次|遍|趟|步|轮|艘|节|海里|锚|帆|吨"
    r"|封|份|项|则|条信息|秒|小时"
    r"|杯|瓶|桶|箱|架|群)"
    r"(\S+)"
)


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ClassifierResolution:
    """量词类型解析结果"""
    noun: str              # 名词
    classifier: str        # 使用的量词
    classifier_type: ClassifierType  # 量词对应的类型
    count: int = 1         # 数量
    confidence: float = 1.0  # 类型置信度 (1.0 = 确定, <1.0 = 回退)
    source: str = "量词"   # 来源: "量词" 或 "语境回退"

    @property
    def type_name(self) -> str:
        return CLASSIFIER_TYPE_NAMES.get(self.classifier_type, "未知")

    def __repr__(self) -> str:
        source_mark = f" [{self.source}]" if self.source != "量词" else ""
        return (
            f"ClassifierResolution({self.count}{self.classifier}{self.noun}"
            f": {self.type_name}, conf={self.confidence:.1f}{source_mark})"
        )


# ══════════════════════════════════════════════════════════════════════
# ClassifierTypeSolver — 量词类型求解器
# ══════════════════════════════════════════════════════════════════════

class ClassifierTypeSolver:
    """
    量词类型求解器 — 将中文量词语法转化为类型约束。

    核心算法:
      1. 解析 "数字 + 量词 + 名词" 模式
      2. 量词 → 类型约束 (主路径)
      3. 名词 → 隐含类型 (回退路径)
      4. 类型兼容性验证

    示例:
        >>> solver = ClassifierTypeSolver()
        >>> solver.resolve("三只猫")
        ClassifierResolution(3只猫: 动物, conf=1.0)
        >>> solver.resolve("五个工程师")
        ClassifierResolution(5个工程师: 人员, conf=0.7)
        >>> solver.validate("三只猫", "只", "猫")
        True
        >>> solver.validate("三本猫", "本", "猫")
        False  # 本用于书籍,不用于动物
    """

    # 类型兼容性矩阵: 哪些类型可以互相替代
    _TYPE_COMPATIBLE: dict[ClassifierType, set[ClassifierType]] = {
        ClassifierType.ANY: {ClassifierType.ANY},
        ClassifierType.ANIMAL: {ClassifierType.ANIMAL},
        ClassifierType.PERSON: {ClassifierType.PERSON, ClassifierType.PERSON_RESPECT},
        ClassifierType.PERSON_RESPECT: {ClassifierType.PERSON, ClassifierType.PERSON_RESPECT},
        ClassifierType.BOOK: {ClassifierType.BOOK},
        ClassifierType.MACHINE: {ClassifierType.MACHINE, ClassifierType.SET},
        ClassifierType.SET: {ClassifierType.MACHINE, ClassifierType.SET},
        ClassifierType.VESSEL: {ClassifierType.VESSEL},
        ClassifierType.LONG: {ClassifierType.LONG},
        ClassifierType.FLAT: {ClassifierType.FLAT},
        ClassifierType.PIECE: {ClassifierType.PIECE, ClassifierType.PLANE},
        ClassifierType.PLANE: {ClassifierType.PIECE, ClassifierType.PLANE},
        ClassifierType.COUNTER: {ClassifierType.COUNTER},
        ClassifierType.STEP: {ClassifierType.STEP},
        ClassifierType.MESSAGE: {ClassifierType.MESSAGE, ClassifierType.LONG},
        ClassifierType.REPORT: {ClassifierType.REPORT, ClassifierType.BOOK},
    }

    # 回退置信度
    FALLBACK_CONFIDENCE = 0.7

    def __init__(self):
        self._type_registry: dict[str, ClassifierType] = {}
        self._resolution_log: list[ClassifierResolution] = []

    def resolve(self, text: str) -> ClassifierResolution:
        """
        解析中文文本中的量词类型约束。

        支持格式:
          - "三只猫" → (猫, 只, AnimalType, 3)
          - "五位客人" → (客人, 位, PersonType(respect), 5)
          - "一台电脑" → (电脑, 台, MachineType, 1)
          - "工程师" → (工程师, 个, PersonType, 1, fallback)

        Args:
            text: 中文文本 (可包含数字+量词+名词)

        Returns:
            ClassifierResolution 解析结果
        """
        text = text.strip()

        # 尝试匹配 "数字+量词+名词" 模式
        match = _CLASSIFIER_PATTERN.search(text)
        if match:
            count_str = match.group(1)
            classifier = match.group(2)
            noun = match.group(3)

            # 解析数量
            count = _parse_zh_number(count_str)

            # 量词 → 类型
            ctype = CLASSIFIER_TO_TYPE.get(classifier, ClassifierType.ANY)

            resolution = ClassifierResolution(
                noun=noun,
                classifier=classifier,
                classifier_type=ctype,
                count=count,
                confidence=1.0,
                source="量词",
            )
            self._resolution_log.append(resolution)
            return resolution

        # 无量词 — 尝试从名词推断
        # 去除数字前缀
        noun_only = re.sub(r'^[零一二两三四五六七八九十百千万\d]+', '', text).strip()
        if noun_only:
            implied_type = NOUN_IMPLIED_TYPE.get(noun_only, ClassifierType.ANY)
            if implied_type != ClassifierType.ANY:
                resolution = ClassifierResolution(
                    noun=noun_only,
                    classifier="个",
                    classifier_type=implied_type,
                    count=1,
                    confidence=self.FALLBACK_CONFIDENCE,
                    source="语境回退",
                )
                self._resolution_log.append(resolution)
                return resolution

        # 完全无法解析
        resolution = ClassifierResolution(
            noun=text,
            classifier="个",
            classifier_type=ClassifierType.ANY,
            count=1,
            confidence=0.0,
            source="未知",
        )
        self._resolution_log.append(resolution)
        return resolution

    def validate(self, number_text: str, classifier: str, noun: str) -> bool:
        """
        验证量词-名词搭配是否合法。

        规则:
          - 量词必须在注册表中
          - 如果量词是专用的 (非"个"), 检查名词是否兼容
          - "个" 永远合法 (通用量词)

        Args:
            number_text: 数字文本 (如 "三", "5")
            classifier: 量词 (如 "只", "本", "台")
            noun: 名词 (如 "猫", "书", "电脑")

        Returns:
            True 如果搭配合法, False 否则
        """
        if not classifier:
            return False

        # 个 永远合法
        if classifier == "个":
            return True

        # 量词必须在表中
        if classifier not in CLASSIFIER_TO_TYPE:
            return False

        # 量词类型的推荐名词检查
        recommended_nouns = CLASSIFIER_NOUN_DB.get(classifier, [])
        if noun in recommended_nouns:
            return True

        # 检查名词的隐含类型是否与量词类型兼容
        noun_type = NOUN_IMPLIED_TYPE.get(noun, ClassifierType.ANY)
        if noun_type == ClassifierType.ANY:
            # 未知名词, 只能用量词"个"
            return classifier == "个"

        clf_type = CLASSIFIER_TO_TYPE.get(classifier, ClassifierType.ANY)
        compatible = self._TYPE_COMPATIBLE.get(clf_type, set())
        if noun_type in compatible:
            return True

        # 不兼容, 但"个"可以搭配任何名词
        return False

    def suggest_classifier(self, noun: str) -> list[tuple[str, float]]:
        """
        为名词推荐最合适的量词, 按匹配度排序。

        Args:
            noun: 名词

        Returns:
            [(量词, 置信度), ...] 列表, 按置信度降序
        """
        suggestions: list[tuple[str, float]] = []

        noun_type = NOUN_IMPLIED_TYPE.get(noun, ClassifierType.ANY)

        # 查找直接匹配的推荐
        for clf, nouns in CLASSIFIER_NOUN_DB.items():
            if noun in nouns:
                suggestions.append((clf, 1.0))

        # 查找类型兼容的量词
        if noun_type != ClassifierType.ANY:
            compatible = self._TYPE_COMPATIBLE.get(noun_type, set())
            for clf, ctype in CLASSIFIER_TO_TYPE.items():
                if ctype in compatible and clf != "个":
                    if not any(s[0] == clf for s in suggestions):
                        suggestions.append((clf, 0.8))

        # "个" 作为最后的通用选项
        suggestions.append(("个", 0.5))

        # 按置信度降序
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions

    def get_type(self, classifier: str) -> ClassifierType:
        """获取量词对应的类型"""
        return CLASSIFIER_TO_TYPE.get(classifier, ClassifierType.ANY)

    def get_type_name(self, classifier: str) -> str:
        """获取量词对应的类型中文名"""
        ctype = self.get_type(classifier)
        return CLASSIFIER_TYPE_NAMES.get(ctype, "未知")

    def register_noun_type(self, noun: str, ctype: ClassifierType) -> None:
        """注册名词的隐含类型"""
        NOUN_IMPLIED_TYPE[noun] = ctype

    def register_classifier(self, classifier: str, ctype: ClassifierType,
                            nouns: list[str] | None = None) -> None:
        """注册自定义量词"""
        CLASSIFIER_TO_TYPE[classifier] = ctype
        CLASSIFIER_TYPE_NAMES[ctype] = ctype.name
        if nouns:
            CLASSIFIER_NOUN_DB[classifier] = nouns
            for n in nouns:
                NOUN_IMPLIED_TYPE[n] = ctype

    @property
    def classifier_count(self) -> int:
        """已注册的量词数量"""
        return len(CLASSIFIER_TO_TYPE)

    @property
    def noun_count(self) -> int:
        """已注册的名词数量"""
        return len(NOUN_IMPLIED_TYPE)

    def resolution_log(self) -> list[ClassifierResolution]:
        """获取所有解析记录"""
        return list(self._resolution_log)

    def clear_log(self) -> None:
        """清除解析记录"""
        self._resolution_log.clear()
