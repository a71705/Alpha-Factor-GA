import random
from typing import Optional, List, Any, Union # Union for type hinting of Node value

# 从配置模块导入遗传算法相关的常量列表
from config import TERMINAL_VALUES, BINARY_OPS, TS_OPS, TS_OPS_VALUES, UNARY_OPS
import logging

logger = logging.getLogger(__name__)

# --- 自定义异常类 ---
class AlphaModelError(Exception):
    """alpha_models模块中发生的错误的基类。"""
    pass

class ExpressionParsingError(AlphaModelError):
    """在解析Alpha表达式字符串时发生错误。"""
    pass

class InvalidTreeStructureError(AlphaModelError):
    """表示Alpha表达式树的结构无效或不完整。"""
    pass

# 此模块用于定义 Alpha 模型的结构 (特别是表达式树) 和相关操作，
# 例如生成不同深度的树结构。

class Node:
    """
    Alpha 表达式树的节点定义。

    每个节点包含一个值 (操作符、终端值或参数)，
    以及指向左子节点和右子节点的引用。
    对于一元操作符，右子节点通常为 None。
    对于终端节点或参数节点，左右子节点均为 None。

    属性:
        value (Any): 节点存储的值。可以是操作符 (str)，数据字段名 (str)，
                     数值常量 (str, int, float)，或时间序列操作的参数 (str)。
        left (Optional[Node]): 左子节点。
        right (Optional[Node]): 右子节点。
    """
    def __init__(self, value: Any):
        self.value: Any = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def __repr__(self) -> str:
        """提供节点的字符串表示形式，方便调试。"""
        return f"Node({self.value!r})"


def depth_one_trees(
    terminal_values: List[str] = TERMINAL_VALUES,
    binary_ops: List[str] = BINARY_OPS,
    ts_ops: List[str] = TS_OPS,
    ts_ops_values: List[str] = TS_OPS_VALUES,
    # unary_ops is not used in depth_one_trees original logic, but kept for signature consistency if needed later
    unary_ops: List[str] = UNARY_OPS,
    flag: int = -1 # -1 表示随机选择，0 为二元，1 为时间序列
) -> Node:
    """
    生成深度为 1 的随机 Alpha 表达式树。

    深度为 1 的树结构如下：
    - 二元操作: `binary_op(terminal, terminal)`
    - 时间序列操作: `ts_op(terminal, ts_param)`

    参数:
        terminal_values (List[str]): 可用的终端值列表。
        binary_ops (List[str]): 可用的二元操作符列表。
        ts_ops (List[str]): 可用的时间序列操作符列表。
        ts_ops_values (List[str]): 可用的时间序列操作参数值列表。
        unary_ops (List[str]): 可用的一元操作符列表 (此函数中未使用)。
        flag (int): 控制生成的树类型。
                    - 0: 生成二元操作符树。
                    - 1: 生成时间序列操作符树。
                    - 其他值 (如默认 -1): 随机选择上述一种类型。

    返回:
        Node: 生成的深度为 1 的树的根节点。
    """
    if flag not in [0, 1]: # 如果 flag 不是 0 或 1，则随机选择
        chosen_flag = random.choice([0, 1])
    else:
        chosen_flag = flag

    if chosen_flag == 0: # 生成二元操作符树
        if not binary_ops or not terminal_values or len(terminal_values) < 2 :
            raise ValueError("二元操作符列表或终端值列表为空或不足以构建二元树。")
        node = Node(random.choice(binary_ops))
        node.left = Node(random.choice(terminal_values))
        node.right = Node(random.choice(terminal_values))
        return node
    # chosen_flag == 1，生成时间序列操作符树
    else:
        if not ts_ops or not terminal_values or not ts_ops_values:
            raise ValueError("时间序列操作符、终端值或时间序列参数列表为空。")
        node = Node(random.choice(ts_ops))
        node.left = Node(random.choice(terminal_values))
        node.right = Node(random.choice(ts_ops_values)) # ts_ops_values 已经是 Node(value)
        return node


def depth_two_tree(
    tree1: Node,
    tree2: Node,
    binary_ops: List[str] = BINARY_OPS,
    ts_ops: List[str] = TS_OPS,
    ts_ops_values: List[str] = TS_OPS_VALUES,
    flag: int = -1 # -1 表示随机选择，0 为二元，1 为时间序列
) -> Node:
    """
    生成深度为 2 的随机 Alpha 表达式树，其子节点是深度为 1 的树。

    深度为 2 的树结构示例:
    - 二元操作: `binary_op(d1_tree, d1_tree)`
    - 时间序列操作: `ts_op(d1_tree, ts_param)`

    参数:
        tree1 (Node): 第一个深度为 1 的树的根节点 (作为潜在子节点)。
        tree2 (Node): 第二个深度为 1 的树的根节点 (作为潜在子节点)。
        binary_ops (List[str]): 可用的二元操作符列表。
        ts_ops (List[str]): 可用的时间序列操作符列表。
        ts_ops_values (List[str]): 可用的时间序列操作参数值列表。
        flag (int): 控制生成的树类型。
                    - 0: 生成以二元操作符为根的树。
                    - 1: 生成以时间序列操作符为根的树。
                    - 其他值 (如默认 -1): 随机选择上述一种类型。
    返回:
        Node: 生成的深度为 2 的树的根节点。
    """
    if flag not in [0, 1]:
        chosen_flag = random.choice([0, 1])
    else:
        chosen_flag = flag

    if chosen_flag == 0: # 以二元操作符为根
        if not binary_ops:
            raise ValueError("二元操作符列表为空。")
        node = Node(random.choice(binary_ops))
        node.left = tree1  # tree1 是一个 Node 对象
        node.right = tree2 # tree2 是一个 Node 对象
        return node
    # chosen_flag == 1，以时间序列操作符为根
    else:
        if not ts_ops or not ts_ops_values:
            raise ValueError("时间序列操作符或时间序列参数列表为空。")
        node = Node(random.choice(ts_ops))
        # 随机选择 tree1 或 tree2 作为其左子树
        node.left = random.choice([tree1, tree2])
        node.right = Node(random.choice(ts_ops_values))
        return node


def depth_three_tree(
    d2_trees: List[Node], # 列表，包含多个深度为2的树节点
    unary_ops: List[str] = UNARY_OPS,
    binary_ops: List[str] = BINARY_OPS,
    ts_ops: List[str] = TS_OPS,
    ts_ops_values: List[str] = TS_OPS_VALUES,
    flag: int = -1 # -1 表示随机，0 为一元，1 为二元，2 为时间序列
) -> Node:
    """
    生成深度为 3 的随机 Alpha 表达式树，其子节点是深度为 2 的树。

    深度为 3 的树结构示例:
    - 一元操作: `unary_op(d2_tree)`
    - 二元操作: `binary_op(d2_tree, d2_tree)`
    - 时间序列操作: `ts_op(d2_tree, ts_param)`

    参数:
        d2_trees (List[Node]): 包含深度为 2 的树的根节点列表，从中选择子树。
        unary_ops (List[str]): 可用的一元操作符列表。
        binary_ops (List[str]): 可用的二元操作符列表。
        ts_ops (List[str]): 可用的时间序列操作符列表。
        ts_ops_values (List[str]): 可用的时间序列操作参数值列表。
        flag (int): 控制生成的树类型。
                    - 0: 生成以一元操作符为根的树。
                    - 1: 生成以二元操作符为根的树。
                    - 2: 生成以时间序列操作符为根的树。
                    - 其他值 (如默认 -1): 随机选择上述一种类型。
    返回:
        Node: 生成的深度为 3 的树的根节点。
    """
    if not d2_trees:
        raise ValueError("深度为2的树列表 (d2_trees) 为空，无法构建深度3的树。")

    if flag not in [0, 1, 2]:
        chosen_flag = random.choice([0, 1, 2])
    else:
        chosen_flag = flag

    selected_d2_tree = random.choice(d2_trees)

    if chosen_flag == 0: # 以一元操作符为根
        if not unary_ops:
            raise ValueError("一元操作符列表为空。")
        node = Node(random.choice(unary_ops))
        node.left = selected_d2_tree
        node.right = None # 一元操作符通常没有右子节点
        return node
    elif chosen_flag == 1: # 以二元操作符为根
        if not binary_ops:
            raise ValueError("二元操作符列表为空。")
        node = Node(random.choice(binary_ops))
        node.left = selected_d2_tree
        node.right = random.choice(d2_trees) # 右子节点也是一个深度2的树
        return node
    # chosen_flag == 2，以时间序列操作符为根
    else:
        if not ts_ops or not ts_ops_values:
            raise ValueError("时间序列操作符或时间序列参数列表为空。")
        node = Node(random.choice(ts_ops))
        node.left = selected_d2_tree
        node.right = Node(random.choice(ts_ops_values))
        return node

import re # 正则表达式模块，用于 parse_expression

# --- 树结构与Alpha表达式字符串的相互转换 ---

def d1tree_to_alpha(tree: Node) -> str:
    """
    将深度为 1 的树结构转换为 Alpha 表达式字符串。

    例如:
        Node("add", left=Node("close"), right=Node("open")) -> "add(close,open)"
        Node("ts_rank", left=Node("vwap"), right=Node("20")) -> "ts_rank(vwap,20)"

    参数:
        tree (Node): 深度为 1 的树的根节点。期望其左右子节点的值可以直接转换为字符串。

    返回:
        str: 生成的 Alpha 表达式字符串。
             如果树结构不符合预期（例如子节点丢失），可能会引发 AttributeError。
    """
    if not tree:
        raise InvalidTreeStructureError("d1tree_to_alpha错误: 输入的树节点为None。")
    if not tree.left:
        logger.error(f"d1tree_to_alpha错误: 树根节点 '{tree.value}' 的左子节点为None。")
        raise InvalidTreeStructureError(f"树根节点 '{tree.value}' 的左子节点为None。")
    if not tree.right:
        logger.error(f"d1tree_to_alpha错误: 树根节点 '{tree.value}' 的右子节点为None。")
        raise InvalidTreeStructureError(f"树根节点 '{tree.value}' 的右子节点为None。")

    if tree.value not in BINARY_OPS and tree.value not in TS_OPS:
        logger.warning(f"d1tree_to_alpha警告: 树根节点值 '{tree.value}' 未在 BINARY_OPS ({BINARY_OPS}) 或 TS_OPS ({TS_OPS}) 中找到。")

    # 子节点应为叶子节点，其value应为字符串或可转换为字符串的类型
    left_val = str(tree.left.value) if tree.left.value is not None else "NoneValue"
    right_val = str(tree.right.value) if tree.right.value is not None else "NoneValue"

    return f"{tree.value}({left_val},{right_val})"


def d2tree_to_alpha(tree: Node) -> str:
    """
    将深度为 2 的树结构转换为 Alpha 表达式字符串。

    此函数假定子节点是符合 `d1tree_to_alpha` 能够处理的简单结构。
    例如:
        tree.value = "add"
        tree.left = Node("ts_rank", left=Node("close"), right=Node("10"))
        tree.right = Node("ts_mean", left=Node("open"), right=Node("5"))
        将会转换为 "add(ts_rank(close,10),ts_mean(open,5))"

        tree.value = "ts_delta"
        tree.left = Node("multiply", left=Node("high"), right=Node("low"))
        tree.right = Node("2") # ts_param
        将会转换为 "ts_delta(multiply(high,low),2)"

    参数:
        tree (Node): 深度为 2 的树的根节点。

    返回:
        str: 生成的 Alpha 表达式字符串。
    """
    if not tree:
        raise InvalidTreeStructureError("d2tree_to_alpha错误: 输入的树节点为None。")
    if not tree.left: # All D2 trees must have a left child.
        logger.error(f"d2tree_to_alpha错误: 树根节点 '{tree.value}' 的左子节点为None。")
        raise InvalidTreeStructureError(f"树根节点 '{tree.value}' 的左子节点为None。")

    if tree.value in BINARY_OPS:
        if not tree.right: # Binary ops also need a right child.
            logger.error(f"d2tree_to_alpha错误: 二元操作树 '{tree.value}' 的右子节点为None。")
            raise InvalidTreeStructureError(f"二元操作树 '{tree.value}' 的右子节点为None。")

        # For D2 binary ops, left and right children are D1 trees.
        left_expr = d1tree_to_alpha(tree.left)
        right_expr = d1tree_to_alpha(tree.right)
        return f"{tree.value}({left_expr},{right_expr})"
    elif tree.value in TS_OPS:
        # For D2 TS_OPS, left child is a D1 tree, right child is a parameter (leaf node).
        if not tree.right or tree.right.left is not None or tree.right.right is not None:
            logger.error(f"d2tree_to_alpha错误: 时间序列操作树 '{tree.value}' 的右子节点 (参数) 结构不正确 (R: {tree.right})。它应该是一个叶子节点。")
            raise InvalidTreeStructureError(f"时间序列操作树 '{tree.value}' 的右子节点 (参数) 结构不正确。")
        left_expr = d1tree_to_alpha(tree.left)
        param_val = str(tree.right.value) if tree.right.value is not None else "NoneValue"
        return f"{tree.value}({left_expr},{param_val})"
    else:
        logger.error(f"d2tree_to_alpha错误: 未知的根节点类型 '{tree.value}'。它不属于 BINARY_OPS ({BINARY_OPS}) 或 TS_OPS ({TS_OPS})。")
        raise InvalidTreeStructureError(f"未知的根节点类型 '{tree.value}' 用于构建深度2 Alpha表达式。")


def d3tree_to_alpha(tree: Node) -> str:
    """
    将深度为 3 的树结构转换为 Alpha 表达式字符串。

    此函数依赖于子树 (深度为 2) 的类型来正确构建表达式。
    由于其复杂性和对特定树结构的强依赖性，这是一个潜在的重构点。
    当前的实现直接翻译自原始 `code.py` 中的逻辑，可能非常脆弱。

    参数:
        tree (Node): 深度为 3 的树的根节点。

    返回:
        str: 生成的 Alpha 表达式字符串。如果树结构不匹配任何已知模式，可能返回空字符串或抛出错误。

    潜在问题:
    - 极度依赖 `tree.left.value` 和 `tree.right.value` 来判断子树类型。
    - 如果子树的结构与 `d2tree_to_alpha` 的预期输出不完全一致，会产生错误。
    - 错误处理不够健壮，很多路径下可能直接抛出 AttributeError。
    """
    if not tree:
        raise InvalidTreeStructureError("d3tree_to_alpha错误: 输入的树节点为None。")
    if not tree.left: # All D3 trees must have a left child.
        logger.error(f"d3tree_to_alpha错误: 树根节点 '{tree.value}' 的左子节点为None。")
        raise InvalidTreeStructureError(f"树根节点 '{tree.value}' 的左子节点为None。")

    if tree.value in BINARY_OPS:
        if not tree.right: # Binary ops also need a right child.
            logger.error(f"d3tree_to_alpha错误: 二元操作树 '{tree.value}' 的右子节点为None。")
            raise InvalidTreeStructureError(f"二元操作树 '{tree.value}' 的右子节点为None。")
        # For D3 binary ops, left and right children are D2 trees.
        left_expr = d2tree_to_alpha(tree.left)
        right_expr = d2tree_to_alpha(tree.right)
        return f"{tree.value}({left_expr},{right_expr})"
    elif tree.value in TS_OPS:
        # For D3 TS_OPS, left child is a D2 tree, right child is a parameter (leaf node).
        if not tree.right or tree.right.left is not None or tree.right.right is not None:
            logger.error(f"d3tree_to_alpha错误: 时间序列操作树 '{tree.value}' 的右子节点 (参数) 结构不正确 (R: {tree.right})。")
            raise InvalidTreeStructureError(f"时间序列操作树 '{tree.value}' 的右子节点 (参数) 结构不正确。")
        left_expr = d2tree_to_alpha(tree.left)
        param_val = str(tree.right.value) if tree.right.value is not None else "NoneValue"
        return f"{tree.value}({left_expr},{param_val})"
    elif tree.value in UNARY_OPS:
        # For D3 UNARY_OPS, left child is a D2 tree, no right child.
        if tree.right is not None:
            logger.error(f"d3tree_to_alpha错误: 一元操作树 '{tree.value}' 的右子节点不为None (R: {tree.right})。")
            raise InvalidTreeStructureError(f"一元操作树 '{tree.value}' 的右子节点不为None。")
        left_expr = d2tree_to_alpha(tree.left)
        return f"{tree.value}({left_expr})"
    else:
        logger.error(f"d3tree_to_alpha错误: 未知的根节点类型 '{tree.value}'。不属于 BINARY_OPS, TS_OPS, 或 UNARY_OPS。")
        raise InvalidTreeStructureError(f"未知的根节点类型 '{tree.value}' 用于构建深度3 Alpha表达式。")


def parse_expression(alpha_expressions: List[str]) -> List[List[str]]:
    """
    解析 Alpha 表达式字符串列表，提取操作符和操作数。
    例如: "add(close,open)" -> ["add", "close", "open"]

    参数:
        alpha_expressions (List[str]): Alpha 表达式字符串列表。

    返回:
        List[List[str]]: 包含每个表达式解析后组件的列表的列表。
                         无效的表达式字符串可能会产生空的子列表或不正确的组件。
    """
    parsed_components_list = []
    # 正则表达式尝试匹配：
    # 1. \w+ : 匹配一个或多个单词字符 (字母、数字、下划线), 用于操作符和终端值。
    # 2. [+\-*/] : 匹配基本的算术操作符 (虽然FASTEXPR通常用函数名如 add, subtract)。
    # 3. 明确列出的 ts_ops 和 unary_ops，以确保它们作为单个token被捕获，特别是包含下划线的。
    # 4. [0-9]+ : 匹配数字，用于时间序列参数。
    # 5. \(|\) : 匹配括号 (后续会被过滤掉)。
    # 将所有操作符列表合并，并进行排序（最长的优先），以处理嵌套或重叠的模式
    # 例如，ts_arg_max 应优先于 max
    # Moved pattern compilation to module level for efficiency
    # all_ops = sorted(TS_OPS + BINARY_OPS + UNARY_OPS, key=len, reverse=True)
    # pattern_str = r'(\w+|[+\-*/]|' + '|'.join(re.escape(op) for op in all_ops) + r'|[0-9]+|\(|\))'
    # _ALPHA_EXPRESSION_PATTERN = re.compile(pattern_str) # Now a module-level constant

    for i, expression_str in enumerate(alpha_expressions):
        if not isinstance(expression_str, str):
            logger.error(f"parse_expression错误: 输入的表达式 #{i} 值 '{expression_str}' 不是字符串 (类型: {type(expression_str)}).")
            raise ExpressionParsingError(f"输入表达式必须是字符串，但收到: {type(expression_str)} for expression: '{expression_str}'")

        matches = _ALPHA_EXPRESSION_PATTERN.findall(expression_str) # Use pre-compiled pattern
        components = [match for match in matches if match and match not in ['(', ')']]

        if not components and expression_str.strip(): # 原始表达式非空/非纯空格，但解析后无组件
            logger.warning(f"parse_expression警告: 表达式 '{expression_str}' 解析后无有效组件。可能表达式无效或不被支持的结构。")
            # 根据严格程度，可以选择抛出 ExpressionParsingError
            # raise ExpressionParsingError(f"表达式 '{expression_str}' 解析后无有效组件。")
        parsed_components_list.append(components)
    return parsed_components_list


def _build_tree_from_components(components: List[str], Waalaikumsalam: str) -> Optional[Node]:
    """
    (内部辅助函数) 根据解析出的组件列表递归构建树。
    这是一个简化的示例，实际的解析和树构建对于复杂的、可变深度的表达式会非常复杂，
    通常需要更复杂的解析技术（如 Shunting-yard 算法或构建 AST 解析器）。
    此版本主要针对 d1_alpha_to_tree, d2_alpha_to_tree, d3_alpha_to_tree 的逆操作。

    参数:
        components (List[str]): 从表达式解析出来的组件列表。
        target_depth (str): 'd1', 'd2', 'd3' 指示目标树的深度和结构。

    返回:
        Optional[Node]: 构建的树的根节点，如果无法构建则返回 None。
    """
    if not components:
        return None

    # 简单的逻辑，仅作为示例，需要根据 dX_alpha_to_tree 的具体结构来定制
    # 这部分非常难通用化，因为原始的 dX_alpha_to_tree 是硬编码的

    # 对于 d1_alpha_to_tree: ["op", "left_val", "right_val"]
    if Waalaikumsalam == 'd1' and len(components) == 3:
        op, left_val, right_val = components[0], components[1], components[2]
        if op in BINARY_OPS or op in TS_OPS: # 假设深度1的操作符是这些
            root = Node(op)
            root.left = Node(left_val)
            root.right = Node(right_val)
            return root

    # 对于 d2_alpha_to_tree:
    #   - TS: [ts_op, d1_op, d1_val1, d1_val2, ts_param_d2] (5 elements)
    #   - BINARY: [op, d1_op_L, val_L1, val_L2, d1_op_R, val_R1, val_R2] (7 elements)
    elif Waalaikumsalam == 'd2':
        op = components[0]
        if op in TS_OPS and len(components) == 5:
            root = Node(op)
            # 左子树是深度1的树
            root.left = Node(components[1]) # d1_op
            root.left.left = Node(components[2]) # d1_val1
            root.left.right = Node(components[3]) # d1_val2
            # 右子节点是参数
            root.right = Node(components[4]) # ts_param_d2
            return root
        elif op in BINARY_OPS and len(components) == 7:
            root = Node(op)
            # 左子树是深度1的树
            root.left = Node(components[1]) # d1_op_L
            root.left.left = Node(components[2]) # val_L1
            root.left.right = Node(components[3]) # val_L2
            # 右子树是深度1的树
            root.right = Node(components[4]) # d1_op_R
            root.right.left = Node(components[5]) # val_R1
            root.right.right = Node(components[6]) # val_R2
            return root

    # 对于 d3_alpha_to_tree: 这部分会更复杂，因为 d3tree_to_alpha 的分支更多
    # 鉴于其复杂性和脆弱性，这里仅作标记，实际实现需要非常小心
    elif Waalaikumsalam == 'd3':
        # print(f"警告: d3_alpha_to_tree 的逆向解析逻辑非常复杂且未完全实现。组件: {components}")
        # 这是一个高度简化的占位符，实际情况复杂得多
        # 需要根据 d3tree_to_alpha 的多个 if 条件来反向构造
        # 例如，如果 components[0] in UNARY_OPS:
        #   node = Node(components[0])
        #   node.left = _build_tree_from_components(components[1:], 'd2') # 递归构建深度2的子树
        #   return node
        # ... 其他情况 ...
        pass # 实际逻辑会非常复杂

    # print(f"警告: 无法从组件 {components} 构建目标深度 '{target_depth}' 的树。")
    return None


def d1_alpha_to_tree(alpha_expressions: List[str]) -> List[Node]:
    """
    将深度为 1 的 Alpha 表达式字符串列表转换为树结构列表。
    """
    trees = []
    parsed_expr_components = parse_expression(alpha_expressions)
    for i, components in enumerate(parsed_expr_components):
        original_expression = alpha_expressions[i]
        if not components and original_expression.strip(): # 解析后为空，但原始表达式非空
             logger.warning(f"d1_alpha_to_tree: 表达式 '{original_expression}' 解析后无组件，跳过。")
             continue

        if len(components) == 3:
            op, left_val, right_val = components[0], components[1], components[2]
            if op not in BINARY_OPS and op not in TS_OPS:
                logger.warning(f"d1_alpha_to_tree: 操作符 '{op}' (来自表达式 '{original_expression}') 未在已知 BINARY_OPS 或 TS_OPS 中。仍尝试构建树。")

            node = Node(op)
            node.left = Node(left_val) # 假设这些是终端值
            node.right = Node(right_val)
            trees.append(node)
        elif components: # 解析后有组件，但不符合长度3
            logger.warning(f"d1_alpha_to_tree: 表达式 '{original_expression}' 解析后组件数量 ({len(components)}) 不符合深度1结构 (预期3)，跳过。组件: {components}")
        # 如果 components 为空且 original_expression 也为空或纯空格，则静默跳过
    return trees


def d2_alpha_to_tree(alpha_expressions: List[str]) -> List[Node]:
    """
    将深度为 2 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数仅适用于 `d2tree_to_alpha` 生成的特定结构。
    """
    trees = []
    parsed_expr_components = parse_expression(alpha_expressions)
    for i, components in enumerate(parsed_expr_components):
        original_expression = alpha_expressions[i]
        if not components and original_expression.strip():
            logger.warning(f"d2_alpha_to_tree: 表达式 '{original_expression}' 解析后无组件，跳过。")
            continue

        if not components: # Skip if components list is empty (e.g. from empty or invalid expression string)
            continue

        op = components[0]
        valid_d2_structure = False
        if op in TS_OPS and len(components) == 5: # Expected: [ts_op, d1_op, d1_val1, d1_val2, ts_param_d2]
            valid_d2_structure = True
        elif op in BINARY_OPS and len(components) == 7: # Expected: [op, d1_op_L, val_L1, val_L2, d1_op_R, val_R1, val_R2]
            valid_d2_structure = True

        if not valid_d2_structure:
            logger.warning(f"d2_alpha_to_tree: 表达式 '{original_expression}' 解析后组件数量 ({len(components)}) 或根操作符 '{op}' 不符合深度2结构，跳过。组件: {components}")
            continue

        # This check is somewhat redundant if the previous one is strict, but acts as a safeguard.
        if op not in BINARY_OPS and op not in TS_OPS:
             logger.warning(f"d2_alpha_to_tree: 根操作符 '{op}' (来自表达式 '{original_expression}') 未在已知 BINARY_OPS 或 TS_OPS 中。仍尝试构建。")
             # Potentially skip here or let _build_tree_from_components handle it if it has its own checks

        try:
            tree_node = _build_tree_from_components(components, 'd2') # _build_tree_from_components handles actual construction
            if tree_node:
                trees.append(tree_node)
            elif components : # _build_tree_from_components returned None but there were components
                logger.warning(f"d2_alpha_to_tree: 未能从组件为表达式 '{original_expression}' 构建深度2树 (组件: {components})，已跳过。")
        except Exception as e:
            logger.error(f"d2_alpha_to_tree: 构建树时发生意外错误 for expression '{original_expression}': {e}", exc_info=True)

    return trees


def d3_alpha_to_tree(alpha_expressions: List[str]) -> List[Node]:
    """
    将深度为 3 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数高度依赖于 `d3tree_to_alpha` 生成的特定、硬编码的模式匹配。
    这是一个潜在的重构点，因其复杂性和脆弱性。
    当前的实现可能无法完美覆盖所有 `d3tree_to_alpha` 的情况。
    """
    trees = []
    parsed_expr_components = parse_expression(alpha_expressions)
    for i, components in enumerate(parsed_expr_components):
        original_expression = alpha_expressions[i]
        if not components and original_expression.strip():
            logger.warning(f"d3_alpha_to_tree: 表达式 '{original_expression}' 解析后无组件，跳过。")
            continue

        if not components:
            continue

        # 深度3的解析非常复杂且脆弱，依赖于组件数量和操作符类型
        # _build_tree_from_components 中对 'd3' 的处理是占位符
        # 标记：这部分是主要的健robustness风险点，应在后续重构中用更通用的解析器替代
        # 当前版本主要依赖于 _build_tree_from_components 的（目前未实现的）'d3'逻辑

        op = components[0]
        if op not in UNARY_OPS and op not in BINARY_OPS and op not in TS_OPS:
            logger.warning(f"d3_alpha_to_tree: 根操作符 '{op}' (来自表达式 '{original_expression}') 未在已知操作符列表中。跳过。")
            continue

        # 由于 _build_tree_from_components['d3'] 未实现，这里会跳过所有D3树的构建
        # 后续如果 _build_tree_from_components['d3'] 实现了，这里的逻辑也需要对应调整
        # 以进行更细致的长度和结构检查。
        try:
            # tree_node = _build_tree_from_components(components, 'd3') # This line is commented out as 'd3' is not implemented in helper
            # if tree_node:
            #     trees.append(tree_node)
            # el
            if components: # 只有在有组件的情况下才记录警告，因为 _build_tree_from_components 的 'd3' 逻辑未实现
                logger.warning(f"d3_alpha_to_tree: 深度3树的解析逻辑 (_build_tree_from_components for 'd3') 未完全实现。跳过表达式 '{original_expression}'. 组件: {components}")
        except Exception as e:
            logger.error(f"d3_alpha_to_tree: 构建树时发生意外错误 for expression '{original_expression}': {e}", exc_info=True)

    return trees


# --- Module-level compiled regex for parse_expression ---
_ALL_OPS_SORTED = sorted(TS_OPS + BINARY_OPS + UNARY_OPS, key=len, reverse=True)
_PATTERN_STR = r'(\w+|[+\-*/]|' + '|'.join(re.escape(op) for op in _ALL_OPS_SORTED) + r'|[0-9]+|\(|\))'
_ALPHA_EXPRESSION_PATTERN = re.compile(_PATTERN_STR)


# --- 遗传算法辅助函数 (树操作) ---

def copy_tree(original_node: Optional[Node]) -> Optional[Node]:
    """
    递归地深度复制一个树（或子树）。

    参数:
        original_node (Optional[Node]): 要复制的树的根节点。如果为 None，则返回 None。

    返回:
        Optional[Node]: 新创建的树的根节点副本。
    """
    if original_node is None:
        return None

    new_node = Node(original_node.value)
    new_node.left = copy_tree(original_node.left)
    new_node.right = copy_tree(original_node.right)
    return new_node


def get_random_node(root_node: Node) -> Optional[Node]:
    """
    从树中随机选择一个节点。
    如果树为空，则返回 None。

    参数:
        root_node (Node): 树的根节点。

    返回:
        Optional[Node]: 随机选择的节点。如果树为空，则为 None。
    """
    if root_node is None:
        return None

    nodes: List[Node] = []
    collect_nodes(root_node, nodes)

    if not nodes: # 应该不会发生，因为已经检查了 root_node is None
        return None
    return random.choice(nodes)


def collect_nodes(current_node: Optional[Node], nodes_list: List[Node]) -> None:
    """
    (辅助函数) 递归地收集树中的所有节点，并将它们添加到提供的列表中。
    采用深度优先遍历（前序）。

    参数:
        current_node (Optional[Node]): 当前遍历到的节点。
        nodes_list (List[Node]): 用于收集节点的列表。
    """
    if current_node:
        nodes_list.append(current_node)
        collect_nodes(current_node.left, nodes_list)
        collect_nodes(current_node.right, nodes_list)

# TODO: 考虑是否需要一个 Alpha 类来封装 Alpha 的属性和行为。
pass
