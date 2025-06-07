# alpha_factory/genetic_programming/generators/legacy_generator.py
import random
from typing import List # 确保导入 List
from alpha_factory.genetic_programming.models import Node # 从模型模块导入Node类

# 遗传编程中使用的操作符和终端值列表
# 这些是从 code.py 迁移过来的，供树生成函数使用。
# 在 AlphaFactory 的后续版本中，这些可能会通过配置系统加载。
terminal_values: List[str] = ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns", "dividend"]
ts_ops: List[str] = ["ts_zscore", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_delta", "ts_ir", "ts_mean","ts_median", "ts_product", "ts_std_dev"]
binary_ops: List[str] = ["add", "subtract", "divide", "multiply", "max", "min"]
ts_ops_values: List[str] = ["20", "40", "60", "120", "240"]
unary_ops: List[str] = ["rank", "zscore", "winsorize", "normalize", "rank_by_side", "sigmoid", "pasteurize", "log"]

def depth_one_trees(
    # 为了与 code.py 保持一致，参数列表暂时保留，尽管在此模块中它们是全局可访问的
    # 在更通用的设计中，这些列表会作为参数传入或从配置中读取
    current_terminal_values: List[str],
    current_binary_ops: List[str],
    current_ts_ops: List[str],
    current_ts_ops_values: List[str],
    current_unary_ops: List[str], # 此参数在 depth_one_trees 中未使用，但保持与原函数签名一致
    flag: int
) -> Node:
    """
    生成深度为 1 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成二元操作符或时间序列操作符的树。
    - 如果 flag 为 0: 生成一个二元操作符作为根节点，其左右子节点为终端值。
    - 如果 flag 为 1: 生成一个时间序列操作符作为根节点，其左子节点为终端值，右子节点为时间序列操作的参数值。

    Args:
        current_terminal_values (List[str]): 可用的终端值。
        current_binary_ops (List[str]): 可用的二元操作符。
        current_ts_ops (List[str]): 可用的时间序列操作符。
        current_ts_ops_values (List[str]): 可用的时间序列操作符参数值。
        current_unary_ops (List[str]): 可用的一元操作符（此函数中未使用）。
        flag (int): 控制树类型的标志 (0 为二元操作，1 为时间序列操作)。

    Returns:
        Node: 生成的深度为 1 的树的根节点。
    """
    # 添加中文注释：根据flag决定生成二元运算树还是时序运算树
    if flag == 0:
        # 添加中文注释：选择一个随机的二元操作符作为根节点
        node = Node(random.choice(current_binary_ops if current_binary_ops else binary_ops))
        # 添加中文注释：左右子节点都从终端值中随机选择
        node.left = Node(random.choice(current_terminal_values if current_terminal_values else terminal_values))
        node.right = Node(random.choice(current_terminal_values if current_terminal_values else terminal_values))
        return node
    # if flag == 1: # 原代码中是 if (flag == 1)，效果一致
    elif flag == 1: # 使用 elif 更清晰
        # 添加中文注释：选择一个随机的时间序列操作符作为根节点
        node = Node(random.choice(current_ts_ops if current_ts_ops else ts_ops))
        # 添加中文注释：左子节点从终端值中随机选择
        node.left = Node(random.choice(current_terminal_values if current_terminal_values else terminal_values))
        # 添加中文注释：右子节点从时间序列操作参数值中随机选择
        node.right = Node(random.choice(current_ts_ops_values if current_ts_ops_values else ts_ops_values))
        return node
    else:
        # 添加中文注释：处理未知的flag值，可以抛出异常或返回默认行为
        # 为了与原始 code.py 的行为（可能隐式出错或不执行）保持一致，这里可以选择不处理
        # 但更健壮的做法是明确处理
        raise ValueError(f"depth_one_trees 收到未知的 flag 值: {flag}")


def depth_two_tree(
    tree1: Node,
    tree2: Node,
    current_ts_ops_values: List[str],
    current_ts_ops: List[str], # 新增 current_binary_ops 以便根节点选择
    current_binary_ops: List[str],
    flag: int
) -> Node:
    """
    生成深度为 2 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成二元操作符或时间序列操作符的树。
    - 如果 flag 为 0: 生成一个二元操作符作为根节点，其左右子节点是输入的深度为 1 的树。
    - 如果 flag 为 1: 生成一个时间序列操作符作为根节点，其左子节点是输入的一个随机深度为 1 的树，
                     右子节点为时间序列操作的参数值。

    Args:
        tree1 (Node): 第一个深度为 1 的树的根节点。
        tree2 (Node): 第二个深度为 1 的树的根节点。
        current_ts_ops_values (List[str]): 可用的时间序列操作符参数值。
        current_ts_ops (List[str]): 可用的时间序列操作符。
        current_binary_ops (List[str]): 可用的二元操作符。
        flag (int): 控制树类型的标志 (0 为二元操作，1 为时间序列操作)。

    Returns:
        Node: 生成的深度为 2 的树的根节点。
    """
    # 添加中文注释：根据flag决定树的根节点类型
    if flag == 0:
        # 添加中文注释：根节点为随机选择的二元操作符
        node = Node(random.choice(current_binary_ops if current_binary_ops else binary_ops))
        # 添加中文注释：左右子节点分别是传入的 tree1 和 tree2
        node.left = tree1
        node.right = tree2
        return node
    # if flag == 1: # 原代码中是 if (flag == 1)
    elif flag == 1: # 使用 elif 更清晰
        # 添加中文注释：根节点为随机选择的时间序列操作符
        node = Node(random.choice(current_ts_ops if current_ts_ops else ts_ops))
        # 添加中文注释：左子节点从 tree1 和 tree2 中随机选择一个
        node.left = random.choice([tree1, tree2])
        # 添加中文注释：右子节点为随机选择的时间序列操作参数值
        node.right = Node(random.choice(current_ts_ops_values if current_ts_ops_values else ts_ops_values))
        return node
    else:
        raise ValueError(f"depth_two_tree 收到未知的 flag 值: {flag}")

def depth_three_tree(
    tree2_list: List[Node], # 原参数名为 tree2，但实际传入的是列表
    # 为了与 code.py 保持一致，参数列表暂时保留，尽管在此模块中它们是全局可访问的
    current_unary_ops: List[str],
    current_binary_ops: List[str],
    current_ts_ops: List[str],
    current_ts_ops_values: List[str],
    flag: int
) -> Node:
    """
    生成深度为 3 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成不同类型的树：
    - 如果 flag 为 0: 生成一个一元操作符作为根节点，其左子节点是从输入列表随机选择的深度为 2 的树。
    - 如果 flag 为 1: 生成一个二元操作符作为根节点，其左右子节点是从输入列表随机选择的深度为 2 的树。
    - 如果 flag 为 2: 生成一个时间序列操作符作为根节点，其左子节点是从输入列表随机选择的深度为 2 的树，
                     右子节点为时间序列操作的参数值。

    Args:
        tree2_list (List[Node]): 包含深度为 2 的树的列表，将从中随机选择作为子节点。
        current_unary_ops (List[str]): 可用的一元操作符。
        current_binary_ops (List[str]): 可用的二元操作符。
        current_ts_ops (List[str]): 可用的时间序列操作符。
        current_ts_ops_values (List[str]): 可用的时间序列操作符参数值。
        flag (int): 控制树类型的标志 (0 为一元操作，1 为二元操作，2 为时间序列操作)。

    Returns:
        Node: 生成的深度为 3 的树的根节点。
    Raises:
        ValueError: 如果 tree2_list 为空或 flag 值未知。
    """
    if not tree2_list:
        # 添加中文注释：如果输入的深度2树列表为空，则无法生成深度3树
        raise ValueError("输入的深度2树列表 (tree2_list) 不能为空。")

    # 添加中文注释：根据flag决定根节点类型
    if flag == 0:
        # 添加中文注释：根节点为随机选择的一元操作符
        node = Node(random.choice(current_unary_ops if current_unary_ops else unary_ops))
        # 添加中文注释：左子节点从输入的深度2树列表中随机选择
        node.left = random.choice(tree2_list)
        node.right = None # 添加中文注释：一元操作符没有右子节点
        return node
    elif flag == 1:
        # 添加中文注释：根节点为随机选择的二元操作符
        node = Node(random.choice(current_binary_ops if current_binary_ops else binary_ops))
        # 添加中文注释：左右子节点均从输入的深度2树列表中随机选择
        node.left = random.choice(tree2_list)
        node.right = random.choice(tree2_list)
        return node
    elif flag == 2:
        # 添加中文注释：根节点为随机选择的时间序列操作符
        node = Node(random.choice(current_ts_ops if current_ts_ops else ts_ops))
        # 添加中文注释：左子节点从输入的深度2树列表中随机选择
        node.left = random.choice(tree2_list)
        # 添加中文注释：右子节点为随机选择的时间序列操作参数值
        node.right = Node(random.choice(current_ts_ops_values if current_ts_ops_values else ts_ops_values))
        return node
    else:
        # 添加中文注释：处理未知的flag值
        raise ValueError(f"depth_three_tree 收到未知的 flag 值: {flag}")
