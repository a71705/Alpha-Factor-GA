# alpha_factory/genetic_programming/operators/legacy_operators.py
import random
from typing import Optional, Tuple, List # 导入 Tuple 和 List
from alpha_factory.genetic_programming.models import Node # 从模型模块导入Node类
# 从旧版生成器导入操作符和终端值列表，供 mutate_random_node 使用
from alpha_factory.genetic_programming.generators.legacy_generator import (
    terminal_values,
    unary_ops,
    binary_ops,
    ts_ops,
    ts_ops_values
)

def copy_tree(original_node: Optional[Node]) -> Optional[Node]:
    """
    递归地复制一个树结构。

    Args:
        original_node (Optional[Node]): 要复制的树的根节点。如果为None，则返回None。

    Returns:
        Optional[Node]: 新创建的树的根节点副本，如果输入为None，则为None。
    """
    # 添加中文注释：如果原始节点为空，则直接返回空
    if original_node is None:
        return None

    # 添加中文注释：创建一个新的节点，其值与原始节点相同
    new_node = Node(original_node.value)
    # 添加中文注释：递归复制左子树和右子树
    new_node.left = copy_tree(original_node.left)
    new_node.right = copy_tree(original_node.right)
    return new_node

def mutate_random_node(
    original_node: Node,
    # 以下参数列表与 code.py 保持一致，但实际使用的是从 legacy_generator 导入的全局列表
    # 在更通用的设计中，这些会通过配置或参数传入
    current_terminal_values: List[str],
    current_unary_ops: List[str],
    current_binary_ops: List[str],
    current_ts_ops: List[str],
    current_ts_ops_values: List[str]
) -> Node:
    """
    随机变异树中的一个节点。实际操作中，该函数将使用从 legacy_generator 模块导入的全局操作符列表。

    Args:
        original_node (Node): 原始树的根节点。
        current_terminal_values (List[str]): （未使用，占位）可用的终端值。
        current_unary_ops (List[str]): （未使用，占位）可用的一元操作符。
        current_binary_ops (List[str]): （未使用，占位）可用的二元操作符。
        current_ts_ops (List[str]): （未使用，占位）可用的时间序列操作符。
        current_ts_ops_values (List[str]): （未使用，占位）可用的时间序列操作符参数值。


    Returns:
        Node: 变异后的树的根节点。
    """
    # 添加中文注释：首先深度复制原始树，确保不在原树上修改
    mutated_tree = copy_tree(original_node)
    if mutated_tree is None: # 如果复制失败（例如原始节点就是None）
        # 根据原始代码行为，如果 original_node 为 None，copy_tree 会返回 None
        # 那么这里 mutated_tree 就是 None，后续 collect_nodes 会传入 None
        # 为了健壮性，如果 mutated_tree 是 None，应该直接返回，或者根据 original_node 是否为 None 抛出错误
        return original_node # 或者更合适的是 None，如果 original_node 是 None

    nodes_to_mutate = []
    def collect_nodes(node: Optional[Node]):
        if node:
            nodes_to_mutate.append(node)
            collect_nodes(node.left)
            collect_nodes(node.right)

    collect_nodes(mutated_tree)

    if not nodes_to_mutate: # 如果树为空（例如，mutated_tree是None，或者是一个没有子节点的单节点且该节点无法变异）
        return mutated_tree # 返回未变异的树（可能是None）

    node_to_change = random.choice(nodes_to_mutate) # 随机选择一个节点进行变异

    original_value = node_to_change.value

    possible_new_values = []
    # 添加中文注释：根据节点当前值的类型从相应的列表中随机选择一个不同的新值
    if original_value in binary_ops:
        possible_new_values = [op for op in binary_ops if op != original_value]
        if possible_new_values: node_to_change.value = random.choice(possible_new_values)
    elif original_value in ts_ops:
        possible_new_values = [op for op in ts_ops if op != original_value]
        if possible_new_values: node_to_change.value = random.choice(possible_new_values)
    elif original_value in ts_ops_values:
        possible_new_values = [val for val in ts_ops_values if val != original_value]
        if possible_new_values: node_to_change.value = random.choice(possible_new_values)
    elif original_value in unary_ops:
        possible_new_values = [op for op in unary_ops if op != original_value]
        if possible_new_values: node_to_change.value = random.choice(possible_new_values)
    elif original_value in terminal_values:
        possible_new_values = [term for term in terminal_values if term != original_value]
        if possible_new_values: node_to_change.value = random.choice(possible_new_values)
    # 如果值不在任何已知列表中，或者列表中只有一个元素（无法选择不同的），则不进行变异

    return mutated_tree


def crossover(parent1: Node, parent2: Node, n: int) -> Tuple[Optional[Node], Optional[Node]]:
    """
    执行树结构之间的交叉操作，生成两个子树。
    注意: 这里的交叉实现非常特定，只在根节点的左右子节点之间进行交换，且依赖于树的深度。
    它不是一个通用的随机子树交换交叉。

    Args:
        parent1 (Node): 第一个父树的根节点。
        parent2 (Node): 第二个父树的根节点。
        n (int): 用于判断树深度的参数 (2 代表深度 2 树，3 代表深度 3 树的交叉规则)。
                 这个参数决定了交叉点选择的逻辑。

    Returns:
        Tuple[Optional[Node], Optional[Node]]: 包含两个子树根节点的元组。
                           如果输入不合法或无法交叉，可能返回原始父本的副本。
    """
    # 添加中文注释：深度复制父树以创建子树，避免修改原始父本
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)

    # 添加中文注释：确保子树存在，才能进行后续操作
    if child1 is None or child2 is None:
        # print("警告: 交叉操作的父代为空，返回原始父代（或其副本）。")
        return child1, child2

    # 针对深度 n 的树进行交叉 (原 code.py 中 n=2 和 n=3 的逻辑几乎一样)
    if n == 2 or n == 3:
        # 情况1：两个父节点都是二元操作符
        if child1.value in binary_ops and child2.value in binary_ops:
            # 确保它们都有左右子节点才能交换
            if child1.left is None or child1.right is None or \
               child2.left is None or child2.right is None:
                # print(f"警告: 交叉时二元操作父代 {child1.value} 或 {child2.value} 的子节点不完整，不进行交换。")
                return child1, child2

            side = random.choice(['L', 'R'])
            same_side_swap = random.choice([True, False])

            if side == 'L':
                if same_side_swap:
                    child1.left, child2.left = child2.left, child1.left
                else:
                    child1.left, child2.right = child2.right, child1.left
            elif side == 'R':
                if same_side_swap:
                    child1.right, child2.right = child2.right, child1.right
                else:
                    child1.right, child2.left = child2.left, child1.right
            return child1, child2

        # 情况2：两个父节点都是时间序列操作符
        elif child1.value in ts_ops and child2.value in ts_ops:
            # 时间序列操作符只交换左子树（表达式部分），右子树是参数值，不参与交换
            if child1.left is None or child2.left is None:
                # print(f"警告: 交叉时TS操作父代 {child1.value} 或 {child2.value} 的左子节点不完整，不进行交换。")
                return child1, child2

            child1.left, child2.left = child2.left, child1.left
            return child1, child2

        # 添加中文注释：如果父节点类型不匹配（例如，一个是二元操作符，另一个是时间序列操作符）
        # 或者不是以上两种情况，则不执行特定的交换逻辑，直接返回副本。
        # print(f"警告: 交叉操作因父代类型不匹配 ({child1.value}, {child2.value}) 或非预期类型，未进行交换。")
        return child1, child2

    # 添加中文注释：如果 n 不是 2 或 3，或者其他未覆盖的情况，则不执行交叉
    # print(f"警告: 交叉操作未针对深度 n={n} 定义规则，返回副本。")
    return child1, child2
