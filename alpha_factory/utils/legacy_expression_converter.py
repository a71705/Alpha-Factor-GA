# alpha_factory/utils/legacy_expression_converter.py
import re
from typing import List, Optional # 确保导入 List 和 Optional
from alpha_factory.genetic_programming.models import Node # 从模型模块导入Node类

# 从 code.py 迁移过来的全局操作符列表，供 d3tree_to_alpha 等函数内部逻辑使用
# 注意：这些列表在 code.py 中是全局定义的，并且 d3tree_to_alpha 的硬编码逻辑依赖它们。
# 虽然更好的做法可能是通过配置或参数传递，但为了“原样复制”，我们暂时在此处复制。
# 在后续的 "generic" 版本中，这些应该被更灵活的机制替代。
terminal_values = ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns", "dividend"]
ts_ops = ["ts_zscore", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_delta", "ts_ir", "ts_mean","ts_median", "ts_product", "ts_std_dev"]
binary_ops = ["add", "subtract", "divide", "multiply", "max", "min"]
ts_ops_values = ["20", "40", "60", "120", "240"] # 虽然转换函数本身不直接用，但原始的树生成依赖它，解析可能间接依赖
unary_ops = ["rank", "zscore", "winsorize", "normalize", "rank_by_side", "sigmoid", "pasteurize", "log"]

def d1tree_to_alpha(tree: Node) -> str:
    """
    将深度为 1 的树结构转换为 Alpha 表达式字符串。
    格式: operation(operand1,operand2) 或 ts_operation(operand1,ts_param)
    此函数仅适用于 depth_one_trees 生成的特定结构。

    Args:
        tree (Node): 深度为 1 的树的根节点。

    Returns:
        str: Alpha 表达式字符串。
    """
    # 添加中文注释：检查左右子节点是否存在，防止None.value的AttributeError
    if tree.left is None or tree.right is None:
        # 或者可以抛出更具体的错误
        # raise ValueError("深度为1的树的左右子节点不能为空")
        # 根据原始代码行为，如果结构不符，可能会隐式出错，这里我们先打印警告并返回空
        print(f"警告: d1tree_to_alpha 中节点 '{tree.value}' 的子节点不完整，可能导致错误。")
        return "" # 或者根据严格模式抛出异常
    return f"{tree.value}({tree.left.value},{tree.right.value})"

def d2tree_to_alpha(tree: Node) -> str:
    """
    将深度为 2 的树结构转换为 Alpha 表达式字符串。
    此函数仅适用于 depth_two_tree 生成的特定结构。

    Args:
        tree (Node): 深度为 2 的树的根节点。

    Returns:
        str: Alpha 表达式字符串。
    """
    # 添加中文注释：检查树的结构是否符合预期，防止AttributeError
    if tree.left is None or tree.right is None:
        print(f"警告: d2tree_to_alpha 中节点 '{tree.value}' 的子节点不完整。")
        return ""
    if tree.left.left is None or tree.left.right is None:
        print(f"警告: d2tree_to_alpha 中节点 '{tree.value}' 的左子树 '{tree.left.value}' 结构不完整。")
        return ""

    if tree.value in binary_ops:
        # 添加中文注释：处理二元操作符的情况
        # 确保右子树结构也完整
        if tree.right.left is None or tree.right.right is None:
            print(f"警告: d2tree_to_alpha 中节点 '{tree.value}' 的右子树 '{tree.right.value}' 结构不完整 (binary_ops)。")
            return ""
        return f"{tree.value}({tree.left.value}({tree.left.left.value},{tree.left.right.value}),{tree.right.value}({tree.right.left.value},{tree.right.right.value}))"
    if tree.value in ts_ops:
        # 添加中文注释：处理时间序列操作符的情况
        # ts_ops 的右子节点直接是参数值，不是一个树节点，所以 tree.right.value 即可
        return f"{tree.value}({tree.left.value}({tree.left.left.value},{tree.left.right.value}),{tree.right.value})"
    # 添加中文注释：如果操作符未知，可以返回空字符串或抛出异常
    print(f"警告: d2tree_to_alpha 遇到未知操作符 '{tree.value}'。")
    return ""


def d3tree_to_alpha(tree: Node) -> str:
    """
    将深度为 3 的树结构转换为 Alpha 表达式字符串。
    此函数根据根节点和其子节点的类型（深度 2 树的类型）进行复杂的模式匹配。
    这部分逻辑非常具体和硬编码，如果树结构有变化，可能需要大量修改。

    Args:
        tree (Node): 深度为 3 的树的根节点。

    Returns:
        str: Alpha 表达式字符串，如果结构不匹配则返回空字符串。
    """
    # 添加中文注释：此函数逻辑复杂，严格依赖输入树的特定结构。
    # 以下是直接从 code.py 迁移过来的逻辑，保持原样。
    # 对所有访问 .left, .right, .value 的地方都需要注意检查Optional类型和AttributeError
    # 为简化，这里假设输入的tree结构总是符合预期，否则原始代码也会出错

    # 检查基本结构
    if tree is None or tree.left is None:
        print(f"警告: d3tree_to_alpha 的输入树 '{tree.value if tree else 'None'}' 或其左子树为空。")
        return ""
    # 对于一元操作符，右子节点应为None
    if tree.value in unary_ops and tree.right is not None:
        print(f"警告: d3tree_to_alpha 中一元操作符 '{tree.value}' 不应有右子节点。")
        # 根据原代码行为，可能继续执行，这里可以视情况处理
    # 对于非一元操作符，如果右子节点是必须的但却为None
    elif tree.value not in unary_ops and tree.right is None:
        print(f"警告: d3tree_to_alpha 中非一元操作符 '{tree.value}' 的右子节点为空。")
        return ""

    # 进一步检查下一层结构
    if tree.left.left is None or tree.left.right is None:
        print(f"警告: d3tree_to_alpha 中节点 '{tree.value}' 的左子树 '{tree.left.value}' 结构不完整。")
        return ""

    # 根节点是二元操作符
    if tree.value in binary_ops:
        if tree.right is None or tree.right.left is None or tree.right.right is None: # 检查右子树的下一层
            print(f"警告: d3tree_to_alpha 中二元操作符 '{tree.value}' 的右子树结构不完整。")
            return ""
        # 左右子节点都是深度2的二元操作符树
        if tree.left.value in binary_ops and tree.right.value in binary_ops:
            # 检查更深层结构
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.left.right.left is None or tree.left.right.right is None or \
               tree.right.left.left is None or tree.right.left.right is None or \
               tree.right.right.left is None or tree.right.right.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}' 的二元子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}({tree.right.right.left.value},{tree.right.right.right.value})))"
        # 左右子节点都是深度2的时间序列操作符树
        if tree.left.value in ts_ops and tree.right.value in ts_ops:
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.right.left.left is None or tree.right.left.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}' 的TS子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}))"
        # 左子节点是深度2的二元操作符树，右子节点是深度2的时间序列操作符树
        if tree.left.value in binary_ops and tree.right.value in ts_ops:
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.left.right.left is None or tree.left.right.right is None or \
               tree.right.left.left is None or tree.right.left.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}' 的混合子树(bin,ts)深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}))"
        # 左子节点是深度2的时间序列操作符树，右子节点是深度2的二元操作符树
        if tree.left.value in ts_ops and tree.right.value in binary_ops:
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.right.left.left is None or tree.right.left.right is None or \
               tree.right.right.left is None or tree.right.right.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}' 的混合子树(ts,bin)深层结构不完整。")
                return ""
            # 原代码中此处有一个多余的括号，已修正
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}({tree.right.right.left.value},{tree.right.right.right.value})))"

    # 根节点是时间序列操作符
    if tree.value in ts_ops:
        if tree.right is None: # ts_ops 必须有右子节点（参数）
             print(f"警告: d3tree_to_alpha 中TS操作符 '{tree.value}' 的右子节点(参数)为空。")
             return ""
        # 左子节点是深度2的二元操作符树
        if tree.left.value in binary_ops:
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.left.right.left is None or tree.left.right.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}({tree.left.value}, ...)' 的二元子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value})"
        # 左子节点是深度2的时间序列操作符树
        if tree.left.value in ts_ops:
            if tree.left.left.left is None or tree.left.left.right is None:
                print(f"警告: d3tree_to_alpha 中 '{tree.value}({tree.left.value}, ...)' 的TS子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),{tree.right.value})"

    # 根节点是一元操作符
    if tree.value in unary_ops:
        # 左子节点是深度2的二元操作符树
        if tree.left.value in binary_ops:
            if tree.left.left.left is None or tree.left.left.right is None or \
               tree.left.right.left is None or tree.left.right.right is None:
                print(f"警告: d3tree_to_alpha 中一元操作 '{tree.value}({tree.left.value})' 的二元子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})))"
        # 左子节点是深度2的时间序列操作符树
        if tree.left.value in ts_ops:
            if tree.left.left.left is None or tree.left.left.right is None:
                print(f"警告: d3tree_to_alpha 中一元操作 '{tree.value}({tree.left.value})' 的TS子树深层结构不完整。")
                return ""
            return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}))"

    # 添加中文注释：如果树的结构不符合任何已知模式，则返回空字符串
    print(f"警告: d3tree_to_alpha 未能匹配任何已知模式，树根: '{tree.value}'，左孩子: '{tree.left.value if tree.left else 'None'}'。")
    return ""

def parse_expression(listr: List[str]) -> List[List[str]]:
    """
    解析 Alpha 表达式字符串列表，提取操作符和操作数。
    例如: "add(close,open)" -> ["add", "close", "open"]

    Args:
        listr (List[str]): Alpha 表达式字符串列表。

    Returns:
        List[List[str]]: 包含每个表达式解析后组件的列表的列表。
    """
    arr = []
    # 添加中文注释：正则表达式用于匹配操作符、操作数（包括数字和单词）以及括号。
    # 更新了正则表达式以更精确地匹配 ts_操作符 和其他特定操作符
    pattern_str = r'(' + '|'.join(re.escape(op) for op in ts_ops + binary_ops + unary_ops + terminal_values + ts_ops_values) + r'|\w+|[+\-*/]|[0-9]+|\(|\))'
    pattern = re.compile(pattern_str)

    for i in range(len(listr)):
        # 添加中文注释：确保处理的是字符串类型
        if not isinstance(listr[i], str):
            # print(f"警告: parse_expression 接收到非字符串输入: {listr[i]}，已跳过。")
            continue # 跳过非字符串输入

        matches = pattern.findall(listr[i])
        # 添加中文注释：过滤掉空字符串和括号本身，只保留有效组件。
        components = [match for match in matches if match and match != '(' and match != ')']
        if components: # 只有当解析出有效组件时才添加
            arr.append(components)
        # else:
            # print(f"警告: parse_expression 未能从 '{listr[i]}' 解析出有效组件。")
    return arr

def d1_alpha_to_tree(alphas: List[str]) -> List[Node]:
    """
    将深度为 1 的 Alpha 表达式字符串列表转换为树结构列表。

    Args:
        alphas (List[str]): 深度为 1 的 Alpha 表达式字符串列表。

    Returns:
        List[Node]: 转换后的树结构列表。
    """
    trees: List[Node] = [] # 明确类型
    parsed_alphas = parse_expression(alphas) # 调用上面定义的解析函数
    for i, alp_components in enumerate(parsed_alphas): # 使用 enumerate 获取原始alpha字符串用于可能的警告信息
        # 添加中文注释：确保解析后的组件列表长度至少为3（操作符，操作数1，操作数2）
        if len(alp_components) >= 3:
            node = Node(alp_components[0]) # 根节点是操作符
            node.left = Node(alp_components[1]) # 左子节点
            node.right = Node(alp_components[2]) # 右子节点
            trees.append(node)
        else:
            # 添加中文注释：如果组件数量不足，记录警告或错误
            original_alpha_str = alphas[i] if i < len(alphas) else "未知原始表达式"
            print(f"警告: 无法解析深度 1 Alpha: '{original_alpha_str}' -> {alp_components}，组件不足，已跳过。")
    return trees

def d2_alpha_to_tree(alpha_list: List[str]) -> List[Node]:
    """
    将深度为 2 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数仅适用于 depth_two_tree 生成的特定结构。

    Args:
        alpha_list (List[str]): 深度为 2 的 Alpha 表达式字符串列表。

    Returns:
        List[Node]: 转换后的树结构列表。
    """
    trees: List[Node] = [] # 明确类型
    parsed_alphas = parse_expression(alpha_list)
    for i, ar_components in enumerate(parsed_alphas):
        # 添加中文注释：根据操作符类型和组件数量判断树的结构
        if not ar_components: # 如果解析结果为空列表
            original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
            print(f"警告: 深度 2 Alpha '{original_alpha_str}' 解析结果为空，已跳过。")
            continue

        op = ar_components[0]
        # 深度为 2 的时间序列操作符树: ts_op(d1_tree_left, ts_param)
        # 解析后组件: [ts_op, op_d1, val_d1_left, val_d1_right, ts_param] (5个元素)
        if op in ts_ops and len(ar_components) >= 5:
            node = Node(op)
            node.left = Node(ar_components[1])      # 深度 1 树的根节点 (op_d1)
            node.left.left = Node(ar_components[2])  # 深度 1 树的左子节点 (val_d1_left)
            node.left.right = Node(ar_components[3]) # 深度 1 树的右子节点 (val_d1_right)
            node.right = Node(ar_components[4])     # 时间序列参数 (ts_param)
            trees.append(node)
        # 深度为 2 的二元操作符树: op(d1_tree_left, d1_tree_right)
        # 解析后组件: [op, op_d1_left, val_d1_left_left, val_d1_left_right, op_d1_right, val_d1_right_left, val_d1_right_right] (7个元素)
        elif op in binary_ops and len(ar_components) >= 7:
            node = Node(op)
            node.left = Node(ar_components[1])       # 左侧深度 1 树的根节点 (op_d1_left)
            node.left.left = Node(ar_components[2])   # 左侧深度 1 树的左子节点 (val_d1_left_left)
            node.left.right = Node(ar_components[3])  # 左侧深度 1 树的右子节点 (val_d1_left_right)
            node.right = Node(ar_components[4])      # 右侧深度 1 树的根节点 (op_d1_right)
            node.right.left = Node(ar_components[5])  # 右侧深度 1 树的左子节点 (val_d1_right_left)
            node.right.right = Node(ar_components[6]) # 右侧深度 1 树的右子节点 (val_d1_right_right)
            trees.append(node)
        else:
            original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
            # 添加中文注释：记录无法解析的情况
            print(f"警告: 无法解析深度 2 Alpha: '{original_alpha_str}' -> {ar_components}，结构不匹配或组件不足，已跳过。")
    return trees

def d3_alpha_to_tree(alpha_list: List[str]) -> List[Node]:
    """
    将深度为 3 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数也高度依赖硬编码的模式匹配，需要确保输入表达式的结构严格符合预期。

    Args:
        alpha_list (List[str]): 深度为 3 的 Alpha 表达式字符串列表。

    Returns:
        List[Node]: 转换后的树结构列表。
    """
    trees: List[Node] = [] # 明确类型
    parsed_alphas = parse_expression(alpha_list)
    for i, ar_components in enumerate(parsed_alphas):
        # 添加中文注释：此函数逻辑复杂，直接从 code.py 迁移并适配。
        # 需要非常小心地处理索引和组件数量。
        if not ar_components:
            original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
            print(f"警告: 深度 3 Alpha '{original_alpha_str}' 解析结果为空，已跳过。")
            continue

        op_d3 = ar_components[0] # 深度3的根操作符

        # 根节点是一元操作符 (rank, zscore, winsorize, etc.)
        # 结构: UnaryOp( D2_Tree )
        if op_d3 in unary_ops:
            # D2_Tree 是 TS 操作符: UnaryOp( TS_Op_D2( D1_Tree, param_d2 ) )
            # 解析: [op_d3, op_ts_d2, op_d1, val_d1_l, val_d1_r, param_d2] (6个元素)
            if len(ar_components) >= 6 and ar_components[1] in ts_ops:
                node = Node(op_d3)
                node.left = Node(ar_components[1])          # op_ts_d2
                node.left.left = Node(ar_components[2])     # op_d1
                node.left.left.left = Node(ar_components[3])# val_d1_l
                node.left.left.right = Node(ar_components[4])# val_d1_r
                node.left.right = Node(ar_components[5])    # param_d2
                trees.append(node)
            # D2_Tree 是二元操作符: UnaryOp( Bin_Op_D2( D1_Tree_L, D1_Tree_R ) )
            # 解析: [op_d3, op_bin_d2, op_d1_L, v_d1L_l, v_d1L_r, op_d1_R, v_d1R_l, v_d1R_r] (8个元素)
            elif len(ar_components) >= 8 and ar_components[1] in binary_ops:
                node = Node(op_d3)
                node.left = Node(ar_components[1])              # op_bin_d2
                node.left.left = Node(ar_components[2])         # op_d1_L
                node.left.left.left = Node(ar_components[3])    # v_d1L_l
                node.left.left.right = Node(ar_components[4])   # v_d1L_r
                node.left.right = Node(ar_components[5])        # op_d1_R
                node.left.right.left = Node(ar_components[6])   # v_d1R_l
                node.left.right.right = Node(ar_components[7])  # v_d1R_r
                trees.append(node)
            else:
                original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
                print(f"警告: 无法解析深度 3 一元操作符 Alpha: '{original_alpha_str}' -> {ar_components}，结构不匹配，已跳过。")

        # 根节点是时间序列操作符 (ts_ops)
        # 结构: TS_Op_D3( D2_Tree, param_d3 )
        elif op_d3 in ts_ops:
            # D2_Tree 是 TS 操作符: TS_Op_D3( TS_Op_D2( D1_Tree, param_d2 ), param_d3 )
            # 解析: [op_ts_d3, op_ts_d2, op_d1, val_d1_l, val_d1_r, param_d2, param_d3] (7个元素)
            if len(ar_components) >= 7 and ar_components[1] in ts_ops:
                node = Node(op_d3)
                node.left = Node(ar_components[1])          # op_ts_d2
                node.left.left = Node(ar_components[2])     # op_d1
                node.left.left.left = Node(ar_components[3])# val_d1_l
                node.left.left.right = Node(ar_components[4])# val_d1_r
                node.left.right = Node(ar_components[5])    # param_d2
                node.right = Node(ar_components[6])         # param_d3
                trees.append(node)
            # D2_Tree 是二元操作符: TS_Op_D3( Bin_Op_D2( D1_Tree_L, D1_Tree_R ), param_d3 )
            # 解析: [op_ts_d3, op_bin_d2, op_d1_L, v_d1L_l, v_d1L_r, op_d1_R, v_d1R_l, v_d1R_r, param_d3] (9个元素)
            elif len(ar_components) >= 9 and ar_components[1] in binary_ops:
                node = Node(op_d3)
                node.left = Node(ar_components[1])              # op_bin_d2
                node.left.left = Node(ar_components[2])         # op_d1_L
                node.left.left.left = Node(ar_components[3])    # v_d1L_l
                node.left.left.right = Node(ar_components[4])   # v_d1L_r
                node.left.right = Node(ar_components[5])        # op_d1_R
                node.left.right.left = Node(ar_components[6])   # v_d1R_l
                node.left.right.right = Node(ar_components[7])  # v_d1R_r
                node.right = Node(ar_components[8])             # param_d3
                trees.append(node)
            else:
                original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
                print(f"警告: 无法解析深度 3 时间序列操作符 Alpha: '{original_alpha_str}' -> {ar_components}，结构不匹配，已跳过。")

        # 根节点是二元操作符 (binary_ops)
        # 结构: Bin_Op_D3( D2_Tree_L, D2_Tree_R )
        elif op_d3 in binary_ops:
            # 尝试找到 D2_Tree_L 和 D2_Tree_R 的分界点。这部分比较复杂，因为 D2_Tree 的长度可变。
            # 我们需要根据 D2_Tree_L 的类型 (TS 或 Binary) 来确定其长度。
            # D2_Tree_L 是 TS: Bin_Op_D3( TS_Op_D2_L(D1_L, p_d2L), D2_Tree_R )
            # 解析的开始部分: [op_bin_d3, op_ts_d2L, op_d1L, v_d1L_l, v_d1L_r, p_d2L, ...] (D2_Tree_L 占5个元素)
            if len(ar_components) > 1 + 5 and ar_components[1] in ts_ops: # D2_L is TS
                d2_L_len = 5
                op_d2_R_start_index = 1 + d2_L_len

                # D2_Tree_R 是 TS: Bin_Op_D3( TS_L, TS_R )
                # 总长度: 1(op_d3) + 5(TS_L) + 5(TS_R) = 11
                if len(ar_components) >= op_d2_R_start_index + 5 and ar_components[op_d2_R_start_index] in ts_ops:
                    node = Node(op_d3)
                    # Left D2 TS Tree
                    node.left = Node(ar_components[1]) # op_ts_d2L
                    node.left.left = Node(ar_components[2]) # op_d1L
                    node.left.left.left = Node(ar_components[3]) # v_d1L_l
                    node.left.left.right = Node(ar_components[4]) # v_d1L_r
                    node.left.right = Node(ar_components[5]) # p_d2L
                    # Right D2 TS Tree
                    node.right = Node(ar_components[op_d2_R_start_index]) # op_ts_d2R
                    node.right.left = Node(ar_components[op_d2_R_start_index+1]) # op_d1R
                    node.right.left.left = Node(ar_components[op_d2_R_start_index+2]) # v_d1R_l
                    node.right.left.right = Node(ar_components[op_d2_R_start_index+3]) # v_d1R_r
                    node.right.right = Node(ar_components[op_d2_R_start_index+4]) # p_d2R
                    trees.append(node)

                # D2_Tree_R 是 Binary: Bin_Op_D3( TS_L, Bin_R )
                # 总长度: 1(op_d3) + 5(TS_L) + 7(Bin_R) = 13
                elif len(ar_components) >= op_d2_R_start_index + 7 and ar_components[op_d2_R_start_index] in binary_ops:
                    node = Node(op_d3)
                    # Left D2 TS Tree
                    node.left = Node(ar_components[1])
                    node.left.left = Node(ar_components[2])
                    node.left.left.left = Node(ar_components[3])
                    node.left.left.right = Node(ar_components[4])
                    node.left.right = Node(ar_components[5])
                    # Right D2 Binary Tree
                    node.right = Node(ar_components[op_d2_R_start_index]) # op_bin_d2R
                    node.right.left = Node(ar_components[op_d2_R_start_index+1]) # op_d1RL
                    node.right.left.left = Node(ar_components[op_d2_R_start_index+2]) # v_d1RL_l
                    node.right.left.right = Node(ar_components[op_d2_R_start_index+3]) # v_d1RL_r
                    node.right.right = Node(ar_components[op_d2_R_start_index+4]) # op_d1RR
                    node.right.right.left = Node(ar_components[op_d2_R_start_index+5]) # v_d1RR_l
                    node.right.right.right = Node(ar_components[op_d2_R_start_index+6]) # v_d1RR_r
                    trees.append(node)
                else:
                    original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
                    print(f"警告: 无法解析深度 3 二元操作符 Alpha (左TS，右未知): '{original_alpha_str}' -> {ar_components}，结构不匹配，已跳过。")


            # D2_Tree_L 是 Binary: Bin_Op_D3( Bin_Op_D2_L(D1_LL, D1_LR), D2_Tree_R )
            # 解析的开始部分: [op_bin_d3, op_bin_d2L, op_d1LL, v_d1LL_l, v_d1LL_r, op_d1LR, v_d1LR_l, v_d1LR_r, ...] (D2_Tree_L 占7个元素)
            elif len(ar_components) > 1 + 7 and ar_components[1] in binary_ops: # D2_L is Binary
                d2_L_len = 7
                op_d2_R_start_index = 1 + d2_L_len

                # D2_Tree_R 是 TS: Bin_Op_D3( Bin_L, TS_R )
                # 总长度: 1(op_d3) + 7(Bin_L) + 5(TS_R) = 13
                if len(ar_components) >= op_d2_R_start_index + 5 and ar_components[op_d2_R_start_index] in ts_ops:
                    node = Node(op_d3)
                    # Left D2 Binary Tree
                    node.left = Node(ar_components[1]) # op_bin_d2L
                    node.left.left = Node(ar_components[2]) # op_d1LL
                    node.left.left.left = Node(ar_components[3]) # v_d1LL_l
                    node.left.left.right = Node(ar_components[4]) # v_d1LL_r
                    node.left.right = Node(ar_components[5]) # op_d1LR
                    node.left.right.left = Node(ar_components[6]) # v_d1LR_l
                    node.left.right.right = Node(ar_components[7]) # v_d1LR_r
                    # Right D2 TS Tree
                    node.right = Node(ar_components[op_d2_R_start_index]) # op_ts_d2R
                    node.right.left = Node(ar_components[op_d2_R_start_index+1]) # op_d1R
                    node.right.left.left = Node(ar_components[op_d2_R_start_index+2]) # v_d1R_l
                    node.right.left.right = Node(ar_components[op_d2_R_start_index+3]) # v_d1R_r
                    node.right.right = Node(ar_components[op_d2_R_start_index+4]) # p_d2R
                    trees.append(node)

                # D2_Tree_R 是 Binary: Bin_Op_D3( Bin_L, Bin_R )
                # 总长度: 1(op_d3) + 7(Bin_L) + 7(Bin_R) = 15
                elif len(ar_components) >= op_d2_R_start_index + 7 and ar_components[op_d2_R_start_index] in binary_ops:
                    node = Node(op_d3)
                    # Left D2 Binary Tree
                    node.left = Node(ar_components[1])
                    node.left.left = Node(ar_components[2])
                    node.left.left.left = Node(ar_components[3])
                    node.left.left.right = Node(ar_components[4])
                    node.left.right = Node(ar_components[5])
                    node.left.right.left = Node(ar_components[6])
                    node.left.right.right = Node(ar_components[7])
                    # Right D2 Binary Tree
                    node.right = Node(ar_components[op_d2_R_start_index]) # op_bin_d2R
                    node.right.left = Node(ar_components[op_d2_R_start_index+1]) # op_d1RL
                    node.right.left.left = Node(ar_components[op_d2_R_start_index+2]) # v_d1RL_l
                    node.right.left.right = Node(ar_components[op_d2_R_start_index+3]) # v_d1RL_r
                    node.right.right = Node(ar_components[op_d2_R_start_index+4]) # op_d1RR
                    node.right.right.left = Node(ar_components[op_d2_R_start_index+5]) # v_d1RR_l
                    node.right.right.right = Node(ar_components[op_d2_R_start_index+6]) # v_d1RR_r
                    trees.append(node)
                else:
                    original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
                    print(f"警告: 无法解析深度 3 二元操作符 Alpha (左Bin，右未知): '{original_alpha_str}' -> {ar_components}，结构不匹配，已跳过。")
            else:
                original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
                print(f"警告: 无法解析深度 3 二元操作符 Alpha (左未知): '{original_alpha_str}' -> {ar_components}，结构不匹配，已跳过。")
        else:
            original_alpha_str = alpha_list[i] if i < len(alpha_list) else "未知原始表达式"
            # 添加中文注释：记录其他所有未知或不完整的情况
            print(f"警告: 未知根操作符或不完整表达式用于深度 3 Alpha: '{original_alpha_str}' -> {ar_components}，已跳过。")

    return trees
