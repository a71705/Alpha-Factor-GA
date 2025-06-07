# alpha_factory/genetic_programming/models.py
from __future__ import annotations
from typing import Optional, Dict, Any

class Node:
    """定义Alpha表达式树的节点，严格遵循code.py的双子节点结构。

    Attributes:
        value (str): 节点存储的值，可以是操作符、终端值或参数。
        left (Optional[Node]): 左子节点。
        right (Optional[Node]): 右子节点。
    """
    def __init__(self, value: str):
        """初始化节点。

        Args:
            value (str): 节点的值。
        """
        self.value: str = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def __repr__(self) -> str:
        """返回节点的字符串表示形式，方便调试。

        Returns:
            str: 节点的字符串表示。
        """
        return f"Node('{self.value}')"

class AlphaIndividual:
    """封装单个Alpha个体所需的所有信息。

    Attributes:
        tree (Node): Alpha表达式的树结构表示。
        expression (Optional[str]): Alpha表达式的字符串表示。
        stats (Optional[Dict[str, Any]]): Alpha的统计数据，例如sharpe, fitness等。
        fitness (float): Alpha的适应度得分。
        is_evaluated (bool): 标记Alpha是否已经被评估。
        node_count (int): Alpha树中的节点数量。
    """
    def __init__(self, tree: Node):
        """初始化Alpha个体。

        Args:
            tree (Node): Alpha表达式的树结构。
        """
        self.tree: Node = tree
        self.expression: Optional[str] = None
        self.stats: Optional[Dict[str, Any]] = None # 更具体的类型可以是 Dict[str, float] 或更复杂的模型
        self.fitness: float = -float('inf') # 初始化为负无穷大，表示尚未评估或适应度极低
        self.is_evaluated: bool = False
        self.node_count: int = 0 # 节点数量可以在树构建或转换时计算
