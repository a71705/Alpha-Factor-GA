# alpha_factory/fitness/base_fitness.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any # 确保导入了需要的类型
import pandas as pd # 虽然基类不用，但子类会用，放在这里供参考

class BaseFitnessCalculator(ABC):
    """
    适应度计算器的抽象基类。
    所有具体的适应度计算器都应继承此类并实现其抽象方法。
    """

    @abstractmethod
    def run(self, results: List[Dict[str, Any]], n: int) -> List[str]:
        """
        运行适应度计算。

        Args:
            results (List[Dict[str, Any]]): 从模拟或评估中获得的原始结果列表。
                                           每个字典代表一个alpha的结果，
                                           具体结构依赖于BrainApiClient的返回。
            n (int): 需要选择的最佳Alpha的数量。

        Returns:
            List[str]: 经过适应度评估后，排名前n的Alpha表达式列表。
        """
        pass
