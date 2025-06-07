# alpha_factory/utils/config_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class BrainSettings(BaseModel):
    """BRAIN平台模拟参数配置。

    Attributes:
        region (str): 模拟区域，例如 "USA", "CHN" 等。
        universe (str): 股票池，例如 "TOP3000", "HS300" 等。
        delay (int): 数据延迟天数。
        truncation (float): 截断比例，用于处理极端值。
        neutralization (str): 中性化方法，例如 "INDUSTRY", "MARKET" 等。
        decay (int): 衰减天数，用于计算持仓权重。
        nan_handling (str): NaN值（缺失数据）的处理方式，例如 "OFF", "FILL_ZERO" 等。
        unit_handling (str): 单位处理方式。
        pasteurization (str): 巴氏消毒开关，用于控制未来数据的偷看。
    """
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    truncation: float = 0.08
    neutralization: str = "INDUSTRY"
    decay: int = 0
    nan_handling: str = "OFF"  # 注意：旧代码中为 "OFF"，但平台可能用 "OFF" 或其他
    unit_handling: str = "VERIFY"
    pasteurization: str = "ON"

class GPParams(BaseModel):
    """遗传编程（Genetic Programming）核心参数配置。

    Attributes:
        d1_population (int): 深度1阶段的种群数量。
        d1_generations (int): 深度1阶段的迭代代数。
        d2_population (int): 深度2阶段的种群数量。
        d2_generations (int): 深度2阶段的迭代代数。
        d3_population (int): 深度3阶段的种群数量。
        d3_generations (int): 深度3阶段的迭代代数。
    """
    d1_population: int
    d1_generations: int
    d2_population: int
    d2_generations: int
    d3_population: int
    d3_generations: int

class OperatorConfig(BaseModel):
    """遗传编程中使用的操作符配置。

    Attributes:
        terminal_values (List[str]): 终端节点的可选值列表，如 "close", "open"。
        ts_ops (List[str]): 时间序列操作符列表，如 "ts_zscore", "ts_rank"。
        binary_ops (List[str]): 二元操作符列表，如 "add", "subtract"。
        ts_ops_values (List[str]): 时间序列操作符的参数值列表，如 "20", "40"。
        unary_ops (List[str]): 一元操作符列表，如 "rank", "zscore"。
    """
    terminal_values: List[str]
    ts_ops: List[str]
    binary_ops: List[str]
    ts_ops_values: List[str]
    unary_ops: List[str]

class FitnessConfig(BaseModel):
    """适应度函数配置。

    Attributes:
        module (str): 适应度计算器所在的模块路径，例如 "alpha_factory.fitness.legacy_fitness"。
        class_name (str): 适应度计算器的类名，例如 "LegacyFitness"。
        params (Dict[str, Any]): 传递给适应度计算器构造函数的额外参数。
    """
    module: str
    class_name: str = Field(..., alias='class') # 允许在YAML中使用 'class' 作为键
    params: Dict[str, Any] = {}

class AlgorithmConfig(BaseModel):
    """核心算法切换配置。

    Attributes:
        engine (str): 使用的遗传编程引擎类型，例如 "staged" (分阶段引擎)。
        crossover (str): 使用的交叉算子类型，例如 "legacy" (旧版交叉算子)。
        mutation (str): 使用的变异算子类型，例如 "legacy" (旧版变异算子)。
        generator (str): 使用的表达式生成器类型，例如 "legacy" (旧版生成器)。
        converter (str): 使用的表达式转换器类型，例如 "legacy" (旧版转换器)。
    """
    engine: str = "staged"
    crossover: str = "legacy"
    mutation: str = "legacy"
    generator: str = "legacy"
    converter: str = "legacy"

class AppConfig(BaseModel):
    """应用程序主配置模型，聚合所有子配置。

    Attributes:
        experiment_name (str): 实验的名称。
        brain (BrainSettings): BRAIN平台相关的配置。
        gp (GPParams): 遗传编程参数配置。
        fitness (FitnessConfig): 适应度函数配置。
        operators (OperatorConfig): 操作符配置。
        algorithm (AlgorithmConfig): 算法切换配置。
    """
    experiment_name: str
    brain: BrainSettings
    gp: GPParams
    fitness: FitnessConfig
    operators: OperatorConfig
    algorithm: AlgorithmConfig = AlgorithmConfig()
