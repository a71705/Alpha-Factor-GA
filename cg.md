

## **AI驱动开发指南："AlphaFactory"量化框架 (v7.0 - 终极施工蓝图)**

**项目目标:** 构建一个**行为上与`code.py`完全对齐**，但**架构上模块化、用户友好且可扩展**的自动化Alpha发现框架。框架必须支持通过配置切换核心算法，以进行科学的A/B测试。

**核心指令:** AI开发者，请严格按照以下任务卡片的顺序进行开发。每个任务都是前一个任务的增量，确保在进入下一步前，当前任务已通过所有测试。本文档是唯一的参考源。

---

### **第一阶段：地基工程 (Foundation Engineering)**

*此阶段旨在构建项目的基本骨架和数据模型。*

**【任务卡片 1.1】: 项目结构与环境设置**
*   **指令:**
    1.  创建一个新的Python虚拟环境并激活。
    2.  执行 `pip install "typer[all]" rich questionary pyyaml pydantic requests pandas` 安装所有依赖。
    3.  创建以下目录结构，包括所有空的`__init__.py`文件：
        ```
        alpha_factory/
        ├── __init__.py
        ├── main.py
        ├── cli/
        │   ├── __init__.py
        │   └── menus.py
        ├── configs/
        │   ├── __init__.py
        │   ├── legacy_default_template.yaml
        │   └── generic_default_template.yaml
        ├── brain_client/
        │   ├── __init__.py
        │   ├── session_manager.py
        │   └── api_client.py
        ├── genetic_programming/
        │   ├── __init__.py
        │   ├── models.py
        │   ├── engine.py
        │   ├── operators/
        │   │   ├── __init__.py
        │   │   ├── legacy_operators.py
        │   │   └── generic_operators.py
        │   └── generators/
        │       ├── __init__.py
        │       ├── legacy_generator.py
        │       └── generic_generator.py
        ├── fitness/
        │   ├── __init__.py
        │   ├── base_fitness.py
        │   └── legacy_fitness.py
        └── utils/
            ├── __init__.py
            ├── config_models.py
            ├── config_loader.py
            ├── legacy_expression_converter.py
            └── generic_parser.py
        ```
*   **验收标准:** 环境配置完毕，目录结构正确。

**【任务卡片 1.2】: 核心数据模型 (`genetic_programming/models.py`)**
*   **指令:** 创建`genetic_programming/models.py`，定义`Node`和`AlphaIndividual`。`Node`必须使用`left`和`right`属性以兼容`code.py`的逻辑。
*   **输出 (代码):**
    ```python
    # alpha_factory/genetic_programming/models.py
    from __future__ import annotations
    from typing import Optional, Dict

    class Node:
        """定义Alpha表达式树的节点，严格遵循code.py的双子节点结构。"""
        def __init__(self, value: str):
            self.value: str = value
            self.left: Optional[Node] = None
            self.right: Optional[Node] = None

        def __repr__(self) -> str:
            return f"Node('{self.value}')"

    class AlphaIndividual:
        """封装单个Alpha的所有信息。"""
        def __init__(self, tree: Node):
            self.tree: Node = tree
            self.expression: Optional[str] = None
            self.stats: Optional[Dict] = None
            self.fitness: float = -float('inf')
            self.is_evaluated: bool = False
            self.node_count: int = 0
    ```
*   **验收标准:** `Node`和`AlphaIndividual`类已正确定义。

**【任务卡片 1.3】: 配置文件Pydantic模型 (`utils/config_models.py`)**
*   **指令:** 创建`utils/config_models.py`。配置模型必须支持算法切换。
*   **输出 (代码):**
    ```python
    # alpha_factory/utils/config_models.py
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any

    class BrainSettings(BaseModel):
        region: str = "USA"
        universe: str = "TOP3000"
        delay: int = 1
        truncation: float = 0.08
        neutralization: str = "INDUSTRY"
        decay: int = 0
        nan_handling: str = "OFF"
        unit_handling: str = "VERIFY"
        pasteurization: str = "ON"

    class GPParams(BaseModel):
        d1_population: int
        d1_generations: int
        d2_population: int
        d2_generations: int
        d3_population: int
        d3_generations: int

    class OperatorConfig(BaseModel):
        terminal_values: List[str]
        ts_ops: List[str]
        binary_ops: List[str]
        ts_ops_values: List[str]
        unary_ops: List[str]

    class FitnessConfig(BaseModel):
        module: str
        class_name: str = Field(..., alias='class')
        params: Dict[str, Any] = {}

    class AlgorithmConfig(BaseModel):
        engine: str = "staged"
        crossover: str = "legacy"
        mutation: str = "legacy"
        generator: str = "legacy"
        converter: str = "legacy"

    class AppConfig(BaseModel):
        experiment_name: str
        brain: BrainSettings
        gp: GPParams
        fitness: FitnessConfig
        operators: OperatorConfig
        algorithm: AlgorithmConfig = AlgorithmConfig()
    ```
*   **验收标准:** 所有模型已定义，`AppConfig`包含一个`algorithm`子模型。

**【任务卡片 1.4】: 配置加载与验证工具 (`utils/config_loader.py`)**
*   **指令:** 实现加载和验证YAML配置文件的函数。
*   **输出 (代码):**
    ```python
    # alpha_factory/utils/config_loader.py
    import yaml
    from pathlib import Path
    from pydantic import ValidationError
    from .config_models import AppConfig

    def load_config(path: Path) -> AppConfig:
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        try:
            return AppConfig.parse_obj(data)
        except ValidationError as e:
            raise ValueError(f"配置文件 '{path.name}' 格式错误: \n{e}") from e
    ```
*   **验收标准:** `load_config`在文件有效时返回`AppConfig`对象，在文件无效时抛出异常。

**【任务卡片 1.5】: 创建默认配置文件**
*   **指令:** 在`configs/`目录下创建`legacy_default_template.yaml`。
*   **输出 (文件内容):**
    ```yaml
    experiment_name: legacy_experiment
    
    brain:
      region: "USA"
      universe: "TOP3000"
      delay: 1
      truncation: 0.08
      neutralization: "INDUSTRY"

    gp:
      d1_population: 30
      d1_generations: 5
      d2_population: 25
      d2_generations: 10
      d3_population: 20
      d3_generations: 15

    fitness:
      module: "alpha_factory.fitness.legacy_fitness"
      class: "LegacyFitness"
      params:
        # 'n' will be set dynamically by the engine for each stage
        # This is a placeholder
        n: 20 

    operators:
      terminal_values: ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns", "dividend"]
      ts_ops: ["ts_zscore", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_delta", "ts_ir", "ts_mean","ts_median", "ts_product", "ts_std_dev"]
      binary_ops: ["add", "subtract", "divide", "multiply", "max", "min"]
      ts_ops_values: ["20", "40", "60", "120", "240"]
      unary_ops: ["rank", "zscore", "winsorize", "normalize", "rank_by_side", "sigmoid", "pasteurize", "log"]

    algorithm:
      engine: "staged"
      crossover: "legacy"
      mutation: "legacy"
      generator: "legacy"
      converter: "legacy"
    ```
*   **验收标准:** YAML文件已创建并包含所有必需字段。

---

### **第二阶段：核心算法与逻辑迁移 (Core Logic Migration)**

*此阶段将`code.py`的核心逻辑原封不动地迁移到新框架中。*

**【任务卡片 2.1】: `code.py`专用表达式转换器 (`utils/legacy_expression_converter.py`)**
*   **指令:** 创建`utils/legacy_expression_converter.py`，并将`code.py`中所有`d*tree_to_alpha`, `d*_alpha_to_tree`和`parse_expression`函数**原样复制**进来。
*   **验收标准:** 所有函数已迁移，代码与`code.py`完全一致。

**【任务卡片 2.2】: `code.py`专用树生成器 (`genetic_programming/generators/legacy_generator.py`)**
*   **指令:** 创建`genetic_programming/generators/legacy_generator.py`，并将`code.py`中所有`depth_*_trees`函数**原样复制**进来。
*   **验收标准:** 所有树生成函数已迁移，代码与`code.py`完全一致。

**【任务卡片 2.3】: `code.py`专用遗传算子 (`genetic_programming/operators/legacy_operators.py`)**
*   **指令:** 创建`genetic_programming/operators/legacy_operators.py`，并将`code.py`中`copy_tree`, `mutate_random_node`, 和**非常特殊**的`crossover`函数**原样复制**进来。
*   **验收标准:** 所有遗传算子已迁移，代码与`code.py`完全一致。

**【任务卡片 2.4】: `code.py`专用适应度函数 (`fitness/legacy_fitness.py` & `base_fitness.py`)**
*   **指令:**
    1.  创建`fitness/base_fitness.py`。
    2.  创建`fitness/legacy_fitness.py`，并将`code.py`中的`prettify_result`和`fitness_fun`函数迁移并封装。
*   **输出 (代码 - `base_fitness.py`):**
    ```python
    # alpha_factory/fitness/base_fitness.py
    from abc import ABC, abstractmethod
    
    class BaseFitnessCalculator(ABC):
        @abstractmethod
        def run(self, results: list, n: int) -> list[str]:
            pass
    ```
*   **输出 (代码 - `legacy_fitness.py`):**
    ```python
    # alpha_factory/fitness/legacy_fitness.py
    import pandas as pd
    from typing import List
    from .base_fitness import BaseFitnessCalculator
    
    class LegacyFitness(BaseFitnessCalculator):
        """封装 code.py 中的 prettify_result 和 fitness_fun 逻辑。"""
        def run(self, results: list, n: int) -> List[str]:
            prettified_df = self._prettify_result(results)
            if prettified_df.empty:
                return []
            return self._fitness_fun(prettified_df, n)

        def _prettify_result(self, result: list) -> pd.DataFrame:
            # --- Start of code migrated from prettify_result ---
            # ... (完全复制 code.py 中 prettify_result 的所有逻辑, 但移除最后的 style.format)
            # ... (它应该返回一个普通的 DataFrame)
            # --- End of migrated code ---
            return alpha_stats

        def _fitness_fun(self, Data: pd.DataFrame, n: int) -> List[str]:
            # --- Start of code migrated from fitness_fun ---
            # ... (完全复制 code.py 中 fitness_fun 的所有逻辑)
            # --- End of migrated code ---
            return top_n_values
    ```
*   **验收标准:** `LegacyFitness`类的行为与原始`prettify_result` + `fitness_fun`的组合一致。

---

### **第三阶段：外部接口实现 (External Interfaces)**

**【任务卡片 3.1】: BRAIN API会话管理器 (`brain_client/session_manager.py`)**
*   **指令:** 实现负责登录和维护会话的`SessionManager`。
*   **输出 (代码):**
    ```python
    # alpha_factory/brain_client/session_manager.py
    import requests, time, os, json, getpass
    from urllib.parse import urljoin
    
    class SessionManager:
        def __init__(self):
            self.session: requests.Session | None = None
            self.token_expiry: int = 0
            self.base_url = "https://api.worldquantbrain.com"

        def get_session(self) -> requests.Session:
            if self.session is None or time.time() > self.token_expiry - 300: # 提前5分钟刷新
                self._login()
            return self.session
        
        def _get_credentials(self) -> (str, str):
            # ... (完全复制 code.py 中的 get_credentials 逻辑) ...
            pass

        def _login(self):
            # ... (完整实现 code.py 中的 start_session 逻辑, 包括 Persona 认证流程) ...
            # ... (成功后，更新 self.token_expiry) ...
            pass
    ```
*   **验收标准:** `get_session()`能够返回一个已认证且有效的`requests.Session`对象。

**【任务卡片 3.2】: BRAIN API客户端 (`brain_client/api_client.py`)**
*   **指令:** 实现与BRAIN API交互的客户端，**完整复刻`code.py`中的所有API调用和并发逻辑**。
*   **输出 (代码):**
    ```python
    # alpha_factory/brain_client/api_client.py
    # ... (imports) ...
    from .session_manager import SessionManager
    
    class BrainApiClient:
        def __init__(self, session_manager: SessionManager):
            self.sm = session_manager
            self.base_url = "https://api.worldquantbrain.com"

        def _request_with_retry(self, method, url, **kwargs):
            # ... (实现一个带重试逻辑的通用请求函数) ...
            pass
            
        def run_simulation_workflow(self, alpha_sim_data_list: List[Dict], limit_concurrent: int, depth: int, iteration: int) -> List[Dict]:
            # **关键实现**:
            # 1. 复制 code.py 中 simulate_alpha_list 的全部逻辑。
            # 2. 将内部调用的 s.get/s.post 替换为 self._request_with_retry。
            # 3. 将内部调用的 simulate_single_alpha, get_specified_alpha_stats 等函数逻辑也迁移进来，
            #    或作为私有方法 _simulate_single_alpha, _get_specified_alpha_stats 实现。
            # 4. 确保所有参数（如 simulation_config）都从方法参数或 self.config 中获取。
            # 5. 返回的结果列表结构必须与原始函数完全一致。
            pass
    ```
*   **验收标准:** `run_simulation_workflow`的行为和产出与原始`simulate_alpha_list`完全相同。

---

### **第四阶段：核心引擎实现 (Engine Implementation)**

**【任务卡片 4.1】: GP分阶段引擎 (`genetic_programming/engine.py`)**
*   **指令:** 实现`StagedGPEngine`，它将**编排（Orchestrate）**所有迁移过来的`legacy`模块，以复刻`code.py`的完整流程。
*   **输出 (代码):**
    ```python
    # alpha_factory/genetic_programming/engine.py
    import importlib, random
    from ..utils.config_models import AppConfig
    from ..brain_client.api_client import BrainApiClient
    from ..fitness.base_fitness import BaseFitnessCalculator
    from .generators import legacy_generator as gen
    from .operators import legacy_operators as ops
    from ..utils import legacy_expression_converter as conv

    class StagedGPEngine:
        def __init__(self, config: AppConfig, api_client: BrainApiClient):
            self.config = config
            self.api_client = api_client
            self.operators = config.operators
            self.fitness_calculator = self._load_fitness_calculator()
        
        def _load_fitness_calculator(self) -> BaseFitnessCalculator:
            # ... (实现动态加载 fitness 模块) ...
            pass

        def run(self):
            best_d1 = self._run_d1_stage()
            best_d2 = self._run_d2_stage(best_d1)
            best_d3 = self._run_d3_stage(best_d2)
            print("--- 最终最佳Alpha (深度3) ---")
            print(best_d3)

        def _run_d1_stage(self) -> List[str]:
            # **完整复刻 code.py 中 best_d1_alphas 的逻辑**
            # 1. 使用 gen.depth_one_trees 和 conv.d1tree_to_alpha 生成初始种群。
            # 2. 调用 self.api_client.run_simulation_workflow 进行模拟。
            # 3. 调用 self.fitness_calculator.run 进行评估和选择。
            # 4. 循环迭代，补充种群。
            # 5. 最后进行变异阶段，使用 ops.mutate_random_node。
            # 6. 返回最终的最佳Alpha表达式列表。
            pass

        def _run_d2_stage(self, onetree_exprs: List[str]) -> List[str]:
            # **完整复刻 code.py 中 best_d2_alphas 的逻辑**
            # 1. 使用 conv.d1_alpha_to_tree 将上一阶段结果转为树。
            # 2. 使用 gen.depth_two_tree 生成初始种群。
            # ... (后续流程与 _run_d1_stage 类似, 但包含交叉 ops.crossover) ...
            pass

        def _run_d3_stage(self, twotree_exprs: List[str]) -> List[str]:
            # **完整复刻 code.py 中 best_d3_alpha 的逻辑**
            # ... (类似地，复刻 best_d3_alpha 的完整流程) ...
            pass
    ```
*   **验收标准:** `StagedGPEngine`的`run`方法能够按顺序执行三个阶段的进化，其最终产出与`code.py`的执行结果一致。

---

### **第五阶段：用户界面与最终组装 (UI & Final Assembly)**

**【任务卡片 5.1】: CLI菜单实现 (`cli/menus.py`)**
*   **指令:** 完整实现所有交互式菜单函数。
*   **输出 (代码):**
    ```python
    # alpha_factory/cli/menus.py
    import questionary, shutil
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    
    CONFIG_DIR = Path("configs")

    def select_experiment() -> Path | None:
        # ... (使用 questionary.select 实现) ...

    def create_new_config():
        # ... (使用 questionary.text 和 shutil.copy 实现) ...

    def list_available_configs():
        # ... (使用 rich.table 实现) ...
    ```
*   **验收标准:** 所有菜单功能正常，交互流畅。

**【任务卡片 5.2】: CLI主程序 (`main.py`)**
*   **指令:** 最终完成`main.py`，连接所有组件。
*   **输出 (代码):**
    ```python
    # alpha_factory/main.py
    import typer
    from pathlib import Path
    from rich.console import Console
    from .cli import menus
    from .utils.config_loader import load_config
    from .genetic_programming.engine import StagedGPEngine # 默认使用Staged引擎
    from .brain_client.session_manager import SessionManager
    from .brain_client.api_client import BrainApiClient

    app = typer.Typer(help="AlphaFactory - 自动化Alpha发现框架")
    console = Console()

    @app.command()
    def run(config_path: Path = typer.Option(None, "-c", help="直接指定配置文件。")):
        """运行一个遗传编程实验。"""
        if not config_path:
            config_path = menus.select_experiment()
            if not config_path:
                return

        try:
            console.print(f"🔩 正在加载配置: [cyan]{config_path.name}[/cyan]")
            app_config = load_config(config_path)
            
            console.print("🤝 正在初始化API客户端...")
            session_manager = SessionManager()
            api_client = BrainApiClient(session_manager)
            
            # **注意**: 此处直接实例化StagedGPEngine以保证与code.py行为一致
            engine = StagedGPEngine(app_config, api_client)
            
            console.print(f"🚀 正在启动 '{app_config.algorithm.engine}' 引擎...")
            engine.run()
            
            console.print("✅ 实验完成！")
        except Exception as e:
            console.print(f"❌ 实验失败: {e}", style="bold red")

    # ... (实现 init, validate, list 命令) ...

    if __name__ == "__main__":
        app()
    ```
*   **验收标准:** 命令行工具功能完整，可以成功启动并运行一个与`code.py`行为完全一致的GP实验流程。
