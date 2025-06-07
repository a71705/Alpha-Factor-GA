# AlphaFactory - 自动化Alpha因子发现框架

## 简介

AlphaFactory 是一个旨在自动化发现量化交易中 Alpha 因子的 Python 框架。它利用遗传编程 (Genetic Programming, GP) 技术，通过多阶段的进化算法来生成、评估和优化潜在的 Alpha 表达式。

该框架与 WorldQuant BRAIN® 平台紧密集成，允许用户利用 BRAIN 平台的强大算力和丰富的市场数据进行 Alpha 因子的回测和模拟。AlphaFactory 的目标是为量化研究员提供一个高效、灵活且可定制的工具，以加速 Alpha 的探索和创新过程。

## 特性

*   **自动化Alpha发现**: 利用遗传编程 (GP) 技术自动搜索和优化 Alpha 因子表达式。
*   **分阶段进化**: 采用独特的分阶段（Staged GP）策略，逐步增加 Alpha 表达式的复杂度（例如，从深度1到深度3），以提高搜索效率和因子质量。
*   **高度可配置**: 通过 YAML 配置文件，用户可以灵活地定义实验的各个方面，包括遗传编程参数、BRAIN 模拟设置、操作符集等。
*   **BRAIN平台集成**: 无缝对接 WorldQuant BRAIN® API，实现 Alpha 表达式的提交、模拟和性能评估。
*   **命令行界面 (CLI)**: 提供简单易用的命令行工具，方便用户进行实验的初始化、运行、配置文件验证和列表查看。
*   **模块化架构**: 清晰的模块划分（如 `genetic_programming`, `brain_client`, `fitness` 等）使得代码更易于理解、维护和扩展。
*   **可定制组件**: 允许用户替换或自定义核心组件，如适应度函数、遗传操作符等，以满足特定的研究需求。
*   **支持多种操作符**: 内置丰富的操作符，包括算术运算符、时间序列函数、横截面操作等，为构建多样化的 Alpha 提供了基础。

## 项目结构

`alpha_factory/`
├── `main.py`: 项目的命令行界面 (CLI) 入口。
├── `brain_client/`: 包含与 WorldQuant BRAIN® API 交互的客户端代码，如会话管理和 API 请求封装。
├── `cli/`: 实现了命令行交互逻辑，如菜单选择、参数处理等。
├── `configs/`: 存放实验的 YAML 配置文件和配置模板。用户可以在此定义和管理不同的实验设置。
├── `fitness/`: 包含适应度评估相关的代码。适应度函数用于评价 Alpha 因子的优劣。
├── `genetic_programming/`: 遗传编程的核心模块。
│   ├── `engine.py`: 实现分阶段遗传编程引擎 (StagedGPEngine)。
│   ├── `generators/`: 负责生成 Alpha 表达式树的初始种群和新个体。
│   ├── `operators/`: 定义遗传操作，如交叉 (crossover) 和变异 (mutation)。
│   └── `models.py`: 定义 Alpha 表达式树的节点等数据模型。
├── `utils/`: 包含项目通用的工具函数和辅助类，如配置加载器、表达式转换器等。
└── `README.md`: (本文档) 项目的介绍和使用指南。

## 安装指南

### 环境要求

*   Python 3.8 或更高版本 (推荐使用最新的稳定版 Python 3.x)。
*   能够访问 WorldQuant BRAIN® 平台 API (通常需要相应的凭证和网络配置)。

### 依赖安装

AlphaFactory 依赖于一些第三方 Python 库。您可以使用 pip 来安装它们。

主要的依赖包括：
*   `typer`: 用于构建强大的命令行界面。
*   `rich`: 用于在控制台输出美观的文本和表格。
*   `questionary`: 用于创建交互式的命令行提示。
*   `PyYAML`: 用于加载和解析 YAML 配置文件。
*   `requests`: (通常由 BRAIN API 的客户端库间接依赖) 用于进行 HTTP 请求。

建议创建一个虚拟环境来管理项目依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate  # Windows
```

然后，您可以通过以下命令安装主要依赖：

```bash
pip install typer rich questionary PyYAML requests
```

**注意**:
*   根据您的 WorldQuant BRAIN® API 客户端库的具体要求，可能需要安装额外的包 (例如 `worldquant-brain-api-client` 或类似的包)。请查阅 BRAIN 平台的相关文档以获取准确的客户端安装说明。
*   如果项目中后续添加了 `requirements.txt` 文件，您可以直接使用 `pip install -r requirements.txt` 来安装所有依赖。

## 使用方法

AlphaFactory 通过命令行界面 (CLI) 进行操作。在项目根目录下，您可以使用以下命令 (假设您的 Python 环境已正确设置，并且您位于 `alpha_factory` 所在的父目录下，或者 `alpha_factory` 包已安装到您的 Python 环境中)：

```bash
python -m alpha_factory <命令> [参数]
```

或者，如果 AlphaFactory 被安装为一个可执行脚本（例如通过 `pip install .` 且在 `pyproject.toml` 中定义了脚本入口点），您可以直接调用：

```bash
alpha-factory <命令> [参数]
```
(以下示例将使用 `python -m alpha_factory` 格式)

### 1. 初始化实验配置文件 (`init`)

如果您想创建一个新的实验配置文件，可以使用 `init` 命令。这将引导您选择一个模板，并为新配置文件命名。

```bash
python -m alpha_factory init
```
程序会提示您输入新配置文件的名称，并选择一个基础模板（例如 `legacy_default_template.yaml` 或 `generic_default_template.yaml`）。新创建的配置文件将保存在 `alpha_factory/configs/` 目录下。

### 2. 列出可用的配置文件 (`list`)

要查看 `alpha_factory/configs/` 目录中所有可用的实验配置文件，请使用 `list` 命令。

```bash
python -m alpha_factory list
```
这将以表格形式显示配置文件的名称、大小和最后修改时间。

### 3. 验证配置文件 (`validate`)

在运行实验之前，您可以验证配置文件的格式是否正确。

```bash
python -m alpha_factory validate alpha_factory/configs/your_experiment_config.yaml
```
将 `your_experiment_config.yaml` 替换为您要验证的实际文件名。如果配置文件有效，将显示成功消息；否则，将指出错误所在。

### 4. 运行Alpha发现实验 (`run`)

这是 AlphaFactory 的核心命令，用于启动遗传编程引擎来发现 Alpha 因子。

有两种方式运行实验：

*   **通过命令行参数指定配置文件：**

    ```bash
    python -m alpha_factory run --config alpha_factory/configs/your_experiment_config.yaml
    ```
    或者使用短格式：
    ```bash
    python -m alpha_factory run -c alpha_factory/configs/your_experiment_config.yaml
    ```
    将 `your_experiment_config.yaml` 替换为您的配置文件名。

*   **交互式选择配置文件：**

    如果您在运行 `run` 命令时不提供配置文件路径，系统将进入交互模式，列出所有可用的配置文件供您选择。
    ```bash
    python -m alpha_factory run
    ```

实验运行时，控制台会输出详细的日志信息，包括当前进行的阶段、种群评估结果、以及最终发现的最佳 Alpha 表达式等。

### WorldQuant BRAIN® 凭证

请确保您的环境中已正确配置了访问 WorldQuant BRAIN® API 所需的凭证。这通常涉及到设置环境变量或配置文件，具体取决于您使用的 BRAIN API 客户端库的要求。AlphaFactory 本身不直接处理凭证的存储，而是依赖于 BRAIN API 客户端库的验证机制。

## 配置说明

AlphaFactory 的实验行为通过位于 `alpha_factory/configs/` 目录下的 YAML 文件进行配置。建议从提供的模板文件（如 `generic_default_template.yaml` 或 `legacy_default_template.yaml`）开始，复制并修改以创建您自己的实验配置。

以下是配置文件中主要部分及其关键参数的说明：

### `experiment_name`
一个字符串，用于标识您的实验。例如：`my_first_alpha_hunt`。

```yaml
experiment_name: generic_experiment
```

### `brain`
此部分包含与 WorldQuant BRAIN® 平台模拟相关的参数。

*   `region`: (字符串) 模拟的地理区域，例如 `"USA"`, `"CHN"`, `"EUR"`。
*   `universe`: (字符串) 模拟所用的股票池，例如 `"TOP3000"`, `"TOP500"`, `"RUSSELL3000"`。
*   `delay`: (整数) 因子生成到模拟执行之间的延迟天数，通常为 `1`。
*   `truncation`: (浮点数) 极端值的截断百分比，例如 `0.08` 表示对最高和最低的 8% 数据进行截断。
*   `neutralization`: (字符串) 中性化方法，例如 `"INDUSTRY"` (行业中性化), `"MARKET"` (市场中性化), 或 `"NONE"`。
*   `decay`: (整数) Alpha 信号的衰减期，例如 `20`。
*   `nan_handling`: (字符串) 如何处理缺失值 (NaN)，例如 `"OFF"` 或 `"ZERO"`。
*   `unit_handling`: (字符串) 单位处理方式，例如 `"VERIFY"` 或 `"CONVERT"`。
*   `pasteurization`: (字符串) 巴氏消毒方法，例如 `"ON"` 或 `"OFF"`。
*   `concurrent_simulations`: (整数) 允许同时向BRAIN API提交的最大模拟请求数，例如 `3`。

```yaml
brain:
  region: "USA"
  universe: "TOP3000"
  delay: 1
  truncation: 0.08
  neutralization: "INDUSTRY"
  # decay, nan_handling, unit_handling, pasteurization 将使用 Pydantic 模型中的默认值
  # concurrent_simulations: 3 # 如果未指定，会使用默认值
```

### `gp` (Genetic Programming)
遗传编程算法的核心参数，主要用于控制不同阶段的进化过程。

*   `d1_population`: (整数) 深度为 1 阶段的种群大小（每一代保留的个体数）。
*   `d1_generations`: (整数) 深度为 1 阶段的进化代数。
*   `d2_population`: (整数) 深度为 2 阶段的种群大小。
*   `d2_generations`: (整数) 深度为 2 阶段的进化代数。
*   `d3_population`: (整数) 深度为 3 阶段的种群大小。
*   `d3_generations`: (整数) 深度为 3 阶段的进化代数。

```yaml
gp:
  d1_population: 30
  d1_generations: 5
  d2_population: 25
  d2_generations: 10
  d3_population: 20
  d3_generations: 15
```

### `fitness`
配置用于评估 Alpha 因子表现的适应度函数。

*   `module`: (字符串) 实现适应度计算器基类 (`BaseFitnessCalculator`) 的 Python 模块路径。
*   `class_name`: (字符串) 适应度计算器类的名称。
*   `params`: (字典) 传递给适应度计算器构造函数的参数。

```yaml
fitness:
  module: "alpha_factory.fitness.legacy_fitness" # 默认使用旧版适应度函数
  class_name: "LegacyFitness" # 对应 Pydantic 模型中的 alias='class'
  params: {} # LegacyFitness 构造函数不接受额外参数
```

### `operators`
定义遗传编程过程中可用于构建 Alpha 表达式树的各类操作符。

*   `terminal_values`: (列表) 终结符节点，通常是基础数据字段，如 `["close", "open", "high", "low", "vwap", "adv20", "volume"]`。
*   `ts_ops`: (列表) 时间序列操作符，如 `["ts_mean", "ts_rank", "ts_delta"]`。
*   `binary_ops`: (列表) 二元操作符，如 `["add", "subtract", "multiply", "divide"]`。
*   `ts_ops_values`: (列表) 时间序列操作符的参数值（通常是回看窗口期），如 `["20", "40", "60"]`。
*   `unary_ops`: (列表) 一元操作符，如 `["rank", "zscore", "log", "sigmoid"]`。

```yaml
operators:
  terminal_values: ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns"]
  ts_ops: ["ts_zscore", "ts_rank", "ts_delta", "ts_mean"]
  binary_ops: ["add", "subtract", "divide", "multiply"]
  ts_ops_values: ["20", "40", "60", "120", "240"]
  unary_ops: ["rank", "zscore", "log"]
```

### `algorithm`
用于选择和配置算法的不同组件。这允许更高级的定制，例如替换默认的遗传编程引擎或特定的遗传算子。

*   `engine`: (字符串) 选择使用的遗传编程引擎，例如 `"staged"` (分阶段引擎)。
*   `crossover`: (字符串) 选择交叉算子的实现，例如 `"legacy"`。
*   `mutation`: (字符串) 选择变异算子的实现，例如 `"legacy"`。
*   `generator`: (字符串) 选择个体生成器的实现，例如 `"legacy"`。
*   `converter`: (字符串) 选择表达式树与字符串之间转换器的实现，例如 `"legacy"`。

```yaml
algorithm:
  engine: "staged"
  crossover: "legacy"
  mutation: "legacy"
  generator: "legacy"
  converter: "legacy"
```

请仔细阅读配置文件中的注释，并根据您的研究需求调整这些参数。不正确的配置可能会导致实验无法运行或结果不符合预期。建议在修改配置后使用 `validate` 命令进行检查。

## 如何贡献

我们欢迎对 AlphaFactory 项目的各种贡献，包括但不限于功能增强、Bug修复、文档改进等。如果您有兴趣为本项目贡献代码或提出建议，请遵循以下基本流程：

1.  **Fork 本仓库**: 点击仓库主页右上角的 "Fork" 按钮，将项目复制到您自己的 GitHub 账户下。
2.  **创建新分支**: 从 `main` (或主开发分支) 创建一个新的分支来进行您的修改。分支名称应清晰描述您所做的工作，例如 `feat/add-new-operator` 或 `fix/config-validation-bug`。
    ```bash
    git checkout -b your-feature-branch
    ```
3.  **进行修改**: 在您的分支上进行代码编写、修改或文档更新。
    *   请确保您的代码风格与项目现有代码保持一致。
    *   为新增的关键功能编写单元测试。
    *   确保所有测试通过。
4.  **提交更改**: 清晰地提交您的更改，并附带描述性的提交信息。
    ```bash
    git add .
    git commit -m "详细说明您所做的更改"
    ```
5.  **推送至您的 Fork**: 将您的分支推送到您在 GitHub 上的 Fork 仓库。
    ```bash
    git push origin your-feature-branch
    ```
6.  **创建 Pull Request (PR)**: 回到原始的 AlphaFactory 仓库页面，选择 "Pull requests" 标签页，然后点击 "New pull request"。选择您的分支，并提交 PR。
    *   在 PR 描述中详细说明您的更改内容、目的以及任何相关的 Issue 编号。
    *   PR 将会经过审查，我们可能会提出修改建议。

如果您不确定如何开始，或者想讨论某个潜在的更改，请随时创建一个 Issue。

感谢您对 AlphaFactory 的关注和支持！

## 许可证
