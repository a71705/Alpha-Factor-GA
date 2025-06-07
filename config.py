import os

# ------------------ API 相关配置 ------------------

# WorldQuant BRAIN API 的基础 URL
API_BASE_URL = "https://api.worldquantbrain.com/"

# WorldQuant BRAIN 平台 Alpha 页面的基础 URL
PLATFORM_ALPHA_URL = "https://platform.worldquantbrain.com/alpha/"

# ------------------ 文件路径配置 ------------------

# 模拟结果保存路径
SIMULATION_RESULTS_PATH = "simulation_results/"

# PnL (盈亏) 数据保存路径
PNL_DATA_PATH = "alphas_pnl/"

# 年度统计数据保存路径
YEARLY_STATS_PATH = "yearly_stats/"

# 用户凭据文件夹路径 (存储 platform-brain.json)
# 使用 os.path.expanduser("~") 来获取用户的主目录，确保跨平台兼容性
CREDENTIALS_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "secrets")

# 用户凭据文件完整路径
CREDENTIALS_FILE_PATH = os.path.join(CREDENTIALS_FOLDER_PATH, "platform-brain.json")


# ------------------ 模拟默认配置 ------------------

# 默认的 Alpha 模拟配置字典
# 这些配置项控制在获取 Alpha 详细信息时，是否执行某些操作，例如：
# - get_pnl: 是否获取 PnL (盈亏) 数据。
# - get_stats: 是否获取年度统计数据。
# - save_pnl_file: 是否将 PnL 数据保存到 CSV 文件。
# - save_stats_file: 是否将年度统计数据保存到 CSV 文件。
# - save_result_file: 是否将完整的模拟结果 (JSON 格式) 保存到文件。
# - check_submission: 是否执行 Alpha 的提交检查 (例如，检查是否符合提交标准)。
# - check_self_corr: 是否执行 Alpha 的自相关性检查。
# - check_prod_corr: 是否执行 Alpha 与生产环境 Alpha 的相关性检查。
DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

# ------------------ 遗传算法相关配置 ------------------

# 定义 Alpha 表达式的构建块

# 终端值：通常是数据字段名或数值常量
TERMINAL_VALUES: list[str] = ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns", "dividend"]

# 时间序列操作符：对单个时间序列数据进行操作的函数
TS_OPS: list[str] = ["ts_zscore", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_delta", "ts_ir", "ts_mean","ts_median", "ts_product", "ts_std_dev"]

# 二元操作符：需要两个操作数的操作符
BINARY_OPS: list[str] = ["add", "subtract", "divide", "multiply", "max", "min"]

# 时间序列操作符的参数值：通常是窗口大小（lookback period）
TS_OPS_VALUES: list[str] = ["20", "40", "60", "120", "240"] # 注意这些是字符串形式的数字

# 一元操作符：只需要一个操作数的操作符
UNARY_OPS: list[str] = ["rank", "zscore", "winsorize", "normalize", "rank_by_side", "sigmoid", "pasteurize", "log"]

# ------------------ Alpha 模拟参数默认值 ------------------
# 这些是 generate_alpha 函数中使用的默认值，集中管理便于修改

# 默认的 Alpha 类型
DEFAULT_ALPHA_TYPE: str = "REGULAR"
# 默认的 Alpha 语言
DEFAULT_ALPHA_LANGUAGE: str = "FASTEXPR"
# 默认的资产类型
DEFAULT_INSTRUMENT_TYPE: str = "EQUITY"

# 默认的区域设置 (示例，generate_alpha 函数参数中也有默认值)
DEFAULT_REGION: str = "USA"
# 默认的股票池 (示例)
DEFAULT_UNIVERSE: str = "TOP3000"
# 默认的延迟天数
DEFAULT_DELAY: int = 1
# 默认的截断值
DEFAULT_TRUNCATION: float = 0.08
# 默认的中性化设置
DEFAULT_NEUTRALIZATION: str = "INDUSTRY"
# 默认的 NaN 处理方式
DEFAULT_NAN_HANDLING: str = "OFF" # 其他选项如 "FILL"
# 默认的单位处理方式
DEFAULT_UNIT_HANDLING: str = "VERIFY" # 其他选项如 "IGNORE", "RESCALE"
# 默认的巴氏消毒设置
DEFAULT_PASTEURIZATION: str = "ON" # 其他选项如 "OFF"
# 默认的衰减值
DEFAULT_DECAY: int = 0

# 模拟进度轮询间隔 (秒)
SIMULATION_POLL_INTERVAL: int = 10 # 正常轮询间隔
SIMULATION_POLL_INTERVAL_ON_ERROR: int = 30 # 发生请求错误时的轮询间隔

# ------------------ 遗传算法执行参数 ------------------
# 这些参数控制遗传算法不同阶段的种群大小和迭代次数

# 深度 1 Alpha 优化参数
GA_D1_POPULATION_SIZE: int = 30  # n: 每代保留的最佳 Alpha 数量
GA_D1_ITERATIONS: int = 5      # m: 遗传算法的迭代次数

# 深度 2 Alpha 优化参数
GA_D2_POPULATION_SIZE: int = 25
GA_D2_ITERATIONS: int = 10

# 深度 3 Alpha 优化参数
GA_D3_POPULATION_SIZE: int = 20
GA_D3_ITERATIONS: int = 15

# 模拟运行模式 (True: 使用MOCK数据/短流程, False: 实际API调用和完整流程)
# TODO: 在 main.py 和其他模块中实现对此标志的响应逻辑
DRY_RUN_MODE: bool = True # 默认为 True 以进行安全测试

# Persona (生物识别) 认证相关配置
PERSONA_MAX_ATTEMPTS: int = 3  # 用户完成 Persona 认证的最大检查次数
PERSONA_POLL_INTERVAL: int = 5 # 每次检查 Persona 状态之间的等待秒数

# API 请求重试配置 (用于5xx服务器错误)
API_MAX_RETRIES: int = 3
API_RETRY_INTERVAL: int = 5 # 秒

# 遗传算法微调参数
GA_MUTATION_PROBABILITY_PER_NODE: float = 0.1 # genetic_algorithm.mutate_random_node 中节点变异概率
# TODO: (Future Extension) Consider loading Alpha generation rules (TERMINAL_VALUES, *_OPS, etc.)
#       from an external JSON/YAML file for easier modification without code changes.

# 测试阈值配置
DEFAULT_PROD_CORR_THRESHOLD: float = 0.7 # simulation_manager.check_prod_corr_test 默认阈值
DEFAULT_SELF_CORR_THRESHOLD: float = 0.7 # simulation_manager.check_self_corr_test 默认阈值

# 并发执行配置
DEFAULT_CONCURRENT_SIMULATIONS: int = 3 # simulation_manager.simulate_alpha_list 中的默认并发数
DEFAULT_CONCURRENT_BATCHES: int = 3   # simulation_manager.simulate_alpha_list_multi 中的默认并发批次数
DEFAULT_ALPHAS_PER_BATCH: int = 3     # simulation_manager.simulate_alpha_list_multi 中每批次的Alpha数量 (原 limit_of_multi_simulations)

# ------------------ 其他全局配置 ------------------
# 可以根据需要在此处添加更多的全局配置项
# 例如：
# - 默认的区域、股票池等
# DEFAULT_REGION = "USA"
# DEFAULT_UNIVERSE = "TOP3000"
# - 并发模拟数量
# MAX_CONCURRENT_SIMULATIONS = 3
# - API 请求重试次数
# API_RETRY_COUNT = 3

# 提示:
# - 保持配置项名称清晰、易懂。
# - 对于敏感信息（如 API 密钥），优先考虑使用环境变量或专门的密钥管理服务，
#   而不是直接硬编码到配置文件中。此处的 CREDENTIALS_FILE_PATH 指向的是包含凭据的文件，
#   凭据本身不在此文件中。
"""
此模块包含项目的所有配置变量。
例如 API URL、文件路径、默认参数等。
通过将配置集中在此文件中，可以方便地管理和修改项目的行为，
而无需在代码的多个位置进行更改。
"""
pass
