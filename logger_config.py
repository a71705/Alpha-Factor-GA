import logging
import sys

# 定义日志格式
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def setup_logger(log_file_path: str = "project.log", level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回一个 logger 实例。

    该函数会配置一个 logger，使其能够将日志信息同时输出到控制台和指定的日志文件。
    可以指定日志文件的路径和最低日志级别。

    参数:
        log_file_path (str): 日志文件的保存路径。默认为 "project.log"。
        level (int): 日志记录的最低级别 (例如 logging.INFO, logging.DEBUG)。默认为 logging.INFO。

    返回:
        logging.Logger: 配置好的 logger 实例。
    """
    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # 创建一个 Formatter 对象，用于定义日志的输出格式
    formatter = logging.Formatter(LOG_FORMAT)

    # 配置控制台日志输出
    # 创建一个 StreamHandler，用于将日志输出到标准错误流 (sys.stderr)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)  # 设置该 handler 的输出格式
    logger.addHandler(console_handler)      # 将该 handler 添加到 logger

    # 配置文件日志输出
    # 创建一个 FileHandler，用于将日志写入到指定的文件
    # mode='a' 表示追加模式，如果日志文件已存在，则在文件末尾追加新的日志记录
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)    # 设置该 handler 的输出格式
    logger.addHandler(file_handler)        # 将该 handler 添加到 logger

    return logger

# 示例：在模块被导入时，可以进行一次默认的配置
# default_logger = setup_logger()
# default_logger.info("日志系统已配置。")

# 或者，让使用者在需要时显式调用 setup_logger
# 例如，在 main.py 中:
# import logger_config
# logger = logger_config.setup_logger(log_file_path="my_app.log", level=logging.DEBUG)
# logger.info("应用启动")

# TODO: (Future Extension) Consider making logger configuration more flexible,
#       e.g., by loading settings from an external file (JSON, YAML, INI)
#       or environment variables. This would allow changing log levels,
#       handlers, and formats without modifying the code.
