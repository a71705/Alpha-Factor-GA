import logging # Standard logging library
import sys # For basic system operations if needed

# Custom module imports
import api_client
import genetic_algorithm
import logger_config # For setting up unified logging
import config # To get GA parameters and other configs
# import utils # If any utility functions are needed directly in main

def run_alpha_optimization():
    """
    主函数，编排整个 Alpha 优化流程。
    1. 初始化日志系统。
    2. 启动并验证 BRAIN API 会话。
    3. 按顺序执行遗传算法的各个深度阶段。
    4. 输出最终结果。
    """
    # 1. 初始化日志系统
    # 调用 logger_config.py 中的函数来设置日志记录器
    # 假设 setup_logger() 返回配置好的根 logger 或特定 logger
    logger = logger_config.setup_logger(log_file_path="alpha_optimization_main.log", level=logging.INFO if not config.DRY_RUN_MODE else logging.DEBUG)
    logger.info("Alpha 优化程序启动。")
    if config.DRY_RUN_MODE:
        logger.warning("注意: 程序正在 DRY_RUN_MODE 下运行。API调用将被模拟，迭代次数可能减少。")

    # 2. 启动 BRAIN API 会话
    logger.info("正在启动 BRAIN API 会话...")
    brain_session = api_client.start_session()

    if not brain_session:
        logger.error("无法启动 BRAIN API 会话。程序将退出。")
        sys.exit(1) # 退出程序，因为后续操作依赖会话

    logger.info("BRAIN API 会话已成功启动。")

    # 检查会话超时 (可选，但良好实践)
    try:
        timeout_seconds = api_client.check_session_timeout(brain_session)
        logger.info(f"当前会话剩余有效时间: {timeout_seconds // 60} 分钟 ({timeout_seconds} 秒)。")
        if timeout_seconds < 300: # 少于5分钟
            logger.warning("会话剩余时间较短，如果优化流程过长，可能会遇到会话过期问题。")
    except Exception as e:
        logger.warning(f"检查会话超时时发生错误: {e}")


    # 3. 执行遗传算法流程
    # 根据 DRY_RUN_MODE 调整迭代次数 (示例)
    d1_iterations = 1 if config.DRY_RUN_MODE else config.GA_D1_ITERATIONS
    d2_iterations = 1 if config.DRY_RUN_MODE else config.GA_D2_ITERATIONS
    d3_iterations = 1 if config.DRY_RUN_MODE else config.GA_D3_ITERATIONS

    # d1_population_size = 5 if config.DRY_RUN_MODE else config.GA_D1_POPULATION_SIZE # 示例调整

    try:
        logger.info(f"开始执行遗传算法 - 优化深度 1 Alpha (N={config.GA_D1_POPULATION_SIZE}, M={d1_iterations})...")
        best_alphas_d1 = genetic_algorithm.best_d1_alphas(
            s=brain_session,
            n_alphas_to_select=config.GA_D1_POPULATION_SIZE,
            n_iterations=d1_iterations,
            simulation_parameters=config.DEFAULT_CONFIG
            # fitness_function=my_custom_fitness, # 示例: 用户可自定义策略
            # mutation_function=my_custom_mutation, # 示例: 用户可自定义策略
        )
        logger.info(f"深度 1 最佳 Alpha ({len(best_alphas_d1)} 个):")
        for i, alpha_expr in enumerate(best_alphas_d1):
            logger.info(f"  D1 Alpha #{i+1}: {alpha_expr}")
        print("\n--- 深度 1 的最佳 Alpha 表达式 ---")
        for alpha_expr in best_alphas_d1:
            print(alpha_expr)

        logger.info(f"开始执行遗传算法 - 优化深度 2 Alpha (N={config.GA_D2_POPULATION_SIZE}, M={d2_iterations})...")
        best_alphas_d2 = genetic_algorithm.best_d2_alphas(
            s=brain_session,
            best_d1_expressions=best_alphas_d1,
            n_alphas_to_select=config.GA_D2_POPULATION_SIZE,
            n_iterations=d2_iterations,
            simulation_parameters=config.DEFAULT_CONFIG
            # crossover_function=my_custom_crossover, # 示例
        )
        logger.info(f"深度 2 最佳 Alpha ({len(best_alphas_d2)} 个):")
        for i, alpha_expr in enumerate(best_alphas_d2):
            logger.info(f"  D2 Alpha #{i+1}: {alpha_expr}")
        print("\n--- 深度 2 的最佳 Alpha 表达式 ---")
        for alpha_expr in best_alphas_d2:
            print(alpha_expr)

        logger.info(f"开始执行遗传算法 - 优化深度 3 Alpha (N={config.GA_D3_POPULATION_SIZE}, M={d3_iterations})...")
        best_alphas_d3 = genetic_algorithm.best_d3_alpha(
            s=brain_session,
            best_d2_expressions=best_alphas_d2,
            n_alphas_to_select=config.GA_D3_POPULATION_SIZE,
            n_iterations=d3_iterations,
            simulation_parameters=config.DEFAULT_CONFIG
        )
        logger.info(f"深度 3 最佳 Alpha ({len(best_alphas_d3)} 个):")
        for i, alpha_expr in enumerate(best_alphas_d3):
            logger.info(f"  D3 Alpha #{i+1}: {alpha_expr}")
        print("\n--- 深度 3 的最佳 Alpha 表达式 ---")
        for alpha_expr in best_alphas_d3:
            print(alpha_expr)

    except Exception as e:
        logger.critical(f"遗传算法执行过程中发生严重错误: {e}", exc_info=True)
        # exc_info=True 会记录堆栈跟踪信息
    finally:
        logger.info("Alpha 优化程序执行完毕。")


if __name__ == "__main__":
    try:
        run_alpha_optimization()
    except api_client.BrainPlatformError as bpe: # Catching specific API/platform related errors
        # Assuming logger is already configured by run_alpha_optimization or globally
        logging.getLogger().critical(f"程序因未处理的 BrainPlatformError 而终止: {bpe}", exc_info=True)
        sys.exit(2) # Exit with a specific error code
    except Exception as e:
        logging.getLogger().critical(f"程序因未捕获的意外错误而终止: {e}", exc_info=True)
        sys.exit(3) # General error code
