import random
import pandas as pd
from typing import List, Optional, Tuple, Callable, Dict, Any
from collections import OrderedDict

# 自定义模块导入
from alpha_models import Node, d1_alpha_to_tree, d2_alpha_to_tree, d3_alpha_to_tree
from alpha_models import depth_one_trees, depth_two_tree, depth_three_tree
from alpha_models import d1tree_to_alpha, d2tree_to_alpha, d3tree_to_alpha
from alpha_models import copy_tree
from config import TERMINAL_VALUES, BINARY_OPS, TS_OPS, TS_OPS_VALUES, UNARY_OPS

import logging

# 模块导入
from simulation_manager import simulate_alpha_list, generate_alpha_simulation_data
from utils import prettify_result
import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --- Default Strategy Function Placeholders ---
# These will be assigned to the actual functions defined later in this file.
# This allows them to be used as default arguments in function signatures before their full definition.
DEFAULT_FITNESS_FUNCTION: Callable
DEFAULT_MUTATION_STRATEGY: Callable
DEFAULT_CROSSOVER_STRATEGY: Callable

"""
此模块包含遗传算法 (Genetic Programming, GP) 的核心实现。
GP 将用于进化和优化 Alpha 表达式（表示为树结构）。
核心流程函数 (best_d*_alphas) 现在支持通过参数传入自定义的
适应度评估 (fitness_function), 变异 (mutation_function),
和交叉 (crossover_function)策略。
"""

def fitness_fun(simulation_results_df: pd.DataFrame, top_n: int) -> List[str]:
    # ... (implementation as before) ...
    logger.debug(f"开始计算适应度，输入DataFrame行数: {len(simulation_results_df if isinstance(simulation_results_df, pd.DataFrame) else 0)}, 选择Top N: {top_n}")

    if not isinstance(simulation_results_df, pd.DataFrame):
        logger.error("fitness_fun错误: 输入数据不是有效的Pandas DataFrame。")
        raise TypeError("输入数据 simulation_results_df 必须是 Pandas DataFrame。")

    if simulation_results_df.empty:
        logger.warning("fitness_fun: 输入的DataFrame为空，无法计算适应度。返回空列表。")
        return []

    required_cols = ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover', 'expression']
    missing_cols = [col for col in required_cols if col not in simulation_results_df.columns]
    if missing_cols:
        logger.error(f"适应度计算错误: 输入的DataFrame中缺失必需列: {', '.join(missing_cols)}。")
        return []

    data_copy = simulation_results_df.copy()
    numeric_cols_for_fitness = ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover']
    for col in numeric_cols_for_fitness:
        if col in data_copy.columns:
            data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
        else:
            logger.error(f"fitness_fun: 在数值转换前发现列 '{col}' 缺失，这不应发生。")
            return []

    turnover_zeros_before = (data_copy['turnover'] == 0).sum()
    turnover_nans_before = data_copy['turnover'].isna().sum()
    data_copy['turnover'] = data_copy['turnover'].fillna(0.0001).replace(0, 0.0001)
    if turnover_zeros_before > 0: logger.info(f"fitness_fun: {turnover_zeros_before} 个 turnover 为 0 的记录被替换为 0.0001。")
    if turnover_nans_before > 0: logger.info(f"fitness_fun: {turnover_nans_before} 个 turnover 为 NaN 的记录被替换为 0.0001。")

    drawdown_zeros_before = (data_copy['drawdown'] == 0).sum()
    drawdown_nans_before = data_copy['drawdown'].isna().sum()
    data_copy['drawdown'] = data_copy['drawdown'].abs()
    data_copy['drawdown'] = data_copy['drawdown'].fillna(0.0001).replace(0, 0.0001)
    if drawdown_zeros_before > 0: logger.info(f"fitness_fun: {drawdown_zeros_before} 个 drawdown 为 0 (或处理后为0) 的记录被替换为 0.0001。")
    if drawdown_nans_before > 0: logger.info(f"fitness_fun: {drawdown_nans_before} 个 drawdown 为 NaN 的记录被替换为 0.0001。")

    cols_to_fillna_zero = ['sharpe', 'fitness', 'returns']
    for col in cols_to_fillna_zero:
        nan_count = data_copy[col].isna().sum()
        if nan_count > 0:
            logger.info(f"fitness_fun: 列 '{col}' 中有 {nan_count} 个 NaN 值被填充为 0。")
            data_copy[col] = data_copy[col].fillna(0)

    data_copy.dropna(subset=numeric_cols_for_fitness, inplace=True)
    if data_copy.empty:
        logger.warning("适应度计算：数据清洗和预处理后 DataFrame 为空。返回空列表。")
        return []

    numerator = data_copy['sharpe'] * data_copy['fitness'] * data_copy['returns']
    denominator = (data_copy['drawdown'] * (data_copy['turnover'] ** 2)) + 0.001
    data_copy['fitness_score'] = numerator / denominator
    data_copy.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    data_copy.dropna(subset=['fitness_score'], inplace=True)

    if data_copy.empty:
        logger.warning("适应度计算：计算并过滤score后 DataFrame 为空。返回空列表。") # Changed from print to logger.warning
        return []

    data_sorted = data_copy.sort_values(by='fitness_score', ascending=False)
    top_n_expressions = data_sorted.head(top_n)['expression'].tolist()
    return top_n_expressions

def mutate_random_node(
    original_tree_root: Node,
    terminal_values: List[str] = TERMINAL_VALUES, # Default to config lists
    unary_ops: List[str] = UNARY_OPS,
    binary_ops: List[str] = BINARY_OPS,
    ts_ops: List[str] = TS_OPS,
    ts_ops_values: List[str] = TS_OPS_VALUES,
    mutation_probability_per_node: float = config.GA_MUTATION_PROBABILITY_PER_NODE
) -> Optional[Node]: # Return Optional[Node] as it can return None
    # ... (implementation as before, ensure logger is used over print) ...
    if not original_tree_root:
        logger.warning("变异函数接收到空树，返回None。")
        return None

    mutated_tree_root = copy_tree(original_tree_root)
    # all_known_values = set(terminal_values + unary_ops + binary_ops + ts_ops + ts_ops_values) # Not strictly needed with current logic

    def mutate_node_recursive(current_node: Optional[Node]):
        if current_node is None: return
        mutate_node_recursive(current_node.left)
        mutate_node_recursive(current_node.right)
        if random.random() < mutation_probability_per_node:
            original_value = current_node.value
            # logger.debug(f"尝试变异节点: {original_value}")
            if original_value in binary_ops and binary_ops: current_node.value = random.choice(binary_ops)
            elif original_value in ts_ops and ts_ops: current_node.value = random.choice(ts_ops)
            elif original_value in ts_ops_values and ts_ops_values: current_node.value = random.choice(ts_ops_values)
            elif original_value in unary_ops and unary_ops: current_node.value = random.choice(unary_ops)
            elif original_value in terminal_values and terminal_values: current_node.value = random.choice(terminal_values)
            # if current_node.value != original_value: logger.info(f"节点 '{original_value}' 变异为 '{current_node.value}'")

    mutate_node_recursive(mutated_tree_root)
    return mutated_tree_root


def crossover(
    parent1_root: Node,
    parent2_root: Node
    # n_depth_indicator: int (Removed as per simplified crossover for pluggability)
) -> Tuple[Optional[Node], Optional[Node]]:
    # ... (implementation as before, ensure logger is used over print) ...
    if not parent1_root or not parent2_root:
        logger.warning("交叉操作接收到空父树，返回None, None。")
        return None, None
    child1_root = copy_tree(parent1_root)
    child2_root = copy_tree(parent2_root)
    if not child1_root or not child2_root: return child1_root, child2_root
    if random.choice([True, False]):
        if child1_root.left and child2_root.left:
            # logger.debug("交叉: 交换左子节点。")
            temp_child1_left = child1_root.left
            child1_root.left = child2_root.left
            child2_root.left = temp_child1_left
    else:
        if child1_root.right and child2_root.right:
            # logger.debug("交叉: 交换右子节点。")
            temp_child1_right = child1_root.right
            child1_root.right = child2_root.right
            child2_root.right = temp_child1_right
    return child1_root, child2_root

# Assign actual functions to the default strategy constants
DEFAULT_FITNESS_FUNCTION = fitness_fun
DEFAULT_MUTATION_STRATEGY = mutate_random_node
DEFAULT_CROSSOVER_STRATEGY = crossover

# --- Core Genetic Algorithm Workflow Functions ---
def best_d1_alphas(
    s: Any,
    n_alphas_to_select: int,
    n_iterations: int,
    simulation_parameters: Dict[str, Any] = config.DEFAULT_CONFIG,
    fitness_function: Callable = DEFAULT_FITNESS_FUNCTION,
    mutation_function: Callable = DEFAULT_MUTATION_STRATEGY
) -> List[str]:
    logger.info(f"开始深度 1 Alpha 优化流程: 选择Top {n_alphas_to_select} Alpha, 进行 {n_iterations} 轮迭代。")
    current_population_expressions: List[str] = []
    logger.info("步骤 1: 初始化深度 1 Alpha 种群...")
    initial_population_size = n_alphas_to_select * 2
    temp_expressions: List[str] = []
    for _ in range(initial_population_size):
        d1_tree = depth_one_trees()
        d1_alpha_expr = d1tree_to_alpha(d1_tree)
        temp_expressions.append(d1_alpha_expr)
    current_population_expressions = list(OrderedDict.fromkeys(temp_expressions))
    logger.info(f"初始种群生成完毕，共 {len(current_population_expressions)} 个独立 Alpha 表达式。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d1_alphas - 模拟评估部分使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in current_population_expressions]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in current_population_expressions]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=1, iteration_context=0)
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("初始深度 1 Alpha 模拟未返回有效数据。流程终止。"); return []
    current_population_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"初始选择完成，选出 {len(current_population_expressions)} 个 Alpha 进入迭代。")

    for iteration_num in range(n_iterations):
        logger.info(f"开始第 {iteration_num + 1}/{n_iterations} 轮深度 1 Alpha 迭代...")
        new_random_expressions: List[str] = [d1tree_to_alpha(depth_one_trees()) for _ in range(n_alphas_to_select)]
        combined_expressions = list(OrderedDict.fromkeys(current_population_expressions + new_random_expressions))
        logger.debug(f"迭代 {iteration_num + 1}: 补充种群后共 {len(combined_expressions)} 个 Alpha。")

        if config.DRY_RUN_MODE:
            logger.warning(f"警告: best_d1_alphas iteration {iteration_num + 1} - 模拟评估部分使用模拟数据 (DRY_RUN_MODE)!")
            mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in combined_expressions]
            results_df = pd.DataFrame(mock_data)
        else:
            alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in combined_expressions]
            simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=1, iteration_context=iteration_num + 1)
            results_df = prettify_result(simulation_output_list)

        if results_df.empty: logger.warning(f"深度 1 Alpha 第 {iteration_num + 1} 轮迭代模拟未返回数据。保留当前种群。"); continue
        current_population_expressions = fitness_function(results_df, n_alphas_to_select)
        logger.info(f"迭代 {iteration_num + 1} 完成选择，当前种群数量: {len(current_population_expressions)}。")

    logger.info("深度 1 Alpha 迭代完成，开始最终变异阶段...")
    if not current_population_expressions: logger.warning("当前种群为空，无法变异。"); return []
    best_trees_d1 = d1_alpha_to_tree(current_population_expressions)
    mutated_expressions: List[str] = []
    for tree_node in best_trees_d1:
        if tree_node:
            mutated_tree = mutation_function(tree_node)
            if mutated_tree:
                try: mutated_expressions.append(d1tree_to_alpha(mutated_tree))
                except Exception as e: logger.error(f"转换变异后D1树到Alpha时出错: {e}", exc_info=True)
    final_expressions_to_evaluate = list(OrderedDict.fromkeys(current_population_expressions + mutated_expressions))
    logger.info(f"变异阶段完成，共 {len(final_expressions_to_evaluate)} 个 Alpha 待最终评估。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d1_alphas final eval - 使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in final_expressions_to_evaluate]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in final_expressions_to_evaluate]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=1, iteration_context=-1)
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("D1 Alpha最终评估未返回数据。返回变异前最佳。"); return current_population_expressions
    final_best_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"深度 1 Alpha 优化流程完成。最终选出 {len(final_best_expressions)} 个 Alpha。")
    return final_best_expressions

def best_d2_alphas(
    s: Any,
    best_d1_expressions: List[str],
    n_alphas_to_select: int,
    n_iterations: int,
    simulation_parameters: Dict[str, Any] = config.DEFAULT_CONFIG,
    fitness_function: Callable = DEFAULT_FITNESS_FUNCTION,
    mutation_function: Callable = DEFAULT_MUTATION_STRATEGY,
    crossover_function: Callable = DEFAULT_CROSSOVER_STRATEGY
) -> List[str]:
    logger.info(f"开始深度 2 Alpha 优化: 基于 {len(best_d1_expressions)} D1 Alphas, Top {n_alphas_to_select}, {n_iterations} 迭代.")
    if not best_d1_expressions: logger.error("输入D1 Alpha列表为空，无法生成D2 Alpha。"); return []
    best_d1_trees = d1_alpha_to_tree(best_d1_expressions)
    if not best_d1_trees: logger.error("从D1表达式转换的树列表为空。"); return []

    current_population_expressions: List[str] = []
    logger.info("步骤 1: 初始化深度 2 Alpha 种群...")
    initial_population_size = n_alphas_to_select * 2
    temp_expressions: List[str] = []
    for _ in range(initial_population_size):
        tree1, tree2 = random.sample(best_d1_trees, 2) if len(best_d1_trees) >= 2 else (best_d1_trees[0], best_d1_trees[0] if best_d1_trees else (None,None))
        if not tree1 or not tree2: logger.warning("D2初始化：未能选出足够的D1父树。"); continue
        d2_tree = depth_two_tree(tree1, tree2)
        try: temp_expressions.append(d2tree_to_alpha(d2_tree))
        except Exception as e: logger.error(f"转换初始D2树到Alpha时出错: {e}", exc_info=True)
    current_population_expressions = list(OrderedDict.fromkeys(temp_expressions))
    logger.info(f"初始D2种群生成完毕，共 {len(current_population_expressions)} 个Alpha。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d2_alphas initial - 使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in current_population_expressions]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in current_population_expressions]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=2, iteration_context=0)
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("初始D2 Alpha模拟未返回数据。流程终止。"); return []
    current_population_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"初始选择完成，选出 {len(current_population_expressions)} 个D2 Alpha进入迭代。")

    for iteration_num in range(n_iterations):
        logger.info(f"开始第 {iteration_num + 1}/{n_iterations} 轮D2 Alpha迭代...")
        new_random_expressions: List[str] = []
        for _ in range(n_alphas_to_select):
            tree1, tree2 = random.sample(best_d1_trees, 2) if len(best_d1_trees) >= 2 else (best_d1_trees[0], best_d1_trees[0] if best_d1_trees else (None,None))
            if not tree1 or not tree2: continue
            d2_tree = depth_two_tree(tree1, tree2)
            try: new_random_expressions.append(d2tree_to_alpha(d2_tree))
            except Exception as e: logger.error(f"迭代中转换D2树到Alpha时出错: {e}", exc_info=True)
        combined_expressions = list(OrderedDict.fromkeys(current_population_expressions + new_random_expressions))

        if config.DRY_RUN_MODE:
            logger.warning(f"警告: best_d2_alphas iteration {iteration_num + 1} - 使用模拟数据 (DRY_RUN_MODE)!")
            mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in combined_expressions]
            results_df = pd.DataFrame(mock_data)
        else:
            alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in combined_expressions]
            simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=2, iteration_context=iteration_num + 1)
            results_df = prettify_result(simulation_output_list)
        if results_df.empty: logger.warning(f"D2 Alpha 第 {iteration_num + 1} 轮迭代模拟未返回数据。"); continue
        current_population_expressions = fitness_function(results_df, n_alphas_to_select)
        logger.info(f"迭代 {iteration_num + 1} (D2)完成选择，当前种群数量: {len(current_population_expressions)}。")

    logger.info("D2 Alpha迭代完成，开始最终变异和交叉...")
    if not current_population_expressions: logger.warning("当前D2种群为空，无法变异/交叉。"); return []
    best_trees_d2 = d2_alpha_to_tree(current_population_expressions)
    mutated_d2_expressions: List[str] = []
    for tree_node in best_trees_d2:
        if tree_node:
            mutated_tree = mutation_function(tree_node)
            if mutated_tree:
                try: mutated_d2_expressions.append(d2tree_to_alpha(mutated_tree))
                except Exception as e: logger.error(f"转换变异后D2树到Alpha时出错: {e}", exc_info=True)
    crossed_d2_expressions: List[str] = []
    if len(best_trees_d2) >= 2:
        for i in range(0, len(best_trees_d2) -1, 2):
            p1, p2 = best_trees_d2[i], best_trees_d2[i+1]
            if p1 and p2:
                c1_tree, c2_tree = crossover_function(p1, p2)
                if c1_tree:
                    try: crossed_d2_expressions.append(d2tree_to_alpha(c1_tree))
                    except Exception as e: logger.error(f"转换交叉子树C1 (D2)到Alpha时出错: {e}", exc_info=True)
                if c2_tree:
                    try: crossed_d2_expressions.append(d2tree_to_alpha(c2_tree))
                    except Exception as e: logger.error(f"转换交叉子树C2 (D2)到Alpha时出错: {e}", exc_info=True)
    final_expressions_to_evaluate = list(OrderedDict.fromkeys(current_population_expressions + mutated_d2_expressions + crossed_d2_expressions))
    logger.info(f"变异/交叉阶段完成，共 {len(final_expressions_to_evaluate)} 个D2 Alpha待最终评估。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d2_alphas final eval - 使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,3), 'fitness': random.uniform(0,2), 'returns': random.uniform(-0.1,0.1), 'drawdown': random.uniform(0.01,0.2), 'turnover': random.uniform(0.05,0.5)} for expr in final_expressions_to_evaluate]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in final_expressions_to_evaluate]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=2, iteration_context=-1)
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("D2 Alpha最终评估未返回数据。"); return current_population_expressions
    final_best_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"深度 2 Alpha优化流程完成。最终选出 {len(final_best_expressions)} 个 Alpha。")
    return final_best_expressions

def best_d3_alpha(
    s: Any,
    best_d2_expressions: List[str],
    n_alphas_to_select: int,
    n_iterations: int,
    simulation_parameters: Dict[str, Any] = config.DEFAULT_CONFIG,
    fitness_function: Callable = DEFAULT_FITNESS_FUNCTION,
    mutation_function: Callable = DEFAULT_MUTATION_STRATEGY,
    crossover_function: Callable = DEFAULT_CROSSOVER_STRATEGY
) -> List[str]:
    logger.info(f"开始深度 3 Alpha 优化: 基于 {len(best_d2_expressions)} D2 Alphas, Top {n_alphas_to_select}, {n_iterations} 迭代.")
    if not best_d2_expressions: logger.error("输入D2 Alpha列表为空，无法生成D3 Alpha。"); return []
    best_d2_trees = d2_alpha_to_tree(best_d2_expressions)
    if not best_d2_trees: logger.error("从D2表达式转换的树列表为空。"); return []

    current_population_expressions: List[str] = []
    logger.info("步骤 1: 初始化深度 3 Alpha 种群...")
    initial_population_size = n_alphas_to_select * 2
    temp_expressions: List[str] = []
    for _ in range(initial_population_size):
        d3_tree = depth_three_tree(best_d2_trees)
        if d3_tree:
            try: temp_expressions.append(d3tree_to_alpha(d3_tree))
            except Exception as e: logger.error(f"转换初始D3树到Alpha时出错: {e}", exc_info=True)
    current_population_expressions = list(OrderedDict.fromkeys(temp_expressions))
    logger.info(f"初始D3种群生成完毕，共 {len(current_population_expressions)} 个Alpha。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d3_alphas initial - 使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,2.5), 'fitness': random.uniform(0,1.5), 'returns': random.uniform(-0.05,0.05), 'drawdown': random.uniform(0.01,0.15), 'turnover': random.uniform(0.1,0.6)} for expr in current_population_expressions]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in current_population_expressions]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=3, iteration_context=0)
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("初始D3 Alpha模拟未返回数据。流程终止。"); return []
    current_population_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"初始选择完成，选出 {len(current_population_expressions)} 个D3 Alpha进入迭代。")

    for iteration_num in range(n_iterations):
        logger.info(f"开始第 {iteration_num + 1}/{n_iterations} 轮D3 Alpha迭代...")
        new_random_expressions: List[str] = []
        for _ in range(n_alphas_to_select):
            d3_tree = depth_three_tree(best_d2_trees)
            if d3_tree:
                try: new_random_expressions.append(d3tree_to_alpha(d3_tree))
                except Exception as e: logger.error(f"迭代中转换D3树到Alpha时出错: {e}", exc_info=True)
        combined_expressions = list(OrderedDict.fromkeys(current_population_expressions + new_random_expressions))

        if config.DRY_RUN_MODE:
            logger.warning(f"警告: best_d3_alphas iteration {iteration_num + 1} - 使用模拟数据 (DRY_RUN_MODE)!")
            mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,2.5), 'fitness': random.uniform(0,1.5), 'returns': random.uniform(-0.05,0.05), 'drawdown': random.uniform(0.01,0.15), 'turnover': random.uniform(0.1,0.6)} for expr in combined_expressions]
            results_df = pd.DataFrame(mock_data)
        else:
            alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in combined_expressions]
            simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=3, iteration_context=iteration_num + 1)
            results_df = prettify_result(simulation_output_list)
        if results_df.empty: logger.warning(f"D3 Alpha 第 {iteration_num + 1} 轮迭代模拟未返回数据。"); continue
        current_population_expressions = fitness_function(results_df, n_alphas_to_select)
        logger.info(f"迭代 {iteration_num + 1} (D3)完成选择，当前种群数量: {len(current_population_expressions)}。")

    logger.info("D3 Alpha迭代完成，开始最终变异和交叉...") # Added crossover for D3 as well
    if not current_population_expressions: logger.warning("当前D3种群为空，无法变异/交叉。"); return []
    best_trees_d3 = d3_alpha_to_tree(current_population_expressions) # d3_alpha_to_tree's robustness is still a concern
    mutated_d3_expressions: List[str] = []
    for tree_node in best_trees_d3: # best_trees_d3 can be empty if d3_alpha_to_tree fails for all
        if tree_node:
            mutated_tree = mutation_function(tree_node)
            if mutated_tree:
                try: mutated_d3_expressions.append(d3tree_to_alpha(mutated_tree))
                except Exception as e: logger.error(f"转换变异后D3树到Alpha时出错: {e}", exc_info=True)

    crossed_d3_expressions: List[str] = [] # Added crossover for D3
    if len(best_trees_d3) >=2:
        for i in range(0, len(best_trees_d3) - 1, 2):
            p1, p2 = best_trees_d3[i], best_trees_d3[i+1]
            if p1 and p2:
                c1_tree, c2_tree = crossover_function(p1, p2)
                if c1_tree:
                    try: crossed_d3_expressions.append(d3tree_to_alpha(c1_tree))
                    except Exception as e: logger.error(f"转换交叉子树C1 (D3)到Alpha时出错: {e}", exc_info=True)
                if c2_tree:
                    try: crossed_d3_expressions.append(d3tree_to_alpha(c2_tree))
                    except Exception as e: logger.error(f"转换交叉子树C2 (D3)到Alpha时出错: {e}", exc_info=True)

    final_expressions_to_evaluate = list(OrderedDict.fromkeys(
        current_population_expressions + mutated_d3_expressions + crossed_d3_expressions
    ))
    logger.info(f"变异/交叉阶段完成，共 {len(final_expressions_to_evaluate)} 个D3 Alpha 待最终评估。")

    if config.DRY_RUN_MODE:
        logger.warning("警告: best_d3_alphas final eval - 使用模拟数据 (DRY_RUN_MODE)!")
        mock_data = [{'expression': expr, 'sharpe': random.uniform(-1,2.5), 'fitness': random.uniform(0,1.5), 'returns': random.uniform(-0.05,0.05), 'drawdown': random.uniform(0.01,0.15), 'turnover': random.uniform(0.1,0.6)} for expr in final_expressions_to_evaluate]
        results_df = pd.DataFrame(mock_data)
    else:
        alpha_sim_data_list = [generate_alpha_simulation_data(expr) for expr in final_expressions_to_evaluate]
        simulation_output_list = simulate_alpha_list(s, alpha_sim_data_list, simulation_parameters=simulation_parameters, depth_context=3, iteration_context=-1) # Final
        results_df = prettify_result(simulation_output_list)

    if results_df.empty: logger.error("D3 Alpha最终评估未返回数据。"); return current_population_expressions
    final_best_expressions = fitness_function(results_df, n_alphas_to_select)
    logger.info(f"深度 3 Alpha优化流程完成。最终选出 {len(final_best_expressions)} 个 Alpha。")
    return final_best_expressions

pass
