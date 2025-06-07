# alpha_factory/genetic_programming/engine.py
import importlib
import random
from typing import List, Dict, Any, Callable # Callable 用于类型提示
from collections import OrderedDict # 用于去重时保持顺序，与 code.py 行为一致

from alpha_factory.utils.config_models import AppConfig
from alpha_factory.brain_client.api_client import BrainApiClient
from alpha_factory.fitness.base_fitness import BaseFitnessCalculator
from alpha_factory.genetic_programming import models # Node 定义
# 别名导入，与 cg.md 一致
from alpha_factory.genetic_programming.generators import legacy_generator as gen
from alpha_factory.genetic_programming.operators import legacy_operators as ops
from alpha_factory.utils import legacy_expression_converter as conv

# code.py 中有一个全局的 generate_alpha 函数，用于将表达式字符串包装成API所需的模拟数据格式。
# 这个函数应该被迁移到 StagedGPEngine 或 BrainApiClient 中。
# cg.md 中未明确指出其位置，但它在 StagedGPEngine 的各阶段方法中被频繁调用。
# 我们将其作为 StagedGPEngine 的一个辅助方法。

class StagedGPEngine:
    """
    分阶段遗传编程引擎。
    负责编排Alpha的生成、模拟、评估和进化过程，以复刻 code.py 的完整流程。
    """
    def __init__(self, config: AppConfig, api_client: BrainApiClient):
        """
        初始化分阶段遗传编程引擎。

        Args:
            config (AppConfig): 应用程序的配置对象。
            api_client (BrainApiClient): 用于与BRAIN API交互的客户端实例。
        """
        self.config = config
        self.api_client = api_client

        # 从配置中获取遗传编程参数
        self.gp_params = config.gp
        # 从配置中获取操作符列表 (尽管旧版模块可能不直接使用这些，但为了结构完整性先获取)
        self.operator_config = config.operators

        # 加载适应度计算器
        self.fitness_calculator: BaseFitnessCalculator = self._load_fitness_calculator()

        # 迁移 generate_alpha 函数所需的参数，这些参数来自 config.brain
        self.alpha_generation_settings = {
            "region": self.config.brain.region,
            "universe": self.config.brain.universe,
            "neutralization": self.config.brain.neutralization,
            "delay": self.config.brain.delay,
            "decay": self.config.brain.decay,
            "truncation": self.config.brain.truncation,
            "nan_handling": self.config.brain.nan_handling,
            "unit_handling": self.config.brain.unit_handling,
            "pasteurization": self.config.brain.pasteurization,
            "visualization": False # 通常在自动化流程中设为 False
        }
        # concurrent_simulations 提取到这里，避免重复 getattr
        self.concurrent_simulations = getattr(self.config.brain, "concurrent_simulations", 3)
        if not isinstance(self.concurrent_simulations, int) or self.concurrent_simulations <=0:
            print(f"警告 (StagedGPEngine): 配置中的 concurrent_simulations ({self.concurrent_simulations}) 无效，将使用默认值 3。")
            self.concurrent_simulations = 3

        print("信息 (StagedGPEngine): 引擎已成功初始化。")

    def _generate_alpha_sim_data(self, regular_expression: str) -> Dict[str, Any]:
        """
        根据Alpha表达式字符串生成用于BRAIN API模拟的数据字典。
        此方法迁移自 code.py 中的全局 `generate_alpha` 函数。

        Args:
            regular_expression (str): Alpha的表达式字符串。

        Returns:
            Dict[str, Any]: 包含Alpha模拟设置和表达式的字典。
        """
        # 使用从 self.config.brain 中提取的设置
        return {
            "type": "REGULAR", # Alpha 类型，通常是 REGULAR
            "settings": {
                "nanHandling": self.alpha_generation_settings["nan_handling"],
                "instrumentType": "EQUITY", # 假设总是 EQUITY，或从配置读取
                "delay": self.alpha_generation_settings["delay"],
                "universe": self.alpha_generation_settings["universe"],
                "truncation": self.alpha_generation_settings["truncation"],
                "unitHandling": self.alpha_generation_settings["unit_handling"],
                "pasteurization": self.alpha_generation_settings["pasteurization"],
                "region": self.alpha_generation_settings["region"],
                "language": "FASTEXPR", # 表达式语言
                "decay": self.alpha_generation_settings["decay"],
                "neutralization": self.alpha_generation_settings["neutralization"],
                "visualization": self.alpha_generation_settings["visualization"],
            },
            "regular": regular_expression,
        }

    def _load_fitness_calculator(self) -> BaseFitnessCalculator:
        """
        根据配置文件动态加载并实例化适应度计算器。
        """
        print(f"信息 (StagedGPEngine): 正在加载适应度计算器: {self.config.fitness.module}.{self.config.fitness.class_name}")
        try:
            module_path = self.config.fitness.module
            class_name = self.config.fitness.class_name

            module = importlib.import_module(module_path)
            calculator_class = getattr(module, class_name)

            # 传递在配置文件中为适应度函数定义的参数
            instance = calculator_class(**self.config.fitness.params)
            print("信息 (StagedGPEngine): 适应度计算器加载成功。")
            return instance
        except ImportError as e:
            print(f"错误 (StagedGPEngine): 无法导入适应度计算器模块 '{module_path}': {e}")
            raise
        except AttributeError as e:
            print(f"错误 (StagedGPEngine): 在模块 '{module_path}' 中未找到适应度计算器类 '{class_name}': {e}")
            raise
        except Exception as e:
            print(f"错误 (StagedGPEngine): 加载适应度计算器时发生未知错误: {e}")
            raise

    def run(self):
        """
        按顺序执行三个阶段的遗传编程进化流程。
        """
        print("\n--- 开始执行AlphaFactory分阶段遗传编程引擎 ---")

        # 运行第一阶段
        print("\n--- [阶段 1] 开始：优化深度为 1 的 Alpha ---")
        best_d1_expressions = self._run_d1_stage()
        if not best_d1_expressions:
            print("错误 (StagedGPEngine.run): 阶段 1 未能产生有效的Alpha表达式，引擎终止。")
            return
        print(f"--- [阶段 1] 完成：选出 {len(best_d1_expressions)} 个最佳深度1 Alpha。")
        # print("阶段1最佳Alpha:", best_d1_expressions)


        # 运行第二阶段
        print("\n--- [阶段 2] 开始：优化深度为 2 的 Alpha ---")
        best_d2_expressions = self._run_d2_stage(best_d1_expressions)
        if not best_d2_expressions:
            print("错误 (StagedGPEngine.run): 阶段 2 未能产生有效的Alpha表达式，引擎终止。")
            return
        print(f"--- [阶段 2] 完成：选出 {len(best_d2_expressions)} 个最佳深度2 Alpha。")
        # print("阶段2最佳Alpha:", best_d2_expressions)

        # 运行第三阶段
        print("\n--- [阶段 3] 开始：优化深度为 3 的 Alpha ---")
        best_d3_expressions = self._run_d3_stage(best_d2_expressions)
        if not best_d3_expressions: # 检查是否为空列表，而非 None
            print("错误 (StagedGPEngine.run): 阶段 3 未能产生有效的Alpha表达式，引擎终止。")
            return
        print(f"--- [阶段 3] 完成：选出 {len(best_d3_expressions)} 个最佳深度3 Alpha。")

        print("\n--- 最终选出的最佳Alpha (来自深度3) ---")
        if best_d3_expressions:
            for i, alpha_expr in enumerate(best_d3_expressions):
                print(f"  {i+1}. {alpha_expr}")
        else:
            print("  未能选出任何最终的Alpha。")

        print("\n--- AlphaFactory分阶段遗传编程引擎执行完毕 ---")


    def _run_d1_stage(self) -> List[str]:
        """
        执行深度1 Alpha的生成、评估和进化。
        复刻 code.py 中 best_d1_alphas 函数的核心逻辑。
        """
        population_exprs: List[str] = []
        num_to_retain = self.gp_params.d1_population
        num_generations = self.gp_params.d1_generations
        initial_pop_size = num_to_retain * 2

        print(f"信息 (阶段1): 目标保留数量 {num_to_retain}, 迭代次数 {num_generations}, 初始种群大小 {initial_pop_size}")
        print("信息 (阶段1): 开始生成初始种群...")

        generated_trees_d1: List[models.Node] = []
        for _ in range(initial_pop_size):
            flag = random.choice([0, 1])
            d1_tree = gen.depth_one_trees(
                current_terminal_values=self.operator_config.terminal_values,
                current_binary_ops=self.operator_config.binary_ops,
                current_ts_ops=self.operator_config.ts_ops,
                current_ts_ops_values=self.operator_config.ts_ops_values,
                current_unary_ops=self.operator_config.unary_ops,
                flag=flag
            )
            generated_trees_d1.append(d1_tree)

        population_exprs = [conv.d1tree_to_alpha(tree) for tree in generated_trees_d1 if tree]
        population_exprs = list(OrderedDict.fromkeys(expr for expr in population_exprs if expr))

        print(f"信息 (阶段1): 初始种群生成完毕，共 {len(population_exprs)} 个不重复 Alpha 表达式。")

        alpha_sim_data_list = [self._generate_alpha_sim_data(expr) for expr in population_exprs]
        print(f"信息 (阶段1): 正在模拟和评估初始种群 ({len(alpha_sim_data_list)} 个Alpha)...")
        simulation_results = self.api_client.run_simulation_workflow(
            alpha_sim_data_list,
            limit_concurrent=self.concurrent_simulations,
            depth=1,
            iteration=0
        )

        population_exprs = self.fitness_calculator.run(simulation_results, num_to_retain)
        if not population_exprs:
            print("警告 (阶段1): 初始种群评估后未能选出任何Alpha。")
            return []
        print(f"信息 (阶段1): 初始评估完成，选出 {len(population_exprs)} 个Alpha进入迭代。")

        for gen_idx in range(num_generations):
            print(f"\n信息 (阶段1): 开始第 {gen_idx + 1}/{num_generations} 代进化...")

            print(f"信息 (阶段1 第{gen_idx+1}代): 正在补充 {num_to_retain} 个新个体到种群...")
            new_individuals_trees: List[models.Node] = []
            for _ in range(num_to_retain):
                flag = random.choice([0, 1])
                d1_tree = gen.depth_one_trees(
                    self.operator_config.terminal_values, self.operator_config.binary_ops,
                    self.operator_config.ts_ops, self.operator_config.ts_ops_values,
                    self.operator_config.unary_ops, flag=flag
                )
                new_individuals_trees.append(d1_tree)

            new_expressions = [conv.d1tree_to_alpha(tree) for tree in new_individuals_trees if tree]
            current_generation_exprs = list(OrderedDict.fromkeys(population_exprs + [expr for expr in new_expressions if expr]))

            print(f"信息 (阶段1 第{gen_idx+1}代): 补充后种群大小 {len(current_generation_exprs)} (去重)。正在评估...")

            alpha_sim_data_list_iter = [self._generate_alpha_sim_data(expr) for expr in current_generation_exprs]
            sim_results_iter = self.api_client.run_simulation_workflow(
                alpha_sim_data_list_iter,
                limit_concurrent=self.concurrent_simulations,
                depth=1,
                iteration=gen_idx + 1
            )

            population_exprs = self.fitness_calculator.run(sim_results_iter, num_to_retain)
            if not population_exprs:
                print(f"警告 (阶段1 第{gen_idx+1}代): 评估后未能选出任何Alpha，可能提前终止此阶段。")
                return []
            print(f"信息 (阶段1 第{gen_idx+1}代): 评估完成，选出 {len(population_exprs)} 个Alpha。")

        print("\n信息 (阶段1): 所有进化代数完成，开始最终的变异阶段...")
        if not population_exprs:
            print("警告 (阶段1 变异): 上一代种群为空，无法进行变异。")
            return []

        best_trees_before_mutation = [conv.d1_alpha_to_tree([expr])[0] for expr in population_exprs if conv.d1_alpha_to_tree([expr])]

        mutated_trees: List[models.Node] = []
        for tree_node in best_trees_before_mutation:
            if tree_node:
                mutated_node = ops.mutate_random_node(
                    tree_node,
                    self.operator_config.terminal_values, self.operator_config.unary_ops,
                    self.operator_config.binary_ops, self.operator_config.ts_ops,
                    self.operator_config.ts_ops_values
                )
                if mutated_node:
                    mutated_trees.append(mutated_node)

        combined_trees_for_final_eval = best_trees_before_mutation + mutated_trees
        final_expressions_to_eval = [conv.d1tree_to_alpha(tree) for tree in combined_trees_for_final_eval if tree]
        final_expressions_to_eval = list(OrderedDict.fromkeys(expr for expr in final_expressions_to_eval if expr))

        print(f"信息 (阶段1 变异): 变异完成，总共 {len(final_expressions_to_eval)} 个Alpha待最终评估。")

        if not final_expressions_to_eval:
            print("警告 (阶段1 变异): 变异后没有有效的Alpha表达式进行最终评估。")
            return population_exprs

        alpha_sim_data_list_final = [self._generate_alpha_sim_data(expr) for expr in final_expressions_to_eval]
        sim_results_final = self.api_client.run_simulation_workflow(
            alpha_sim_data_list_final,
            limit_concurrent=self.concurrent_simulations,
            depth=1,
            iteration=-1
        )

        final_best_expressions = self.fitness_calculator.run(sim_results_final, num_to_retain)
        if not final_best_expressions:
            print("警告 (阶段1 变异): 最终评估后未能选出任何Alpha。")
            return population_exprs if population_exprs else []

        print(f"信息 (阶段1): 深度1 Alpha优化完成。最终选出 {len(final_best_expressions)} 个Alpha。")
        return final_best_expressions


    def _run_d2_stage(self, onetree_exprs: List[str]) -> List[str]:
        """
        执行深度2 Alpha的生成、评估和进化。
        """
        population_exprs: List[str] = []
        num_to_retain = self.gp_params.d2_population
        num_generations = self.gp_params.d2_generations
        initial_pop_size = num_to_retain * 2

        print(f"信息 (阶段2): 目标保留数量 {num_to_retain}, 迭代次数 {num_generations}, 初始种群大小 {initial_pop_size}")

        best_d1_trees = [conv.d1_alpha_to_tree([expr])[0] for expr in onetree_exprs if conv.d1_alpha_to_tree([expr])]
        if not best_d1_trees:
            print("错误 (阶段2): 无法从阶段1的结果转换出有效的深度1树，阶段2终止。")
            return []
        print(f"信息 (阶段2): 成功从阶段1结果转换 {len(best_d1_trees)} 个深度1树作为构建块。")
        print("信息 (阶段2): 开始生成初始种群...")

        generated_trees_d2: List[models.Node] = []
        for _ in range(initial_pop_size):
            flag = random.choice([0, 1])
            tree1 = random.choice(best_d1_trees)
            tree2 = random.choice(best_d1_trees)
            d2_tree = gen.depth_two_tree(
                tree1, tree2,
                self.operator_config.ts_ops_values,
                self.operator_config.ts_ops,
                self.operator_config.binary_ops,
                flag=flag
            )
            generated_trees_d2.append(d2_tree)

        population_exprs = [conv.d2tree_to_alpha(tree) for tree in generated_trees_d2 if tree]
        population_exprs = list(OrderedDict.fromkeys(expr for expr in population_exprs if expr))

        print(f"信息 (阶段2): 初始种群生成完毕，共 {len(population_exprs)} 个不重复 Alpha 表达式。")

        alpha_sim_data_list = [self._generate_alpha_sim_data(expr) for expr in population_exprs]
        print(f"信息 (阶段2): 正在模拟和评估初始种群 ({len(alpha_sim_data_list)} 个Alpha)...")
        simulation_results = self.api_client.run_simulation_workflow(
            alpha_sim_data_list,
            limit_concurrent=self.concurrent_simulations,
            depth=2, iteration=0
        )
        population_exprs = self.fitness_calculator.run(simulation_results, num_to_retain)
        if not population_exprs:
            print("警告 (阶段2): 初始种群评估后未能选出任何Alpha。")
            return []
        print(f"信息 (阶段2): 初始评估完成，选出 {len(population_exprs)} 个Alpha进入迭代。")

        for gen_idx in range(num_generations):
            print(f"\n信息 (阶段2): 开始第 {gen_idx + 1}/{num_generations} 代进化...")
            print(f"信息 (阶段2 第{gen_idx+1}代): 正在补充 {num_to_retain} 个新个体到种群...")
            new_individuals_trees_d2: List[models.Node] = []
            for _ in range(num_to_retain):
                flag = random.choice([0, 1])
                tree1 = random.choice(best_d1_trees)
                tree2 = random.choice(best_d1_trees)
                d2_tree = gen.depth_two_tree(
                    tree1, tree2, self.operator_config.ts_ops_values,
                    self.operator_config.ts_ops, self.operator_config.binary_ops, flag=flag
                )
                new_individuals_trees_d2.append(d2_tree)

            new_expressions = [conv.d2tree_to_alpha(tree) for tree in new_individuals_trees_d2 if tree]
            current_generation_exprs = list(OrderedDict.fromkeys(population_exprs + [expr for expr in new_expressions if expr]))

            print(f"信息 (阶段2 第{gen_idx+1}代): 补充后种群大小 {len(current_generation_exprs)}。正在评估...")
            alpha_sim_data_list_iter = [self._generate_alpha_sim_data(expr) for expr in current_generation_exprs]
            sim_results_iter = self.api_client.run_simulation_workflow(
                alpha_sim_data_list_iter,
                limit_concurrent=self.concurrent_simulations,
                depth=2, iteration=gen_idx + 1
            )
            population_exprs = self.fitness_calculator.run(sim_results_iter, num_to_retain)
            if not population_exprs:
                print(f"警告 (阶段2 第{gen_idx+1}代): 评估后未能选出任何Alpha。")
                return []
            print(f"信息 (阶段2 第{gen_idx+1}代): 评估完成，选出 {len(population_exprs)} 个Alpha。")

        print("\n信息 (阶段2): 所有进化代数完成，开始最终的变异和交叉...")
        if not population_exprs:
            print("警告 (阶段2 变异/交叉): 上一代种群为空。")
            return []

        print("信息 (阶段2): 执行变异操作...")
        best_d2_trees_for_mutation = [conv.d2_alpha_to_tree([expr])[0] for expr in population_exprs if conv.d2_alpha_to_tree([expr])]

        mutated_d2_trees: List[models.Node] = []
        for tree_node in best_d2_trees_for_mutation:
            if tree_node:
                mutated_node = ops.mutate_random_node(
                    tree_node, self.operator_config.terminal_values, self.operator_config.unary_ops,
                    self.operator_config.binary_ops, self.operator_config.ts_ops, self.operator_config.ts_ops_values
                )
                if mutated_node:
                    mutated_d2_trees.append(mutated_node)

        expressions_after_mutation_raw = [conv.d2tree_to_alpha(tree) for tree in best_d2_trees_for_mutation + mutated_d2_trees if tree]
        expressions_after_mutation = list(OrderedDict.fromkeys(expr for expr in expressions_after_mutation_raw if expr))

        print(f"信息 (阶段2 变异): 变异后共 {len(expressions_after_mutation)} 个Alpha。正在评估...")
        sim_data_mut = [self._generate_alpha_sim_data(expr) for expr in expressions_after_mutation]
        results_mut = self.api_client.run_simulation_workflow(
            sim_data_mut, self.concurrent_simulations,
            depth=2, iteration=-1
        )
        population_after_mutation = self.fitness_calculator.run(results_mut, num_to_retain)
        if not population_after_mutation:
            print("警告 (阶段2 变异): 变异种群评估后未能选出任何Alpha。保留变异前种群。")
            population_after_mutation = list(population_exprs)

        print("\n信息 (阶段2): 执行交叉操作...")
        trees_for_crossover = [conv.d2_alpha_to_tree([expr])[0] for expr in population_after_mutation if conv.d2_alpha_to_tree([expr])]
        if not trees_for_crossover or len(trees_for_crossover) < 2:
            print("警告 (阶段2 交叉): 用于交叉的树不足两个，跳过交叉。")
            final_best_expressions_d2 = population_after_mutation
        else:
            crossed_d2_trees: List[models.Node] = []
            # Iterate to almost the end to ensure pairs for crossover
            for i in range(0, len(trees_for_crossover) - (len(trees_for_crossover) % 2), 2):
                p1, p2 = trees_for_crossover[i], trees_for_crossover[i+1]
                if p1 and p2: # Ensure both parents are valid nodes
                    c1, c2 = ops.crossover(p1, p2, n=2)
                    if c1: crossed_d2_trees.append(c1)
                    if c2: crossed_d2_trees.append(c2)

            expressions_after_crossover_raw = [conv.d2tree_to_alpha(tree) for tree in trees_for_crossover + crossed_d2_trees if tree]
            expressions_after_crossover = list(OrderedDict.fromkeys(expr for expr in expressions_after_crossover_raw if expr))

            print(f"信息 (阶段2 交叉): 交叉后共 {len(expressions_after_crossover)} 个Alpha。正在评估...")
            sim_data_cross = [self._generate_alpha_sim_data(expr) for expr in expressions_after_crossover]
            results_cross = self.api_client.run_simulation_workflow(
                sim_data_cross, self.concurrent_simulations,
                depth=2, iteration=-2
            )
            population_after_crossover = self.fitness_calculator.run(results_cross, num_to_retain)
            if not population_after_crossover:
                print("警告 (阶段2 交叉): 交叉种群评估后未能选出任何Alpha。保留交叉前种群。")
                final_best_expressions_d2 = population_after_mutation
            else:
                final_best_expressions_d2 = population_after_crossover

        print(f"信息 (阶段2): 深度2 Alpha优化完成。最终选出 {len(final_best_expressions_d2)} 个Alpha。")
        return final_best_expressions_d2


    def _run_d3_stage(self, twotree_exprs: List[str]) -> List[str]:
        """
        执行深度3 Alpha的生成、评估和进化。
        """
        population_exprs: List[str] = []
        num_to_retain = self.gp_params.d3_population
        num_generations = self.gp_params.d3_generations
        initial_pop_size = num_to_retain * 2

        print(f"信息 (阶段3): 目标保留数量 {num_to_retain}, 迭代次数 {num_generations}, 初始种群大小 {initial_pop_size}")

        best_d2_trees = [conv.d2_alpha_to_tree([expr])[0] for expr in twotree_exprs if conv.d2_alpha_to_tree([expr])]
        if not best_d2_trees:
            print("错误 (阶段3): 无法从阶段2的结果转换出有效的深度2树，阶段3终止。")
            return []
        print(f"信息 (阶段3): 成功从阶段2结果转换 {len(best_d2_trees)} 个深度2树作为构建块。")
        print("信息 (阶段3): 开始生成初始种群...")

        generated_trees_d3: List[models.Node] = []
        for _ in range(initial_pop_size):
            flag = random.choice([0, 1, 2])
            d3_tree = gen.depth_three_tree(
                best_d2_trees,
                self.operator_config.unary_ops,
                self.operator_config.binary_ops,
                self.operator_config.ts_ops,
                self.operator_config.ts_ops_values,
                flag=flag
            )
            if d3_tree : generated_trees_d3.append(d3_tree)

        population_exprs = [conv.d3tree_to_alpha(tree) for tree in generated_trees_d3 if tree]
        population_exprs = list(OrderedDict.fromkeys(expr for expr in population_exprs if expr))
        print(f"信息 (阶段3): 初始种群生成完毕，共 {len(population_exprs)} 个不重复 Alpha 表达式。")

        alpha_sim_data_list = [self._generate_alpha_sim_data(expr) for expr in population_exprs]
        print(f"信息 (阶段3): 正在模拟和评估初始种群 ({len(alpha_sim_data_list)} 个Alpha)...")
        simulation_results = self.api_client.run_simulation_workflow(
            alpha_sim_data_list,
            limit_concurrent=self.concurrent_simulations,
            depth=3, iteration=0
        )
        population_exprs = self.fitness_calculator.run(simulation_results, num_to_retain)
        if not population_exprs:
            print("警告 (阶段3): 初始种群评估后未能选出任何Alpha。")
            return []
        print(f"信息 (阶段3): 初始评估完成，选出 {len(population_exprs)} 个Alpha进入迭代。")

        for gen_idx in range(num_generations):
            print(f"\n信息 (阶段3): 开始第 {gen_idx + 1}/{num_generations} 代进化...")
            print(f"信息 (阶段3 第{gen_idx+1}代): 正在补充 {num_to_retain} 个新个体到种群...")
            new_individuals_trees_d3: List[models.Node] = []
            for _ in range(num_to_retain):
                flag = random.choice([0, 1, 2])
                d3_tree = gen.depth_three_tree(
                    best_d2_trees, self.operator_config.unary_ops, self.operator_config.binary_ops,
                    self.operator_config.ts_ops, self.operator_config.ts_ops_values, flag=flag
                )
                if d3_tree: new_individuals_trees_d3.append(d3_tree)

            new_expressions = [conv.d3tree_to_alpha(tree) for tree in new_individuals_trees_d3 if tree]
            current_generation_exprs = list(OrderedDict.fromkeys(population_exprs + [expr for expr in new_expressions if expr]))

            print(f"信息 (阶段3 第{gen_idx+1}代): 补充后种群大小 {len(current_generation_exprs)}。正在评估...")
            alpha_sim_data_list_iter = [self._generate_alpha_sim_data(expr) for expr in current_generation_exprs]
            sim_results_iter = self.api_client.run_simulation_workflow(
                alpha_sim_data_list_iter,
                limit_concurrent=self.concurrent_simulations,
                depth=3, iteration=gen_idx + 1
            )
            population_exprs = self.fitness_calculator.run(sim_results_iter, num_to_retain)
            if not population_exprs:
                print(f"警告 (阶段3 第{gen_idx+1}代): 评估后未能选出任何Alpha。")
                return []
            print(f"信息 (阶段3 第{gen_idx+1}代): 评估完成，选出 {len(population_exprs)} 个Alpha。")

        print("\n信息 (阶段3): 所有进化代数完成，开始最终的变异阶段...")
        if not population_exprs:
            print("警告 (阶段3 变异): 上一代种群为空。")
            return []

        best_d3_trees_for_mutation = [conv.d3_alpha_to_tree([expr])[0] for expr in population_exprs if conv.d3_alpha_to_tree([expr])]

        mutated_d3_trees: List[models.Node] = []
        for tree_node in best_d3_trees_for_mutation:
            if tree_node:
                mutated_node = ops.mutate_random_node(
                    tree_node, self.operator_config.terminal_values, self.operator_config.unary_ops,
                    self.operator_config.binary_ops, self.operator_config.ts_ops, self.operator_config.ts_ops_values
                )
                if mutated_node:
                    mutated_d3_trees.append(mutated_node)

        combined_trees_final_d3 = best_d3_trees_for_mutation + mutated_d3_trees
        final_expressions_to_eval_d3 = [conv.d3tree_to_alpha(tree) for tree in combined_trees_final_d3 if tree]
        final_expressions_to_eval_d3 = list(OrderedDict.fromkeys(expr for expr in final_expressions_to_eval_d3 if expr))

        print(f"信息 (阶段3 变异): 变异完成，总共 {len(final_expressions_to_eval_d3)} 个Alpha待最终评估。")

        if not final_expressions_to_eval_d3:
            print("警告 (阶段3 变异): 变异后没有有效的Alpha表达式进行最终评估。")
            return population_exprs

        alpha_sim_data_list_final_d3 = [self._generate_alpha_sim_data(expr) for expr in final_expressions_to_eval_d3]
        sim_results_final_d3 = self.api_client.run_simulation_workflow(
            alpha_sim_data_list_final_d3,
            limit_concurrent=self.concurrent_simulations,
            depth=3, iteration=-1
        )

        final_best_expressions_d3 = self.fitness_calculator.run(sim_results_final_d3, num_to_retain)
        if not final_best_expressions_d3:
            print("警告 (阶段3 变异): 最终评估后未能选出任何Alpha。")
            return population_exprs if population_exprs else []

        print(f"信息 (阶段3): 深度3 Alpha优化完成。最终选出 {len(final_best_expressions_d3)} 个Alpha。")
        return final_best_expressions_d3
