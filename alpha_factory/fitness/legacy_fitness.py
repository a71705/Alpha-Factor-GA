# alpha_factory/fitness/legacy_fitness.py
import pandas as pd
from typing import List, Dict, Any, Optional # 确保导入了需要的类型 (加入 Optional)
from .base_fitness import BaseFitnessCalculator # 从同级目录的 base_fitness 导入基类

class LegacyFitness(BaseFitnessCalculator):
    """
    封装了源自 code.py 的旧版适应度计算逻辑。
    这包括结果的格式化（prettify_result）和基于特定公式的适应度排序（fitness_fun）。
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化 LegacyFitness 计算器。

        Args:
            params (Optional[Dict[str, Any]]): 适应度计算相关的参数。
                                              例如，可以包含 `prettify_result` 或 `fitness_fun` 所需的特定设置。
                                              在当前旧版实现中，此参数未使用，但为未来扩展保留。
        """
        self.params = params if params is not None else {}
        # n 是动态的，由 run 方法传入，不在构造时固定

    def run(self, results: List[Dict[str, Any]], n: int) -> List[str]:
        """
        执行旧版的适应度计算流程。

        Args:
            results (List[Dict[str, Any]]): Alpha模拟结果列表，
                                           每个元素是一个包含 'is_stats', 'alpha_id', 'simulate_data', 'is_tests' 等键的字典。
            n (int): 要选择的最佳Alpha数量。

        Returns:
            List[str]: 排名最高的n个Alpha表达式。如果无有效结果或计算出错，则返回空列表。
        """
        # 添加中文注释：调用内部方法 _prettify_result 处理和格式化原始结果
        prettified_df = self._prettify_result(results)

        # 添加中文注释：如果格式化后的DataFrame为空，则无法进行适应度计算，返回空列表
        if prettified_df.empty:
            print("警告 (LegacyFitness): 经过 prettify_result 处理后没有有效的 Alpha 数据，无法计算适应度。")
            return []

        # 添加中文注释：调用内部方法 _fitness_fun 计算适应度并选出最优的n个
        top_n_alphas = self._fitness_fun(prettified_df, n)
        return top_n_alphas

    def _prettify_result(self, result: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        将模拟结果整合到一个 DataFrame 中，以便分析 Alpha。
        结果按 fitness 绝对值降序排列。此逻辑迁移自 code.py 中的 prettify_result。

        Args:
            result (List[Dict[str, Any]]): 包含每个 Alpha 模拟结果的列表。
                                          每个字典应包含 'is_stats', 'alpha_id', 'simulate_data', 'is_tests'。

        Returns:
            pd.DataFrame: 包含 Alpha 统计信息、表达式和测试结果的 DataFrame。
                          如果无法生成有效的DataFrame，则返回空DataFrame。
        """
        # --- Start of code migrated from prettify_result ---
        # 添加中文注释：提取并合并所有 Alpha 的 IS (In-Sample) 统计数据
        list_of_is_stats = []
        for x_res in result: # 使用更明确的变量名
            if x_res and isinstance(x_res.get("is_stats"), pd.DataFrame) and not x_res["is_stats"].empty:
                list_of_is_stats.append(x_res["is_stats"])

        is_stats_df = pd.DataFrame() # 初始化为空DataFrame
        if list_of_is_stats:
            valid_stats = [df for df in list_of_is_stats if not df.empty and not df.isna().all().all()]
            if valid_stats:
                try:
                    is_stats_df = pd.concat(valid_stats, ignore_index=True)
                except Exception as e:
                    print(f"警告 (_prettify_result): 合并is_stats时出错: {e}")
                    # is_stats_df 保持为空

        if is_stats_df.empty or "fitness" not in is_stats_df.columns:
             # print("警告 (_prettify_result): is_stats_df 为空或缺少 'fitness' 列，无法排序。")
             # 如果没有有效的 is_stats，后续合并可能意义不大，但为了尽量保持原逻辑，我们继续
             pass # 保持 is_stats_df 为空或无fitness列的状态
        else:
            # 添加中文注释：按 fitness 降序排序
            is_stats_df = is_stats_df.sort_values("fitness", ascending=False)

        # 添加中文注释：提取所有 Alpha 的表达式
        expressions = {}
        for x_res in result:
            if x_res and x_res.get("alpha_id") and isinstance(x_res.get("simulate_data"), dict) and x_res["simulate_data"].get("regular"):
                expressions[x_res["alpha_id"]] = x_res["simulate_data"]["regular"]

        expression_df = pd.DataFrame(list(expressions.items()), columns=["alpha_id", "expression"])

        # 添加中文注释：提取并合并所有 Alpha 的 IS 测试结果
        list_of_is_tests = []
        for x_res in result:
            if x_res and isinstance(x_res.get("is_tests"), pd.DataFrame) and not x_res["is_tests"].empty:
                list_of_is_tests.append(x_res["is_tests"])

        is_tests_df = pd.DataFrame() # 初始化为空DataFrame
        if list_of_is_tests:
            try:
                is_tests_df = pd.concat(list_of_is_tests).reset_index(drop=True)
            except Exception as e:
                print(f"警告 (_prettify_result): 合并is_tests时出错: {e}")
                # is_tests_df 保持为空

        # 原 prettify_result 中有 detailed_tests_view 参数，此处我们按 False（默认）处理
        # 如果 is_tests_df 非空且包含必要列
        if not is_tests_df.empty and all(col in is_tests_df.columns for col in ["alpha_id", "name", "result"]):
            try:
                is_tests_df = is_tests_df.pivot(
                    index="alpha_id", columns="name", values="result"
                ).reset_index()
            except Exception as e: # 可能因为重复的 (alpha_id, name) 导致 pivot 失败
                print(f"警告 (_prettify_result): pivot is_tests_df 时出错: {e}。尝试去重后pivot。")
                try:
                    is_tests_df_unique = is_tests_df.drop_duplicates(subset=["alpha_id", "name"], keep="last")
                    is_tests_df = is_tests_df_unique.pivot(
                        index="alpha_id", columns="name", values="result"
                    ).reset_index()
                except Exception as e_pivot_again:
                    print(f"警告 (_prettify_result): 去重后 pivot is_tests_df 仍然失败: {e_pivot_again}。is_tests_df 将不被合并。")
                    is_tests_df = pd.DataFrame() # 重置为empty，避免合并错误数据
        else:
            if not is_tests_df.empty: # 如果非空，但缺少列
                 print("警告 (_prettify_result): is_tests_df 缺少必要的列 (alpha_id, name, result) 进行 pivot。")
            is_tests_df = pd.DataFrame() # 确保为空，以便后续合并安全

        # 添加中文注释：合并统计、表达式和测试结果
        # 确保 alpha_id 列存在于要合并的DataFrame中
        alpha_stats = pd.DataFrame()

        if not is_stats_df.empty and "alpha_id" in is_stats_df.columns:
            alpha_stats = is_stats_df
            if not expression_df.empty and "alpha_id" in expression_df.columns:
                try:
                    alpha_stats = pd.merge(alpha_stats, expression_df, on="alpha_id", how="left")
                except Exception as e:
                    print(f"警告 (_prettify_result): 合并 is_stats_df 和 expression_df 时出错: {e}")
            if not is_tests_df.empty and "alpha_id" in is_tests_df.columns:
                try:
                    alpha_stats = pd.merge(alpha_stats, is_tests_df, on="alpha_id", how="left")
                except Exception as e:
                    print(f"警告 (_prettify_result): 合并 alpha_stats 和 is_tests_df 时出错: {e}")
        elif not expression_df.empty and "alpha_id" in expression_df.columns: # 如果 is_stats_df 为空，但 expression_df 非空
            alpha_stats = expression_df
            if not is_tests_df.empty and "alpha_id" in is_tests_df.columns:
                try:
                    alpha_stats = pd.merge(alpha_stats, is_tests_df, on="alpha_id", how="left")
                except Exception as e:
                    print(f"警告 (_prettify_result): 合并 expression_df 和 is_tests_df 时出错: {e}")
        else: # 如果 is_stats_df 和 expression_df 都为空或无 alpha_id
            # print("警告 (_prettify_result): is_stats_df 和 expression_df 均为空或缺少 alpha_id，无法进行核心合并。")
            return pd.DataFrame() # 返回空DataFrame

        if alpha_stats.empty:
            # print("警告 (_prettify_result): 合并后 alpha_stats 为空。")
            return pd.DataFrame()

        # 添加中文注释：删除包含“PENDING”值的列（表示测试还在进行中或失败）
        # 检查列是否存在，避免KeyError
        cols_to_drop = [col for col in alpha_stats.columns if (alpha_stats[col].astype(str) == "PENDING").any()]
        if cols_to_drop:
            alpha_stats = alpha_stats.drop(columns=cols_to_drop)

        # 添加中文注释：将列名转换为小写并用下划线分隔
        alpha_stats.columns = alpha_stats.columns.str.replace(
            r"(?<=[a-z])(?=[A-Z])", "_", regex=True # 正则表达式前加r
        ).str.lower()

        # 移除原始代码中的 style.format，直接返回 DataFrame
        # --- End of migrated code ---
        return alpha_stats

    def _fitness_fun(self, Data: pd.DataFrame, n: int) -> List[str]:
        """
        根据 Alpha 模拟结果计算适应度得分，并返回前 N 个最佳 Alpha 的表达式。
        适应度函数: (sharpe * fitness * returns) / ((drawdown * turnover^2) + 0.001)
        此逻辑迁移自 code.py 中的 fitness_fun。

        Args:
            Data (pd.DataFrame): 包含 Alpha 模拟结果的 DataFrame (通常是 _prettify_result 的输出)。
            n (int): 要选择的最佳 Alpha 数量。

        Returns:
            List[str]: 包含前 N 个最佳 Alpha 表达式的列表。
        """
        # --- Start of code migrated from fitness_fun ---
        # 添加中文注释：确保适应度计算所需的列都存在于DataFrame中
        required_cols = ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover', 'expression']
        missing_cols = [col for col in required_cols if col not in Data.columns]
        if missing_cols:
            print(f"错误 (_fitness_fun): DataFrame中缺失以下必要列: {', '.join(missing_cols)}，无法计算适应度。")
            return []

        # 添加中文注释：复制DataFrame以避免修改原始数据
        Data_filtered = Data.copy()

        # 添加中文注释：将可能影响计算的列转换为数值类型，错误值转为NaN，然后填充0
        cols_to_numeric = ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover']
        for col in cols_to_numeric:
            Data_filtered[col] = pd.to_numeric(Data_filtered[col], errors='coerce').fillna(0)

        # 添加中文注释：处理可能导致除零或无效计算的值
        # turnover 为零时替换为一个极小值以防除零
        Data_filtered['turnover'] = Data_filtered['turnover'].replace(0, 0.0001)
        # drawdown 通常是负值或表示亏损百分比的正值。适应度公式中期望其为正。
        # 取绝对值，如果为0，则替换为一个极小正值。
        Data_filtered['drawdown'] = Data_filtered['drawdown'].abs().replace(0, 0.0001)

        # 添加中文注释：计算适应度列 (fitness_score)
        # 分子：sharpe * fitness * returns
        numerator = Data_filtered['sharpe'] * Data_filtered['fitness'] * Data_filtered['returns']
        # 分母：(drawdown * turnover^2) + epsilon
        denominator = (Data_filtered['drawdown'] * (Data_filtered['turnover']**2)) + 0.001 # epsilon 避免除零

        Data_filtered['fitness_score'] = numerator / denominator

        # 添加中文注释：处理计算结果中可能出现的无穷大或NaN值
        Data_filtered = Data_filtered.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=['fitness_score'])

        # 添加中文注释：按新计算的 fitness_score 列降序排序
        Data_filtered = Data_filtered.sort_values(by='fitness_score', ascending=False)

        # 添加中文注释：获取前 N 个最佳 Alpha 的表达式
        top_n_values = Data_filtered.head(n)['expression'].tolist()
        # --- End of migrated code ---
        return top_n_values
