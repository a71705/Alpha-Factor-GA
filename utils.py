import os
import json
import pandas as pd
from typing import Dict, Any
import config # 导入配置模块

# 确保在函数外部定义路径，或者在函数内部从 config 模块动态获取
# 这样可以避免在函数默认参数中捕获旧的或不正确的路径值

def make_clickable_alpha_id(alpha_id: str) -> str:
    """
    在数据框中将 alpha_id 变为可点击链接，
    以便用户可以直接跳转到平台查看模拟结果。

    参数:
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        str: 包含可点击链接的 HTML 字符串。
    """
    # 从 config 模块获取平台 URL
    url = config.PLATFORM_ALPHA_URL
    return f'<a href="{url}{alpha_id}">{alpha_id}</a>'


def save_simulation_result(result: Dict[str, Any]) -> None:
    """
    将单个 Alpha 的完整模拟结果转储到 JSON 文件。

    文件将保存在 `config.SIMULATION_RESULTS_PATH` 定义的目录中。
    文件名格式为: {alpha_id}_{region}.json。

    参数:
        result (Dict[str, Any]): 单个 Alpha 的完整模拟结果字典。
                                 期望包含 'id' (Alpha ID) 和 'settings'>'region' 键。
    """
    alpha_id = result.get("id")
    if not alpha_id:
        print("错误: 模拟结果字典中缺少 'id' 键，无法保存。")
        return

    settings = result.get("settings", {})
    region = settings.get("region")
    if not region:
        print(f"警告: Alpha ID {alpha_id} 的模拟结果中缺少区域信息，文件名中将不包含区域。")
        file_name = f"{alpha_id}.json"
    else:
        file_name = f"{alpha_id}_{region}.json"

    if not isinstance(result, dict):
        logger.error(f"save_simulation_result错误: 输入的 'result' 不是字典类型 (类型: {type(result)})。Alpha ID: {result.get('id', 'N/A') if isinstance(result, dict) else 'N/A'}")
        # Optionally raise TypeError or FileOperationError here
        return

    # 从 config 模块获取保存路径
    folder_path = config.SIMULATION_RESULTS_PATH
    file_path = os.path.join(folder_path, file_name)

    try:
        # 如果文件夹不存在则创建
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        logger.error(f"创建目录 {folder_path} 失败: {e}", exc_info=True)
        # raise FileOperationError(f"创建目录失败: {folder_path}", filepath=folder_path, original_exception=e) from e
        # For now, just log and attempt to write, assuming path might exist or be creatable by open() in some cases.
        # A stricter approach would be to re-raise here.
        pass # Let the open() call below handle the error if directory still doesn't exist or isn't writable.


    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        logger.info(f"模拟结果已保存到: {file_path}")
    except (IOError, OSError) as e: # Catch more general OS errors too
        logger.error(f"保存模拟结果到文件 {file_path} 失败: {e}", exc_info=True)
        # raise FileOperationError(f"保存文件失败: {file_path}", filepath=file_path, original_exception=e) from e
    except TypeError as e: # json.dump might raise this for non-serializable content
        logger.error(f"序列化模拟结果 (Alpha ID: {alpha_id}) 时发生类型错误: {e}", exc_info=True)
        # raise FileOperationError(f"序列化JSON失败 for {file_path}", filepath=file_path, original_exception=e) from e


def save_pnl(pnl_df: pd.DataFrame, alpha_id: str, region: str) -> None:
    """
    将 PnL (盈亏) 数据转储到 CSV 文件。

    文件将保存在 `config.PNL_DATA_PATH` 定义的目录中。
    文件名格式为: {alpha_id}_{region}.csv。

    参数:
        pnl_df (pd.DataFrame): 包含 PnL 数据的 DataFrame。
        alpha_id (str): Alpha 的唯一标识符。
        region (str): Alpha 所在的区域。
    """
    if pnl_df is None or pnl_df.empty:
        # print(f"信息: PnL 数据为空 (Alpha ID: {alpha_id}, Region: {region})，不保存文件。") # 可以取消注释以获取更详细的日志
        return

    if not isinstance(pnl_df, pd.DataFrame):
        logger.error(f"save_pnl错误: 输入的 'pnl_df' 不是Pandas DataFrame (类型: {type(pnl_df)})。Alpha ID: {alpha_id}, Region: {region}")
        # Optionally raise TypeError or FileOperationError here
        return

    if pnl_df is None or pnl_df.empty: # Original check was just for None or empty, keeping it.
        logger.info(f"PnL 数据为空 (Alpha ID: {alpha_id}, Region: {region})，不保存文件。")
        return

    # 从 config 模块获取保存路径
    folder_path = config.PNL_DATA_PATH
    file_name = f"{alpha_id}_{region}.csv"
    file_path = os.path.join(folder_path, file_name)

    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        logger.error(f"创建目录 {folder_path} 失败: {e}", exc_info=True)
        # For now, log and attempt write. Stricter: raise FileOperationError
        pass

    try:
        pnl_df.to_csv(file_path, index=False)
        logger.info(f"PnL 数据已保存到: {file_path}")
    except (IOError, OSError) as e:
        logger.error(f"保存 PnL 数据到文件 {file_path} 失败: {e}", exc_info=True)
        # raise FileOperationError(f"保存PnL CSV失败: {file_path}", filepath=file_path, original_exception=e) from e
    except Exception as e: # Pandas to_csv can raise other errors (e.g. on data types if not handled)
        logger.error(f"将PnL DataFrame导出到CSV ({file_path})时发生未知错误: {e}", exc_info=True)
        # raise FileOperationError(f"导出PnL到CSV失败: {file_path}", filepath=file_path, original_exception=e) from e


def save_yearly_stats(yearly_stats_df: pd.DataFrame, alpha_id: str, region: str) -> None:
    """
    将年度统计数据转储到 CSV 文件。

    文件将保存在 `config.YEARLY_STATS_PATH` 定义的目录中。
    文件名格式为: {alpha_id}_{region}.csv。

    参数:
        yearly_stats_df (pd.DataFrame): 包含年度统计数据的 DataFrame。
        alpha_id (str): Alpha 的唯一标识符。
        region (str): Alpha 所在的区域。
    """
    if yearly_stats_df is None or yearly_stats_df.empty:
        # print(f"信息: 年度统计数据为空 (Alpha ID: {alpha_id}, Region: {region})，不保存文件。") # 可以取消注释以获取更详细的日志
        return

    if not isinstance(yearly_stats_df, pd.DataFrame):
        logger.error(f"save_yearly_stats错误: 输入的 'yearly_stats_df' 不是Pandas DataFrame (类型: {type(yearly_stats_df)})。Alpha ID: {alpha_id}, Region: {region}")
        return

    if yearly_stats_df is None or yearly_stats_df.empty: # Original check
        logger.info(f"年度统计数据为空 (Alpha ID: {alpha_id}, Region: {region})，不保存文件。")
        return

    # 从 config 模块获取保存路径
    folder_path = config.YEARLY_STATS_PATH
    file_name = f"{alpha_id}_{region}.csv"
    file_path = os.path.join(folder_path, file_name)

    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        logger.error(f"创建目录 {folder_path} 失败: {e}", exc_info=True)
        pass # Attempt write

    try:
        yearly_stats_df.to_csv(file_path, index=False)
        logger.info(f"年度统计数据已保存到: {file_path}")
    except (IOError, OSError) as e:
        logger.error(f"保存年度统计数据到文件 {file_path} 失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"将年度统计DataFrame导出到CSV ({file_path})时发生未知错误: {e}", exc_info=True)

import pandas as pd # Ensure pandas is imported for DataFrame operations
import logging # For logging
from typing import List, Dict, Any, Union # Added for type hints

# 获取模块级别的 logger
logger = logging.getLogger(__name__)
# TODO: 主程序中统一配置 logger handler 和 level


# --- 结果数据处理与聚合函数 ---

def prettify_result(
    simulation_outputs: List[Dict[str, Any]], # 期望列表每个元素是 get_specified_alpha_stats 的输出
    detailed_tests_view: bool = False,
    clickable_alpha_id_links: bool = False # Renamed for clarity
) -> pd.DataFrame:
    """
    将多个 Alpha 的模拟结果（通常来自 get_specified_alpha_stats 的输出列表）
    整合到一个美化后的 DataFrame 中，以便于分析。
    结果按 'fitness' (来自 is_stats) 降序排列。

    参数:
        simulation_outputs (List[Dict[str, Any]]): 包含每个 Alpha 模拟结果的字典列表。
            每个字典应至少包含:
            - 'alpha_id' (str)
            - 'simulate_data' (Dict): 包含 'regular' (Alpha表达式)
            - 'is_stats' (pd.DataFrame): 来自 IS 统计，包含 'fitness' 等。
            - 'is_tests' (pd.DataFrame): 来自 IS 测试结果。
        detailed_tests_view (bool): 如果为 True，将显示详细的测试结果视图（包含limit, result, value）。
                                    默认为 False，只显示测试的 'result'。
        clickable_alpha_id_links (bool): 如果为 True，'alpha_id' 列将格式化为可点击的HTML链接。
                                         默认为 False。

    返回:
        pd.DataFrame: 包含 Alpha 统计信息、表达式和测试结果的整合 DataFrame。
                      如果输入为空或处理失败，返回空 DataFrame。
    """
    logger.info(f"开始美化处理 {len(simulation_outputs)} 个Alpha的模拟结果...")

    if not simulation_outputs:
        logger.warning("输入的美化结果列表为空。")
        return pd.DataFrame()

    # 提取并合并所有 Alpha 的 IS (In-Sample) 统计数据
    list_of_is_stats_dfs = []
    for x_output in simulation_outputs:
        if x_output and isinstance(x_output.get('is_stats'), pd.DataFrame) and not x_output['is_stats'].empty:
            list_of_is_stats_dfs.append(x_output['is_stats'])

    if not list_of_is_stats_dfs:
        logger.warning("所有输入结果中均未找到有效的 'is_stats' DataFrame。")
        is_stats_df_combined = pd.DataFrame()
    else:
        is_stats_df_combined = pd.concat(list_of_is_stats_dfs, ignore_index=True)
        # 按 fitness 降序排序 (确保 fitness 列存在且为数值)
        if 'fitness' in is_stats_df_combined.columns:
            is_stats_df_combined['fitness'] = pd.to_numeric(is_stats_df_combined['fitness'], errors='coerce')
            is_stats_df_combined = is_stats_df_combined.sort_values("fitness", ascending=False)
        # else: # No sorting if 'fitness' column is missing from is_stats_df_combined
            # logger.warning("'fitness' 列不存在于合并后的 is_stats_df 中，排序被跳过。")

        # 尝试向下转换数值类型以减少内存
        for col in is_stats_df_combined.select_dtypes(include=['number']).columns:
            is_stats_df_combined[col] = pd.to_numeric(is_stats_df_combined[col], downcast='float')

    # 提取所有 Alpha 的表达式
    expressions_map = {}
    for x_output in simulation_outputs:
        if x_output and x_output.get('alpha_id') and isinstance(x_output.get('simulate_data'), dict):
            # 表达式通常在 simulate_data['regular'] 或 simulate_data['regular']['code']
            sim_data = x_output['simulate_data']
            expr = sim_data.get('regular')
            if isinstance(expr, dict): # 例如 {'code': 'alpha_expr_str'}
                expr = expr.get('code')
            if isinstance(expr, str):
                 expressions_map[x_output['alpha_id']] = expr
            else:
                logger.debug(f"Alpha ID {x_output['alpha_id']} 未找到有效的表达式字符串。")

    expression_df = pd.DataFrame(list(expressions_map.items()), columns=["alpha_id", "expression"])
    if 'alpha_id' in expression_df.columns: # Ensure alpha_id is present before trying to convert
        expression_df['alpha_id'] = expression_df['alpha_id'].astype('category') # alpha_id can be category if many merges happen on it

    # 提取并合并所有 Alpha 的 IS 测试结果
    list_of_is_tests_dfs = []
    for x_output in simulation_outputs:
        if x_output and isinstance(x_output.get('is_tests'), pd.DataFrame) and not x_output['is_tests'].empty:
            list_of_is_tests_dfs.append(x_output['is_tests'])

    if not list_of_is_tests_dfs:
        logger.warning("所有输入结果中均未找到有效的 'is_tests' DataFrame。")
        # 如果没有测试数据，后续合并会产生问题，可以返回部分结果或空DF
        # 为保持原逻辑的鲁棒性，允许只包含统计和表达式
        alpha_stats_merged = pd.merge(is_stats_df_combined, expression_df, on="alpha_id", how="left")
    else:
        is_tests_df_combined = pd.concat(list_of_is_tests_dfs).reset_index(drop=True)

        # 根据 detailed_tests_view 重塑测试结果 DataFrame
        # 确保 'alpha_id' 和 'name' 列存在以进行 pivot
        if 'alpha_id' not in is_tests_df_combined.columns or 'name' not in is_tests_df_combined.columns:
            logger.error("'is_tests' DataFrame 中缺少 'alpha_id' 或 'name' 列，无法进行 pivot 操作。")
            # 返回已合并的统计和表达式数据
            alpha_stats_merged = pd.merge(is_stats_df_combined, expression_df, on="alpha_id", how="left")
        else:
            try:
                if detailed_tests_view:
                    # 确保 'limit', 'result', 'value' 列存在
                    pivot_cols = ["limit", "result", "value"]
                    if not all(col in is_tests_df_combined.columns for col in pivot_cols):
                        logger.warning(f"详细测试视图所需的列 {pivot_cols} 不完全存在于 'is_tests' DataFrame。")
                        # 可以选择回退到非详细视图或只使用存在的列
                        # 为了简单，如果列不全，则不创建 'details' 列，可能导致pivot失败或产生NaN
                        is_tests_df_pivoted = is_tests_df_combined.pivot(index="alpha_id", columns="name", values="result").reset_index()

                    else:
                        is_tests_df_combined["details"] = is_tests_df_combined[pivot_cols].to_dict(orient="records")
                        is_tests_df_pivoted = is_tests_df_combined.pivot(
                            index="alpha_id", columns="name", values="details"
                        ).reset_index()
                else:
                    if 'result' not in is_tests_df_combined.columns:
                        logger.error("'result' 列不存在于 'is_tests' DataFrame，无法进行非详细视图的 pivot 操作。")
                        is_tests_df_pivoted = pd.DataFrame(columns=['alpha_id']) # 空DF，避免后续合并失败
                    else:
                        is_tests_df_pivoted = is_tests_df_combined.pivot(
                            index="alpha_id", columns="name", values="result"
                        ).reset_index()
            except Exception as e:
                logger.error(f"Pivot 操作 'is_tests' DataFrame 时出错: {e}", exc_info=True)
                unique_alpha_ids = is_tests_df_combined['alpha_id'].unique() if 'alpha_id' in is_tests_df_combined.columns else []
                is_tests_df_pivoted = pd.DataFrame({'alpha_id': unique_alpha_ids})

            # 尝试向下转换 is_tests_df_pivoted 中的数值类型
            for col in is_tests_df_pivoted.select_dtypes(include=['number']).columns:
                is_tests_df_pivoted[col] = pd.to_numeric(is_tests_df_pivoted[col], downcast='float')
            if 'alpha_id' in is_tests_df_pivoted.columns:
                 is_tests_df_pivoted['alpha_id'] = is_tests_df_pivoted['alpha_id'].astype('category')


            # 合并所有数据: expression_df, is_stats_df_combined, is_tests_df_pivoted
            # Start with expression_df as it's most likely to contain all alpha_ids derived from simulate_data
            if expression_df.empty and is_stats_df_combined.empty and is_tests_df_pivoted.empty:
                 alpha_stats_merged = pd.DataFrame() # All inputs are empty
            elif not expression_df.empty:
                alpha_stats_merged = expression_df
                if not is_stats_df_combined.empty:
                    alpha_stats_merged = pd.merge(alpha_stats_merged, is_stats_df_combined, on="alpha_id", how="left")
                if not is_tests_df_pivoted.empty:
                    alpha_stats_merged = pd.merge(alpha_stats_merged, is_tests_df_pivoted, on="alpha_id", how="left")
            elif not is_stats_df_combined.empty: # expression_df is empty, but is_stats_df_combined is not
                alpha_stats_merged = is_stats_df_combined # This will be base
                # expression_df is empty, so no expression column unless it's in is_stats_df_combined
                if not is_tests_df_pivoted.empty:
                    alpha_stats_merged = pd.merge(alpha_stats_merged, is_tests_df_pivoted, on="alpha_id", how="left")
            else: # expression_df and is_stats_df_combined are empty
                alpha_stats_merged = is_tests_df_pivoted # Only tests data available, or empty if all are empty

    if alpha_stats_merged.empty:
        logger.warning("最终合并的 Alpha 统计 DataFrame 为空。")
        return pd.DataFrame()

    # 在所有数据合并后，且在列名清理前，进行排序 (如果 fitness 列存在)
    if 'fitness' in alpha_stats_merged.columns:
        # Ensure fitness is numeric before sorting, in case it was re-introduced as non-numeric by a merge
        alpha_stats_merged['fitness'] = pd.to_numeric(alpha_stats_merged['fitness'], errors='coerce')
        # Handle NaNs in fitness for sorting, e.g., put them last or first based on preference
        alpha_stats_merged = alpha_stats_merged.sort_values("fitness", ascending=False, na_position='last')
        logger.debug("alpha_stats_merged 已按 'fitness' 列排序。")
    else:
        logger.warning("排序跳过：'fitness' 列在最终合并的 DataFrame 中不存在。")


    # 删除包含“PENDING”值的列 (原代码逻辑) - 这可能过于激进
    # 更安全的做法是替换 PENDING 或保留它们，取决于分析需求
    # for col in alpha_stats_merged.columns:
    #     if (alpha_stats_merged[col].astype(str) == "PENDING").any():
    #         alpha_stats_merged = alpha_stats_merged.drop(columns=col)
    #         logger.debug(f"已删除包含 'PENDING' 状态的列: {col}")

    # 将列名转换为小写并用下划线分隔 (snake_case) - 使用原 code.py 中的稳定版本
    try:
        alpha_stats_merged.columns = alpha_stats_merged.columns.str.replace(
            r"(?<=[a-z])(?=[A-Z])", "_", regex=True
        ).str.lower().str.replace(r"[^a-z0-9_]+", "_", regex=True).str.replace(r"_{2,}", "_", regex=True).str.strip("_")
    except Exception as e:
        logger.error(f"转换列名为 snake_case 时出错: {e}", exc_info=True)


    if clickable_alpha_id_links and 'alpha_id' in alpha_stats_merged.columns:
        # 返回 Styler 对象以便 HTML 渲染，而不是直接修改 DataFrame
        logger.info("美化结果完成，应用可点击 Alpha ID 链接。")
        return alpha_stats_merged.style.format({"alpha_id": make_clickable_alpha_id})

    logger.info(f"美化结果完成，生成 DataFrame，行数: {len(alpha_stats_merged)}。")
    return alpha_stats_merged


def concat_pnl(simulation_outputs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将多个 Alpha 的 PnL (盈亏) 数据合并到一个 DataFrame 中。

    参数:
        simulation_outputs (List[Dict[str, Any]]): 包含每个 Alpha 模拟结果的字典列表。
            每个字典期望包含 'pnl' (pd.DataFrame) 和 'alpha_id' (str) 键。

    返回:
        pd.DataFrame: 包含所有 Alpha PnL 数据的合并 DataFrame。
                      列通常包括 'Date', 'Pnl', 'alpha_id'。
                      如果无有效 PnL 数据，返回空 DataFrame。
    """
    list_of_pnls_dfs = []
    for x_output in simulation_outputs:
        if x_output and isinstance(x_output.get('pnl'), pd.DataFrame) and not x_output['pnl'].empty:
            # 确保 'alpha_id' 列存在于 PnL DataFrame 中，如果不存在则添加
            pnl_df = x_output['pnl']
            if 'alpha_id' not in pnl_df.columns and 'alpha_id' in x_output:
                pnl_df = pnl_df.assign(alpha_id=x_output['alpha_id'])
            list_of_pnls_dfs.append(pnl_df)

    if not list_of_pnls_dfs:
        logger.info("未找到可合并的 PnL 数据。")
        return pd.DataFrame()

    try:
        # concat 可能会因索引或列名不一致而出问题，需确保 PnL DataFrame 结构一致
        # 假设 PnL DataFrame 已经有 'Date' 索引和 'alpha_id' 列
        # 如果 'Date' 不是索引，而是列，则 reset_index 可能不需要或需要调整
        # 原代码中 pnls_df = pd.concat(list_of_pnls).reset_index()
        # 假设 get_alpha_pnl 返回的 DataFrame 是 Date 作为索引

        # 检查Date是否为索引
        # if all(df.index.name == 'Date' for df in list_of_pnls_dfs):
        #     pnls_df_combined = pd.concat(list_of_pnls_dfs) # Date索引会自动对齐
        # else: # 如果Date是列
        #     pnls_df_combined = pd.concat(list_of_pnls_dfs, ignore_index=True)

        # 简单起见，先尝试直接 concat，如果 Date 是索引，它会尝试对齐
        # 如果 Date 是列，而我们想保留原始索引，则 ignore_index=True
        # 原代码是 reset_index() 在 concat 之后，这意味着原索引（如有）被丢弃并创建新RangeIndex
        pnls_df_combined = pd.concat(list_of_pnls_dfs)
        if pnls_df_combined.index.name == 'Date' or 'Date' in pnls_df_combined.columns:
             # 如果Date是索引，reset_index会将其变为列。如果Date已是列，reset_index只重置外部索引。
            pnls_df_combined = pnls_df_combined.reset_index()
        else: # 如果没有Date索引或列，则简单重置当前索引
            pnls_df_combined = pnls_df_combined.reset_index(drop=True)


        logger.info(f"成功合并 {len(list_of_pnls_dfs)} 个 Alpha 的 PnL 数据。")
        return pnls_df_combined
    except Exception as e:
        logger.error(f"合并 PnL 数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def concat_is_tests(simulation_outputs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将多个 Alpha 的 IS (In-Sample) 测试结果合并到一个 DataFrame 中。

    参数:
        simulation_outputs (List[Dict[str, Any]]): 包含每个 Alpha 模拟结果的字典列表。
            每个字典期望包含 'is_tests' (pd.DataFrame) 和 'alpha_id' (str) 键。

    返回:
        pd.DataFrame: 包含所有 Alpha IS 测试结果的合并 DataFrame。
                      如果无有效测试数据，返回空 DataFrame。
    """
    list_of_is_tests_dfs = []
    for x_output in simulation_outputs:
        if x_output and isinstance(x_output.get('is_tests'), pd.DataFrame) and not x_output['is_tests'].empty:
            # 确保 'alpha_id' 列存在
            is_tests_df = x_output['is_tests']
            if 'alpha_id' not in is_tests_df.columns and 'alpha_id' in x_output:
                 is_tests_df = is_tests_df.assign(alpha_id=x_output['alpha_id'])
            list_of_is_tests_dfs.append(is_tests_df)

    if not list_of_is_tests_dfs:
        logger.info("未找到可合并的 IS 测试数据。")
        return pd.DataFrame()

    try:
        # 原代码是 reset_index(drop=True)，表示不保留原始索引
        is_tests_df_combined = pd.concat(list_of_is_tests_dfs, ignore_index=True)
        logger.info(f"成功合并 {len(list_of_is_tests_dfs)} 个 Alpha 的 IS 测试数据。")
        return is_tests_df_combined
    except Exception as e:
        logger.error(f"合并 IS 测试数据时出错: {e}", exc_info=True)
        return pd.DataFrame()

# TODO: 从 code.py 迁移其他可以作为通用工具的函数。
# 例如: (部分已移至 api_client 或 simulation_manager)
# - get_alpha_pnl (已在 api_client.py)
# - get_alpha_yearly_stats (已在 api_client.py)
# - get_datasets (已在 api_client.py)
# - get_datafields (已在 api_client.py)

"""
此模块包含项目中使用的各种辅助函数。
这些函数通常是通用性的，可以在项目的不同部分被重用。
例如，数据格式化、文件保存、结果美化与聚合等。
"""
pass
