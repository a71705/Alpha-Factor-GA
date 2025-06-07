from typing import List, Dict, Optional, Any

# 导入配置模块
import config

# 导入API客户端模块 (用于执行实际的API调用)
# import api_client

# 导入工具模块 (例如用于保存结果)
# import utils

"""
此模块负责管理和协调 Alpha 的模拟、数据获取和结果处理流程。
它作为业务逻辑层，调用 api_client.py 中的底层 API 函数，
并可能使用 utils.py 中的辅助函数来处理数据。
"""

def generate_alpha_simulation_data(
    alpha_expression: str,
    region: str = config.DEFAULT_REGION,
    universe: str = config.DEFAULT_UNIVERSE,
    neutralization: str = config.DEFAULT_NEUTRALIZATION,
    delay: int = config.DEFAULT_DELAY,
    decay: int = config.DEFAULT_DECAY,
    truncation: float = config.DEFAULT_TRUNCATION,
    nan_handling: str = config.DEFAULT_NAN_HANDLING,
    unit_handling: str = config.DEFAULT_UNIT_HANDLING,
    pasteurization: str = config.DEFAULT_PASTEURIZATION,
    visualization: bool = False, # 可视化通常根据单次需求开启，不设全局默认
    alpha_type: str = config.DEFAULT_ALPHA_TYPE,
    instrument_type: str = config.DEFAULT_INSTRUMENT_TYPE,
    language: str = config.DEFAULT_ALPHA_LANGUAGE
) -> Dict[str, Any]:
    """
    生成用于 Alpha 模拟请求的标准化数据字典。

    此函数整合所有必要的参数和配置，构建一个符合 BRAIN API 模拟接口要求的字典。
    默认参数值优先从 `config.py` 模块中获取。

    参数:
        alpha_expression (str): Alpha 的核心表达式字符串 (例如 "close - open")。
        region (str): 模拟区域 (例如 "USA", "CHN", "EUR")。
        universe (str): 股票池 (例如 "TOP3000", "ALLSHARES")。
        neutralization (str): 中性化类型 (例如 "INDUSTRY", "MARKET", "SUBINDUSTRY")。
        delay (int): 数据延迟天数 (通常为 1)。
        decay (int): Alpha 衰减值 (通常用于调整 Alpha 的半衰期)。
        truncation (float): 极值处理的截断阈值 (例如 0.01 表示上下1%截断)。
        nan_handling (str): NaN (缺失值) 处理方式 (例如 "OFF", "FILL")。
        unit_handling (str): 单位处理方式 (例如 "VERIFY", "IGNORE", "RESCALE")。
        pasteurization (str): 巴氏消毒开关 ("ON" 或 "OFF")。
        visualization (bool): 是否为此次模拟启用可视化 (通常在需要分析时设为 True)。
        alpha_type (str): Alpha 类型，通常为 "REGULAR"。
        instrument_type (str): 资产类型，通常为 "EQUITY"。
        language (str): Alpha 表达式语言，通常为 "FASTEXPR"。

    返回:
        Dict[str, Any]: 包含 Alpha 模拟完整设置的字典，可直接用于 API 请求。
    """
    simulation_data = {
        "type": alpha_type,
        "settings": {
            "nanHandling": nan_handling,
            "instrumentType": instrument_type,
            "delay": delay,
            "universe": universe,
            "truncation": truncation,
            "unitHandling": unit_handling,
            "pasteurization": pasteurization,
            "region": region,
            "language": language,
            "decay": decay,
            "neutralization": neutralization,
            "visualization": visualization,
        },
        "regular": alpha_expression, # "regular" 键对应的是 Alpha 表达式本身
    }
    return simulation_data


# TODO: 从 api_client.py 导入需要的函数或整个模块
# from api_client import (
#     start_simulation, simulation_progress, multisimulation_progress,
#     get_simulation_result_json, simulate_single_alpha, simulate_multi_alpha,
#     get_alpha_pnl, get_alpha_yearly_stats, get_prod_corr, check_prod_corr_test,
#     get_self_corr, check_self_corr_test, get_check_submission, submit_alpha,
#     set_alpha_properties
)
import time # 用于处理 API 重试等待
import logging # 用于日志记录
from urllib.parse import urljoin # 用于构建 URL
import requests # requests.Session, requests.Response 类型提示

# 自定义模块导入
# Full import of api_client and utils for clarity, or import specific functions
import api_client
import utils
# from api_client import get_simulation_result_json, get_alpha_pnl, get_alpha_yearly_stats, get_check_submission, set_alpha_properties
# from utils import save_pnl, save_yearly_stats, save_simulation_result

import pandas as pd # 确保 pandas 导入，因为测试函数中使用了 MOCK 数据创建 DataFrame
import random # 确保 random 导入，因为测试函数中使用了 MOCK 数据
from typing import Union # For type hinting start_simulation

# 获取模块级别的 logger
logger = logging.getLogger(__name__)
# TODO: 主程序中统一配置 logger handler 和 level


# --- 模拟流程控制函数 ---

def start_simulation(
    s: requests.Session,
    simulation_data: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Optional[requests.Response]:
    """
    启动单个或多个 Alpha 的模拟。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulation_data (Union[Dict[str, Any], List[Dict[str, Any]]]):
            单个 Alpha 的模拟数据字典 (由 `generate_alpha_simulation_data` 生成)，
            或多个此类字典组成的列表 (用于批量模拟)。

    返回:
        Optional[requests.Response]: 来自 BRAIN API 的模拟启动响应对象。
                                     如果请求失败，则返回 None。包含 Location 头部用于跟踪进度。
    """
    simulations_url = urljoin(config.API_BASE_URL, "simulations")
    logger.info(f"向 {simulations_url} 发起模拟请求...")
    logger.debug(f"模拟请求数据: {simulation_data}")

    try:
        response = s.post(simulations_url, json=simulation_data)
        response.raise_for_status() # 对 4xx/5xx 错误抛出 HTTPError

        # API成功启动模拟后，状态码通常是 201 (Created) 或 200 (OK)
        # Location 头部包含了跟踪模拟进度的 URL
        if 'Location' not in response.headers:
            logger.error("模拟启动成功，但响应中缺少 'Location' 头部，无法跟踪进度。")
            # 根据API行为，这里可能仍视为部分成功，或者返回None表示无法继续
            return response # 或者返回 None，取决于是否认为这是致命错误

        logger.info(f"模拟请求已成功发送，状态码: {response.status_code}。进度跟踪URL: {response.headers.get('Location')}")
        return response

    except requests.exceptions.HTTPError as e:
        logger.error(f"启动模拟 API 请求失败 (HTTPError): {e.response.status_code} - {e.response.text}", exc_info=True)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"启动模拟 API 请求失败 (RequestException): {e}", exc_info=True)
        return None


def _handle_simulation_progress(
    s: requests.Session,
    progress_url: str,
    is_batch_simulation: bool
) -> Dict[str, Any]:
    """
    (内部辅助函数) 循环跟踪单个或批量模拟的进度，直到完成或出错。

    参数:
        s (requests.Session): 已认证的会话对象。
        progress_url (str): 用于查询模拟进度的 URL (来自 Location 头部)。
        is_batch_simulation (bool): 指示是否为批量模拟。批量模拟的完成条件和结果结构不同。

    返回:
        Dict[str, Any]: 包含模拟状态和结果的字典。
                        结构: {'completed': bool, 'result': Optional[Any], 'error': Optional[str]}
                        - 'completed': True 表示模拟成功完成。
                        - 'result': 如果成功，单个模拟是结果JSON，批量模拟是子模拟结果列表。
                        - 'error': 如果失败，包含错误信息字符串。
    """
    logger.info(f"开始跟踪模拟进度: {progress_url}")
    error_message: Optional[str] = None

    while True:
        try:
            progress_response = s.get(progress_url)
            progress_response.raise_for_status() # 检查HTTP错误

            # 检查是否需要等待后重试 (API 限流)
            if "retry-after" in progress_response.headers:
                wait_time = float(progress_response.headers["Retry-After"])
                logger.info(f"模拟进度API限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue # 继续下一次轮询

            progress_json = progress_response.json()
            status = progress_json.get("status", "UNKNOWN").upper()
            logger.debug(f"模拟状态: {status}, 响应: {progress_json}")

            if status == "ERROR":
                error_message = progress_json.get("message", "模拟发生未知错误。")
                logger.error(f"模拟失败: {error_message}")
                return {"completed": False, "result": None, "error": error_message}

            if status == "COMPLETED":
                logger.info("模拟已完成。")
                if is_batch_simulation:
                    # 对于批量模拟，结果在 'children' 字段中，每个child是一个simulation ID
                    children_ids = progress_json.get("children", [])
                    if not children_ids:
                        logger.warning("批量模拟已完成，但未找到子模拟ID。")
                        return {"completed": True, "result": [], "error": None}

                    children_results = []
                    logger.info(f"批量模拟完成，获取 {len(children_ids)} 个子模拟的结果...")
                    for child_id in children_ids:
                        # 获取每个子模拟的最终结果 (通常包含alpha ID)
                        # 这里假设子模拟ID可以直接用于获取其状态和alpha ID
                        child_sim_url = urljoin(config.API_BASE_URL, f"simulations/{child_id}")
                        child_sim_status_resp = s.get(child_sim_url)
                        child_sim_status_resp.raise_for_status()
                        child_sim_json = child_sim_status_resp.json()

                        if child_sim_json.get("status", "").upper() == "COMPLETED" and "alpha" in child_sim_json:
                            alpha_id = child_sim_json["alpha"]
                            # 获取该 alpha_id 的详细模拟结果
                            alpha_result_json = get_simulation_result_json(s, alpha_id)
                            if alpha_result_json:
                                children_results.append(alpha_result_json)
                            else:
                                logger.warning(f"未能获取子模拟 (ID: {child_id}, Alpha ID: {alpha_id}) 的详细结果。")
                                children_results.append({"id": alpha_id, "error": "Failed to retrieve full result"}) # 添加占位符
                        else:
                            logger.warning(f"子模拟 (ID: {child_id}) 未成功完成或缺少 Alpha ID。状态: {child_sim_json.get('status')}")
                            children_results.append({"id": child_id, "error": f"Sub-simulation not completed successfully. Status: {child_sim_json.get('status')}"})

                    return {"completed": True, "result": children_results, "error": None}
                else: # 单个模拟
                    alpha_id = progress_json.get("alpha")
                    if not alpha_id:
                        error_message = "模拟完成但未返回 Alpha ID。"
                        logger.error(error_message)
                        return {"completed": False, "result": None, "error": error_message}

                    # 获取完整的模拟结果JSON
                    simulation_result_data = get_simulation_result_json(s, alpha_id)
                    if simulation_result_data:
                        return {"completed": True, "result": simulation_result_data, "error": None}
                    else:
                        error_message = f"成功获取 Alpha ID {alpha_id}，但未能获取其详细模拟结果。"
                        logger.error(error_message)
                        return {"completed": False, "result": {"id": alpha_id}, "error": error_message}

            # 如果状态不是 COMPLETED 或 ERROR，且没有 retry-after，则稍等后继续轮询
            # (原代码中，若无 retry-after 头部，则认为已完成或出错并跳出循环。
            #  更稳健的做法可能是根据状态判断，例如 PENDING, RUNNING 等状态下继续轮询)
            if status in ["PENDING", "RUNNING", "QUEUED"]: # 假设这些是进行中的状态
                 logger.debug(f"模拟仍在进行中 (状态: {status})，等待后轮询...")
                 time.sleep(config.SIMULATION_POLL_INTERVAL) # 需要在config.py中定义一个轮询间隔
                 continue
            else: # 未知或非预期状态
                error_message = f"模拟进入未知或非预期状态: {status}。响应: {progress_json}"
                logger.error(error_message)
                return {"completed": False, "result": None, "error": error_message}

        except requests.exceptions.HTTPError as e:
            logger.error(f"跟踪模拟进度时发生 HTTP 错误: {e.response.status_code} - {e.response.text}", exc_info=True)
            return {"completed": False, "result": None, "error": str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"跟踪模拟进度时发生网络或请求错误: {e}", exc_info=True)
            # 在这种情况下，可能需要更复杂的重试逻辑或直接失败
            time.sleep(config.SIMULATION_POLL_INTERVAL_ON_ERROR) # 错误后等待更长时间
            # return {"completed": False, "result": None, "error": str(e)} # 或者尝试重试几次
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析模拟进度响应时出错: {e}", exc_info=True)
            return {"completed": False, "result": None, "error": "Failed to parse progress response"}
        except Exception as e: # 其他未知错误
            logger.error(f"跟踪模拟进度时发生未知错误: {e}", exc_info=True)
            return {"completed": False, "result": None, "error": str(e)}


def simulation_progress(s: requests.Session, simulate_response: requests.Response) -> Dict[str, Any]:
    """
    跟踪单个 Alpha 模拟的进度并获取最终结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_response (requests.Response): `start_simulation` 函数返回的响应对象，
                                             其头部应包含 'Location' URL 用于跟踪进度。

    返回:
        Dict[str, Any]: 包含模拟完成状态和结果的字典。
                        如果成功: {'completed': True, 'result': <simulation_json_data>}
                        如果失败: {'completed': False, 'result': None, 'error': <error_message>}
    """
    if not simulate_response or 'Location' not in simulate_response.headers:
        logger.error("无效的模拟响应或缺少 'Location' 头部，无法跟踪进度。")
        return {"completed": False, "result": None, "error": "Invalid simulation response for progress tracking."}

    progress_url = simulate_response.headers['Location']
    # 确保 progress_url 是完整的 URL
    if not progress_url.startswith(('http://', 'https://')):
        # 尝试从响应的 URL (原始请求的 URL) 构建完整 URL
        base_url_for_location = simulate_response.url
        progress_url = urljoin(base_url_for_location, progress_url)
        logger.debug(f"修正后的进度跟踪 URL: {progress_url}")

    return _handle_simulation_progress(s, progress_url, is_batch_simulation=False)


def multisimulation_progress(s: requests.Session, simulate_response: requests.Response) -> Dict[str, Any]:
    """
    跟踪多个 Alpha (批量) 模拟的进度并获取所有子模拟的结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_response (requests.Response): `start_simulation` 函数返回的响应对象 (用于批量模拟)。

    返回:
        Dict[str, Any]: 包含批量模拟完成状态和结果列表的字典。
                        如果成功: {'completed': True, 'result': List[<simulation_json_data>]}
                        如果失败: {'completed': False, 'result': None, 'error': <error_message>}
                        如果部分子模拟失败，result 列表可能包含错误信息或部分数据。
    """
    if not simulate_response or 'Location' not in simulate_response.headers:
        logger.error("无效的批量模拟响应或缺少 'Location' 头部，无法跟踪进度。")
        return {"completed": False, "result": None, "error": "Invalid batch simulation response for progress tracking."}

    progress_url = simulate_response.headers['Location']
    if not progress_url.startswith(('http://', 'https://')):
        base_url_for_location = simulate_response.url
        progress_url = urljoin(base_url_for_location, progress_url)
        logger.debug(f"修正后的批量模拟进度跟踪 URL: {progress_url}")

    return _handle_simulation_progress(s, progress_url, is_batch_simulation=True)


# --- 高级模拟业务逻辑与编排 ---

def simulate_single_alpha(
    s: requests.Session,
    simulation_data_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    模拟单个 Alpha，并设置其属性。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulation_data_dict (Dict[str, Any]): 单个 Alpha 的模拟数据字典 (由 `generate_alpha_simulation_data` 生成)。

    返回:
        Dict[str, Any]: 包含 Alpha ID 和原始模拟数据的字典。
                        如果模拟失败或未能获取 Alpha ID, 'alpha_id' 键可能为 None。
                        结构: {'alpha_id': Optional[str], 'simulate_data': Dict[str, Any]}
    """
    logger.info(f"开始单个 Alpha 模拟流程...")
    # TODO: 后续版本可以将 check_session_timeout 移到 api_client 或 main 调用前
    # if api_client.check_session_timeout(s) < 1000: # 假设 api_client 已导入
    #     logger.info("会话即将过期或已过期，尝试重新启动会话...")
    #     s_new = api_client.start_session()
    #     if not s_new:
    #         logger.error("无法重新启动会话，模拟中止。")
    #         return {'alpha_id': None, 'simulate_data': simulation_data_dict, 'error': "Session refresh failed"}
    #     s = s_new # 更新会话对象

    # 启动模拟
    sim_response = start_simulation(s, simulation_data_dict)
    if not sim_response:
        logger.error("单个 Alpha 模拟启动失败。")
        return {'alpha_id': None, 'simulate_data': simulation_data_dict, 'error': "Failed to start simulation"}

    # 跟踪模拟进度
    progress_result = simulation_progress(s, sim_response)
    if not progress_result.get("completed"):
        logger.error(f"单个 Alpha 模拟未成功完成。错误: {progress_result.get('error')}")
        return {'alpha_id': None, 'simulate_data': simulation_data_dict, 'error': progress_result.get('error', "Simulation did not complete")}

    simulation_details = progress_result.get("result", {})
    alpha_id = simulation_details.get("id")

    if not alpha_id:
        logger.error("模拟完成但未能从结果中提取 Alpha ID。")
        return {'alpha_id': None, 'simulate_data': simulation_data_dict, 'error': "Alpha ID missing in completed simulation result"}

    logger.info(f"单个 Alpha (ID: {alpha_id}) 模拟成功。")

    # 设置 Alpha 属性 (例如，默认标签)
    # TODO: 考虑 set_alpha_properties 是否应该在此处调用，或由调用方决定
    #       如果调用，确保 api_client 已导入或 set_alpha_properties 已移至本模块
    # success_props = api_client.set_alpha_properties(s, alpha_id) # 默认标签 "gen"
    # if not success_props:
    #     logger.warning(f"未能为 Alpha ID {alpha_id} 设置属性。")

    return {'alpha_id': alpha_id, 'simulate_data': simulation_data_dict}


def simulate_multi_alpha(
    s: requests.Session,
    simulation_data_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    模拟多个 Alpha (批量模拟)，并设置属性。
    如果列表只有一个 Alpha，则调用 simulate_single_alpha。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulation_data_list (List[Dict[str, Any]]): 包含多个 Alpha 模拟数据字典的列表。

    返回:
        List[Dict[str, Any]]: 包含每个 Alpha ID 和模拟数据的字典列表。
                              对于失败的模拟，'alpha_id' 可能为 None，并可能包含 'error' 键。
    """
    logger.info(f"开始批量 Alpha 模拟流程，共 {len(simulation_data_list)} 个 Alpha。")
    # TODO: Session check similar to simulate_single_alpha

    if not simulation_data_list:
        logger.warning("输入的模拟数据列表为空。")
        return []

    if len(simulation_data_list) == 1:
        logger.info("列表中只有一个 Alpha，转为单个 Alpha 模拟流程。")
        return [simulate_single_alpha(s, simulation_data_list[0])]

    # 启动批量模拟
    sim_response = start_simulation(s, simulation_data_list) # API支持直接传入列表
    if not sim_response:
        logger.error("批量 Alpha 模拟启动失败。")
        # 为每个输入 Alpha 返回失败状态
        return [{'alpha_id': None, 'simulate_data': data, 'error': "Failed to start batch simulation"} for data in simulation_data_list]

    # 跟踪批量模拟进度
    progress_result = multisimulation_progress(s, sim_response)
    if not progress_result.get("completed"):
        logger.error(f"批量 Alpha 模拟未成功完成。错误: {progress_result.get('error')}")
        return [{'alpha_id': None, 'simulate_data': data, 'error': progress_result.get('error', "Batch simulation did not complete")} for data in simulation_data_list]

    # 处理子模拟结果
    # progress_result['result'] 应该是一个包含每个子 Alpha 完整模拟结果 JSON 的列表
    children_simulation_details_list = progress_result.get("result", [])

    processed_results: List[Dict[str, Any]] = []
    original_expr_map: Dict[str, Dict[str, Any]] = {} # 用于通过表达式代码匹配原始输入数据

    # 为了能将结果与原始 simulate_data 对应起来，可能需要更复杂的匹配逻辑
    # 假设 API 返回的子模拟结果中包含可用于匹配原始请求的信息，例如表达式
    # 或者，假设返回顺序与请求顺序一致 (这通常不保证)
    # 原 code.py 的 simulate_multi_alpha 通过 regular code 进行匹配，这里尝试类似逻辑

    # 构建一个从表达式到原始数据的映射（如果模拟数据包含表达式）
    for original_data in simulation_data_list:
        if 'regular' in original_data and isinstance(original_data['regular'], str):
             original_expr_map[original_data['regular']] = original_data
        # 如果 simulate_data 中没有 'regular' 键或其不是字符串，则无法通过表达式匹配
        # 此时，依赖返回顺序可能是唯一的（但不保证可靠的）方法

    for child_sim_details in children_simulation_details_list:
        alpha_id = child_sim_details.get("id")
        if not alpha_id or child_sim_details.get("error"): # 如果子模拟本身就有错误
            logger.warning(f"子模拟 (ID: {alpha_id or '未知'}) 处理失败或已标记错误: {child_sim_details.get('error', '未知错误')}")
            # 尝试找到对应的原始数据，如果找不到，则无法返回
            # 这是一个简化的匹配，实际可能需要更鲁棒的机制
            found_original_data = None
            if alpha_id and 'regular' in child_sim_details and isinstance(child_sim_details['regular'], dict) and 'code' in child_sim_details['regular']:
                expr_code = child_sim_details['regular']['code']
                found_original_data = original_expr_map.get(expr_code)

            processed_results.append({
                'alpha_id': alpha_id,
                'simulate_data': found_original_data or {"expression_unknown": True}, # 标记数据源不确定
                'error': child_sim_details.get('error', 'Sub-simulation failed or Alpha ID missing')
            })
            continue

        logger.info(f"子模拟 (Alpha ID: {alpha_id}) 处理成功。")
        # api_client.set_alpha_properties(s, alpha_id) # TODO: 考虑是否调用，并确保导入

        # 查找对应的原始 simulate_data
        # 'regular' 字段在模拟结果中可能是一个字典 {'code': 'expression_str'}
        alpha_expression_code = None
        if 'regular' in child_sim_details and isinstance(child_sim_details['regular'], dict):
            alpha_expression_code = child_sim_details['regular'].get('code')

        original_input_data = original_expr_map.get(alpha_expression_code) if alpha_expression_code else None

        if original_input_data:
            processed_results.append({'alpha_id': alpha_id, 'simulate_data': original_input_data})
        else:
            # 如果无法通过表达式匹配，则记录一个问题，并可能基于顺序（如果适用）或仅ID返回
            logger.warning(f"无法通过表达式 '{alpha_expression_code}' 将 Alpha ID {alpha_id} 匹配到原始模拟数据。")
            # 返回包含完整结果，但标记 simulate_data 可能不准确
            processed_results.append({
                'alpha_id': alpha_id,
                'simulate_data': {'expression_code_from_api': alpha_expression_code}, # 表明这是从API结果中获取的
                'full_api_result': child_sim_details # 可以选择包含完整的API结果以供调试
            })

    # 如果 processed_results 的数量与 simulation_data_list 不匹配，可能需要填充错误信息
    if len(processed_results) != len(simulation_data_list):
        logger.warning(f"批量模拟后处理结果数量 ({len(processed_results)}) 与输入数量 ({len(simulation_data_list)}) 不匹配。")
        # 这是一个复杂的情况，可能需要更仔细地处理哪些Alpha失败了。
        # 目前的逻辑是基于成功从API获取的子模拟结果进行处理。

    return processed_results


def get_specified_alpha_stats(
    s: requests.Session,
    alpha_id: Optional[str],
    simulation_data_dict: Dict[str, Any], # 原始模拟数据，用于返回给调用者
    simulation_params: Dict[str, Any] # 控制获取哪些数据及是否保存，来自 config.DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    获取指定 Alpha 的统计信息、PnL、测试结果等，并根据配置保存。

    参数:
        s (requests.Session): 已认证的会话对象。
        alpha_id (Optional[str]): Alpha 的唯一标识符。如果为 None，表示此 Alpha 的模拟已失败。
        simulation_data_dict (Dict[str, Any]): 此 Alpha 的原始模拟数据字典。
        simulation_params (Dict[str, Any]): 一个字典，包含多个布尔标志，控制行为：
            - 'get_pnl': 是否获取 PnL 数据。
            - 'get_stats': 是否获取年度统计数据。
            - 'save_pnl_file': 是否将 PnL 数据保存到文件。
            - 'save_stats_file': 是否将年度统计数据保存到文件。
            - 'save_result_file': 是否将完整模拟结果 (JSON) 保存到文件。
            - 'check_submission': 是否执行提交检查。
            - 'check_self_corr': 是否执行自相关性检查。
            - 'check_prod_corr': 是否执行生产相关性检查。

    返回:
        Dict[str, Any]: 包含 Alpha ID、原始模拟数据、IS统计、PnL、年度统计和各项测试结果的字典。
                        如果 alpha_id 为 None，则许多字段也会是 None。
    """
    logger.debug(f"开始为 Alpha ID '{alpha_id or 'N/A'}' 获取指定统计数据。Params: {simulation_params}")

    # 初始化返回字典的结构
    result_package: Dict[str, Any] = {
        'alpha_id': alpha_id,
        'simulate_data': simulation_data_dict,
        'is_stats': None,
        'pnl': None,
        'stats': None, # 年度统计
        'is_tests': None, # IS 测试结果 (DataFrame)
        'error': None
    }

    if alpha_id is None:
        logger.warning("Alpha ID 为 None，无法获取统计数据。可能由于模拟创建失败。")
        result_package['error'] = "Alpha ID is None, simulation likely failed."
        return result_package

    # 获取完整的模拟结果 JSON，其中包含 IS 统计和基础测试
    full_sim_result = get_simulation_result_json(s, alpha_id)
    if not full_sim_result:
        logger.error(f"未能获取 Alpha ID {alpha_id} 的完整模拟结果 JSON。")
        result_package['error'] = "Failed to retrieve full simulation JSON."
        return result_package

    alpha_region = full_sim_result.get("settings", {}).get("region", config.DEFAULT_REGION)

    # 提取 IS 统计数据 (不包括 'checks' 部分)
    is_stats_data = full_sim_result.get('is', {})
    if is_stats_data:
        is_stats_df = pd.DataFrame([{k: v for k, v in is_stats_data.items() if k != 'checks'}]).assign(alpha_id=alpha_id)
        result_package['is_stats'] = is_stats_df
    else:
        logger.warning(f"Alpha ID {alpha_id}: 模拟结果中缺少 'is' 统计数据。")

    # 提取基础 IS 测试结果
    base_is_tests_list = is_stats_data.get('checks', [])
    if base_is_tests_list:
        current_is_tests_df = pd.DataFrame(base_is_tests_list).assign(alpha_id=alpha_id)
    else:
        logger.warning(f"Alpha ID {alpha_id}: 模拟结果中缺少 'is.checks' 测试数据。")
        current_is_tests_df = pd.DataFrame() # 初始化为空DF

    # 根据参数获取和保存 PnL
    if simulation_params.get('get_pnl', False):
        pnl_df = api_client.get_alpha_pnl(s, alpha_id) # 假设api_client已导入
        if pnl_df is not None and not pnl_df.empty:
            result_package['pnl'] = pnl_df
            if simulation_params.get('save_pnl_file', False):
                utils.save_pnl(pnl_df, alpha_id, alpha_region) # 假设utils已导入
        else:
            logger.info(f"Alpha ID {alpha_id}: 未获取到 PnL 数据。")

    # 根据参数获取和保存年度统计
    if simulation_params.get('get_stats', False):
        yearly_stats_df = api_client.get_alpha_yearly_stats(s, alpha_id)
        if yearly_stats_df is not None and not yearly_stats_df.empty:
            result_package['stats'] = yearly_stats_df
            if simulation_params.get('save_stats_file', False):
                utils.save_yearly_stats(yearly_stats_df, alpha_id, alpha_region)
        else:
            logger.info(f"Alpha ID {alpha_id}: 未获取到年度统计数据。")

    # 根据参数保存完整模拟结果 JSON
    if simulation_params.get('save_result_file', False):
        utils.save_simulation_result(full_sim_result) # full_sim_result 已包含id和settings.region

    # 执行额外的检查测试
    if simulation_params.get('check_submission', False):
        submission_checks_df = api_client.get_check_submission(s, alpha_id)
        if not submission_checks_df.empty:
            # get_check_submission 通常返回更全面的测试结果，可能覆盖或合并 base_is_tests
            current_is_tests_df = submission_checks_df
            # logger.debug(f"Alpha ID {alpha_id}: 已获取并使用提交检查结果 ({len(current_is_tests_df)} 项)。")
        else:
            logger.warning(f"Alpha ID {alpha_id}: 提交检查未返回结果。")

    # 如果不检查提交，但需要检查自相关或生产相关，则追加这些测试结果
    # （原代码逻辑是如果 check_submission 为 True，则不执行后续的 self/prod corr）
    # 这里调整为：如果 check_submission 获取了结果，则后续检查追加到这个结果上。
    # 如果 check_submission 未获取结果（或未执行），则追加到 base_is_tests 上。

    additional_tests_to_run = []
    if simulation_params.get('check_self_corr', False):
        additional_tests_to_run.append(check_self_corr_test) # 函数本身，非调用结果
    if simulation_params.get('check_prod_corr', False):
        additional_tests_to_run.append(check_prod_corr_test)

    for test_func in additional_tests_to_run:
        try:
            # 假设这些测试函数 (check_self_corr_test, check_prod_corr_test)
            # 已被迁移到本模块或导入，并返回 DataFrame
            test_result_df = test_func(s, alpha_id)
            if not test_result_df.empty:
                if not current_is_tests_df.empty:
                    current_is_tests_df = pd.concat([current_is_tests_df, test_result_df]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True)
                else:
                    current_is_tests_df = test_result_df
                # logger.debug(f"Alpha ID {alpha_id}: 已追加测试 '{test_func.__name__}' 的结果。")
            else:
                logger.info(f"Alpha ID {alpha_id}: 测试 '{test_func.__name__}' 未返回结果。")
        except Exception as e:
            logger.error(f"Alpha ID {alpha_id}: 执行测试 '{test_func.__name__}' 时出错: {e}", exc_info=True)

    result_package['is_tests'] = current_is_tests_df
    logger.info(f"完成 Alpha ID {alpha_id} 的指定统计数据获取和处理。")
    return result_package


# 导入并发处理库
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool # 原代码使用了这个，但 ThreadPoolExecutor 更现代
from functools import partial
import tqdm # 用于显示进度条

# TODO: 确认 utils 和 api_client 模块已导入，并且其中包含被调用的函数
# 例如: import utils, api_client

def simulate_alpha_list(
    s: requests.Session,
    alpha_simulation_data_list: List[Dict[str, Any]], # 列表，每个元素是 generate_alpha_simulation_data 的输出
    limit_of_concurrent_simulations: int = config.DEFAULT_CONCURRENT_SIMULATIONS,
    simulation_parameters: Dict[str, Any] = None, # 即 DEFAULT_CONFIG
    # 以下参数用于日志/进度条描述，来自遗传算法的上下文
    depth_context: int = 0,
    iteration_context: int = 0
) -> List[Dict[str, Any]]:
    """
    并发模拟 Alpha 列表，并获取其指定的统计数据。

    此函数管理两个阶段的并发：
    1.  **并发启动模拟**: 使用 ThreadPoolExecutor 限制并发启动 Alpha 模拟的数量。
        仅启动模拟，不等待每个模拟完成，以尽快将任务提交给 BRAIN 平台。
    2.  **并发获取统计数据**: 对已成功启动并获得 Alpha ID 的模拟，
        并发地调用 `get_specified_alpha_stats` 来获取详细数据。

    参数:
        s (requests.Session): 已认证的会话对象。
        alpha_simulation_data_list (List[Dict[str, Any]]): 包含多个 Alpha 模拟数据字典的列表。
        limit_of_concurrent_simulations (int): 最大并发模拟/数据获取数量。
        simulation_parameters (Dict[str, Any]): 传递给 `get_specified_alpha_stats` 的参数，
                                               控制获取哪些数据及是否保存。默认为 `config.DEFAULT_CONFIG`。
        depth_context (int): (可选) 当前遗传算法的深度，用于日志/进度条。
        iteration_context (int): (可选) 当前遗传算法的迭代次数，用于日志/进度条。

    返回:
        List[Dict[str, Any]]: 包含每个 Alpha 完整结果 (ID、数据、统计、PnL、测试等) 的列表。
                              结构与 `get_specified_alpha_stats` 的输出一致。
    """
    if simulation_parameters is None:
        simulation_parameters = config.DEFAULT_CONFIG

    total_alphas = len(alpha_simulation_data_list)
    logger.info(f"开始并发模拟 Alpha 列表，总数: {total_alphas}, 并发限制: {limit_of_concurrent_simulations}.")

    # --- 阶段 1: 并发启动所有模拟 ---
    # 此阶段的目标是快速提交所有模拟请求，获取初步的 {'alpha_id': id, 'simulate_data': data} 或错误信息

    # 准备进度条描述
    iteration_desc_part = f"迭代{iteration_context}" if iteration_context > 0 else "初始化"
    pbar_desc_stage1 = f"动态并发模拟Alpha (深度{depth_context}, {iteration_desc_part})"

    initial_simulation_results: List[Dict[str, Any]] = [] # 存储 simulate_single_alpha 的直接输出

    with ThreadPoolExecutor(max_workers=limit_of_concurrent_simulations, thread_name_prefix='SimLaunch') as executor:
        # 使用 submit + as_completed 来动态管理任务，类似于原代码的 tqdm 进度更新方式
        futures_map = {
            executor.submit(simulate_single_alpha, s, alpha_data): alpha_data
            for alpha_data in alpha_simulation_data_list
        }

        with tqdm.tqdm(total=total_alphas, desc=pbar_desc_stage1) as pbar:
            for future in concurrent.futures.as_completed(futures_map):
                original_alpha_data = futures_map[future]
                try:
                    # future.result() 会重新引发在子线程中发生的异常
                    result = future.result()
                    initial_simulation_results.append(result)
                except Exception as e:
                    # 此处的 original_alpha_data 是 futures_map[future]
                    alpha_expr_for_log = original_alpha_data.get('regular', '未知表达式')
                    logger.error(f"并发启动模拟任务时捕获到异常 (Alpha表达式: {alpha_expr_for_log}): {e}", exc_info=True)
                    # 记录失败，并继续处理其他 futures
                    initial_simulation_results.append({
                        'alpha_id': None,
                        'simulate_data': original_alpha_data, # 保留原始数据以供调试
                        'error': f"模拟启动阶段任务执行失败: {str(e)}"
                    })
                finally:
                    pbar.update(1) # 确保进度条总是更新
                # pbar.set_description(f"{pbar_desc_stage1} (当前并发: {executor._work_queue.qsize() + len(executor._threads)})") # 近似并发数

    logger.info(f"所有 Alpha 模拟启动尝试完成。初步结果的数量(成功+失败): {len(initial_simulation_results)}。")

    # --- 阶段 2: 并发获取已成功启动的 Alpha 的详细统计数据 ---
    # 只处理那些在阶段1中没有报告错误的、并且成功获取了alpha_id的结果
    valid_for_stats_fetching = [res for res in initial_simulation_results if res.get('alpha_id') and not res.get('error')]
    # 那些在阶段1就失败的结果，直接加入最终列表
    failed_at_launch_stage = [res for res in initial_simulation_results if not res.get('alpha_id') or res.get('error')]

    final_results_list: List[Dict[str, Any]] = list(failed_at_launch_stage) # 直接加入已失败的

    if valid_for_stats_fetching:
        logger.info(f"开始为 {len(valid_for_stats_fetching)} 个成功启动的 Alpha 并发获取详细统计数据。")

        # 构建 get_specified_alpha_stats 的参数元组列表
        tasks_for_stats = []
        for res in valid_for_stats_fetching:
            tasks_for_stats.append(
                (s, res['alpha_id'], res['simulate_data'], simulation_parameters)
            )

        # 使用 ThreadPoolExecutor (推荐) 或 ThreadPool
        with ThreadPoolExecutor(max_workers=limit_of_concurrent_simulations, thread_name_prefix='StatsFetch') as executor_stats:
            futures_stats_map = {
                # executor_stats.submit(func, *args_tuple)
                executor_stats.submit(get_specified_alpha_stats, *task_args): task_args[1] # key is alpha_id for context
                for task_args in tasks_for_stats
            }
            with tqdm.tqdm(total=len(valid_for_stats_fetching), desc="获取Alpha详细统计数据") as pbar_stats:
                for future_stat in concurrent.futures.as_completed(futures_stats_map):
                    alpha_id_context = futures_stats_map[future_stat]
                    try:
                        detailed_result = future_stat.result()
                        final_results_list.append(detailed_result)
                    except Exception as e:
                        logger.error(f"获取 Alpha ID {alpha_id_context} 的详细统计数据时发生严重错误: {e}", exc_info=True)
                        # 找到此 alpha_id 对应的原始 simulate_data (它在 valid_for_stats_fetching 中)
                        original_sim_data = next((r['simulate_data'] for r in valid_for_stats_fetching if r['alpha_id'] == alpha_id_context), {"unknown_alpha": alpha_id_context})
                        final_results_list.append({
                            'alpha_id': alpha_id_context,
                            'simulate_data': original_sim_data,
                            'is_stats': None, 'pnl': None, 'stats': None, 'is_tests': None,
                            'error': f"获取统计数据阶段任务执行失败: {str(e)}"
                        })
                    finally:
                        pbar_stats.update(1)
    else:
        logger.warning("没有成功启动的 Alpha 可供获取详细统计数据。")

    logger.info(f"并发模拟 Alpha 列表流程结束。总共处理 Alpha 数量: {len(final_results_list)}。")
    return final_results_list


def simulate_alpha_list_multi(
    s: requests.Session,
    alpha_simulation_data_list: List[Dict[str, Any]],
    limit_of_concurrent_batches: int = config.DEFAULT_CONCURRENT_BATCHES,
    alphas_per_batch: int = config.DEFAULT_ALPHAS_PER_BATCH,
    simulation_parameters: Dict[str, Any] = None # 即 DEFAULT_CONFIG
) -> List[Dict[str, Any]]:
    """
    使用平台的批量模拟功能并发模拟 Alpha 列表，并获取统计数据。
    """
    if simulation_parameters is None:
        simulation_parameters = config.DEFAULT_CONFIG

    total_alphas = len(alpha_simulation_data_list)
    logger.info(f"开始批量并发模拟 Alpha 列表 (平台多模拟接口)，总数: {total_alphas}, 每批次: {alphas_per_batch}, 并发批次: {limit_of_concurrent_batches}.")

    if (alphas_per_batch < 2 or alphas_per_batch > 10): # 平台限制
        logger.warning(f"每批次 Alpha 数量 ({alphas_per_batch}) 超出建议范围 (2-10)。强制设为3。")
        alphas_per_batch = 3

    # 根据原代码，如果 Alpha 总数较少，则回退到 simulate_alpha_list (单Alpha并发)
    # TODO: 这个阈值 (e.g., 10) 可以配置化
    if total_alphas < 10: #  and total_alphas > 1 (避免单个alpha也走这个，因为simulate_multi_alpha会处理)
        logger.warning("Alpha 列表较短，转为使用 simulate_alpha_list (单Alpha并发模式) 进行模拟。")
        return simulate_alpha_list(
            s, alpha_simulation_data_list,
            limit_of_concurrent_simulations=limit_of_concurrent_batches, # 复用并发批次限制
            simulation_parameters=simulation_parameters
        )

    # 将 Alpha 列表分成多个批次
    batches = [
        alpha_simulation_data_list[i : i + alphas_per_batch]
        for i in range(0, total_alphas, alphas_per_batch)
    ]

    logger.info(f"已将 {total_alphas} 个 Alpha 分为 {len(batches)} 个批次进行模拟。")

    # --- 阶段 1: 并发启动所有批处理模拟 ---
    # 每个批处理的结果是 List[Dict{'alpha_id': id, 'simulate_data': data, 'error': ...}]
    initial_batch_results_flat: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=limit_of_concurrent_batches, thread_name_prefix='BatchSimLaunch') as executor:
        futures_map = {
            executor.submit(simulate_multi_alpha, s, batch_data): batch_data
            for batch_data in batches
        }
        with tqdm.tqdm(total=len(batches), desc="启动多Alpha批处理模拟") as pbar_batch_launch:
            for future in concurrent.futures.as_completed(futures_map):
                # original_batch_data = futures_map[future] # 未使用
                try:
                    # simulate_multi_alpha 返回 List[Dict{'alpha_id': ..., 'simulate_data': ..., 'error':...}]
                    batch_result_list = future.result()
                    initial_batch_results_flat.extend(batch_result_list)
                except Exception as e:
                    original_batch_data_for_error = futures_map[future] # 获取此失败批次的原始数据
                    logger.error(f"并发启动批处理模拟任务时发生严重错误 (影响批次中 {len(original_batch_data_for_error)} 个Alpha): {e}", exc_info=True)
                    # 标记这个批次的所有Alpha都失败了
                    for alpha_data_in_failed_batch in original_batch_data_for_error:
                        initial_batch_results_flat.append({
                            'alpha_id': None,
                            'simulate_data': alpha_data_in_failed_batch,
                            'error': f"批处理任务执行失败: {str(e)}"
                        })
                finally:
                    pbar_batch_launch.update(1)

    logger.info(f"所有批处理模拟启动尝试完成。初步结果条目数 (可能包含错误标记): {len(initial_batch_results_flat)}。")

    # --- 阶段 2: 并发获取详细统计数据 (与 simulate_alpha_list 类似) ---
    valid_for_stats_fetching = [res for res in initial_batch_results_flat if res.get('alpha_id') and not res.get('error')]
    failed_at_launch_stage = [res for res in initial_batch_results_flat if not res.get('alpha_id') or res.get('error')]

    final_results_list: List[Dict[str, Any]] = list(failed_at_launch_stage)

    if valid_for_stats_fetching:
        logger.info(f"开始为 {len(valid_for_stats_fetching)} 个成功获取 Alpha ID 的模拟 (批量模式) 并发获取详细统计数据。")

        tasks_for_stats_multi = [
            (s, res['alpha_id'], res['simulate_data'], simulation_parameters)
            for res in valid_for_stats_fetching
        ]

        with ThreadPoolExecutor(max_workers=limit_of_concurrent_batches, thread_name_prefix='BatchStatsFetch') as executor_stats_multi:
            futures_stats_map_multi = {
                executor_stats_multi.submit(get_specified_alpha_stats, *task_args): task_args[1] # key is alpha_id
                for task_args in tasks_for_stats_multi
            }
            with tqdm.tqdm(total=len(valid_for_stats_fetching), desc="获取Alpha详细统计(批量模式)") as pbar_stats_multi:
                for future_stat_multi in concurrent.futures.as_completed(futures_stats_map_multi):
                    alpha_id_context = futures_stats_map_multi[future_stat_multi]
                    try:
                        detailed_result = future_stat_multi.result()
                        final_results_list.append(detailed_result)
                    except Exception as e:
                        logger.error(f"获取 Alpha ID {alpha_id_context} (批量模式) 的详细统计数据时发生严重错误: {e}", exc_info=True)
                        original_sim_data = next((r['simulate_data'] for r in valid_for_stats_fetching if r['alpha_id'] == alpha_id_context), {"unknown_alpha": alpha_id_context})
                        final_results_list.append({
                            'alpha_id': alpha_id_context,
                            'simulate_data': original_sim_data,
                            'is_stats': None, 'pnl': None, 'stats': None, 'is_tests': None,
                            'error': f"获取统计数据阶段任务执行失败 (批量模式): {str(e)}"
                        })
                    finally:
                        pbar_stats_multi.update(1)
    else:
        logger.warning("没有成功启动的 Alpha (通过批量模式) 可供获取详细统计数据。")

    logger.info(f"批量并发模拟 Alpha 列表流程结束。总共处理 Alpha 数量: {len(final_results_list)}。")
    return final_results_list

# --- Alpha 测试相关函数 (从原 code.py 迁移) ---

def check_prod_corr_test(s: requests.Session, alpha_id: str, threshold: float = config.DEFAULT_PROD_CORR_THRESHOLD) -> pd.DataFrame:
    """
    检查 Alpha 的生产相关性测试是否通过。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。
        threshold (float): 生产相关性的阈值。默认为 0.7。

    返回:
        pd.DataFrame: 包含生产相关性测试结果的 DataFrame (单行)。
                      列: 'test', 'result', 'limit', 'value', 'alpha_id'
                      如果获取数据失败，返回空 DataFrame。
    """
    logger.debug(f"执行 Alpha ID {alpha_id} 的生产相关性测试，阈值: {threshold}...")
    # prod_corr_df = api_client.get_prod_corr(s, alpha_id) # 依赖 api_client
    # TODO: 确保 api_client.get_prod_corr 可用
    # --- MOCK ---
    if random.random() < 0.1: # 模拟API失败
        logger.warning(f"MOCK: get_prod_corr for {alpha_id} failed")
        prod_corr_df = pd.DataFrame()
    else:
        mock_val = random.uniform(0, 1) if random.random() > 0.2 else random.uniform(0.7, 0.9) # 模拟高相关性
        prod_corr_df = pd.DataFrame({'alphas': [10], 'max': [mock_val]}) # 简化模拟
    # --- END MOCK ---

    if prod_corr_df.empty:
        logger.warning(f"未能获取 Alpha ID {alpha_id} 的生产相关性数据，测试跳过。")
        # 返回一个表示测试未执行或数据不足的特定结果，而不是空DF可能更好
        # 但遵循原代码逻辑，是如果DF为空则value为0
        value = 0.0
    else:
        # 获取 alphas > 0 的最大相关性值 (原逻辑)
        value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max() if not prod_corr_df.empty else 0.0
        if pd.isna(value): value = 0.0 # 处理max()在空Series上返回NaN的情况

    test_result_str = "PASS" if value <= threshold else "FAIL"
    logger.info(f"Alpha ID {alpha_id} 生产相关性测试结果: {test_result_str} (值: {value:.4f}, 阈值: {threshold})")

    result_data = [{
        "test": "PROD_CORRELATION",
        "result": test_result_str,
        "limit": threshold,
        "value": value,
        "alpha_id": alpha_id
    }]
    return pd.DataFrame(result_data)


def check_self_corr_test(s: requests.Session, alpha_id: str, threshold: float = config.DEFAULT_SELF_CORR_THRESHOLD) -> pd.DataFrame:
    """
    检查 Alpha 的自相关性测试是否通过。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。
        threshold (float): 自相关性的阈值。默认为 0.7。

    返回:
        pd.DataFrame: 包含自相关性测试结果的 DataFrame (单行)。
                      列: 'test', 'result', 'limit', 'value', 'alpha_id'
                      如果获取数据失败，默认测试通过 (value=0)。
    """
    logger.debug(f"执行 Alpha ID {alpha_id} 的自相关性测试，阈值: {threshold}...")
    # self_corr_df = api_client.get_self_corr(s, alpha_id) # 依赖 api_client
    # TODO: 确保 api_client.get_self_corr 可用
    # --- MOCK ---
    if random.random() < 0.1: # 模拟API失败
        logger.warning(f"MOCK: get_self_corr for {alpha_id} failed")
        self_corr_df = pd.DataFrame()
    else:
        mock_val = random.uniform(0, 1) if random.random() > 0.2 else random.uniform(0.7, 0.9)
        self_corr_df = pd.DataFrame({'correlation': [mock_val, random.uniform(0, mock_val)]})
    # --- END MOCK ---

    value = 0.0 # 默认值
    if self_corr_df.empty:
        logger.warning(f"未能获取 Alpha ID {alpha_id} 的自相关性数据，测试默认通过 (value=0)。")
    else:
        value = self_corr_df["correlation"].max()
        if pd.isna(value): value = 0.0

    test_result_str = "PASS" if value < threshold else "FAIL" # 注意原逻辑是 < threshold
    logger.info(f"Alpha ID {alpha_id} 自相关性测试结果: {test_result_str} (值: {value:.4f}, 阈值: {threshold})")

    result_data = [{
        "test": "SELF_CORRELATION",
        "result": test_result_str,
        "limit": threshold,
        "value": value,
        "alpha_id": alpha_id
    }]
    return pd.DataFrame(result_data)

# TODO: 从 utils.py 导入保存结果等辅助函数 (如果 get_specified_alpha_stats 等函数需要)
# from utils import save_simulation_result, save_pnl, save_yearly_stats

# TODO: 确保 api_client 中所需的函数 (如 get_alpha_pnl, get_alpha_yearly_stats 等) 已被迁移并可导入
# import api_client (或 from . import api_client)

# 思考:
# - 并发限制参数 (limit_of_concurrent_simulations, limit_of_concurrent_batches, alphas_per_batch)
#   可以考虑移到 config.py 中作为默认值。
# - `simulation_parameters` (即 DEFAULT_CONFIG) 的传递可以更明确。
# - 错误处理：当前主要通过日志记录，部分函数返回包含 'error' 键的字典。
#   可以考虑定义更具体的异常类型。
pass
#       它调用了 get_simulation_result_json, get_alpha_pnl, get_alpha_yearly_stats,
#       save_simulation_result, save_pnl, save_yearly_stats, get_check_submission,
#       check_self_corr_test, check_prod_corr_test。
#       参数: s, alpha_id, simulate_data,以及各种布尔标志 (get_pnl, save_pnl_file 等)。
# def get_alpha_full_details(
#     s: 'requests.Session',
#     alpha_id: Optional[str],
#     simulate_data: Dict[str, Any], # 原始模拟数据，用于返回
#     config_flags: Dict[str, bool] # 从 DEFAULT_CONFIG 传入的配置
# ) -> Dict[str, Any]:
# pass


# TODO: 迁移 simulate_alpha_list 函数 (code.py)
#       这是主要的并发模拟函数，使用了 ThreadPoolExecutor 和 tqdm。
#       它编排了 simulate_single_alpha (只启动模拟) 和 get_specified_alpha_stats 的调用。
#       参数: s, alpha_list (模拟数据字典列表), limit_of_concurrent_simulations, simulation_config, depth, iteration.
# from multiprocessing.pool import ThreadPool # 或 concurrent.futures.ThreadPoolExecutor
# from functools import partial
# import tqdm
# def manage_simulation_workflow(
#     s: 'requests.Session',
#     alpha_expression_list: List[Dict[str, Any]], # 注意这里是包含 simulate_data 的列表
#     concurrent_sim_limit: int = 3,
#     sim_params: Dict[str, Any] = None, # 对应 DEFAULT_CONFIG
#     # depth 和 iteration 参数是遗传算法的上下文，可以考虑是否在此模块保留
#     # 或者让调用方 (genetic_algorithm.py) 处理这些日志/描述信息
#     depth_info: int = 0,
#     iteration_info: int = 0
# ) -> List[Dict[str, Any]]:
# pass

# TODO: 迁移 simulate_alpha_list_multi 函数 (code.py)
#       与 simulate_alpha_list 类似，但使用了平台的批量模拟功能 (simulate_multi_alpha)。
#       编排了 simulate_multi_alpha 和 get_specified_alpha_stats 的调用。
# def manage_batch_simulation_workflow(
#     s: 'requests.Session',
#     alpha_expression_list: List[Dict[str, Any]],
#     concurrent_batch_limit: int = 3,
#     alphas_per_batch: int = 3, # 原 limit_of_multi_simulations
#     sim_params: Dict[str, Any] = None
# ) -> List[Dict[str, Any]]:
# pass

# --- 辅助数据处理和API调用封装 (部分可能仍在api_client.py) ---
# 以下函数在 code.py 中主要围绕 API 调用展开，部分已计划移至 api_client.py。
# simulation_manager 可以提供更高层次的封装，或者直接使用 api_client 中的函数。

# - prettify_result: 决定放入 utils.py，因为它主要是数据展示工具。
# - concat_pnl: 决定放入 utils.py。
# - concat_is_tests: 决定放入 utils.py。

# 思考:
# - `simulation_manager` 是否应该直接依赖 `requests.Session` (s) 作为参数，
#   还是应该内部持有一个 `APIClient` 实例？
#   倾向于接收 Session 对象，这样更灵活，调用方 (main.py) 负责初始化会话。
# - 错误处理和重试逻辑：部分已在 `api_client` 中初步实现，`simulation_manager`
#   可以添加更高级别的重试或回退策略。
pass
