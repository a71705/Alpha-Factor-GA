# alpha_factory/brain_client/api_client.py
import requests
import time
import json
import pandas as pd
import random # 确保导入 random
from typing import List, Dict, Any, Optional, Callable, Union # Callable用于类型提示, Union for simulate_data
from urllib.parse import urljoin, urlparse # 导入 urlparse
from multiprocessing.pool import ThreadPool # 与 code.py 一致，用于并发获取统计数据
# from concurrent.futures import ThreadPoolExecutor # 也可以用这个，但为了保持一致性，先用ThreadPool
import tqdm # 用于显示进度条

from .session_manager import SessionManager # 从同级目录导入 SessionManager
from alpha_factory.utils.config_models import AppConfig # 导入 AppConfig 模型
from alpha_factory.genetic_programming.models import Node # 虽然不直接用Node，但Alpha数据结构可能间接关联

# 默认的模拟配置，如果 AppConfig 中没有提供，可以使用这个作为后备
# 但 cg.md 的设计是 AppConfig 会提供，这里只是为了完整性参考
DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False, # 文件保存逻辑在新框架中可能由其他模块处理或移除
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

class BrainApiClient:
    """
    与 WorldQuant BRAIN API 交互的客户端。
    负责提交Alpha模拟、获取结果、统计数据等。
    """
    def __init__(self, session_manager: SessionManager, config: AppConfig):
        """
        初始化 BrainApiClient。

        Args:
            session_manager (SessionManager): 用于管理API会话的 SessionManager 实例。
            config (AppConfig): 应用程序的配置对象，包含BRAIN设置、模拟参数等。
        """
        self.sm = session_manager
        self.config = config # AppConfig 实例
        self.base_url = session_manager.base_url # 从 session_manager 获取 base_url

    def _request_with_retry(
        self,
        method: str,
        url_path: str,
        max_retries: int = 3,
        retry_delay_base: float = 1.0, # 秒
        **kwargs: Any
    ) -> requests.Response:
        """
        执行API请求，并在遇到可重试的错误时自动重试。

        Args:
            method (str): HTTP方法 (例如 "GET", "POST", "PATCH").
            url_path (str): API的路径 (例如 "simulations", "alphas/alpha_id").
            max_retries (int): 最大重试次数。
            retry_delay_base (float): 初始重试延迟的基数（秒）。延迟会指数增加。
            **kwargs: 传递给 requests.<method> 的其他参数 (例如 json, params, data).

        Returns:
            requests.Response: API的响应对象。

        Raises:
            requests.exceptions.RequestException: 如果在多次重试后请求仍然失败。
            ValueError: 如果方法名无效。
        """
        # 如果 url_path 已经是完整的 URL (例如从 Location header 获取的)，则直接使用
        if url_path.startswith("http://") or url_path.startswith("https://"):
            full_url = url_path
        else:
            full_url = urljoin(self.base_url, url_path)

        session = self.sm.get_session() # 获取最新的有效会话

        last_exception: Optional[Exception] = None
        response: Optional[requests.Response] = None # 初始化 response

        for attempt in range(max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = session.get(full_url, **kwargs)
                elif method.upper() == "POST":
                    response = session.post(full_url, **kwargs)
                elif method.upper() == "PATCH":
                    response = session.patch(full_url, **kwargs)
                elif method.upper() == "DELETE":
                    response = session.delete(full_url, **kwargs)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")

                # 检查是否有 'Retry-After' 头部
                if "retry-after" in response.headers:
                    wait_time = float(response.headers["retry-after"])
                    print(f"信息 (_request_with_retry): API要求等待 {wait_time:.2f} 秒后重试 ({method} {url_path})。")
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        last_exception = requests.exceptions.RetryError(f"API请求等待后重试 (Retry-After: {wait_time}s)")
                        session = self.sm.get_session() # 刷新会话以防万一
                        continue # 进入下一次重试
                    else:
                        print(f"错误 (_request_with_retry): 达到最大重试次数后，API仍要求等待 ({method} {url_path})。")
                        response.raise_for_status() # 如果最后一次还是retry-after，则抛出

                response.raise_for_status() # 如果不是2xx状态码，则抛出HTTPError
                return response # 请求成功

            except requests.exceptions.HTTPError as http_err:
                last_exception = http_err
                # response 对象可能在 session.get/post 等调用中未被成功赋值（例如，在连接错误时）
                # 所以在使用 response.status_code 前要确保 response 不是 None
                status_code = http_err.response.status_code if http_err.response is not None else None

                if status_code in [500, 502, 503, 504, 429] and attempt < max_retries:
                    delay = retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1 * (2**attempt)) # 指数退避 + 抖动
                    print(f"警告 (_request_with_retry): 请求 {method} {full_url} 失败 (状态码 {status_code})，将在 {delay:.2f} 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(delay)
                    session = self.sm.get_session() # 刷新会话
                else: # 不可重试的HTTP错误或达到最大重试次数
                    print(f"错误 (_request_with_retry): 请求 {method} {full_url} 发生HTTP错误: {http_err} (状态码 {status_code})。")
                    raise
            except requests.exceptions.RequestException as req_err: # 其他网络问题，例如超时、连接错误
                last_exception = req_err
                if attempt < max_retries:
                    delay = retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1 * (2**attempt))
                    print(f"警告 (_request_with_retry): 请求 {method} {full_url} 失败: {req_err}，将在 {delay:.2f} 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(delay)
                    session = self.sm.get_session() # 刷新会话
                else:
                    print(f"错误 (_request_with_retry): 请求 {method} {full_url} 在多次重试后仍然失败: {req_err}。")
                    raise

        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(f"请求 {method} {full_url} 在未知情况下失败。")


    def _start_simulation_request(self, simulate_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> requests.Response:
        """
        内部辅助函数：向BRAIN API提交模拟请求。
        迁移自 code.py 中的 start_simulation。
        """
        return self._request_with_retry("POST", "simulations", json=simulate_data)

    def _get_simulation_progress(self, simulation_progress_url: str) -> Dict[str, Any]:
        """
        内部辅助函数：跟踪单个或批量模拟的进度直到完成或出错。
        此函数结合了 code.py 中 simulation_progress 和 multisimulation_progress 的轮询逻辑。
        参数 simulation_progress_url 是从 _start_simulation_request 返回的响应头中的 Location。
        """
        error_flag = False
        progress_data: Dict[str, Any] = {} # Initialize progress_data

        while True:
            try:
                response = self._request_with_retry("GET", simulation_progress_url, max_retries=5)
                progress_data = response.json()

                if response.headers.get("Retry-After", "0") == "0":
                    if progress_data.get("status", "ERROR").upper() == "ERROR":
                        error_flag = True
                    break

                wait_time = float(response.headers["Retry-After"])
                time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                print(f"错误 (_get_simulation_progress): 查询模拟进度 {simulation_progress_url} 时出错: {e}。")
                return {"status": "ERROR", "message": f"查询模拟进度时出错: {e}"}


        if error_flag:
            message = progress_data.get("message", "模拟过程中发生未知错误。")
            print(f"错误 (_get_simulation_progress): 模拟失败。BRAIN报告: {message}")
            return {"status": "ERROR", "message": message, "raw_data": progress_data}

        return progress_data

    def _check_session_timeout(self) -> int:
        """
        内部辅助函数：检查会话的过期时间。
        迁移自 code.py 中的 check_session_timeout。
        
        返回:
            int: 会话的剩余有效期（秒），如果获取失败则返回 0。
        """
        try:
            response = self._request_with_retry("GET", "authentication")
            result = response.json()["token"]["expiry"]
            # 计算剩余时间 (当前时间戳到过期时间戳的差值)
            return result - int(time.time())
        except Exception as e:
            print(f"警告 (_check_session_timeout): 检查会话超时失败: {e}")
            return 0

    def _get_simulation_result_json(self, alpha_id: str) -> Dict[str, Any]:
        """
        内部辅助函数：通过 Alpha ID 获取完整的模拟结果 JSON。
        迁移自 code.py 中的 get_simulation_result_json。
        """
        return self._request_with_retry("GET", f"alphas/{alpha_id}").json()

    def _set_alpha_properties(self, alpha_id: str, tags: Optional[List[str]] = None) -> None:
        """
        内部辅助函数：修改 Alpha 的描述参数，主要是打标签。
        迁移自 code.py 中的 set_alpha_properties。
        """
        # 如果 AppConfig 中有 brain.default_alpha_tags，则使用它，否则使用函数默认或传入的 tags
        default_tags_from_config = getattr(getattr(self.config, 'brain', {}), 'default_alpha_tags', ["gen_default_tag"])
        current_tags = tags if tags is not None else default_tags_from_config

        params = {
            "tags": current_tags,
        }
        try:
            self._request_with_retry("PATCH", f"alphas/{alpha_id}", json=params)
        except requests.exceptions.RequestException as e:
            print(f"警告 (_set_alpha_properties): 为 Alpha {alpha_id} 设置属性失败: {e}")


    def _get_alpha_pnl(self, alpha_id: str) -> pd.DataFrame:
        """内部辅助函数：获取Alpha的PnL数据。迁移自 code.py get_alpha_pnl。"""
        try:
            response = self._request_with_retry("GET", f"alphas/{alpha_id}/recordsets/pnl")
            pnl_data = response.json()
            if pnl_data.get("records", 0) == 0 or not pnl_data.get("records"):
                return pd.DataFrame()

            pnl_df = (
                pd.DataFrame(pnl_data["records"], columns=["Date", "Pnl"])
                .assign(
                    alpha_id=alpha_id,
                    Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
                )
                .set_index("Date")
            )
            return pnl_df
        except Exception as e:
            print(f"警告 (_get_alpha_pnl): 获取 Alpha {alpha_id} 的 PnL 数据失败: {e}")
            return pd.DataFrame()

    def _get_alpha_yearly_stats(self, alpha_id: str) -> pd.DataFrame:
        """内部辅助函数：获取Alpha的年度统计数据。迁移自 code.py get_alpha_yearly_stats。"""
        try:
            response = self._request_with_retry("GET", f"alphas/{alpha_id}/recordsets/yearly-stats")
            stats_data = response.json()

            if stats_data.get("records", 0) == 0 or not stats_data.get("records") or not stats_data.get("schema"):
                return pd.DataFrame()

            columns = [dct["name"] for dct in stats_data["schema"]["properties"]]
            yearly_stats_df = pd.DataFrame(stats_data["records"], columns=columns).assign(alpha_id=alpha_id)
            return yearly_stats_df
        except Exception as e:
            print(f"警告 (_get_alpha_yearly_stats): 获取 Alpha {alpha_id} 的年度统计数据失败: {e}")
            return pd.DataFrame()

    def _get_check_submission_data(self, alpha_id: str) -> pd.DataFrame:
        """内部辅助函数：获取Alpha的提交检查结果。迁移自 code.py get_check_submission。"""
        try:
            response = self._request_with_retry("GET", f"alphas/{alpha_id}/check")
            checks_data = response.json()
            if not checks_data.get("is") or not checks_data["is"].get("checks"): # 更安全的检查
                return pd.DataFrame()

            checks_df = pd.DataFrame(checks_data["is"]["checks"]).assign(alpha_id=alpha_id)

            if 'year' in checks_df.columns and 'name' in checks_df.columns:
                ladder_rows_indices = checks_df[checks_df['name'] == 'IS_LADDER_SHARPE'].index
                if not ladder_rows_indices.empty:
                    for idx in ladder_rows_indices:
                        # 创建新的value结构，避免直接修改迭代中的行 (虽然这里可能安全)
                        new_value = [{'value': checks_df.loc[idx, 'value'], 'year': checks_df.loc[idx, 'year']}]
                        checks_df.loc[idx, 'value'] = new_value

                cols_to_drop_submission = [col for col in ['endDate', 'startDate', 'year'] if col in checks_df.columns]
                if cols_to_drop_submission:
                    checks_df = checks_df.drop(columns=cols_to_drop_submission)
            return checks_df
        except Exception as e:
            print(f"警告 (_get_check_submission_data): 获取 Alpha {alpha_id} 的提交检查数据失败: {e}")
            return pd.DataFrame()

    def _check_self_correlation_test(self, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
        """内部辅助函数：检查Alpha的自相关性测试。迁移自 code.py check_self_corr_test。"""
        try:
            response = self._request_with_retry("GET", f"alphas/{alpha_id}/correlations/self")
            self_corr_data = response.json()

            value = 0.0
            if self_corr_data.get("records"): # 检查 records 是否存在且非空
                columns = [dct["name"] for dct in self_corr_data.get("schema", {}).get("properties", [])]
                if columns: # 确保 schema 和 properties 存在
                    self_corr_df = pd.DataFrame(self_corr_data["records"], columns=columns)
                    if not self_corr_df.empty and "correlation" in self_corr_df.columns:
                        value = self_corr_df["correlation"].max()

            result_val = "PASS" if value < threshold else "FAIL"
            return pd.DataFrame([
                {"test": "SELF_CORRELATION", "result": result_val, "limit": threshold, "value": value, "alpha_id": alpha_id}
            ])
        except Exception as e:
            print(f"警告 (_check_self_correlation_test): Alpha {alpha_id} 自相关性检查失败: {e}")
            return pd.DataFrame([
                {"test": "SELF_CORRELATION", "result": "ERROR", "limit": threshold, "value": "N/A", "alpha_id": alpha_id}
            ])


    def _check_prod_correlation_test(self, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
        """内部辅助函数：检查Alpha的生产相关性测试。迁移自 code.py check_prod_corr_test。"""
        try:
            response = self._request_with_retry("GET", f"alphas/{alpha_id}/correlations/prod")
            prod_corr_data = response.json()

            value = 0.0
            if prod_corr_data.get("records"):
                columns = [dct["name"] for dct in prod_corr_data.get("schema", {}).get("properties", [])]
                if columns:
                    prod_corr_df = pd.DataFrame(prod_corr_data["records"], columns=columns)
                    if not prod_corr_df.empty and "alphas" in prod_corr_df.columns and "max" in prod_corr_df.columns:
                        filtered_corr = prod_corr_df[prod_corr_df.alphas > 0]
                        if not filtered_corr.empty:
                            value = filtered_corr["max"].max()

            result_val = "PASS" if value <= threshold else "FAIL"
            return pd.DataFrame([
                {"test": "PROD_CORRELATION", "result": result_val, "limit": threshold, "value": value, "alpha_id": alpha_id}
            ])
        except Exception as e:
            print(f"警告 (_check_prod_correlation_test): Alpha {alpha_id} 生产相关性检查失败: {e}")
            return pd.DataFrame([
                {"test": "PROD_CORRELATION", "result": "ERROR", "limit": threshold, "value": "N/A", "alpha_id": alpha_id}
            ])

    def _get_specified_alpha_stats_details(
        self,
        alpha_id: Optional[str],
        simulate_data_for_this_alpha: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        内部辅助函数：获取单个Alpha的详细统计数据和测试结果。
        迁移自 code.py 中的 get_specified_alpha_stats。
        配置从 self.config.brain (假设的配置路径) 获取。
        """
        sim_settings = vars(self.config.brain) # Convert BrainSettings model to dict

        cfg_get_pnl = sim_settings.get("get_pnl", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["get_pnl"])
        cfg_get_stats = sim_settings.get("get_stats", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["get_stats"])
        cfg_check_submission = sim_settings.get("check_submission", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_submission"])
        cfg_check_self_corr = sim_settings.get("check_self_corr", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_self_corr"])
        cfg_check_prod_corr = sim_settings.get("check_prod_corr", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_prod_corr"])
        self_corr_threshold = sim_settings.get("self_correlation_threshold", 0.7)
        prod_corr_threshold = sim_settings.get("prod_correlation_threshold", 0.7)

        pnl_df = pd.DataFrame()
        stats_df = pd.DataFrame()
        is_tests_df = pd.DataFrame()
        is_stats_df = pd.DataFrame()

        if alpha_id is None:
            return {
                'alpha_id': None, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': is_stats_df, 'pnl': pnl_df, 'stats': stats_df, 'is_tests': is_tests_df
            }

        try:
            full_simulation_result = self._get_simulation_result_json(alpha_id)

            if "is" in full_simulation_result and isinstance(full_simulation_result["is"], dict):
                is_data_no_checks = {k: v for k, v in full_simulation_result['is'].items() if k != 'checks'}
                is_stats_df = pd.DataFrame([is_data_no_checks]).assign(alpha_id=alpha_id)

            if "is" in full_simulation_result and isinstance(full_simulation_result["is"], dict) and \
               "checks" in full_simulation_result["is"] and isinstance(full_simulation_result["is"]["checks"], list):
                is_tests_df = pd.DataFrame(full_simulation_result["is"]["checks"]).assign(alpha_id=alpha_id)
            else:
                is_tests_df = pd.DataFrame()

            if cfg_get_pnl: pnl_df = self._get_alpha_pnl(alpha_id)
            if cfg_get_stats: stats_df = self._get_alpha_yearly_stats(alpha_id)

            if cfg_check_submission:
                submission_checks_df = self._get_check_submission_data(alpha_id)
                if not submission_checks_df.empty: is_tests_df = submission_checks_df

            # Always append self_corr and prod_corr if flags are set, regardless of check_submission
            # This differs slightly from original code.py's nested ifs, but seems more logical as separate checks.
            if cfg_check_self_corr:
                self_corr_test_df = self._check_self_correlation_test(alpha_id, self_corr_threshold)
                is_tests_df = pd.concat([is_tests_df, self_corr_test_df]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True) if not is_tests_df.empty else self_corr_test_df

            if cfg_check_prod_corr:
                prod_corr_test_df = self._check_prod_correlation_test(alpha_id, prod_corr_threshold)
                is_tests_df = pd.concat([is_tests_df, prod_corr_test_df]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True) if not is_tests_df.empty else prod_corr_test_df

            return {
                'alpha_id': alpha_id, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': is_stats_df, 'pnl': pnl_df, 'stats': stats_df, 'is_tests': is_tests_df
            }

        except Exception as e:
            print(f"错误 (_get_specified_alpha_stats_details): 获取 Alpha {alpha_id} 的详细数据时发生错误: {e}")
            return {
                'alpha_id': alpha_id, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': pd.DataFrame(), 'pnl': pd.DataFrame(), 'stats': pd.DataFrame(),
                'is_tests': pd.DataFrame(), 'error': str(e)
            }

    def _handle_single_alpha_simulation_and_tagging(self, alpha_sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        内部辅助函数：处理单个Alpha的模拟启动和标签设置。
        完全复刻 code.py 中 simulate_single_alpha 的逻辑，确保等待模拟完成。
        """
        alpha_id_for_this_sim = None
        try:
            # 检查会话是否即将过期（复刻 code.py 逻辑）
            if self._check_session_timeout() < 1000:
                print("信息 (_handle_single_alpha_simulation_and_tagging): 会话即将过期，重新登录...")
                self.sm.refresh_session()

            # 启动模拟
            response_start_sim = self._start_simulation_request(alpha_sim_data)

            if response_start_sim.status_code // 100 != 2:
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): 启动模拟失败: {response_start_sim.text}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            if 'Location' not in response_start_sim.headers:
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): 启动模拟后未找到Location头部。响应: {response_start_sim.text[:200]}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            # 等待模拟完成（复刻 code.py 的 simulation_progress 逻辑）
            progress_url_path = urlparse(response_start_sim.headers['Location']).path
            progress_result = self._get_simulation_progress(progress_url_path)

            if progress_result.get("status", "ERROR").upper() == "ERROR" or "alpha" not in progress_result:
                message = progress_result.get("message", "模拟未成功返回Alpha ID。")
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): Alpha 模拟失败或未返回 Alpha ID。BRAIN消息: {message}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            # 模拟成功完成，获取 Alpha ID
            alpha_id_for_this_sim = progress_result["alpha"]
            
            # 设置 Alpha 属性（标签等）
            self._set_alpha_properties(alpha_id_for_this_sim)

        except Exception as e:
            print(f"警告 (_handle_single_alpha_simulation_and_tagging): 处理单个 Alpha 模拟时发生错误: {e}。Alpha表达式: {alpha_sim_data.get('regular', '未知')}")
            return {'alpha_id': None, 'simulate_data': alpha_sim_data}

        return {'alpha_id': alpha_id_for_this_sim, 'simulate_data': alpha_sim_data}


    def run_simulation_workflow(
        self,
        alpha_sim_data_list: List[Dict[str, Any]],
        limit_concurrent: int, # 此参数在新逻辑中不再直接使用，但保留以兼容旧接口
        depth: int,
        iteration: int
    ) -> List[Dict[str, Any]]:
        """
        执行Alpha模拟工作流，完整复刻 code.py 中 simulate_alpha_list 的行为。
        改为顺序处理以避免速率限制问题。

        参数:
            alpha_sim_data_list (List[Dict[str, Any]]): 包含多个Alpha模拟数据的列表。
            limit_concurrent (int): 并发限制数 (在此顺序版本中未使用)。
            depth (int): 当前优化的深度。
            iteration (int): 当前的迭代次数。

        返回:
            List[Dict[str, Any]]: 包含每个Alpha模拟结果（alpha_id 和原始 simulate_data）的列表。
        """
        results = []
        total_alphas = len(alpha_sim_data_list)
        progress_desc_sim = f"顺序模拟Alpha (深度{depth}, {'迭代'+str(iteration) if iteration > 0 else '初始化'})"

        print(f"信息: 开始执行模拟工作流 (深度 {depth}, 迭代 {'初始化' if iteration == 0 else iteration}), 总计 {total_alphas} 个Alpha, 使用顺序处理.")

        with tqdm.tqdm(total=total_alphas, desc=progress_desc_sim, unit="alpha") as pbar:
            for alpha_sim_data in alpha_sim_data_list:
                # 顺序处理每个 Alpha
                result = self._handle_single_alpha_simulation_and_tagging(alpha_sim_data)
                results.append(result)
                pbar.update(1)

        return results


    def get_simulation_results_for_alphas(
        self,
        alpha_id: Optional[str],
        simulate_data_for_this_alpha: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        内部辅助函数：获取单个Alpha的详细统计数据和测试结果。
        迁移自 code.py 中的 get_specified_alpha_stats。
        配置从 self.config.brain (假设的配置路径) 获取。
        """
        sim_settings = vars(self.config.brain) # Convert BrainSettings model to dict

        cfg_get_pnl = sim_settings.get("get_pnl", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["get_pnl"])
        cfg_get_stats = sim_settings.get("get_stats", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["get_stats"])
        cfg_check_submission = sim_settings.get("check_submission", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_submission"])
        cfg_check_self_corr = sim_settings.get("check_self_corr", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_self_corr"])
        cfg_check_prod_corr = sim_settings.get("check_prod_corr", DEFAULT_SIMULATION_PARAMS_FROM_CODE_PY["check_prod_corr"])
        self_corr_threshold = sim_settings.get("self_correlation_threshold", 0.7)
        prod_corr_threshold = sim_settings.get("prod_correlation_threshold", 0.7)

        pnl_df = pd.DataFrame()
        stats_df = pd.DataFrame()
        is_tests_df = pd.DataFrame()
        is_stats_df = pd.DataFrame()

        if alpha_id is None:
            return {
                'alpha_id': None, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': is_stats_df, 'pnl': pnl_df, 'stats': stats_df, 'is_tests': is_tests_df
            }

        try:
            full_simulation_result = self._get_simulation_result_json(alpha_id)

            if "is" in full_simulation_result and isinstance(full_simulation_result["is"], dict):
                is_data_no_checks = {k: v for k, v in full_simulation_result['is'].items() if k != 'checks'}
                is_stats_df = pd.DataFrame([is_data_no_checks]).assign(alpha_id=alpha_id)

            if "is" in full_simulation_result and isinstance(full_simulation_result["is"], dict) and \
               "checks" in full_simulation_result["is"] and isinstance(full_simulation_result["is"]["checks"], list):
                is_tests_df = pd.DataFrame(full_simulation_result["is"]["checks"]).assign(alpha_id=alpha_id)
            else:
                is_tests_df = pd.DataFrame()

            if cfg_get_pnl: pnl_df = self._get_alpha_pnl(alpha_id)
            if cfg_get_stats: stats_df = self._get_alpha_yearly_stats(alpha_id)

            if cfg_check_submission:
                submission_checks_df = self._get_check_submission_data(alpha_id)
                if not submission_checks_df.empty: is_tests_df = submission_checks_df

            # Always append self_corr and prod_corr if flags are set, regardless of check_submission
            # This differs slightly from original code.py's nested ifs, but seems more logical as separate checks.
            if cfg_check_self_corr:
                self_corr_test_df = self._check_self_correlation_test(alpha_id, self_corr_threshold)
                is_tests_df = pd.concat([is_tests_df, self_corr_test_df]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True) if not is_tests_df.empty else self_corr_test_df

            if cfg_check_prod_corr:
                prod_corr_test_df = self._check_prod_correlation_test(alpha_id, prod_corr_threshold)
                is_tests_df = pd.concat([is_tests_df, prod_corr_test_df]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True) if not is_tests_df.empty else prod_corr_test_df

            return {
                'alpha_id': alpha_id, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': is_stats_df, 'pnl': pnl_df, 'stats': stats_df, 'is_tests': is_tests_df
            }

        except Exception as e:
            print(f"错误 (_get_specified_alpha_stats_details): 获取 Alpha {alpha_id} 的详细数据时发生错误: {e}")
            return {
                'alpha_id': alpha_id, 'simulate_data': simulate_data_for_this_alpha,
                'is_stats': pd.DataFrame(), 'pnl': pd.DataFrame(), 'stats': pd.DataFrame(),
                'is_tests': pd.DataFrame(), 'error': str(e)
            }

    def _handle_single_alpha_simulation_and_tagging(self, alpha_sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        内部辅助函数：处理单个Alpha的模拟启动和标签设置。
        完全复刻 code.py 中 simulate_single_alpha 的逻辑，确保等待模拟完成。
        """
        alpha_id_for_this_sim = None
        try:
            # 检查会话是否即将过期（复刻 code.py 逻辑）
            if self._check_session_timeout() < 1000:
                print("信息 (_handle_single_alpha_simulation_and_tagging): 会话即将过期，重新登录...")
                self.sm.refresh_session()

            # 启动模拟
            response_start_sim = self._start_simulation_request(alpha_sim_data)

            if response_start_sim.status_code // 100 != 2:
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): 启动模拟失败: {response_start_sim.text}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            if 'Location' not in response_start_sim.headers:
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): 启动模拟后未找到Location头部。响应: {response_start_sim.text[:200]}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            # 等待模拟完成（复刻 code.py 的 simulation_progress 逻辑）
            progress_url_path = urlparse(response_start_sim.headers['Location']).path
            progress_result = self._get_simulation_progress(progress_url_path)

            if progress_result.get("status", "ERROR").upper() == "ERROR" or "alpha" not in progress_result:
                message = progress_result.get("message", "模拟未成功返回Alpha ID。")
                print(f"错误 (_handle_single_alpha_simulation_and_tagging): Alpha 模拟失败或未返回 Alpha ID。BRAIN消息: {message}")
                return {'alpha_id': None, 'simulate_data': alpha_sim_data}

            # 模拟成功完成，获取 Alpha ID
            alpha_id_for_this_sim = progress_result["alpha"]
            
            # 设置 Alpha 属性（标签等）
            self._set_alpha_properties(alpha_id_for_this_sim)

        except Exception as e:
            print(f"警告 (_handle_single_alpha_simulation_and_tagging): 处理单个 Alpha 模拟时发生错误: {e}。Alpha表达式: {alpha_sim_data.get('regular', '未知')}")
            return {'alpha_id': None, 'simulate_data': alpha_sim_data}

        return {'alpha_id': alpha_id_for_this_sim, 'simulate_data': alpha_sim_data}


    def run_simulation_workflow(
        self,
        alpha_sim_data_list: List[Dict[str, Any]],
        limit_concurrent: int, # 此参数在新逻辑中不再直接使用，但保留以兼容旧接口
        depth: int,
        iteration: int
    ) -> List[Dict[str, Any]]:
        """
        执行Alpha模拟工作流，完整复刻 code.py 中 simulate_alpha_list 的行为。
        改为顺序处理以避免速率限制问题。

        参数:
            alpha_sim_data_list (List[Dict[str, Any]]): 包含多个Alpha模拟数据的列表。
            limit_concurrent (int): 并发限制数 (在此顺序版本中未使用)。
            depth (int): 当前优化的深度。
            iteration (int): 当前的迭代次数。

        返回:
            List[Dict[str, Any]]: 包含每个Alpha模拟结果（alpha_id 和原始 simulate_data）的列表。
        """
        results = []
        total_alphas = len(alpha_sim_data_list)
        progress_desc_sim = f"顺序模拟Alpha (深度{depth}, {'迭代'+str(iteration) if iteration > 0 else '初始化'})"

        print(f"信息: 开始执行模拟工作流 (深度 {depth}, 迭代 {'初始化' if iteration == 0 else iteration}), 总计 {total_alphas} 个Alpha, 使用顺序处理.")

        with tqdm.tqdm(total=total_alphas, desc=progress_desc_sim, unit="alpha") as pbar:
            for alpha_sim_data in alpha_sim_data_list:
                # 顺序处理每个 Alpha
                result = self._handle_single_alpha_simulation_and_tagging(alpha_sim_data)
                results.append(result)
                pbar.update(1)

        return results
