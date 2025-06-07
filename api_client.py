import requests
import os
import getpass
import json
from pathlib import Path
import time
from typing import Tuple, Optional, List, Dict, Any
from urllib.parse import urljoin
import pandas as pd
import logging

import config # 导入配置模块

logger = logging.getLogger(__name__)
# TODO: 主程序中统一配置 logger handler 和 level (e.g., in main.py using logger_config)

# --- 自定义异常类 ---
class BrainPlatformError(Exception):
    """与 WorldQuant BRAIN 平台 API 相关的错误的基类。"""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.message = message

    def __str__(self):
        return f"{self.message} (Status Code: {self.status_code or 'N/A'})"

class ApiClientError(BrainPlatformError):
    """表示客户端错误 (例如，4xx HTTP 状态码)。"""
    pass

class ApiServerError(BrainPlatformError):
    """表示服务器端错误 (例如，5xx HTTP 状态码)。"""
    pass

class AuthenticationError(ApiClientError):
    """专用于认证失败 (例如，401 Unauthorized)。"""
    def __init__(self, message="Authentication failed.", status_code=401, response_text=None, original_exception=None):
        super().__init__(message, status_code, response_text)
        self.original_exception = original_exception


class RateLimitError(ApiClientError):
    """专用于处理速率限制错误 (例如，429 Too Many Requests)。"""
    def __init__(self, message="Rate limit exceeded.", status_code=429, retry_after=None, response_text=None):
        super().__init__(message, status_code, response_text)
        self.retry_after = retry_after

    def __str__(self):
        return f"{super().__str__()} Retry after: {self.retry_after or 'N/A'} seconds."

class NetworkError(BrainPlatformError):
    """表示网络相关错误 (例如，连接超时, DNS解析失败)。"""
    def __init__(self, message="A network error occurred.", original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

class FileOperationError(BrainPlatformError):
    """表示文件操作（读/写）时发生错误。"""
    def __init__(self, message="File operation failed.", filepath=None, original_exception=None):
        super().__init__(message) # Pass message to BrainPlatformError
        self.filepath = filepath
        self.original_exception = original_exception
        # self.message is already set by BrainPlatformError

    def __str__(self):
        return f"{self.message} (File: {self.filepath or 'N/A'})"

class CredentialFileError(AuthenticationError):
    """专用于凭据文件读取或写入失败的错误。"""
    def __init__(self, message="Credential file operation failed.", filepath=None, original_exception=None):
        # Call AuthenticationError's constructor, which in turn calls ApiClientError -> BrainPlatformError
        super().__init__(message, original_exception=original_exception)
        self.filepath = filepath
        # self.message is set by AuthenticationError's constructor

    def __str__(self):
        base_str = super().__str__() # Gets message from AuthenticationError
        return f"{base_str} (File: {self.filepath or 'N/A'})"


# --- 凭据管理 ---
def get_credentials() -> Tuple[str, str]:
    """
    获取平台凭据（邮箱和密码）。
    流程: 环境变量 -> 凭据文件 -> 用户输入。
    """
    credential_email = os.environ.get('BRAIN_CREDENTIAL_EMAIL')
    credential_password = os.environ.get('BRAIN_CREDENTIAL_PASSWORD')

    credentials_file_path = config.CREDENTIALS_FILE_PATH
    credentials_folder_path = config.CREDENTIALS_FOLDER_PATH

    if credential_email and credential_password:
        logger.info("已从环境变量加载凭据。")
        return credential_email, credential_password

    logger.debug(f"尝试从文件加载凭据: {credentials_file_path}")
    try:
        if Path(credentials_file_path).exists() and os.path.getsize(credentials_file_path) > 2:
            with open(credentials_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            if "email" in data and "password" in data:
                logger.info(f"已从文件 {credentials_file_path} 加载凭据。")
                return data["email"], data["password"]
            else:
                logger.warning(f"凭据文件 {credentials_file_path} 格式不正确。")
        else:
            logger.info(f"凭据文件 {credentials_file_path} 不存在或为空。")
    except (IOError, OSError) as e: # More general file errors
        raise CredentialFileError(f"读取凭据文件失败: {e}", filepath=credentials_file_path, original_exception=e) from e
    except json.JSONDecodeError as e:
        raise CredentialFileError(f"解析凭据文件JSON失败: {e}", filepath=credentials_file_path, original_exception=e) from e

    logger.info("未从环境变量或文件加载凭据，提示用户输入。")
    try:
        email = input("请输入您的 WorldQuant BRAIN 邮箱:\n")
        password = getpass.getpass(prompt="请输入您的 WorldQuant BRAIN 密码 (输入时不可见):\n")
    except Exception as e:
        logger.error(f"无法从用户处获取凭据输入 (非交互式环境?): {e}")
        raise AuthenticationError("无法通过交互式输入获取凭据。", original_exception=e) from e

    if not email or not password:
        logger.error("用户输入的邮箱或密码为空。")
        raise AuthenticationError("用户输入的邮箱或密码为空。")

    try:
        os.makedirs(credentials_folder_path, exist_ok=True)
        with open(credentials_file_path, "w", encoding='utf-8') as file:
            json.dump({"email": email, "password": password}, file, indent=4)
        logger.info(f"凭据已保存到 {credentials_file_path}。请确保此文件权限安全 (例如 chmod 600)。")
    except (IOError, OSError) as e:
        # Log error but proceed with current credentials for this session
        logger.error(f"保存凭据到文件 {credentials_file_path} 失败: {e}", exc_info=True)
        # Not raising CredentialFileError here as we still have the credentials for current use.
        # The error is about saving, not obtaining for current session.

    return email, password

# --- 会话管理 ---
def start_session() -> requests.Session:
    """
    登录到 WorldQuant BRAIN 平台并返回一个认证后的会话对象。
    处理标准认证及 Persona 生物识别认证。
    """
    s = requests.Session()
    try:
        s.auth = get_credentials()
    except AuthenticationError as e: # Raised by get_credentials
        logger.error(f"会话启动失败：获取凭据时认证错误: {e}")
        raise # Re-raise to be handled by the caller

    api_auth_url = urljoin(config.API_BASE_URL, "authentication")
    logger.info(f"向 {api_auth_url} 发起认证请求...")

    try:
        response = s.post(api_auth_url) # Initial authentication request

        if response.status_code == 401:
            if response.headers.get("WWW-Authenticate") == "persona":
                persona_url_path = response.headers.get("Location")
                if not persona_url_path:
                    err_msg = "Persona认证需要，但响应中缺少Location头部。"
                    logger.error(err_msg)
                    raise AuthenticationError(err_msg, status_code=response.status_code, response_text=response.text)

                persona_full_url = urljoin(response.url, persona_url_path)
                logger.info(f"需要 Persona 生物识别认证。请在浏览器中访问以下链接，完成后按 Enter 键:\n{persona_full_url}")
                try:
                    input() # Wait for user confirmation
                except RuntimeError: # input() fails in non-interactive environment
                     err_msg = "Persona认证等待用户输入失败 (非交互式环境?)"
                     logger.error(err_msg)
                     raise AuthenticationError(err_msg) from None # Python 3: None to break chain

                logger.info("用户已确认完成 Persona 认证，正在验证...")
                for attempt in range(config.PERSONA_MAX_ATTEMPTS):
                    logger.info(f"Persona认证尝试 #{attempt + 1}/{config.PERSONA_MAX_ATTEMPTS}...")
                    try:
                        # According to original notebook, POST to Persona URL, then check main auth
                        persona_confirm_resp = s.post(persona_full_url)
                        # Some APIs might return 200 or 204 on successful Persona post before main auth is active
                        # Others might return 409 if still pending. raise_for_status might be too aggressive here.
                        # For now, proceed to check main auth URL regardless of persona_confirm_resp status,
                        # unless it's a clear error like 5xx.
                        if persona_confirm_resp.status_code >= 500:
                             persona_confirm_resp.raise_for_status() # Raise for server errors on Persona endpoint

                        final_auth_response = s.post(api_auth_url) # Re-check main authentication
                        if final_auth_response.ok: # 2xx status
                             logger.info("Persona 生物识别认证成功，主会话已激活。")
                             return s
                        logger.warning(f"Persona确认后，主认证仍未成功 (状态: {final_auth_response.status_code})。等待 {config.PERSONA_POLL_INTERVAL} 秒后重试...")
                        time.sleep(config.PERSONA_POLL_INTERVAL)
                    except requests.exceptions.HTTPError as pe: # HTTP error during persona post or final_auth post
                        if pe.response.status_code == 409: # Conflict - Persona still in progress
                            logger.info(f"Persona认证仍在进行中 (409 Conflict)。等待 {config.PERSONA_POLL_INTERVAL} 秒...")
                            time.sleep(config.PERSONA_POLL_INTERVAL)
                        else: # Other HTTP errors
                            raise AuthenticationError(f"Persona认证或确认失败: {pe.response.status_code}", status_code=pe.response.status_code, response_text=pe.response.text) from pe
                    except requests.exceptions.RequestException as pre: # Network error during persona post
                        raise NetworkError(f"Persona认证网络请求失败: {pre}", original_exception=pre) from pre
                raise AuthenticationError(f"已达到 Persona 认证最大尝试次数 ({config.PERSONA_MAX_ATTEMPTS})，认证失败。")
            else: # Non-Persona 401
                raise AuthenticationError(f"认证失败 (非Persona): {response.status_code}", status_code=response.status_code, response_text=response.text)

        response.raise_for_status() # For other non-401 initial errors, or if flow changes and first post can succeed.
        logger.info(f"BRAIN API 会话成功启动。状态码: {response.status_code}")
        return s

    except requests.exceptions.RequestException as e:
        raise NetworkError(f"API认证网络请求失败 for URL {api_auth_url}: {e}", original_exception=e) from e
    # AuthenticationError from get_credentials or Persona flow will pass through
    except BrainPlatformError: # Catch specific custom errors already raised
        raise
    except Exception as e: # Catch-all for other unexpected errors
        raise BrainPlatformError(f"启动会话时发生未知错误: {e}") from e

# --- API Request Helper ---
def _request_with_retry(...) -> requests.Response: # Full implementation as before
    # ... (Full implementation of _request_with_retry as provided in previous correct diffs)
    attempt = 0
    last_exception: Optional[Exception] = None

    while attempt <= max_retries:
        attempt_message = f"(Attempt {attempt + 1}/{max_retries + 1})" if max_retries > 0 else ""
        logger.debug(f"发起 API 请求 {attempt_message}: {method} {url}, Params: {params}, JSON: {json_data is not None}")

        try:
            response = session.request(method, url, params=params, json=json_data)

            if "retry-after" in response.headers:
                try:
                    wait_time = int(response.headers["Retry-After"])
                except ValueError:
                    logger.warning(f"无法解析 Retry-After 头部值: {response.headers['Retry-After']}. 使用默认间隔 {retry_interval}s.")
                    wait_time = retry_interval

                logger.info(f"API响应包含 Retry-After: {wait_time} 秒。URL: {url}")
                if attempt < max_retries:
                    time.sleep(wait_time)
                    attempt +=1
                    last_exception = RateLimitError(f"Rate limit (Retry-After), waited {wait_time}s.", retry_after=wait_time, status_code=response.status_code, response_text=response.text)
                    continue
                else:
                    raise RateLimitError(f"Rate limit (Retry-After) and no retries left.", retry_after=wait_time, status_code=response.status_code, response_text=response.text)

            if response.status_code == 401:
                raise AuthenticationError(status_code=401, response_text=response.text)
            elif response.status_code == 403:
                raise ApiClientError("Forbidden.", status_code=403, response_text=response.text)
            elif response.status_code == 404:
                raise ApiClientError("Resource not found.", status_code=404, response_text=response.text)
            elif response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                wait_time = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else retry_interval
                if attempt < max_retries:
                    logger.warning(f"API请求速率限制 (429): {method} {url}. Retry-After: {retry_after_header or 'N/A'}. 将在 {wait_time} 秒后重试。")
                    time.sleep(wait_time)
                    last_exception = RateLimitError(f"Rate limit (429). Waited {wait_time}s.", retry_after=wait_time, status_code=429, response_text=response.text)
                    attempt += 1
                    continue
                else:
                    raise RateLimitError("Rate limit (429) and no retries left.", retry_after=wait_time, status_code=429, response_text=response.text)

            elif 400 <= response.status_code < 500:
                raise ApiClientError(f"Client error: {response.status_code}", status_code=response.status_code, response_text=response.text)

            elif 500 <= response.status_code < 600:
                if attempt < max_retries:
                    logger.warning(f"API请求服务器错误 ({response.status_code}): {method} {url}. 响应: {response.text[:200]}. 将在 {retry_interval} 秒后重试。")
                    time.sleep(retry_interval)
                    last_exception = ApiServerError(f"Server error: {response.status_code}. Retrying.", status_code=response.status_code, response_text=response.text)
                    attempt += 1
                    continue
                else:
                    raise ApiServerError(f"Server error ({response.status_code}) and no retries left.", status_code=response.status_code, response_text=response.text)

            response.raise_for_status() # For any other unhandled non-2xx status codes not caught above.
            logger.debug(f"API请求成功: {method} {url}, Status: {response.status_code}")
            return response

        except requests.exceptions.Timeout as e:
            last_exception = NetworkError(f"Request timed out: {e}", original_exception=e)
        except requests.exceptions.ConnectionError as e:
            last_exception = NetworkError(f"Connection error: {e}", original_exception=e)
        except requests.exceptions.RequestException as e:
            last_exception = NetworkError(f"Generic request exception: {e}", original_exception=e)

        if attempt < max_retries:
            logger.warning(f"API请求失败 (错误: {type(last_exception).__name__}), 将在 {retry_interval} 秒后重试 ({attempt +1}/{max_retries}). URL: {url}")
            time.sleep(retry_interval)
            attempt += 1
        else:
            logger.error(f"API请求达到最大重试次数 ({max_retries}) 失败。最后一次错误: {last_exception or 'Unknown'}. URL: {url}")
            if last_exception:
                raise last_exception
            else:
                raise BrainPlatformError(f"API request failed after {max_retries} retries for URL {url}.")

    # Should not be reached if logic is correct
    final_error_msg = f"API请求最终失败，即使在重试后。URL: {url}. 最后异常: {last_exception}"
    logger.critical(final_error_msg)
    if last_exception: raise last_exception
    raise BrainPlatformError(final_error_msg)

# --- Refactored Data Fetching Functions ---
# (get_alpha_pnl, get_alpha_yearly_stats, etc., all using _request_with_retry)
# ... (Full implementation of all data fetching functions as provided in previous correct diffs,
#      now using _request_with_retry and updated error handling/logging)

# Example for one function:
def get_alpha_pnl(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    pnl_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/recordsets/pnl")
    logger.debug(f"请求 Alpha PnL 数据 for ID: {alpha_id} from URL: {pnl_url}")
    try:
        response = _request_with_retry(s, "GET", pnl_url)
        pnl_json = response.json()
        pnl_records = pnl_json.get("records", [])
        if not pnl_records:
            logger.info(f"Alpha ID {alpha_id} 没有 PnL 数据返回。")
            return pd.DataFrame()
        pnl_df = pd.DataFrame(pnl_records, columns=["Date", "Pnl"]).assign(
            alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
        ).set_index("Date")
        logger.info(f"成功获取并处理了 Alpha ID {alpha_id} 的 PnL 数据，共 {len(pnl_df)} 条记录。")
        return pnl_df
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 的 PnL 数据失败: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析 Alpha ID {alpha_id} 的 PnL 响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} PnL 数据时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_alpha_yearly_stats(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    stats_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/recordsets/yearly-stats")
    logger.debug(f"请求 Alpha 年度统计数据 for ID: {alpha_id} from URL: {stats_url}")
    try:
        response = _request_with_retry(s, "GET", stats_url)
        stats_json = response.json()
        if not stats_json.get("records"):
            logger.info(f"Alpha ID {alpha_id} 没有年度统计数据返回。")
            return pd.DataFrame()
        schema = stats_json.get("schema", {})
        properties = schema.get("properties", [])
        if not properties:
            logger.warning(f"Alpha ID {alpha_id}: 年度统计响应中缺少 schema.properties。")
            if stats_json["records"] and isinstance(stats_json["records"][0], dict):
                 columns = list(stats_json["records"][0].keys())
            else:
                 logger.error(f"Alpha ID {alpha_id}: 无法从年度统计响应中确定列名。")
                 return pd.DataFrame()
        else:
            columns = [dct["name"] for dct in properties if "name" in dct]
        yearly_stats_df = pd.DataFrame(stats_json["records"], columns=columns).assign(alpha_id=alpha_id)
        logger.info(f"成功获取并处理了 Alpha ID {alpha_id} 的年度统计数据。")
        return yearly_stats_df
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 的年度统计数据失败: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析 Alpha ID {alpha_id} 的年度统计响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} 年度统计数据时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_datasets(s: requests.Session, instrument_type: str = config.DEFAULT_INSTRUMENT_TYPE, region: str = config.DEFAULT_REGION, delay: int = config.DEFAULT_DELAY, universe: str = config.DEFAULT_UNIVERSE) -> pd.DataFrame:
    datasets_url = urljoin(config.API_BASE_URL, "data-sets")
    request_params = {"instrumentType": instrument_type, "region": region, "delay": str(delay), "universe": universe}
    logger.debug(f"请求数据集信息: {datasets_url} with params {request_params}")
    try:
        response = _request_with_retry(s, "GET", datasets_url, params=request_params)
        datasets_json = response.json()
        datasets_df = pd.DataFrame(datasets_json.get('results', []))
        logger.info(f"成功获取 {len(datasets_df)} 个数据集信息。")
        return datasets_df
    except BrainPlatformError as e:
        logger.error(f"获取数据集信息时发生错误: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析数据集响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取数据集信息时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_datafields(s: requests.Session, instrument_type: str = config.DEFAULT_INSTRUMENT_TYPE, region: str = config.DEFAULT_REGION, delay: int = config.DEFAULT_DELAY, universe: str = config.DEFAULT_UNIVERSE, dataset_id: str = '', search: str = '') -> pd.DataFrame:
    datafields_url = urljoin(config.API_BASE_URL, "data-fields")
    request_params = {"instrumentType": instrument_type, "region": region, "delay": str(delay), "universe": universe, "limit": 50}
    if dataset_id: request_params["dataset.id"] = dataset_id
    if search: request_params["search"] = search
    logger.debug(f"请求数据字段信息: {datafields_url} with initial params {request_params}")
    all_datafields_list = []
    current_offset = 0
    total_count = -1
    try:
        while True:
            request_params["offset"] = current_offset
            response = _request_with_retry(s, "GET", datafields_url, params=request_params)
            json_response = response.json()
            results = json_response.get('results', [])
            all_datafields_list.extend(results)
            if total_count == -1: total_count = json_response.get('count', 0)
            logger.debug(f"获取到 {len(results)} 个数据字段, 总计 {len(all_datafields_list)} / {total_count}")
            if not results or len(all_datafields_list) >= total_count: break
            current_offset += request_params["limit"]
        datafields_df = pd.DataFrame(all_datafields_list)
        logger.info(f"成功获取 {len(datafields_df)} 个数据字段信息 (总计预期: {total_count})。")
        return datafields_df
    except BrainPlatformError as e:
        logger.error(f"获取数据字段时发生错误: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析数据字段响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取数据字段时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_prod_corr(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    corr_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/correlations/prod")
    logger.debug(f"请求 Alpha 生产相关性数据 for ID: {alpha_id} from URL: {corr_url}")
    try:
        response = _request_with_retry(s, "GET", corr_url)
        json_response = response.json()
        if not json_response.get("records"):
            logger.info(f"Alpha ID {alpha_id} 没有生产相关性数据。")
            return pd.DataFrame()
        schema = json_response.get("schema", {})
        properties = schema.get("properties", [])
        if not properties:
            logger.warning(f"Alpha ID {alpha_id}: 生产相关性响应中缺少 schema.properties。")
            if json_response["records"] and isinstance(json_response["records"][0], dict): columns = list(json_response["records"][0].keys())
            else: logger.error(f"Alpha ID {alpha_id}: 无法从生产相关性响应中确定列名。"); return pd.DataFrame()
        else: columns = [dct["name"] for dct in properties if "name" in dct]
        prod_corr_df = pd.DataFrame(json_response["records"], columns=columns).assign(alpha_id=alpha_id)
        logger.info(f"成功获取 Alpha ID {alpha_id} 的生产相关性数据。")
        return prod_corr_df
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 的生产相关性数据时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析 Alpha ID {alpha_id} 的生产相关性响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} 生产相关性数据时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_self_corr(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    corr_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/correlations/self")
    logger.debug(f"请求 Alpha 自相关性数据 for ID: {alpha_id} from URL: {corr_url}")
    try:
        response = _request_with_retry(s, "GET", corr_url)
        json_response = response.json()
        if not json_response.get("records"):
            logger.info(f"Alpha ID {alpha_id} 没有自相关性数据。")
            return pd.DataFrame()
        schema = json_response.get("schema", {})
        properties = schema.get("properties", [])
        if not properties:
            logger.warning(f"Alpha ID {alpha_id}: 自相关性响应中缺少 schema.properties。")
            if json_response["records"] and isinstance(json_response["records"][0], dict): columns = list(json_response["records"][0].keys())
            else: logger.error(f"Alpha ID {alpha_id}: 无法从自相关性响应中确定列名。"); return pd.DataFrame()
        else: columns = [dct["name"] for dct in properties if "name" in dct]
        self_corr_df = pd.DataFrame(json_response["records"], columns=columns).assign(alpha_id=alpha_id)
        logger.info(f"成功获取 Alpha ID {alpha_id} 的自相关性数据。")
        return self_corr_df
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 的自相关性数据时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"解析 Alpha ID {alpha_id} 的自相关性响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} 自相关性数据时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_check_submission(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    check_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/check")
    logger.debug(f"请求 Alpha 提交检查结果 for ID: {alpha_id} from URL: {check_url}")
    try:
        response = _request_with_retry(s, "GET", check_url)
        json_response = response.json()
        is_data = json_response.get("is", {})
        if not isinstance(is_data, dict): logger.info(f"Alpha ID {alpha_id}: 'is' 字段为空或非预期类型。"); return pd.DataFrame()
        is_checks = is_data.get("checks", [])
        if not is_checks: logger.info(f"Alpha ID {alpha_id} 没有提交检查数据 ('is.checks' 为空或不存在)。"); return pd.DataFrame()
        checks_df = pd.DataFrame(is_checks).assign(alpha_id=alpha_id)
        if 'year' in checks_df.columns and 'name' in checks_df.columns:
            ladder_rows_filter = checks_df['name'] == 'IS_LADDER_SHARPE'
            if ladder_rows_filter.any():
                def create_ladder_detail(row): return [{'value': row['value'], 'year': row['year']}]
                checks_df['value'] = checks_df['value'].astype('object')
                checks_df.loc[ladder_rows_filter, 'value'] = checks_df[ladder_rows_filter].apply(create_ladder_detail, axis=1)
        logger.info(f"成功获取 Alpha ID {alpha_id} 的提交检查结果。")
        return checks_df
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 的提交检查结果时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"解析 Alpha ID {alpha_id} 的提交检查响应时出错: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} 提交检查结果时发生未知错误: {e}", exc_info=True)
        return pd.DataFrame()

def get_simulation_result_json(s: requests.Session, alpha_id: str) -> Optional[Dict[str, Any]]:
    alpha_details_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}")
    logger.debug(f"请求 Alpha 详细结果 JSON for ID: {alpha_id} from URL: {alpha_details_url}")
    try:
        response = _request_with_retry(s, "GET", alpha_details_url)
        logger.info(f"成功获取 Alpha ID {alpha_id} 的详细结果 JSON。")
        return response.json()
    except ApiClientError as e:
        if e.status_code == 404: logger.warning(f"Alpha ID {alpha_id} 未找到 (404)。")
        else: logger.error(f"获取 Alpha ID {alpha_id} 详细结果时发生客户端API错误: {e}", exc_info=True)
        return None
    except BrainPlatformError as e:
        logger.error(f"获取 Alpha ID {alpha_id} 详细结果时发生API或网络错误: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"解析 Alpha ID {alpha_id} 详细结果响应时出错: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"获取 Alpha ID {alpha_id} 详细结果时发生未知错误: {e}", exc_info=True)
        return None

def submit_alpha(s: requests.Session, alpha_id: str) -> bool:
    submit_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}/submit")
    logger.info(f"尝试提交 Alpha ID {alpha_id} to URL: {submit_url}")
    try:
        response = _request_with_retry(s, "POST", submit_url)
        logger.info(f"Alpha ID {alpha_id} 提交请求成功完成，状态码: {response.status_code}。响应: {response.text[:200]}")
        return True
    except BrainPlatformError as e:
        logger.error(f"提交 Alpha ID {alpha_id} 失败: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"提交 Alpha ID {alpha_id} 时发生未知错误: {e}", exc_info=True)
        return False

def set_alpha_properties(s: requests.Session, alpha_id: str, name: Optional[str] = None, color: Optional[str] = None, selection_desc: str = "None", combo_desc: str = "None", tags: Optional[List[str]] = None) -> bool:
    if not s or not s.auth:
        logger.error(f"更新 Alpha ({alpha_id}) 属性失败: 会话无效或未认证。")
        return False
    if tags is None: tags = ["gen"]
    alpha_url = urljoin(config.API_BASE_URL, f"alphas/{alpha_id}")
    logger.info(f"尝试更新 Alpha ID {alpha_id} 的属性 at URL: {alpha_url}")
    payload: Dict[str, Any] = {"selection": {"description": selection_desc}, "combo": {"description": combo_desc}, "tags": tags}
    if name is not None: payload["name"] = name
    if color is not None: payload["color"] = color
    logger.debug(f"更新 Alpha 属性的请求体: {payload}")
    try:
        response = _request_with_retry(s, "PATCH", alpha_url, json_data=payload)
        logger.info(f"Alpha ID {alpha_id} 属性更新成功。状态码: {response.status_code}, 响应: {response.text[:200]}")
        return True
    except BrainPlatformError as e:
        logger.error(f"更新 Alpha ID {alpha_id} 属性失败 (API/网络层面错误): {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"更新 Alpha ID {alpha_id} 属性时发生未知错误: {e}", exc_info=True)
        return False

# TODOs for functions to be moved to simulation_manager.py remain valid.
pass
