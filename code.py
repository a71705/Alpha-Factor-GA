# %%
import random
# import graphviz # Graphviz 库在此代码中被导入但未实际使用，如果需要可视化树结构，请取消注释并确保安装 graphviz。
from collections import OrderedDict
from typing import Optional
import requests
from urllib.parse import urljoin
import time
import json
import os
import getpass
from pathlib import Path
import pandas as pd
from multiprocessing.pool import ThreadPool
from functools import partial
import tqdm


# %%
def make_clickable_alpha_id(alpha_id: str) -> str:
    """
    在数据框中将 alpha_id 变为可点击链接，
    以便用户可以直接跳转到平台查看模拟结果。

    参数:
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        str: 包含可点击链接的 HTML 字符串。
    """
    url = "https://platform.worldquantbrain.com/alpha/"
    return f'<a href=\"{url}{alpha_id}\">{alpha_id}</a>'


def prettify_result(
    result: list, detailed_tests_view: bool = False, clickable_alpha_id: bool = False
) -> pd.DataFrame:
    """
    将模拟结果整合到一个 DataFrame 中，以便分析 Alpha。
    结果按 fitness 绝对值降序排列。

    参数:
        result (list): 包含每个 Alpha 模拟结果的列表。
        detailed_tests_view (bool): 如果为 True，将显示详细的测试结果视图。默认为 False。
        clickable_alpha_id (bool): 如果为 True，alpha_id 将在 DataFrame 中显示为可点击链接。默认为 False。

    返回:
        pd.DataFrame: 包含 Alpha 统计信息、表达式和测试结果的 DataFrame。
    """
    # 提取并合并所有 Alpha 的 IS (In-Sample) 统计数据
    list_of_is_stats = [
        result[x]["is_stats"]
        for x in range(len(result))
        if result[x]["is_stats"] is not None and not result[x]["is_stats"].empty
    ]
    
    # 添加额外检查
    if list_of_is_stats:
        # 过滤掉空的或全为NA的DataFrame
        valid_stats = [df for df in list_of_is_stats if not df.empty and not df.isna().all().all()]
        if valid_stats:
            is_stats_df = pd.concat(valid_stats, ignore_index=True)
        else:
            is_stats_df = pd.DataFrame()  # 创建空DataFrame
    else:
        is_stats_df = pd.DataFrame()  # 创建空DataFrame
    # 按 fitness 降序排序
    is_stats_df = is_stats_df.sort_values("fitness", ascending=False)

    # 提取所有 Alpha 的表达式
    expressions = {
        result[x]["alpha_id"]: result[x]["simulate_data"]["regular"]
        for x in range(len(result))
        if result[x]["is_stats"] is not None
    }
    expression_df = pd.DataFrame(
        list(expressions.items()), columns=["alpha_id", "expression"]
    )

    # 提取并合并所有 Alpha 的 IS 测试结果
    list_of_is_tests = [
        result[x]["is_tests"]
        for x in range(len(result))
        if result[x]["is_tests"] is not None
    ]
    is_tests_df = pd.concat(list_of_is_tests).reset_index(drop=True)

    if detailed_tests_view:
        # 如果需要详细视图，将 limit, result, value 合并为字典
        cols = ["limit", "result", "value"]
        is_tests_df["details"] = is_tests_df[cols].to_dict(orient="records")
        is_tests_df = is_tests_df.pivot(
            index="alpha_id", columns="name", values="details"
        ).reset_index()
    else:
        # 否则，只使用 result 列
        is_tests_df = is_tests_df.pivot(
            index="alpha_id", columns="name", values="result"
        ).reset_index()

    # 合并统计、表达式和测试结果
    alpha_stats = pd.merge(is_stats_df, expression_df, on="alpha_id")
    alpha_stats = pd.merge(alpha_stats, is_tests_df, on="alpha_id")

    # 删除包含“PENDING”值的列（表示测试还在进行中或失败）
    alpha_stats = alpha_stats.drop(
        columns=alpha_stats.columns[(alpha_stats == "PENDING").any()]
    )
    # 将列名转换为小写并用下划线分隔
    alpha_stats.columns = alpha_stats.columns.str.replace(
        "(?<=[a-z])(?=[A-Z])", "_", regex=True
    ).str.lower()

    if clickable_alpha_id:
        # 如果需要，将 alpha_id 格式化为可点击链接
        return alpha_stats.style.format({"alpha_id": make_clickable_alpha_id})
    return alpha_stats


def concat_pnl(result: list) -> pd.DataFrame:
    """
    将所有 Alpha 的 PnL (盈亏) 数据合并到一个 DataFrame 中。

    参数:
        result (list): 包含每个 Alpha 模拟结果的列表。

    返回:
        pd.DataFrame: 包含所有 Alpha PnL 数据的 DataFrame。
    """
    list_of_pnls = [
        result[x]["pnl"]
        for x in range(len(result))
        if result[x]["pnl"] is not None
    ]
    pnls_df = pd.concat(list_of_pnls).reset_index()
    return pnls_df


def concat_is_tests(result: list) -> pd.DataFrame:
    """
    将所有 Alpha 的 IS (In-Sample) 测试结果合并到一个 DataFrame 中。

    参数:
        result (list): 包含每个 Alpha 模拟结果的列表。

    返回:
        pd.DataFrame: 包含所有 Alpha IS 测试结果的 DataFrame。
    """
    is_tests_list = [
        result[x]["is_tests"]
        for x in range(len(result))
        if result[x]["is_tests"] is not None
    ]
    is_tests_df = pd.concat(is_tests_list).reset_index(drop=True)
    return is_tests_df


def save_simulation_result(result: dict) -> None:
    """
    将模拟结果转储到 'simulation_results' 文件夹中的 JSON 文件。

    参数:
        result (dict): 单个 Alpha 的完整模拟结果字典。
    """
    alpha_id = result["id"]
    region = result["settings"]["region"]
    folder_path = "simulation_results/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}")

    os.makedirs(folder_path, exist_ok=True) # 如果文件夹不存在则创建

    with open(file_path, "w") as file:
        json.dump(result, file, indent=4) # 以可读性更好的方式保存 JSON


def set_alpha_properties(
    s: requests.Session,
    alpha_id: str,
    name: Optional[str] = None,
    color: Optional[str] = None,
    selection_desc: str = "None",
    combo_desc: str = "None",
    tags: list = ["gen"], # 类型提示更准确，默认为列表
) -> None:
    """
    修改 Alpha 的描述参数。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。
        name (Optional[str]): Alpha 的名称。默认为 None。
        color (Optional[str]): Alpha 的颜色。默认为 None。
        selection_desc (str): Alpha 的选择描述。默认为 "None"。
        combo_desc (str): Alpha 的组合描述。默认为 "None"。
        tags (list): Alpha 的标签列表。默认为 ["gen"]。
    """
    params = {
        "color": color,
        "name": name,
        "tags": tags,
        "category": None,
        "regular": {"description": None},
        "combo": {"description": combo_desc},
        "selection": {"description": selection_desc},
    }
    response = s.patch(
        "https://api.worldquantbrain.com/alphas/" + alpha_id, json=params
    )
    # 可以在此添加错误处理，例如 if response.status_code != 200: print(response.text)


def save_pnl(pnl_df: pd.DataFrame, alpha_id: str, region: str) -> None:
    """
    将 PnL 数据转储到 'alphas_pnl' 文件夹中的 CSV 文件。

    参数:
        pnl_df (pd.DataFrame): PnL 数据帧。
        alpha_id (str): Alpha 的唯一标识符。
        region (str): Alpha 所在的区域。
    """
    folder_path = "alphas_pnl/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}.csv") # 添加 .csv 扩展名
    os.makedirs(folder_path, exist_ok=True)

    pnl_df.to_csv(file_path)


def save_yearly_stats(yearly_stats: pd.DataFrame, alpha_id: str, region: str) -> None:
    """
    将年度统计数据转储到 'yearly_stats' 文件夹中的 CSV 文件。

    参数:
        yearly_stats (pd.DataFrame): 年度统计数据帧。
        alpha_id (str): Alpha 的唯一标识符。
        region (str): Alpha 所在的区域。
    """
    folder_path = "yearly_stats/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}.csv") # 添加 .csv 扩展名
    os.makedirs(folder_path, exist_ok=True)

    yearly_stats.to_csv(file_path, index=False)


def get_alpha_pnl(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    获取 Alpha 模拟的 PnL (盈亏) 数据。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        pd.DataFrame: 包含 PnL 数据的 DataFrame，如果无数据则返回空 DataFrame。
    """
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/recordsets/pnl"
        )
        # 检查是否需要重试（例如，平台限流）
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json().get("records", 0)
    if pnl == 0:
        return pd.DataFrame() # 如果没有记录，返回空 DataFrame
    pnl_df = (
        pd.DataFrame(pnl, columns=["Date", "Pnl"])
        .assign(
            alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
        )
        .set_index("Date")
    )
    return pnl_df


def get_alpha_yearly_stats(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    获取 Alpha 模拟的年度统计数据。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        pd.DataFrame: 包含年度统计数据的 DataFrame，如果无数据则返回空 DataFrame。
    """
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/"
            + alpha_id
            + "/recordsets/yearly-stats"
        )
        # 检查是否需要重试
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    stats = result.json()

    if stats.get("records", 0) == 0:
        return pd.DataFrame() # 如果没有记录，返回空 DataFrame

    columns = [dct["name"] for dct in stats["schema"]["properties"]]
    yearly_stats_df = pd.DataFrame(stats["records"], columns=columns).assign(alpha_id=alpha_id)
    return yearly_stats_df


def get_datasets(
    s: requests.Session,
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000'
) -> pd.DataFrame:
    """
    获取平台上的数据集信息。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        instrument_type (str): 资产类型（如 'EQUITY'）。
        region (str): 区域（如 'USA'）。
        delay (int): 延迟（如 1）。
        universe (str): 股票池（如 'TOP3000'）。

    返回:
        pd.DataFrame: 包含数据集信息的 DataFrame。
    """
    url = "https://api.worldquantbrain.com/data-sets?" +\
        f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}"
    result = s.get(url)
    datasets_df = pd.DataFrame(result.json()['results'])
    return datasets_df


def get_datafields(
    s: requests.Session,
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000',
    dataset_id: str = '',
    search: str = ''
) -> pd.DataFrame:
    """
    获取平台上的数据字段信息。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        instrument_type (str): 资产类型（如 'EQUITY'）。
        region (str): 区域（如 'USA'）。
        delay (int): 延迟（如 1）。
        universe (str): 股票池（如 'TOP3000'）。
        dataset_id (str): 数据集ID，用于过滤特定数据集的数据字段。
        search (str): 搜索关键词，用于模糊匹配数据字段名称。

    返回:
        pd.DataFrame: 包含数据字段信息的 DataFrame。
    """
    # 根据是否有搜索关键词构建不同的 URL 模板
    if len(search) == 0:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
            "&offset={x}"
        # 获取总数以确定分页次数
        count = s.get(url_template.format(x=0)).json()['count']
    else:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
            f"&search={search}" +\
            "&offset={x}"
        # 如果有搜索，通常不需要精确的总数，可以设定一个较大的限制
        count = 1000 # 默认为1000，防止无限循环，如果数据字段很多，可能需要更大
    
    datafields_list = []
    for x in range(0, count, 50): # 每次请求获取50条数据
        datafields = s.get(url_template.format(x=x))
        # 确保请求成功，并且有 'results' 键
        if datafields.status_code == 200 and 'results' in datafields.json():
            datafields_list.append(datafields.json()['results'])
        else:
            print(f"获取数据字段失败，状态码: {datafields.status_code}, 响应: {datafields.text}")
            break # 出现错误则停止

    # 将列表的列表扁平化为一个列表
    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    return datafields_df


# 默认的模拟配置
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


def get_credentials() -> tuple[str, str]:
    """
    获取平台凭据（邮箱和密码）。
    如果 '~/secrets/platform-brain.json' 文件存在且不为空，则从中读取。
    否则，会提示用户输入凭据并保存到文件中。
    也可以从环境变量 BRAIN_CREDENTIAL_EMAIL 和 BRAIN_CREDENTIAL_PASSWORD 中获取。

    返回:
        tuple[str, str]: 包含邮箱和密码的元组。
    """
    credential_email = os.environ.get('BRAIN_CREDENTIAL_EMAIL')
    credential_password = os.environ.get('BRAIN_CREDENTIAL_PASSWORD')

    credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
    credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")

    if (
        Path(credentials_file_path).exists()
        and os.path.getsize(credentials_file_path) > 2 # 检查文件大小是否大于2字节（即不是空字典 {}）
    ):
        with open(credentials_file_path) as file:
            data = json.loads(file.read())
    else:
        os.makedirs(credentials_folder_path, exist_ok=True) # 如果文件夹不存在则创建
        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            email = input("邮箱:\n")
            password = getpass.getpass(prompt="密码:")
        data = {"email": email, "password": password}
        with open(credentials_file_path, "w") as file:
            json.dump(data, file)
    return (data["email"], data["password"])


def start_session() -> requests.Session:
    """
    登录到 WorldQuant Brain 平台并返回一个认证后的会话对象。
    如果需要，会处理生物识别认证（Persona）。

    返回:
        requests.Session: 已认证的请求会话对象。
    """
    s = requests.Session()
    s.auth = get_credentials() # 设置会话的认证信息
    r = s.post("https://api.worldquantbrain.com/authentication")
    

    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers["WWW-Authenticate"] == "persona":
            # 处理生物识别认证流程
            print(
                "请完成生物识别认证，然后按任意键继续: \n"
                + urljoin(r.url, r.headers["Location"]) + "\n"
            )
            input() # 等待用户输入
            s.post(urljoin(r.url, r.headers["Location"]))

            while True:
                # 循环检查认证是否完成
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input("生物识别认证未完成。请重试并在完成后按任意键继续。\n")
                else:
                    break
        else:
            # 处理错误的邮箱或密码
            print("\n邮箱或密码不正确。\n")
            # 清空保存的凭据文件，以便下次重新输入
            with open(
                os.path.join(os.path.expanduser("~"), "secrets/platform-brain.json"),
                "w",
            ) as file:
                json.dump({}, file)
            return start_session() # 递归调用以重新获取凭据
    return s


def check_session_timeout(s: requests.Session) -> int:
    """
    检查会话的过期时间。

    参数:
        s (requests.Session): 已认证的请求会话对象。

    返回:
        int: 会话的剩余有效期（秒），如果获取失败则返回 0。
    """
    authentication_url = "https://api.worldquantbrain.com/authentication"
    try:
        # 获取 token 的过期时间戳
        result = s.get(authentication_url).json()["token"]["expiry"]
        # 计算剩余时间 (当前时间戳到过期时间戳的差值)
        return result - int(time.time())
    except:
        return 0


def generate_alpha(
    regular: str,
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "INDUSTRY",
    delay: int = 1,
    decay: int = 0,
    truncation: float = 0.08,
    nan_handling: str = "OFF",
    unit_handling: str = "VERIFY",
    pasteurization: str = "ON",
    visualization: bool = False,
) -> dict:
    """
    生成用于 Alpha 模拟的数据字典，包含默认参数。

    参数:
        regular (str): Alpha 的表达式字符串。
        region (str): 区域（如 'USA'）。
        universe (str): 股票池（如 'TOP3000'）。
        neutralization (str): 中性化类型（如 'INDUSTRY'）。
        delay (int): 延迟（如 1）。
        decay (int): 衰减值。
        truncation (float): 截断值。
        nan_handling (str): NaN 处理方式（如 'OFF'）。
        unit_handling (str): 单位处理方式（如 'VERIFY'）。
        pasteurization (str): 巴氏消毒（如 'ON'）。
        visualization (bool): 是否启用可视化。

    返回:
        dict: 包含 Alpha 模拟设置的字典。
    """
    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "nanHandling": nan_handling,
            "instrumentType": "EQUITY",
            "delay": delay,
            "universe": universe,
            "truncation": truncation,
            "unitHandling": unit_handling,
            "pasteurization": pasteurization,
            "region": region,
            "language": "FASTEXPR",
            "decay": decay,
            "neutralization": neutralization,
            "visualization": visualization,
        },
        "regular": regular,
    }
    return simulation_data


def start_simulation(s: requests.Session, simulate_data: dict or list) -> requests.Response:
    """
    启动 Alpha 模拟。可以是单个 Alpha 或多个 Alpha 的批处理模拟。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_data (dict or list): 单个 Alpha 的模拟数据字典，或多个 Alpha 模拟数据字典的列表。

    返回:
        requests.Response: 模拟请求的响应对象。
    """
    simulate_response = s.post(
        "https://api.worldquantbrain.com/simulations", json=simulate_data
    )
    return simulate_response


def simulation_progress(s: requests.Session, simulate_response: requests.Response) -> dict:
    """
    跟踪单个 Alpha 模拟的进度并获取结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_response (requests.Response): 启动模拟请求的响应。

    返回:
        dict: 包含模拟完成状态和结果的字典。如果完成，'completed' 为 True，'result' 为模拟结果；
              否则 'completed' 为 False，'result' 为空字典。
    """
    if simulate_response.status_code // 100 != 2:
        # 如果请求状态码不是 2xx (成功)，则打印错误信息
        print(f"启动模拟失败: {simulate_response.text}")
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        # 循环检查模拟进度
        simulation_progress = s.get(simulation_progress_url)
        # 检查 'Retry-After' 头部，如果存在则等待指定秒数
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            # 如果没有 'Retry-After' 头部，表示模拟已完成或出错
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break
        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:
        print("发生错误。")
        if "message" in simulation_progress.json():
            print(f"错误信息: {simulation_progress.json()['message']}")
        return {"completed": False, "result": {}}

    alpha = simulation_progress.json().get("alpha", 0)
    if alpha == 0:
        # 如果 Alpha ID 未返回，表示模拟失败或没有 Alpha 生成
        print("未获取到 Alpha ID。")
        return {"completed": False, "result": {}}
    simulation_result = get_simulation_result_json(s, alpha)
    return {"completed": True, "result": simulation_result}


def multisimulation_progress(s: requests.Session, simulate_response: requests.Response) -> dict:
    """
    跟踪多个 Alpha 批处理模拟的进度并获取所有子模拟的结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_response (requests.Response): 启动批处理模拟请求的响应。

    返回:
        dict: 包含模拟完成状态和结果的字典。如果完成，'completed' 为 True，
              'result' 为所有子 Alpha 模拟结果的列表；否则 'completed' 为 False，
              'result' 为空字典。
    """
    if simulate_response.status_code // 100 != 2:
        # 如果请求状态码不是 2xx (成功)，则打印错误信息
        print(f"启动多模拟失败: {simulate_response.text}")
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        # 循环检查模拟进度
        simulation_progress = s.get(simulation_progress_url)
        # 检查 'Retry-After' 头部，如果存在则等待指定秒数
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            # 如果没有 'Retry-After' 头部，表示模拟已完成或出错
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break
        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:
        print("发生错误。")
        if "message" in simulation_progress.json():
            print(f"错误信息: {simulation_progress.json()['message']}")
        return {"completed": False, "result": {}}

    children = simulation_progress.json().get("children", [])
    if len(children) == 0:
        # 如果没有子 Alpha ID，表示模拟失败或没有 Alpha 生成
        print("未获取到子 Alpha ID。")
        return {"completed": False, "result": {}}

    children_list = []
    for child_id in children:
        # 为每个子 Alpha 获取其详细模拟结果
        child_progress = s.get("https://api.worldquantbrain.com/simulations/" + child_id)
        # 检查 child_progress 的状态码和内容，以避免 KeyError
        if child_progress.status_code == 200 and "alpha" in child_progress.json():
            alpha = child_progress.json()["alpha"]
            child_result = get_simulation_result_json(s, alpha)
            children_list.append(child_result)
        else:
            print(f"获取子模拟 {child_id} 结果失败。状态码: {child_progress.status_code}, 响应: {child_progress.text}")

    return {"completed": True, "result": children_list}


def get_prod_corr(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    获取 Alpha 的生产相关性数据。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        pd.DataFrame: 包含生产相关性数据的 DataFrame，如果无数据则返回空 DataFrame。
    """
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/prod"
        )
        # 检查是否需要重试
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    return prod_corr_df


def check_prod_corr_test(s: requests.Session, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    检查 Alpha 的生产相关性测试是否通过。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。
        threshold (float): 生产相关性的阈值。

    返回:
        pd.DataFrame: 包含生产相关性测试结果的 DataFrame。
    """
    prod_corr_df = get_prod_corr(s, alpha_id)
    # 获取 alphas > 0 的最大相关性值
    value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max() if not prod_corr_df.empty else 0.0
    result = [
        {"test": "PROD_CORRELATION", "result": "PASS" if value <= threshold else "FAIL", "limit": threshold, "value": value, "alpha_id": alpha_id}
    ]
    return pd.DataFrame(result)


def get_self_corr(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    获取 Alpha 的自相关性数据。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        pd.DataFrame: 包含自相关性数据的 DataFrame，如果无数据则返回空 DataFrame。
    """
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/self"
        )
        # 检查是否需要重试
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()

    records_len = len(result.json()["records"])
    if records_len == 0:
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    return self_corr_df


def check_self_corr_test(s: requests.Session, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    检查 Alpha 的自相关性测试是否通过。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。
        threshold (float): 自相关性的阈值。

    返回:
        pd.DataFrame: 包含自相关性测试结果的 DataFrame。
    """
    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        # 如果没有自相关性数据，默认认为通过
        result = [{"test": "SELF_CORRELATION", "result": "PASS", "limit": threshold, "value": 0, "alpha_id": alpha_id}]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id
            }
        ]
    return pd.DataFrame(result)


def get_check_submission(s: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    获取 Alpha 的提交检查结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        pd.DataFrame: 包含提交检查结果的 DataFrame。
    """
    while True:
        result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
        # 检查是否需要重试
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        return pd.DataFrame()

    checks_df = pd.DataFrame(
            result.json()["is"]["checks"]
    ).assign(alpha_id=alpha_id)

    # 特殊处理 'IS_LADDER_SHARPE' 测试结果
    if 'year' in checks_df.columns and 'name' in checks_df.columns: # 确保列存在
        ladder_row = checks_df.loc[checks_df['name']=='IS_LADDER_SHARPE']
        if not ladder_row.empty:
            # 将梯形夏普比率的年份和值作为一个字典列表存储
            ladder_dict = [ladder_row[['value', 'year']].iloc[0].to_dict()]
            checks_df.loc[checks_df['name']=='IS_LADDER_SHARPE', 'value'] = [ladder_dict] * len(ladder_row) # 确保广播正确
            # 删除不再需要的列
            checks_df = checks_df.drop(['endDate', 'startDate', 'year'], axis=1) # inplace=True 已废弃

    return checks_df


def submit_alpha(s: requests.Session, alpha_id: str) -> bool:
    """
    提交一个 Alpha。
    注意: 此函数在原 Notebook 中定义但未被调用。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        bool: 如果提交成功则返回 True，否则返回 False。
    """
    result = s.post("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit")
    while True:
        # 检查是否需要重试
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
            result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit") # 重新获取状态
        else:
            break
    return result.status_code == 200


def get_simulation_result_json(s: requests.Session, alpha_id: str) -> dict:
    """
    通过 Alpha ID 获取完整的模拟结果 JSON。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (str): Alpha 的唯一标识符。

    返回:
        dict: 包含 Alpha 模拟结果的 JSON 字典。
    """
    return s.get("https://api.worldquantbrain.com/alphas/" + alpha_id).json()


def simulate_single_alpha(s: requests.Session, simulate_data: dict) -> dict:
    """
    模拟单个 Alpha。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_data (dict): 单个 Alpha 的模拟数据字典。

    返回:
        dict: 包含 Alpha ID 和模拟数据的字典。如果模拟失败，alpha_id 为 None。
    """
    # 检查会话是否即将过期，如果过期则重新登录
    if check_session_timeout(s) < 1000:
        s = start_session()

    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)
    

    if not simulation_result["completed"]:
        return {'alpha_id': None, 'simulate_data': simulate_data}
    
    # 设置 Alpha 属性（例如标签为 'gen'）
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {'alpha_id': simulation_result["result"]["id"], 'simulate_data': simulate_data}


def simulate_multi_alpha(s: requests.Session, simulate_data_list: list[dict]) -> list[dict]:
    """
    模拟多个 Alpha（批处理模拟）。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        simulate_data_list (list[dict]): 包含多个 Alpha 模拟数据字典的列表。

    返回:
        list[dict]: 包含每个 Alpha ID 和模拟数据的列表。如果模拟失败，对应 Alpha ID 为 None。
    """
    # 检查会话是否即将过期，如果过期则重新登录
    if check_session_timeout(s) < 1000:
        s = start_session()
    
    if len(simulate_data_list) == 1:
        # 如果只有一个 Alpha，则调用单 Alpha 模拟函数
        return [simulate_single_alpha(s, simulate_data_list[0])]
    
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)

    if not simulation_result["completed"]:
        # 如果批处理模拟未完成，为每个输入数据返回 None 的 Alpha ID
        return [{'alpha_id': None, 'simulate_data': x} for x in simulate_data_list]
    
    # 解析批处理模拟结果，提取 Alpha ID 和原始模拟数据
    result = []
    for x in simulation_result["result"]:
        # 确保 x['regular'] 是一个字典，且包含 'code' 键
        regular_code = x['regular']['code'] if isinstance(x.get('regular'), dict) and 'code' in x['regular'] else None
        if regular_code:
            result.append({
                "alpha_id": x["id"],
                "simulate_data": {
                    "type": x["type"],
                    "settings": x["settings"],
                    "regular": regular_code
                }
            })
        else:
            print(f"警告: Alpha ID {x.get('id', '未知')} 的常规表达式代码缺失或格式不正确。")
            result.append({'alpha_id': None, 'simulate_data': x}) # 返回失败状态

    # 为所有成功生成的 Alpha 设置属性
    # Note: original code passes simulation_result["result"] which is a list of full alpha JSONs, not just IDs.
    # set_alpha_properties expects an ID, so this part needs adjustment if the full result isn't always present.
    # Assuming x is the full alpha object from simulation_result["result"]
    for alpha_obj in simulation_result["result"]:
        set_alpha_properties(s, alpha_obj["id"])
    
    return result


def get_specified_alpha_stats(
    s: requests.Session,
    alpha_id: Optional[str],
    simulate_data: dict,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
) -> dict:
    """
    获取指定 Alpha 的统计信息和测试结果。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_id (Optional[str]): Alpha 的唯一标识符。如果为 None，则直接返回空结果。
        simulate_data (dict): 原始模拟数据字典。
        get_pnl (bool): 是否获取 PnL 数据。
        get_stats (bool): 是否获取年度统计数据。
        save_pnl_file (bool): 是否将 PnL 数据保存到文件。
        save_stats_file (bool): 是否将年度统计数据保存到文件。
        save_result_file (bool): 是否将完整模拟结果保存到文件。
        check_submission (bool): 是否执行提交检查。
        check_self_corr (bool): 是否执行自相关性检查。
        check_prod_corr (bool): 是否执行生产相关性检查。

    返回:
        dict: 包含 Alpha ID、模拟数据、IS 统计、PnL、年度统计和测试结果的字典。
    """
    pnl = None
    stats = None
    is_tests = None

    if alpha_id is None:
        return {'alpha_id': None, 'simulate_data': simulate_data, 'is_stats': None, 'pnl': pnl, 'stats': stats, 'is_tests': None}

    result = get_simulation_result_json(s, alpha_id)
    region = result["settings"]["region"]
    # 提取 IS 统计数据，排除 'checks' 键
    is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key!='checks'}]).assign(alpha_id=alpha_id)

    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)

    if save_result_file:
        save_simulation_result(result)
    if save_pnl_file and get_pnl:
        save_pnl(pnl, alpha_id, region)
    if save_stats_file and get_stats:
        save_yearly_stats(stats, alpha_id, region)

    # 初始获取 IS 测试结果
    is_tests = pd.DataFrame(result["is"]["checks"]).assign(alpha_id=alpha_id)

    if check_submission:
        # 如果需要执行提交检查，则调用专门的函数
        # 注意: get_check_submission 包含了 IS_LADDER_SHARPE 的特殊处理，可能覆盖is_tests的其他部分
        is_tests = get_check_submission(s, alpha_id)
        # 如果只检查提交，则直接返回结果
        return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}

    # 如果不检查提交，但需要检查自相关或生产相关，则追加这些测试结果
    if check_self_corr: # and not check_submission (implicitly handled by the if above)
        self_corr_test = check_self_corr_test(s, alpha_id)
        # 合并并去重，保留最新（追加的）测试结果
        is_tests = pd.concat([is_tests, self_corr_test]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True)
    
    if check_prod_corr: # and not check_submission (implicitly handled by the if above)
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        # 合并并去重，保留最新（追加的）测试结果
        is_tests = pd.concat([is_tests, prod_corr_test]).drop_duplicates(subset=["test"], keep="last").reset_index(drop=True)

    return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}


def simulate_alpha_list(
    s: requests.Session,
    alpha_list: list[dict], # alpha_list 包含模拟数据字典，不是 Alpha ID
    limit_of_concurrent_simulations: int = 3,
    batch_size: int = 10,  # 批处理大小，保留参数但不使用
    simulation_config: dict = DEFAULT_CONFIG,
    depth: int = 1,  # 添加深度参数
    iteration: int = 0,  # 添加迭代次数参数
) -> list[dict]:
    """
    模拟 Alpha 列表，使用动态并发控制：单个Alpha模拟 + 动态补充机制。
    当一个Alpha完成时，立即启动下一个，保持指定数量的并发。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_list (list[dict]): 包含 Alpha 模拟数据字典的列表。
        limit_of_concurrent_simulations (int): 并发模拟的数量限制。
        batch_size (int): 保留参数，兼容性考虑（本函数中不使用）。
        simulation_config (dict): 模拟配置字典。

    返回:
        list[dict]: 包含每个 Alpha 模拟结果（包括 ID、数据、统计、PnL、测试等）的列表。
    """
    import concurrent.futures
    import threading
    
    result_list = []
    result_lock = threading.Lock()
    
    # 第一阶段：启动所有模拟（控制并发数量）
    simulation_results = []
    
    def start_single_simulation(alpha_data):
        """只启动单个Alpha模拟，不获取统计数据"""
        try:
            single_result = simulate_single_alpha(s, alpha_data)
            return single_result
        except Exception as e:
            print(f"警告: Alpha模拟启动失败: {e}")
            return {'alpha_id': None, 'simulate_data': alpha_data}
    
    # 使用ThreadPoolExecutor控制模拟启动的并发数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=limit_of_concurrent_simulations) as executor:
        # 创建任务队列
        alpha_queue = list(alpha_list)
        running_futures = {}
        completed_count = 0
        
        # 使用进度条显示进度
        iteration_info = f"迭代{iteration}" if iteration > 0 else "初始化"
        with tqdm.tqdm(total=len(alpha_list), desc=f"正在动态并发模拟Alpha (深度{depth}, {iteration_info})") as pbar:
            # 初始启动最多limit_of_concurrent_simulations个任务
            while len(running_futures) < limit_of_concurrent_simulations and alpha_queue:
                alpha_data = alpha_queue.pop(0)
                future = executor.submit(start_single_simulation, alpha_data)
                running_futures[future] = alpha_data
                # 更新进度条描述
                iteration_info = f"迭代{iteration}" if iteration > 0 else "初始化"
                pbar.set_description(f"正在动态并发模拟Alpha (深度{depth}, {iteration_info}, 当前并发: {len(running_futures)})")
            
            # 动态管理任务：当有任务完成时，立即启动新任务
            while running_futures:
                # 等待至少一个任务完成
                done_futures = set()
                for future in concurrent.futures.as_completed(running_futures.keys()):
                    done_futures.add(future)
                    break  # 只处理一个完成的任务，然后立即启动新任务
                
                # 处理完成的任务
                for future in done_futures:
                    alpha_data = running_futures.pop(future)
                    try:
                        result = future.result()
                        with result_lock:
                            simulation_results.append(result)
                    except Exception as e:
                        print(f"警告: Alpha模拟任务异常: {e}")
                        with result_lock:
                            simulation_results.append({'alpha_id': None, 'simulate_data': alpha_data})
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # 如果还有待处理的Alpha，立即启动新任务
                    if alpha_queue:
                        new_alpha_data = alpha_queue.pop(0)
                        new_future = executor.submit(start_single_simulation, new_alpha_data)
                        running_futures[new_future] = new_alpha_data
                    
                    # 更新进度条描述显示深度、迭代次数和当前并发数
                    iteration_info = f"迭代{iteration}" if iteration > 0 else "初始化"
                    pbar.set_description(f"正在动态并发模拟Alpha (深度{depth}, {iteration_info}, 当前并发: {len(running_futures)})")
    
    # 第二阶段：并发获取统计数据
    valid_results = [r for r in simulation_results if r['alpha_id'] is not None]
    
    if valid_results:
        def get_stats_for_alpha(sim_result):
            try:
                return get_specified_alpha_stats(
                    s, sim_result['alpha_id'], sim_result['simulate_data'], **simulation_config
                )
            except Exception as e:
                print(f"警告: 获取Alpha统计数据失败: {e}")
                return {
                    'alpha_id': sim_result['alpha_id'],
                    'simulate_data': sim_result['simulate_data'],
                    'is_stats': None,
                    'pnl': None,
                    'stats': None,
                    'is_tests': None
                }
        
        with ThreadPool(limit_of_concurrent_simulations) as pool:
            with tqdm.tqdm(total=len(valid_results), desc="正在获取Alpha统计数据") as pbar:
                for result in pool.imap_unordered(get_stats_for_alpha, valid_results):
                    result_list.append(result)
                    pbar.update()
    
    # 为失败的Alpha添加空结果
    failed_results = [r for r in simulation_results if r['alpha_id'] is None]
    for failed_result in failed_results:
        result_list.append({
            'alpha_id': None,
            'simulate_data': failed_result['simulate_data'],
            'is_stats': None,
            'pnl': None,
            'stats': None,
            'is_tests': None
        })

    return result_list


def simulate_alpha_list_multi(
    s: requests.Session,
    alpha_list: list[dict],
    limit_of_concurrent_simulations: int = 3,
    limit_of_multi_simulations: int = 3,
    simulation_config: dict = DEFAULT_CONFIG,
) -> list[dict]:
    """
    模拟 Alpha 列表，使用多 Alpha 批处理模拟功能。

    参数:
        s (requests.Session): 已认证的请求会话对象。
        alpha_list (list[dict]): 包含 Alpha 模拟数据字典的列表。
        limit_of_concurrent_simulations (int): 并发批处理模拟的数量限制。
        limit_of_multi_simulations (int): 每个批次中包含的 Alpha 数量（2到10之间）。
        simulation_config (dict): 模拟配置字典。

    返回:
        list[dict]: 包含每个 Alpha 模拟结果（包括 ID、数据、统计、PnL、测试等）的列表。
    """
    if (limit_of_multi_simulations < 2) or (limit_of_multi_simulations > 10):
        print('警告: 多模拟的数量限制应在 2 到 10 之间。')
        limit_of_multi_simulations = 3 # 强制设为默认值

    if len(alpha_list) < 10: # 经验值，如果 Alpha 数量太少，多模拟可能效率不高
        print('警告: Alpha 列表太短，将使用单并发模拟代替多模拟。')
        return simulate_alpha_list(s, alpha_list, simulation_config=simulation_config, limit_of_concurrent_simulations=limit_of_concurrent_simulations, depth=1, iteration=0)

    # 将 Alpha 列表分成多个批次（任务）
    tasks = [alpha_list[i:i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)]
    result_list = []

    # 第一阶段：并发启动批处理模拟并获取所有子 Alpha ID
    with ThreadPool(limit_of_concurrent_simulations) as pool:
        with tqdm.tqdm(total=len(tasks), desc="正在启动多Alpha批处理模拟") as pbar:
            for result in pool.imap_unordered(
                partial(simulate_multi_alpha, s), tasks
            ):
                result_list.append(result)
                pbar.update()
    
    # 扁平化结果列表（从列表的列表变为单一列表）
    result_list_flat = [item for sublist in result_list for item in sublist]

    stats_list_result = []
    # 第二阶段：并发获取所有 Alpha 的统计数据和测试结果
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(limit_of_concurrent_simulations) as pool: # 建议这里也用limit_of_concurrent_simulations
        with tqdm.tqdm(total=len(result_list_flat), desc="正在获取Alpha统计数据") as pbar:
            for result in pool.imap_unordered(func, result_list_flat): # 使用 imap_unordered 保持与进度条一致
                stats_list_result.append(result)
                pbar.update()

    return stats_list_result


# %%
def main() -> requests.Session:
    """
    主函数，用于启动会话。
    """
    s = start_session()
    return s

# 全局会话对象
s: requests.Session = main()

# %%
# # 深度为一的树结构生成
# # 该部分为生成遗传编程中的树结构，用于构建 Alpha 表达式

class Node:
    """
    树节点的定义。
    每个节点包含一个值，以及指向左子节点和右子节点的引用。
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def depth_one_trees(terminal_values: list[str], binary_ops: list[str], ts_ops: list[str], ts_ops_values: list[str], unary_ops: list[str], flag: int) -> Node:
    """
    生成深度为 1 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成二元操作符或时间序列操作符的树。
    - 如果 flag 为 0: 生成一个二元操作符作为根节点，其左右子节点为终端值。
    - 如果 flag 为 1: 生成一个时间序列操作符作为根节点，其左子节点为终端值，右子节点为时间序列操作的参数值。

    参数:
        terminal_values (list[str]): 可用的终端值（如 "close", "open"）。
        binary_ops (list[str]): 可用的二元操作符（如 "add", "subtract"）。
        ts_ops (list[str]): 可用的时间序列操作符（如 "ts_zscore", "ts_rank"）。
        ts_ops_values (list[str]): 可用的时间序列操作符参数值（如 "20", "40"）。
        unary_ops (list[str]): 可用的一元操作符（未在深度 1 树中直接使用）。
        flag (int): 控制树类型的标志 (0 为二元操作，1 为时间序列操作)。

    返回:
        Node: 生成的深度为 1 的树的根节点。
    """
    if (flag == 0):
        node = Node(random.choice(binary_ops))
        node.left = Node(random.choice(terminal_values))
        node.right = Node(random.choice(terminal_values))
        return node
    if (flag == 1):
        node = Node(random.choice(ts_ops))
        node.left = Node(random.choice(terminal_values))
        node.right = Node(random.choice(ts_ops_values))
        return node

# 定义 Alpha 表达式的构建块
terminal_values = ["close", "open", "high", "low", "vwap", "adv20", "volume", "cap", "returns", "dividend"]
ts_ops = ["ts_zscore", "ts_rank", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_delta", "ts_ir", "ts_mean","ts_median", "ts_product", "ts_std_dev"]
binary_ops = ["add", "subtract", "divide", "multiply", "max", "min"]
ts_ops_values = ["20", "40", "60", "120", "240"]
unary_ops = ["rank", "zscore", "winsorize", "normalize", "rank_by_side", "sigmoid", "pasteurize", "log"]

# 生成 100 个深度为 1 的随机树
one_depth_tree = []
for i in range(100):
    flag = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
    one_tree = depth_one_trees(terminal_values, binary_ops,ts_ops,ts_ops_values, unary_ops, flag)
    one_depth_tree.append(one_tree)

# %%
# # 树可视化工具（如果需要，请安装 graphviz 库）

# def generate_dot_tree(node, dot, parent_id=""):
#     """
#     递归地生成 Graphviz DOT 语言的树结构。
#     此函数未直接在代码中调用，但可用于可视化。
#     """
#     if node:
#         current_id = str(id(node)) # 使用节点对象的内存地址作为唯一ID
#         dot.node(current_id, label=str(node.value)) # 添加节点

#         if parent_id:
#             dot.edge(parent_id, current_id) # 添加父子节点之间的边

#         # 递归处理左右子节点
#         generate_dot_tree(node.left, dot, current_id)
#         generate_dot_tree(node.right, dot, current_id)

# def display_tree_with_graphviz(node):
#     """
#     使用 Graphviz 可视化树结构并保存为 PNG 文件。
#     此函数未直接在代码中调用，但可用于可视化。
#     需要安装 graphviz 库和 Graphviz 软件。
#     """
#     dot = graphviz.Digraph(comment="Genetic Programming Tree", format="png")
#     generate_dot_tree(node, dot)
#     # 渲染树并保存到文件，cleanup=True 会删除中间的 .dot 文件
#     dot.render("genetic_programming_tree", format="png", cleanup=True)

# %%
def depth_two_tree(tree1: Node, tree2: Node, ts_ops_values: list[str], ts_ops: list[str], flag: int) -> Node:
    """
    生成深度为 2 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成二元操作符或时间序列操作符的树。
    - 如果 flag 为 0: 生成一个二元操作符作为根节点，其左右子节点是输入的深度为 1 的树。
    - 如果 flag 为 1: 生成一个时间序列操作符作为根节点，其左子节点是输入的一个随机深度为 1 的树，
                     右子节点为时间序列操作的参数值。

    参数:
        tree1 (Node): 第一个深度为 1 的树的根节点。
        tree2 (Node): 第二个深度为 1 的树的根节点。
        ts_ops_values (list[str]): 可用的时间序列操作符参数值。
        ts_ops (list[str]): 可用的时间序列操作符。
        flag (int): 控制树类型的标志 (0 为二元操作，1 为时间序列操作)。

    返回:
        Node: 生成的深度为 2 的树的根节点。
    """
    if (flag == 0):
        node = Node(random.choice(binary_ops))
        node.left = tree1
        node.right = tree2
        return node
    if (flag == 1):
        node = Node(random.choice(ts_ops))
        node.left = random.choice([tree1, tree2]) # 随机选择一个深度1树作为左子节点
        node.right = Node(random.choice(ts_ops_values))
        return node

# %%
# 生成 100 个深度为 2 的随机树
tree_two = []
for i in range(100):
    jhanda = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
    tree1 = random.choice(one_depth_tree) # 从深度 1 树列表中随机选择
    tree2 = random.choice(one_depth_tree) # 从深度 1 树列表中随机选择
    tree22 = depth_two_tree(tree1, tree2, ts_ops_values, ts_ops, jhanda) # 传入 jhanda 作为 flag
    tree_two.append(tree22)

# %%
def depth_three_tree(tree2: list[Node], flag: int) -> Node:
    """
    生成深度为 3 的随机 Alpha 表达式树。

    根据 'flag' 的值，生成不同类型的树：
    - 如果 flag 为 0: 生成一个一元操作符作为根节点，其左子节点是输入的深度为 2 的树。
    - 如果 flag 为 1: 生成一个二元操作符作为根节点，其左右子节点是输入的深度为 2 的树。
    - 如果 flag 为 2: 生成一个时间序列操作符作为根节点，其左子节点是输入的深度为 2 的树，
                     右子节点为时间序列操作的参数值。

    参数:
        tree2 (list[Node]): 包含深度为 2 的树的列表，将从中随机选择作为子节点。
        flag (int): 控制树类型的标志 (0 为一元操作，1 为二元操作，2 为时间序列操作)。

    返回:
        Node: 生成的深度为 3 的树的根节点。
    """
    if flag == 0 :
        node = Node(random.choice(unary_ops))
        node.left = random.choice(tree2)
        node.right = None # 一元操作符没有右子节点
        return node
    if flag == 1 :
        node = Node(random.choice(binary_ops))
        node.left = random.choice(tree2)
        node.right =  random.choice(tree2)
        return node
    if flag == 2 :
        node = Node(random.choice(ts_ops))
        node.left = random.choice(tree2)
        node.right =  Node(random.choice(ts_ops_values))
        return node

# 生成 100 个深度为 3 的随机树
tree3 = []
for i in range(100):
    f = random.choice([0,1,2]) # 随机选择生成一元、二元或时间序列操作符树
    tree33 = depth_three_tree(tree_two, f) # 传入 f 作为 flag
    tree3.append(tree33)


# %%
# # 树结构转换为 Alpha 表达式字符串

def d1tree_to_alpha(tree: Node) -> str:
    """
    将深度为 1 的树结构转换为 Alpha 表达式字符串。
    格式: operation(operand1,operand2) 或 ts_operation(operand1,ts_param)
    此函数仅适用于 depth_one_trees 生成的特定结构。
    """
    return f"{tree.value}({tree.left.value},{tree.right.value})"

def d2tree_to_alpha(tree: Node) -> str:
    """
    将深度为 2 的树结构转换为 Alpha 表达式字符串。
    此函数仅适用于 depth_two_tree 生成的特定结构。
    假定子节点是深度 1 的树结构，即 `tree.left` 和 `tree.right` 都有 `left` 和 `right` 属性。
    """
    if tree.value in binary_ops:
        # 如果是二元操作符，左右子节点都是深度1的树
        # 格式: op(d1_tree_left, d1_tree_right)
        return f"{tree.value}({tree.left.value}({tree.left.left.value},{tree.left.right.value}),{tree.right.value}({tree.right.left.value},{tree.right.right.value}))"
    if tree.value in ts_ops:
        # 如果是时间序列操作符，左子节点是深度1的树，右子节点是参数值
        # 格式: ts_op(d1_tree_left, ts_param)
        return f"{tree.value}({tree.left.value}({tree.left.left.value},{tree.left.right.value}),{tree.right.value})"

def d3tree_to_alpha(tree: Node) -> str:
    """
    将深度为 3 的树结构转换为 Alpha 表达式字符串。
    此函数根据根节点和其子节点的类型（深度 2 树的类型）进行复杂的模式匹配。
    这部分逻辑非常具体和硬编码，如果树结构有变化，可能需要大量修改。
    """
    # 根节点是二元操作符，且左右子节点都是深度2的二元操作符树
    if tree.value in binary_ops and tree.left.value in binary_ops and tree.right.value in binary_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}({tree.right.right.left.value},{tree.right.right.right.value})))"
    # 根节点是二元操作符，且左右子节点都是深度2的时间序列操作符树
    if tree.value in binary_ops and tree.left.value in ts_ops and tree.right.value in ts_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}))"
    # 根节点是二元操作符，左子节点是深度2的二元操作符树，右子节点是深度2的时间序列操作符树
    if tree.value in binary_ops and tree.left.value in binary_ops and tree.right.value in ts_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}))"
    # 根节点是二元操作符，左子节点是深度2的时间序列操作符树，右子节点是深度2的二元操作符树
    if tree.value in binary_ops and tree.left.value in ts_ops and tree.right.value in binary_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),({tree.right.value}({tree.right.left.value}({tree.right.left.left.value},{tree.right.left.right.value}),{tree.right.right.value}({tree.right.right.left.value},{tree.right.right.right.value}))))"
    # 根节点是时间序列操作符，其左子节点是深度2的二元操作符树
    if tree.value in ts_ops and tree.left.value in binary_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})),{tree.right.value})"
    # 根节点是时间序列操作符，其左子节点是深度2的时间序列操作符树
    if tree.value in ts_ops and tree.left.value in ts_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}),{tree.right.value})"
    # 根节点是一元操作符，其左子节点是深度2的二元操作符树
    if  tree.value in unary_ops and tree.left.value in binary_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}({tree.left.right.left.value},{tree.left.right.right.value})))"
    # 根节点是一元操作符，其左子节点是深度2的时间序列操作符树
    if tree.value in unary_ops and tree.left.value in ts_ops:
        return f"{tree.value}({tree.left.value}({tree.left.left.value}({tree.left.left.left.value},{tree.left.left.right.value}),{tree.left.right.value}))"
    return "" # 如果不匹配任何已知模式，返回空字符串

# %%
# 将深度 3 树转换为 Alpha 字符串
t3 = []
for i in range(100):
    # 确保 tree3[i] 不是 None，并且 d3tree_to_alpha 返回非 None 值
    if tree3[i] is not None:
        r = d3tree_to_alpha(tree3[i])
        if r: # 确保 r 不是空字符串
            t3.append(r)

# %%
def fitness_fun(Data: pd.DataFrame, n: int) -> list[str]:
    """
    根据 Alpha 模拟结果计算适应度得分，并返回前 N 个最佳 Alpha 的表达式。
    适应度函数: (sharpe * fitness * returns) / ((drawdown * turnover^2) + 0.001)

    参数:
        Data (pd.DataFrame): 包含 Alpha 模拟结果的 DataFrame (由 prettify_result 生成)。
        n (int): 要选择的最佳 Alpha 数量。

    返回:
        list[str]: 包含前 N 个最佳 Alpha 表达式的列表。
    """
    # 确保所需列存在，避免计算错误
    required_cols = ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover', 'expression']
    for col in required_cols:
        if col not in Data.columns:
            print(f"错误: 缺失 '{col}' 列，无法计算适应度。")
            return [] # 返回空列表或根据需要处理错误

    # 替换可能导致除以零或异常值的零或 NaN
    # 将 turnover 为零的替换为一个非常小的值，防止除以零
    Data['turnover'] = Data['turnover'].replace(0, 0.0001)
    # 确保 drawdown 不为零，并且是正数（通常 drawdown 是负值或正值表示亏损百分比）
    # 如果 drawdown 是负值，取其绝对值，如果为0，则给一个很小的正值
    Data['drawdown'] = Data['drawdown'].abs().replace(0, 0.0001)
    
    # 过滤掉缺失关键统计数据的行
    Data_filtered = Data.dropna(subset=['sharpe', 'fitness', 'returns', 'drawdown', 'turnover']).copy()
    
    # 过滤掉非数字的列值，并转换为数值类型
    for col in ['sharpe', 'fitness', 'returns', 'drawdown', 'turnover']:
        Data_filtered[col] = pd.to_numeric(Data_filtered[col], errors='coerce').fillna(0) # 将无法转换的设为0

    # 重新计算适应度列
    Data_filtered['fitness_column'] = (
        Data_filtered['sharpe'] * Data_filtered['fitness'] * Data_filtered['returns']
    ) / (
        (Data_filtered['drawdown'] * Data_filtered['turnover']**2) + 0.001
    )

    # 过滤掉无穷大或 NaN 的适应度值
    Data_filtered = Data_filtered.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=['fitness_column'])


    # 排除所有 fitness_column 为 0 的行，或根据需求调整
    # Data_filtered = Data_filtered[Data_filtered['fitness_column'] != 0]

    # 按适应度降序排序
    Data_filtered = Data_filtered.sort_values(by='fitness_column', ascending=False)
    
    # 获取前 N 个最佳 Alpha 的表达式
    top_n_values = Data_filtered.head(n)['expression'].tolist()
    return top_n_values

# %%
# # Alpha 表达式字符串转换为树结构

import re

def parse_expression(listr: list[str]) -> list[list[str]]:
    """
    解析 Alpha 表达式字符串列表，提取操作符和操作数。
    例如: "add(close,open)" -> ["add", "close", "open"]

    参数:
        listr (list[str]): Alpha 表达式字符串列表。

    返回:
        list[list[str]]: 包含每个表达式解析后组件的列表的列表。
    """
    arr = []
    # 匹配单词字符（包括字母、数字、下划线），数字，或括号
    pattern = re.compile(r'(\w+|[+\-*/]|ts_arg_max|ts_arg_min|ts_backfill|ts_delta|ts_ir|ts_mean|ts_median|ts_product|ts_std_dev|ts_rank|ts_zscore|rank|zscore|winsorize|normalize|rank_by_side|sigmoid|pasteurize|log|[0-9]+|\(|\))')
    for i in range(len(listr)):
        # 确保 listr[i] 是字符串类型
        if not isinstance(listr[i], str):
            continue
        matches = pattern.findall(listr[i])
        # 过滤掉空字符串和括号
        components = [match for match in matches if match and match != '(' and match != ')']
        arr.append(components)
    return arr


def d1_alpha_to_tree(alphas: list[str]) -> list[Node]:
    """
    将深度为 1 的 Alpha 表达式字符串列表转换为树结构列表。
    """
    trees = []
    ary = parse_expression(alphas)
    for i in range(len(ary)):
        alp = ary[i]
        # 确保 alp 长度足够，至少包含操作符和两个操作数
        if len(alp) >= 3:
            node = Node(alp[0]) # 根节点是操作符
            node.left = Node(alp[1]) # 左子节点
            node.right = Node(alp[2]) # 右子节点
            trees.append(node)
        else:
            print(f"警告: 无法解析深度 1 Alpha: {alphas[i]}，跳过。")
    return trees

def d2_alpha_to_tree(alpha_list: list[str]) -> list[Node]:
    """
    将深度为 2 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数仅适用于 depth_two_tree 生成的特定结构。
    """
    trees = []
    ary = parse_expression(alpha_list)
    for i in range (len(ary)):
        ar = ary[i]
        # 深度为 2 的时间序列操作符树: ts_op(d1_tree_left, ts_param) => [ts_op, op_d1, val_d1_left, val_d1_right, ts_param] (5 elements)
        if ar[0] in ts_ops and len(ar) >= 5:
            node = Node(ar[0])
            node.left = Node(ar[1]) # 深度 1 树的根节点
            node.left.left = Node(ar[2]) # 深度 1 树的左子节点
            node.left.right = Node(ar[3]) # 深度 1 树的右子节点
            node.right = Node(ar[4]) # 时间序列参数
            trees.append(node)
        # 深度为 2 的二元操作符树: op(d1_tree_left, d1_tree_right) => [op, op_d1_left, val_d1_left_left, val_d1_left_right, op_d1_right, val_d1_right_left, val_d1_right_right] (7 elements)
        elif ar[0] in binary_ops and len(ar) >= 7:
            node = Node(ar[0])
            node.left = Node(ar[1])
            node.left.left = Node(ar[2])
            node.left.right = Node(ar[3])
            node.right = Node(ar[4])
            node.right.left = Node(ar[5])
            node.right.right = Node(ar[6])
            trees.append(node)
        else:
            print(f"警告: 无法解析深度 2 Alpha: {''.join(ar)}，跳过。")
    return trees

def d3_alpha_to_tree(alpha_list: list[str]) -> list[Node]:
    """
    将深度为 3 的 Alpha 表达式字符串列表转换为树结构列表。
    此函数也高度依赖硬编码的模式匹配，可能需要确保输入表达式的结构严格符合预期。
    """
    trees = []
    ary = parse_expression(alpha_list)
    for i in range (len(ary)):
        ar = ary[i]

        # 根节点是一元操作符 (rank, zscore, winsorize, etc.)
        if ar[0] in unary_ops:
            # 左子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # Pattern: Unary(TS_op(d1_op(d1_left, d1_right), ts_param))
            if ar[1] in ts_ops and len(ar) >= 6: # ar[0]=unary, ar[1]=ts_op, ar[2]=d1_op, ar[3]=d1_left, ar[4]=d1_right, ar[5]=ts_param
                node = Node(ar[0])
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # This expects ar[2] to be the d1_op, but it could be the first terminal value if d2 structure simplified.
                                             # This is where the manual parsing could break easily.
                                             # Based on d2tree_to_alpha, d1_op's children are ar[2] and ar[3]
                if ar[2] in terminal_values: # This branch assumes direct terminal values after ts_op which is not how d2 operates.
                                             # Assuming ar[2] is the d1_left and ar[3] is d1_right and ar[4] is the ts_param from the d1 tree.
                                             # This is messy. Let's try to infer from the `d2tree_to_alpha` structure for a nested `d1` tree.
                                             # d2 ts_op: [ts_op, d1_op, d1_val1, d1_val2, ts_param_d2]
                                             # So, ar[1] = d2_op (ts_op), ar[2] = d1_op, ar[3] = d1_val1, ar[4] = d1_val2, ar[5] = ts_param_d2
                    node.left.left = Node(ar[3]) # d1_op left
                    node.left.right = Node(ar[4]) # d1_op right
                    node.left.right = Node(ar[5]) # d2 ts_param
                    trees.append(node)
                else: # The ar[2] is the d1 operator (like 'multiply')
                    node.left.left = Node(ar[2]) # The d1 tree root (e.g., 'multiply')
                    if len(ar) >= 5 and ar[3] and ar[4]:
                        node.left.left.left = Node(ar[3]) # d1_op left (e.g., 'high')
                        node.left.left.right = Node(ar[4]) # d1_op right (e.g., 'low')
                    if len(ar) >= 6:
                        node.left.right = Node(ar[5]) # d2 ts_param (e.g., '60')
                    trees.append(node)
            # 左子节点是二元操作符 (binary_ops) 组成的深度2树
            # Pattern: Unary(Binary_op(d1_tree_left, d1_tree_right))
            elif ar[1] in binary_ops and len(ar) >= 8: # ar[0]=unary, ar[1]=bin_op_d2, ar[2]=d1_op_left, ar[3]=d1_left_val1, ar[4]=d1_left_val2, ar[5]=d1_op_right, ar[6]=d1_right_val1, ar[7]=d1_right_val2
                node = Node(ar[0])
                node.left = Node(ar[1]) # The d2 binary operator (e.g., 'add')
                node.left.left = Node(ar[2]) # The left d1 tree root (e.g., 'multiply')
                if len(ar) >= 4 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3]) # d1_left_op left (e.g., 'close')
                    node.left.left.right = Node(ar[4]) # d1_left_op right (e.g., 'open')
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # The right d1 tree root (e.g., 'divide')
                    if len(ar) >= 8 and ar[6] and ar[7]:
                        node.left.right.left = Node(ar[6]) # d1_right_op left (e.g., 'high')
                        node.left.right.right = Node(ar[7]) # d1_right_op right (e.g., 'low')
                trees.append(node)
            else:
                print(f"警告: 无法解析深度 3 一元操作符 Alpha (结构1): {''.join(ar)}，跳过。")

        # 根节点是时间序列操作符 (ts_ops)
        elif ar[0] in ts_ops:
            # 左子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # Pattern: TS_op(TS_op(d1_op(d1_left, d1_right), ts_param_d2), ts_param_d3)
            if ar[1] in ts_ops and len(ar) >= 7: # ar[0]=ts_op_d3, ar[1]=ts_op_d2, ar[2]=d1_op, ar[3]=d1_left, ar[4]=d1_right, ar[5]=ts_param_d2, ar[6]=ts_param_d3
                node = Node(ar[0])
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # d1_op
                if len(ar) >= 5 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3]) # d1_op left
                    node.left.left.right = Node(ar[4]) # d1_op right
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # ts_param_d2
                if len(ar) >= 7 and ar[6]:
                    node.right = Node(ar[6]) # ts_param_d3
                trees.append(node)
            # 左子节点是二元操作符 (binary_ops) 组成的深度2树
            # Pattern: TS_op(Binary_op(d1_tree_left, d1_tree_right), ts_param_d3)
            elif ar[1] in binary_ops and len(ar) >= 9: # ar[0]=ts_op_d3, ar[1]=bin_op_d2, ar[2]=d1_op_left, ar[3]=d1_left_val1, ar[4]=d1_left_val2, ar[5]=d1_op_right, ar[6]=d1_right_val1, ar[7]=d1_right_val2, ar[8]=ts_param_d3
                node = Node(ar[0])
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # left d1_op
                if len(ar) >= 4 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3]) # left d1_op left
                    node.left.left.right = Node(ar[4]) # left d1_op right
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # right d1_op
                    if len(ar) >= 8 and ar[6] and ar[7]:
                        node.left.right.left = Node(ar[6]) # right d1_op left
                        node.left.right.right = Node(ar[7]) # right d1_op right
                if len(ar) >= 9 and ar[8]:
                    node.right = Node(ar[8]) # ts_param_d3
                trees.append(node)
            else:
                print(f"警告: 无法解析深度 3 时间序列操作符 Alpha (结构2): {''.join(ar)}，跳过。")

        # 根节点是二元操作符 (binary_ops)
        elif ar[0] in binary_ops:
            # 左子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # 右子节点是二元操作符 (binary_ops) 组成的深度2树
            # Pattern: Binary(TS_op(d1_tree, ts_param), Binary_op(d1_tree_left, d1_tree_right))
            if ar[1] in ts_ops and ar[6] in binary_ops and len(ar) >= 13:
                node = Node(ar[0])
                # 左子树 (深度2 TS_op)
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # d1_op
                if len(ar) >= 5 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3])
                    node.left.left.right = Node(ar[4])
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # ts_param_d2
                # 右子树 (深度2 Binary_op)
                node.right = Node(ar[6])
                node.right.left = Node(ar[7]) # left d1_op
                if len(ar) >= 10 and ar[8] and ar[9]:
                    node.right.left.left = Node(ar[8])
                    node.right.left.right = Node(ar[9])
                if len(ar) >= 11 and ar[10]:
                    node.right.right = Node(ar[10]) # right d1_op
                    if len(ar) >= 13 and ar[11] and ar[12]:
                        node.right.right.left = Node(ar[11])
                        node.right.right.right = Node(ar[12])
                trees.append(node)
            # 左子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # 右子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # Pattern: Binary(TS_op(d1_tree, ts_param), TS_op(d1_tree, ts_param))
            elif ar[1] in ts_ops and ar[6] in ts_ops and len(ar) >= 11:
                node = Node(ar[0])
                # 左子树 (深度2 TS_op)
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # d1_op
                if len(ar) >= 5 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3])
                    node.left.left.right = Node(ar[4])
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # ts_param_d2_left
                # 右子树 (深度2 TS_op)
                node.right = Node(ar[6])
                node.right.left = Node(ar[7]) # d1_op
                if len(ar) >= 10 and ar[8] and ar[9]:
                    node.right.left.left = Node(ar[8])
                    node.right.left.right = Node(ar[9])
                if len(ar) >= 11 and ar[10]:
                    node.right.right = Node(ar[10]) # ts_param_d2_right
                trees.append(node)
            # 左右子节点都是二元操作符 (binary_ops) 组成的深度2树
            # Pattern: Binary(Binary_op(d1_tree_left, d1_tree_right), Binary_op(d1_tree_left, d1_tree_right))
            elif ar[1] in binary_ops and ar[8] in binary_ops and len(ar) >= 15:
                node = Node(ar[0])
                # 左子树 (深度2 Binary_op)
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # left d1_op
                if len(ar) >= 5 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3])
                    node.left.left.right = Node(ar[4])
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # right d1_op
                    if len(ar) >= 8 and ar[6] and ar[7]:
                        node.left.right.left = Node(ar[6])
                        node.left.right.right = Node(ar[7])
                # 右子树 (深度2 Binary_op)
                node.right = Node(ar[8])
                node.right.left = Node(ar[9]) # left d1_op
                if len(ar) >= 12 and ar[10] and ar[11]:
                    node.right.left.left = Node(ar[10])
                    node.right.left.right = Node(ar[11])
                if len(ar) >= 13 and ar[12]:
                    node.right.right = Node(ar[12]) # right d1_op
                    if len(ar) >= 15 and ar[13] and ar[14]:
                        node.right.right.left = Node(ar[13])
                        node.right.right.right = Node(ar[14])
                trees.append(node)
            # 左子节点是二元操作符 (binary_ops) 组成的深度2树
            # 右子节点是时间序列操作符 (ts_ops) 组成的深度2树
            # Pattern: Binary(Binary_op(d1_tree_left, d1_tree_right), TS_op(d1_tree, ts_param))
            elif ar[1] in binary_ops and ar[8] in ts_ops and len(ar) >= 13:
                node = Node(ar[0])
                # 左子树 (深度2 Binary_op)
                node.left = Node(ar[1])
                node.left.left = Node(ar[2]) # left d1_op
                if len(ar) >= 5 and ar[3] and ar[4]:
                    node.left.left.left = Node(ar[3])
                    node.left.left.right = Node(ar[4])
                if len(ar) >= 6 and ar[5]:
                    node.left.right = Node(ar[5]) # right d1_op
                    if len(ar) >= 8 and ar[6] and ar[7]:
                        node.left.right.left = Node(ar[6])
                        node.left.right.right = Node(ar[7])
                # 右子树 (深度2 TS_op)
                node.right = Node(ar[8])
                node.right.left = Node(ar[9]) # d1_op
                if len(ar) >= 12 and ar[10] and ar[11]:
                    node.right.left.left = Node(ar[10])
                    node.right.left.right = Node(ar[11])
                if len(ar) >= 13 and ar[12]:
                    node.right.right = Node(ar[12]) # ts_param_d2_right
                trees.append(node)
            else:
                print(f"警告: 无法解析深度 3 二元操作符 Alpha (结构3): {''.join(ar)}，跳过。")
        else:
            print(f"警告: 未知根操作符或不完整表达式: {''.join(ar)}，跳过。")
    return trees

# %%
# # 遗传算法的核心操作：复制、变异、交叉

def copy_tree(original_node: Optional[Node]) -> Optional[Node]:
    """
    递归地复制一个树结构。
    """
    if original_node is None:
        return None

    new_node = Node(original_node.value)
    new_node.left = copy_tree(original_node.left)
    new_node.right = copy_tree(original_node.right)
    return new_node

def mutate_random_node(original_node: Node, terminal_values: list[str], unary_ops: list[str], binary_ops: list[str], ts_ops: list[str], ts_ops_values: list[str]) -> Node:
    """
    随机变异树中的一个节点。

    参数:
        original_node (Node): 原始树的根节点。
        terminal_values (list[str]): 可用的终端值。
        unary_ops (list[str]): 可用的一元操作符。
        binary_ops (list[str]): 可用的二元操作符。
        ts_ops (list[str]): 可用的时间序列操作符。
        ts_ops_values (list[str]): 可用的时间序列操作符参数值。

    返回:
        Node: 变异后的树的根节点。
    """
    mutated_tree = copy_tree(original_node)
    mutation_probability = 0.5 # 节点变异的概率

    def mutate(node):
        if node is None:
            return
        
        # 随机决定是否变异当前节点
        if random.random() < mutation_probability:
            if isinstance(node.value, str):
                if node.value in binary_ops:
                    node.value = random.choice(binary_ops)
                elif node.value in ts_ops:
                    node.value = random.choice(ts_ops)
                elif node.value in ts_ops_values:
                    node.value = random.choice(ts_ops_values)
                elif node.value in unary_ops:
                    node.value = random.choice(unary_ops)
                elif node.value in terminal_values:
                    node.value = random.choice(terminal_values)
            # 如果节点值不是已知操作符/终端值，或者上面已经变异，则不再深入。
            # 如果未变异当前节点，则尝试变异其子节点。
            # 原始代码中存在一个逻辑问题：如果当前节点被变异，它还会尝试变异其子节点
            # 导致一个节点可能被多次随机变异，或子节点被变异而父节点未变异。
            # 更常见的做法是：如果当前节点被选中变异，则只变异它本身，不继续深入。
            # 如果未被选中变异，则递归地对子节点进行变异检查。
            return # 如果当前节点已变异，则停止递归，避免重复变异。

        # 如果当前节点未变异，则递归变异其子节点
        mutate(node.left)
        mutate(node.right)

    mutate(mutated_tree) # 从根节点开始变异过程

    return mutated_tree

def crossover(parent1: Node, parent2: Node, n: int) -> tuple[Node, Node]:
    """
    执行树结构之间的交叉操作，生成两个子树。
    注意: 这里的交叉实现非常特定，只在根节点的左右子节点之间进行交换，且依赖于树的深度。
    它不是一个通用的随机子树交换交叉。

    参数:
        parent1 (Node): 第一个父树的根节点。
        parent2 (Node): 第二个父树的根节点。
        n (int): 用于判断树深度的参数 (2 代表深度 2 树，3 代表深度 3 树)。

    返回:
        tuple[Node, Node]: 包含两个子树根节点的元组。
    """
    # 深度复制父树以创建子树
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)

    # 确保父节点存在且有左右子节点，避免 AttributeError
    if child1 is None or child2 is None or child1.left is None or child1.right is None or child2.left is None or child2.right is None:
        return child1, child2 # 如果结构不符合预期，直接返回副本

    if n == 2: # 针对深度 2 的树进行交叉
        # 针对二元操作符作为根节点的树
        if child1.value in binary_ops and child2.value in binary_ops:
            side =  random.choice(['R','L']) # 随机选择交换左子树还是右子树
            same = random.choice(['Y','N']) # 随机选择是否在同侧交换（Y）或异侧交换（N）
            if side == 'L' and same == 'Y':
                z = child1.left
                child1.left = child2.left
                child2.left = z
                return child1,child2
            if side == 'R' and same == 'Y':
                z = child1.right
                child1.right = child2.right
                child2.right = z
                return child1,child2
            if side == 'L' and same =='N': # 注意原代码 ' N' 有空格，应为 'N'
                z = child1.left
                child1.left = child2.right
                child2.right = z
                return child1,child2
            if side == 'R' and same == 'N':
                z = child1.right
                child1.right = child2.left
                child2.left = z
                return child1,child2
        # 针对时间序列操作符作为根节点的树
        if child1.value in ts_ops and child2.value in ts_ops:
            # 只有左子节点是树结构，交换左子节点
            z = child1.left
            child1.left = child2.left
            child2.left = z
            return child1,child2

    if n == 3: # 针对深度 3 的树进行交叉 (逻辑与深度 2 类似，可能需要更深的节点交换)
        # 针对二元操作符作为根节点的树
        if child1.value in binary_ops and child2.value in binary_ops:
            side =  random.choice(['R','L'])
            same = random.choice(['Y','N'])
            if side == 'L' and same == 'Y':
                z = child1.left
                child1.left = child2.left
                child2.left = z
                return child1,child2
            if side == 'R' and same == 'Y':
                z = child1.right
                child1.right = child2.right
                child2.right = z
                return child1,child2
            if side == 'L' and same =='N': # 注意原代码 ' N' 有空格，应为 'N'
                z = child1.left
                child1.left = child2.right
                child2.right = z
                return child1,child2
            if side == 'R' and same == 'N':
                z = child1.right
                child1.right = child2.left
                child2.left = z
                return child1,child2
        # 针对时间序列操作符作为根节点的树
        if child1.value in ts_ops and child2.value in ts_ops:
            # 只有左子节点是树结构，交换左子节点
            z = child1.left
            child1.left = child2.left
            child2.left = z
            return child1,child2
    
    # 如果不满足任何交叉条件，返回原始副本
    return child1, child2


def get_random_node(node: Node) -> Node:
    """
    从树中随机选择一个节点。
    此函数在提供的代码中未被使用，但通常用于通用遗传编程的交叉和变异操作。
    """
    nodes = []
    collect_nodes(node, nodes)
    return random.choice(nodes)

def collect_nodes(node: Optional[Node], nodes: list[Node]) -> None:
    """
    递归地收集树中的所有节点。
    此函数在提供的代码中未被使用，但通常用于通用遗传编程的交叉和变异操作。
    """
    if node:
        nodes.append(node)
        collect_nodes(node.left, nodes)
        collect_nodes(node.right, nodes)


# %%
# # 遗传算法迭代：深度 1 Alpha 的优化

def best_d1_alphas(n: int, m: int) -> list[str]:
    """
    使用遗传算法优化生成深度为 1 的最佳 Alpha。

    流程:
    1.  初始化种群: 随机生成 2*n 个深度为 1 的 Alpha 树，转换为表达式并去重。
    2.  模拟和评估: 模拟当前种群中的所有 Alpha，并使用 fitness_fun 评估其适应度。
    3.  选择: 根据适应度选择前 n 个最佳 Alpha 作为下一代的父本。
    4.  迭代演化: 重复 m 次以下步骤：
        a.  补充种群: 随机生成 n 个新的深度为 1 的 Alpha 树，添加到当前种群。
        b.  模拟和评估: 再次模拟并评估整个种群。
        c.  选择: 选择前 n 个最佳 Alpha。
    5.  变异: 对当前最佳 Alpha 树进行变异，生成新的 Alpha，重新评估并选择。

    参数:
        n (int): 每代保留的最佳 Alpha 数量（种群大小）。
        m (int): 遗传算法的迭代次数。

    返回:
        list[str]: 经过遗传算法优化后，表现最佳的深度为 1 的 Alpha 表达式列表。
    """
    population = []
    print("开始深度 1 Alpha 的初始父代选择。")

    # 初始种群生成：随机生成 2*n 个深度 1 的 Alpha
    for i in range(n * 2):
        current_flag = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
        d1_tree = depth_one_trees(terminal_values, binary_ops, ts_ops, ts_ops_values, unary_ops, current_flag)
        d1_alpha = d1tree_to_alpha(d1_tree)
        population.append(d1_alpha)
    
    # 去重并模拟评估初始种群
    k = list(OrderedDict.fromkeys(population)) # 去重
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=1, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
    
    # 检查 Data 是否为空，以避免 fitness_fun 错误
    if Data.empty:
        print("警告: 初始深度 1 Alpha 模拟未返回有效数据。")
        return []

    population = fitness_fun(Data, n) # 根据适应度选择最佳 n 个

    # 遗传算法迭代
    for j in range(m):
        for i in range(n):
            current_flag = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
            d1 = depth_one_trees(terminal_values, binary_ops, ts_ops, ts_ops_values, unary_ops, current_flag)
            d1a = d1tree_to_alpha(d1)
            population.append(d1a) # 将新生成的 Alpha 添加到种群中

        # 每轮迭代后重新评估整个种群
        k = list(OrderedDict.fromkeys(population))
        alpha_list = [generate_alpha(x) for x in k]
        cntx = simulate_alpha_list(s, alpha_list, depth=1, iteration=j+1)
        Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
        
        if Data.empty:
            print(f"警告: 深度 1 Alpha 第 {j+1} 轮迭代模拟未返回有效数据。")
            # 可以选择在这里返回当前最佳或继续，取决于错误容忍度
            continue 

        population = fitness_fun(Data, n) # 选择本轮最佳 n 个

    print("深度 1 Alpha 的初始父代选择完成，开始进行变异。")
    
    # 变异阶段：将当前最佳 Alpha 转换为树结构进行变异
    best_trees = d1_alpha_to_tree(population)
    mut = []
    # 先收集所有变异后的树，避免在迭代过程中修改列表
    mutated_trees = []
    for tree_node in best_trees:
        # 变异后的树收集到单独的列表中
        mutated_node = mutate_random_node(tree_node, terminal_values, unary_ops, binary_ops, ts_ops, ts_ops_values)
        if mutated_node: # 确保变异后的节点非空
            mutated_trees.append(mutated_node)
    
    # 将变异后的树添加到 best_trees 中
    best_trees.extend(mutated_trees)

    # 将所有树（包括原始最佳和变异后的）转换为 Alpha 表达式
    for tree_node in best_trees:
        try:
            ft = d1tree_to_alpha(tree_node)
            mut.append(ft)
        except AttributeError:
            print(f"警告: 转换深度 1 树到 Alpha 表达式时出错，跳过该树。树值: {tree_node.value}")
            continue

    mut = list(OrderedDict.fromkeys(mut)) # 再次去重

    # 模拟评估变异后的 Alpha
    k = mut
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=1, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)

    if Data.empty:
        print("警告: 深度 1 Alpha 变异阶段模拟未返回有效数据。")
        return population # 返回变异前的最佳种群

    population = fitness_fun(Data, n) # 最终选择最佳 n 个

    print("深度 1 Alpha 优化完成。")
    return population


# %%
# # 遗传算法迭代：深度 2 Alpha 的优化

def best_d2_alphas(onetree: list[str], n: int, m: int) -> list[str]:
    """
    使用遗传算法优化生成深度为 2 的最佳 Alpha。

    流程:
    1.  初始化种群: 根据深度 1 的最佳 Alpha 树，随机组合生成 2*n 个深度为 2 的 Alpha 树，转换为表达式并去重。
    2.  模拟和评估: 模拟当前种群中的所有 Alpha，并使用 fitness_fun 评估其适应度。
    3.  选择: 根据适应度选择前 n 个最佳 Alpha 作为下一代的父本。
    4.  迭代演化: 重复 m 次以下步骤：
        a.  补充种群: 随机生成 n 个新的深度为 2 的 Alpha 树，添加到当前种群。
        b.  模拟和评估: 再次模拟并评估整个种群。
        c.  选择: 选择前 n 个最佳 Alpha。
    5.  变异: 对当前最佳 Alpha 树进行变异，生成新的 Alpha，重新评估。
    6.  交叉: 对变异后的 Alpha 树进行交叉，生成新的 Alpha，重新评估。
    7.  最终选择: 合并变异和交叉后的最佳 Alpha，进行最终评估和选择。

    参数:
        onetree (list[str]): 深度 1 的最佳 Alpha 表达式列表，用于构建深度 2 的树。
        n (int): 每代保留的最佳 Alpha 数量（种群大小）。
        m (int): 遗传算法的迭代次数。

    返回:
        list[str]: 经过遗传算法优化后，表现最佳的深度为 2 的 Alpha 表达式列表。
    """
    population = []
    print("开始深度 2 Alpha 的初始父代选择。")
    # 将深度 1 的最佳 Alpha 表达式转换为树结构
    best_one_trees = d1_alpha_to_tree(onetree)
    
    if not best_one_trees:
        print("错误: 深度 1 最佳 Alpha 列表为空，无法生成深度 2 Alpha。")
        return []

    # 初始种群生成：随机组合深度 1 树生成 2*n 个深度 2 Alpha
    for i in range(n * 2):
        jhanda = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
        tree1 = random.choice(best_one_trees)
        tree2 = random.choice(best_one_trees)
        tree22 = depth_two_tree(tree1, tree2, ts_ops_values, ts_ops, jhanda) # 传入 jhanda 作为 flag
        d2_alpha = d2tree_to_alpha(tree22) # 转换为 Alpha 表达式
        population.append(d2_alpha)

    # 去重并模拟评估初始种群
    k = list(OrderedDict.fromkeys(population))
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=2, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
    
    if Data.empty:
        print("警告: 初始深度 2 Alpha 模拟未返回有效数据。")
        return []

    population = fitness_fun(Data, n) # 根据适应度选择最佳 n 个

    # 遗传算法迭代
    for j in range(m):
        for i in range(n):
            jhanda = random.choice([0,1]) # 随机选择生成二元操作符树或时间序列操作符树
            tree1 = random.choice(best_one_trees)
            tree2 = random.choice(best_one_trees)
            tree22 = depth_two_tree(tree1, tree2, ts_ops_values, ts_ops, jhanda)
            d2_alpha = d2tree_to_alpha(tree22)
            population.append(d2_alpha) # 将新生成的 Alpha 添加到种群中

        # 每轮迭代后重新评估整个种群
        k = list(OrderedDict.fromkeys(population))
        alpha_list = [generate_alpha(x) for x in k]
        cntx = simulate_alpha_list(s, alpha_list, depth=2, iteration=0)
        Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
         
        if Data.empty:
            print(f"警告: 深度 2 Alpha 第 {j+1} 轮迭代模拟未返回有效数据。")
            continue

        population = fitness_fun(Data, n) # 选择本轮最佳 n 个

    print("深度 2 Alpha 的初始父代选择完成，开始进行变异和交叉。")
    
    # 变异阶段
    best_trees = d2_alpha_to_tree(population)
    mut = []
    # 创建一个可修改的树列表，包含原始最佳树和它们的变异体
    all_trees_for_mutation = list(best_trees) # 复制一份，避免在迭代中修改被迭代的列表

    for tree_node in all_trees_for_mutation:
        mutated_node = mutate_random_node(tree_node, terminal_values, unary_ops, binary_ops, ts_ops, ts_ops_values)
        if mutated_node:
            best_trees.append(mutated_node) # 将变异体添加到 best_trees 中

    # 将所有树（原始最佳和变异后的）转换为 Alpha 表达式
    for tree_node in best_trees:
        try:
            ft = d2tree_to_alpha(tree_node)
            if ft: # 确保非空字符串
                mut.append(ft)
        except (AttributeError, TypeError):
            print(f"警告: 转换深度 2 变异树到 Alpha 表达式时出错，跳过该树。")
            continue

    mut = list(OrderedDict.fromkeys(mut)) # 去重
    
    # 模拟评估变异后的 Alpha
    k = mut
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=2, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
 
    if Data.empty:
        print("警告: 深度 2 Alpha 变异阶段模拟未返回有效数据。")
        d2m_population = population # 如果变异没有有效结果，则保留原始最佳
    else:
        d2m_population = fitness_fun(Data, n) # 变异后的最佳 Alpha

    # 交叉阶段
    cross = []
    # 使用 best_trees 进行交叉，注意索引范围
    # 确保 len(best_trees) >= 2 才能进行交叉
    if len(best_trees) >= 2:
        # 使用 len(best_trees) // 2 确保不会越界，或者更稳妥地使用 len(best_trees) - 1
        for i in range(0, len(best_trees) - 1, 2): # 每两个进行交叉
            parent1 = best_trees[i]
            parent2 = best_trees[i+1]
            try:
                a, b = crossover(parent1, parent2, 2) # n=2 表示深度 2 树的交叉规则
                if a:
                    best_trees.append(a)
                if b:
                    best_trees.append(b)
            except AttributeError:
                print(f"警告: 深度 2 交叉操作失败，跳过父代 {parent1.value}, {parent2.value}。")
                continue
    else:
        print("警告: 最佳深度 2 树数量不足，无法进行交叉操作。")

    # 将所有树（包括原始最佳、变异体和交叉后的）转换为 Alpha 表达式
    for tree_node in best_trees:
        try:
            ft = d2tree_to_alpha(tree_node)
            if ft:
                cross.append(ft)
        except (AttributeError, TypeError):
            # print(f"警告: 转换深度 2 交叉树到 Alpha 表达式时出错，跳过该树。")
            continue # 忽略错误的转换，继续处理下一个

    cross = list(OrderedDict.fromkeys(cross)) # 去重

    # 模拟评估交叉后的 Alpha
    k = cross
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=2, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
     
    if Data.empty:
        print("警告: 深度 2 Alpha 交叉阶段模拟未返回有效数据。")
        d2c_population = population # 如果交叉没有有效结果，则保留原始最佳
    else:
        d2c_population = fitness_fun(Data, n) # 交叉后的最佳 Alpha

    # 合并变异和交叉后的最佳种群，进行最终评估
    prefinal_d2_pop = []
    prefinal_d2_pop.extend(d2m_population) # 添加变异后的最佳
    prefinal_d2_pop.extend(d2c_population) # 添加交叉后的最佳

    prefinal_d2_pop = list(OrderedDict.fromkeys(prefinal_d2_pop)) # 最终去重
    
    k = prefinal_d2_pop
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=2, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
     
    if Data.empty:
        print("警告: 深度 2 Alpha 最终阶段模拟未返回有效数据。")
        return [] # 返回空列表
    
    best_depth_two_population = fitness_fun(Data, n) # 最终选择最佳 n 个

    print("深度 2 Alpha 优化完成。")
    return best_depth_two_population


# %%
# # 遗传算法迭代：深度 3 Alpha 的优化

def best_d3_alpha(best_depth_two_population: list[str], n: int, m: int) -> list[str]:
    """
    使用遗传算法优化生成深度为 3 的最佳 Alpha。

    流程:
    1.  初始化种群: 根据深度 2 的最佳 Alpha 树，随机组合生成 2*n 个深度为 3 的 Alpha 树，转换为表达式并去重。
    2.  模拟和评估: 模拟当前种群中的所有 Alpha，并使用 fitness_fun 评估其适应度。
    3.  选择: 根据适应度选择前 n 个最佳 Alpha 作为下一代的父本。
    4.  迭代演化: 重复 m 次以下步骤：
        a.  补充种群: 随机生成 n 个新的深度为 3 的 Alpha 树，添加到当前种群。
        b.  模拟和评估: 再次模拟并评估整个种群。
        c.  选择: 选择前 n 个最佳 Alpha。
    5.  变异: 对当前最佳 Alpha 树进行变异，生成新的 Alpha，重新评估。
    6.  （原代码中此处缺少交叉阶段，与深度2不一致，但为保持一致性，遵循原代码）
    7.  最终选择: 合并变异后的最佳 Alpha，进行最终评估和选择。

    参数:
        best_depth_two_population (list[str]): 深度 2 的最佳 Alpha 表达式列表，用于构建深度 3 的树。
        n (int): 每代保留的最佳 Alpha 数量（种群大小）。
        m (int): 遗传算法的迭代次数。

    返回:
        list[str]: 经过遗传算法优化后，表现最佳的深度为 3 的 Alpha 表达式列表。
    """
    popu = []
    print("开始深度 3 Alpha 的初始父代选择。")
    # 将深度 2 的最佳 Alpha 表达式转换为树结构
    d2 = d2_alpha_to_tree(best_depth_two_population)

    if not d2:
        print("错误: 深度 2 最佳 Alpha 列表为空，无法生成深度 3 Alpha。")
        return []

    # 初始种群生成：随机组合深度 2 树生成 2*n 个深度 3 Alpha
    for i in range(n * 2):
        f = random.choice([0,1,2]) # 随机选择生成一元、二元或时间序列操作符树
        tree33_node = depth_three_tree(d2, f) # 生成深度 3 树节点
        if tree33_node: # 确保树节点非空
            tree33_alpha = d3tree_to_alpha(tree33_node) # 转换为 Alpha 表达式
            if tree33_alpha: # 确保表达式非空
                popu.append(tree33_alpha)

    # 去重并模拟评估初始种群
    k = list(OrderedDict.fromkeys(popu))
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=3, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
     
    if Data.empty:
        print("警告: 初始深度 3 Alpha 模拟未返回有效数据。")
        return []

    popu = fitness_fun(Data, n) # 根据适应度选择最佳 n 个

    # 遗传算法迭代
    for j in range(m):
        for i in range(n):
            f = random.choice([0,1,2]) # 随机选择生成一元、二元或时间序列操作符树
            tree33_node = depth_three_tree(d2, f)
            if tree33_node:
                tree33_alpha = d3tree_to_alpha(tree33_node)
                if tree33_alpha:
                    popu.append(tree33_alpha)

        # 每轮迭代后重新评估整个种群
        k = list(OrderedDict.fromkeys(popu))
        alpha_list = [generate_alpha(x) for x in k]
        cntx = simulate_alpha_list(s, alpha_list, depth=3, iteration=0)
        Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
         
        if Data.empty:
            print(f"警告: 深度 3 Alpha 第 {j+1} 轮迭代模拟未返回有效数据。")
            continue

        popu = fitness_fun(Data, n) # 选择本轮最佳 n 个

    print("深度 3 Alpha 的初始父代选择完成，开始进行变异和交叉（注意：交叉部分在原 Notebook 中未明确实现）。")
    
    # 变异阶段
    best_t3_trees = d3_alpha_to_tree(popu)
    mut3 = []
    prefinal_d3_pop = []
    
    # 创建一个可修改的树列表，包含原始最佳树和它们的变异体
    all_trees_for_mutation_d3 = list(best_t3_trees) # 复制一份，避免在迭代中修改被迭代的列表

    for tree_node in all_trees_for_mutation_d3:
        mutated_node = mutate_random_node(tree_node, terminal_values, unary_ops, binary_ops, ts_ops, ts_ops_values)
        if mutated_node:
            best_t3_trees.append(mutated_node) # 将变异体添加到 best_t3_trees 中

    # 将所有树（原始最佳和变异后的）转换为 Alpha 表达式
    for tree_node in best_t3_trees:
        try:
            ft = d3tree_to_alpha(tree_node)
            if ft:
                mut3.append(ft)
        except (AttributeError, TypeError):
            # print(f"警告: 转换深度 3 变异树到 Alpha 表达式时出错，跳过该树。")
            continue

    mut3 = list(OrderedDict.fromkeys(mut3)) # 去重

    # 模拟评估变异后的 Alpha
    k = mut3
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=3, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)
     
    if Data.empty:
        print("警告: 深度 3 Alpha 变异阶段模拟未返回有效数据。")
        d3m_population = popu # 如果变异没有有效结果，则保留原始最佳
    else:
        d3m_population = fitness_fun(Data, n) # 变异后的最佳 Alpha
    
    # 将变异后的最佳 Alpha 添加到预最终种群
    prefinal_d3_pop.extend(d3m_population)

    # Note: 原 Notebook 在深度 3 阶段没有明确的交叉操作步骤。
    # 如果需要，可以在这里添加类似深度 2 的交叉逻辑。
    # 否则， prefinal_d3_pop 将只包含原始选择和变异后的 Alpha。

    prefinal_d3_pop = list(OrderedDict.fromkeys(prefinal_d3_pop)) # 最终去重

    # 最终模拟评估
    k = prefinal_d3_pop
    alpha_list = [generate_alpha(x) for x in k]
    cntx = simulate_alpha_list(s, alpha_list, depth=3, iteration=0)
    Data = prettify_result(cntx, detailed_tests_view=False, clickable_alpha_id=False)

    if Data.empty:
        print("警告: 深度 3 Alpha 最终阶段模拟未返回有效数据。")
        return [] # 返回空列表
    
    best_depth_three_population = fitness_fun(Data, n) # 最终选择最佳 n 个

    print("深度 3 Alpha 优化完成。")
    return best_depth_three_population


# %%
# 执行遗传算法优化，获取不同深度的最佳 Alpha
# 参数 n: 每代保留的最佳 Alpha 数量 (种群大小)
# 参数 m: 遗传算法的迭代次数

# 优化深度 1 的 Alpha
best_depth_one_alphas = best_d1_alphas(30, 5)

# 优化深度 2 的 Alpha，以上一代的深度 1 最佳 Alpha 作为构建块
best_depth_two_alphas =  best_d2_alphas(best_depth_one_alphas, 25, 10)

# 优化深度 3 的 Alpha，以上一代的深度 2 最佳 Alpha 作为构建块
best_depth_three_alphas = best_d3_alpha(best_depth_two_alphas, 20, 15)


# %%
# 打印深度 1 的最佳 Alpha 表达式
print("深度 1 的最佳 Alpha 表达式:")
print(best_depth_one_alphas)

# %%
# 打印深度 2 的最佳 Alpha 表达式
print("深度 2 的最佳 Alpha 表达式:")
print(best_depth_two_alphas)

# %%
# 打印深度 3 的最佳 Alpha 表达式
print("深度 3 的最佳 Alpha 表达式:")
print(best_depth_three_alphas)