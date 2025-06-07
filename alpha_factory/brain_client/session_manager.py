# alpha_factory/brain_client/session_manager.py
import requests
import time
import os
import json
import getpass
from urllib.parse import urljoin, urlparse # 导入 urlparse 用于提取主机名作为 token 的一部分
from pathlib import Path # 导入 Path 以便与 code.py 的实现更一致
from typing import Optional, Tuple # 确保导入 Optional 和 Tuple
import threading # 导入 threading 用于实现锁

class SessionManager:
    """
    管理与 WorldQuant BRAIN API 的会话，包括登录、凭据管理和会话刷新。
    """
    def __init__(self, base_url: str = "https://api.worldquantbrain.com"):
        """
        初始化 SessionManager。

        Args:
            base_url (str): BRAIN API 的基础URL。
        """
        self.session: Optional[requests.Session] = None
        self.token_expiry: int = 0 # UNIX时间戳，表示token的过期时间
        self.base_url: str = base_url
        self._login_lock = threading.Lock() # 添加线程锁以防止并发登录
        # self.user_email: Optional[str] = None # 可以考虑存储email用于 Persona 认证提示

    def _get_credentials(self) -> Tuple[str, str]:
        """
        获取平台凭据（邮箱和密码）。
        与 code.py 中的 get_credentials 函数逻辑一致。
        优先从环境变量 BRAIN_CREDENTIAL_EMAIL 和 BRAIN_CREDENTIAL_PASSWORD 读取。
        其次尝试从 '~/secrets/platform-brain.json' 文件读取。
        如果均未找到，则提示用户输入并保存到文件中。

        Returns:
            Tuple[str, str]: 包含邮箱和密码的元组。

        Raises:
            Exception: 如果无法获取凭据。
        """
        # 中文注释：尝试从环境变量获取凭据
        credential_email = os.environ.get('BRAIN_CREDENTIAL_EMAIL')
        credential_password = os.environ.get('BRAIN_CREDENTIAL_PASSWORD')

        # 中文注释：定义凭据文件的路径
        credentials_folder_path = Path(os.path.expanduser("~")) / "secrets"
        credentials_file_path = credentials_folder_path / "platform-brain.json"

        if credential_email and credential_password:
            # 中文注释：如果环境变量中存在凭据，则直接使用
            # print("信息：从环境变量中获取了BRAIN凭据。") # 示例日志
            # self.user_email = credential_email
            return credential_email, credential_password

        # 中文注释：尝试从本地文件读取凭据
        if credentials_file_path.exists() and credentials_file_path.stat().st_size > 2: # 检查文件存在且非空
            try:
                with open(credentials_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if "email" in data and "password" in data:
                        # print(f"信息：从文件 {credentials_file_path} 中获取了BRAIN凭据。")
                        # self.user_email = data["email"]
                        return data["email"], data["password"]
                    else:
                        print(f"警告：凭据文件 {credentials_file_path} 格式不正确，缺少email或password字段。")
            except json.JSONDecodeError:
                print(f"警告：无法解析凭据文件 {credentials_file_path}。")
            except Exception as e:
                print(f"警告：读取凭据文件 {credentials_file_path} 时发生未知错误: {e}")

        # 中文注释：如果环境变量和文件均无有效凭据，则提示用户输入
        print("信息：未能从环境变量或本地文件找到有效的BRAIN凭据。")
        email = input("请输入您的BRAIN邮箱地址: ")
        password = getpass.getpass(prompt="请输入您的BRAIN密码: ")

        # self.user_email = email

        # 中文注释：创建目录（如果不存在）并保存用户输入的凭据到文件
        try:
            credentials_folder_path.mkdir(parents=True, exist_ok=True)
            with open(credentials_file_path, 'w', encoding='utf-8') as file:
                json.dump({"email": email, "password": password}, file)
            print(f"信息：凭据已保存到 {credentials_file_path}。")
        except Exception as e:
            print(f"错误：保存凭据到文件 {credentials_file_path} 失败: {e}")
            # 即使保存失败，本次输入的凭据仍然可以使用

        return email, password

    def _login(self) -> None:
        """
        执行登录到 WorldQuant BRAIN 平台的逻辑。
        与 code.py 中的 start_session 函数的核心逻辑一致。
        成功登录后，会更新 self.session 和 self.token_expiry。

        Raises:
            requests.exceptions.RequestException: 如果登录过程中发生网络或其他请求相关的错误。
            ValueError: 如果登录失败或无法获取token。
        """
        print("信息：正在尝试登录到 WorldQuant BRAIN 平台...")
        s = requests.Session()
        try:
            s.auth = self._get_credentials()
        except Exception as e: # _get_credentials 内部可能抛出异常或无法获取
            print(f"错误 (SessionManager._login): 获取凭据失败: {e}")
            raise ValueError("无法获取登录凭据，登录中止。") from e

        auth_url = urljoin(self.base_url, "authentication")

        try:
            r = s.post(auth_url)
            r.raise_for_status() # 检查初始POST请求是否有HTTP错误 (例如 400, 500)

            # 处理生物识别认证 (Persona)
            if r.status_code == 200 and r.headers.get("WWW-Authenticate", "").lower().strip() == "persona":
                persona_path = r.headers.get("Location")
                if not persona_path:
                    print("错误 (SessionManager._login): Persona认证需要，但响应头中未找到Location。")
                    raise ValueError("Persona认证失败：缺少Location头部。")

                persona_url = urljoin(auth_url, persona_path) # 使用 auth_url 作为 persona_path 的基础

                print(f"重要提示：需要进行生物识别认证 (Persona)。")
                print(f"请在浏览器中打开以下链接完成认证，然后按 Enter键 继续：\n{persona_url}")
                input() # 等待用户操作

                # 循环检查认证是否完成
                while True:
                    print("信息：正在检查Persona认证状态...")
                    try:
                        # 根据 code.py，再次 POST 到 Location URL 以确认或推进认证
                        persona_check_response = s.post(persona_url)
                        if persona_check_response.status_code == 201: # 201 Created 通常表示认证成功并创建了会话
                            print("信息：Persona认证成功。")
                            break
                        elif persona_check_response.status_code == 200 and persona_check_response.headers.get("WWW-Authenticate", "").lower().strip() == "persona":
                            # 某些情况下，可能还是返回200和persona头，提示用户重试
                            print("提示：Persona认证似乎尚未完成或已超时。")
                            print(f"如果认证已完成，请按 Enter键 重试检查。如果认证失败或超时，请重新认证并按 Enter键。")
                            input()
                        else: # 其他非预期状态码
                            print(f"错误 (SessionManager._login): Persona认证检查返回非预期状态 {persona_check_response.status_code}。响应内容: {persona_check_response.text[:200]}")
                            print(f"如果认证已在浏览器中完成，请按 Enter键 重试。否则，请检查认证流程。")
                            input() # 给用户机会重试
                            # raise ValueError(f"Persona认证失败：状态检查返回 {persona_check_response.status_code}")
                    except requests.exceptions.RequestException as e_persona_check:
                        print(f"错误 (SessionManager._login): Persona认证状态检查请求失败: {e_persona_check}")
                        print("请确保网络连接正常，并在浏览器中完成认证后按 Enter键 重试。")
                        input() # 给用户机会重试

            elif r.status_code != 201: # 期望直接登录成功是 201 Created (code.py 中 start_session 最终检查的是 s.get(auth_url).json()["token"]["expiry"])
                                    # 但实际POST到 /authentication 后，如果不需要Persona，会直接在响应中包含token信息或需要GET一次
                # 尝试GET一下 /authentication 看看是否有token
                check_r = s.get(auth_url)
                if check_r.status_code == 200 and "token" in check_r.json():
                    pass # Token 在 GET 响应中，后面会处理
                else:
                    print(f"错误 (SessionManager._login): 登录失败。初始POST状态: {r.status_code}, GET状态: {check_r.status_code}, 响应: {check_r.text[:200]}")
                    # 清空可能已保存的错误凭据
                    credentials_file_path = Path(os.path.expanduser("~")) / "secrets" / "platform-brain.json"
                    if credentials_file_path.exists():
                        try:
                            with open(credentials_file_path, 'w', encoding='utf-8') as file:
                                json.dump({}, file)
                            print(f"提示：已清空本地保存的凭据文件 {credentials_file_path}，下次将提示重新输入。")
                        except Exception as e_clear:
                            print(f"警告：清空本地凭据文件 {credentials_file_path} 失败: {e_clear}")
                    raise ValueError("登录失败，请检查您的邮箱和密码。")

            # 获取并设置 token 过期时间
            # 无论是否经过Persona，最终都应该可以通过GET /authentication获取token信息
            final_auth_check = s.get(auth_url)
            final_auth_check.raise_for_status() # 确保获取token的请求成功

            token_data = final_auth_check.json()
            print(f"调试信息 (SessionManager._login): 从API获取的原始token_data: {token_data}") # 打印原始token数据以供调试
            if "token" in token_data and isinstance(token_data["token"], dict) and "expiry" in token_data["token"]:
                try:
                    expiry_value = token_data["token"]["expiry"]
                    current_timestamp = int(time.time())
                    
                    # 检查API返回的是相对时间还是绝对时间戳
                    expiry_int = int(expiry_value)
                    if expiry_int < current_timestamp - 60*60*24*365:  # 如果过期时间远早于当前时间（例如超过1年），可能是相对时间
                        # API返回的是相对时间（秒数），需要转换为绝对时间戳
                        self.token_expiry = current_timestamp + expiry_int
                        print(f"信息 (SessionManager._login): 检测到相对过期时间 {expiry_int} 秒，转换为绝对时间戳。")
                    else:
                        # API返回的是绝对时间戳
                        self.token_expiry = expiry_int
                    
                    self.session = s
                    print(f"信息：成功登录到 WorldQuant BRAIN 平台。会话有效期至: {time.ctime(self.token_expiry)}")
                except ValueError as ve:
                    print(f"错误 (SessionManager._login): token中的expiry值 '{expiry_value}' 无法转换为整数: {ve}")
                    raise ValueError("登录后未能获取有效的token过期时间格式。") from ve
            else:
                print(f"错误 (SessionManager._login): 登录后无法获取有效的token信息或token结构不正确。Token data: {token_data}")
                raise ValueError("登录后未能获取token或token结构不正确。")

        except requests.exceptions.HTTPError as http_err:
            # 特别处理401 Unauthorized，可能需要清除旧凭据
            if http_err.response.status_code == 401:
                print(f"错误 (SessionManager._login): 登录认证失败 (401 Unauthorized)。请检查您的凭据。")
                credentials_file_path = Path(os.path.expanduser("~")) / "secrets" / "platform-brain.json"
                if credentials_file_path.exists():
                    try:
                        with open(credentials_file_path, 'w', encoding='utf-8') as file:
                            json.dump({}, file) # 清空错误的凭据
                        print(f"提示：由于认证失败，已清空本地保存的凭据文件 {credentials_file_path}。")
                    except Exception as e_clear_401:
                         print(f"警告：清空本地凭据文件 {credentials_file_path} 失败: {e_clear_401}")
                raise ValueError("登录认证失败 (401)。") from http_err
            else:
                print(f"错误 (SessionManager._login): 登录过程中发生HTTP错误: {http_err}")
                raise
        except requests.exceptions.RequestException as req_err:
            print(f"错误 (SessionManager._login): 登录过程中发生网络请求错误: {req_err}")
            raise
        except Exception as e:
            print(f"错误 (SessionManager._login): 登录过程中发生未知错误: {e}")
            raise ValueError(f"登录过程中发生未知错误: {e}") from e

    def refresh_session(self) -> None:
        """
        Refreshes the current session by performing a login.
        This method is called when an existing session is about to expire.
        """
        print("信息 (SessionManager.refresh_session): 会话即将过期或已过期，正在调用 _login 刷新...")
        try:
            self._login()
            print("信息 (SessionManager.refresh_session): 会话刷新成功。")
        except Exception as e:
            print(f"错误 (SessionManager.refresh_session): 调用 _login 刷新会话时发生错误: {e}")
            # Re-raise the exception to allow the caller to handle it
            raise

    def get_session(self) -> requests.Session:
        """
        获取一个有效的、经过认证的 requests.Session 对象。
        如果当前没有会话，或者会话已过期/即将过期（提前5分钟刷新），则会自动执行登录。
        使用线程锁确保 _login 只被调用一次。

        Returns:
            requests.Session: 一个有效的会话对象。

        Raises:
            ValueError: 如果登录失败且无法建立会话。
        """
        # 中文注释：检查会话是否存在以及token是否即将过期（提前300秒/5分钟刷新）
        current_time = int(time.time())
        should_login = False
        if self.session is None or self.token_expiry == 0 or current_time >= (self.token_expiry - 300):
            should_login = True

        if should_login:
            with self._login_lock:
                # 再次检查条件，因为在等待锁的过程中，其他线程可能已经完成了登录
                current_time_after_lock = int(time.time())
                if self.session is None or self.token_expiry == 0 or current_time_after_lock >= (self.token_expiry - 300):
                    if self.session is not None and self.token_expiry > 0:
                        remaining_minutes = (self.token_expiry - current_time_after_lock) // 60
                        if self.token_expiry < current_time_after_lock : # 如果已经过期
                             print(f"信息：当前会话已过期 (过期时间: {time.ctime(self.token_expiry)})，正在尝试重新登录...")
                        else:
                             print(f"信息：当前会话即将过期 (剩余 {remaining_minutes} 分钟)，正在尝试重新登录...")
                    else:
                        print("信息：无有效会话，正在尝试登录...")
                    try:
                        self._login()  # 执行登录流程
                    except Exception as e:  # _login 可能会抛出各种异常
                        print(f"错误 (SessionManager.get_session): 尝试登录或刷新会话失败: {e}")
                        raise ValueError("无法获取有效的API会话。") from e
                else:
                    print("信息：在等待锁期间，其他线程已成功登录/刷新会话。")

        # 中文注释：如果登录或刷新后仍然没有session，说明存在严重问题
        if self.session is None:
            print("严重错误 (SessionManager.get_session): 登录后会话对象依然为None。")
            raise ValueError("无法建立API会话。")

        return self.session

    def check_session_timeout_seconds(self) -> int:
        """
        检查当前会话的剩余有效时间（秒）。
        如果会话无效或未登录，返回0。
        与 code.py 中的 check_session_timeout 函数逻辑类似。

        Returns:
            int: 会话的剩余有效期（秒）。
        """
        if self.session is None or self.token_expiry == 0:
            return 0

        # 尝试通过API刷新token过期时间信息，因为token_expiry是登录时设置的，可能已不准确
        # 但频繁调用 /authentication 可能不是最佳实践，cg.md 要求 get_session 中包含刷新逻辑
        # 此函数可以简单返回基于 self.token_expiry 的剩余时间
        # get_session 会在需要时调用 _login 来刷新 token_expiry

        # 如果要实时检查，可以像 code.py 一样调用 GET /authentication
        # auth_url = urljoin(self.base_url, "authentication")
        # try:
        #     response = self.session.get(auth_url)
        #     response.raise_for_status()
        #     token_data = response.json()
        #     if "token" in token_data and "expiry" in token_data["token"]:
        #         self.token_expiry = int(token_data["token"]["expiry"]) # 更新存储的过期时间
        # except requests.RequestException as e:
        #     print(f"警告 (check_session_timeout_seconds): 无法从API获取最新的token过期时间: {e}")
        #     # 如果获取失败，则依赖已存储的 self.token_expiry

        remaining_time = self.token_expiry - int(time.time())
        return max(0, remaining_time) # 确保不返回负数
