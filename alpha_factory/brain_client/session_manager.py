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


# Custom exception for persistent credential errors
class InvalidCredentialsError(ValueError):
    pass


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
        self.token_expiry: int = 0 # UNIX时间戳（绝对时间），表示token的过期时间
        self.base_url: str = base_url
        # self._login_lock = threading.Lock() # Lock removed as per plan for simplified get_session

    def _get_credentials(self) -> Tuple[str, str]:
        """
        获取平台凭据（邮箱和密码）。
        严格按照 code.py 中 get_credentials() 的顺序和逻辑:
        1. 环境变量 BRAIN_CREDENTIAL_EMAIL, BRAIN_CREDENTIAL_PASSWORD。
        2. '~/secrets/platform-brain.json' 文件。
        3. 提示用户输入。
        确保文件检查 os.path.getsize(credentials_file_path) > 2 的逻辑被正确迁移。
        确保凭据保存回文件的逻辑与 code.py 一致。

        Returns:
            Tuple[str, str]: 包含邮箱和密码的元组。
        """
        email = os.environ.get('BRAIN_CREDENTIAL_EMAIL')
        password = os.environ.get('BRAIN_CREDENTIAL_PASSWORD')

        if email and password:
            print("信息 (_get_credentials): 使用环境变量中的凭据。")
            return email, password

        credentials_file_path = Path.home() / "secrets" / "platform-brain.json"

        if credentials_file_path.exists():
            if credentials_file_path.stat().st_size > 2: # Check if file is not empty (more than '{}')
                try:
                    with open(credentials_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        email = data.get("email")
                        password = data.get("password")
                        if email and password:
                            print(f"信息 (_get_credentials): 使用文件 {credentials_file_path} 中的凭据。")
                            return email, password
                        else:
                            print(f"警告 (_get_credentials): 文件 {credentials_file_path} 中的凭据格式不正确。")
                except json.JSONDecodeError:
                    print(f"警告 (_get_credentials): 无法解析文件 {credentials_file_path}。")
                except Exception as e:
                    print(f"警告 (_get_credentials): 读取文件 {credentials_file_path} 时出错: {e}")
            else:
                print(f"警告 (_get_credentials): 凭据文件 {credentials_file_path} 为空或过小。")


        print("信息 (_get_credentials): 未能从环境变量或有效文件中获取凭据，将提示用户输入。")
        email = input("请输入您的BRAIN邮箱地址: ")
        password = getpass.getpass("请输入您的BRAIN密码: ")

        try:
            credentials_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credentials_file_path, 'w', encoding='utf-8') as f:
                json.dump({'email': email, 'password': password}, f)
            print(f"信息 (_get_credentials): 凭据已保存到 {credentials_file_path}。")
        except Exception as e:
            print(f"错误 (_get_credentials): 保存凭据到 {credentials_file_path} 失败: {e}")
            # 即使保存失败，本次输入的凭据仍然可以使用

        return email, password

    def _login(self) -> None:
        """
        执行登录到 WorldQuant BRAIN 平台的逻辑。
        **完全重写**此方法以匹配 `code.py` 中 `start_session()` 的行为。
        处理 Persona 认证流程。
        处理错误凭据 (非Persona 401)。
        获取并存储绝对 token 过期时间。
        移除所有旧的相对时间逻辑。
        打印信息与 code.py 风格一致。

        Raises:
            InvalidCredentialsError: 如果凭据错误且非Persona认证导致401。
            requests.exceptions.RequestException: 网络或请求相关错误。
            ValueError: 其他登录过程中的逻辑错误。
        """
        print("信息 (_login): 开始登录流程...")
        s = requests.Session()

        try:
            email, password = self._get_credentials()
            s.auth = (email, password)
        except Exception as e: # Should cover if _get_credentials itself has an issue not resulting in returned creds
            print(f"错误 (_login): 获取凭据时发生内部错误: {e}")
            raise ValueError("获取登录凭据失败。") from e

        auth_url = urljoin(self.base_url, "authentication")

        try:
            print(f"信息 (_login): POST {auth_url}")
            r = s.post(auth_url)

            # Persona 认证流程
            # code.py: if r.status_code == requests.status_codes.codes.unauthorized and r.headers["WWW-Authenticate"] == "persona":
            if r.status_code == 401 and r.headers.get("WWW-Authenticate", "").lower().strip() == "persona":
                persona_location = r.headers.get("Location")
                if not persona_location:
                    print("错误 (_login): Persona认证响应缺少Location头部。")
                    raise ValueError("Persona认证错误：响应缺少Location。")

                # code.py: url = urljoin(r.url, r.headers["Location"]) -> r.url is the original auth_url
                persona_auth_url = urljoin(auth_url, persona_location)

                print(f"\n重要提示：需要进行用户认证。")
                print(f"请在浏览器中打开以下链接完成认证，然后按 Enter键 继续：\n{persona_auth_url}")
                input() # 等待用户操作

                # code.py: s.post(url)
                print(f"信息 (_login): 第一次 POST {persona_auth_url} (Persona)")
                persona_resp = s.post(persona_auth_url)

                # code.py: while s.post(url).status_code != 201:
                # Loop while status is not 201 Created
                while persona_resp.status_code != 201:
                    print(f"提示 (_login): Persona认证尚未完成 (状态码: {persona_resp.status_code})。")
                    print(f"请确保您已在浏览器中完成认证。")
                    input("完成后请按 Enter键 重试...")
                    print(f"信息 (_login): 重新 POST {persona_auth_url} (Persona)")
                    persona_resp = s.post(persona_auth_url)

                print("信息 (_login): Persona认证成功 (状态码: 201)。")
                # Session 's' is now authenticated through Persona. Proceed to get token expiry.

            # 错误凭据处理 (非Persona 401)
            # code.py: elif r.status_code == 401:
            elif r.status_code == 401:
                print("错误 (_login): 凭据错误或认证失败 (401)。")
                credentials_file_path = Path.home() / "secrets" / "platform-brain.json"
                print(f"信息 (_login): 尝试清空凭据文件: {credentials_file_path}")
                try:
                    with open(credentials_file_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f) # Write empty JSON object to clear it
                    print(f"信息 (_login): 已成功清空凭据文件: {credentials_file_path}")
                except Exception as e_clear:
                    print(f"警告 (_login): 清空凭据文件 {credentials_file_path} 失败: {e_clear}")
                # code.py: return start_session() -> here we raise an exception
                raise InvalidCredentialsError("无效的用户名或密码。请检查您的凭据。")

            # 其他 HTTP 错误 (非401 and not successful 2xx)
            elif r.status_code // 100 != 2: # If not a 2xx status
                print(f"错误 (_login): 认证请求失败，状态码: {r.status_code}, 响应: {r.text[:500]}")
                r.raise_for_status() # This will raise an HTTPError

            # 如果初始POST成功 (e.g. 200 or 201 without Persona), 或 Persona 认证成功
            # 获取 Token 过期时间
            # code.py: expiry = s.get(auth_url).json()["token"]["expiry"]
            print(f"信息 (_login): GET {auth_url} (获取token详情)")
            auth_check_response = s.get(auth_url)
            auth_check_response.raise_for_status() # Ensure this GET request is successful

            token_info = auth_check_response.json()
            if "token" in token_info and isinstance(token_info["token"], dict) and "expiry" in token_info["token"]:
                self.token_expiry = int(token_info["token"]["expiry"]) # Store absolute timestamp
                self.session = s # Store the successfully authenticated session
                print(f"信息 (_login): 登录成功。会话已建立。Token有效期至 (绝对时间戳): {self.token_expiry} ({time.ctime(self.token_expiry)})")
            else:
                print(f"错误 (_login): 未能从认证响应中获取有效的token过期时间。响应: {token_info}")
                raise ValueError("无法解析token过期时间。")

        except InvalidCredentialsError: # Ensure custom error is re-raised
            raise
        except requests.exceptions.RequestException as e:
            print(f"错误 (_login): API请求失败或网络问题: {e}")
            raise # Re-raise the requests exception
        except Exception as e:
            print(f"错误 (_login): 登录过程中发生未预料的错误: {e}")
            raise ValueError(f"登录时发生未知错误: {e}") # Wrap other errors


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
        如果当前没有会话 (self.session is None) 或 token_expiry 未设置 (0),
        则尝试登录。移除主动过期检查逻辑。

        Returns:
            requests.Session: 一个有效的会话对象。

        Raises:
            InvalidCredentialsError: 如果_login因凭据错误失败。
            ValueError: 如果登录失败且无法建立会话 (其他原因)。
        """
        if self.session is None or self.token_expiry == 0:
            print("信息 (get_session): 无有效会话或token_expiry未设置，尝试登录...")
            try:
                self._login()
            except InvalidCredentialsError: # Catch the specific error from _login
                # Log or handle as per desired behavior for failed first login
                # For now, re-raise to let the caller handle it or program to exit
                print("错误 (get_session): _login 失败，无效的凭据。")
                raise
            except Exception as e: # Catch other exceptions from _login
                print(f"错误 (get_session): _login 失败: {e}")
                raise ValueError("无法获取有效的API会话。") from e

        if self.session is None: # Should ideally not happen if _login succeeded or raised
            print("严重错误 (get_session): _login 调用后 self.session 仍为 None 且未抛出预期异常。")
            raise ValueError("无法建立API会话，内部状态异常。")

        return self.session

    def check_session_timeout(self, s: requests.Session) -> int:
        """
        此方法严格按照 code.py 中的 check_session_timeout(s) 函数实现。
        它接收一个 requests.Session 对象 s作为参数。
        向 https://api.worldquantbrain.com/authentication 发起 GET 请求。
        从返回的 JSON 中解析 ["token"]["expiry"] (这是一个绝对时间戳)。
        计算当前时间到这个过期时间戳的剩余秒数 (expiry_timestamp - int(time.time()))。
        如果请求失败或解析出错，应返回 0 (与 code.py 的 except: return 0 行为一致)。
        """
        if not isinstance(s, requests.Session):
            print("警告 (check_session_timeout): 传入的不是有效的Session对象。")
            return 0

        auth_url = urljoin(self.base_url, "authentication")
        try:
            print(f"信息 (check_session_timeout): GET {auth_url}")
            r = s.get(auth_url)
            r.raise_for_status()
            expiry_timestamp = int(r.json()["token"]["expiry"])
            remaining_time = expiry_timestamp - int(time.time())
            return max(0, remaining_time) # Ensure non-negative
        except requests.exceptions.RequestException as e:
            print(f"警告 (check_session_timeout): 请求失败: {e}")
            return 0
        except (KeyError, ValueError, TypeError) as e:
            print(f"警告 (check_session_timeout): 解析token expiry失败: {e}")
            return 0
        except Exception as e: # Catch-all for any other unexpected error
            print(f"警告 (check_session_timeout): 未知错误: {e}")
            return 0

    def force_refresh_session(self) -> None:
        """
        此方法的主体是直接调用 self._login()。
        这将执行一次全新的登录，并更新 self.session 和 self.token_expiry。
        此方法模拟 code.py 中 s = start_session() 的重新赋值行为，用于显式刷新。
        如果 _login() 失败并抛出 InvalidCredentialsError，此方法也应允许该异常向上传播。
        """
        print("信息 (force_refresh_session): 强制执行新的登录...")
        self._login() # This will update self.session and self.token_expiry

    def check_session_timeout_seconds(self) -> int:
        """
        检查当前内部存储的会话 (self.session) 的剩余有效时间（秒）。
        如果会话无效或未登录，返回0。
        这是基于 self.token_expiry 的检查，不主动调用API。
        """
        if self.session is None or self.token_expiry == 0:
            return 0
        remaining_time = self.token_expiry - int(time.time())
        return max(0, remaining_time) # 确保不返回负数
