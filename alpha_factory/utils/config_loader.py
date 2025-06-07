# alpha_factory/utils/config_loader.py
import yaml
from pathlib import Path
from pydantic import ValidationError
from .config_models import AppConfig # 从同级目录的 config_models 导入 AppConfig

def load_config(path: Path) -> AppConfig:
    """加载并验证YAML配置文件。

    Args:
        path (Path): YAML配置文件的路径。

    Raises:
        FileNotFoundError: 如果指定的配置文件路径不存在。
        ValueError: 如果配置文件内容无法被正确解析或验证失败，
                    会包含 Pydantic ValidationError 的详细信息。

    Returns:
        AppConfig: 解析并验证通过的配置对象。
    """
    if not path.exists():
        # 日志记录配置文件未找到的错误
        # print(f"错误：配置文件 {path} 未找到。") # 示例日志，实际项目中可能使用 logging 模块
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, 'r', encoding='utf-8') as f: # 指定utf-8编码以支持中文
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # 日志记录YAML解析错误
            # print(f"错误：解析配置文件 {path} 时发生YAML错误: {e}")
            raise ValueError(f"配置文件 '{path.name}' 解析错误: \n{e}") from e

    if data is None:
        # 日志记录配置文件为空的错误
        # print(f"错误：配置文件 {path} 为空或格式不正确。")
        raise ValueError(f"配置文件 '{path.name}' 为空或不是有效的YAML格式。")

    try:
        config = AppConfig.parse_obj(data)
        # 日志记录配置加载成功
        # print(f"配置 {path.name} 加载并验证成功。")
        return config
    except ValidationError as e:
        # 日志记录Pydantic验证错误
        # print(f"错误：配置文件 {path.name} 验证失败: {e}")
        raise ValueError(f"配置文件 '{path.name}' 格式错误: \n{e}") from e
