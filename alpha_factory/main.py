# alpha_factory/main.py
import typer # 用于创建命令行界面应用
from pathlib import Path # 用于处理文件路径
from typing import Optional # 用于类型提示 Optional

from rich.console import Console # Rich库，用于更美观的控制台输出
from rich.traceback import install as install_rich_traceback # Rich库，用于美化异常回溯信息

# 导入项目内部模块
from alpha_factory.cli import menus # 从 cli 子包导入 menus 模块
from alpha_factory.utils import config_loader # 从 utils 子包导入 config_loader 模块
from alpha_factory.utils.config_models import AppConfig # 显式导入AppConfig供类型提示
from alpha_factory.genetic_programming.engine import StagedGPEngine # 默认使用Staged引擎
from alpha_factory.brain_client.session_manager import SessionManager
from alpha_factory.brain_client.api_client import BrainApiClient

# 初始化 Rich Console 和 Traceback (美化错误输出)
console = Console()
install_rich_traceback(show_locals=False) # show_locals=False 避免显示过多局部变量信息

# 创建 Typer 应用实例
app = typer.Typer(
    help="AlphaFactory - 自动化Alpha因子发现框架 (v7.0)",
    add_completion=False, # 可以禁用或启用shell自动补全功能
    no_args_is_help=True, # 没有参数时显示帮助信息
    rich_markup_mode="markdown" # 允许在帮助信息中使用markdown
)

@app.command(name="run", help="🚀 运行一个遗传编程实验来发现Alpha因子。")
def run_experiment(
    config_path: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="直接指定要运行的实验配置文件路径。如果未提供，将交互式选择。",
        show_default=False, # 不显示默认值 None
        exists=True, # Typer会自动检查文件是否存在
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True # 将路径解析为绝对路径
    )
):
    """
    运行一个遗传编程实验。
    可以通过 -c/--config 选项直接指定配置文件，否则会进入交互模式选择。
    """
    console.print("---  AlphaFactory 实验运行开始 ---", style="bold blue")

    selected_config_path: Optional[Path] = config_path

    if not selected_config_path:
        console.print("提示：未通过命令行参数指定配置文件，进入交互式选择模式...")
        selected_config_path = menus.select_experiment()
        if not selected_config_path:
            console.print("[yellow]取消操作：未选择配置文件，实验终止。[/yellow]")
            raise typer.Exit(code=1) # 正常退出，但标记为非成功

    console.print(f"🔩 正在加载配置: [cyan]{selected_config_path.name}[/cyan] (路径: {selected_config_path})")
    try:
        app_config: AppConfig = config_loader.load_config(selected_config_path)
        console.print(f"✅ 配置文件 '{selected_config_path.name}' 加载并验证成功。实验名称: '{app_config.experiment_name}'")
    except FileNotFoundError:
        console.print(f"[bold red]错误：配置文件 '{selected_config_path}' 未找到。[/bold red]")
        raise typer.Exit(code=1)
    except ValueError as ve:
        console.print(f"[bold red]错误：配置文件 '{selected_config_path.name}' 格式无效或验证失败。[/bold red]")
        console.print(f"详细错误信息:\n{ve}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]错误：加载配置文件时发生未知错误: {e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        console.print("🤝 正在初始化API会话管理器...")
        # 假设BRAIN API基础URL是固定的，或者未来从更高层配置读取
        # 目前 SessionManager 构造函数有默认值 "https://api.worldquantbrain.com"
        session_manager = SessionManager()
        # 尝试获取一次会话以触发登录（如果需要）
        _ = session_manager.get_session()
        console.print("✅ API会话管理器初始化成功。")

        console.print("📡 正在初始化BRAIN API客户端...")
        api_client = BrainApiClient(session_manager, app_config)
        console.print("✅ BRAIN API客户端初始化成功。")

        # 根据 cg.md 要求，此处直接实例化 StagedGPEngine
        # 未来可以根据 app_config.algorithm.engine 的值来动态选择引擎
        console.print(f"🧬 正在初始化 '{app_config.algorithm.engine}' 遗传编程引擎...")
        if app_config.algorithm.engine == "staged":
            engine = StagedGPEngine(app_config, api_client)
            console.print("✅ 分阶段遗传编程引擎 (StagedGPEngine) 初始化成功。")
        else:
            console.print(f"[bold red]错误：不支持的引擎类型 '{app_config.algorithm.engine}'。目前仅支持 'staged'。[/bold red]")
            raise typer.Exit(code=1)

        console.print(f"🚀 引擎 '{app_config.algorithm.engine}' 开始运行实验 '{app_config.experiment_name}'...")
        engine.run() # 核心运行逻辑

        console.print("\n🎉 [bold green]实验运行圆满结束！[/bold green]")
        console.print("--- AlphaFactory 实验运行完毕 ---", style="bold blue")

    except typer.Exit: # 捕获typer的退出，避免被下面的通用Exception捕获
        raise
    except KeyboardInterrupt:
        console.print("\n[bold yellow]操作被用户中断 (Ctrl+C)。实验提前终止。[/bold yellow]")
        raise typer.Exit(code=130) # 符合Ctrl+C的退出码
    except Exception as e:
        console.print(f"\n❌ [bold red]实验执行过程中发生严重错误:[/bold red]")
        # Rich会自动打印美化的traceback
        console.print_exception(show_locals=False) # 再次确保不显示局部变量
        # console.print(f"错误类型: {type(e).__name__}")
        # console.print(f"错误详情: {e}")
        console.print("--- AlphaFactory 实验运行因错误中止 ---", style="bold red")
        raise typer.Exit(code=1)


@app.command(name="init", help="🛠️  初始化一个新的实验配置文件。")
def init_config():
    """
    引导用户创建一个新的配置文件，从可用模板中选择。
    """
    console.print("--- AlphaFactory 新配置文件初始化 ---", style="bold blue")
    new_file_path = menus.create_new_config()
    if new_file_path:
        console.print(f"💡 提示：您现在可以编辑 '{new_file_path}' 文件，然后使用 'run -c {new_file_path.name}' 来运行它。")
    else:
        console.print("信息：未创建新的配置文件。")
    console.print("--- 初始化流程结束 ---", style="bold blue")


@app.command(name="validate", help="校验一个指定的配置文件是否符合格式要求。")
def validate_config_file(
    config_path: Path = typer.Argument(
        ..., # ... 表示此参数是必需的
        help="需要验证的实验配置文件的完整路径。",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    )
):
    """
    加载并验证指定的配置文件。
    会输出验证结果，包括详细的Pydantic验证错误（如果存在）。
    """
    console.print(f"--- AlphaFactory 配置文件验证 ('{config_path.name}') ---", style="bold blue")
    try:
        _ = config_loader.load_config(config_path) # 尝试加载
        console.print(f"[bold green]✅ 配置文件 '{config_path.name}' 格式正确，验证通过！[/bold green]")
    except FileNotFoundError: # 虽然Typer的exists=True会检查，但以防万一
        console.print(f"[bold red]错误：配置文件 '{config_path}' 未找到。[/bold red]")
        raise typer.Exit(code=1)
    except ValueError as ve: # config_loader 在验证失败时抛出 ValueError
        console.print(f"[bold red]❌ 配置文件 '{config_path.name}' 验证失败。[/bold red]")
        console.print("详细错误信息:")
        console.print(str(ve)) # Pydantic的ValidationError可以直接转为字符串显示错误细节
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]错误：验证配置文件时发生未知错误: {e}[/bold red]")
        console.print_exception(show_locals=False)
        raise typer.Exit(code=1)
    console.print("--- 验证流程结束 ---", style="bold blue")


@app.command(name="list", help="📜 列出所有在 'configs' 目录中可用的实验配置文件。")
def list_configs_command(): # Typer 会自动将函数名中的下划线转为命令中的连字符
    """
    列出 `alpha_factory/configs` 目录中所有可用的 YAML 配置文件。
    """
    console.print("--- AlphaFactory 可用配置文件列表 ---", style="bold blue")
    menus.list_available_configs()
    console.print("--- 列表显示完毕 ---", style="bold blue")


if __name__ == "__main__":
    # 这里是程序的入口点
    # typer.run(main) # 如果只有一个函数作为命令，可以用这个
    app() # 对于多命令应用，调用 Typer app 实例
