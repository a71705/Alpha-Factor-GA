# alpha_factory/cli/menus.py
import questionary # 用于交互式命令行提示
import shutil # 用于文件操作，如复制
from pathlib import Path
from typing import Optional, List # 确保导入 Optional 和 List
from rich.console import Console # 用于漂亮的打印输出
from rich.table import Table # 用于以表格形式展示数据
import os # 用于获取文件信息
import time # 导入 time 模块以使用 time.ctime

# 定义配置文件目录的全局常量，方便维护
CONFIG_DIR = Path("alpha_factory/configs") # 指向 alpha_factory/configs
# 确保 CONFIG_DIR 是相对于项目根目录的正确路径。
# 如果脚本是从 alpha_factory 外部运行，可能需要调整。
# 假设脚本的执行上下文的当前工作目录是项目的根目录。

def select_experiment() -> Optional[Path]:
    """
    提示用户从 configs/ 目录下选择一个 YAML 实验配置文件。

    Returns:
        Optional[Path]: 用户选择的配置文件的 Path 对象，如果取消或无文件则返回 None。
    """
    console = Console()
    if not CONFIG_DIR.exists() or not CONFIG_DIR.is_dir():
        console.print(f"[bold red]错误：配置目录 '{CONFIG_DIR}' 不存在或不是一个目录。[/bold red]")
        return None

    # 查找所有 .yaml 或 .yml 文件
    config_files = list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))

    if not config_files:
        console.print(f"[yellow]提示：在 '{CONFIG_DIR}' 目录下未找到任何 YAML 配置文件。[/yellow]")
        console.print("您可以使用 'init' 命令创建一个新的配置文件。")
        return None

    # 创建选择列表，只显示文件名
    choices = sorted([file.name for file in config_files])

    try:
        selected_filename = questionary.select(
            "请选择一个实验配置文件:",
            choices=choices,
            qmark="🔬",
            pointer="👉"
        ).ask()
    except Exception as e: # 捕获 questionary 可能的异常，例如在非交互式环境运行
        console.print(f"[bold red]错误：选择配置文件时发生错误: {e}[/bold red]")
        return None


    if selected_filename:
        return CONFIG_DIR / selected_filename
    else:
        # 用户可能按了 Esc 或 Ctrl+C
        console.print("[yellow]提示：未选择任何配置文件。[/yellow]")
        return None

def create_new_config() -> Optional[Path]:
    """
    引导用户创建一个新的配置文件。
    用户可以输入新配置文件的名称，并选择一个模板进行复制。

    Returns:
        Optional[Path]: 新创建的配置文件的 Path 对象，如果创建失败或用户取消则返回 None。
    """
    console = Console()
    if not CONFIG_DIR.exists():
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            console.print(f"信息：配置目录 '{CONFIG_DIR}' 已创建。")
        except OSError as e:
            console.print(f"[bold red]错误：无法创建配置目录 '{CONFIG_DIR}': {e}[/bold red]")
            return None

    try:
        new_config_name = questionary.text(
            "请输入新配置文件的名称 (例如: my_awesome_experiment):",
            validate=lambda text: True if len(text.strip()) > 0 else "名称不能为空白。",
            qmark="📝"
        ).ask()

        if not new_config_name: # 用户取消
            console.print("[yellow]提示：已取消创建新配置文件。[/yellow]")
            return None

        new_config_name = new_config_name.strip()
        # 确保文件名不包含非法字符，并添加 .yaml 后缀 (如果用户没有提供)
        # 简单的处理：替换空格为下划线
        safe_filename = new_config_name.replace(" ", "_")
        if not safe_filename.endswith((".yaml", ".yml")):
            safe_filename += ".yaml"

        target_path = CONFIG_DIR / safe_filename

        if target_path.exists():
            overwrite = questionary.confirm(
                f"文件 '{target_path.name}' 已存在。是否覆盖?",
                default=False,
                qmark="❓"
            ).ask()
            if not overwrite:
                console.print(f"[yellow]提示：已取消覆盖现有文件 '{target_path.name}'。[/yellow]")
                return None

        # 列出可用的模板文件
        template_files = list(CONFIG_DIR.glob("*_template.yaml")) + list(CONFIG_DIR.glob("*_template.yml"))
        if not template_files:
            console.print(f"[bold red]错误：在 '{CONFIG_DIR}' 中未找到可用的模板文件 (例如 legacy_default_template.yaml)。[/bold red]")
            # 此时可以考虑是否依然允许创建空文件，或强制需要模板
            return None

        template_choices = sorted([t.name for t in template_files])

        selected_template_name = questionary.select(
            "请选择一个模板来创建新配置:",
            choices=template_choices,
            qmark="📋"
        ).ask()

        if not selected_template_name: # 用户取消
            console.print("[yellow]提示：未选择模板，已取消创建。[/yellow]")
            return None

        source_template_path = CONFIG_DIR / selected_template_name

        shutil.copy(source_template_path, target_path)
        console.print(f"[green]成功：新的配置文件 '{target_path.name}' 已从模板 '{selected_template_name}' 创建。[/green]")
        return target_path

    except Exception as e:
        console.print(f"[bold red]错误：创建新配置文件时发生错误: {e}[/bold red]")
        return None


def list_available_configs() -> None:
    """
    扫描 configs/ 目录，并以表格形式列出所有可用的 YAML 配置文件及其信息。
    """
    console = Console()
    if not CONFIG_DIR.exists() or not CONFIG_DIR.is_dir():
        console.print(f"[bold red]错误：配置目录 '{CONFIG_DIR}' 不存在或不是一个目录。[/bold red]")
        return

    config_files_paths = list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))

    if not config_files_paths:
        console.print(f"[yellow]提示：在 '{CONFIG_DIR}' 目录下未找到任何 YAML 配置文件。[/yellow]")
        return

    table = Table(title=f"AlphaFactory 可用的配置文件 (位于: {CONFIG_DIR.resolve()})", title_style="bold magenta")
    table.add_column("序号", style="dim", width=6, justify="center")
    table.add_column("配置文件名", style="cyan", no_wrap=True, min_width=20)
    table.add_column("大小 (KB)", style="green", justify="right", min_width=10)
    table.add_column("最后修改时间", style="yellow", min_width=20)

    for idx, file_path in enumerate(sorted(config_files_paths, key=lambda p: p.name)):
        try:
            stat_info = os.stat(file_path)
            size_kb = round(stat_info.st_size / 1024, 2)
            last_modified = Path(file_path).stat().st_mtime # 使用Path.stat()更一致
            # last_modified_str = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
            # 为了避免引入datetime，使用time.ctime
            last_modified_str = time.ctime(last_modified)

            table.add_row(
                str(idx + 1),
                file_path.name,
                f"{size_kb:.2f} KB",
                last_modified_str
            )
        except Exception as e:
            table.add_row(str(idx+1), file_path.name, "[red]错误[/red]", f"[red]{e}[/red]")


    console.print(table)
