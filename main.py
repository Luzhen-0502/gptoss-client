# -*- coding: utf-8 -*-
"""
main.py —— 程序入口（大肥鱼手写）

职责很单一：
  1. 打开高 DPI 感知，别让界面糊成一片；
  2. 初始化配置 / 日志 / 记忆 / 模型管理器 / 工具注册表；
  3. 把所有东西塞进 UI，开跑；
  4. 兜住一切未捕获异常（打包后没有黑框，出错只能弹窗 + 写日志）。

打包命令见 README.md / build.bat。
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import sys
import traceback

from config import Config, app_dir, setup_logging
from loader import ModelManager
from memory import SharedMemory
from tools import build_default_registry
from ui import AgentApp


class _LogStream:
    """把第三方库的 print() 重定向到日志文件。

    打包成 --noconsole 的单 exe 后，sys.stdout 是 None，
    某些库（tqdm、diffusers）往里 print 会直接抛异常，所以必须兜住。
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def write(self, s: str) -> int:
        text = (s or "").strip()
        if text:
            self.logger.debug("[stdout] %s", text)
        return len(s or "")

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _enable_high_dpi() -> None:
    """Windows 高分屏下让界面清晰，而不是被系统拉伸糊掉。"""
    try:
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            pass


def _install_global_handler(root, logger: logging.Logger) -> None:
    """未捕获异常 → 写日志 + 中文弹窗，绝不让程序默默消失。"""
    from tkinter import messagebox

    def handle(exc_type, exc_value, exc_tb) -> None:
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("未捕获异常：\n%s", detail)
        try:
            messagebox.showerror(
                "程序出错了",
                f"发生了一个没预料到的错误：\n\n{exc_type.__name__}: {exc_value}\n\n"
                "详细日志已经写到程序目录下的 logs/app.log。\n\n"
                "你可以关掉窗口重开一次；如果一直这样，把日志文件发给开发者。",
            )
        except Exception:
            pass

    sys.excepthook = handle
    if root is not None:
        # Tkinter 回调里抛的异常走这个口子
        root.report_callback_exception = handle


def main() -> int:
    multiprocessing.freeze_support()  # PyInstaller + torch 打包后必须
    _enable_high_dpi()

    cfg = Config()
    logger = setup_logging(cfg)
    # 重定向 stdout/stderr，防止无控制台环境下第三方库 print 崩掉
    if sys.stdout is None:
        sys.stdout = _LogStream(logger)  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _LogStream(logger)  # type: ignore[assignment]

    logger.info("程序启动，根目录：%s", app_dir())

    try:
        memory = SharedMemory(cfg.memory_path)
        logger.info("共享记忆已加载：%s", cfg.memory_path)
    except Exception as e:  # noqa: BLE001
        logger.exception("加载记忆失败，改用全新记忆")
        memory = SharedMemory(cfg.memory_path.with_suffix(".new.json"))

    try:
        registry = build_default_registry()
        logger.info("工具注册表已就绪，共 %d 个工具", len(registry.names()))
    except Exception as e:  # noqa: BLE001
        logger.exception("工具注册表初始化失败")
        from tools import ToolRegistry

        registry = ToolRegistry()

    try:
        loader = ModelManager(cfg)
    except Exception as e:  # noqa: BLE001
        logger.exception("模型管理器初始化失败")
        raise

    app = AgentApp(cfg, memory, loader, registry)
    _install_global_handler(app.root, logger)

    try:
        app.run()
    finally:
        try:
            memory.save()
        except Exception:
            pass
        try:
            loader.shutdown()
        except Exception:
            pass
        logger.info("程序退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
