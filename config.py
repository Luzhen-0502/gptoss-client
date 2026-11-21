# -*- coding: utf-8 -*-
"""
config.py —— 全局配置模块（大肥鱼手写）

职责：
  1. 计算程序根目录（区分「源码运行」和「PyInstaller 打包后的 exe 运行」）；
  2. 读写 config.json（模型路径、推理参数、界面参数都在这里）；
  3. 统一日志落盘（因为打包后是无控制台窗口的，报错只能写文件 + 弹窗）；
  4. 提供「模型文件体检」函数，缺哪个模型就用中文告诉用户缺什么、放哪里。

设计原则：
  - 模型权重一律外置到 exe 同级的 models/ 目录，绝不塞进 exe；
  - 所有相对路径都相对「程序根目录」解析，双击 exe 在哪都能跑。
"""

from __future__ import annotations

import copy
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 目录计算
# ---------------------------------------------------------------------------


def app_dir() -> Path:
    """返回程序根目录。

    - 源码运行时：本文件所在目录；
    - 打包成单 exe 后：exe 文件所在目录（不是临时解压目录 _MEIPASS）。

    注意：模型权重、config.json、data/、outputs/ 都放在这个目录下，
    而不是 sys._MEIPASS —— 那个临时目录每次启动都会重建/清理。
    """
    if getattr(sys, "frozen", False):
        # PyInstaller onefile 会把 exe 路径写进 sys.executable
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    """返回打包时被塞进 exe 的内部资源目录（源码运行时等同 app_dir）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """把配置里的相对路径解析成绝对路径。"""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (app_dir() / p).resolve()


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

# 各模型在 models/ 下的默认文件名（README 里也会写一遍，方便对照）
DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "title": "GPT-OSS 专属 Agent",
        "version": "1.0.0",
        # 界面字号（微软雅黑），高分屏下可自行调大
        "font_size": 11,
        "window_size": [1280, 820],
    },
    "llm": {
        # ★ 语言模型：GPT-OSS-20B，4bit（MXFP4）GGUF，常驻内存当大脑
        "model_path": "models/gpt-oss-20b-mxfp4.gguf",
        "n_ctx": 8192,          # 上下文窗口；越大越吃内存
        "n_batch": 512,         # 批大小
        "n_threads": 0,         # 0 = 自动（物理核数）
        "n_gpu_layers": 0,      # 0 = 纯 CPU（核显跑 llama.cpp 没有收益，容易崩）
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1200,     # 单次生成上限
        "repeat_penalty": 1.05,
        "seed": -1,             # -1 = 随机
        "flash_attn": False,    # CPU 上一般别开
    },
    "vision": {
        # ★ 视觉理解：轻量小 VLM，按需加载，用完卸载
        # 后端 auto 会先试 llama.cpp GGUF，失败再试 transformers
        "backend": "auto",      # auto / llama_cpp / transformers
        "model_path": "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_path": "models/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf",
        # transformers 后备后端（当 GGUF 不可用时）
        "hf_model_path": "models/Qwen2.5-VL-3B-Instruct",
        "n_ctx": 4096,
        "n_threads": 0,
        "max_tokens": 512,
        "temperature": 0.2,
        "max_image_side": 896,  # 送模型前把图缩到最长边 896，省内存省时间
    },
    "image": {
        # ★ 生图：轻量文生图（SD1.5 / SDXL-Turbo），按需加载，用完卸载
        #   既支持 diffusers 目录，也支持单个 .safetensors 文件
        "model_path": "models/sd-v1-5",   # 或 models/dreamshaper_8.safetensors
        "variant": "sd15",                # sd15 / sdxl-turbo（影响提示词规整策略）
        "steps": 12,
        "guidance": 7.5,
        "width": 512,
        "height": 512,
        "negative_prompt": "lowres, bad anatomy, bad hands, text, watermark, blurry, jpeg artifacts",
        "quality_suffix": "masterpiece, best quality, highly detailed, sharp focus",
        "max_prompt_chars": 600,
        "seed": -1,
        "max_side": 768,          # 生成的边长上限（CPU 上再大就慢到不能忍）
    },
    "agent": {
        "max_steps": 6,           # 调度闭环最多转几圈（防止死循环）
        "history_turns": 8,       # 塞进上下文的历史对话条数
        "memory_block_chars": 1800,  # 记忆摘要注入 prompt 的最大字数
        "language": "zh",         # 模型最终回答用中文
        "auto_image_goal": (
            "请先用 see_image 工具看懂我刚上传的图片，"
            "然后把画面整理成一段英文的生图提示词，再调用 generate_image 生成一张新图。"
        ),
    },
    "computer": {
        # ★ 电脑操作权限：让 Agent 能真的帮你干活（执行命令 / 读写文件 / 截图）
        "enabled": True,
        "command_timeout": 120,          # 单条命令最长跑多少秒
        "max_output_chars": 4000,        # 命令输出回填给模型的最大字数
        "max_read_chars": 12000,         # 读文件的最大字数
        "confirm_delete": True,          # 删除文件/文件夹要不要弹窗问用户
        "confirm_dangerous": True,       # 命中黑名单的命令要不要弹窗问用户
        # 命中这些词就会先弹窗确认（不是禁用，是问一句）
        "blacklist": [
            "format ",
            "diskpart",
            "del /s",
            "del /f /s",
            "rd /s",
            "rmdir /s",
            "shutdown",
            "bcdedit",
            "reg delete",
            "vssadmin",
            "cipher /w",
            "takeown",
            "icacls",
            "net user",
            "net localgroup",
            "taskkill /f /im",
            "sc delete",
        ],
    },
    "paths": {
        "memory_file": "data/memory.json",
        "output_dir": "outputs",
        "log_file": "logs/app.log",
        "config_file": "config.json",
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    """把 patch 深度合并进 base 的副本（用于老配置文件升级，缺的键自动补默认值）。"""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------


class Config:
    """配置单例：读一次，全局共享；需要改参数就改 config.json 后重启程序。"""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = resolve_path(
            path or (app_dir() / DEFAULT_CONFIG["paths"]["config_file"])
        )
        # ★ 所有相对路径都以「config.json 所在目录」为基准，而不是代码目录。
        #   这样无论从哪启动、配置文件放哪，模型/输出/日志都能找对地方。
        self.base_dir = self.path.parent
        self.data: dict[str, Any] = {}
        self.load()

    # -- 读写 --------------------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                # 深度合并，保证新版本新增的配置项自动补齐
                self.data = _deep_merge(DEFAULT_CONFIG, user_cfg if isinstance(user_cfg, dict) else {})
            except Exception:
                # 配置文件坏了不能直接崩，用默认值继续跑，坏文件另存一份
                self.data = copy.deepcopy(DEFAULT_CONFIG)
                try:
                    self.path.rename(self.path.with_suffix(".broken.json"))
                except Exception:
                    pass
        else:
            self.data = copy.deepcopy(DEFAULT_CONFIG)
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # -- 便捷访问 ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        """取一段配置，比如 cfg.section("llm")。"""
        return self.data.setdefault(name, {})

    def get(self, dotted: str, default: Any = None) -> Any:
        """按 "llm.max_tokens" 这种点号路径取值。"""
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    # -- 路径 --------------------------------------------------------------
    def resolve_relative(self, value: str) -> Path:
        """把配置里的相对路径解析成绝对路径（基准 = config.json 所在目录）。"""
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def path_of(self, dotted: str, default: str = "") -> Path:
        return self.resolve_relative(self.get(dotted, default) or default)

    @property
    def output_dir(self) -> Path:
        d = self.path_of("paths.output_dir", "outputs")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def memory_path(self) -> Path:
        return self.path_of("paths.memory_file", "data/memory.json")

    @property
    def log_path(self) -> Path:
        return self.path_of("paths.log_file", "logs/app.log")


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


def setup_logging(cfg: Config, level: int = logging.INFO) -> logging.Logger:
    """初始化日志：写 logs/app.log（滚动），同时尽量写 console（源码运行时能看见）。

    打包成无控制台 exe 后 stdout 可能是 None，这里用 try 包住，绝不让日志把程序搞崩。
    """
    log_path = cfg.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # 清掉可能存在的旧 handler，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    if sys.stdout is not None and sys.stderr is not None:
        try:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            root.addHandler(sh)
        except Exception:
            pass

    log = logging.getLogger("gptoss")
    log.info("=" * 70)
    log.info("程序根目录：%s", app_dir())
    log.info("Python：%s", sys.version.split()[0])
    return log


# ---------------------------------------------------------------------------
# 模型文件体检
# ---------------------------------------------------------------------------


def check_model_files(cfg: Config) -> list[dict[str, str]]:
    """检查三个模型文件在不在，返回缺什么（全中文，给 UI 直接弹窗用）。

    返回示例：
        [{"kind": "llm", "path": "...", "title": "语言模型 GPT-OSS-20B", "hint": "..."}]
    """
    problems: list[dict[str, str]] = []

    def _add(kind: str, title: str, path: Path, hint: str) -> None:
        problems.append(
            {"kind": kind, "title": title, "path": str(path), "hint": hint}
        )

    # 1) 语言模型
    llm_path = cfg.path_of("llm.model_path", "models/gpt-oss-20b-mxfp4.gguf")
    if not llm_path.exists():
        _add(
            "llm",
            "语言模型（GPT-OSS-20B 4bit GGUF）",
            llm_path,
            "去 HuggingFace 搜 gpt-oss-20b GGUF（MXFP4 版约 11~12GB），"
            "下载后改名放进 models/ 目录。文件名要和 config.json 里的 llm.model_path 一致。",
        )

    # 2) 视觉模型（GGUF 后端至少要模型本体；mmproj 缺了反而能用 transformers 后备）
    vis_path = cfg.path_of("vision.model_path", "")
    hf_path = cfg.path_of("vision.hf_model_path", "")
    backend = str(cfg.get("vision.backend", "auto")).lower()
    if backend in ("auto", "llama_cpp"):
        if not vis_path.exists() and not hf_path.exists():
            _add(
                "vision",
                "视觉理解模型（Qwen2.5-VL-3B）",
                vis_path,
                "下载 Qwen2.5-VL-3B-Instruct 的 GGUF（Q4_K_M 约 2GB）+ mmproj 投影文件（约 600MB），"
                "两个都放进 models/；或者把 transformers 版模型目录放到 models/Qwen2.5-VL-3B-Instruct。",
            )
    elif not hf_path.exists():
        _add("vision", "视觉理解模型（Qwen2.5-VL-3B）", hf_path, "把 HF 版模型目录放进 models/。")

    # 3) 生图模型
    img_path = cfg.path_of("image.model_path", "")
    if not img_path.exists():
        _add(
            "image",
            "生图模型（SD1.5 / SDXL-Turbo 量化）",
            img_path,
            "把 SD1.5 的 diffusers 目录（或单个 .safetensors 文件）放进 models/。"
            "推荐：stable-diffusion-v1-5（约 4GB）或 DreamShaper 8 单文件（约 2GB）。",
        )

    return problems


def model_file_report(cfg: Config) -> str:
    """把体检结果渲染成一段中文文本（给启动时打印/写日志）。"""
    problems = check_model_files(cfg)
    if not problems:
        return "三个模型文件都已就位。"
    lines = [f"缺少 {len(problems)} 个模型文件："]
    for i, p in enumerate(problems, 1):
        lines.append(f"{i}. {p['title']}")
        lines.append(f"   期望路径：{p['path']}")
        lines.append(f"   怎么办：{p['hint']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 单独跑这个文件可以做一次体检：python config.py
    logging.basicConfig(level=logging.INFO)
    c = Config()
    print("配置文件：", c.path)
    print(model_file_report(c))
