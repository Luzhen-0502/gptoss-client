# -*- mode: python ; coding: utf-8 -*-
"""
gptoss-agent.spec —— PyInstaller 打包配置（大肥鱼手写）

作用：把整个项目打成一个 Windows 单 exe，双击就开，没有黑框。

关键点：
  1. console=False —— 不弹控制台窗口；
  2. 模型权重【不打包】—— models/ 目录留在 exe 旁边，用户自己放；
  3. diffusers 大量动态导入，必须 collect_all，否则运行时找不到管线；
  4. llama_cpp 的 .pyd/.dll 要 collect_dynamic_libs 收进去。

用法（在项目根目录）：
    python -m PyInstaller --noconfirm --clean ^
        --distpath "D:\\gptoss-build\\dist" ^
        --workpath "D:\\gptoss-build\\build" ^
        gptoss-agent.spec
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH)  # noqa: F821  (SPECPATH 是 PyInstaller 注入的变量)

datas = []
binaries = []
hiddenimports = []

# ---------------------------------------------------------------------------
# 1. diffusers：它里面每个 pipeline 都是运行时按名字 import 的，
#    静态分析找不到，必须整包收进来（体积大一点，但能保证跑得起来）
# ---------------------------------------------------------------------------
try:
    _d, _b, _h = collect_all("diffusers")
    datas += _d
    binaries += _b
    hiddenimports += _h
except Exception as e:  # noqa: BLE001
    print(f"[warn] collect_all(diffusers) 失败：{e}")

# ---------------------------------------------------------------------------
# 2. transformers：主要收它的 tokenizer 配置文件
# ---------------------------------------------------------------------------
try:
    datas += collect_data_files("transformers")
except Exception as e:  # noqa: BLE001
    print(f"[warn] collect_data_files(transformers) 失败：{e}")

# ---------------------------------------------------------------------------
# 3. llama.cpp 的原生扩展
# ---------------------------------------------------------------------------
try:
    binaries += collect_dynamic_libs("llama_cpp")
except Exception as e:  # noqa: BLE001
    print(f"[warn] collect_dynamic_libs(llama_cpp) 失败：{e}")

hiddenimports += [
    "llama_cpp",
    "llama_cpp._llama_cpp",
    "llama_cpp.llama_chat_format",
    "llama_cpp.llama_grammar",
    "llama_cpp.llama_tokenizer",
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    "PIL.ImageDraw",
    "PIL.ImageOps",
    "numpy",
    "safetensors",
    "safetensors.torch",
]

try:
    hiddenimports += collect_submodules("safetensors")
except Exception:
    pass

# 项目自己的模块（正常情况下 Analysis 能顺着 import 找到，写上是双保险）
hiddenimports += ["config", "memory", "tools", "loader", "agent_core", "ui"]

# ---------------------------------------------------------------------------
# 4. 排掉明显用不到的大块头，控制体积
# ---------------------------------------------------------------------------
excludes = [
    "matplotlib",
    "IPython",
    "notebook",
    "jupyter",
    "jupyterlab",
    "pytest",
    "sphinx",
    "tensorboard",
    "pandas",
    "PyQt5",
    "PySide2",
    "PySide6",
    "torchvision",   # 只做文生图，不需要 torchvision
    "torchaudio",
]

# ---------------------------------------------------------------------------
# 5. 分析
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# 6. 单文件 exe（onefile：把 binaries/datas 全塞进 exe）
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GPTOSS-Agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # ★ 无控制台黑框
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "app.ico"),   # ★ 自己的图标
)
