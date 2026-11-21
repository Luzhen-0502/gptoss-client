# -*- coding: utf-8 -*-
"""
ui.py —— 豆包风格的桌面客户端（大肥鱼重写版）

这一版把上一版的「输入框被挤没」问题从根上修掉：
    - 整个窗口用 grid 布局，中间聊天区 weight=1 伸展，
      底部输入框和状态栏固定高度，永远不会被挤掉。
    - 输入区常驻：多行输入框 + 发送按钮 + 上传图片 + 画图 + 停止。
    - 右侧图片预览可以收起，收起后纯聊天（像豆包）。
    - 顶部自带模型状态和「下载模型」入口，缺模型不用去外面跑 bat。

线程模型不变：推理在后台线程，所有 UI 更新走 queue + after 轮询。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from config import Config, app_dir, check_model_files

log = logging.getLogger("gptoss.ui")

try:
    from PIL import Image, ImageTk  # type: ignore

    HAS_PIL = True
except Exception:  # noqa: BLE001
    HAS_PIL = False

from agent_core import AgentCore
from memory import SharedMemory
from tools import ToolRegistry, build_default_registry

TOOL_NAMES_ZH = {
    "see_image": "看图",
    "generate_image": "画图",
    "search_knowledge": "联网查资料",
    "get_current_time": "看时间",
    "run_command": "执行命令",
    "read_file": "读文件",
    "write_file": "写文件",
    "list_dir": "看目录",
    "open_path": "打开",
    "delete_path": "删除",
    "screenshot": "截屏",
    "system_info": "系统信息",
}

FONT = "Microsoft YaHei UI"

# 配色（豆包风，浅色）
BG = "#F4F5F7"
CARD = "#FFFFFF"
TEXT = "#1F2329"
SUB = "#8F959E"
BLUE = "#4D6BFE"
BLUE_DARK = "#3B55D9"
GREEN = "#12B76A"
RED = "#E5484D"
BORDER = "#E5E7EB"


def tool_zh(name: str) -> str:
    return TOOL_NAMES_ZH.get(name, name)


class ModelDownloader:
    """在界面里直接下载模型（走 hf-mirror，curl 断点续传）。"""

    MIRROR = "https://hf-mirror.com"
    FILES = [
        ("视觉模型眼睛 mmproj（1.2GB）", "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
         "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf", "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"),
        ("视觉模型本体（1.8GB）", "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
         "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
        ("生图模型 DreamShaper 8（4GB）", "Lykon/dreamshaper-8",
         None, "dreamshaper-8"),
        ("语言模型 GPT-OSS-20B（11.3GB）", "ggml-org/gpt-oss-20b-GGUF",
         "gpt-oss-20b-MXFP4.gguf", "gpt-oss-20b-mxfp4.gguf"),
    ]
    DS_FILES = [
        "model_index.json", "feature_extractor/preprocessor_config.json",
        "scheduler/scheduler_config.json", "text_encoder/config.json",
        "text_encoder/model.safetensors", "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json", "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]

    def __init__(self, cfg: Config, emit) -> None:
        self.cfg = cfg
        self.emit = emit
        self.models_dir = cfg.resolve_relative("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- 状态 ----------------------------------------------------------
    def missing(self) -> list[str]:
        return [p["title"] for p in check_model_files(self.cfg)]

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()

    # -- 下载 ----------------------------------------------------------
    def start(self) -> None:
        if self.running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="model-dl")
        self._thread.start()

    def _run(self) -> None:
        try:
            for title, repo, remote, local in self.FILES:
                if self._stop.is_set():
                    self.emit("dl_status", "已暂停")
                    return
                target = self.models_dir / local
                if remote is None:
                    # DreamShaper 目录
                    if (target / "unet" / "diffusion_pytorch_model.safetensors").exists():
                        self.emit("dl_status", f"{title} 已存在，跳过")
                        continue
                    for rel in self.DS_FILES:
                        if self._stop.is_set():
                            return
                        out = target / rel.replace("/", os.sep)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if out.exists() and out.stat().st_size > 0:
                            continue
                        self.emit("dl_status", f"下载 生图/{Path(rel).name}")
                        self._curl(f"{self.MIRROR}/{repo}/resolve/main/{rel}", out)
                else:
                    if target.exists() and target.stat().st_size > 50 * 1024 * 1024:
                        self.emit("dl_status", f"{title} 已存在，跳过")
                        continue
                    self.emit("dl_status", f"下载 {title}")
                    self._curl(f"{self.MIRROR}/{repo}/resolve/main/{remote}", target)
                self.emit("dl_done_one", title)
            # 补 config
            try:
                c = json.loads((app_dir() / "config.json").read_text(encoding="utf-8"))
            except Exception:
                c = {}
            c.setdefault("llm", {})["model_path"] = "models/gpt-oss-20b-mxfp4.gguf"
            c.setdefault("vision", {})["model_path"] = "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
            c["vision"]["mmproj_path"] = "models/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"
            c.setdefault("image", {})["model_path"] = "models/dreamshaper-8"
            (app_dir() / "config.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
            self.emit("dl_status", "全部下载完成 ✓")
            self.emit("dl_finished", True)
        except Exception as e:  # noqa: BLE001
            log.exception("模型下载失败")
            self.emit("dl_status", f"下载出错：{e}")
            self.emit("dl_finished", False)

    def _curl(self, url: str, out: Path, timeout: int = 3600) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["curl.exe", "-L", "-C", "-", "--retry", "8", "--retry-delay", "3",
               "--retry-all-errors", "--connect-timeout", "30", "-sS", "-o", str(out), url]
        try:
            subprocess.run(cmd, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"curl 出错：{e}") from e


class AgentApp:
    def __init__(self, cfg: Config, memory: SharedMemory, loader: Any,
                 registry: ToolRegistry | None = None) -> None:
        self.cfg = cfg
        self.memory = memory
        self.loader = loader
        self.registry = registry or build_default_registry()

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.downloader = ModelDownloader(cfg, self._emit)
        self.core = AgentCore(cfg=cfg, memory=memory, loader=loader, registry=self.registry,
                              emit=self._emit, confirm=self._ask_confirm)

        self.busy = False
        self.pending_image = ""
        self._photo: Any = None
        self._pil_image: Any = None
        self._resize_job: str | None = None
        self._network_warned = False
        self._worker: threading.Thread | None = None
        self._confirm_event: threading.Event | None = None
        self._confirm_answer = False
        self._right_visible = True

        self._build_window()
        self._build_ui()
        self._welcome()

        try:
            self.loader.set_status_callback(self._on_model_status)
        except Exception:
            pass

        self.root.after(80, self._poll)

    # ==================================================================
    # 窗口
    # ==================================================================
    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("GPT-OSS 专属 Agent")
        self.root.configure(bg=BG)
        w, h = 1360, 860
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(w, sw - 80)
        h = min(h, sh - 100)
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 30)}")
        self.root.minsize(900, 600)

        fs = int(self.cfg.get("app.font_size", 11))
        self.f_normal = (FONT, fs)
        self.f_bold = (FONT, fs, "bold")
        self.f_small = (FONT, max(8, fs - 1))
        self.f_title = (FONT, fs + 3, "bold")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # 布局：grid 四行，中间伸展，底部固定 —— 输入框永远不会被挤掉
    # ==================================================================
    def _build_ui(self) -> None:
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=0)   # 顶栏
        root.grid_rowconfigure(1, weight=1)   # 中间（聊天 + 预览）伸展
        root.grid_rowconfigure(2, weight=0)   # 输入区固定
        root.grid_rowconfigure(3, weight=0)   # 状态栏固定

        # ---------------- 0. 顶栏 ----------------
        top = tk.Frame(root, bg=CARD, height=54)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)
        top.grid_columnconfigure(1, weight=1)

        tk.Label(top, text="🐟", font=(FONT, 18), bg=CARD).grid(row=0, column=0, padx=(16, 6), pady=8)
        tk.Label(top, text="GPT-OSS 专属 Agent", font=self.f_title, bg=CARD, fg=TEXT)\
            .grid(row=0, column=1, sticky="w")
        tk.Label(top, text="看图 · 查资料 · 画图 · 干活", font=self.f_small, bg=CARD, fg=SUB)\
            .grid(row=0, column=2, padx=10)

        self.model_var = tk.StringVar(value="模型未加载")
        tk.Label(top, textvariable=self.model_var, font=self.f_small, bg=CARD, fg=SUB)\
            .grid(row=0, column=3, padx=8)

        self.btn_dl = tk.Button(top, text="下载模型", font=self.f_small, bg=BLUE, fg="white",
                                relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
                                command=self._open_downloader)
        if self.downloader.missing():
            self.btn_dl.grid(row=0, column=4, padx=(0, 16))
        else:
            self.btn_dl.configure(bg="#E8EAED", fg=SUB, text="模型已就绪")
            self.btn_dl.grid(row=0, column=4, padx=(0, 16), pady=14)

        tk.Frame(root, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew")

        # ---------------- 1. 中间：左聊天 + 右预览 ----------------
        mid = tk.Frame(root, bg=BG)
        mid.grid(row=1, column=0, sticky="nsew", padx=12, pady=(10, 0))
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=2)

        # 左：聊天
        left = tk.Frame(mid, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.chat = tk.Text(left, wrap=tk.WORD, font=self.f_normal, bg=CARD, fg=TEXT,
                            relief=tk.FLAT, padx=14, pady=12, spacing1=2, spacing3=8,
                            state=tk.DISABLED, cursor="arrow", selectbackground="#DCE3FF")
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        self.chat.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.chat.tag_configure("user", foreground=BLUE, font=self.f_bold, lmargin1=80, justify=tk.RIGHT)
        self.chat.tag_configure("user_body", foreground=TEXT, lmargin1=80, justify=tk.RIGHT)
        self.chat.tag_configure("bot", foreground=BLUE, font=self.f_bold)
        self.chat.tag_configure("bot_body", foreground=TEXT)
        self.chat.tag_configure("tool", foreground=GREEN, font=self.f_bold)
        self.chat.tag_configure("tool_body", foreground="#4B5563", lmargin1=18)
        self.chat.tag_configure("sys", foreground=SUB, font=self.f_small)
        self.chat.tag_configure("err", foreground=RED, font=self.f_bold)
        self.chat.tag_configure("warn", foreground="#B45309")
        self.chat.tag_configure("think", foreground="#9CA3AF", font=self.f_small)

        # 右：图片预览
        self.right = tk.Frame(mid, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.grid_rowconfigure(1, weight=1)
        self.right.grid_columnconfigure(0, weight=1)

        rhead = tk.Frame(self.right, bg=CARD)
        rhead.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        rhead.grid_columnconfigure(0, weight=1)
        tk.Label(rhead, text="图片", font=self.f_bold, bg=CARD, fg=TEXT).grid(row=0, column=0, sticky="w")
        self.img_var = tk.StringVar(value="")
        tk.Label(rhead, textvariable=self.img_var, font=self.f_small, bg=CARD, fg=SUB)\
            .grid(row=0, column=1, sticky="e")

        self.canvas = tk.Canvas(self.right, bg="#FAFBFC", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.desc = tk.Text(self.right, height=5, wrap=tk.WORD, font=self.f_small, bg="#FAFBFC",
                            fg="#4B5563", relief=tk.FLAT, padx=10, pady=8, state=tk.DISABLED)
        self.desc.grid(row=2, column=0, sticky="ew", padx=8, pady=(6, 8))

        # ---------------- 2. 输入区（固定） ----------------
        bottom = tk.Frame(root, bg=BG)
        bottom.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 6))
        bottom.grid_columnconfigure(0, weight=1)

        box = tk.Frame(bottom, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        box.grid(row=0, column=0, sticky="ew")
        box.grid_columnconfigure(0, weight=1)

        self.input = tk.Text(box, height=4, wrap=tk.WORD, font=self.f_normal, bg=CARD, fg=TEXT,
                             relief=tk.FLAT, padx=12, pady=8, insertbackground=BLUE)
        self.input.grid(row=0, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)
        self.input.focus_set()

        bar = tk.Frame(box, bg=CARD)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        bar.grid_columnconfigure(2, weight=1)

        self.btn_up = tk.Button(bar, text="📎 上传图片", font=self.f_small, bg="#F2F3F5", fg="#4B5563",
                                relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=self._on_upload)
        self.btn_up.grid(row=0, column=0, padx=(0, 6))

        self.btn_stop = tk.Button(bar, text="■ 停止", font=self.f_small, bg="#F2F3F5", fg=SUB,
                                  relief=tk.FLAT, padx=10, pady=5, cursor="hand2", state=tk.DISABLED,
                                  command=self._on_stop)
        self.btn_stop.grid(row=0, column=1, padx=(0, 6))

        tk.Label(bar, text="Enter 发送 · Shift+Enter 换行", font=self.f_small, bg=CARD, fg=SUB)\
            .grid(row=0, column=2, sticky="w", padx=8)

        self.btn_send = tk.Button(bar, text="发送  ➤", font=self.f_bold, bg=BLUE, fg="white",
                                  relief=tk.FLAT, padx=22, pady=6, cursor="hand2",
                                  activebackground=BLUE_DARK, activeforeground="white",
                                  command=self._on_send)
        self.btn_send.grid(row=0, column=3, padx=(6, 0))

        # ---------------- 3. 状态栏（固定） ----------------
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(root, textvariable=self.status_var, font=self.f_small, bg=BG, fg=SUB, anchor="w")\
            .grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 6))

    # ==================================================================
    # 欢迎语
    # ==================================================================
    def _welcome(self) -> None:
        self._append("你好呀，我是你的 GPT-OSS 专属助手。\n", "bot")
        self._append(
            "· 直接打字聊天，我能联网查资料、画图、帮你操作这台电脑\n"
            "· 上传一张图，我能看懂它，再按你的要求改图或出新图\n"
            "· 删文件、跑危险命令之前，我一定会先弹窗问你\n\n",
            "sys",
        )
        problems = check_model_files(self.cfg)
        if problems:
            self._append(f"⚠️ 还缺 {len(problems)} 个模型文件，点右上角【下载模型】就能一键下好。\n"
                         "   （合计约 19GB，第一次下载要等一会儿）\n\n", "warn")
        else:
            self._append("模型文件齐全，可以直接开始。\n\n", "sys")

    # ==================================================================
    # 聊天区写入
    # ==================================================================
    def _append(self, text: str, tag: str = "bot_body") -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        try:
            if int(self.chat.index("end-1c").split(".")[0]) > 3000:
                self.chat.delete("1.0", "1000.0")
        except Exception:
            pass
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    # ==================================================================
    # 线程桥
    # ==================================================================
    def _emit(self, kind: str, payload: Any = None) -> None:
        self.events.put((kind, payload))

    def _on_model_status(self, msg: str) -> None:
        self.events.put(("status", msg))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                try:
                    self._handle(kind, payload)
                except Exception:
                    log.exception("处理事件 %s 出错", kind)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "user":
            self._append("你\n", "user")
            self._append(f"{payload}\n\n", "user_body")

        elif kind == "thinking":
            t = str(payload).strip()
            if t:
                self._append(f"💭 {t[:300]}{'…' if len(t) > 300 else ''}\n\n", "think")

        elif kind == "narrate":
            self._append(f"{payload}\n\n", "bot")

        elif kind == "tool_call":
            name = str(payload.get("name", ""))
            args = payload.get("args") or {}
            brief = "，".join(f"{k}={v}" for k, v in list(args.items())[:2])
            if len(brief) > 100:
                brief = brief[:100] + "…"
            self._append(f"🔧 {tool_zh(name)}", "tool")
            self._append(f"  {brief}\n", "tool_body")
            self.status_var.set(f"正在 {tool_zh(name)}…")

        elif kind == "tool_result":
            r = str(payload.get("result", "")).strip().replace("\n", " ")
            self._append(f"   {r[:200]}{'…' if len(r) > 200 else ''}\n\n", "tool_body")

        elif kind == "status":
            self.status_var.set(str(payload))

        elif kind == "progress":
            try:
                s, t = payload
                self.status_var.set(f"思考中… 第 {s}/{t} 步")
            except Exception:
                pass

        elif kind == "retry":
            self._append(f"⚠️ {payload}\n\n", "warn")

        elif kind == "vision_done":
            self._set_desc(str((payload or {}).get("description", "")))

        elif kind == "knowledge_done":
            info = payload or {}
            s = str(info.get("summary", ""))[:160]
            self._append(f"🌐 查到「{info.get('title', '')}」：{s}…\n\n", "tool_body")

        elif kind == "image_ready":
            self._show_image(str(payload), "生成")
            self._append("🖼 图在右边，看看行不行。\n\n", "bot_body")

        elif kind == "network_error":
            if not self._network_warned:
                self._network_warned = True
                messagebox.showwarning("没有网络", f"连不上网，这次没法查资料。\n\n{payload}")

        elif kind == "final":
            self._append("助手\n", "bot")
            self._append(f"{payload}\n\n", "bot_body")
            self.status_var.set("就绪")

        elif kind == "error":
            self._append(f"❌ {payload}\n\n", "err")
            messagebox.showerror("出错了", str(payload))

        elif kind == "ask_confirm":
            ans = messagebox.askyesno("⚠️ 需要你确认", str(payload))
            self._confirm_answer = bool(ans)
            ev = self._confirm_event
            if ev:
                ev.set()
            if not ans:
                self._append("⛔ 你拒绝了这次操作。\n\n", "warn")

        elif kind == "done":
            self._set_busy(False)

        elif kind == "cancelled":
            self._append("⏹ 已停止。\n\n", "warn")
            self._set_busy(False)

        elif kind == "exception":
            self._append(f"❌ {payload}\n\n", "err")
            self._set_busy(False)
            messagebox.showerror("出错了", str(payload))

        # ---- 模型下载 ----
        elif kind == "dl_status":
            self.status_var.set(str(payload))
            self._append(f"{payload}\n", "sys")

        elif kind == "dl_done_one":
            self._append(f"✓ {payload} 完成\n", "sys")

        elif kind == "dl_finished":
            self._set_busy(False)
            if payload:
                self._append("\n全部模型下载完成，可以开始用了。\n\n", "bot")
                self.btn_dl.configure(text="模型已就绪", bg="#E8EAED", fg=SUB)
            else:
                self._append("\n下载没完成，可以再点一次继续。\n\n", "warn")

    # ==================================================================
    # 图片
    # ==================================================================
    def _show_image(self, path: str, source: str = "") -> None:
        if not HAS_PIL:
            return
        try:
            img = Image.open(path)
            self._pil_image = img.copy()
            img.close()
        except Exception:
            return
        self.img_var.set(f"{source}：{Path(path).name}")
        self._render()
        rec = self.memory.image_by_path(path)
        if rec:
            txt = str(rec.get("description", ""))
            if rec.get("prompt"):
                txt += f"\n\n【提示词】{rec['prompt']}"
            self._set_desc(txt)

    def _set_desc(self, text: str) -> None:
        self.desc.configure(state=tk.NORMAL)
        self.desc.delete("1.0", tk.END)
        self.desc.insert("1.0", text.strip())
        self.desc.configure(state=tk.DISABLED)

    def _on_canvas_resize(self, _e=None) -> None:
        if self._resize_job:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(120, self._render)

    def _render(self) -> None:
        self._resize_job = None
        if not HAS_PIL or self._pil_image is None:
            return
        cw, ch = max(60, self.canvas.winfo_width()), max(60, self.canvas.winfo_height())
        try:
            im = self._pil_image.copy()
            im.thumbnail((cw - 12, ch - 12), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(im)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        except Exception:
            pass

    # ==================================================================
    # 动作
    # ==================================================================
    def _on_enter(self, e) -> str:
        if e.state & 0x0001:
            return ""
        self._on_send()
        return "break"

    def _on_upload(self) -> None:
        if self.busy:
            return
        p = filedialog.askopenfilename(
            title="选一张图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")],
        )
        if not p:
            return
        self.pending_image = p
        self._show_image(p, "已上传")
        self.memory.set_task(pending_image=p)
        self._append(f"📎 已选图：{Path(p).name}\n", "sys")
        self._append("  写一句要求（比如「改成水彩风」），然后发送。\n\n", "sys")
        self.status_var.set("图片已就绪")

    def _on_stop(self) -> None:
        self.core.cancel()
        self.status_var.set("正在停止…")
        self.btn_stop.configure(state=tk.DISABLED)

    def _on_send(self) -> None:
        if self.busy:
            return
        text = self.input.get("1.0", tk.END).strip()
        img = self.pending_image
        if not text and not img:
            messagebox.showinfo("还没说做什么", "先在下面输入框写一句，或者点【上传图片】。")
            self.input.focus_set()
            return
        if not HAS_PIL:
            messagebox.showerror("缺少组件", "没有 Pillow，图片功能不可用。")
            return

        problems = check_model_files(self.cfg)
        if problems:
            if not messagebox.askyesno("模型还没准备好",
                                       f"还缺 {len(problems)} 个模型文件，现在去下载吗？"):
                return
            self._open_downloader()
            return

        self.input.delete("1.0", tk.END)
        self.pending_image = ""
        self._network_warned = False
        self._set_busy(True)
        self.status_var.set("思考中…")
        if img:
            self.memory.set_task(pending_image=img)

        self._worker = threading.Thread(target=self._run, args=(text, img), daemon=True)
        self._worker.start()

    def _run(self, text: str, img: str) -> None:
        try:
            self.core.run(text, img or "")
            self.events.put(("done", None))
        except Exception as e:  # noqa: BLE001
            from agent_core import AgentCancelled

            if isinstance(e, AgentCancelled):
                self.events.put(("cancelled", None))
                return
            log.exception("Agent 运行失败")
            self.events.put(("exception", f"{type(e).__name__}: {e}"))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        st = tk.DISABLED if busy else tk.NORMAL
        try:
            self.btn_send.configure(state=st, bg="#C7CEDB" if busy else BLUE)
            self.btn_up.configure(state=st)
            self.btn_stop.configure(state=tk.NORMAL if busy else tk.DISABLED)
        except Exception:
            pass
        if not busy:
            self.status_var.set("就绪")

    def _ask_confirm(self, msg: str) -> bool:
        self._confirm_event = threading.Event()
        self._confirm_answer = False
        self.events.put(("ask_confirm", msg))
        if not self._confirm_event.wait(timeout=300):
            return False
        self._confirm_event = None
        return bool(self._confirm_answer)

    # ==================================================================
    # 模型下载对话框
    # ==================================================================
    def _open_downloader(self) -> None:
        if self.downloader.running():
            messagebox.showinfo("正在下载", "模型正在下载中，看下面的进度就行。")
            return
        if not self.downloader.missing():
            messagebox.showinfo("模型齐全", "三个模型文件都在，不用再下了。")
            return
        miss = "\n".join(f"· {m}" for m in self.downloader.missing())
        if not messagebox.askyesno(
            "下载模型",
            f"还缺这些：\n{miss}\n\n"
            "合计约 19GB，走 hf-mirror 镜像，支持断点续传。\n"
            "下载期间可以继续看界面，只是不能用 AI。\n\n现在开始下载吗？",
        ):
            return
        self._set_busy(True)
        self.downloader.start()
        self.status_var.set("模型下载中…")

    # ==================================================================
    # 关闭
    # ==================================================================
    def _on_close(self) -> None:
        try:
            self.core.cancel()
        except Exception:
            pass
        try:
            self.downloader.stop()
        except Exception:
            pass
        try:
            self.memory.save()
        except Exception:
            pass
        try:
            self.loader.shutdown()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO)
    from config import Config as Cfg
    from loader import ModelManager
    from memory import SharedMemory as Mem

    c = Cfg()
    m = Mem(Path(tempfile.mkdtemp()) / "mem.json")
    mm = ModelManager(c)
    app = AgentApp(c, m, mm, build_default_registry())
    if "--smoke" in sys.argv:
        app.root.after(1500, app.root.destroy)
    app.run()
