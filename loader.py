# -*- coding: utf-8 -*-
"""
loader.py —— 三个模型的加载 / 卸载（大肥鱼手写）

本机只有 32GB 内存、16GB 核显、没有独显，三个模型同时常驻会直接把内存撑爆。
所以这里做了严格的「单槽互斥」：

    内存里同一时刻只允许一个模型：
      语言模型（GPT-OSS-20B，常驻） ←→ 视觉模型（Qwen2.5-VL-3B，按需）
                                     ←→ 生图模型（SD1.5，按需）

调度器要换模型时只有一个入口：with mm.borrow("vision") as vlm: ...
    - 进去前：先把当前模型卸干净（del + gc.collect + 关掉 llama.cpp 上下文）
    - 用完退出：自动卸载视觉/生图模型，并把语言模型重新加载回来（继续当大脑）

为什么重新加载语言模型不算慢：
    llama.cpp 默认 mmap 权重文件，卸载后文件页还留在系统缓存里，
    第二次加载通常只要几秒，而不用重新读 12GB 硬盘。
"""

from __future__ import annotations

import base64
import ctypes
import gc
import logging
import math
import os
import random
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from config import Config

log = logging.getLogger("gptoss.loader")


# ===========================================================================
# 异常
# ===========================================================================


class ModelNotReady(RuntimeError):
    """模型没准备好（文件缺失、加载失败、依赖没装）。"""


class ModelFileMissing(ModelNotReady):
    """模型文件在磁盘上找不到。"""


# ===========================================================================
# 内存监控（纯 ctypes，不引入 psutil）
# ===========================================================================


def available_ram_gb() -> float:
    """返回当前可用物理内存（GB）。取不到就返回 0。"""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _fmt_gb(v: float) -> str:
    return f"{v:.1f} GB"


# ===========================================================================
# 模型基类
# ===========================================================================


class BaseModel:
    """所有模型的共同接口。"""

    kind: str = "base"
    name: str = "模型"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._loaded = False
        self._loaded_at = 0.0
        self._ram_before = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    # 子类实现
    def _do_load(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def _do_unload(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def load(self) -> None:
        if self._loaded:
            return
        self._ram_before = available_ram_gb()
        t0 = time.time()
        self._do_load()
        self._loaded = True
        self._loaded_at = time.time()
        cost = time.time() - t0
        after = available_ram_gb()
        log.info(
            "%s 加载完成，耗时 %.1fs，可用内存 %s → %s",
            self.name, cost, _fmt_gb(self._ram_before), _fmt_gb(after),
        )

    def unload(self) -> None:
        if not self._loaded:
            return
        try:
            self._do_unload()
        except Exception:
            log.exception("卸载 %s 时出错（忽略）", self.name)
        finally:
            self._loaded = False
            gc.collect()
            log.info("%s 已卸载，可用内存 %s", self.name, _fmt_gb(available_ram_gb()))

    def info(self) -> str:
        return f"{self.name}（{'已加载' if self.loaded else '未加载'}）"


# ===========================================================================
# 1. 语言模型：GPT-OSS-20B（llama.cpp，常驻）
# ===========================================================================


class LLM(BaseModel):
    """用 llama-cpp-python 跑 GPT-OSS-20B。

    注意：我们不走它的 chat template，而是自己拼 Harmony 原文再送进去，
    这样才能完全掌控 <|channel|>/<|constrain|>/<|message|> 这套标记，
    也让「专属协议解析器」有的可解析。
    """

    kind = "llm"
    name = "GPT-OSS-20B"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.conf = cfg.section("llm")
        self.model_path = cfg.path_of("llm.model_path", "models/gpt-oss-20b-mxfp4.gguf")
        self._llama: Any = None

    # -- 生命周期 ------------------------------------------------------
    def _do_load(self) -> None:
        try:
            from llama_cpp import Llama
        except Exception as e:  # noqa: BLE001
            raise ModelNotReady(
                "没有找到 llama-cpp-python。请先安装：\n"
                "pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from e

        if not self.model_path.exists():
            raise ModelFileMissing(
                f"语言模型文件不存在：\n{self.model_path}\n"
                "请把 GPT-OSS-20B 的 4bit GGUF（约 11~12GB）放到这个路径。"
            )

        n_threads = int(self.conf.get("n_threads", 0) or 0) or max(2, (os.cpu_count() or 4))
        ram = available_ram_gb()
        if ram and ram < 15:
            log.warning("可用内存只有 %s，加载 20B 模型可能失败", _fmt_gb(ram))

        kwargs: dict[str, Any] = dict(
            model_path=str(self.model_path),
            n_ctx=int(self.conf.get("n_ctx", 8192)),
            n_batch=int(self.conf.get("n_batch", 512)),
            n_threads=n_threads,
            n_gpu_layers=int(self.conf.get("n_gpu_layers", 0)),
            use_mmap=True,
            verbose=False,
        )
        # flash_attn 只在支持的时候传，老版本传了会报错
        if bool(self.conf.get("flash_attn", False)):
            kwargs["flash_attn"] = True

        try:
            self._llama = Llama(**kwargs)
        except TypeError:
            # 某些版本不接受 flash_attn / use_mmap 参数，去掉重试
            for key in ("flash_attn", "use_mmap"):
                kwargs.pop(key, None)
            try:
                self._llama = Llama(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise ModelNotReady(
                    f"加载语言模型失败：{e}\n"
                    "常见原因：① llama-cpp-python 版本太老不支持 GPT-OSS 的 MXFP4 量化；"
                    "② GGUF 文件损坏。建议升级到 0.3.x 最新版再看日志。"
                ) from e
        except Exception as e:  # noqa: BLE001
            raise ModelNotReady(
                f"加载语言模型失败：{e}\n"
                "常见原因：① llama-cpp-python 版本太老不支持 GPT-OSS 的 MXFP4 量化；"
                "② GGUF 文件损坏。建议升级到 0.3.x 最新版再看日志。"
            ) from e

    def _do_unload(self) -> None:
        llama = self._llama
        self._llama = None
        if llama is not None:
            # 新版有 close()，老版本只能靠析构
            close = getattr(llama, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            del llama
        gc.collect()

    # -- 编码 ----------------------------------------------------------
    def tokenize(self, text: str, add_bos: bool = False) -> list[int]:
        """把 Harmony 原文编码成 token。

        关键：special=True —— 让 <|start|>、<|channel|> 这些特殊 token
        被编码成单个特殊 token，而不是拆成普通字符，否则 GPT-OSS 根本不认。
        """
        data = text.encode("utf-8")
        try:
            return self._llama.tokenize(data, add_bos=add_bos, special=True)
        except TypeError:
            # 老版本没有 special 参数
            return self._llama.tokenize(data, add_bos=add_bos)

    # -- 生成 ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """输入 Harmony 原文，输出模型续写的原文（不含 prompt 本身）。"""
        if not self._loaded or self._llama is None:
            raise ModelNotReady("语言模型还没加载")

        tokens = self.tokenize(prompt)
        n_ctx = int(self.conf.get("n_ctx", 8192))
        if len(tokens) >= n_ctx - 64:
            # 上下文塞满了，从前面砍（保留 system 开头一段 + 最近的对话）
            keep_head = min(400, n_ctx // 6)
            tokens = tokens[:keep_head] + tokens[-(n_ctx - keep_head - 256):]
            log.warning("提示词超过上下文，已裁剪到 %d token", len(tokens))

        seed = int(self.conf.get("seed", -1))
        # 用 create_completion 明确传 token 列表：保证 <|start|> 这类特殊 token
        # 作为一个整体进模型，而不是被拆成普通字符
        resp = self._llama.create_completion(
            prompt=tokens,
            max_tokens=int(max_tokens or self.conf.get("max_tokens", 1200)),
            temperature=float(
                self.conf.get("temperature", 0.7) if temperature is None else temperature
            ),
            top_p=float(self.conf.get("top_p", 0.95) if top_p is None else top_p),
            repeat_penalty=float(self.conf.get("repeat_penalty", 1.05)),
            stop=stop if stop is not None else ["<|return|>", "<|call|>", "<|end|>"],
            echo=False,
            seed=seed if seed >= 0 else None,
        )
        try:
            return str(resp["choices"][0]["text"])
        except Exception:  # noqa: BLE001
            return str(resp)

    def info(self) -> str:
        return f"{self.name}（{'已加载' if self.loaded else '未加载'}，{self.model_path.name}）"


# ===========================================================================
# 2. 视觉理解模型：Qwen2.5-VL-3B（按需加载，用完卸载）
# ===========================================================================


VISION_PROMPT_RULES = (
    "你是一个图片描述助手。请用中文回答，客观、具体、详细。"
    "重点描述：画面主体是什么、主体外观与服饰细节、颜色、构图、光线、画风。"
    "不要编造看不见的内容，不要输出英文，不要输出 Markdown 表格。"
)


class VLM(BaseModel):
    """轻量视觉理解模型。优先 llama.cpp（GGUF + mmproj），失败再退 transformers。"""

    kind = "vision"
    name = "Qwen2.5-VL-3B"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.conf = cfg.section("vision")
        self.model_path = cfg.path_of("vision.model_path", "")
        self.mmproj_path = cfg.path_of("vision.mmproj_path", "")
        self.hf_path = cfg.path_of("vision.hf_model_path", "")
        self._llama: Any = None
        self._tf_model: Any = None
        self._tf_processor: Any = None
        self._backend = ""  # llama_cpp / transformers
        self._temp_files: list[str] = []

    # -- 图片预处理 ----------------------------------------------------
    def _prepare_image(self, image_path: str) -> str:
        """把图缩到最长边 max_image_side，转 RGB，存成临时 PNG。

        缩图能显著减少视觉 token 数量：既省内存又提速，对 3B 小模型也更稳。
        """
        from PIL import Image

        max_side = int(self.conf.get("max_image_side", 896))
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        img.save(tmp.name, format="PNG")
        self._temp_files.append(tmp.name)
        return tmp.name

    def _clean_temp(self) -> None:
        for p in self._temp_files:
            try:
                os.unlink(p)
            except Exception:
                pass
        self._temp_files.clear()

    # -- 加载 ----------------------------------------------------------
    def _do_load(self) -> None:
        backend = str(self.conf.get("backend", "auto")).lower()
        errors: list[str] = []

        if backend in ("auto", "llama_cpp"):
            try:
                self._load_llama()
                self._backend = "llama_cpp"
                return
            except ModelFileMissing as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001
                errors.append(f"llama.cpp 后端失败：{e}")

        if backend in ("auto", "transformers"):
            try:
                self._load_transformers()
                self._backend = "transformers"
                return
            except Exception as e:  # noqa: BLE001
                errors.append(f"transformers 后端失败：{e}")

        raise ModelNotReady(
            "视觉模型加载失败，两个后端都没起来：\n  - " + "\n  - ".join(errors)
            + "\n请检查 models/ 下的视觉模型文件，或看 config.json 里的 vision 配置。"
        )

    def _load_llama(self) -> None:
        from llama_cpp import Llama

        if not self.model_path.exists():
            raise ModelFileMissing(
                f"视觉模型文件不存在：{self.model_path}\n"
                "请下载 Qwen2.5-VL-3B-Instruct 的 GGUF（Q4_K_M 约 2GB）。"
            )
        if not self.mmproj_path.exists():
            raise ModelFileMissing(
                f"视觉模型的 mmproj 投影文件不存在：{self.mmproj_path}\n"
                "这是多模态必需文件（约 600MB），和模型本体一起下载。"
            )

        handler = self._build_chat_handler(self.mmproj_path)
        n_threads = int(self.conf.get("n_threads", 0) or 0) or max(2, (os.cpu_count() or 4))
        try:
            self._llama = Llama(
                model_path=str(self.model_path),
                chat_handler=handler,
                n_ctx=int(self.conf.get("n_ctx", 4096)),
                n_threads=n_threads,
                n_gpu_layers=0,
                verbose=False,
                use_mmap=True,
            )
        except TypeError:
            self._llama = Llama(
                model_path=str(self.model_path),
                chat_handler=handler,
                n_ctx=int(self.conf.get("n_ctx", 4096)),
                n_threads=n_threads,
                n_gpu_layers=0,
                verbose=False,
            )
        # 空跑一次，确认 handler 真的能工作（有些版本会静默失败）
        if self._llama is None:
            raise ModelNotReady("llama.cpp 返回了空模型")

    @staticmethod
    def _build_chat_handler(mmproj_path: Path) -> Any:
        """llama-cpp-python 各版本 chat handler 名字不一样，挨个探测。"""
        try:
            from llama_cpp import llama_chat_format as cf
        except Exception as e:  # noqa: BLE001
            raise ModelNotReady("llama-cpp-python 没有 llama_chat_format 模块，版本太老") from e

        candidates = [
            "Qwen25VLChatHandler",
            "Qwen2VLChatHandler",
            "MTMDChatHandler",
            "Llava15ChatHandler",
        ]
        for name in candidates:
            cls = getattr(cf, name, None)
            if cls is None:
                continue
            try:
                return cls(clip_model_path=str(mmproj_path))
            except TypeError:
                try:
                    return cls(mmproj_path=str(mmproj_path))
                except Exception:
                    continue
            except Exception:
                continue
        raise ModelNotReady(
            "当前 llama-cpp-python 版本不认识 Qwen2.5-VL 的多模态处理器。\n"
            "请升级：pip install -U llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
            "或者在 config.json 里把 vision.backend 改成 transformers。"
        )

    def _load_transformers(self) -> None:
        """transformers 后备后端：直接加载 HF 格式的 Qwen2.5-VL。"""
        import torch
        from transformers import AutoProcessor

        if not self.hf_path.exists():
            raise ModelFileMissing(
                f"transformers 视觉模型目录不存在：{self.hf_path}\n"
                "把 Qwen2.5-VL-3B-Instruct 的 HF 模型目录放到这个位置。"
            )

        # transformers 5.x 用 AutoModelForImageTextToText
        model = None
        last_err: Exception | None = None
        for loader_name in ("AutoModelForImageTextToText", "Qwen2_5_VLForConditionalGeneration"):
            try:
                import transformers as tf

                loader = getattr(tf, loader_name)
                model = loader.from_pretrained(
                    str(self.hf_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if model is None:
            raise ModelNotReady(f"transformers 加载视觉模型失败：{last_err}")

        model.eval()
        self._tf_model = model
        self._tf_processor = AutoProcessor.from_pretrained(str(self.hf_path))

    def _do_unload(self) -> None:
        self._clean_temp()
        llama = self._llama
        self._llama = None
        if llama is not None:
            close = getattr(llama, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            del llama
        if self._tf_model is not None:
            try:
                del self._tf_model
            except Exception:
                pass
            self._tf_model = None
            self._tf_processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # -- 推理 ----------------------------------------------------------
    def describe(self, image_path: str, question: str = "") -> str:
        """输入图片 + 问题，输出中文描述。"""
        if not self._loaded:
            raise ModelNotReady("视觉模型还没加载")
        q = (question or "请详细描述这张图片。").strip()
        prompt = f"{VISION_PROMPT_RULES}\n\n用户的问题：{q}"
        img_path = self._prepare_image(image_path)
        try:
            if self._backend == "transformers":
                return self._describe_transformers(img_path, prompt)
            return self._describe_llama(img_path, prompt)
        finally:
            self._clean_temp()

    def _describe_llama(self, img_path: str, prompt: str) -> str:
        # ★ 踩坑记录：llama-cpp-python 的 _load_image 只认两种东西 ——
        #   1) "data:image/png;base64,..." 这种 data URI
        #   2) http/https 链接（它内部用 urllib.urlopen 去下）
        #   它【不认本地文件路径】，会把 "C:\xxx.png" 直接丢给 urlopen 然后报
        #   "unknown url type"。所以这里必须先读文件、转成 base64 data URI 再喂。
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resp = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=int(self.conf.get("max_tokens", 512)),
            temperature=float(self.conf.get("temperature", 0.2)),
        )
        try:
            return str(resp["choices"][0]["message"]["content"]).strip()
        except Exception:
            return str(resp).strip()

    def _describe_transformers(self, img_path: str, prompt: str) -> str:
        import torch
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        proc = self._tf_processor
        model = self._tf_model

        # 新版 transformers 的消息格式是 {"type": "image"} + images=[...]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[image], return_tensors="pt")
        except Exception:
            # 老格式：图片直接挂在 content 里
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[image], return_tensors="pt")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(self.conf.get("max_tokens", 512)),
                do_sample=False,
            )
        n_in = inputs["input_ids"].shape[1]
        answer = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return answer.strip()

    def info(self) -> str:
        state = "已加载" if self.loaded else "未加载"
        return f"{self.name}（{state}，后端 {self._backend or '未确定'}）"


# ===========================================================================
# 3. 生图模型：SD1.5 / SDXL-Turbo（按需加载，用完卸载）
# ===========================================================================


class ImageModel(BaseModel):
    """用 diffusers 跑轻量文生图（CPU）。

    「按模型整理提示词」就在这里做：
      - SD1.5：补质量词、补默认反向词、截断长度、检查是不是中文（中文模型看不懂）
      - SDXL-Turbo：不加质量词，guidance=0，步数 1~4
    """

    kind = "image"
    name = "SD1.5"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.conf = cfg.section("image")
        self.model_path = cfg.path_of("image.model_path", "models/sd-v1-5")
        self.variant = str(self.conf.get("variant", "sd15")).lower()
        self._pipe: Any = None

    # -- 加载 ----------------------------------------------------------
    def _do_load(self) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as e:  # noqa: BLE001
            raise ModelNotReady(
                "没有找到 diffusers / torch。请安装：\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "pip install diffusers transformers accelerate safetensors"
            ) from e

        if not self.model_path.exists():
            raise ModelFileMissing(
                f"生图模型不存在：{self.model_path}\n"
                "请把 SD1.5 的 diffusers 目录（或单个 .safetensors 文件）放到这个路径。"
            )

        # CPU 上用 float32 最稳（float16 很多算子不支持）
        dtype = torch.float32
        common = dict(torch_dtype=dtype, use_safetensors=True)

        try:
            if self.model_path.is_file():
                pipe = AutoPipelineForText2Image.from_single_file(str(self.model_path), **common)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(str(self.model_path), **common)
        except Exception as first_err:  # noqa: BLE001
            log.warning("AutoPipeline 加载失败，退回 StableDiffusionPipeline：%s", first_err)
            try:
                from diffusers import StableDiffusionPipeline

                if self.model_path.is_file():
                    pipe = StableDiffusionPipeline.from_single_file(
                        str(self.model_path), torch_dtype=dtype, safety_checker=None,
                        requires_safety_checker=False,
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        str(self.model_path), torch_dtype=dtype, safety_checker=None,
                        requires_safety_checker=False,
                    )
            except Exception as e:  # noqa: BLE001
                raise ModelNotReady(f"加载生图模型失败：{e}") from e

        pipe = pipe.to("cpu")
        # 省内存三件套：注意力切片 + 关掉进度条（无控制台会报错）
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        try:
            pipe.safety_checker = None
        except Exception:
            pass

        self._pipe = pipe
        try:
            cls_name = type(pipe).__name__
            log.info("生图管线就绪：%s", cls_name)
            self.name = cls_name.replace("Pipeline", "") or "SD"
        except Exception:
            pass

    def _do_unload(self) -> None:
        pipe = self._pipe
        self._pipe = None
        if pipe is not None:
            try:
                del pipe
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # -- 提示词规整 ----------------------------------------------------
    def prepare_prompt(self, prompt: str, negative: str | None = None) -> tuple[str, str | None]:
        """按模型整理提示词：这是「生图模型只负责画图」的关键一步。"""
        p = re.sub(r"\s+", " ", (prompt or "").strip())
        max_chars = int(self.conf.get("max_prompt_chars", 600))

        # 中文检测：SD1.5 的 CLIP 只认英文，中文进去基本等于噪音
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", p))
        if cn_chars and cn_chars >= max(2, len(p) // 4):
            log.warning("生图提示词里中文偏多（%d 个汉字），SD 可能画不准，建议让模型输出英文", cn_chars)

        quality = str(self.conf.get("quality_suffix", "")).strip()
        if self.variant.startswith("sd15") and quality and quality.lower() not in p.lower():
            p = f"{p}, {quality}"

        if len(p) > max_chars:
            p = p[:max_chars].rstrip(",， ")

        neg = (negative or "").strip()
        default_neg = str(self.conf.get("negative_prompt", "")).strip()
        if not neg:
            neg = default_neg or None
        elif default_neg and neg.lower() not in default_neg.lower():
            neg = f"{neg}, {default_neg}"

        return p, neg

    # -- 生成 ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        on_step: Callable[[int, int], None] | None = None,
    ) -> Path:
        if not self._loaded or self._pipe is None:
            raise ModelNotReady("生图模型还没加载")

        import torch

        prompt, neg = self.prepare_prompt(prompt, negative_prompt)

        max_side = int(self.conf.get("max_side", 768))
        w = int(width or self.conf.get("width", 512))
        h = int(height or self.conf.get("height", 512))
        # SD 要求边长是 8 的倍数，且别超过上限（CPU 上太大就是折磨）
        w = max(256, min(max_side, w // 8 * 8))
        h = max(256, min(max_side, h // 8 * 8))

        s = int(steps or self.conf.get("steps", 12))
        if self.variant.startswith("sdxl"):
            s = max(1, min(8, s))
            guidance = 0.0
        else:
            guidance = float(self.conf.get("guidance", 7.5))

        seed = int(self.conf.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2 ** 31 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=s,
            guidance_scale=guidance,
            width=w,
            height=h,
            generator=generator,
        )
        if neg:
            kwargs["negative_prompt"] = neg

        if on_step is not None:
            def _cb(pipe, step_index, timestep, callback_kwargs):  # noqa: ANN001
                try:
                    on_step(int(step_index) + 1, s)
                except Exception:
                    pass
                return callback_kwargs

            kwargs["callback_on_step_end"] = _cb

        log.info("开始生图：%dx%d, %d 步, seed=%d", w, h, s, seed)
        t0 = time.time()
        result = self._pipe(**kwargs)
        image = result.images[0]

        out_dir = self.cfg.output_dir
        filename = time.strftime("gen_%Y%m%d_%H%M%S") + f"_{seed % 10000:04d}.png"
        out_path = out_dir / filename
        image.save(out_path)
        log.info("生图完成，耗时 %.1fs，文件：%s", time.time() - t0, out_path)
        return out_path

    def info(self) -> str:
        state = "已加载" if self.loaded else "未加载"
        return f"{self.name}（{state}，{self.variant}）"


# ===========================================================================
# 4. 模型管理器：单槽互斥的总闸门
# ===========================================================================


class ModelManager:
    """模型常驻管理器：加载后**一直留着**，只有内存紧张才按 LRU 踢掉。

    为什么改：上一版「用一下卸一下」，看张图要把 20B 语言模型卸载再重载，
    光重载 + 重新算 KV cache 就要十几分钟。现在改成：

      - 语言模型：启动后常驻，永不主动卸
      - 视觉 / 生图：第一次用时加载，之后一直留着
      - 只有当「可用内存 < 本次加载预估 + 安全余量」或「驻留数超上限」时，
        才按最近最少使用顺序卸载一个

    32GB 机器上三个模型（13.5 + 3 + 4.5 ≈ 21GB）可以同时驻留。
    """

    # 每个模型的粗略内存占用（GB），只用于内存预算判断
    EST_GB = {"llm": 13.5, "vision": 3.0, "image": 4.5}

    def __init__(
        self,
        cfg: Config,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.slots: dict[str, BaseModel] = {
            "llm": LLM(cfg),
            "vision": VLM(cfg),
            "image": ImageModel(cfg),
        }
        self._resident: list[str] = []          # LRU 顺序：index 0 最久没用
        self._lock = threading.RLock()
        self._on_status = on_status or (lambda msg: None)
        self._max_resident = int(cfg.get("loader.max_resident", 3))
        self._min_free_gb = float(cfg.get("loader.min_free_gb", 2.5))

    # -- 状态 ----------------------------------------------------------
    def _notify(self, msg: str) -> None:
        log.info(msg)
        try:
            self._on_status(msg)
        except Exception:
            pass

    def current_kind(self) -> str | None:
        return self._resident[-1] if self._resident else None

    def set_status_callback(self, cb: Callable[[str], None]) -> None:
        self._on_status = cb or (lambda msg: None)

    def current_name(self) -> str:
        k = self.current_kind()
        return self.slots[k].name if k else "无"

    def resident_names(self) -> list[str]:
        return [self.slots[k].name for k in self._resident]

    def status_text(self) -> str:
        if not self._resident:
            return "没有模型驻留"
        names = "、".join(self.resident_names())
        return f"驻留：{names}｜可用内存 {_fmt_gb(available_ram_gb())}"

    # -- LRU ------------------------------------------------------------
    def _touch(self, kind: str) -> None:
        if kind in self._resident:
            self._resident.remove(kind)
        self._resident.append(kind)

    def _unload_locked(self, kind: str) -> None:
        if kind in self._resident:
            self._resident.remove(kind)
        try:
            self.slots[kind].unload()
        finally:
            gc.collect()

    def _ensure_room_locked(self, kind: str) -> None:
        """为即将加载的模型腾地方：只在真的不够时才卸，而且先卸最久没用的。"""
        est = self.EST_GB.get(kind, 4.0)
        while self._resident:
            avail = available_ram_gb()
            too_many = len(self._resident) >= self._max_resident
            low_mem = (avail > 0) and (avail < est + self._min_free_gb)
            if not too_many and not low_mem:
                break
            victim = self._resident[0]
            if victim == kind:
                break
            self._notify(f"内存紧张，先释放{self.slots[victim].name}")
            self._unload_locked(victim)

    # -- 对外 ------------------------------------------------------------
    def get(self, kind: str) -> BaseModel:
        """拿到某个模型；已驻留就直接用，没驻留就加载（之后常驻）。"""
        if kind not in self.slots:
            raise KeyError(f"未知模型类型：{kind}")
        with self._lock:
            slot = self.slots[kind]
            if slot.loaded:
                self._touch(kind)
                return slot

            self._ensure_room_locked(kind)
            self._notify(f"正在加载 {slot.name}……（只需这一次，之后常驻不再等）")
            slot.load()
            self._touch(kind)
            self._notify(f"{slot.name} 已就绪并常驻｜可用内存 {_fmt_gb(available_ram_gb())}")
            return slot

    def unload(self, kind: str) -> None:
        with self._lock:
            if kind in self.slots and self.slots[kind].loaded:
                self._notify(f"手动释放 {self.slots[kind].name}")
                self._unload_locked(kind)

    def unload_all(self) -> None:
        with self._lock:
            for kind in list(self._resident):
                self._unload_locked(kind)

    # -- 借用（现在不再自动卸载）----------------------------------------
    @contextmanager
    def borrow(self, kind: str):
        """借用模型。★ 用完不会卸载，留着下次秒用。"""
        yield self.get(kind)

    # -- 语言模型快捷方式 ----------------------------------------------
    def chat_model(self) -> LLM:
        return self.get("llm")  # type: ignore[return-value]

    def preload_llm(self) -> None:
        """后台预热语言模型（启动时调用，避免第一次对话干等）。"""
        def _worker() -> None:
            try:
                self.get("llm")
            except Exception as e:  # noqa: BLE001
                log.warning("预热语言模型失败：%s", e)

        threading.Thread(target=_worker, daemon=True, name="preload-llm").start()

    def shutdown(self) -> None:
        self.unload_all()


# ===========================================================================
# 自测：python loader.py
# ===========================================================================
if __name__ == "__main__":
    from config import Config as Cfg

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = Cfg()
    mm = ModelManager(c, on_status=lambda m: print("[状态]", m))
    print("可用内存：", _fmt_gb(available_ram_gb()))
    print(mm.status_text())
    for kind in ("llm", "vision", "image"):
        print(f"--- 尝试加载 {kind} ---")
        try:
            mm.get(kind)
            print(mm.status_text())
        except Exception as e:  # noqa: BLE001
            print(f"（预期内的失败，因为还没放模型文件）{e}")
    mm.shutdown()
