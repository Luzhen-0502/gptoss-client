# -*- coding: utf-8 -*-
"""
agent_core.py —— 专属协议解析器 + 自研调度闭环（大肥鱼手写）

这个文件是整个 Agent 的大脑皮层，分两半：

上半截：HarmonyParser
    专治 GPT-OSS-20B 的 Harmony 通道格式。模型真实输出长这样：

        <|start|>assistant<|channel|>analysis<|message|>用户要一张洛丽塔的图……<|end|>
        <|start|>assistant<|channel|>commentary to=functions.search_knowledge <|constrain|>json<|message|>{"query":"洛丽塔服饰"}<|call|>
        <|start|>functions.search_knowledge<|channel|>commentary<|message|>【联网资料】……<|end|>
        <|start|>assistant<|channel|>final<|message|>画好了，你看！<|end|>

    解析器要做的事：
      1. 切出每一条消息（start / channel / constrain / message / call / return）
      2. 认出 commentary 通道里 to=functions.xxx 的工具调用
      3. 把 <|message|> 后面的 JSON 抠出来并解析
      4. JSON 坏了不算完 —— 交给调度器重试一次，还坏就降级纯文本

下半截：AgentCore
    完全自己写的调度闭环（一行 LangChain / AutoGen 都没有）：

        while 还有步数:
            模型输出 → 解析 → 是不是工具调用？
                是  → 执行工具（该换模型就换）→ 结果回填上下文和记忆 → 继续
                否  → 就是最终回答，收工

    工具执行前后自动装卸模型，保证内存里永远只有一个模型。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config import Config
from tools import ToolContext, ToolRegistry, WebKnowledge

log = logging.getLogger("gptoss.agent")


# ===========================================================================
# 零、Harmony 特殊标记
# ===========================================================================

T_START = "<|start|>"
T_END = "<|end|>"
T_MESSAGE = "<|message|>"
T_CHANNEL = "<|channel|>"
T_CONSTRAIN = "<|constrain|>"
T_RETURN = "<|return|>"
T_CALL = "<|call|>"

# 模型偶尔会把 constrain 写成 constraint，也一起认
_CONSTRAIN_VARIANTS = (T_CONSTRAIN, "<|constraint|>")

# 一条消息块：从 <|start|> 到 <|message|> 是头部，后面是正文，正文结束于结束标记或下一条开始
_BLOCK_RE = re.compile(
    r"<\|start\|>(?P<header>.*?)<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|call\|>|<\|return\|>|<\|start\|>|$)",
    re.S,
)


# ===========================================================================
# 一、数据结构
# ===========================================================================


@dataclass
class ToolCall:
    """一次工具调用请求。"""

    name: str                      # 工具名（去掉 functions. 前缀）
    recipient: str = ""            # 原始 recipient，例如 functions.see_image
    arguments: dict[str, Any] | None = None   # 解析好的参数；None 表示 JSON 坏了
    raw_args: str = ""             # 原始参数文本（调试/重试提示用）
    error: str = ""                # JSON 解析失败原因


@dataclass
class ParsedOutput:
    """一次模型输出的解析结果。"""

    raw: str = ""
    analysis: str = ""             # analysis 通道（模型的思考，不直接给用户看）
    commentary: str = ""           # commentary 通道里的普通文字
    final: str = ""                # final 通道（给用户的最终回答）
    tool_calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    malformed: bool = False        # 有工具调用意图但参数解析失败
    plain_text_only: bool = False  # 完全没有 harmony 标记（降级情况）

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)


# ===========================================================================
# 二、Harmony 协议解析器
# ===========================================================================


class HarmonyParser:
    """GPT-OSS-20B 的 Harmony 通道解析器（纯手写，无第三方依赖）。"""

    # -- 工具名归一化 --------------------------------------------------
    @staticmethod
    def _normalize_recipient(recipient: str) -> str:
        """functions.see_image / functions::see_image / see_image → see_image"""
        r = (recipient or "").strip().strip("<>" + "|").strip()
        r = r.replace("::", ".")
        if "." in r:
            r = r.split(".")[-1]
        return r.strip()

    # -- JSON 提取 -----------------------------------------------------
    @classmethod
    def extract_json(cls, text: str) -> tuple[dict[str, Any] | None, str]:
        """从一段文本里抠出 JSON 对象，尽最大努力。

        返回 (对象, 错误说明)。失败时对象为 None。
        """
        body = (text or "").strip()
        if not body:
            return None, "参数为空"

        # 去掉 ```json ... ``` 代码围栏
        fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", body, re.S)
        if fence:
            body = fence.group(1).strip()

        candidates = [body]
        # 抠第一个 { 到最后一个 }（模型可能在前面说了一句中文）
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            candidates.append(body[start : end + 1])
        # 也支持数组形式的参数
        if body.lstrip().startswith("["):
            candidates.append(body)

        for cand in candidates:
            for attempt in (cand, cls._repair_json(cand)):
                try:
                    obj = json.loads(attempt)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj, ""
                if isinstance(obj, list):
                    # 有的模型把参数直接写成数组，包一层
                    return {"items": obj}, ""
        return None, f"不是合法 JSON：{body[:120]}"

    @staticmethod
    def _repair_json(text: str) -> str:
        """修修常见小毛病：尾逗号、中文引号、单引号。"写坏的 JSON 先救一次"。"""
        s = text
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = re.sub(r",\s*([}\]])", r"\1", s)       # {"a":1,} → {"a":1}
        s = re.sub(r"(?<=[{\[,])\s*'", '"', s)     # {'a' → {"a
        s = re.sub(r"'\s*(?=[:\],}])", '"', s)     # 'a': → "a":
        return s

    # -- 头部解析 ------------------------------------------------------
    @classmethod
    def _parse_header(cls, header: str) -> tuple[str, str, str, str]:
        """拆头部，返回 (author, channel, recipient, constrain)。"""
        h = header or ""
        author = ""
        rest = h

        if T_CHANNEL in h:
            author_part, rest = h.split(T_CHANNEL, 1)
            author = author_part.strip()
        else:
            # 没有 channel 标记：整段当 author（有些版本省略 channel）
            author = h.strip()

        # 去掉 author 里可能混进来的空白和特殊符号
        author = author.replace("\n", " ").strip()

        constrain = ""
        for tok in _CONSTRAIN_VARIANTS:
            if tok in rest:
                before, after = rest.split(tok, 1)
                rest = before
                # constrain 后面可能还有 json 字样
                constrain = after.strip().split("<|")[0].strip()
                break

        channel_raw = rest.replace("\n", " ").strip()
        recipient = ""
        m = re.search(r"to=([^\s>]+)", channel_raw)
        if m:
            recipient = m.group(1)
            channel_raw = (channel_raw[: m.start()] + " " + channel_raw[m.end() :]).strip()

        channel = channel_raw.strip()
        return author, channel, recipient, constrain

    # -- 主解析 --------------------------------------------------------
    @classmethod
    def parse(cls, raw: str) -> ParsedOutput:
        """把模型输出拆成 分析 / 评论 / 最终回答 / 工具调用。"""
        raw = raw or ""
        out = ParsedOutput(raw=raw)

        blocks = list(_BLOCK_RE.finditer(raw))

        # 完全没有 <|start|> → 模型没用 harmony 格式，整段当纯文本回答（降级）
        if not blocks:
            text = raw.strip()
            # 尝试识别「裸 JSON 工具调用」这种非标准输出
            bare = cls._try_bare_tool_call(text)
            if bare is not None:
                out.tool_calls.append(bare)
                if bare.arguments is None:
                    out.malformed = True
                    out.errors.append(bare.error)
                return out
            out.final = text
            out.plain_text_only = True
            return out

        for m in blocks:
            header = m.group("header") or ""
            body = (m.group("body") or "").strip()
            author, channel, recipient, constrain = cls._parse_header(header)

            # author 可能是 assistant / functions.xxx / user / tool
            is_assistant = (not author) or author.endswith("assistant")

            if is_assistant:
                if recipient:
                    # ★ 这就是工具调用：commentary to=functions.xxx
                    call = cls._build_tool_call(recipient, body)
                    out.tool_calls.append(call)
                    if call.arguments is None:
                        out.malformed = True
                        out.errors.append(f"工具 {call.name} 参数解析失败：{call.error}")
                    continue

                if channel.startswith("analysis"):
                    out.analysis += (("\n" if out.analysis else "") + body)
                elif channel.startswith("commentary"):
                    out.commentary += (("\n" if out.commentary else "") + body)
                elif channel.startswith("final"):
                    out.final += (("\n" if out.final else "") + body)
                else:
                    # 没写 channel 的 assistant 消息，当最终回答
                    out.final += (("\n" if out.final else "") + body)
            else:
                # 别的 author 的块（工具结果等）不属于本次要解析的模型输出，忽略
                continue

        # 工具调用参数一个都没解析出来，但有 to= 的痕迹 → 也算 malformed
        if out.tool_calls and all(tc.arguments is None for tc in out.tool_calls):
            out.malformed = True

        # 模型既没给 final 也没给工具调用，只有 commentary → 把 commentary 当回答
        if not out.final and not out.tool_calls and out.commentary:
            out.final = out.commentary

        return out

    @classmethod
    def _build_tool_call(cls, recipient: str, body: str) -> ToolCall:
        name = cls._normalize_recipient(recipient)
        args, err = cls.extract_json(body)
        return ToolCall(
            name=name,
            recipient=recipient,
            arguments=args,
            raw_args=body,
            error=err,
        )

    @classmethod
    def _try_bare_tool_call(cls, text: str) -> ToolCall | None:
        """兜底：识别 {"name": "...", "arguments": {...}} 这种非 harmony 的调用格式。"""
        if not text or "{" not in text:
            return None
        obj, _err = cls.extract_json(text)
        if not isinstance(obj, dict):
            return None
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters")
        if isinstance(name, str) and isinstance(args, dict):
            return ToolCall(name=cls._normalize_recipient(name), recipient=f"functions.{name}", arguments=args)
        return None


# ===========================================================================
# 三、Harmony 提示词渲染器
# ===========================================================================


SYSTEM_HEADER = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {date}

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.
"""

TOOL_USAGE_RULES = """
# 工具调用格式（必须严格遵守）

需要调用工具时，你必须输出这样的结构（先写 channel，再写 to=，最后是 json）：

<|start|>assistant<|channel|>commentary to=functions.工具名 <|constrain|>json<|message|>{{"参数名": "参数值"}}<|call|>

规则：
1. 工具调用只能出现在 commentary 通道，并带上 to=functions.工具名 和 <|constrain|>json。
2. 参数必须是一个合法的 JSON 对象，不要写注释、不要用单引号、不要漏引号。
3. 一次可以调用多个工具，每个工具一条消息。
4. 工具执行结果会以 functions.工具名 的身份回给你，你接着推理。
5. 最终给用户的回答放在 final 通道：<|start|>assistant<|channel|>final<|message|>……<|end|>
6. 全程用中文和用户交流（工具参数里的生图提示词除外）。

# 你的身份与职责

你是这台电脑上的「GPT-OSS 专属 Agent」，能看图片、能上网查资料、能画图、能看时间。
你要自己决定：先看图？先查资料？先写提示词？还是直接回答。

工作原则：
1. 用户上传了图片、让你「看这张图」时，先调用 see_image 拿到中文描述。
2. 要生成图片时：先想清楚画面要素 → 遇到你不确定视觉细节的小众词（洛丽塔、蒸汽波、哥特、赛博朋克、新艺术运动、特定民族服饰等）先调用 search_knowledge 查资料 → 把资料翻译成英文视觉描述 → 调用 generate_image。
3. generate_image 的 prompt 必须是英文，包含主体、服装款式、颜色、场景、画风、镜头，用逗号分隔关键词。
4. 联网失败不代表任务失败：用你已有的常识继续写提示词并出图，同时告诉用户这次没查到资料。
5. 不要凭空编造工具结果。工具没返回的内容，就说不知道。

# 你能操作这台电脑（不只是画图）

你运行在用户自己的 Windows 电脑上，除了看图和画图，你还能真的动手干活：

| 工具 | 能干什么 |
|---|---|
| run_command | 执行命令行命令：装软件、跑脚本、查系统状态、批量处理文件 |
| read_file / write_file | 读 / 写文本文件：写代码、改配置、写笔记 |
| list_dir | 看一个文件夹里有什么 |
| open_path | 用系统默认程序打开文件、文件夹或网址 |
| delete_path | 删除文件/文件夹（会先弹窗问用户） |
| screenshot | 截屏，截完可以接着调 see_image 看 |
| system_info | 查 CPU、磁盘剩余空间、用户目录 |

干活原则：
1. 先看清楚再动手：不知道路径就先 list_dir，不知道文件内容就先 read_file，别瞎猜。
2. 用户没让你做的事不要做，尤其是删除文件、改系统设置、装软件。
3. 删除和危险命令会自己弹窗问用户；用户拒绝就换一个更安全的方案，不要反复纠缠。
4. 干完要说清楚：你改了什么、存在哪、下一步要用户做什么。
5. 命令写成一行，别用会一直等输入的交互式命令。
"""


class HarmonyRenderer:
    """把「系统提示 + 工具声明 + 记忆 + 历史」拼成 GPT-OSS 能吃的 Harmony 原文。"""

    def __init__(self, cfg: Config, registry: ToolRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.history_turns = int(cfg.get("agent.history_turns", 8))
        self.memory_block_chars = int(cfg.get("agent.memory_block_chars", 1800))

    # -- 类型转换 ------------------------------------------------------
    @staticmethod
    def _ts_type(schema: dict[str, Any]) -> str:
        t = str(schema.get("type", "string"))
        if t == "array":
            item = schema.get("items", {})
            return f"{HarmonyRenderer._ts_type(item)}[]" if item else "any[]"
        if t == "object":
            return "object"
        return {"integer": "number", "number": "number", "boolean": "boolean"}.get(t, "string")

    def _render_tools(self) -> str:
        """按 openai/harmony 的 TypeScript 风格渲染工具声明。"""
        tools = self.registry.model_tools()
        if not tools:
            return ""
        lines = ["", "# Tools", "", "## Namespace: functions", "", "### Target channel: commentary", "", "### Tool definitions"]
        for tool in tools:
            desc = re.sub(r"\s+", " ", str(tool["description"])).strip()
            lines.append(f"// {desc}")
            lines.append(f"type {tool['name']} = (_: {{")
            schema = tool.get("parameters") or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            for pname, pschema in props.items():
                pdesc = re.sub(r"\s+", " ", str(pschema.get("description", ""))).strip()
                if pdesc:
                    lines.append(f"// {pdesc}")
                optional = "" if pname in required else "?"
                lines.append(f"{pname}{optional}: {self._ts_type(pschema)},")
            lines.append("}) => any;")
            lines.append("")
        return "\n".join(lines)

    # -- 系统消息 ------------------------------------------------------
    def system_block(self) -> str:
        today = time.strftime("%Y-%m-%d")
        content = SYSTEM_HEADER.format(date=today) + self._render_tools() + "\n" + TOOL_USAGE_RULES
        return f"{T_START}system{T_MESSAGE}{content}{T_END}"

    # -- 单条历史消息 --------------------------------------------------
    def render_message(self, msg: dict[str, Any]) -> str:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        meta = msg.get("meta") or {}

        if role == "user":
            return f"{T_START}user{T_MESSAGE}{content}{T_END}"

        if role == "assistant":
            kind = str(meta.get("kind", "final"))
            if kind == "tool_call":
                tool = str(meta.get("tool", ""))
                args = meta.get("arguments")
                if isinstance(args, (dict, list)):
                    arg_text = json.dumps(args, ensure_ascii=False)
                else:
                    arg_text = content
                return (
                    f"{T_START}assistant{T_CHANNEL}commentary to=functions.{tool} "
                    f"{T_CONSTRAIN}json{T_MESSAGE}{arg_text}{T_CALL}"
                )
            channel = "commentary" if kind == "commentary" else "final"
            return f"{T_START}assistant{T_CHANNEL}{channel}{T_MESSAGE}{content}{T_END}"

        if role == "tool":
            tool = str(meta.get("tool", "tool"))
            return f"{T_START}functions.{tool}{T_CHANNEL}commentary{T_MESSAGE}{content}{T_END}"

        # 其它类型当 user 处理，别丢信息
        return f"{T_START}user{T_MESSAGE}{content}{T_END}"

    # -- 完整提示词 ----------------------------------------------------
    def render(self, memory: Any, *, extra_note: str = "") -> str:
        """拼出一次完整推理的输入。

        结构：
            [system（含工具声明和工作规则）]
            [记忆摘要（图片描述 / 联网资料 / 任务状态）]  ← 用 system 段注入，避免污染对话
            [最近若干轮历史]
            <|start|>assistant
        """
        parts: list[str] = [self.system_block()]

        # 记忆摘要：单独一段 system，让模型「记得」之前看到了什么、查到了什么
        mem_block = memory.render_memory_block(self.memory_block_chars) if hasattr(memory, "render_memory_block") else ""
        if mem_block:
            parts.append(f"{T_START}system{T_MESSAGE}# 共享记忆（三个模块共用，你可以直接使用）\n{mem_block}{T_END}")

        if extra_note:
            parts.append(f"{T_START}system{T_MESSAGE}{extra_note}{T_END}")

        # 历史对话
        for msg in memory.history(self.history_turns * 3):
            parts.append(self.render_message(msg))

        # 生成起点：让模型自己补 <|channel|>
        parts.append(f"{T_START}assistant")
        return "".join(parts)


# ===========================================================================
# 四、自研调度闭环
# ===========================================================================


class AgentCancelled(RuntimeError):
    """用户点了取消。"""


class AgentCore:
    """自己写的 Agent 循环：模型 → 解析 → 工具 → 回填 → 再推理……直到 final。"""

    def __init__(
        self,
        cfg: Config,
        memory: Any,
        loader: Any,
        registry: ToolRegistry | None = None,
        emit: Callable[[str, Any], None] | None = None,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.memory = memory
        self.loader = loader
        self.registry = registry or ToolRegistry()
        self.emit = emit or (lambda kind, payload=None: None)
        self.renderer = HarmonyRenderer(cfg, self.registry)
        self.web = WebKnowledge(timeout=float(cfg.get("agent.web_timeout", 8)))
        # ★ confirm：危险操作（删文件、高危命令）前问用户，默认拒绝
        self.ctx = ToolContext(
            memory=memory,
            loader=loader,
            config=cfg,
            web=self.web,
            emit=self.emit,
            confirm=confirm or (lambda message: False),
        )
        self.max_steps = int(cfg.get("agent.max_steps", 6))

        self._cancel = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self.last_raw_outputs: list[str] = []   # 调试用：保留每次原始输出

    # -- 事件 ----------------------------------------------------------
    def _emit(self, kind: str, payload: Any = None) -> None:
        try:
            self.emit(kind, payload)
        except Exception:
            log.exception("事件回调出错")

    # -- 取消 ----------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        self._emit("status", "正在取消……")

    def reset_cancel(self) -> None:
        self._cancel.clear()

    # -- 主入口 --------------------------------------------------------
    def run(self, user_text: str, image_path: str = "") -> str:
        """跑一轮完整任务，返回最终回答文本。

        这就是「自研调度闭环」的全部：
            模型输出 → 解析是否调用工具 → 执行工具 → 结果回填上下文和记忆 → 再推理
        """
        with self._lock:
            if self._running:
                return "（上一轮还没跑完，请稍等）"
            self._running = True
        try:
            return self._run_inner(user_text, image_path)
        finally:
            with self._lock:
                self._running = False
            # 只在任务还没落状态时才兜底标记失败；
            # 正常结束（done）和显式报错（error）都由 _run_inner 自己记，别覆盖它
            if self.memory.get_task().get("stage") not in ("done", "error"):
                self.memory.finish_turn(ok=False, error="任务中断")

    def _run_inner(self, user_text: str, image_path: str) -> str:
        self.reset_cancel()
        self.last_raw_outputs = []
        user_text = (user_text or "").strip()

        # ---- 1. 记录任务起点，写进共享记忆 -------------------------
        self.memory.begin_turn(user_text or "（看图片）", image_path)
        display_text = user_text
        if image_path:
            note = f"[上传图片] {Path(image_path).name}"
            display_text = f"{user_text}\n{note}".strip() if user_text else note
        self.memory.add_user(display_text, image_path)
        if image_path:
            self.memory.set_task(pending_image=image_path)
        self._emit("user", display_text)

        # ---- 2. 如果只传了图没说话，自动补一句任务指令 -----------
        if not user_text and image_path:
            user_text = str(self.cfg.get("agent.auto_image_goal", "请先看懂这张图，然后生成一张新图。"))

        retry_used = False       # 「解析失败重试一次」用掉没有
        final_text = ""
        last_parse: ParsedOutput | None = None

        for step in range(1, self.max_steps + 1):
            if self._cancel.is_set():
                raise AgentCancelled("用户取消了任务")

            self.memory.set_task(stage="thinking")
            self._emit("status", f"第 {step} 步：模型正在思考……")
            self._emit("progress", (step, self.max_steps))

            # ---- 3. 让语言模型推理（这一步不换模型，语言模型本来就在）----
            try:
                llm = self.loader.chat_model()
                prompt = self.renderer.render(self.memory)
                raw = llm.generate(prompt)
            except Exception as e:  # noqa: BLE001
                msg = self._friendly_error(e)
                self.memory.finish_turn(ok=False, error=msg)
                self._emit("error", msg)
                return f"出错了：{msg}"

            self.last_raw_outputs.append(raw)
            log.debug("模型原始输出：%s", raw[:2000])

            # ---- 4. 专属协议解析 ---------------------------------
            parsed = HarmonyParser.parse(raw)
            last_parse = parsed

            if parsed.analysis:
                self._emit("thinking", parsed.analysis)

            # ---- 5a. 参数 JSON 坏了 → 重试一次，再坏就降级 ---------
            if parsed.malformed and parsed.has_tool_call:
                broken = next((tc for tc in parsed.tool_calls if tc.arguments is None), None)
                broken_name = broken.name if broken else "工具"

                if not retry_used:
                    retry_used = True
                    tip = (
                        f"你刚才调用 {broken_name} 时参数不是合法 JSON。"
                        f"错误信息：{broken.error if broken else '未知'}。\n"
                        "请重新输出一次工具调用，参数必须是一个合法的 JSON 对象，例如：\n"
                        '<|start|>assistant<|channel|>commentary to=functions.工具名 <|constrain|>json'
                        '<|message|>{"参数名": "参数值"}<|call|>'
                    )
                    self.memory.add_message("user", f"【系统纠正】{tip}")
                    self._emit("retry", f"{broken_name} 的参数不是合法 JSON，正在让模型重试一次……")
                    continue

                # 第二次还坏 → 降级为纯文本回复
                fallback = parsed.final or parsed.commentary or broken.raw_args or raw
                fallback = fallback.strip() or "模型输出的工具参数无法解析。"
                self._emit("degrade", "工具参数连续两次解析失败，已降级为纯文本回复。")
                self.memory.add_assistant(fallback, kind="final")
                self.memory.set_task(stage="done")
                self._emit("final", fallback)
                return fallback

            # ---- 5b. 有工具调用 → 执行、回填、继续下一圈 ----------
            if parsed.has_tool_call:
                # 把「模型要调什么工具」也记进记忆，下一轮拼模板时上下文才完整
                for tc in parsed.tool_calls:
                    if tc.arguments is None:
                        continue
                    self.memory.add_message(
                        "assistant",
                        json.dumps(tc.arguments, ensure_ascii=False),
                        meta={"kind": "tool_call", "tool": tc.name, "arguments": tc.arguments},
                    )

                # 如果模型在 commentary 里说了话（比如「我先查一下资料」），也给用户看
                if parsed.commentary.strip():
                    self.memory.add_assistant(parsed.commentary.strip(), kind="commentary")
                    self._emit("narrate", parsed.commentary.strip())

                for tc in parsed.tool_calls:
                    if tc.arguments is None:
                        continue
                    if self._cancel.is_set():
                        raise AgentCancelled("用户取消了任务")

                    self.memory.set_task(stage="tool")
                    self._emit("tool_call", {"name": tc.name, "args": tc.arguments})

                    # ★ 这里就是「工具执行」：注册表负责跑，需要换模型时自己装卸
                    result = self.registry.call(tc.name, tc.arguments, self.ctx)

                    ok = not result.lstrip().startswith("【工具错误】") and "【联网检索失败】" not in result
                    self.memory.add_tool_result(tc.name, result, ok=ok)
                    self._emit("tool_result", {"name": tc.name, "result": result})

                    if "【联网检索失败】" in result:
                        self._emit("network_error", result)

                # 工具跑完，回到循环顶部让大脑接着推理
                continue

            # ---- 5c. 普通文字回复（含降级）-----------------------
            final_text = (parsed.final or parsed.commentary or raw).strip()
            if not final_text:
                final_text = "（模型没有给出有效回复）"

            self.memory.add_assistant(final_text, kind="final")
            self.memory.set_task(stage="done", pending_image="")
            self._emit("final", final_text)
            return final_text

        # ---- 6. 步数用完了还没 final：把最后一轮内容当回答 ----------
        if last_parse is not None:
            final_text = (last_parse.final or last_parse.commentary or "").strip()
        if not final_text:
            final_text = f"任务转了 {self.max_steps} 步还没得到最终回答。你可以把要求说得更具体一点，或者直接说「生成一张……」我再试。"
        self.memory.add_assistant(final_text, kind="final")
        self.memory.set_task(stage="done")
        self._emit("final", final_text)
        return final_text

    # -- 中文错误转换 --------------------------------------------------
    def _friendly_error(self, e: Exception) -> str:
        name = type(e).__name__
        text = str(e)
        if "ModelNotReady" in name or "ModelFileMissing" in name:
            return text
        if "memory" in text.lower() or "alloc" in text.lower():
            return f"内存不够了：{text}\n建议关掉其他大程序（浏览器、游戏），或者在 config.json 里把 llm.n_ctx 调小。"
        if isinstance(e, AgentCancelled):
            return "任务已取消。"
        return f"{name}: {text}\n详细日志见 logs/app.log。"


# ===========================================================================
# 自测：python agent_core.py
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SAMPLE_CALL = (
        "<|start|>assistant<|channel|>analysis<|message|>用户要洛丽塔裙子，我得先查一下它的视觉特征。<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.search_knowledge "
        '<|constrain|>json<|message|>{"query":"洛丽塔服饰"}<|call|>'
    )
    SAMPLE_BAD = (
        "<|start|>assistant<|channel|>commentary to=functions.generate_image "
        "<|constrain|>json<|message|>{prompt: lolita dress,<|call|>"
    )
    SAMPLE_FINAL = (
        "<|start|>assistant<|channel|>final<|message|>已经帮你画好啦，右边就是生成的图片。<|end|>"
    )
    SAMPLE_FENCE = (
        "<|start|>assistant<|channel|>commentary to=functions.generate_image "
        '<|constrain|>json<|message|>```json\n{"prompt": "lolita girl, frilly dress"}\n```<|call|>'
    )
    SAMPLE_CONSTRAINT = (
        "<|start|>assistant<|channel|>commentary to=functions.get_current_time "
        '<|constraint|>json<|message|>{}<|call|>'
    )

    print("=" * 60)
    print("测试 1：标准工具调用")
    p = HarmonyParser.parse(SAMPLE_CALL)
    print("  analysis:", p.analysis)
    print("  tool_calls:", [(t.name, t.arguments) for t in p.tool_calls])
    assert p.tool_calls[0].name == "search_knowledge"
    assert p.tool_calls[0].arguments == {"query": "洛丽塔服饰"}

    print("\n测试 2：坏 JSON（应标记 malformed）")
    p = HarmonyParser.parse(SAMPLE_BAD)
    print("  malformed:", p.malformed, "| tool:", p.tool_calls[0].name, "| err:", p.tool_calls[0].error[:60])
    assert p.malformed is True
    assert p.tool_calls[0].arguments is None

    print("\n测试 3：最终回答")
    p = HarmonyParser.parse(SAMPLE_FINAL)
    print("  final:", p.final)
    assert "已经帮你画好啦" in p.final

    print("\n测试 4：代码围栏里的 JSON")
    p = HarmonyParser.parse(SAMPLE_FENCE)
    print("  args:", p.tool_calls[0].arguments)
    assert p.tool_calls[0].arguments["prompt"].startswith("lolita")

    print("\n测试 5：constraint 拼写变体")
    p = HarmonyParser.parse(SAMPLE_CONSTRAINT)
    print("  tool:", p.tool_calls[0].name, p.tool_calls[0].arguments)
    assert p.tool_calls[0].name == "get_current_time"

    print("\n测试 6：完全没有 harmony 标记（纯文本降级）")
    p = HarmonyParser.parse("你好，我是一段普通回答。")
    print("  final:", p.final, "| plain:", p.plain_text_only)
    assert p.plain_text_only is True

    print("\n测试 7：裸 JSON 工具调用（非标准兜底）")
    p = HarmonyParser.parse('{"name": "see_image", "arguments": {"question": "描述"}}')
    print("  tool:", p.tool_calls[0].name, p.tool_calls[0].arguments)
    assert p.tool_calls[0].name == "see_image"

    print("\n" + "=" * 60)
    print("全部解析测试通过 ✓ 本鱼这条鱼没白吃 token")
