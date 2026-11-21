# -*- coding: utf-8 -*-
"""
tools.py —— 工具注册表 + 四个内置工具（大肥鱼手写）

一、工具注册表（可无限扩展）
    - 每个工具 = 名称 + 中文描述 + 参数 JSON Schema + 执行函数 + 依赖哪种模型
    - 用 @registry.tool(...) 装饰器就能注册，第三方工具一行都不用改主循环
    - 「依赖哪种模型」告诉调度器：跑这个工具前后要不要换模型（视觉/生图/不占模型）

二、四个内置工具
    1. see_image           看图理解（需要视觉模型，用完卸载）
    2. generate_image      生成图片（需要生图模型，用完卸载）
    3. search_knowledge    联网知识检索 ★ 解决「生图模型看不懂小众词汇」
    4. get_current_time    获取时间（不占模型）

三、联网检索的设计要点
    生图模型（SD1.5）只认英文视觉描述，不认识「洛丽塔」「蒸汽波」这种词。
    所以流程是：
       用户说「生成洛丽塔裙子的人物」
         → 大脑（GPT-OSS）意识到这个词自己说不准视觉细节
         → 调 search_knowledge 查中文百科/搜索结果
         → 资料写进共享记忆（knowledge 区）
         → 大脑读着资料，翻译成 SD 听得懂的英文提示词
         → 调 generate_image
    断网时：工具返回中文失败说明 + 通知界面弹窗，大脑继续用已有常识降级生成。
"""

from __future__ import annotations

import html
import json
import logging
import os
import random
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("gptoss.tools")


# ===========================================================================
# 一、网络小工具（不依赖 requests，纯标准库，打包体积小）
# ===========================================================================


class NetworkError(RuntimeError):
    """断网 / 网络被阻断，UI 收到这个要弹中文提示。"""


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def _decode_body(raw: bytes, content_type: str = "") -> str:
    """按响应头/常见编码依次尝试解码，中文网页别乱码。"""
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    encodings: list[str] = []
    if m:
        encodings.append(m.group(1))
    encodings += ["utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", "ignore")


def http_get(url: str, timeout: float = 8.0, headers: dict[str, str] | None = None) -> str:
    """GET 一个网页/接口，返回文本。网络层错误统一抛 NetworkError。"""
    hdrs = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "") if resp.headers else ""
            # 处理 gzip 压缩（有的服务器即使没要也会压）
            enc = (resp.headers.get("Content-Encoding", "") if resp.headers else "").lower()
            if enc == "gzip":
                import gzip as _gz

                try:
                    raw = _gz.decompress(raw)
                except Exception:
                    pass
            return _decode_body(raw, ctype)
    except urllib.error.HTTPError as e:
        # 4xx/5xx 不算断网，只是这个源没结果
        raise RuntimeError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, socket.timeout, TimeoutError, ssl.SSLError, ConnectionError) as e:
        raise NetworkError(f"网络连接失败：{e}") from e
    except Exception as e:  # noqa: BLE001
        raise NetworkError(f"网络请求异常：{e}") from e


def strip_tags(s: str) -> str:
    """去掉 HTML 标签 + 反转义实体，压成一行干净文本。"""
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s or "")
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ===========================================================================
# 二、联网知识检索器
# ===========================================================================


@dataclass
class KnowledgeResult:
    ok: bool
    query: str
    title: str = ""
    summary: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    error: str = ""


class WebKnowledge:
    """联网查资料的实现：中文维基 → 英文维基 → Bing 搜索摘要，三级兜底。

    为什么不用付费 API：不花钱、不填 key，用户开箱即用。
    为什么维基不总通：国内访问 zh.wikipedia.org 时好时坏，所以必须用 Bing 兜底。
    """

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout
        self._network_failures = 0
        self._network_ok = False

    # -- 维基百科 ------------------------------------------------------
    def _wiki_opensearch(self, query: str, lang: str) -> tuple[str, list[dict[str, str]]]:
        """用维基的搜索接口找最可能的词条标题。"""
        url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=opensearch&format=json&limit=3&namespace=0"
            f"&search={urllib.parse.quote(query)}"
        )
        text = http_get(url, timeout=self.timeout)
        data = json.loads(text)
        self._network_ok = True
        if not isinstance(data, list) or len(data) < 2:
            return "", []
        titles = data[1] or []
        urls = data[3] if len(data) > 3 else []
        srcs = [
            {"title": t, "url": urls[i] if i < len(urls) else ""}
            for i, t in enumerate(titles[:3])
        ]
        return (titles[0] if titles else ""), srcs

    def _wiki_summary(self, title: str, lang: str) -> str:
        """取维基词条的摘要正文（就是我们要的「视觉特征」来源）。"""
        url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(title.replace(' ', '_'))}"
        )
        text = http_get(url, timeout=self.timeout)
        data = json.loads(text)
        self._network_ok = True
        extract = data.get("extract") or ""
        return strip_tags(extract)

    def _wiki_lookup(self, query: str, lang: str) -> KnowledgeResult | None:
        title, srcs = self._wiki_opensearch(query, lang)
        if not title:
            return None
        summary = self._wiki_summary(title, lang)
        if not summary:
            return None
        return KnowledgeResult(
            ok=True,
            query=query,
            title=title,
            summary=summary,
            sources=srcs or [{"title": title, "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"}],
        )

    # -- Bing 搜索 -----------------------------------------------------
    def _bing_lookup(self, query: str) -> KnowledgeResult | None:
        url = (
            "https://cn.bing.com/search?q="
            + urllib.parse.quote(query)
            + "&setlang=zh-CN&ensearch=0"
        )
        page = http_get(url, timeout=self.timeout)
        self._network_ok = True

        # Bing 的搜索结果条目大致长这样：
        # <li class="b_algo"><h2><a href="URL">标题</a></h2>...<p>摘要</p></li>
        sources: list[dict[str, str]] = []
        snippets: list[str] = []
        for m in re.finditer(
            r'(?is)<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>', page
        ):
            block = m.group(1)
            link = re.search(r'(?is)<h2[^>]*>.*?<a[^>]+href="([^"]+)"', block)
            if not link:
                link = re.search(r'(?is)<a[^>]+href="(https?://[^"]+)"', block)
            href = html.unescape(link.group(1)) if link else ""
            if not href or "bing.com" in href or href.startswith("javascript"):
                continue
            title_m = re.search(r"(?is)<h2[^>]*>(.*?)</h2>", block)
            title = strip_tags(title_m.group(1)) if title_m else ""
            snip_m = re.search(r"(?is)<p[^>]*>(.*?)</p>", block)
            snippet = strip_tags(snip_m.group(1)) if snip_m else ""
            if snippet or title:
                sources.append({"title": title, "url": href})
                if snippet:
                    snippets.append(snippet)
            if len(sources) >= 3:
                break

        if not snippets:
            # 网页结构变了也不能崩，退化成「整页正文里抓中文段落」
            plain = strip_tags(page)
            chunks = [c for c in re.split(r"[。！？\n]", plain) if len(c) > 25]
            # 跳过导航栏那些噪音，取中间偏后的段落
            snippets = chunks[3:9] if len(chunks) > 9 else chunks[:6]

        if not snippets:
            return None
        return KnowledgeResult(
            ok=True,
            query=query,
            title=query,
            summary="；".join(snippets[:5]),
            sources=sources,
        )

    # -- 对外主入口 ----------------------------------------------------
    def lookup(self, query: str, lang: str = "zh") -> KnowledgeResult:
        """查一个词的资料。返回结构化结果；彻底断网时抛 NetworkError。

        策略：
          1) 中文维基（中文互联网最好的百科源）
          2) 英文维基（很多艺术流派/服饰术语只有英文条目，但英文描述对 SD 反而更好）
          3) Bing 搜索结果摘要（前两个都不通或被屏蔽时的兜底）
        """
        query = (query or "").strip()
        if not query:
            return KnowledgeResult(ok=False, query=query, error="查询词为空")

        last_error = ""
        errors_are_network = 0
        attempts = 0

        # 1) 中文维基
        attempts += 1
        try:
            r = self._wiki_lookup(query, "zh")
            if r:
                return r
            last_error = "中文维基没有该词条"
        except NetworkError as e:
            errors_are_network += 1
            last_error = str(e)
        except Exception as e:  # noqa: BLE001
            last_error = f"中文维基查询失败：{e}"

        # 2) 英文维基
        attempts += 1
        try:
            r = self._wiki_lookup(query, "en")
            if r:
                return r
            last_error = "英文维基没有该词条"
        except NetworkError as e:
            errors_are_network += 1
            last_error = str(e)
        except Exception as e:  # noqa: BLE001
            last_error = f"英文维基查询失败：{e}"

        # 3) Bing
        attempts += 1
        try:
            r = self._bing_lookup(query)
            if r:
                return r
            last_error = "搜索引擎没有返回有效摘要"
        except NetworkError as e:
            errors_are_network += 1
            last_error = str(e)
        except Exception as e:  # noqa: BLE001
            last_error = f"搜索引擎查询失败：{e}"

        # 三次全是网络错误 → 判定断网（这个要弹窗）
        if errors_are_network >= attempts:
            raise NetworkError(last_error or "网络不可用")

        return KnowledgeResult(ok=False, query=query, error=last_error or "没查到资料")

    def is_online(self) -> bool:
        """轻量探活：有没有网（给界面状态用，最多等 6 秒）。"""
        for url in ("https://cn.bing.com/", "https://zh.wikipedia.org/"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                with urllib.request.urlopen(req, timeout=3) as _resp:
                    return True
            except Exception:
                continue
        return False


# ===========================================================================
# 三、工具注册表
# ===========================================================================


@dataclass
class ToolContext:
    """工具执行时能拿到的东西：记忆、模型管理器、配置、事件回调、人工确认。"""

    memory: Any                      # memory.SharedMemory
    loader: Any                      # loader.ModelManager
    config: Any                      # config.Config
    web: WebKnowledge                # 联网检索器
    emit: Callable[[str, Any], None] = lambda kind, payload: None  # 往 UI 推事件
    # ★ 危险操作前问用户：返回 True 才是同意。默认拒绝，安全第一。
    confirm: Callable[[str], bool] = lambda message: False

    def emit_event(self, kind: str, payload: Any = None) -> None:
        try:
            self.emit(kind, payload)
        except Exception:
            pass


@dataclass
class ToolSpec:
    """一个工具的全部信息。"""

    name: str                              # 模型看到的函数名（英文，snake_case）
    description: str                       # 中文描述（给模型看的「什么时候用我」）
    parameters: dict[str, Any]             # JSON Schema
    func: Callable[[dict[str, Any], ToolContext], str]
    requires: str = "none"                 # none / vision / image —— 决定要不要换模型


class ToolRegistry:
    """工具注册表：注册、查询、校验、执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # -- 注册 ----------------------------------------------------------
    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        *,
        requires: str = "none",
    ) -> Callable[[Callable[[dict[str, Any], ToolContext], str]], Callable[[dict[str, Any], ToolContext], str]]:
        """装饰器注册方式：

            @registry.tool("my_tool", "我的工具说明", {...}, requires="none")
            def my_tool(args, ctx):
                return "结果"
        """

        def deco(fn: Callable[[dict[str, Any], ToolContext], str]):
            self.register(ToolSpec(name, description, parameters or {"type": "object", "properties": {}}, fn, requires))
            return fn

        return deco

    def register(self, spec: ToolSpec) -> None:
        """直接注册一个 ToolSpec（也支持运行中动态加工具）。"""
        if not spec.name or not spec.name.isascii():
            raise ValueError(f"工具名必须是英文标识符：{spec.name!r}")
        self._tools[spec.name] = spec
        log.info("已注册工具：%s（依赖模型：%s）", spec.name, spec.requires)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- 查询 ----------------------------------------------------------
    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def model_tools(self) -> list[dict[str, Any]]:
        """给 prompt 渲染用的工具清单。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "requires": t.requires,
            }
            for t in self._tools.values()
        ]

    def describe_for_human(self) -> str:
        """给用户看的中文工具说明（UI 里展示「这个 Agent 会什么」）。"""
        lines = []
        for t in self._tools.values():
            tag = {"vision": "需要视觉模型", "image": "需要生图模型"}.get(t.requires, "不占模型")
            lines.append(f"· {t.name} —— {t.description}（{tag}）")
        return "\n".join(lines)

    # -- 执行 ----------------------------------------------------------
    def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        """执行工具。所有异常都吞掉转成中文错误串返回给模型，绝不让主循环崩。"""
        spec = self._tools.get(name)
        if spec is None:
            return f"【工具错误】不存在的工具：{name}。可用工具：{', '.join(self._tools) }"

        # 参数校验：必填字段不能少
        missing = [
            k
            for k, v in (spec.parameters.get("properties") or {}).items()
            if k in (spec.parameters.get("required") or []) and k not in args
        ]
        if missing:
            return f"【工具错误】{name} 缺少必填参数：{', '.join(missing)}"

        t0 = time.time()
        try:
            result = spec.func(args, ctx)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except NetworkError as e:
            ctx.emit_event("network_error", str(e))
            result = f"【联网检索失败】{e}。已跳过联网，继续根据已有知识生成。"
        except Exception as e:  # noqa: BLE001
            log.exception("工具 %s 执行失败", name)
            result = f"【工具错误】{name} 执行失败：{type(e).__name__}: {e}"
        finally:
            log.info("工具 %s 耗时 %.2fs", name, time.time() - t0)
        return result


# ===========================================================================
# 四、内置工具实现
# ===========================================================================


def _resolve_image_arg(args: dict[str, Any], ctx: ToolContext) -> str:
    """决定要看哪张图：参数 > 任务里的待处理图 > 最近上传的图。"""
    p = (args.get("image_path") or "").strip()
    if p:
        return p
    task = ctx.memory.get_task()
    if task.get("pending_image"):
        return str(task["pending_image"])
    latest = ctx.memory.latest_input_image()
    if latest:
        return str(latest.get("path", ""))
    return ""


def tool_see_image(args: dict[str, Any], ctx: ToolContext) -> str:
    """看图理解：把图片交给视觉模型，得到中文描述，写进共享记忆。"""
    path = _resolve_image_arg(args, ctx)
    question = (args.get("question") or "请用中文详细描述这张图片：画面主体、构图、颜色、风格、细节。").strip()

    if not path:
        return "【看图失败】没有找到要看的图片。请先让用户上传图片，或在参数 image_path 里给一个路径。"
    if not Path(path).exists():
        return f"【看图失败】图片不存在：{path}"

    ctx.emit_event("status", f"正在看图片：{Path(path).name}（加载视觉模型）")
    # borrow() 会自动：卸载语言模型 → 加载视觉模型 → 用完卸载视觉模型 → 重新加载语言模型
    with ctx.loader.borrow("vision") as vlm:
        desc = vlm.describe(path, question)

    # ★ 视觉模块的产出写进共享记忆，语言模型下一轮就能读到
    ctx.memory.add_image_description(path, desc, question=question, model=vlm.name)
    ctx.emit_event("vision_done", {"path": path, "description": desc})
    return f"【看图结果】{desc}"


def tool_generate_image(args: dict[str, Any], ctx: ToolContext) -> str:
    """生成图片：交给生图模型画，落盘到 outputs/，结果写进共享记忆。"""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return "【出图失败】缺少 prompt（英文提示词）。"

    neg = (args.get("negative_prompt") or "").strip()
    width = int(args.get("width") or 0)
    height = int(args.get("height") or 0)
    steps = int(args.get("steps") or 0)

    ctx.emit_event("status", "正在生成图片（加载生图模型）")
    with ctx.loader.borrow("image") as sd:
        out_path = sd.generate(
            prompt,
            negative_prompt=neg or None,
            width=width or None,
            height=height or None,
            steps=steps or None,
        )

    # ★ 生图模块的产出也写进共享记忆
    ctx.memory.add_generated_image(
        str(out_path),
        prompt,
        params={"negative_prompt": neg, "width": width, "height": height, "steps": steps},
        model=sd.name,
    )
    ctx.emit_event("image_ready", str(out_path))
    return f"【出图完成】图片已保存到：{out_path}"


def tool_search_knowledge(args: dict[str, Any], ctx: ToolContext) -> str:
    """联网查知识：把陌生词的视觉特征查回来，存进共享记忆。"""
    query = (args.get("query") or "").strip()
    if not query:
        return "【检索失败】缺少 query。"

    ctx.emit_event("status", f"正在联网查资料：{query}")
    try:
        result = ctx.web.lookup(query, lang=(args.get("lang") or "zh"))
    except NetworkError as e:
        # 断网：写一条失败记录进记忆（让模型知道「查过了没查到」），然后让主循环继续
        ctx.memory.add_knowledge(query, "", ok=False)
        ctx.emit_event("network_error", str(e))
        return (
            f"【联网检索失败】{e}。当前没有可用网络，已跳过联网检索。"
            "请基于你已有的常识给出英文视觉描述，并提醒用户这次没有联网补充资料。"
        )

    if not result.ok:
        ctx.memory.add_knowledge(query, "", ok=False)
        return f"【检索无结果】「{query}」没有查到资料（{result.error}）。请基于已有常识继续。"

    # ★★ 关键：查到的资料进共享记忆，生图模型看不懂的词，大脑已经把它变成了具体描述
    ctx.memory.add_knowledge(
        query,
        result.summary,
        sources=[s for s in result.sources if s.get("url")],
        lang="zh",
        ok=True,
    )
    src_text = "；".join(
        f"{s.get('title', '')} {s.get('url', '')}".strip() for s in result.sources[:3] if s.get("url")
    )
    ctx.emit_event(
        "knowledge_done",
        {"query": query, "title": result.title, "summary": result.summary, "sources": result.sources},
    )
    return (
        f"【联网资料】查询词：{query}\n"
        f"标题：{result.title}\n"
        f"摘要：{result.summary}\n"
        f"来源：{src_text or '（未记录）'}\n"
        "请把上面这段资料提炼成生图模型能懂的英文视觉描述。\n"
        "如果这段资料主要在讲剧情、历史、人物关系，而不是「外观长什么样」，"
        "请换一个更具体的查询词再查一次（例如把「洛丽塔」改成「洛丽塔服饰」或英文「Lolita fashion」）。"
    )


def tool_get_current_time(args: dict[str, Any], ctx: ToolContext) -> str:
    """获取当前时间（本地时区）。"""
    now = time.localtime()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return time.strftime(f"%Y年%m月%d日 {weekdays[now.tm_wday]} %H:%M:%S", now)


# ===========================================================================
# 五、电脑操作工具（让 Agent 真的能帮你干活，而不只是画图）
# ===========================================================================

def _decode_bytes(raw: bytes) -> str:
    """Windows 命令输出可能是 UTF-8 也可能是 GBK，挨个试。"""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _human_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def _blacklist_hit(command: str, blacklist: list[str]) -> str:
    """检查命令有没有踩到危险词。返回命中的词，没命中返回空串。"""
    low = (command or "").lower()
    for word in blacklist:
        if word.lower() in low:
            return word
    return ""


def tool_run_command(args: dict[str, Any], ctx: ToolContext) -> str:
    """执行一条命令行命令（cmd / powershell），把输出返回给模型。

    这是 Agent 的「手」：装软件、跑脚本、查系统信息、批量改名……都靠它。
    """
    import subprocess

    conf = ctx.config.section("computer") if hasattr(ctx.config, "section") else {}
    if conf.get("enabled", True) is False:
        return "【命令被禁用】config.json 里 computer.enabled 是 false。"

    command = str(args.get("command") or "").strip()
    if not command:
        return "【命令失败】command 不能为空。"
    if len(command) > 4000:
        return "【命令失败】命令太长（超过 4000 字符），分几步来吧。"

    cwd = str(args.get("cwd") or Path.home())
    if not Path(cwd).exists():
        cwd = str(Path.home())

    timeout = int(args.get("timeout") or conf.get("command_timeout", 120))
    timeout = max(5, min(timeout, 600))

    # 危险命令先问用户（不是禁用）
    hit = _blacklist_hit(command, list(conf.get("blacklist") or []))
    if hit and conf.get("confirm_dangerous", True):
        ok = ctx.confirm(
            f"Agent 想执行一条可能有风险的操作：\n\n{command}\n\n（命中敏感词：{hit}）\n\n要放行吗？"
        )
        if not ok:
            return f"【已取消】用户不同意执行这条命令（命中敏感词 {hit}）。请换一种更安全的做法。"

    ctx.emit_event("status", f"正在执行命令：{command[:60]}")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"【命令超时】跑了 {timeout} 秒还没结束，已经掐掉。可以把 timeout 调大或拆成小步骤。"
    except Exception as e:  # noqa: BLE001
        return f"【命令执行失败】{type(e).__name__}: {e}"

    stdout = _decode_bytes(proc.stdout or b"").strip()
    stderr = _decode_bytes(proc.stderr or b"").strip()
    limit = int(conf.get("max_output_chars", 4000))

    def clip(s: str) -> str:
        if len(s) > limit:
            return s[:limit] + f"\n……（输出太长，已截断，共 {len(s)} 字符）"
        return s

    lines = [f"【命令完成】退出码 {proc.returncode}", f"工作目录：{cwd}", f"命令：{command}"]
    if stdout:
        lines.append(f"标准输出：\n{clip(stdout)}")
    if stderr:
        lines.append(f"错误输出：\n{clip(stderr)}")
    if not stdout and not stderr:
        lines.append("（没有任何输出）")
    return "\n".join(lines)


def tool_read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """读一个文本文件的内容。"""
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return "【读文件失败】path 不能为空。"
    p = Path(raw_path)
    if not p.exists():
        return f"【读文件失败】文件不存在：{p}"
    if p.is_dir():
        return f"【读文件失败】{p} 是个目录，请用 list_dir。"

    conf = ctx.config.section("computer") if hasattr(ctx.config, "section") else {}
    limit = int(args.get("max_chars") or conf.get("max_read_chars", 12000))
    try:
        if p.stat().st_size > 20 * 1024 * 1024:
            return f"【读文件失败】文件太大（{_human_size(p.stat().st_size)}），超过 20MB 不读。"
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"【读文件失败】{type(e).__name__}: {e}"

    if len(text) > limit:
        text = text[:limit] + f"\n……（文件共 {len(text)} 字符，已截断）"
    return f"【文件内容】{p}\n{text}"


def tool_write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """写一个文本文件（覆盖或追加），目录不存在会自动创建。"""
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return "【写文件失败】path 不能为空。"
    content = args.get("content")
    if content is None:
        return "【写文件失败】content 不能为空。"
    content = str(content)
    append = bool(args.get("append"))
    p = Path(raw_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a" if append else "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:  # noqa: BLE001
        return f"【写文件失败】{type(e).__name__}: {e}"
    action = "追加写入" if append else "写入"
    return f"【写文件成功】{action} {len(content)} 个字符到 {p}"


def tool_list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    """列出一个目录里有什么。"""
    raw_path = str(args.get("path") or str(Path.home())).strip()
    p = Path(raw_path)
    if not p.exists():
        return f"【列目录失败】目录不存在：{p}"
    if p.is_file():
        return f"【列目录失败】{p} 是个文件，不是目录。"

    try:
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except Exception as e:  # noqa: BLE001
        return f"【列目录失败】{type(e).__name__}: {e}"

    if not items:
        return f"【目录内容】{p}\n（空目录）"
    lines = [f"【目录内容】{p}（共 {len(items)} 项）"]
    for item in items[:120]:
        try:
            if item.is_dir():
                lines.append(f"[目录] {item.name}")
            else:
                lines.append(f"       {item.name}  {_human_size(item.stat().st_size)}")
        except Exception:
            lines.append(f"       {item.name}")
    if len(items) > 120:
        lines.append(f"……还有 {len(items) - 120} 项没显示")
    return "\n".join(lines)


def tool_open_path(args: dict[str, Any], ctx: ToolContext) -> str:
    """用系统默认程序打开文件/文件夹/网址。"""
    import webbrowser

    target = str(args.get("path") or "").strip()
    if not target:
        return "【打开失败】path 不能为空。"
    try:
        if target.startswith(("http://", "https://")):
            webbrowser.open(target)
            return f"【已打开】在浏览器里打开：{target}"
        p = Path(target)
        if not p.exists():
            return f"【打开失败】路径不存在：{p}"
        os.startfile(str(p))  # type: ignore[attr-defined]
        return f"【已打开】用默认程序打开：{p}"
    except Exception as e:  # noqa: BLE001
        return f"【打开失败】{type(e).__name__}: {e}"


def tool_delete_path(args: dict[str, Any], ctx: ToolContext) -> str:
    """删除文件或文件夹。★ 每次都会弹窗问用户，不会先斩后奏。"""
    import shutil

    target = str(args.get("path") or "").strip()
    if not target:
        return "【删除失败】path 不能为空。"
    p = Path(target)
    if not p.exists():
        return f"【删除失败】路径不存在：{p}"

    # 防呆：盘符根目录 / 用户主目录 / 系统目录，一律不给删
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    dangerous_roots = {
        Path(resolved.anchor) if resolved.anchor else None,
        Path.home(),
        Path(os.environ.get("SystemRoot", "C:/Windows")),
        Path(os.environ.get("ProgramFiles", "C:/Program Files")),
    }
    if resolved in {r for r in dangerous_roots if r is not None}:
        return "【拒绝】这是系统根目录/用户主目录，不能删。"

    conf = ctx.config.section("computer") if hasattr(ctx.config, "section") else {}
    if conf.get("confirm_delete", True):
        kind = "文件夹（里面所有东西一起没）" if resolved.is_dir() else "文件"
        ok = ctx.confirm(f"要删除这个{kind}吗？\n\n{resolved}\n\n删了不能恢复，确定吗？")
        if not ok:
            return "【已取消】用户不同意删除。请换个方案或告诉用户让他自己删。"

    try:
        if resolved.is_dir():
            shutil.rmtree(resolved)
            return f"【删除成功】已删除文件夹（连同里面所有内容）：{resolved}"
        resolved.unlink()
        return f"【删除成功】已删除文件：{resolved}"
    except Exception as e:  # noqa: BLE001
        return f"【删除失败】{type(e).__name__}: {e}"


def tool_screenshot(args: dict[str, Any], ctx: ToolContext) -> str:
    """截屏保存成图片，Agent 可以接着用 see_image 看自己截的图。"""
    try:
        from PIL import ImageGrab
    except Exception as e:  # noqa: BLE001
        return f"【截屏失败】缺少 Pillow：{e}"

    try:
        out_dir = ctx.config.output_dir if hasattr(ctx.config, "output_dir") else Path("outputs")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        img = ImageGrab.grab()
        img.save(out)
        ctx.emit_event("image_ready", str(out))
        return f"【截屏成功】已保存：{out}（想看图里的内容就调用 see_image）"
    except Exception as e:  # noqa: BLE001
        return f"【截屏失败】{type(e).__name__}: {e}"


def tool_system_info(args: dict[str, Any], ctx: ToolContext) -> str:
    """报一下这台电脑的基本情况（内存、磁盘、CPU）。"""
    import platform
    import shutil

    lines = [f"操作系统：{platform.system()} {platform.release()}（{platform.version()}）"]
    lines.append(f"CPU：{platform.processor() or '未知'}，{os.cpu_count()} 个逻辑核心")
    try:
        total = 0
        free = 0
        for letter in "CDEFG":
            try:
                t, _u, f = shutil.disk_usage(f"{letter}:\\")
                if t > 0:
                    lines.append(f"{letter} 盘：空闲 {f / 2**30:.1f} GB / 共 {t / 2**30:.1f} GB")
                    total += t
                    free += f
            except Exception:
                continue
        if total:
            lines.append(f"磁盘合计：空闲 {free / 2**30:.1f} GB / 共 {total / 2**30:.1f} GB")
    except Exception:
        pass
    try:
        lines.append(f"用户目录：{Path.home()}")
    except Exception:
        pass
    return "【系统信息】\n" + "\n".join(lines)


# ===========================================================================
# 六、组装默认注册表
# ===========================================================================


def build_default_registry() -> ToolRegistry:
    """建立带四个内置工具的工具注册表。想加工具就在这里加一行。"""
    reg = ToolRegistry()

    reg.register(
        ToolSpec(
            name="see_image",
            description=(
                "看图理解。把用户上传的图片交给视觉模型，返回画面的中文详细描述（主体、构图、颜色、风格、细节）。"
                "当用户让你「看这张图」「分析这张图」「照这张图改」时必须先调用它。"
                "不传 image_path 时自动使用用户刚上传的那张图。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片的本地路径。留空表示使用用户刚上传的图片。",
                    },
                    "question": {
                        "type": "string",
                        "description": "你希望视觉模型重点描述什么，例如「描述人物服装的款式和颜色」。",
                    },
                },
                "required": [],
            },
            func=tool_see_image,
            requires="vision",
        )
    )

    reg.register(
        ToolSpec(
            name="generate_image",
            description=(
                "生成图片。根据一段【英文】提示词画一张新图并保存到本地。"
                "prompt 必须是英文！Stable Diffusion 只认英文，中文提示词画不出东西。"
                "如果提示词里涉及你不确定的小众风格、服饰、艺术流派、专有名词，"
                "请先调用 search_knowledge 查清它的视觉特征，再把特征翻译成英文写进 prompt。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "英文生图提示词，用逗号分隔关键词，例如：lolita fashion girl, frilly dress, lace, ribbon, street",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "英文反向提示词，表示不要什么。可留空用默认值。",
                    },
                    "width": {"type": "integer", "description": "图片宽度，默认 512"},
                    "height": {"type": "integer", "description": "图片高度，默认 512"},
                    "steps": {"type": "integer", "description": "采样步数，默认 12，越大越慢"},
                },
                "required": ["prompt"],
            },
            func=tool_generate_image,
            requires="image",
        )
    )

    reg.register(
        ToolSpec(
            name="search_knowledge",
            description=(
                "联网知识检索。查一个小众词、风格名、服饰名、艺术流派、专有名词的资料，"
                "重点是要拿到「它在视觉上具体长什么样」的描述。"
                "当你准备写生图提示词，但发现某个词你讲不清视觉细节（例如洛丽塔、蒸汽波、哥特、赛博朋克、新艺术运动），"
                "必须先调用本工具查资料，查到的内容会自动存进共享记忆，你再据此写英文提示词。"
                "断网时本工具会返回失败说明，你应当基于已有常识继续完成任务。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查的词，例如：洛丽塔服饰"},
                    "lang": {"type": "string", "description": "语言偏好，zh 或 en，默认 zh"},
                },
                "required": ["query"],
            },
            func=tool_search_knowledge,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="get_current_time",
            description="获取当前的日期、星期和时间。用户问现在几点、今天几号时调用。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=tool_get_current_time,
            requires="none",
        )
    )

    # ------------------------------------------------------------------
    # 电脑操作工具：让 Agent 真的能帮用户干活，而不只是画图
    # ------------------------------------------------------------------
    reg.register(
        ToolSpec(
            name="run_command",
            description=(
                "在用户的 Windows 电脑上执行一条命令行命令（cmd），返回退出码和输出。"
                "能用来装软件、跑脚本、查系统状态、批量处理文件、调用其他程序。"
                "遇到高危命令（格式化磁盘、删系统文件、改注册表等）会先弹窗让用户确认。"
                "命令要写成一行，别用交互式命令（会卡住）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令，例如 dir D:\\ 或 python -V"},
                    "cwd": {"type": "string", "description": "在哪个目录下执行，默认用户主目录"},
                    "timeout": {"type": "integer", "description": "最多等多少秒，默认 120"},
                },
                "required": ["command"],
            },
            func=tool_run_command,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="read_file",
            description="读一个文本文件的内容（代码、配置、日志、txt 都行）。看文件之前调用它。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件的完整路径"},
                    "max_chars": {"type": "integer", "description": "最多读多少字符，默认 12000"},
                },
                "required": ["path"],
            },
            func=tool_read_file,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="write_file",
            description=(
                "写一个文本文件（自动创建不存在的目录）。append=true 是追加，否则覆盖。"
                "写代码、写配置、写笔记都用它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件的完整路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                    "append": {"type": "boolean", "description": "是否追加到文件末尾，默认 false（覆盖）"},
                },
                "required": ["path", "content"],
            },
            func=tool_write_file,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="list_dir",
            description="列出一个文件夹里有哪些文件和子文件夹，带大小。不知道东西在哪时先用它看看。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件夹路径，默认用户主目录"},
                },
                "required": [],
            },
            func=tool_list_dir,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="open_path",
            description="用系统默认程序打开文件、文件夹或网址（用户能直接看到）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件/文件夹路径，或 http(s) 网址"},
                },
                "required": ["path"],
            },
            func=tool_open_path,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="delete_path",
            description=(
                "删除一个文件或文件夹。★ 每次都会弹窗让用户确认，用户不同意就不删。"
                "只在用户明确要求删除时调用，别自作主张清理东西。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要删除的文件或文件夹路径"},
                },
                "required": ["path"],
            },
            func=tool_delete_path,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="screenshot",
            description=(
                "给当前屏幕截一张图并保存。截完可以接着调用 see_image 看这张图，"
                "用来确认程序界面、看用户桌面上有什么。"
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            func=tool_screenshot,
            requires="none",
        )
    )

    reg.register(
        ToolSpec(
            name="system_info",
            description="报一下这台电脑的 CPU、磁盘剩余空间、用户目录等基本情况。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=tool_system_info,
            requires="none",
        )
    )

    return reg


# ===========================================================================
# 自测：python tools.py
# ===========================================================================
if __name__ == "__main__":
    registry = build_default_registry()
    print("=== 已注册工具 ===")
    print(registry.describe_for_human())
    print("\n=== 给模型看的工具名 ===", registry.names())

    # 用假的 ctx 测一下不需要模型的部分
    class _FakeMem:
        def __init__(self):
            self.knowledge = []

        def get_task(self):
            return {}

        def latest_input_image(self):
            return None

        def add_knowledge(self, q, s, **kw):
            self.knowledge.append((q, s, kw))

    class _FakeCtx:
        memory = _FakeMem()
        loader = None
        config = None

        def emit_event(self, k, p=None):
            pass

    registry_ctx = _FakeCtx()
    registry_ctx.web = WebKnowledge(timeout=6)
    print("\n=== 获取时间 ===")
    print(registry.call("get_current_time", {}, registry_ctx))
    print("\n=== 联网查「洛丽塔」 ===")
    print(registry.call("search_knowledge", {"query": "洛丽塔服饰"}, registry_ctx)[:800])
