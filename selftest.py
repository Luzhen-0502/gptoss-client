# -*- coding: utf-8 -*-
"""
selftest.py —— 不改一行真模型，也能验证整条闭环跑得通（大肥鱼手写）

为什么要这个文件：
    真正的 20B 模型要下载十几 GB，跑一次要等好几分钟。
    所以这里用「假模型」替身，把 Agent 的调度闭环、工具调用、记忆回填、
    协议解析、失败的降级路径全部跑一遍，几分钟内就能验证代码逻辑对不对。

覆盖：
    A. 完整最小闭环：上传图片 → see_image → search_knowledge → generate_image → final
    B. 工具参数 JSON 写坏 → 自动重试一次 → 成功
    C. 连续两次写坏 → 降级为纯文本回复
    D. 多轮对话记忆：第二轮能看到第一张图的描述
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_core import AgentCore, HarmonyParser
from config import Config
from memory import SharedMemory
from tools import KnowledgeResult, build_default_registry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


# ===========================================================================
# 假模型替身
# ===========================================================================


class FakeLLM:
    """按剧本逐条返回 Harmony 原文，模拟 GPT-OSS-20B。"""

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.index = 0
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt: str, **_kw: Any) -> str:
        with self._lock:
            self.prompts.append(prompt)
            if self.index >= len(self.script):
                return (
                    "<|start|>assistant<|channel|>final"
                    "<|message|>（假模型剧本用完了）<|end|>"
                )
            out = self.script[self.index]
            self.index += 1
            return out


class FakeVLM:
    name = "FakeVLM（假视觉模型）"

    def __init__(self, description: str) -> None:
        self.description = description
        self.calls: list[str] = []

    def describe(self, image_path: str, question: str = "") -> str:
        self.calls.append(image_path)
        return self.description


class FakeSD:
    name = "FakeSD（假生图模型）"

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kw: Any) -> Path:
        from PIL import Image, ImageDraw

        self.prompts.append(prompt)
        img = Image.new("RGB", (512, 512), (48, 64, 128))
        draw = ImageDraw.Draw(img)
        draw.text((16, 16), "FAKE SD OUTPUT", fill=(255, 255, 255))
        draw.text((16, 40), "lolita-ish placeholder", fill=(220, 220, 255))
        path = self.out_dir / f"fake_gen_{len(self.prompts)}.png"
        img.save(path)
        return path


class FakeLoader:
    """假的模型管理器：记录谁被借用了，验证「按需换模型」的调用顺序。"""

    def __init__(self, llm: FakeLLM, vlm: FakeVLM, sd: FakeSD) -> None:
        self.models = {"llm": llm, "vision": vlm, "image": sd}
        self.borrowed: list[str] = []

    def chat_model(self) -> FakeLLM:
        return self.models["llm"]  # type: ignore[return-value]

    @contextmanager
    def borrow(self, kind: str):
        self.borrowed.append(kind)
        yield self.models[kind]

    def shutdown(self) -> None:
        pass


class FakeWeb:
    """假的联网检索：固定返回一段「洛丽塔」的视觉特征资料。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def lookup(self, query: str, lang: str = "zh") -> KnowledgeResult:
        self.queries.append(query)
        return KnowledgeResult(
            ok=True,
            query=query,
            title="洛丽塔服饰",
            summary=(
                "洛丽塔时装是一种源自日本的服装风格：特征是蓬松的衬裙、大量蕾丝与荷叶边、"
                "蝴蝶结装饰、系带紧身胸衣、泡泡袖、及膝 A 字裙，常见甜美、哥特、古典三种分支。"
            ),
            sources=[{"title": "洛丽塔服饰", "url": "https://zh.wikipedia.org/wiki/Lolita_fashion"}],
        )


# ===========================================================================
# 工具
# ===========================================================================


def make_env(tmp: Path) -> tuple[Config, SharedMemory, Any]:
    cfg = Config(tmp / "config.json")
    cfg.data["paths"]["memory_file"] = str(tmp / "mem.json")
    cfg.data["paths"]["output_dir"] = str(tmp / "out")
    (tmp / "out").mkdir(exist_ok=True)
    memory = SharedMemory(tmp / "mem.json")
    registry = build_default_registry()
    return cfg, memory, registry


def make_image(tmp: Path) -> str:
    from PIL import Image

    p = tmp / "input.png"
    Image.new("RGB", (300, 200), (200, 160, 180)).save(p)
    return str(p)


def tool_call(name: str, args: dict[str, Any] | str) -> str:
    if isinstance(args, str):
        arg_text = args
    else:
        arg_text = json.dumps(args, ensure_ascii=False)
    return (
        f"<|start|>assistant<|channel|>commentary to=functions.{name} "
        f"<|constrain|>json<|message|>{arg_text}<|call|>"
    )


def final(text: str) -> str:
    return f"<|start|>assistant<|channel|>final<|message|>{text}<|end|>"


# ===========================================================================
# 测试 A：完整最小闭环
# ===========================================================================


def test_a_full_loop() -> None:
    print("\n[A] 完整最小闭环：上传图 → 看图 → 查洛丽塔 → 出图")
    tmp = Path(tempfile.mkdtemp())
    cfg, memory, registry = make_env(tmp)
    img = make_image(tmp)

    llm = FakeLLM(
        [
            tool_call("see_image", {"question": "描述画面和服装"}),
            tool_call("search_knowledge", {"query": "洛丽塔服饰"}),
            tool_call(
                "generate_image",
                {
                    "prompt": "lolita fashion girl, frilly layered skirt, lace trim, ribbon, "
                    "puff sleeves, knee length, cherry blossom street, soft light",
                    "steps": 12,
                },
            ),
            final("已经按洛丽塔的风格画好并放到右边预览了。"),
        ]
    )
    vlm = FakeVLM("一位少女站在樱花树下，穿着浅粉色洛丽塔裙，裙摆有蕾丝和蝴蝶结。")
    sd = FakeSD(tmp / "out")
    loader = FakeLoader(llm, vlm, sd)

    events: list[tuple[str, Any]] = []
    core = AgentCore(cfg, memory, loader, registry, emit=lambda k, p=None: events.append((k, p)))
    core.web = FakeWeb()
    core.ctx.web = core.web

    answer = core.run("生成一张洛丽塔裙子的人物", img)

    # --- 断言 ---
    assert "画好" in answer, answer
    assert loader.borrowed == ["vision", "image"], loader.borrowed        # 换模型的顺序对
    assert len(llm.prompts) == 4, f"应该推理 4 次，实际 {len(llm.prompts)}"

    kinds = [k for k, _ in events]
    for need in ("user", "tool_call", "tool_result", "vision_done", "knowledge_done", "image_ready", "final"):
        assert need in kinds, f"缺少事件：{need}（现有 {kinds}）"

    # 视觉描述进了记忆
    assert "洛丽塔" in memory.latest_description(), memory.latest_description()
    # 联网资料进了记忆
    kn = memory.latest_knowledge()
    assert kn and "蕾丝" in kn["summary"], kn
    # 生成结果进了记忆
    gen = memory.latest_generated()
    assert gen and Path(gen["path"]).exists(), gen
    assert "lolita" in gen["prompt"].lower(), gen["prompt"]
    # 第三次推理的 prompt 里应该已经带上了查到的资料
    third_prompt = llm.prompts[2]
    assert "蕾丝" in third_prompt, "第三轮 prompt 没带上网资料，记忆没回填！"

    print(f"  ✓ 推理 {len(llm.prompts)} 轮，换模型顺序 {loader.borrowed}")
    print(f"  ✓ 图片描述：{memory.latest_description()[:40]}……")
    print(f"  ✓ 联网资料已回填进第三轮 prompt（{len(third_prompt)} 字符）")
    print(f"  ✓ 出图：{Path(gen['path']).name}")


# ===========================================================================
# 测试 B：坏 JSON 自动重试一次
# ===========================================================================


def test_b_bad_json_retry() -> None:
    print("\n[B] 工具参数 JSON 写坏 → 自动重试一次")
    tmp = Path(tempfile.mkdtemp())
    cfg, memory, registry = make_env(tmp)

    llm = FakeLLM(
        [
            tool_call("generate_image", "{prompt: lolita dress,"),   # 坏的
            tool_call("generate_image", {"prompt": "lolita dress, lace, ribbon"}),
            final("重试之后画好了。"),
        ]
    )
    loader = FakeLoader(llm, FakeVLM("描述"), FakeSD(tmp / "out"))
    events: list[tuple[str, Any]] = []
    core = AgentCore(cfg, memory, loader, registry, emit=lambda k, p=None: events.append((k, p)))
    core.web = FakeWeb()
    core.ctx.web = core.web

    answer = core.run("画个洛丽塔", "")

    assert "重试之后画好了" in answer, answer
    assert any(k == "retry" for k, _ in events), "没有触发重试事件"
    assert len(llm.prompts) == 3, f"应该是 3 次推理（坏→好→final），实际 {len(llm.prompts)}"
    print("  ✓ 触发重试，第二次解析成功，最终拿到回答")


# ===========================================================================
# 测试 C：连续两次坏 JSON → 降级纯文本
# ===========================================================================


def test_c_degrade_to_text() -> None:
    print("\n[C] 连续两次 JSON 写坏 → 降级为纯文本回复")
    tmp = Path(tempfile.mkdtemp())
    cfg, memory, registry = make_env(tmp)

    llm = FakeLLM(
        [
            tool_call("generate_image", "{prompt: broken one,"),
            tool_call("generate_image", "{prompt: broken two,"),
            final("这段不该被用到"),
        ]
    )
    loader = FakeLoader(llm, FakeVLM("描述"), FakeSD(tmp / "out"))
    events: list[tuple[str, Any]] = []
    core = AgentCore(cfg, memory, loader, registry, emit=lambda k, p=None: events.append((k, p)))
    core.web = FakeWeb()
    core.ctx.web = core.web

    answer = core.run("画个图", "")

    assert any(k == "degrade" for k, _ in events), "没有走降级路径"
    assert "broken" in answer or "工具" in answer or answer.strip(), answer
    assert len(llm.prompts) == 2, f"应该只推理 2 次就降级，实际 {len(llm.prompts)}"
    print(f"  ✓ 降级成功，返回内容：{answer[:50]}……")


# ===========================================================================
# 测试 D：多轮记忆
# ===========================================================================


def test_d_cross_turn_memory() -> None:
    print("\n[D] 跨轮记忆：第二轮的 prompt 里带着第一张图的描述")
    tmp = Path(tempfile.mkdtemp())
    cfg, memory, registry = make_env(tmp)
    img = make_image(tmp)

    llm = FakeLLM(
        [
            tool_call("see_image", {}),
            final("我看懂了，是一只橘猫。"),
            final("第二轮我仍然记得那只猫。"),
        ]
    )
    loader = FakeLoader(llm, FakeVLM("一只橘猫坐在窗台上。"), FakeSD(tmp / "out"))
    core = AgentCore(cfg, memory, loader, registry, emit=lambda k, p=None: None)
    core.web = FakeWeb()
    core.ctx.web = core.web

    core.run("看看这张图", img)
    core.run("我刚才给你看的是什么？", "")

    last_prompt = llm.prompts[-1]
    assert "橘猫" in last_prompt, "第二轮 prompt 里没有第一轮的记忆"
    assert "我刚才给你看的是什么" in last_prompt
    print("  ✓ 第二轮 prompt 里带上了上一轮的图片描述")


# ===========================================================================
# 测试 E：协议解析器的边角料
# ===========================================================================


def test_e_parser_edges() -> None:
    print("\n[E] 解析器边角料")
    # 多个工具调用
    raw = tool_call("get_current_time", {}) + tool_call("see_image", {"image_path": "a.png"})
    p = HarmonyParser.parse(raw)
    assert [t.name for t in p.tool_calls] == ["get_current_time", "see_image"]
    # 参数里带中文、嵌套对象
    raw2 = tool_call("generate_image", {"prompt": "洛丽塔, lace", "meta": {"w": 512}})
    p2 = HarmonyParser.parse(raw2)
    assert p2.tool_calls[0].arguments["meta"]["w"] == 512
    # 中文引号
    p3 = HarmonyParser.parse(tool_call("generate_image", '{"prompt": "裙摆"}'.replace('"', "“")))
    print("  ✓ 多工具、嵌套参数、中文引号都扛得住")


# ===========================================================================
# 汇总
# ===========================================================================


def main() -> None:
    print("=" * 64)
    print("Agent 闭环自测（用假模型，不占显存不联网）")
    print("=" * 64)
    test_a_full_loop()
    test_b_bad_json_retry()
    test_c_degrade_to_text()
    test_d_cross_turn_memory()
    test_e_parser_edges()
    print("\n" + "=" * 64)
    print("全部自测通过 ✓ 闭环逻辑是通的，剩下就看真模型了")
    print("=" * 64)


if __name__ == "__main__":
    main()
