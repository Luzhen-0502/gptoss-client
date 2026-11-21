# -*- coding: utf-8 -*-
"""
memory.py —— 共享记忆池（大肥鱼手写）

这是三模型共用的「一份脑子」：
  - 语言模型：读写对话历史、读图片描述、读任务状态
  - 视觉模型：把图片的中文描述写进来
  - 生图模型：把生成结果（提示词、输出路径、参数）写进来

存储：一个 JSON 文件（默认 data/memory.json），人可读、可备份、可删除。
线程安全：UI 线程和推理线程都会碰它，所以内部用 RLock 保护，写盘用「临时文件+替换」防写坏。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

# 单条消息/描述在注入 prompt 时的最大保留字数，防止记忆太长把上下文撑爆
ENTRY_CLIP = 400
# 记忆里最多保留多少条消息（再老的就裁掉，避免文件无限膨胀）
MAX_MESSAGES = 400
MAX_IMAGES = 200
MAX_KNOWLEDGE = 60


def _now() -> float:
    return time.time()


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _clip(text: str, n: int = ENTRY_CLIP) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "……（已截断）"


class SharedMemory:
    """共享记忆池：对话 + 图片描述 + 任务状态，三模块共用一份。"""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._dirty = False
        self._data: dict[str, Any] = {
            "version": 1,
            "created_at": _now(),
            "messages": [],   # 对话历史（user / assistant / tool / system）
            "images": [],     # 图片记录（input 原图 / output 生成图，含中文描述）
            "knowledge": [],  # 联网查到的参考资料（小众词条的视觉特征描述）
            "task": {         # 当前任务状态
                "goal": "",
                "stage": "idle",       # idle / thinking / tool / done / error
                "pending_image": "",   # 待处理的图片路径
                "last_error": "",
                "turn": 0,
            },
            "stats": {"visions": 0, "generations": 0, "turns": 0},
        }
        self.load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load(self) -> None:
        """从磁盘读取记忆；文件不存在/损坏都不报错，直接从头开始。"""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with self._lock:
                    base = self._data
                    for key in ("messages", "images", "knowledge"):
                        if isinstance(data.get(key), list):
                            base[key] = data[key]
                    if isinstance(data.get("task"), dict):
                        base["task"].update(data["task"])
                    if isinstance(data.get("stats"), dict):
                        base["stats"].update(data["stats"])
                    base["version"] = data.get("version", 1)
                    base["created_at"] = data.get("created_at", base["created_at"])
        except Exception:
            # 记忆文件坏了就当新的用，但把坏文件留个备份方便排查
            try:
                self.path.rename(self.path.with_suffix(".broken.json"))
            except Exception:
                pass

    def save(self) -> None:
        """写盘（原子替换，防止写一半断电把文件弄坏）。"""
        with self._lock:
            data = json.loads(json.dumps(self._data, ensure_ascii=False))  # 拷贝一份
            messages = data.get("messages", [])[-MAX_MESSAGES:]
            images = data.get("images", [])[-MAX_IMAGES:]
            knowledge = data.get("knowledge", [])[-MAX_KNOWLEDGE:]
            data["messages"] = messages
            data["images"] = images
            data["knowledge"] = knowledge
            tmp = self.path.with_suffix(".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
                self._dirty = False
            except Exception:
                # 写盘失败不能影响对话，内存里的记忆照常可用
                pass

    def _mark_dirty(self, autosave: bool) -> None:
        self._dirty = True
        if autosave:
            self.save()

    # ------------------------------------------------------------------
    # 对话历史
    # ------------------------------------------------------------------
    def add_message(
        self,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
        autosave: bool = True,
    ) -> dict[str, Any]:
        """写入一条对话。role 用 user / assistant / tool / system。"""
        item = {
            "id": uuid.uuid4().hex[:12],
            "role": role,
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            "ts": _now(),
            "meta": meta or {},
        }
        with self._lock:
            self._data["messages"].append(item)
            self._mark_dirty(autosave)
        return item

    def add_user(self, text: str, image_path: str = "") -> dict[str, Any]:
        return self.add_message("user", text, meta={"image_path": image_path} if image_path else None)

    def add_assistant(self, text: str, *, kind: str = "final") -> dict[str, Any]:
        return self.add_message("assistant", text, meta={"kind": kind})

    def add_tool_result(self, tool_name: str, result: str, *, ok: bool = True) -> dict[str, Any]:
        """工具执行结果也进对话历史，语言模型下一轮就能看到。"""
        return self.add_message("tool", result, meta={"tool": tool_name, "ok": ok})

    def history(self, limit: int = 8) -> list[dict[str, Any]]:
        """取最近 limit 条对话（不含 system）。"""
        with self._lock:
            msgs = [m for m in self._data["messages"] if m.get("role") != "system"]
            return [dict(m) for m in msgs[-limit:]]

    def clear_conversation(self) -> None:
        """清空对话（图片记录和记忆摘要保留）。"""
        with self._lock:
            self._data["messages"] = []
            self._data["task"].update({"goal": "", "stage": "idle", "turn": 0})
            self.save()

    # ------------------------------------------------------------------
    # 图片 / 视觉描述
    # ------------------------------------------------------------------
    def add_image_description(
        self,
        image_path: str,
        description: str,
        *,
        question: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """视觉模块专用：把图片的中文描述写进共享记忆。"""
        item = {
            "id": uuid.uuid4().hex[:12],
            "kind": "input",
            "path": str(image_path),
            "description": (description or "").strip(),
            "question": question,
            "model": model,
            "ts": _now(),
        }
        with self._lock:
            # 同一张图重复描述时更新旧的，不堆垃圾
            for old in self._data["images"]:
                if old.get("kind") == "input" and old.get("path") == item["path"]:
                    old.update(item)
                    break
            else:
                self._data["images"].append(item)
            self._data["stats"]["visions"] = int(self._data["stats"].get("visions", 0)) + 1
            self._mark_dirty(True)
        return item

    def add_generated_image(
        self,
        image_path: str,
        prompt: str,
        *,
        params: dict[str, Any] | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        """生图模块专用：把出图结果（英文提示词、路径、参数）写进共享记忆。"""
        item = {
            "id": uuid.uuid4().hex[:12],
            "kind": "output",
            "path": str(image_path),
            "prompt": prompt,
            "description": f"根据提示词生成的图片：{_clip(prompt, 160)}",
            "params": params or {},
            "model": model,
            "ts": _now(),
        }
        with self._lock:
            self._data["images"].append(item)
            self._data["stats"]["generations"] = int(self._data["stats"].get("generations", 0)) + 1
            self._mark_dirty(True)
        return item

    def images(self, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = [dict(i) for i in self._data["images"]]
        if kind:
            items = [i for i in items if i.get("kind") == kind]
        return items[-limit:]

    def latest_input_image(self) -> dict[str, Any] | None:
        """最近一张用户上传的图（含描述）。"""
        with self._lock:
            for item in reversed(self._data["images"]):
                if item.get("kind") == "input":
                    return dict(item)
        return None

    def latest_generated(self) -> dict[str, Any] | None:
        """最近一张生成出来的图。"""
        with self._lock:
            for item in reversed(self._data["images"]):
                if item.get("kind") == "output":
                    return dict(item)
        return None

    def latest_description(self) -> str:
        """最近一张已描述图片的中文描述（语言模型改图时直接读它）。"""
        inp = self.latest_input_image()
        if inp and inp.get("description"):
            return str(inp["description"])
        with self._lock:
            for item in reversed(self._data["images"]):
                if item.get("kind") == "output":
                    return str(item.get("description", ""))
        return ""

    def image_by_path(self, path: str) -> dict[str, Any] | None:
        target = str(Path(path).resolve()).lower()
        with self._lock:
            for item in self._data["images"]:
                try:
                    if str(Path(item.get("path", "")).resolve()).lower() == target:
                        return dict(item)
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # 联网参考资料（解决「生图模型看不懂小众词汇」的核心：先查再画）
    # ------------------------------------------------------------------
    def add_knowledge(
        self,
        query: str,
        summary: str,
        *,
        sources: Iterable[dict[str, str]] | None = None,
        lang: str = "zh",
        ok: bool = True,
    ) -> dict[str, Any]:
        """联网检索模块专用：把查到的资料写进共享记忆。

        语言模型下一轮生成提示词时会从记忆摘要里读到它，
        生图模型本身不需要理解「洛丽塔」，它只吃翻译好的英文视觉描述。
        """
        item = {
            "id": uuid.uuid4().hex[:12],
            "query": (query or "").strip(),
            "summary": (summary or "").strip(),
            "sources": list(sources or []),
            "lang": lang,
            "ok": bool(ok),
            "ts": _now(),
        }
        with self._lock:
            # 同一个查询词重复查时覆盖旧的，节约上下文
            key = item["query"].lower()
            for old in self._data["knowledge"]:
                if str(old.get("query", "")).lower() == key and bool(old.get("ok", True)) == item["ok"]:
                    old.update(item)
                    break
            else:
                self._data["knowledge"].append(item)
            self._mark_dirty(True)
        return item

    def knowledge_items(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(i) for i in self._data["knowledge"][-limit:]]

    def latest_knowledge(self) -> dict[str, Any] | None:
        with self._lock:
            for item in reversed(self._data["knowledge"]):
                if item.get("ok", True):
                    return dict(item)
        return None

    def render_knowledge_block(self, max_chars: int = 900) -> str:
        """把最近几条联网资料压成文本，给语言模型改写成英文视觉描述用。"""
        items = [i for i in self.knowledge_items(6) if i.get("ok", True) and i.get("summary")]
        if not items:
            return ""
        lines = ["【联网查到的参考资料（请据此把概念翻译成英文视觉描述）】"]
        used = 0
        for it in items:
            piece = f"- 「{it['query']}」：{_clip(it['summary'], 300)}"
            if used + len(piece) > max_chars:
                break
            lines.append(piece)
            used += len(piece)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 任务状态
    # ------------------------------------------------------------------
    def get_task(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data["task"])

    def set_task(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self._data["task"].update(kwargs)
            self._mark_dirty(True)
            return dict(self._data["task"])

    def begin_turn(self, goal: str, pending_image: str = "") -> int:
        """开始新一轮任务：轮次 +1，记录目标与待处理图片。"""
        with self._lock:
            task = self._data["task"]
            task["turn"] = int(task.get("turn", 0)) + 1
            task["goal"] = goal
            task["stage"] = "thinking"
            task["pending_image"] = pending_image
            task["last_error"] = ""
            self._data["stats"]["turns"] = int(self._data["stats"].get("turns", 0)) + 1
            self._mark_dirty(True)
            return task["turn"]

    def finish_turn(self, ok: bool = True, error: str = "") -> None:
        with self._lock:
            self._data["task"]["stage"] = "done" if ok else "error"
            self._data["task"]["last_error"] = error
            self._mark_dirty(True)

    # ------------------------------------------------------------------
    # 注入 prompt / UI 用
    # ------------------------------------------------------------------
    def render_memory_block(self, max_chars: int = 1800) -> str:
        """把记忆压成一段中文文本，给语言模型当「记忆摘要」用。

        包含：当前任务目标、最近图片的中文描述、最近一次生成提示词。
        """
        lines: list[str] = []
        task = self.get_task()
        if task.get("goal"):
            lines.append(f"【当前任务】{_clip(task['goal'], 200)}")
        if task.get("stage") and task["stage"] != "idle":
            lines.append(f"【任务阶段】{task['stage']}")

        inp = self.latest_input_image()
        if inp and inp.get("description"):
            lines.append(f"【最近上传图片的中文描述（文件 {Path(inp['path']).name}）】")
            lines.append(_clip(inp["description"], 600))

        gen = self.latest_generated()
        if gen and gen.get("prompt"):
            lines.append("【最近一次生成图片用的英文提示词】")
            lines.append(_clip(gen["prompt"], 300))

        # 联网查到的参考资料 —— 这是「生图模型看不懂小众词」的解药：
        # 大脑先把「洛丽塔」查成具体的裙型/蕾丝/蝴蝶结/层叠裙摆，再翻译成英文喂给 SD。
        knowledge_block = self.render_knowledge_block()
        if knowledge_block:
            lines.append(knowledge_block)

        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars] + "……（记忆过长已截断）"
        return out

    def render_history_block(self, limit: int = 8) -> str:
        """把最近几轮对话压成纯文本块（给 prompt 用，角色用中文标注）。"""
        role_names = {"user": "用户", "assistant": "助手", "tool": "工具结果"}
        lines = []
        for m in self.history(limit):
            role = role_names.get(m.get("role", ""), m.get("role", ""))
            content = _clip(m.get("content", ""), 300)
            if m.get("role") == "tool":
                tool = m.get("meta", {}).get("tool", "")
                role = f"工具结果({tool})" if tool else role
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """给 UI 的全量快照（拷一份，避免跨线程直接读内部结构）。"""
        with self._lock:
            return json.loads(json.dumps(self._data, ensure_ascii=False))

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {k: int(v) for k, v in self._data.get("stats", {}).items()}

    def reset_all(self) -> None:
        """彻底清空（UI 里用不到也行，留着做调试）。"""
        with self._lock:
            self._data["messages"] = []
            self._data["images"] = []
            self._data["knowledge"] = []
            self._data["task"] = {
                "goal": "",
                "stage": "idle",
                "pending_image": "",
                "last_error": "",
                "turn": 0,
            }
            self._data["stats"] = {"visions": 0, "generations": 0, "turns": 0}
            self.save()


# ---------------------------------------------------------------------------
# 自测：python memory.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mem.json"
    mem = SharedMemory(tmp)

    mem.add_user("帮我把这张图改成赛博朋克风格", "C:/x/a.png")
    mem.add_image_description("C:/x/a.png", "一只橘猫坐在窗台上，背景是白天的城市。", question="描述这张图")
    mem.add_generated_image("C:/x/out.png", "a cyberpunk orange cat on a windowsill, neon lights", params={"steps": 12})
    mem.begin_turn("改图")
    mem.add_tool_result("see_image", "一只橘猫坐在窗台上……")
    mem.add_assistant("已经帮你生成好了。")

    print("=== 记忆摘要 ===")
    print(mem.render_memory_block())
    print("\n=== 对话历史 ===")
    print(mem.render_history_block())
    print("\n=== 统计 ===", mem.stats())

    # 重新加载，验证落盘
    mem2 = SharedMemory(tmp)
    assert mem2.latest_description() == "一只橘猫坐在窗台上，背景是白天的城市。"
    assert mem2.latest_generated()["path"] == "C:/x/out.png"
    print("\n落盘/重载测试通过 ✓")
