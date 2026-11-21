<div align="center">
  <img src="app_icon.png" width="128" alt="GPT-OSS Agent">
  <h1>GPT-OSS 专属 Agent</h1>
  <p>为 <b>GPT-OSS-20B</b> 手写的 Windows 桌面 AI 助手：会看图、会上网查资料、会画图，<br>还能直接操作你的电脑。</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue" alt="python">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="license">
    <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey" alt="platform">
    <img src="https://img.shields.io/badge/Framework-手写%20(无%20LangChain)-orange" alt="no framework">
  </p>
</div>

---

## ⚡ 一键安装（Windows，推荐）

打开 **PowerShell**（开始菜单搜 PowerShell），把下面这一整行粘进去，回车：

```powershell
irm https://github.com/Luzhen-0502/gptoss-client/releases/latest/download/install.ps1 | iex
```

它会自动做四件事：

1. 下载 `GPTOSS-Agent.exe` 到 `%USERPROFILE%\GPTOSS-Agent\`
2. 下载模型下载脚本
3. 在桌面建一个「GPT-OSS Agent」快捷方式
4. 问你要不要**现在就把 19GB 模型下好**（选 y 就一条龙搞定）

装完桌面双击就能用。

<details>
<summary>如果这条命令连不上（个别网络环境），点这里用镜像版</summary>

```powershell
irm https://gh-proxy.com/https://raw.githubusercontent.com/Luzhen-0502/gptoss-client/main/install.ps1 | iex
```

</details>

> 如果 PowerShell 提示「禁止运行脚本」，先跑这条放开当前会话：
> `Set-ExecutionPolicy -Scope Process Bypass`

> 不想跑脚本？直接下 exe 也行：
> [**GPTOSS-Agent.exe**](https://github.com/Luzhen-0502/gptoss-client/releases/latest/download/GPTOSS-Agent.exe)（305 MB）

---

## 这是什么

一个**完全手写**的桌面 Agent。不用 LangChain、不用 AutoGen、不用 LlamaIndex、不用 OpenAI Agents SDK。
调度循环、Harmony 协议解析、记忆管理、模型装卸，全部自己实现，代码全中文注释。

它不是一个"聊天窗口 + 生图按钮"，而是三个模型各司其职的完整 Agent：

```
     ┌─────────────┐   看图    ┌─────────────┐   查资料   ┌─────────────┐
     │  GPT-OSS-20B │ ────────▶ │ Qwen2.5-VL-3B│ ────────▶ │   联网检索   │
     │  （大脑）    │ ◀──────── │ （眼睛）     │           │ （百科/搜索）│
     └──────┬──────┘   中文描述  └─────────────┘           └──────┬──────┘
            │                                                    │
            │  写英文提示词                                        │ 视觉特征资料
            ▼                                                    ▼
     ┌─────────────┐                                        共享记忆池
     │ DreamShaper 8│ ◀──────────────────────────────────────────┘
     │ （画笔）     │
     └─────────────┘
```

> 核心思路：**大脑负责查资料、翻译概念，生图模型只负责画图。**
> 生图模型看不懂「洛丽塔」，那就让大脑先上网把洛丽塔的裙型、蕾丝、蝴蝶结查清楚，
> 翻成英文塞进共享记忆，再喂给 SD —— 这样它就能画对了。

---

## 功能

### 🖼 看图和画图

- 上传图片 → 视觉模型看懂 → 中文描述写进记忆 → 大脑据此改写英文提示词 → 出图 → 右侧预览
- 冷门风格（洛丽塔、蒸汽波、哥特、新艺术运动…）先联网查视觉特征，再翻译成生图模型听得懂的话

### 💻 直接操作电脑

| 工具 | 能干什么 |
|---|---|
| `run_command` | 执行命令行命令：装软件、跑脚本、查系统状态、批量处理文件 |
| `read_file` / `write_file` | 读写文本文件：写代码、改配置、写笔记 |
| `list_dir` | 看文件夹里有什么（带大小） |
| `open_path` | 用默认程序打开文件/文件夹/网址 |
| `delete_path` | 删除文件/文件夹（**每次都弹窗问你**） |
| `screenshot` | 截屏，可以接着用视觉模型看 |
| `system_info` | 查 CPU、磁盘剩余、用户目录 |

**安全设计不是摆设：**

- 删除文件、命中高危词（`format`、`diskpart`、`del /s`、`reg delete` 等）的命令，**一定弹窗让你点「是」**；
- 盘符根目录、用户主目录、Windows 目录、Program Files **直接拒绝删除**，连问都不问；
- 默认行为是「拒绝」——没有界面确认时，危险操作一律不做；
- 全部可以在 `config.json` 的 `computer` 段里关掉或调参。

### 🧠 手写的 Agent 内核

- **专属协议解析器**：专治 GPT-OSS 的 Harmony 通道格式
  `<|channel|>commentary to=functions.xxx <|constrain|>json<|message|>{...}<|call|>`
  解析失败自动重试一次，再坏就降级为纯文本回复
- **自研调度闭环**：模型输出 → 解析 → 调工具 → 结果回填记忆 → 再推理 → 直到 final
- **三模型严格单槽互斥**：内存里任何时刻只有一个模型，32GB 内存也不会爆
- **共享记忆池**：对话 / 图片描述 / 联网资料 / 任务状态，三个模型共读共写

---

## 快速开始

### 1. 下载

到 [Releases](https://github.com/Luzhen-0502/gptoss-client/releases) 下载 `GPTOSS-Agent.exe`（约 305 MB），放进一个**普通文件夹**（比如 `D:\GPTOSS-Agent\`）。

> ⚠️ 别放 `C:\Users\...\Documents` 这种被安全软件重点盯的目录，PyInstaller 解压可能被拦，
> 会报 `Could not create temporary directory!`。

### 2. 一键下载模型

把 exe 旁边的 `一键下载模型.bat` 双击跑一遍（或者从本仓库下载它）。

它会走 `hf-mirror.com` 镜像把三个模型下到 `models\`，合计约 19 GB，支持断点续传：

```
models\
├── gpt-oss-20b-mxfp4.gguf                        11.3 GB
├── Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf            1.8 GB
├── Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf        1.2 GB
└── dreamshaper-8\                                4.1 GB
```

> 不想用脚本？手动下载地址和文件名核对表见 [models/README.md](models/README.md)。

### 3. 双击运行

双击 `GPTOSS-Agent.exe`。

> **第一次启动要等 1~2 分钟**：单文件 exe 会把内部 300MB 解压到临时目录，期间没有窗口是正常的。
> 日志 `logs\app.log` 出现「程序启动」就说明起来了。
>
> 第一次点「开始生成」还要加载 20B 模型（约 13GB），再等几十秒。

---

## 用起来是什么样

**改图 / 出图：**

```
你：生成一张洛丽塔裙子的人物
  ↓
大脑：这个词我讲不准 → 调 search_knowledge("洛丽塔服饰")
  ↓
联网：蓬松衬裙 / 蕾丝荷叶边 / 蝴蝶结 / 系带胸衣 / A字及膝裙 → 写进共享记忆
  ↓
大脑：翻译成英文视觉描述 → 调 generate_image
  ↓
内存换入 DreamShaper（语言模型先卸载）→ 出图 → 右侧预览
```

**干活：**

```
你：帮我看看 D 盘还剩多少空间
  ↓
大脑：调 system_info
  ↓
返回：D 盘空闲 63.1 GB / 共 275.0 GB
```

**删除前会问你：**

```
┌────────────────────────────────────────┐
│ ⚠️ 需要你确认                            │
│ 要删除这个文件吗？                       │
│ D:\test\old.txt                         │
│ 删了不能恢复，确定吗？                   │
│              [ 是 ]   [ 否 ]            │
└────────────────────────────────────────┘
```

---

## 架构

| 文件 | 职责 |
|---|---|
| `agent_core.py` | Harmony 协议解析器 + 自研调度闭环 |
| `tools.py` | 工具注册表 + 12 个内置工具（看图/生图/联网/时间/电脑操作） |
| `memory.py` | 共享记忆池（JSON 落盘） |
| `loader.py` | 三模型单槽互斥加载/卸载 |
| `ui.py` | 中文 Tkinter 界面 |
| `config.py` | 配置 + 模型体检 + 日志 |
| `main.py` | 入口 + 全局异常兜底 |

### 工具注册表怎么扩展

```python
@registry.tool(
    "my_tool",
    "中文描述：这个工具是干什么的、什么时候该用",
    {"type": "object",
     "properties": {"x": {"type": "string", "description": "参数说明"}},
     "required": ["x"]},
    requires="none",   # none / vision / image
)
def my_tool(args, ctx):
    return "执行结果"
```

### 三模型互斥怎么保证

```python
with loader.borrow("vision") as vlm:
    desc = vlm.describe(img_path, question)
# 离开 with：视觉模型卸载，语言模型自动请回来
```

`loader.py` 里只有一个 `_current` 槽位，加载新模型前必定先卸旧的。
llama.cpp 用 mmap，卸载后页缓存还在，重新加载只要几秒。

---

## 从源码运行 / 自己打包

```bash
git clone https://github.com/Luzhen-0502/gptoss-client.git
cd gptoss-client
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```

打包成单 exe：

```bash
python -m PyInstaller --noconfirm --clean ^
    --distpath "D:\gptoss-build\dist" ^
    --workpath "D:\gptoss-build\build" ^
    gptoss-agent.spec
```

或者双击 `打包.bat`。

自测（不需要模型，用假模型验证闭环）：

```bash
python selftest.py
```

---

## 合规与许可

- **本仓库源代码**：[MIT](LICENSE)
- **不打包任何模型权重**：模型由用户从原始授权方自行下载，各自的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

| 模型 | 许可 |
|---|---|
| GPT-OSS-20B | Apache-2.0 |
| Qwen2.5-VL-3B-Instruct | Apache-2.0 |
| DreamShaper 8 | CreativeML OpenRAIL-M（含使用限制） |

三条硬承诺：

1. 不含任何破解、绕过网络限制、绕过付费墙的功能；
2. 不内置任何 API Key、账号或代理；
3. 联网检索只访问公开网页。

生成内容的合规责任在使用者。请勿用于违法、侵权、冒充或误导性用途。

---

## 已知限制

| 项目 | 说明 |
|---|---|
| 推理速度 | 纯 CPU，20B 模型约 4~8 token/s，一轮回答等 20~60 秒 |
| 生图速度 | DreamShaper 8 在 CPU 上 512×512 / 12 步约 40~90 秒（4 步约 76 秒） |
| 模型装卸 | 看图 → 出图之间会重新加载语言模型（靠页缓存压到几秒） |
| 启动 | 单 exe 每次启动解压 300MB，1~2 分钟；想快可以用 onedir 模式 |
| 联网源 | 维基/必应页面结构变化可能影响解析，做了降级但不保证永远准 |
| 没有流式输出 | 一次性出完整段，但工具调用是分步的，能看到过程 |

---

## 致谢

- [OpenAI GPT-OSS](https://github.com/openai/gpt-oss) —— 开源的语言模型
- [ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp) —— 让 GGUF 能跑在普通电脑上
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) —— 视觉理解
- [DreamShaper](https://huggingface.co/Lykon/dreamshaper-8) —— 文生图
- [Hugging Face](https://huggingface.co/) / [hf-mirror](https://hf-mirror.com/) —— 模型托管与镜像
