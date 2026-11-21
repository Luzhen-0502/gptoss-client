# 第三方许可与模型声明（THIRD-PARTY NOTICES）

本项目（GPT-OSS 专属 Agent）的**源代码**以 MIT 许可证发布，见 [LICENSE](LICENSE)。

本项目**不在仓库或安装包里打包任何模型权重**。程序首次运行前，由用户通过
`一键下载模型.bat`（或手动）从模型原始发布方下载。这样做有两个原因：
一是模型合计约 19 GB，二是让每个模型都直接从它的原始授权方获取，许可关系最清楚。

下面列出运行本程序需要用到的模型和开源库，以及它们各自的许可证。

---

## 一、模型

### 1. GPT-OSS-20B（语言模型 / Agent 大脑）

| 项目 | 内容 |
|---|---|
| 提供方 | OpenAI |
| 原始仓库 | https://huggingface.co/openai/gpt-oss-20b |
| 本项目使用的 GGUF | `ggml-org/gpt-oss-20b-GGUF`（llama.cpp 官方转换）的 `gpt-oss-20b-MXFP4.gguf` |
| 许可证 | **Apache License 2.0** |
| 许可文本 | https://www.apache.org/licenses/LICENSE-2.0 |
| 说明 | Apache-2.0 允许商用、修改、再分发，需保留版权与许可声明。本项目仅通过脚本让用户自行下载，不再分发权重。 |

### 2. Qwen2.5-VL-3B-Instruct（视觉理解 / Agent 的眼睛）

| 项目 | 内容 |
|---|---|
| 提供方 | 阿里云通义千问团队 |
| 原始仓库 | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| 本项目使用的 GGUF | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`（Q4_K_M + mmproj-f16） |
| 许可证 | **Apache License 2.0** |
| 许可文本 | https://www.apache.org/licenses/LICENSE-2.0 |
| 说明 | 同上，仅引导用户从原始仓库下载。 |

### 3. DreamShaper 8（文生图 / Agent 的画笔）

| 项目 | 内容 |
|---|---|
| 提供方 | Lykon |
| 原始仓库 | https://huggingface.co/Lykon/dreamshaper-8 |
| 许可证 | **CreativeML OpenRAIL-M** |
| 许可文本 | https://huggingface.co/spaces/CompVis/stable-diffusion-license |
| 说明 | OpenRAIL-M 允许使用与再分发，但附带**使用限制**（不得用于违法、伤害、欺骗等用途），并要求把许可传递给下游使用者。本项目不打包权重，用户自行下载时即接受该许可；若你要再分发微调版或打包版，请务必附上完整许可文本。 |

> ⚠️ 生成内容的合规责任在使用者。请勿用本工具生成违法、侵权、冒充他人或
> 误导性内容；不要把生成的图片当作真实照片传播。

---

## 二、开源依赖库

| 库 | 用途 | 许可证 |
|---|---|---|
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | llama.cpp 的 Python 绑定（跑 GGUF） | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 底层推理引擎 | MIT |
| [PyTorch](https://pytorch.org/) | 深度学习运行时 | BSD-3-Clause |
| [diffusers](https://github.com/huggingface/diffusers) | 文生图管线 | Apache-2.0 |
| [transformers](https://github.com/huggingface/transformers) | 模型加载工具 | Apache-2.0 |
| [accelerate](https://github.com/huggingface/accelerate) | 推理加速辅助 | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | 权重格式 | Apache-2.0 |
| [Pillow](https://python-pillow.org/) | 图片处理与界面预览 | HPND |
| [NumPy](https://numpy.org/) | 数值计算 | BSD-3-Clause |
| [PyInstaller](https://pyinstaller.org/) | 打包成 exe | GPL-2.0 with exception（打包产物不受 GPL 约束） |

以上依赖均通过 pip 安装，各自的许可证随包分发，未做修改。

---

## 三、本项目的合规承诺

1. **不包含任何破解、绕过网络限制、绕过付费墙的功能**；
2. **不内置任何 API Key、账号或代理**；
3. 联网检索功能只访问公开网页（维基百科、必应搜索）的公开内容；
4. 模型权重不随项目分发，用户从原始授权方获取；
5. 代码开源（MIT），任何人均可审计。

如果你认为本项目的某处内容侵犯了你的权利，请提 Issue，我们会立即处理。
