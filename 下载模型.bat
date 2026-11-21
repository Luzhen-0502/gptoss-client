@echo off
chcp 936 >nul
cd /d "%~dp0"
setlocal

echo ==========================================
echo   自动下载三个模型（共约 15.3 GB）
echo   走 hf-mirror.com 镜像，国内网络也能下
echo   下载慢的话别关窗口，脚本支持断点续传
echo ==========================================
echo.
echo 按任意键开始，或者直接关窗口取消。
pause >nul

set HF_ENDPOINT=https://hf-mirror.com

where huggingface-cli >nul 2>nul
if errorlevel 1 (
    echo 没找到下载工具，正在安装 huggingface_hub ...
    python -m pip install -U huggingface_hub
)

echo.
echo [1/3] 语言模型 GPT-OSS-20B（12.8 GB，最慢，可能要很久）
huggingface-cli download axteam/gpt-oss-20b-GGUF gpt-oss-20b-GGUF-MXFP4.gguf --local-dir models
if errorlevel 1 echo [警告] 这一步失败了。别急，重新双击本脚本会接着传。

echo.
echo [2/3] 视觉模型 Qwen2.5-VL-3B（本体 + mmproj）
huggingface-cli download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --include "*Q4_K_M*" "*mmproj*" --local-dir models

echo.
echo [3/3] 生图模型 DreamShaper 8（约 2 GB）
huggingface-cli download Lykon/dreamshaper-8 DreamShaper8.safetensors --local-dir models

echo.
echo ==========================================
echo   下载结束。
echo   请打开 models 文件夹核对文件名：
echo     - 如果和 config.json 里的路径对不上，直接改 config.json
echo     - 生图是单文件的话，把 image.model_path 改成那个 .safetensors
echo ==========================================
pause
