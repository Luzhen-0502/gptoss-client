@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==========================================================
echo   GPT-OSS 专属 Agent - 一键下载模型
echo   三个模型合计约 19 GB，放在本目录的 models\ 里
echo   走 hf-mirror.com 镜像，国内网络也能下
echo   中断了重新双击本文件会接着下
echo ==========================================================
echo.
echo 按任意键开始，或直接关窗口取消。
pause >nul

set MIRROR=https://hf-mirror.com
set MODELS=%~dp0models
if not exist "%MODELS%" mkdir "%MODELS%"

where curl >nul 2>nul
if errorlevel 1 (
    echo [错误] 系统里没有 curl，请用 Windows 10 以上版本，或手动下载。
    pause
    exit /b 1
)

rem ---------- 1. 视觉模型 ----------
echo.
echo [1/4] 视觉模型 mmproj（约 1.2 GB）
curl.exe -L -C - --retry 5 --retry-delay 3 -sS -o "%MODELS%\Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf" "%MIRROR%/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
if errorlevel 1 echo [警告] 这一步没成功，可以重跑续传。

echo [2/4] 视觉模型本体（约 1.8 GB）
curl.exe -L -C - --retry 5 --retry-delay 3 -sS -o "%MODELS%\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" "%MIRROR%/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
if errorlevel 1 echo [警告] 这一步没成功，可以重跑续传。

rem ---------- 2. 生图模型 ----------
echo.
echo [3/4] 生图模型 DreamShaper 8（约 4 GB）
set DS=%MODELS%\dreamshaper-8
if not exist "%DS%\feature_extractor" mkdir "%DS%\feature_extractor"
if not exist "%DS%\scheduler" mkdir "%DS%\scheduler"
if not exist "%DS%\text_encoder" mkdir "%DS%\text_encoder"
if not exist "%DS%\tokenizer" mkdir "%DS%\tokenizer"
if not exist "%DS%\unet" mkdir "%DS%\unet"
if not exist "%DS%\vae" mkdir "%DS%\vae"

call :get "Lykon/dreamshaper-8" "model_index.json"                            "%DS%\model_index.json"
call :get "Lykon/dreamshaper-8" "feature_extractor/preprocessor_config.json"  "%DS%\feature_extractor\preprocessor_config.json"
call :get "Lykon/dreamshaper-8" "scheduler/scheduler_config.json"             "%DS%\scheduler\scheduler_config.json"
call :get "Lykon/dreamshaper-8" "text_encoder/config.json"                    "%DS%\text_encoder\config.json"
call :get "Lykon/dreamshaper-8" "text_encoder/model.safetensors"              "%DS%\text_encoder\model.safetensors"
call :get "Lykon/dreamshaper-8" "tokenizer/merges.txt"                        "%DS%\tokenizer\merges.txt"
call :get "Lykon/dreamshaper-8" "tokenizer/special_tokens_map.json"           "%DS%\tokenizer\special_tokens_map.json"
call :get "Lykon/dreamshaper-8" "tokenizer/tokenizer_config.json"             "%DS%\tokenizer\tokenizer_config.json"
call :get "Lykon/dreamshaper-8" "tokenizer/vocab.json"                        "%DS%\tokenizer\vocab.json"
call :get "Lykon/dreamshaper-8" "unet/config.json"                            "%DS%\unet\config.json"
call :get "Lykon/dreamshaper-8" "unet/diffusion_pytorch_model.safetensors"    "%DS%\unet\diffusion_pytorch_model.safetensors"
call :get "Lykon/dreamshaper-8" "vae/config.json"                             "%DS%\vae\config.json"
call :get "Lykon/dreamshaper-8" "vae/diffusion_pytorch_model.safetensors"     "%DS%\vae\diffusion_pytorch_model.safetensors"

rem ---------- 3. 语言模型 ----------
echo.
echo [4/4] 语言模型 GPT-OSS-20B（约 11.3 GB，最慢，请耐心）
curl.exe -L -C - --retry 5 --retry-delay 3 -sS -o "%MODELS%\gpt-oss-20b-mxfp4.gguf" "%MIRROR%/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf"
if errorlevel 1 echo [警告] 这一步没成功，可以重跑续传。

echo.
echo ==========================================================
echo   下载结束。可以直接双击 GPTOSS-Agent.exe 使用了。
echo   如果某个文件没下完，重新双击本脚本会从断点继续。
echo ==========================================================
pause
exit /b 0

:get
rem 用法: call :get <repo> <仓库内路径> <本地完整路径>
if exist "%~3" (
    echo   已存在，跳过：%~nx3
    exit /b 0
)
echo   下载 %~2
curl.exe -L -C - --retry 5 --retry-delay 3 -sS -o "%~3" "%MIRROR%/%~1/resolve/main/%~2"
exit /b 0
