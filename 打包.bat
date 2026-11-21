@echo off
chcp 936 >nul
cd /d "%~dp0"
echo ==========================================
echo   正在打包 GPT-OSS 专属 Agent
echo   exe 会输出到 D:\gptoss-build\dist
echo   第一次打包比较慢（几分钟），别关窗口
echo ==========================================
echo.

python -m PyInstaller --noconfirm --clean --distpath "D:\gptoss-build\dist" --workpath "D:\gptoss-build\build" gptoss-agent.spec

if errorlevel 1 (
    echo.
    echo [失败] 打包出错了，看上面的报错信息。
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   打包完成！
echo   exe 位置：D:\gptoss-build\dist\GPTOSS-Agent.exe
echo   记得把 models 文件夹放到 exe 旁边
echo ==========================================
pause
