@echo off
chcp 936 >nul
set EXE=D:\GPTOSS-Agent\GPTOSS-Agent.exe

if not exist "%EXE%" (
    echo [错误] 找不到 GPTOSS-Agent.exe
    echo 它应该在 D:\GPTOSS-Agent\ 目录里。
    echo 如果你把它挪到别的地方了，请打开那边的 exe。
    pause
    exit /b 1
)

rem 正式版放在 D 盘，因为 C 盘 Documents 目录会拦 PyInstaller 的解压
start "" "%EXE%"
