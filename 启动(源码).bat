@echo off
chcp 936 >nul
cd /d "%~dp0"
rem 不打包直接跑源码。用 pythonw 不弹黑框，出错信息会写进 logs\app.log
start "" pythonw main.py
