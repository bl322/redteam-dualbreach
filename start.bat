@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] 创建虚拟环境并安装依赖...
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
  echo [1/2] 已存在虚拟环境，跳过安装
)

echo [2/2] 启动 DUALBREACH 评测系统: http://localhost:8088
.venv\Scripts\python.exe -m system.server --host 0.0.0.0 --port 8088
pause
