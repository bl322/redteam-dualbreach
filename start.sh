#!/usr/bin/env bash
# DUALBREACH 评测系统一键启动（Linux / 服务器）
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[1/2] 创建虚拟环境并安装依赖..."
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt
else
  echo "[1/2] 已存在虚拟环境，跳过安装"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8088}"
echo "[2/2] 启动 DUALBREACH 评测系统: http://localhost:${PORT}"
exec ./.venv/bin/python -m system.server --host "$HOST" --port "$PORT"
