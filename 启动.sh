#!/bin/bash
# ============================================================
# Linux/macOS 启动脚本
# ============================================================
cd "$(dirname "$0")"

echo "============================================"
echo "  中国42部门CGE政策模拟平台 v2.0"
echo "============================================"
echo

# 检查 Python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "[错误] 未找到 Python，请先安装 Python 3.9+"
    exit 1
fi

# 检查依赖
$PY -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[安装依赖] 首次运行，正在安装依赖包..."
    $PY -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败，请手动运行：pip install -r requirements.txt"
        exit 1
    fi
    echo "[完成] 依赖安装成功"
    echo
fi

echo "正在启动平台..."
echo "浏览器将自动打开 http://localhost:8501"
echo "按 Ctrl+C 可停止服务"
echo

streamlit run dashboard/app.py --server.runOnSave true
