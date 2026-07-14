# ============================================================
# 启动脚本 —— 双击运行即可
# ============================================================
@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================
echo   中国42部门CGE政策模拟平台 v2.0
echo ============================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查依赖
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [安装依赖] 首次运行，正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动运行：pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [完成] 依赖安装成功
    echo.
)

echo 正在启动平台...
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 可停止服务
echo.

streamlit run dashboard/app.py --server.runOnSave true

pause
