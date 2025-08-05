@echo off
echo ========================================
echo    VTuber AI CustomTkinter GUI 啟動器
echo ========================================
echo.

cd /d "%~dp0"

echo 檢查 Python 環境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 未安裝或不在 PATH 中
    echo 請先安裝 Python 3.8+
    pause
    exit /b 1
)

echo 檢查所需套件...
python -c "import yaml" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安裝 PyYAML...
    pip install PyYAML
)

python -c "import colorama" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安裝 colorama...
    pip install colorama
)

echo.
echo 🚀 啟動 VTuber AI GUI...
echo.

python gui_launcher.py

echo.
echo 👋 GUI 已關閉
pause
