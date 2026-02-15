@echo off
echo.
echo  ==============================
echo   MirrorMetrics - Starting...
echo  ==============================
echo.

call "%~dp0venv\Scripts\activate.bat"
python "%~dp0mirror_metrics.py"

echo.
pause
