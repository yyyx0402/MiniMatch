@echo off
cd /d "%~dp0"
if not defined MINIMATCH_PYTHON (
  if exist "runtime\python.path" set /p MINIMATCH_PYTHON=<"runtime\python.path"
)
if not defined MINIMATCH_PYTHON set "MINIMATCH_PYTHON=python"
if not exist "models\application\manifest.json" (
  "%MINIMATCH_PYTHON%" scripts\run_windows.py scripts\prepare_application.py
  if errorlevel 1 (
    echo Search artifacts are not ready. See README.md for first-time setup.
    pause
    exit /b 1
  )
)
echo MiniMatch: http://127.0.0.1:8899
"%MINIMATCH_PYTHON%" scripts\run_windows.py -m productsearch
pause
