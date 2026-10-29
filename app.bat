@echo off
rem Stable React (SD 1.5) starten: http://127.0.0.1:7860
setlocal
set "PYTHONIOENCODING=utf-8"
"%~dp0.venv\Scripts\python.exe" "%~dp0app\dmk_app.py"
