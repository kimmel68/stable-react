@echo off
rem Aufruf:  dmk.bat bench --seq-root seq\demo --out out\bench
setlocal
set "PYTHONPATH=%~dp0deforum_mosh"
set "PYTHONUNBUFFERED=1"
"%~dp0.venv\Scripts\python.exe" -m dmk %*
