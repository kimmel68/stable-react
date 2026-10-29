@echo off
rem ComfyUI auf den neuesten stabilen Tag heben und Abhaengigkeiten nachziehen
setlocal
cd /d "%~dp0ComfyUI"
git fetch --tags --quiet || exit /b 1
for /f "delims=" %%t in ('git tag --sort^=-v:refname') do (set "TAG=%%t" & goto :found)
:found
echo Neuester Tag: %TAG%
git checkout --quiet %TAG% || exit /b 1
set "UV_CACHE_DIR=%~dp0.uv-cache"
uv pip install --python "%~dp0.venv\Scripts\python.exe" -r requirements.txt
git describe --tags
