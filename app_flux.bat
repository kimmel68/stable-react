@echo off
rem Stable React · Flux Hybrid starten: http://127.0.0.1:7861
rem FLUX.1 Kontext [dev] zeichnet die Welten auf dem Beat, die dmk-Schleife (SD 1.5) den Fluss dazwischen.
rem Modelle: install.ps1 -Flux (ca. 19 GB). Lizenz der Flux-Gewichte: FLUX.1 [dev] Non-Commercial License.
setlocal
set "PYTHONIOENCODING=utf-8"
set "DMK_ENGINE=flux"
"%~dp0.venv\Scripts\python.exe" "%~dp0app\dmk_app.py"
