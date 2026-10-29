@echo off
rem ComfyUI headless als Renderer fuer Weg A (http://127.0.0.1:8188)
rem --highvram: Modelle bleiben im VRAM. Mit dem dynamischen VRAM-Modus lagert ComfyUI aus,
rem sobald andere Programme (Unreal, Blender) VRAM halten: gemessen 0,5 statt 3,5 it/s.
rem --cache-ram 8 8: mindestens 8 GB RAM frei lassen (Standard 10 %% = 3 GB reichte nicht, PC fror am 20.10. ein)
rem Fuer grosse Modelle (Flux, Krea) stattdessen:  start_comfy.bat --lowvram
cd /d "%~dp0ComfyUI"
if "%~1"=="" (
  "%~dp0.venv\Scripts\python.exe" main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --highvram --cache-ram 8 8
) else (
  "%~dp0.venv\Scripts\python.exe" main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch %*
)
