# dmk installer (Windows, NVIDIA GPU). Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Options:
#   -TorchIndex <url>   PyTorch wheel index for your CUDA version (default: CUDA 12.8)
#   -SkipModels         do not download the models (if you already have them)
#   -Flux               also download the models for the Flux Hybrid app (app_flux.bat, about 19 GB;
#                       FLUX.1 Kontext [dev] is under the FLUX.1 [dev] Non-Commercial License)
#   -DryRun             only print what would be done
param(
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipModels,
    [switch]$Flux,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComfyTag = "v0.39.1"          # tested ComfyUI version
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Run($exe, [string[]]$argList) {
    Write-Host "    $exe $($argList -join ' ')"
    if (-not $DryRun) {
        & $exe @argList
        if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($argList -join ' ')" }
    }
}

Step "Checking prerequisites"
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) { throw "git is required (https://git-scm.com)." }
$uv = Get-Command uv -ErrorAction SilentlyContinue
$sysPy = Get-Command python -ErrorAction SilentlyContinue
if (-not $uv -and -not $sysPy) { throw "Python 3.12 (or uv) is required (https://www.python.org or https://docs.astral.sh/uv)." }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found in PATH. Install it (e.g. 'winget install Gyan.FFmpeg') before rendering."
}

Step "Python environment (.venv)"
if (Test-Path $Py) { Write-Host "    .venv exists" }
elseif ($uv) { Run "uv" @("venv", "--python", "3.12", (Join-Path $Root ".venv")) }
else { Run "python" @("-m", "venv", (Join-Path $Root ".venv")) }
if ($uv) { $Pip = @("uv", "pip", "install", "--python", $Py) } else { $Pip = @($Py, "-m", "pip", "install") }
function PipInstall([string[]]$pkgs) { Run $Pip[0] (@($Pip[1..($Pip.Length - 1)]) + $pkgs) }

Step "PyTorch with CUDA ($TorchIndex)"
PipInstall @("torch", "torchvision", "--index-url", $TorchIndex)

Step "ComfyUI $ComfyTag"
$Comfy = Join-Path $Root "ComfyUI"
if (Test-Path (Join-Path $Comfy "main.py")) { Write-Host "    ComfyUI folder exists (make sure it is $ComfyTag)" }
else { Run "git" @("clone", "--depth", "1", "--branch", $ComfyTag, "https://github.com/comfyanonymous/ComfyUI.git", $Comfy) }
PipInstall @("-r", (Join-Path $Comfy "requirements.txt"))

Step "dmk requirements"
PipInstall @("-r", (Join-Path $Root "requirements.txt"))

Step "dmk ComfyUI nodes (comfyui_dmk, GPL-3.0)"
$Nodes = Join-Path $Comfy "custom_nodes\comfyui_dmk"
Write-Host "    copy comfyui_dmk -> $Nodes"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force $Nodes | Out-Null
    Copy-Item (Join-Path $Root "comfyui_dmk\*") $Nodes -Recurse -Force   # incl. example_workflows\
    # the example workflow loads the example track from ComfyUI\input
    New-Item -ItemType Directory -Force (Join-Path $Comfy "input") | Out-Null
    Copy-Item (Join-Path $Root "examples\signal_decay_20s.m4a") (Join-Path $Comfy "input") -Force
}

if (-not $SkipModels) {
    Step "Models (open weights, from Hugging Face)"
    # Up to 4 attempts per model: twice with huggingface_hub, then with curl.exe (built into Windows 10/11, resumes partial
    # files). Clean-clone test 22.-25.10.: the hub's Xet transfer failed sporadically ("CAS Client Error"), and its
    # HF_HUB_DISABLE_XET fallback could not even locate the file, while curl on the same URL worked every time.
    $dl = @"
import os, shutil, subprocess, sys, time
root, flux = sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "flux"
models = [
    ("Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", "checkpoints"),
    ("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1p_sd15_depth_fp16.safetensors", "controlnet"),
]
if flux:   # Flux Hybrid: FLUX.1 Kontext [dev] fp8 (Comfy-Org repackaged), text encoders, VAE - core ComfyUI nodes only
    models += [
        ("Comfy-Org/flux1-kontext-dev_ComfyUI", "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", "diffusion_models"),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "text_encoders"),
        ("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors", "vae"),
    ]
code = ("import os, sys; from huggingface_hub import hf_hub_download; "
        "p = hf_hub_download(sys.argv[1], sys.argv[2], local_dir=os.path.join(sys.argv[3], '.dl')); "
        "os.replace(p, os.path.join(sys.argv[3], os.path.basename(sys.argv[2])))")
curl = shutil.which("curl.exe") or shutil.which("curl")
for repo, remote, sub in models:
    fname = os.path.basename(remote)
    ziel = os.path.join(root, "models", sub)
    if os.path.exists(os.path.join(ziel, fname)):
        print("    exists:", fname); continue
    os.makedirs(ziel, exist_ok=True)
    for attempt in range(4):
        mit_curl = attempt >= 2 and curl
        print("    downloading", repo, fname, "(attempt %d%s)" % (attempt + 1, ", curl" if mit_curl else ""), flush=True)
        if mit_curl:
            teil = os.path.join(ziel, fname + ".part")
            url = "https://huggingface.co/%s/resolve/main/%s" % (repo, remote)
            if subprocess.run([curl, "-L", "--fail", "--retry", "3", "-C", "-", "-o", teil, url]).returncode == 0:
                os.replace(teil, os.path.join(ziel, fname))
                break
        elif subprocess.run([sys.executable, "-c", code, repo, remote, ziel]).returncode == 0:
            break
        time.sleep(5)
    else:
        sys.exit("download failed: %s %s - check the internet connection and run install.ps1 again" % (repo, fname))
print("    Depth-Anything-V2-Small and RAFT download automatically on first use.")
"@
    $tmp = Join-Path $env:TEMP "dmk_models.py"
    Set-Content -Path $tmp -Value $dl -Encoding utf8
    if ($Flux) { Run $Py @($tmp, $Comfy, "flux") } else { Run $Py @($tmp, $Comfy) }
}

Step "Done"
Write-Host "    Start the app with app.bat and open http://127.0.0.1:7860, then press 'Start services'."
Write-Host "    Flux Hybrid (after install.ps1 -Flux): app_flux.bat, http://127.0.0.1:7861"
