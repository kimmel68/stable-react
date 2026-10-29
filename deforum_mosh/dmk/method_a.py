"""dmk.method_a: ComfyUI-API-Client + Deforum-Feedback-Loop.

    init_n = blend(warp(out_{n-1}), input_n, alpha)
    out_n  = sampler(init_n, denoise)        (--dry-run: out_n = init_n)
"""
from __future__ import annotations

import copy
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import core

DEFAULT_WORKFLOW = Path(__file__).parent / "workflows" / "sd15_img2img.json"
DEPTH_WORKFLOW = Path(__file__).parent / "workflows" / "sd15_depth_cn.json"
_CN_PREFER = {"tile": ("tile",),
              "depth": ("control_v11f1p_sd15_depth", "sd15_depth", "depth"),
              "canny": ("control_v11p_sd15_canny", "sd15_canny", "canny")}


def add_rausch_transport(wf: dict, sampler: str = "dpmpp_2m", scheduler: str = "karras") -> dict:
    """KSampler durch SamplerCustomAdvanced mit von außen eingespeistem Rauschen ersetzen. Das Rauschen ε ~ N(0,1)
    kommt als .latent (LoadLatent, Titel "rausch") und wird über VOIDWarpedNoiseSource (Kern-Knoten, wandelt ein
    LATENT in eine NOISE-Quelle) eingespeist; ComfyUI skaliert es dann selbst korrekt mit σ₀ im skalierten Latentraum.
    (Eigenes Vorverrauschen per LatentAdd scheitert an der Skalierung: Latent ×0.18215 passiert erst danach.)"""
    wf = copy.deepcopy(wf)
    ks_id = next(k for k, v in wf.items() if v.get("class_type") in ("KSampler", "KSamplerAdvanced"))
    ks = wf.pop(ks_id)["inputs"]
    wf["r1"] = {"class_type": "LoadLatent", "inputs": {"latent": ""}, "_meta": {"title": "rausch"}}
    wf["r2"] = {"class_type": "VOIDWarpedNoiseSource", "inputs": {"warped_noise": ["r1", 0]}}
    wf["r4"] = {"class_type": "BasicScheduler", "inputs": {"model": ks["model"], "scheduler": scheduler,
                                                          "steps": ks.get("steps", 20), "denoise": ks.get("denoise", 1.0)}}
    wf["r5"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    wf["r6"] = {"class_type": "CFGGuider", "inputs": {"model": ks["model"], "positive": ks["positive"],
                                                     "negative": ks["negative"], "cfg": ks.get("cfg", 6.0)}}
    wf["r8"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["r2", 0], "guider": ["r6", 0],
                                                                 "sampler": ["r5", 0], "sigmas": ["r4", 0],
                                                                 "latent_image": ks["latent_image"]}}
    for v in wf.values():  # alles, was am KSampler hing (VAEDecode), an den neuen Sampler
        for key, link in list(v["inputs"].items()):
            if isinstance(link, list) and str(link[0]) == str(ks_id):
                v["inputs"][key] = ["r8", 0]
    return wf


REF_LADEN = "dmk_ref_laden"


def add_lora(wf: dict, lora_name: str, staerke: float = 1.0) -> dict:
    """LoRA (z. B. Hyper-SD15 für wenige, ruhige Schritte) über den Kern-Knoten LoraLoaderModelOnly vor alle Modell-Nutzer
    setzen."""
    wf = copy.deepcopy(wf)
    ck = next(k for k, v in wf.items() if v.get("class_type") == "CheckpointLoaderSimple")
    for v in wf.values():
        for key, val in list(v.get("inputs", {}).items()):
            if val == [ck, 0]:
                v["inputs"][key] = ["dmk_lora", 0]
    wf["dmk_lora"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": [ck, 0], "lora_name": lora_name, "strength_model": float(staerke)}}
    return wf


def add_referenz(wf: dict, bloecke: str = "all", sigma_min: float = 0.0) -> dict:
    """Workflow um die Referenz-Aufmerksamkeit erweitern (Node-Paket comfyui_dmk, Weg 2 "Ruhe von innen"):
    LoadImage(Referenz) -> VAEEncode -> DMKReferenceAttention(model, latent, reference) -> KSampler; nach dem Decode wird
    mit dem Kern-Knoten ImageFromBatch das Ziel (Index 1) ausgewählt. Die Referenz ist das mitbewegte vorige Ergebnis."""
    wf = copy.deepcopy(wf)
    ks = next(k for k, v in wf.items() if v.get("class_type") in ("KSampler", "SamplerCustomAdvanced"))
    ck = next(k for k, v in wf.items() if v.get("class_type") == "CheckpointLoaderSimple")
    dec = next(k for k, v in wf.items() if v.get("class_type") == "VAEDecode")
    vor = [k for k, v in wf.items() if v.get("class_type") in ("PreviewImage", "SaveImage")]
    wf[REF_LADEN] = {"class_type": "LoadImage", "inputs": {"image": ""}, "_meta": {"title": "referenz"}}
    wf["dmk_ref_enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": [REF_LADEN, 0], "vae": [ck, 2]}}
    lat = wf[ks]["inputs"].get("latent_image")
    mdl = wf[ks]["inputs"].get("model")
    wf["dmk_ref_attn"] = {"class_type": "DMKReferenceAttention", "inputs": {
        "model": mdl, "latent": lat, "reference": ["dmk_ref_enc", 0], "blocks": bloecke, "sigma_min": sigma_min}}
    wf[ks]["inputs"]["model"] = ["dmk_ref_attn", 0]
    wf[ks]["inputs"]["latent_image"] = ["dmk_ref_attn", 1]
    wf["dmk_ref_pick"] = {"class_type": "ImageFromBatch", "inputs": {"image": [dec, 0], "batch_index": 1, "length": 1}}
    for k in vor:
        wf[k]["inputs"]["images"] = ["dmk_ref_pick", 0]
    return wf


def add_wandelkarte(wf: dict) -> dict:
    """Workflow um eine Wandelkarte erweitern: pro Pixel unterschiedlich starkes Umzeichnen
    (DifferentialDiffusion + Latent-Rauschmaske). Weiß = voller denoise, Schwarz = Bereich bleibt erhalten.
    Der neue LoadImage-Knoten trägt den Titel "wandel" und wird von patch_workflow befüllt."""
    wf = copy.deepcopy(wf)
    ks_id = next(k for k, v in wf.items() if v.get("class_type") in ("KSampler", "KSamplerAdvanced"))
    ks = wf[ks_id]["inputs"]
    wf["w1"] = {"class_type": "LoadImage", "inputs": {"image": ""}, "_meta": {"title": "wandel"}}
    wf["w2"] = {"class_type": "ImageToMask", "inputs": {"image": ["w1", 0], "channel": "red"}}
    wf["w3"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ks["latent_image"], "mask": ["w2", 0]}}
    wf["w4"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ks["model"]}}
    ks["latent_image"], ks["model"] = ["w3", 0], ["w4", 0]
    return wf


def add_controlnet2(wf: dict, control_net_name: str, strength: float, end: float = 1.0) -> dict:
    """Zweites ControlNet hinter das erste hängen (z. B. Kanten zusätzlich zur Tiefe). Der neue LoadImage-
    Knoten trägt den Titel "control2"; Lader und Apply sind ebenfalls so betitelt, damit patch_workflow die
    Werte des ersten ControlNets nicht darüberschreibt."""
    wf = copy.deepcopy(wf)
    ks = next(v for v in wf.values() if v.get("class_type") in ("KSampler", "KSamplerAdvanced"))["inputs"]
    vae = next((v["inputs"]["vae"] for v in wf.values() if v.get("class_type") == "ControlNetApplyAdvanced"
                and "vae" in v["inputs"]), None)
    t = {"_meta": {"title": "control2"}}
    wf["c2a"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": control_net_name}, **t}
    wf["c2b"] = {"class_type": "LoadImage", "inputs": {"image": ""}, **t}
    inputs = {"positive": ks["positive"], "negative": ks["negative"], "control_net": ["c2a", 0],
              "image": ["c2b", 0], "strength": strength, "start_percent": 0.0, "end_percent": end}
    if vae is not None:
        inputs["vae"] = vae
    wf["c2c"] = {"class_type": "ControlNetApplyAdvanced", "inputs": inputs, **t}
    ks["positive"], ks["negative"] = ["c2c", 0], ["c2c", 1]
    return wf


def wandel_bild(depth: np.ndarray, mode: str, floor: float) -> np.ndarray:
    """Wandelkarte aus der Tiefe (hell = nah). fern: die Ferne wandelt sich, das Nahe bleibt erhalten;
    nah: umgekehrt. floor = Mindestanteil Wandel überall, damit nichts einfriert."""
    x = 1.0 - depth if mode == "fern" else depth
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 3)
    m = np.clip(floor + (1.0 - floor) * x, 0, 1)
    return cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def parse_prompt_schedule(text: str) -> list[tuple[int, str]]:
    """Deforum-artiger Prompt-Fahrplan: "0: a man in a tuxedo | 60: a tiger mask | 120: a space station".
    Ohne Keyframe-Präfix gilt der Text für alle Frames. Harte Wechsel am Keyframe; der Feedback-Loop
    blendet sie weich, wie im Original."""
    parts = [p.strip() for p in str(text).split("|") if p.strip()]
    keys = []
    for p in parts:
        m = re.match(r"^(\d+)\s*:\s*(.+)$", p, re.S)
        keys.append((int(m.group(1)), m.group(2).strip()) if m else (0, p))
    keys.sort(key=lambda k: k[0])
    return keys or [(0, "")]


def prompt_at(keys: list[tuple[int, str]], i: int) -> str:
    cur = keys[0][1]
    for f, t in keys:
        if f <= i:
            cur = t
    return cur
DEFAULT_PROMPT = "highly detailed, sharp focus, cinematic light"
DEFAULT_NEGATIVE = "blurry, lowres, jpeg artifacts, text, watermark"
# Reihenfolge = Vorliebe, wenn --ckpt fehlt (SD 1.5 passt bequem in 8 GB)
_CKPT_PREFER = ("dreamshaper_8", "juggernaut_reborn", "epicrealism", "photon")


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, server: str = "127.0.0.1:8188"):
        self.base = f"http://{server}"
        self.client_id = uuid.uuid4().hex
        self.tag = self.client_id[:8]  # Uploads pro Prozess eindeutig: parallele Läufe überschreiben sich sonst

    def _req(self, path, data=None, headers=None, timeout=30) -> bytes:
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def uploads_aufraeumen(self, input_dir=None) -> int:
        """Eigene Uploads dieses Laufs aus ComfyUIs input-Ordner löschen (Muster <tag>_dmk*). Im SDK-Modus legt der Proxy
        jedes Bild als neue Datei an ("<tag>_dmk_init (N).png"); ohne Aufräumen wuchs der Ordner auf 3.500 Dateien / 750 MB."""
        import os
        d = Path(input_dir or os.environ.get("DMK_COMFY_INPUT") or Path(__file__).resolve().parents[2] / "ComfyUI" / "input")
        n = 0
        if d.is_dir():
            for f in d.glob(f"{self.tag}_dmk*"):
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def alive(self) -> bool:
        try:
            self._req("/system_stats", timeout=3)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def stats(self) -> dict:
        return json.loads(self._req("/system_stats", timeout=5))

    def options(self, node: str, field: str) -> list[str]:
        info = json.loads(self._req(f"/object_info/{node}"))
        spec = info[node]["input"]["required"][field]
        # alt: [[namen...], {...}]   neu: ["COMBO", {"options": [...]}]
        if isinstance(spec[0], list):
            return spec[0]
        return spec[1].get("options", [])

    def checkpoints(self) -> list[str]:
        return self.options("CheckpointLoaderSimple", "ckpt_name")

    def controlnets(self) -> list[str]:
        return self.options("ControlNetLoader", "control_net_name")

    def upload(self, img: np.ndarray, name: str = "dmk_init.png") -> str:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return self.upload_bytes(buf.tobytes(), name, "image/png")

    def upload_bytes(self, data: bytes, name: str, mime: str = "application/octet-stream", subfolder: str = "dmk") -> str:
        """Beliebige Datei in ComfyUIs input-Ordner legen (der Upload-Endpunkt prüft keine Endung).
        LoadLatent listet nur Dateien direkt im input-Ordner, dafür subfolder=""."""
        name = f"{self.tag}_{name}"
        b = uuid.uuid4().hex
        field = lambda k, v: f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body = b"".join([
            f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f'Content-Type: {mime}\r\n\r\n'.encode(), data, b"\r\n",
            field("overwrite", "true"), field("subfolder", subfolder), f"--{b}--\r\n".encode()])
        r = json.loads(self._req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
        return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]

    def queue(self, workflow: dict) -> str:
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        try:
            r = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))
        except urllib.error.HTTPError as e:
            raise ComfyError(f"ComfyUI rejected the workflow: {e.read().decode('utf-8', 'replace')[:1500]}") from None
        return r["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 900) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = json.loads(self._req(f"/history/{prompt_id}"))
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m[1].get("exception_message", "") for m in status.get("messages", [])
                            if m[0] == "execution_error"]
                    raise ComfyError("ComfyUI error: " + (" | ".join(msgs) or json.dumps(status)[:800]))
                return entry
            time.sleep(0.2)
        raise ComfyError(f"Timeout after {timeout:.0f} s (prompt {prompt_id}).")

    def image(self, entry: dict) -> np.ndarray:
        for out in entry.get("outputs", {}).values():
            for im in out.get("images", []):
                q = urllib.parse.urlencode({k: im.get(k, "") for k in ("filename", "subfolder", "type")})
                data = np.frombuffer(self._req("/view?" + q, timeout=60), np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
        raise ComfyError("ComfyUI returned no image (workflow without Save/PreviewImage?).")


class ComfySDK(Comfy):
    """Gleiche Schnittstelle wie `Comfy`, aber Uploads und Jobs laufen über das offizielle Comfy SDK
    (Comfy Developer Platform, API v2). Selbst gehostet über `comfy-api-proxy` neben ComfyUI:
        comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189
    `sdk_url` ist die Adresse des Proxys (setzt COMFY_BASE_URL). Abfragen der Knotenoptionen (Checkpoints,
    ControlNets) gehen weiter direkt an ComfyUI. Bilder werden als Platzhalter in den Workflow geschrieben
    (patch_workflow kopiert per deepcopy) und erst beim Absenden durch Asset-Handles ersetzt."""

    def __init__(self, server: str = "127.0.0.1:8188", sdk_url: str = "http://127.0.0.1:8189"):
        super().__init__(server)
        import os
        os.environ["COMFY_BASE_URL"] = sdk_url
        from comfy_sdk import Comfy as _SDK
        self.sdk = _SDK()
        self._assets: dict[str, object] = {}
        self._n = 0

    def upload_bytes(self, data: bytes, name: str, mime: str = "application/octet-stream", subfolder: str = "dmk") -> str:
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return super().upload_bytes(data, name, mime, subfolder)   # z. B. .latent für LoadLatent: direkter Weg
        self._n += 1
        token = f"dmk-asset:{self._n}"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            # Lauf-Kennung als PNG-Textchunk: das SDK lädt gleiche Bytes nicht erneut hoch (Hash-Dedup). Ohne Kennung
            # verwies ein neuer Lauf auf eine Datei, die das Aufräumen des vorigen Laufs schon gelöscht hatte.
            import struct
            import zlib
            txt = b"dmk\x00" + self.tag.encode()
            chunk = struct.pack(">I", len(txt)) + b"tEXt" + txt + struct.pack(">I", zlib.crc32(b"tEXt" + txt) & 0xFFFFFFFF)
            data = data[:33] + chunk + data[33:]
        self._assets[token] = self.sdk.assets.from_bytes(data, filename=f"{self.tag}_{name}", content_type=mime)
        return token

    def queue(self, workflow: dict):
        used = set()

        def ersetze(v):
            if isinstance(v, str) and v in self._assets:
                used.add(v)
                return self._assets[v]
            return v
        for node in workflow.values():
            ins = node.get("inputs", {})
            for k in list(ins):
                ins[k] = ersetze(ins[k])
        wf = self.sdk.workflows.from_json(workflow)
        job = None
        for versuch in range(3):
            try:
                job = self.sdk.submit(wf)
                break
            except Exception as e:  # noqa: BLE001
                # Zeitüberschreitung (ComfyUI kurz blockiert, Modell wird geladen): kurz warten, neu senden. Ein neuer
                # Idempotenz-Schlüssel kann im schlimmsten Fall ein Sample doppelt rechnen lassen, das ist harmlos.
                if "timed out" not in str(e).lower() and "timeout" not in type(e).__name__.lower() or versuch == 2:
                    raise ComfyError(f"Comfy SDK rejected the workflow: {e}") from None
                time.sleep(10)
        for t in used:          # Handles sind hochgeladen; Speicher pro Frame freigeben
            self._assets.pop(t, None)
        return job

    def wait(self, job, timeout: float = 900):
        # job.result() pollt mit wachsendem Abstand bis 5 s; unsere Jobs dauern 1–3 s, das kostete ~1 s pro Sample.
        # Deshalb festes kurzes Intervall über die öffentliche refresh()-Methode.
        t0 = time.time()
        try:
            while time.time() - t0 < timeout:
                job.refresh()
                if str(job.status).lower() in ("succeeded", "failed", "canceled", "expired"):
                    return job.result()
                time.sleep(0.15)
        except Exception as e:  # noqa: BLE001
            raise ComfyError(f"Comfy SDK: job failed: {e}") from None
        raise ComfyError(f"Comfy SDK: timeout after {timeout:.0f} s")

    def image(self, job) -> np.ndarray:
        for out in job.outputs:
            if str(out.name).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                data = np.frombuffer(out.to_bytes(), np.uint8)
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
        raise ComfyError(f"Comfy SDK: no image in the outputs ({[o.name for o in job.outputs]})")


def make_client(server: str = "127.0.0.1:8188", backend: str = "http", sdk_url: str = "http://127.0.0.1:8189") -> Comfy:
    """backend="http": ComfyUI-HTTP-API direkt (Entwicklung). backend="sdk": Comfy SDK über comfy-api-proxy."""
    return ComfySDK(server, sdk_url) if backend == "sdk" else Comfy(server)


def patch_workflow(wf: dict, image=None, prompt=None, negative=None, ckpt=None, seed=None,
                   steps=None, cfg=None, denoise=None, sampler=None, scheduler=None,
                   control_image=None, control_net=None, control_strength=None, control_end=None,
                   wandel_image=None, control2_image=None, rausch_latent=None, sigma0=None) -> dict:
    """Setzt die Laufparameter in einem API-Workflow. Sucht die Knoten über class_type bzw.
    _meta.title (init/control, positive/negative), damit eigene Workflows (--workflow)
    ohne Platzhalter funktionieren."""
    wf = copy.deepcopy(wf)

    def put(node, key, val):
        if val is not None and key in node["inputs"]:
            node["inputs"][key] = val

    def title(node):
        return str(node.get("_meta", {}).get("title", "")).lower()

    def text_node(start_id):
        """Folgt der Konditionierungs-Kette (ControlNet-Apply o. ä.) bis zum CLIPTextEncode."""
        seen = set()
        nid = str(start_id)
        while nid in wf and nid not in seen:
            seen.add(nid)
            node = wf[nid]
            if node.get("class_type") == "CLIPTextEncode":
                return node
            link = node["inputs"].get("conditioning") or node["inputs"].get("positive")
            if not isinstance(link, list):
                return None
            nid = str(link[0])
        return None

    loaders = [n for n in wf.values() if n.get("class_type") == "LoadImage"]
    plain = [n for n in loaders if title(n) not in ("control", "wandel", "control2")]
    for node in loaders:
        if title(node) == "control":
            put(node, "image", control_image)
        elif title(node) == "control2":
            put(node, "image", control2_image)
        elif title(node) == "wandel":
            put(node, "image", wandel_image)
        elif title(node) == "init" or len(plain) == 1:
            put(node, "image", image)
    if plain and not any(title(n) == "init" for n in plain) and len(plain) > 1:
        put(plain[0], "image", image)

    for node in wf.values():
        ct = node.get("class_type", "")
        if ct == "CheckpointLoaderSimple":
            put(node, "ckpt_name", ckpt)
        elif ct == "ControlNetLoader" and title(node) != "control2":
            put(node, "control_net_name", control_net)
        elif ct in ("ControlNetApplyAdvanced", "ControlNetApply") and title(node) != "control2":
            put(node, "strength", control_strength)
            put(node, "end_percent", control_end)
        elif ct == "CLIPTextEncode" and title(node) in ("positive", "negative"):
            put(node, "text", prompt if title(node) == "positive" else negative)
        elif ct == "LoadLatent" and title(node) == "rausch":
            put(node, "latent", rausch_latent)
        elif ct == "LatentMultiply" and title(node) == "sigma0":
            put(node, "multiplier", sigma0)
        elif ct == "BasicScheduler":
            put(node, "steps", steps)
            put(node, "denoise", denoise)
            put(node, "scheduler", scheduler)
        elif ct == "KSamplerSelect":
            put(node, "sampler_name", sampler)
        elif ct == "CFGGuider":
            put(node, "cfg", cfg)
            for key, text in (("positive", prompt), ("negative", negative)):
                link = node["inputs"].get(key)
                if text is not None and isinstance(link, list):
                    tn = text_node(link[0])
                    if tn is not None and title(tn) not in ("positive", "negative"):
                        put(tn, "text", text)
        elif ct in ("KSampler", "KSamplerAdvanced"):
            put(node, "seed", seed)
            put(node, "noise_seed", seed)
            put(node, "steps", steps)
            put(node, "cfg", cfg)
            put(node, "denoise", denoise)
            put(node, "sampler_name", sampler)
            put(node, "scheduler", scheduler)
            for key, text in (("positive", prompt), ("negative", negative)):
                link = node["inputs"].get(key)
                if text is not None and isinstance(link, list):
                    tn = text_node(link[0])
                    if tn is not None and title(tn) not in ("positive", "negative"):
                        put(tn, "text", text)
    return wf


def run(seq_root, out_dir, count=0, start=0, zoom="0:(1.02)", max_side=768, fps=24,
        alpha=0.3, denoise=0.45, steps=20, cfg=6.0, seed=1234, seed_mode="iter",
        angle="0:(0)", tx="0:(0)", ty="0:(0)", border="reflect", parallax=0.0, depth_invert=False,
        hf=0.0, hf_sigma=2.0, hf_mode="replace", controlnet="none", control_strength=0.8, control_net_name=None,
        control_gamma=1.0, stride=1, motion="zoom", flow_backend="auto", flow_side=640, occl_thresh=1.5,
        cadence=1, wrap=False, control_end=1.0, first_denoise=0.0,
        licht=0.0, licht_sigma=14.0, wandel="none", wandel_floor=0.35, antiblur=0.0, noise=0.0,
        kanten=0.0, kanten_end=0.8, quellzoom="0:(1.0)", quellzoom_zentrum=(0.5, 0.5), quellzoom_max=0.0, anker_zoom=0.0, anker_blend=12, fokus_tiefe=0.0, film=0.0,
        rausch="zufall", rausch_frisch=0.0, rausch_rho=0.0, gedaechtnis=0.0, ged_beta=0.25, ged_sigma=12.0, morph=True,
        pump_fix=True, mono=False, weich=0.0, weich_modus="init", schaerfe_halten=0.0, glaettung=0.0, referenz=0.0, referenz_bloecke="all",
        ram_min_gb=4.0, abbruch=None, fuehrung=None, tiefe_bild=None, welt_anker=None, beat_hart=False, lora=None, lora_staerke=1.0, audio=None, backend="http", sdk_url="http://127.0.0.1:8189",
        prompt=DEFAULT_PROMPT, negative=DEFAULT_NEGATIVE, ckpt=None, sampler=None, scheduler=None,
        server="127.0.0.1:8188", workflow=None, color_match="input", dry_run=False,
        pattern=None, depth_cache=None, mask=None, log=print, **_):
    """hf > 0: Hochfrequenzen des aktuellen Quellframes vor dem Sampler zurückspeisen (Schärfe).
    controlnet=depth|canny: Raumstruktur pro Frame über ControlNet festhalten (Tiefenkarte der
    Quelle bzw. Canny-Kanten des Quellframes); der Inhalt wandelt sich über Prompt und denoise.
    Fehlen Tiefenkarten, werden sie geschätzt (dmk.depth) und unter depth_cache abgelegt.
    motion="flow": das vorige Ergebnis wird mit dem optischen Fluss der QUELLE weitergetragen (RAFT),
    der Zoom-Schedule kommt nur obendrauf. Ein fester Zoom, der nicht zur Kamerafahrt der Quelle passt,
    zieht das Bild gegen die per ControlNet festgehaltene Geometrie und erzeugt radiale Schlieren."""
    t0 = time.time()
    seq = core.load_sequence(seq_root, count, start, pattern, stride=stride, wrap=wrap)
    n = len(seq)
    # fuehrung: Frame-Ordner, aus dem Tiefe (ControlNet) und Bewegung (Fluss) kommen, während das Bild aus seq_root kommt.
    # Flux Hybrid (Nutzer 27.10.: "die räumliche Vorlage soll immer dieselbe bleiben"): Raum und Bewegung vom
    # Originalvideo, nicht von den Flux-Bildern – sonst übernimmt die Schleife jede Neukomposition von Flux
    fseq = None
    if fuehrung:
        fseq = core.load_sequence(fuehrung, n, start, stride=stride)
        from . import depth as depth_mod
        depth_mod.ensure(fseq, depth_cache or Path(out_dir).parent / "_depth", log=log)
        depth_mod.freigeben()
        seq.depth = list(fseq.depth[:n])
        log(f"[A] Depth and motion from the guide sequence {Path(fuehrung).name} ({len(fseq)} frames)")
    if tiefe_bild:   # eigene Tiefenkarte zum Einzelbild (hell = nah, 8 oder 16 bit): exakt statt geschätzt (Nutzer 28.10.)
        seq.depth = [Path(tiefe_bild)] * n
        log(f"[A] Own depth map: {Path(tiefe_bild).name}")
    zs, angs, txs, tys = (core.schedule(s, n) for s in (zoom, angle, tx, ty))
    fdir = core.clear_frames(out_dir)

    # Pro-Frame-Werte: diese Parameter dürfen Zahl, Deforum-Schedule ("0:(0.1), 60:(0.3)") oder Array sein.
    # Eine Audio-Bindung (dmk.audio, Parameter `audio`) überschreibt sie mit Kurven aus dem Track; so werden
    # Fließen (quellzoom/alpha), Verschwimmen (weich), Wandel (denoise) und Gedächtnis audioreaktiv.
    kurven: dict = {}
    force = None          # Frames, an denen ein Sample erzwungen wird (Beat), unabhängig von der Cadence
    audio_info = None
    if audio:
        from . import audio as audio_mod
        kurven, force, _, audio_info = audio_mod.aufloesen(audio, n, stride, start, out_dir, log)
    if "zoom" in kurven:
        zs = kurven["zoom"]

    def pro_frame(v):
        if isinstance(v, str):
            return core.schedule(v, n)
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=np.float64)
            return a if len(a) == n else np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
        return np.full(n, float(v))
    _pf_namen = ("alpha", "antiblur", "noise", "gedaechtnis", "licht", "hf", "control_strength", "cfg", "weich",
                 "wandel_floor", "schaerfe_halten", "glaettung", "referenz")
    _loc = locals()
    pf = {nm: pro_frame(kurven.get(nm, _loc[nm])) for nm in _pf_namen}
    (alpha, antiblur, noise, gedaechtnis, licht, hf, control_strength, cfg, weich, wandel_floor, schaerfe_halten, glaettung,
     referenz) = (
        float(pf[nm][0]) for nm in _pf_namen)
    for nm in _pf_namen:
        if np.ptp(pf[nm]) > 0:
            log(f"[A] {nm} per frame: {pf[nm].min():.3g}..{pf[nm].max():.3g} (mean {pf[nm].mean():.3g})")

    if (controlnet == "depth" or parallax or wandel != "none" or motion in ("flow", "kamera", "vektor", "virtuell")) and not seq.depth:
        from . import depth as depth_mod
        try:
            depth_mod.ensure(seq, depth_cache or Path(out_dir).parent / "_depth", log=log)
            depth_mod.freigeben()   # Modell wird danach nicht mehr gebraucht: RAM/VRAM sofort zurückgeben
        except Exception as e:  # Modell nicht ladbar, kein Netz, OOM
            log(f"[A] Depth estimation failed ({type(e).__name__}: {e}).")
            if controlnet == "depth":
                log("[A] Using Canny edges instead of depth for the ControlNet.")
                controlnet = "canny"

    comfy = wf = None
    wf_ref = None
    if not dry_run:
        comfy = make_client(server, backend, sdk_url)
        if not comfy.alive():
            raise ComfyError(f"ComfyUI is not reachable at {server}. Start it first (app: 'Start services', or start_comfy.bat) "
                             f"or use --dry-run.")
        default_wf = DEPTH_WORKFLOW if controlnet != "none" else DEFAULT_WORKFLOW
        wf = json.loads(Path(workflow or default_wf).read_text(encoding="utf-8"))
        if wandel != "none" and seq.depth:
            wf = add_wandelkarte(wf)
        if lora:
            namen = comfy.options("LoraLoaderModelOnly", "lora_name")
            treffer = next((x for x in namen if lora.lower() in x.lower()), None)
            if treffer is None:
                raise ComfyError(f"LoRA {lora!r} not found in ComfyUI models/loras: {namen}")
            wf = add_lora(wf, treffer, lora_staerke)
            log(f"[A] LoRA {treffer} @ {lora_staerke}")
        wf_ref = None
        if pf["referenz"].max() > 0.05:
            wf_ref = "ausstehend"   # wird unten gebaut, nachdem alle anderen Zusätze im Workflow sind
        if rausch == "transport":
            if motion not in ("flow", "kamera", "vektor", "virtuell"):
                log("[A] Noise transport needs --motion flow|kamera; using random noise.")
                rausch = "zufall"
            else:
                wf = add_rausch_transport(wf, sampler or "dpmpp_2m", scheduler or "karras")
                log("[A] Noise transport: initial noise is carried along with the motion (integral noise)")
        if kanten > 0 and controlnet != "none":
            # zweiter Raum-Anker: Kantenbild (Blender-Pass "kanten", sonst Tiefen- und Bildkanten) am Canny-ControlNet
            cn2 = next((c for p in _CN_PREFER["canny"] for c in comfy.controlnets()
                        if p in c.lower() and "sd 1.5" in c.lower()), None)
            if cn2 is None:
                log("[A] No Canny ControlNet found, --kanten is skipped.")
                kanten = 0.0
            else:
                wf = add_controlnet2(wf, cn2, kanten, kanten_end)
                log(f"[A] Kanten-ControlNet: {cn2} @ {kanten} "
                    f"({'Blender-Pass' if seq.passes and seq.passes.get('kanten') else 'aus Tiefe und Bild berechnet'})")
        if ckpt is None and any(v.get("class_type") == "CheckpointLoaderSimple" for v in wf.values()):
            names = comfy.checkpoints()
            if not names:
                raise ComfyError("ComfyUI has no checkpoints (check models/checkpoints or extra_model_paths.yaml).")
            ckpt = next((c for p in _CKPT_PREFER for c in names if p in c.lower()), None)
            if ckpt is None:   # nicht still irgendein Modell nehmen (z. B. SDXL): das gäbe Fehler oder Datenmüll
                raise ComfyError("DreamShaper 8 (SD 1.5) not found in ComfyUI models/checkpoints. Download "
                                 "dreamshaper_8.safetensors (see README) or pass ckpt=... explicitly.")
        if controlnet != "none" and control_net_name is None:
            names = comfy.controlnets()
            control_net_name = next((c for p in _CN_PREFER[controlnet] for c in names
                                     if p in c.lower() and "sd 1.5" in c.lower()), None) \
                or next((c for p in _CN_PREFER[controlnet] for c in names if p in c.lower()), None)
            if control_net_name is None:
                raise ComfyError(f"No {controlnet} ControlNet found in ComfyUI. Expected control_v11f1p_sd15_depth in models/controlnet. Found: {names}")
        if wf_ref == "ausstehend":
            if any(v.get("class_type") == "KSampler" for v in wf.values()):
                wf_ref = add_referenz(wf, referenz_bloecke)
                log(f"[A] Reference attention (dmk node): model sees the warped previous frame, blocks={referenz_bloecke}")
            else:
                log("[A] Reference attention needs the KSampler workflow (not noise transport); skipped.")
                wf_ref = None
        log(f"[A] ComfyUI {server}, Checkpoint: {ckpt}" +
            (f", ControlNet {controlnet}: {control_net_name} @ {control_strength}" if controlnet != "none" else ""))

    # Wie in Deforum/A1111 zählt --steps für den vollen Rauschbereich; gerechnet werden nur
    # steps * denoise Schritte. ComfyUIs KSampler würde sonst immer alle steps ausführen.
    # denoise darf ein Schedule sein ("0:(0.4), 40:(0.65), 43:(0.4)" = Motiv-Impuls an Prompt-Keyframes, wie
    # Deforums strength_schedule); gehalten wird pro Frame der interpolierte Wert
    if "denoise" in kurven:
        denoise_s = kurven["denoise"]
    else:
        denoise_s = core.schedule(denoise, n) if isinstance(denoise, str) else None
    denoise = float(denoise_s[0]) if denoise_s is not None else float(denoise)
    real_steps = max(1, round(steps * denoise))
    pseq = None      # Blender-Pass-Sequenz (EXR), wird unten gesetzt; sample() greift darauf zu
    frame_nr: list = []
    prompt_keys = parse_prompt_schedule(prompt)
    if len(prompt_keys) > 1:
        log(f"[A] Prompt schedule: " + " | ".join(f"{f}: {t[:40]}…" if len(t) > 40 else f"{f}: {t}" for f, t in prompt_keys))

    # Digitale Kamerafahrt in die Quelle: Quellbild, Tiefe und Pässe werden gemeinsam kumulativ gezoomt.
    # So bekommt auch Material mit stehender Kamera eine echte, in sich stimmige Vorwärtsbewegung, und
    # Rückkopplung, Führung (ControlNet) und Quelle arbeiten nicht gegeneinander.
    qz_s = kurven["quellzoom"] if "quellzoom" in kurven else core.schedule(quellzoom, n)
    qz = np.cumprod(np.concatenate([[1.0], qz_s[1:]]))
    zoomt = bool(qz.max() > 1.0 + 1e-6)
    # Kamera als Folge von Ausschnitten in normierten Quellkoordinaten (Breite = Höhe = 1): Ursprung o_i, Maßstab qz_i.
    # Punkt (x, y): Ausschnitt um diesen Punkt (wie bisher). Pfad [[x, y], ...]: Fokuspunkt IM BILD, der über die Zeit
    # den Pfad entlang wandert; jeder Frame zoomt um diesen Punkt. Der neue Ausschnitt liegt so immer im alten, die
    # Fahrt driftet zur Fokusseite und ist in jeder Tiefe gleich schnell (Nutzer 21.10.: "durchs ganze Bild gleiten").
    zp = np.asarray(quellzoom_zentrum, dtype=np.float64)
    pfad = zp.ndim == 2 and len(zp) >= 2
    if pfad or fokus_tiefe:
        zp = zp if pfad else np.array([zp if zp.ndim == 1 else zp[0]] * 2)
        stuetz = np.linspace(0.0, 1.0, len(zp))
        t_n = np.arange(n) / max(1, n - 1)
        fx, fy = np.interp(t_n, stuetz, zp[:, 0]), np.interp(t_n, stuetz, zp[:, 1])
        ox_s, oy_s = np.zeros(n), np.zeros(n)
        for i in range(1, n):
            k = 1.0 - qz[i - 1] / qz[i]
            ox_s[i] = ox_s[i - 1] + fx[i] / qz[i - 1] * k
            oy_s[i] = oy_s[i - 1] + fy[i] / qz[i - 1] * k
    else:
        c = zp if zp.ndim == 1 else zp[0]
        oo = [core.zoom_origin(1, 1, float(z), float(c[0]), float(c[1])) for z in qz]
        ox_s, oy_s = np.array([o[0] for o in oo]), np.array([o[1] for o in oo])
    if zoomt:
        log(f"[A] Virtual camera: zoom up to {qz.max():.2f}x" + (" along a focus path" if pfad else ""))
        (Path(out_dir) / "quelle").mkdir(parents=True, exist_ok=True)

    def ausschnitt(img, i, o, qa, interp=cv2.INTER_LANCZOS4):
        """Ausschnitt von Frame i aus einem Bild, das den Bereich ab Ursprung o (normiert) im Maßstab qa zeigt."""
        if not zoomt:
            return img
        z = float(qz[i]) / qa
        m = np.float32([[1 / z, 0, (ox_s[i] - o[0]) * qa * img.shape[1]], [0, 1 / z, (oy_s[i] - o[1]) * qa * img.shape[0]]])
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=interp | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT_101)

    def gezoomt(img, i, interp=cv2.INTER_LANCZOS4):   # Pässe/IDs/Tiefe im Koordinatensystem der Quelle
        return ausschnitt(img, i, (0.0, 0.0), 1.0, interp)

    # Endlose Fahrt (anker_zoom > 0): statt an den Anfang zu springen (Nutzer 21.10.: "der Fluss darf nicht unterbrochen
    # werden"), wird ab einer Tiefe das eigene zuletzt gezeigte Bild zur neuen Quelle (Anker) und bekommt eine eigene
    # Tiefe; die Kamera fliegt darin weiter. Über `anker_blend` Frames wird von der alten auf die neue Quelle überblendet.
    # Der erste Anker ist das Quellbild selbst, gültig bis `quellzoom_max` (so tief, wie es Pixel hat).
    anker = [{"img": None, "dep": None, "o": (0.0, 0.0), "qa": 1.0, "start": 0,
              "lim": float(quellzoom_max) if quellzoom_max else float("inf")}]
    anker_log: list = []
    fokus_zustand: dict = {}

    def _mix(i, nimm):
        """Quelle bzw. Tiefe aus dem aktiven Anker, während eines Übergangs aus altem und neuem gemischt."""
        neu_a = anker[-1]
        x = nimm(neu_a, i)
        ab = int(neu_a.get("blend", anker_blend))
        if len(anker) > 1 and i - neu_a["start"] < ab:
            t = (i - neu_a["start"] + 1) / float(ab + 1)
            t = t * t * (3 - 2 * t)
            alt = nimm(anker[-2], i)
            alt = cv2.resize(alt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA) if alt.shape[:2] != x.shape[:2] else alt
            x = cv2.addWeighted(alt, 1.0 - t, x, t, 0)
        return x

    _orig: dict = {}

    def _quelle_von(a, i):
        if a["img"] is None:
            if "img" not in _orig:
                _orig["img"] = core.imread(seq.frames[i])
            src = _orig["img"] if len(set(map(str, seq.frames[:2]))) == 1 else core.imread(seq.frames[i])
        else:
            src = a["img"]
        img = core.prep(ausschnitt(src, i, a["o"], a["qa"]), max_side, 8)
        return img if "w" not in _orig else cv2.resize(img, (_orig["w"], _orig["h"]), interpolation=cv2.INTER_AREA)

    def _tiefe_von(a, i):
        if a["dep"] is None:
            return gezoomt(core.read_depth(seq.depth[i], None, depth_invert), i, cv2.INTER_LINEAR)
        return ausschnitt(a["dep"], i, a["o"], a["qa"], cv2.INTER_LINEAR)

    def _entrauschen(img, h=7):
        if mono:
            g = cv2.fastNlMeansDenoising(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, h, 7, 21)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def anker_pruefen(i, bild):
        """Vor Frame i: ist der aktive Anker ausgereizt, wird `bild` (Frame i-1, scharf) der neue Anker."""
        if not anker_zoom or not zoomt or not seq.depth:
            return
        a = anker[-1]
        if float(qz[i]) / a["qa"] <= a["lim"]:
            return
        from . import depth as depth_mod
        # Anker entrauschen (kantenerhaltend): sonst schaukelt sich die Nachschärfung von Anker zu Anker auf – Nacht
        # 21./22.10.: nach 29 Ankern (40 s) Rauschen und am Ende ein Pixelmosaik
        bild = _entrauschen(bild)
        g = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY) if mono else None
        dep = depth_mod.schaetze(bild)
        anker.append({"img": cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g is not None else bild.copy(), "dep": dep,
                      "o": (float(ox_s[i - 1]), float(oy_s[i - 1])), "qa": float(qz[i - 1]), "start": i,
                      "lim": float(anker_zoom)})
        if len(anker) > 3:
            del anker[0]
        anker_log.append(i)
        log(f"[A] Endless glide: frame {i - 1} becomes the new source (anchor {len(anker_log)}, depth {float(qz[i - 1]):.1f}x)")

    def welt_anker_setzen(i, bild):
        """Flux Hybrid mit Standbild (Nutzer 29.10.: Flur in die Tiefe): auf dem Weltwechsel zeichnet `welt_anker` den
        AKTUELLEN Blick (schon tief in der Fahrt) in die neue Welt um; das Bild wird sofort der neue Anker. Sonst übernimmt
        nach 1,5x Zoom das eigene Bild die Quelle und spätere Welten aus der Flux-Stufe kämen nie an (Test f2_flur_flux)."""
        neu = welt_anker(bild, kurven["_prompt"][i])
        if neu is None:
            return
        from . import depth as depth_mod
        neu = cv2.resize(neu, (bild.shape[1], bild.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        anker.append({"img": neu, "dep": depth_mod.schaetze(neu), "o": (float(ox_s[i - 1]), float(oy_s[i - 1])),
                      "qa": float(qz[i - 1]), "start": i, "lim": float(anker_zoom) if anker_zoom else 1e9, "blend": 2})
        if len(anker) > 3:
            del anker[0]
        anker_log.append(i)
        log(f"[A] World change at frame {i}: the current view is redrawn as the new world and becomes the source")

    def _fuehr_von(a, i):
        src = core.imread(fseq.frames[min(i, len(fseq) - 1)]) if a["img"] is None else a["img"]
        img = core.prep(ausschnitt(src, i, a["o"], a["qa"]), max_side, 8)
        return img if "w" not in _orig else cv2.resize(img, (_orig["w"], _orig["h"]), interpolation=cv2.INTER_AREA)

    def fuehr_bild(i, size):
        return cv2.resize(_mix(i, _fuehr_von), size, interpolation=cv2.INTER_AREA)

    def quelle(i):
        img = _mix(i, _quelle_von)
        if mono:   # neu aufgedeckte Flächen zeigen die Quelle direkt: bei Schwarzweiß sonst Farbblitze (Fahrt 21.10.)
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if zoomt:  # die tatsächlich benutzte Quelle ist die Referenz für alle Kennzahlen
            core.imwrite(Path(out_dir) / "quelle" / f"{i:05d}.png", img)
        return img

    def tiefe(i, size):
        d = cv2.resize(_mix(i, _tiefe_von), size, interpolation=cv2.INTER_LINEAR)
        if zoomt:
            lo, hi = float(d.min()), float(d.max())
            d = (d - lo) / (hi - lo) if hi > lo else d
            p = Path(out_dir) / "quelle_depth" / f"{i:05d}_depth.png"
            if not p.exists():
                core.imwrite(p, (d * 65535).astype(np.uint16))
        return d

    def control_image(i, inp):
        if controlnet == "tile":   # Quellbild selbst als Führung: hält grafische Struktur, wo Tiefe nichts hergibt
            return inp
        if controlnet == "depth":
            d = tiefe(i, (inp.shape[1], inp.shape[0]))
            if control_gamma != 1.0:  # < 1 hebt die Ferne an, > 1 spreizt das Nahfeld: Raumtiefe betonen
                d = np.power(d, control_gamma)
            return cv2.cvtColor((d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if controlnet == "canny":
            e = cv2.Canny(cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY), 80, 180)
            return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
        return None

    def kanten_bild(i, inp):
        """Geometriekanten: Blender-Pass, sonst Sprünge in der Tiefenkarte plus kräftige Bildkanten."""
        if seq.passes and seq.passes.get("kanten"):
            e = gezoomt(core.imread(seq.passes["kanten"][i], gray=True), i, cv2.INTER_LINEAR)
            return cv2.cvtColor(cv2.resize(e, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_GRAY2BGR)
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY), (0, 0), 1.0), 60, 150)
        if seq.depth:
            d8 = (tiefe(i, (inp.shape[1], inp.shape[0])) * 255).astype(np.uint8)
            e = cv2.max(e, cv2.Canny(d8, 20, 60))
        return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

    def licht_quelle(i, inp):
        """Lichtkarte: Blender-Pass "licht" (Licht ohne Material), sonst das Quellbild selbst."""
        if seq.passes and seq.passes.get("licht"):
            l = gezoomt(core.imread(seq.passes["licht"][i]), i, cv2.INTER_LINEAR)
            return cv2.resize(l, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_AREA)
        return inp

    def sample(init, i, inp, dn=None, ref=None):
        if dry_run:
            return init
        if ram_min_gb:
            from . import gpu as _gpu
            _gpu.ram_waechter(ram_min_gb, ram_min_gb + 2.0, log=log)
        if dn is None:
            dn = float(denoise_s[min(i, n - 1)]) if denoise_s is not None else denoise
        name = comfy.upload(init)
        cname = None
        ctrl = control_image(i, inp)
        if ctrl is not None:
            cname = comfy.upload(ctrl, "dmk_control.png")
        k2name = None
        if kanten > 0:
            k2name = comfy.upload(kanten_bild(i, inp), "dmk_kanten.png")
        wname = None
        if isinstance(wandel, str) and wandel.startswith("ids") and pseq is not None and i > 0:
            # Wandelkarte aus Objekt-IDs: "ids:6,5" = diese Objekte bleiben (nur wandel_floor), alles andere wandelt voll
            halten = [int(x) for x in re.findall(r"\d+", wandel.split(":", 1)[1])] if ":" in wandel else []
            ids = pseq.objekt_ids(frame_nr[i], (inp.shape[1], inp.shape[0]))
            if ids is not None and zoomt:
                ids = gezoomt(ids, i, cv2.INTER_NEAREST)
            keep = np.isin(np.rint(ids).astype(int), halten) if ids is not None else np.zeros(inp.shape[:2], bool)
            m = np.where(keep, float(pf["wandel_floor"][i]), 1.0).astype(np.float32)
            m = cv2.GaussianBlur(m, (0, 0), 2)
            wname = comfy.upload(cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), "dmk_wandel.png")
        elif wandel != "none" and seq.depth and i > 0:
            dmap = tiefe(i, (inp.shape[1], inp.shape[0]))
            wname = comfy.upload(wandel_bild(dmap, wandel, float(pf["wandel_floor"][i])), "dmk_wandel.png")
        elif wandel != "none" and seq.depth:  # Frame 0 darf überall voll umgezeichnet werden
            wname = comfy.upload(np.full(inp.shape, 255, np.uint8), "dmk_wandel.png")
        if "_seed" in kurven:            # Seed-Sprung aus dem Ton (z. B. pro Takt)
            s = seed + int(kurven["_seed"][i])
        else:
            s = seed + i if seed_mode == "iter" else seed
        rname = s0 = None
        if transport is not None:
            from . import rauschen
            # eindeutiger Name pro Sample: LoadLatent cacht über den Dateinamen/-hash
            rname = comfy.upload_bytes(rauschen.latent_datei(transport.latent()), f"dmk_rausch_{i % 2}.latent",
                                       "application/octet-stream", subfolder="")
        # Weltwechsel aus dem Ton (audio "welt_wechsel") ersetzt den Prompt-Fahrplan
        ptxt = (kurven["_prompt"][i] if "_prompt" in kurven else prompt_at(prompt_keys, i)) +             (kurven["_prompt_zusatz"][i] if "_prompt_zusatz" in kurven else "")
        mit_ref = ref is not None and wf_ref is not None and float(pf["referenz"][i]) > 0.05
        wfp = patch_workflow(wf_ref if mit_ref else wf, name, ptxt, negative, ckpt, s,
                             max(1, round(steps * dn)), float(pf["cfg"][i]), dn, sampler, scheduler, cname,
                             control_net_name, float(pf["control_strength"][i]), control_end, wname, k2name,
                             rname, s0)
        if mit_ref:   # Referenz-Aufmerksamkeit: das Modell sieht das mitbewegte vorige Ergebnis beim Zeichnen
            wfp[REF_LADEN]["inputs"]["image"] = comfy.upload(ref, "dmk_ref.png")
        pid = comfy.queue(wfp)
        img = comfy.image(comfy.wait(pid))
        if img.shape[:2] != init.shape[:2]:
            img = cv2.resize(img, (init.shape[1], init.shape[0]), interpolation=cv2.INTER_AREA)
        return img

    first = quelle(0)
    h, w = first.shape[:2]
    _orig.update(w=w, h=h)
    transport = None
    if rausch == "transport" and not dry_run:
        from . import rauschen
        transport = rauschen.NoiseTransport(h, w, seed)
    mk = core.load_mask(mask, (w, h)) if mask else None
    if mk is not None:
        log(f"[A] Mask {mask}: {mk.mean() * 100:.0f} % of the image active")
    # first_denoise > denoise: Frame 0 stärker umzeichnen, damit Palette und Texturreichtum vom Modell kommen
    # (eine dunkle Quelle bleibt bei denoise 0.45 dunkel; gen0 schreibt das dann für den ganzen Lauf fest)
    out = sample(first, 0, first, first_denoise or None)
    if mono:  # Schwarzweiß: der Negativ-Prompt "color" reicht nicht, gen0 würde die Palette sonst festschreiben
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    gen0 = out.copy()  # Farbreferenz "gen0": erster generierter Frame, wie Deforums Color Coherence
    lv0 = max(1e-6, core.lap_var(gen0))
    if color_match in ("input", "first"):  # sonst fällt Frame 0 farblich aus der Reihe
        out = core.match_color(out, first)
    out = core.apply_mask(out, mk)
    core.imwrite(fdir / "00000.png", core.film_finish(out, 0, film, mono, seed))

    flow_be = None
    soll: list = []   # Soll-Bewegung pro Frame (Kameramodell, als Anteil der Bildbreite) für die Kennzahl mitfluss
    if motion == "vektor" or (isinstance(wandel, str) and wandel.startswith("ids")):
        from . import passes as passes_mod
        pseq = passes_mod.PassSequenz(seq.root.parent)
        frame_nr = [int((re.findall(r"\d+", p.stem) or ["0"])[-1]) for p in seq.frames]
        if motion == "vektor" and not pseq.exr:
            log("[A] No EXR passes next to the sequence, --motion vektor falls back to kamera.")
            motion = "kamera"
        elif motion == "vektor":
            log(f"[A] Motion from the Blender vector pass ({len(pseq.exr)} EXR frames), stride {stride}")
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    if motion == "virtuell":
        # Virtuelle Kamera ohne Blender-Pass: die Szene steht (Foto, Render, Projektion mit fester Kamera), die
        # Bewegung ist allein die digitale Kamerafahrt (quellzoom + Fluchtpunkt) und damit exakt berechenbar
        log("[A] Motion from the virtual camera (zoom around the vanishing point, no passes needed)")
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    if motion in ("flow", "kamera"):
        from . import method_c
        flow_be = method_c.make_backend(flow_backend, log)
        log(f"[A] Motion from the optical flow of the source ({flow_be.name}), stride {stride}")
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    prev_in = first
    prev_f = fuehr_bild(0, (w, h)) if fseq is not None else None
    cad = max(1, int(cadence))
    prev_d = out
    shown = out            # zuletzt geschriebenes Bild: Ausgangspunkt, wenn ein Beat mitten im Morph ein Sample erzwingt
    letztes_sample = 0
    # Mitbewegtes Gedächtnisbild (Review-Vorschlag 3): hält das Grobband (Formen in Objektgröße) über viele
    # Frames; Wandel wird rückkehrend statt Zufallslauf, ein neues Objekt blendet ein statt aufzuploppen.
    mem = out.astype(np.float32) if pf["gedaechtnis"].max() > 0 else None
    lp = lambda img: cv2.GaussianBlur(img.astype(np.float32), (0, 0), ged_sigma)
    # Cadence als Fluss-Morph (Deforums optical_flow_cadence) statt linearer Überblendung: die lineare Mischung
    # zweier Samples zeigt Doppelbilder und pumpt in der Schärfe; der Morph schiebt beide aufeinander zu.
    morph_flow = None
    dis = None
    if morph and cad > 1:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ys_m, xs_m = np.mgrid[0:h, 0:w].astype(np.float32)

    def morph_frame(a, b, t):
        """a (altes Sample) nach b (neues Sample) zu Anteil t morphen; beide liegen in derselben Pose."""
        if morph_flow is None:
            return cv2.addWeighted(a, 1.0 - t, b, t, 0)
        fa = cv2.remap(a, xs_m - t * morph_flow[..., 0], ys_m - t * morph_flow[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)             # a entlang a→b vorwärts
        fb = cv2.remap(b, xs_m + (1.0 - t) * morph_flow[..., 0], ys_m + (1.0 - t) * morph_flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)  # b entlang b→a zurück
        return cv2.addWeighted(fa, 1.0 - t, fb, t, 0)

    for i in range(1, n):
        if abbruch is not None and abbruch():   # App: Cancel-Knopf; bisherige Frames bleiben erhalten
            log(f"[A] cancelled after {i} frames")
            if comfy is not None:
                comfy.uploads_aufraeumen()
            raise KeyboardInterrupt("render cancelled")
        occ = None   # Verdeckungskarte dieses Frames (nur bei Fluss/Vektor/virtuell gesetzt)
        if motion == "virtuell" and fokus_tiefe and zoomt:
            # Tiefe folgen: der Fokus steuert weich auf den fernsten Bereich des letzten Bildausschnitts zu (Öffnungen,
            # Gänge, Fluchten) – die Kamera fliegt durch den Raum statt in Wände (Nacht 21./22.10.: tiefe Fahrten
            # landeten frontal auf Balken). Gemischt mit dem Pfad nach Gewicht fokus_tiefe.
            dv = cv2.GaussianBlur(cv2.resize(tiefe(i - 1, (96, 54)), (96, 54)), (0, 0), 6)
            # fern UND strukturiert: Himmel ist auch "fern", hat aber keine Kanten (Look-Test 21.10.: Außenaufnahmen
            # zogen die Kamera in den Himmel). Kantendichte des zuletzt gezeigten Bildes gewichtet die Ferne.
            kg = cv2.resize(cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
            kd = cv2.GaussianBlur(np.abs(cv2.Laplacian(kg, cv2.CV_32F)), (0, 0), 6)
            kd = kd / (kd.max() + 1e-6)
            hell = cv2.GaussianBlur(kg / 255.0, (0, 0), 6)            # dunkle Bereiche meiden (tiefes Wasser, Nacht)
            score = (1.0 - (dv - dv.min()) / (np.ptp(dv) + 1e-6)) * (0.15 + kd) * np.clip(hell / (hell.mean() + 1e-6), 0.2, 1.5)
            yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)    # hell = nah → fern und strukturiert
            ziel = (np.clip(xx / 95.0, 0.2, 0.8), np.clip(yy / 53.0, 0.25, 0.75))
            fz = fokus_zustand.get("f", (float(fx[i]), float(fy[i])))
            fz = (fz[0] + 0.04 * (ziel[0] - fz[0]), fz[1] + 0.04 * (ziel[1] - fz[1]))
            fokus_zustand["f"] = fz
            gx = (1 - fokus_tiefe) * float(fx[i]) + fokus_tiefe * fz[0]
            gy = (1 - fokus_tiefe) * float(fy[i]) + fokus_tiefe * fz[1]
            k = 1.0 - qz[i - 1] / qz[i]
            ox_s[i] = ox_s[i - 1] + gx / qz[i - 1] * k
            oy_s[i] = oy_s[i - 1] + gy / qz[i - 1] * k
        if motion == "virtuell":
            if welt_anker and "_prompt" in kurven and kurven["_prompt"][i] != kurven["_prompt"][i - 1]:
                welt_anker_setzen(i, shown)
            else:
                anker_pruefen(i, shown)
        inp = quelle(i)
        if inp.shape[:2] != (h, w):
            inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_AREA)
        depth = None
        if parallax and seq.depth:
            depth = tiefe(i - 1, (w, h))
        if (motion == "vektor" and pseq is not None) or motion == "virtuell":
            if zoomt:
                # digitale Kamerafahrt: Pixel p des gezoomten Frames i liegt in der Quelle bei o_i + p/qz[i];
                # dort die Vektoren verketten; die Quellposition zurück in den gezoomten Frame i-1: (s - o_{i-1})·qz[i-1]
                ox, oy = float(ox_s[i] * w), float(oy_s[i] * h)
                sx, sy = ox + xs / float(qz[i]), oy + ys / float(qz[i])
                rb = ((sx, sy, np.ones((h, w), bool)) if motion == "virtuell"
                      else pseq.rueckabbildung(frame_nr[i], frame_nr[i - 1], (w, h), start=(sx, sy)))
                if rb is None:
                    raise RuntimeError(f"Vector pass missing for frames {frame_nr[i - 1]}..{frame_nr[i]} (EXR not rendered yet?)")
                mx, my, ok = rb
                px, py = float(ox_s[i - 1] * w), float(oy_s[i - 1] * h)
                mx, my = (mx - px) * float(qz[i - 1]), (my - py) * float(qz[i - 1])
                ok &= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            else:
                rb = ((xs.copy(), ys.copy(), np.ones((h, w), bool)) if motion == "virtuell"
                      else pseq.rueckabbildung(frame_nr[i], frame_nr[i - 1], (w, h)))
                if rb is None:
                    raise RuntimeError(f"Vector pass missing for frames {frame_nr[i - 1]}..{frame_nr[i]} (EXR not rendered yet?)")
                mx, my, ok = rb
            occ = cv2.GaussianBlur((~ok).astype(np.float32), (0, 0), 1.0)
            z = float(zs[i])
            if abs(z - 1.0) > 1e-6:  # Zoom-Schedule obendrauf
                mx, my = cx + (mx - cx) / z, cy + (my - cy) / z
            soll.append(cv2.resize(np.stack([mx - xs, my - ys], -1), (96, max(8, round(96 * h / w))),
                                   interpolation=cv2.INTER_AREA) / w)
            if transport is not None:
                frisch = np.maximum(occ, rausch_frisch) if rausch_frisch > 0 else occ
                transport.advect(mx, my, frisch, rausch_rho)
            occ = occ[..., None]

            def advect(img, mx=mx, my=my, occ=occ, inp=inp):
                wimg = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=core.BORDERS[border])
                return (occ * inp + (1.0 - occ) * wimg).astype(np.uint8)
            prev_in = inp
        elif flow_be is not None:
            f_a, f_b = (prev_in, inp) if fseq is None else (prev_f, fuehr_bild(i, (w, h)))
            try:
                fwd, bwd = method_c._flow_pair(flow_be, f_a, f_b, flow_side)
            except RuntimeError as e:  # z. B. CUDA out of memory neben ComfyUI
                if flow_be.name == "farneback":
                    raise
                log(f"[A] {flow_be.name} failed ({str(e).splitlines()[0]}), continuing with Farneback.")
                flow_be = method_c.FarnebackFlow()
                fwd, bwd = method_c._flow_pair(flow_be, f_a, f_b, flow_side)
            occ = method_c.occlusion(fwd, bwd, xs, ys, occl_thresh)
            if seq.depth:
                # Kameramodell aus Tiefe + glaubwürdigem Fluss: Soll-Bewegung auch für glatte Flächen
                conf = method_c.flow_konfidenz(f_b, fwd, bwd, xs, ys)
                dmap = tiefe(i, (w, h))
                modell = method_c.kamera_fluss(bwd, dmap, conf)
                soll.append(cv2.resize(modell, (96, max(8, round(96 * h / w))), interpolation=cv2.INTER_AREA) / w)
                if motion == "kamera":
                    c = cv2.GaussianBlur(np.clip(conf * 1.5, 0, 1), (0, 0), 5)[..., None]
                    bwd = c * bwd + (1.0 - c) * modell   # Bildschätzung nur, wo sie glaubwürdig ist
                    occ = occ * c[..., 0]                # Verdeckung ebenfalls nur dort glauben
            z = float(zs[i])
            zx, zy = cx + (xs - cx) / z, cy + (ys - cy) / z
            b = cv2.remap(bwd, zx, zy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = zx + b[..., 0], zy + b[..., 1]
            if transport is not None:  # Rauschfeld mit derselben Abbildung mitführen wie das Bild
                frisch = np.maximum(occ, rausch_frisch) if rausch_frisch > 0 else occ
                transport.advect(mx, my, frisch, rausch_rho)
            # neu aufgedeckte Flächen haben keinen Vorgänger: dort aus der Quelle starten
            occ = occ[..., None]

            def advect(img, mx=mx, my=my, occ=occ, inp=inp):
                wimg = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=core.BORDERS[border])
                return (occ * inp + (1.0 - occ) * wimg).astype(np.uint8)
            prev_in = inp
            prev_f = f_b
        else:
            def advect(img, i=i, depth=depth):
                return core.warp(img, zs[i], angs[i], txs[i], tys[i], border, depth, parallax)

        # Cadence wie in Deforum: nur jeder cad-te Frame geht durch den Sampler. `out` ist das jüngste
        # Sampler-Bild, `prev_d` das davor; beide werden zur aktuellen Pose getragen und überblendet.
        warped = advect(out)
        # Cadence wie in Deforum (jeder cad-te Frame), dazu erzwungene Samples auf dem Beat (Audio `takt_sample`):
        # dann startet der Morph vom zuletzt gezeigten Bild, nicht vom alten Sample
        sampeln = i - letztes_sample >= cad
        erzwungen = (not sampeln) and force is not None and bool(force[i])
        if sampeln or erzwungen:
            prev_d = advect(shown) if erzwungen else warped
        elif cad > 1:
            prev_d = advect(prev_d)
        if mem is not None:
            mem = advect(np.clip(mem, 0, 255).astype(np.uint8)).astype(np.float32)
        if sampeln or erzwungen:
            letztes_sample = i
            a_i = float(pf["alpha"][i])
            init = cv2.addWeighted(warped, 1.0 - a_i, inp, a_i, 0)
            init = core.hf_inject(init, inp, float(pf["hf"][i]), hf_sigma, hf_mode)
            w_i = float(pf["weich"][i]) if weich_modus == "init" else 0.0
            if w_i > 0.05:     # Verschwimmen: der Sampler zeichnet aus dem weichen Bild neu, Feinformen lösen sich
                init = cv2.GaussianBlur(init, (0, 0), w_i)
            ab_i = float(pf["antiblur"][i])
            if ab_i > 0:       # wie Deforums Anti-Blur: Resampling und Überblenden kosten sonst stetig Schärfe
                soft = cv2.GaussianBlur(init, (0, 0), 1.5)
                init = cv2.addWeighted(init, 1.0 + ab_i, soft, -ab_i, 0)
            nz_i = float(pf["noise"][i])
            if nz_i > 0:       # etwas Korn als Futter für den Sampler, pro Frame anders
                rng = np.random.default_rng(seed + i)
                init = np.clip(init.astype(np.float32) + rng.normal(0, nz_i * 255, init.shape), 0, 255).astype(np.uint8)
            d = sample(init, i, inp, ref=warped)
            if color_match == "input":
                # endlose Fahrt: die Quelle ist dann das eigene Bild – angeglichen wird an das Originalbild, sonst bleibt
                # jede kleine Abdunklung hängen (Langlauf 21.10.: nach ~25 s schwarz)
                d = core.match_color(d, first if len(anker) > 1 else inp)
            elif color_match == "first":
                d = core.match_color(d, first)
            elif color_match == "gen0":  # hält Kontrast/Farbe des Modells statt der (evtl. dunklen, flachen) Quelle
                if "_prompt" in kurven and kurven["_prompt"][i] != kurven["_prompt"][i - 1]:
                    gen0 = d.copy()   # Weltwechsel: neue Welt bringt ihre eigene Palette mit (sonst bleibt die erste)
                else:
                    d = core.match_color(d, gen0)
            if mono:
                d = cv2.cvtColor(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            d = core.licht_transfer(d, licht_quelle(i, inp), float(pf["licht"][i]), licht_sigma)
            gl_i = float(pf["glaettung"][i])
            if gl_i > 0:
                # Bewegungsgeführte Glättung gegen Brodeln: neues Sample mit dem entlang der Bewegung mitgetragenen
                # Vorbild mischen; wo Neues aufgedeckt wird (Verdeckung), gilt nur das neue Sample
                o = occ if (occ is not None and np.ndim(occ) == 3) else (occ[..., None] if occ is not None else 0.0)
                kk = gl_i * (1.0 - o)
                d = np.clip(kk * warped.astype(np.float32) + (1.0 - kk) * d.astype(np.float32), 0, 255).astype(np.uint8)
            sh_i = float(pf["schaerfe_halten"][i])
            if sh_i > 0:   # Schärfe halten: jedes Sample auf die Schärfe des ersten generierten Bildes zurückholen,
                # sonst schaukelt sich Weichheit über den Loop auf (gemessen: Laplace-Varianz 2600 -> 9 in 4 s)
                d = cv2.addWeighted(core.schaerfe_angleichen(d, gen0, max_gain=1.5), sh_i, d, 1.0 - sh_i, 0)
            if anker_zoom and len(anker) > 1:
                # Rauschbremse der endlosen Fahrt: wird das Sample deutlich körniger als das erste Bild, kantenerhaltend
                # entrauschen (Hochfrequenz schaukelt sich in tiefen Fahrten sonst über die Rückkopplung auf)
                lv = core.lap_var(d)
                if lv > 1.6 * lv0:
                    d = _entrauschen(d, h=int(np.clip(4 + 4 * (lv / lv0 - 1.6), 4, 10)))
            if mem is not None:
                if "_prompt" in kurven and i > 0 and kurven["_prompt"][i] != kurven["_prompt"][i - 1]:
                    # Weltwechsel: Gedächtnis beginnt mit der neuen Welt (Test 26.10.: sonst mischte es die alte Welt
                    # über Sekunden wieder ein – die Lava blieb blass, weil das Grobband noch Eis war)
                    mem = d.astype(np.float32)
                g = float(pf["gedaechtnis"][i])
                df = d.astype(np.float32)
                df = df - lp(df) + (1.0 - g) * lp(df) + g * lp(mem)   # Grobband aus dem Gedächtnis
                d = np.clip(df, 0, 255).astype(np.uint8)
                mem = (1.0 - ged_beta) * mem + ged_beta * d.astype(np.float32)
                if motion in ("flow", "kamera", "vektor", "virtuell") and occ is not None:  # bei Verdeckung neu anfangen
                    o = occ if occ.ndim == 3 else occ[..., None]
                    mem = o * d.astype(np.float32) + (1.0 - o) * mem
            out = core.apply_mask(d, mk)
            if beat_hart and force is not None and bool(force[i]):
                # Harter Schnitt auf den Beat: das Beat-Sample sofort voll zeigen statt über die Cadence einzublenden
                # (gemessen: im Beat-Frame war nur ein Drittel der Verwandlung sichtbar, voll erst 2 Frames später)
                prev_d = out.copy()
            if dis is not None:  # Fluss altes Sample (in dieser Pose) → neues Sample, einmal pro Sample
                ga, gb = cv2.cvtColor(prev_d, cv2.COLOR_BGR2GRAY), cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                morph_flow = dis.calc(gb, ga, None)  # für Pixel in b: wo liegt er in a → a→b-Verschiebung = -flow
                morph_flow = -morph_flow
        else:
            out = warped
        t = min(1.0, (i - letztes_sample + 1) / cad)
        frame = out if t >= 1.0 else morph_frame(prev_d, out, t)
        if t < 1.0 and pump_fix:  # Zwischenframes auf die Schärfe des Samples bringen (sonst Pumpen im Cadence-Takt)
            frame = core.schaerfe_angleichen(frame, out)
        shown = core.apply_mask(frame, mk)
        if weich_modus == "anzeige" and float(pf["weich"][i]) > 0.05:
            # Verschwimmen nur im ausgegebenen Bild: der Loop bleibt scharf, die Unschärfe klingt mit der Kurve ab
            shown = core.apply_mask(cv2.GaussianBlur(shown, (0, 0), float(pf["weich"][i])), mk)
            core.imwrite(fdir / f"{i:05d}.png", core.film_finish(shown, i, film, mono, seed))
            shown = core.apply_mask(frame, mk)   # weitergetragen wird das scharfe Bild
        else:
            core.imwrite(fdir / f"{i:05d}.png", core.film_finish(shown, i, film, mono, seed))
        if i % 10 == 0 or i == n - 1:
            log(f"[A] {i + 1}/{n}  {(time.time() - t0) / (i + 1):.2f} s/frame")

    if soll:
        np.savez_compressed(Path(out_dir) / "soll_fluss.npz", bwd=np.stack(soll).astype(np.float16))
    if comfy is not None:
        weg = comfy.uploads_aufraeumen()
        if weg:
            log(f"[A] removed {weg} temporary uploads from ComfyUI/input")
    meta = core.finish_run(out_dir, fps, {
        "backend": backend, "weich": weich, "weich_modus": weich_modus, "schaerfe_halten": schaerfe_halten,
        "glaettung": glaettung, "beat_hart": bool(beat_hart), "lora": lora, "referenz": referenz, "referenz_bloecke": referenz_bloecke, "audio": ({k: v for k, v in audio_info.items() if k not in ("beat_frames", "takt_frames")}
                                  if audio_info else None),
        "pro_frame": {nm: [round(float(pf[nm].min()), 4), round(float(pf[nm].max()), 4)] for nm in _pf_namen
                      if np.ptp(pf[nm]) > 0},
        "antiblur": antiblur, "noise": noise, "kanten": kanten, "quellzoom": str(quellzoom), "rausch": rausch,
        "gedaechtnis": gedaechtnis, "morph": bool(morph and cad > 1), "rausch_rho": rausch_rho, "pump_fix": bool(pump_fix), "mono": bool(mono),
        "quellzoom_gesamt": round(float(qz[-1]), 3),
        "paesse": sorted(seq.passes) if seq.passes else [],
        "method": "a", "dry_run": dry_run, "zoom": str(zoom), "alpha": alpha, "denoise": denoise,
        "steps": steps, "sampler_steps": real_steps, "cfg": cfg, "seed": seed, "ckpt": ckpt,
        "hf": hf, "hf_mode": hf_mode, "stride": stride, "control_gamma": control_gamma,
        "motion": motion, "flow_backend": flow_be.name if flow_be is not None else None,
        "seed_mode": seed_mode, "color_match": color_match, "cadence": cad, "wrap": bool(wrap),
        "control_end": control_end, "first_denoise": first_denoise, "licht": licht, "wandel": wandel,
        "controlnet": controlnet, "control_strength": control_strength if controlnet != "none" else None,
        "control_net_name": control_net_name, "mask": str(mask) if mask else None, "size": f"{w}x{h}", "sub": 1,
        "parallax": bool(parallax and seq.depth)}, t0)
    if audio and not dry_run and (Path(out_dir) / "video.mp4").exists():
        # Ton zeitrichtig anlegen: Quellzeit n·stride/basis_fps wird auf die Videolänge n/fps gezogen
        from . import audio as audio_mod
        bf = float(audio.get("basis_fps", 30))
        try:
            audio_mod.ton_anlegen(Path(out_dir) / "video.mp4", audio["datei"], Path(out_dir) / "video_ton.mp4",
                                  float(audio.get("offset", 0.0)) + start / bf, n * stride / bf, n / fps)
            meta["video_ton"] = str(Path(out_dir) / "video_ton.mp4")
        except Exception as e:  # noqa: BLE001
            log(f"[A] Adding the soundtrack failed: {str(e).splitlines()[0]}")
    return meta
