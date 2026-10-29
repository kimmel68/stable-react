"""dmk.method_b: echter Datamosh über den MPEG-4-Codec.

Ablauf: Frames (mit Zoom) -> AVI/MPEG-4 ohne B-Frames -> RIFF parsen, VOP-Typen lesen ->
Mosh-Operatoren auf der Frameliste -> AVI neu schreiben -> mit ffmpeg dekodieren.

Der Zoom läuft VOR dem Encoding, damit er in den Motion-Vectors steckt. Er wird pro GOP
kumuliert und am I-Frame zurückgesetzt; da die I-Frames entfernt werden, bleibt nur die
Zoombewegung der P-Frames übrig und läuft im Mosh endlos weiter.
"""
from __future__ import annotations

import random
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import core, proc

VOP_START = b"\x00\x00\x01\xb6"
VOP_TYPES = "IPBS"


@dataclass
class Frame:
    fourcc: bytes
    data: bytes
    type: str  # I, P, B, S oder ? (kein VOP gefunden)


def vop_type(data: bytes) -> str:
    i = data.find(VOP_START)
    if i < 0 or i + 4 >= len(data):
        return "?"
    return VOP_TYPES[data[i + 4] >> 6]


# ---------------------------------------------------------------- RIFF/AVI

def _chunks(buf: bytes, pos: int, end: int):
    """Liefert (fourcc, daten_start, groesse, listentyp) für jede Chunk-Ebene zwischen pos und end."""
    while pos + 8 <= end:
        fcc = buf[pos:pos + 4]
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        ltype = buf[pos + 8:pos + 12] if fcc in (b"LIST", b"RIFF") else None
        yield fcc, pos + 8, size, ltype
        pos += 8 + size + (size & 1)


def read_avi(path) -> tuple[bytearray, list[Frame]]:
    """Gibt den hdrl-LIST-Chunk (komplett, änderbar) und die Videoframes aus movi zurück."""
    buf = Path(path).read_bytes()
    if buf[:4] != b"RIFF" or buf[8:12] != b"AVI ":
        raise ValueError(f"Keine AVI-Datei: {path}")
    hdrl, frames = None, []
    riff_end = min(len(buf), 8 + struct.unpack_from("<I", buf, 4)[0])
    for fcc, start, size, ltype in _chunks(buf, 12, riff_end):
        if ltype == b"hdrl":
            hdrl = bytearray(buf[start - 8:start + size])
        elif ltype == b"movi":
            stack = [(start + 4, start + size)]
            while stack:
                a, b = stack.pop()
                for f2, s2, z2, l2 in _chunks(buf, a, b):
                    if l2 == b"rec ":
                        stack.append((s2 + 4, s2 + z2))
                    elif f2[2:4] in (b"dc", b"db"):
                        data = buf[s2:s2 + z2]
                        frames.append(Frame(f2, data, vop_type(data)))
    if hdrl is None or not frames:
        raise ValueError(f"AVI ohne hdrl oder Videoframes: {path}")
    return hdrl, frames


def _patch_hdrl(hdrl: bytearray, n: int) -> None:
    """Framezahl in avih/strh/dmlh setzen und OpenDML-Indizes stilllegen (zeigen sonst ins alte movi)."""
    stack = [(12, len(hdrl))]
    while stack:
        a, b = stack.pop()
        for fcc, start, size, ltype in _chunks(hdrl, a, b):
            if fcc == b"LIST":
                stack.append((start + 4, start + size))
            elif fcc == b"avih":
                struct.pack_into("<I", hdrl, start + 16, n)
            elif fcc == b"strh" and hdrl[start:start + 4] == b"vids":
                struct.pack_into("<I", hdrl, start + 32, n)
            elif fcc == b"dmlh":
                struct.pack_into("<I", hdrl, start, n)
            elif fcc == b"indx":
                hdrl[start - 8:start - 4] = b"JUNK"


def write_avi(path, hdrl: bytearray, frames: list[Frame]) -> None:
    hdrl = bytearray(hdrl)
    _patch_hdrl(hdrl, len(frames))
    movi, idx = bytearray(b"movi"), bytearray()
    for i, f in enumerate(frames):
        flags = 0x10 if f.type == "I" else 0
        idx += f.fourcc + struct.pack("<III", flags, len(movi), len(f.data))
        movi += f.fourcc + struct.pack("<I", len(f.data)) + f.data
        if len(f.data) & 1:
            movi += b"\x00"
    body = (b"AVI " + bytes(hdrl) + b"LIST" + struct.pack("<I", len(movi)) + bytes(movi)
            + b"idx1" + struct.pack("<I", len(idx)) + bytes(idx))
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


# ---------------------------------------------------------------- Mosh-Operatoren

def op_idrop(frames: list[Frame], mode: str = "replace") -> list[Frame]:
    """I-Frames nach dem ersten entfernen. replace: durch den vorherigen P-Frame ersetzen
    (Länge bleibt), remove: ersatzlos streichen, keep: nichts tun."""
    if mode == "keep":
        return list(frames)
    out = [frames[0]]
    for f in frames[1:]:
        if f.type != "I":
            out.append(f)
        elif mode == "replace":
            prev = next((g for g in reversed(out) if g.type == "P"), None)
            if prev is not None:
                out.append(prev)
    return out


def op_loop(frames: list[Frame], start: int, length: int, times: int) -> list[Frame]:
    """Segment aus P-Frames mehrfach abspielen; die Gesamtlänge bleibt erhalten."""
    if times < 2 or length < 1 or start < 1:
        return frames
    seg = [f for f in frames[start:start + length] if f.type == "P"]
    out = frames[:start + length] + seg * (times - 1) + frames[start + length:]
    return out[:len(frames)]


def op_dup(frames: list[Frame], every: int, times: int) -> list[Frame]:
    """Jeden every-ten P-Frame times-mal wiederholen (Bloom); die Folgeframes fallen dafür weg."""
    if every < 1 or times < 1:
        return frames
    out = list(frames)
    for i in range(every, len(out), every):
        if out[i].type == "P":
            for k in range(i + 1, min(len(out), i + 1 + times)):
                out[k] = out[i]
    return out


def op_jitter(frames: list[Frame], prob: float, seed: int = 0, reach: int = 3) -> list[Frame]:
    """Mit Wahrscheinlichkeit prob einen P-Frame durch einen Nachbarn (+-reach) ersetzen."""
    if prob <= 0:
        return frames
    rng = random.Random(seed)
    out = list(frames)
    for i in range(1, len(frames)):
        if frames[i].type == "P" and rng.random() < prob:
            j = min(len(frames) - 1, max(1, i + rng.randint(-reach, reach)))
            if frames[j].type == "P":
                out[i] = frames[j]
    return out


# ---------------------------------------------------------------- ffedit (optional)

_FFEDIT_JS = """\
// Motion-Vector-Gain. Die JS-API von ffglitch ändert sich zwischen Releases;
// diese Fassung nutzt nur Indexzugriffe, die in 0.9 und 0.10 vorhanden sind.
const GAIN = %GAIN%;
export function setup(args) { args.features = ["mv"]; }
export function glitch_frame(frame) {
  const fwd = frame.mv && frame.mv.forward;
  if (!fwd) return;
  for (let i = 0; i < fwd.length; i++) {
    const row = fwd[i];
    for (let j = 0; j < row.length; j++) {
      const mv = row[j];
      if (!mv) continue;
      mv[0] = Math.round(mv[0] * GAIN);
      mv[1] = Math.round(mv[1] * GAIN);
    }
  }
}
"""


def mv_gain(avi_in: Path, avi_out: Path, gain: float, log=print) -> bool:
    """Motion-Vectors mit ffedit skalieren. Gibt False zurück, wenn ffedit fehlt oder scheitert."""
    ffedit = proc.tool("ffedit", required=False)
    if ffedit is None:
        log("[B] ffedit nicht im PATH, --mv-gain wird übersprungen.")
        return False
    script = avi_in.with_suffix(".mvgain.js")
    script.write_text(_FFEDIT_JS.replace("%GAIN%", repr(float(gain))), encoding="utf-8")
    try:
        proc.run([ffedit, "-i", avi_in, "-f", "mv", "-s", script, "-o", avi_out, "-y"], "ffedit")
    except proc.ProcError as e:
        log(f"[B] ffedit gescheitert, --mv-gain wird übersprungen:\n{e}")
        return False
    return True


# ---------------------------------------------------------------- Lauf

def encode_avi(seq, zs, avi_path, max_side, gop, qscale, fps, border) -> tuple[int, int]:
    first = core.prep(core.imread(seq.frames[0]), max_side, 16)
    h, w = first.shape[:2]
    cmd = [proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
           "-c:v", "mpeg4", "-vtag", "xvid", "-bf", "0", "-g", str(gop), "-sc_threshold", "1000000000",
           "-qscale:v", str(qscale), "-pix_fmt", "yuv420p", "-an", "-f", "avi", str(avi_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        z = 1.0
        for i, path in enumerate(seq.frames):
            img = core.prep(core.imread(path), max_side, 16)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            z = 1.0 if i % gop == 0 else z * float(zs[i])
            if abs(z - 1.0) > 1e-6:
                img = core.warp(img, z, border=border)
            p.stdin.write(img.tobytes())
        p.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    err = p.stderr.read().decode("utf-8", "replace")
    if p.wait() != 0:
        raise proc.ProcError(f"MPEG-4-Encoding fehlgeschlagen:\n{err[-1500:]}")
    return w, h


def run(seq_root, out_dir, count=0, start=0, zoom="0:(1.0)", max_side=768, fps=24,
        gop=20, qscale=4, idrop="replace", dup_every=0, dup_times=0,
        loop_start=0, loop_len=0, loop_times=0, jitter=0.0, seed=0,
        mv_gain_factor=1.0, border="reflect", pattern=None, log=print, **_):
    t0 = time.time()
    out_dir = Path(out_dir)
    seq = core.load_sequence(seq_root, count, start, pattern)
    zs = core.schedule(zoom, len(seq))
    fdir = core.clear_frames(out_dir)

    src, mosh = out_dir / "quelle.avi", out_dir / "mosh.avi"
    w, h = encode_avi(seq, zs, src, max_side, gop, qscale, fps, border)
    if mv_gain_factor != 1.0:
        gained = out_dir / "quelle_mv.avi"
        if mv_gain(src, gained, mv_gain_factor, log):
            src = gained

    hdrl, frames = read_avi(src)
    types_in = "".join(f.type for f in frames)
    frames = op_idrop(frames, idrop)
    frames = op_loop(frames, loop_start, loop_len, loop_times)
    frames = op_dup(frames, dup_every, dup_times)
    frames = op_jitter(frames, jitter, seed)
    write_avi(mosh, hdrl, frames)
    log(f"[B] {len(types_in)} Frames kodiert ({types_in.count('I')} I, {types_in.count('P')} P), "
        f"{len(frames)} nach Mosh ({sum(f.type == 'I' for f in frames)} I)")

    proc.run([proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", mosh,
              *proc.sync_args(), "-start_number", "0", fdir / "%05d.png"], "Mosh-Dekodierung")

    return core.finish_run(out_dir, fps, {
        "method": "b", "zoom": str(zoom), "gop": gop, "qscale": qscale, "idrop": idrop,
        "dup_every": dup_every, "dup_times": dup_times, "jitter": jitter,
        "i_frames_in": types_in.count("I"), "size": f"{w}x{h}", "sub": 1}, t0)
