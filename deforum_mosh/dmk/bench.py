"""dmk.bench: Wege und Ketten auf derselben Sequenz laufen lassen, messen, bewerten, berichten."""
from __future__ import annotations

import importlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import core

# Ziel: Bildbereiche bleiben erhalten und fließen mit der Bewegung wie bei Deforum (Leitgröße, Nutzer
# 2026-10-08); dabei Detailschärfe und Räumlichkeit der Quelle halten, Bildinhalt wandeln.
WEIGHTS = {"fluss": 0.35, "raum": 0.20, "schaerfe": 0.15, "wandel": 0.10, "ruhe": 0.05, "kohaerenz": 0.05,
           "mosh": 0.05, "kosten": 0.05}
# Deforum-Referenz (seq\ref_bond): fluss_gewinn 0.32, flow_global 0.75, flow_jitter 0.32, erhalt_8 0.58
REF_FLUSS = {"fluss_gewinn": 0.32, "flow_global": 0.75, "flow_jitter": 0.32, "erhalt_8": 0.58}

# "from" hängt einen Lauf an die Ausgabe eines früheren (Kette); die Stufe zoomt dann nicht erneut.
DEFAULT_RUNS = [
    {"name": "A_dry", "method": "a", "dry_run": True},
    {"name": "A_dry_hf", "method": "a", "dry_run": True, "hf": 0.8},
    {"name": "A", "method": "a"},
    {"name": "D_depth", "method": "a", "controlnet": "depth", "hf": 0.8, "denoise": 0.5},
    {"name": "D_depth_par", "method": "a", "controlnet": "depth", "hf": 0.8, "denoise": 0.5, "parallax": 0.6},
    {"name": "D_canny", "method": "a", "controlnet": "canny", "hf": 0.8, "denoise": 0.5},
    {"name": "B_gop10", "method": "b", "gop": 10},
    {"name": "B_gop20_dup", "method": "b", "gop": 20, "dup_every": 12, "dup_times": 3},
    {"name": "C", "method": "c"},
    {"name": "C_hf", "method": "c", "hf": 0.8},
    {"name": "C_sub3_hold", "method": "c", "sub": 3, "occl": "hold", "hf": 0.8},
    {"name": "C_hf+B", "method": "b", "from": "C_hf", "gop": 20},
    {"name": "D_depth+B", "method": "b", "from": "D_depth", "gop": 20},
]

# Referenz: Deforum-Vorlage "james bond" (512 px, 15 fps, 248 Frames), gemessen 2026-10-08:
# ssim_prev 0.53, flow 2.4 px/F, jerk 0.05, warp_error 10.1, zoom 1.006/F, det_drift 1.04, Tiefe n vs n+4: 0.88
TARGETS = {
    "det_ratio": "0.9–3.0 (Schärfe relativ zur Quelle; darüber Schlieren)",
    "raum_tiefe": "> 0.9 (Tiefen-Korrelation gegen Quelle)", "kanten": "nur zur Info (Kanten-SSIM)",
    "ssim_ref": "0.4–0.7 (Inhalt wandelt sich, Quelle bleibt erkennbar)",
    "ssim_prev": "0.5–0.7 (Wandel pro Frame wie Deforum-Referenz 0.53; > 0.9 = eingefroren)",
    "flow_px": "1.5–3 px/Frame (Referenz 2.4)", "warp_error": "< 12 (Referenz 10)",
    "jerk": "< 0.15 (Referenz 0.05)", "zoom_std": "< 0.01", "blockiness": "1.15–1.80",
}


def _method(key: str):
    """Wege a-c sind eingebaut; d/e/f werden geladen, falls dmk.method_<x> existiert."""
    try:
        return importlib.import_module(f".method_{key}", __package__).run
    except ModuleNotFoundError:
        return None


def _corridor(x, lo, hi, zero_lo, zero_hi) -> float:
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return float(np.clip((x - zero_lo) / (lo - zero_lo), 0, 1))
    return float(np.clip((zero_hi - x) / (zero_hi - hi), 0, 1))


def score(m: dict, zoom_target: float, src: dict | None) -> dict:
    """Teilnoten 0..1 und gewichtete Summe. det_drift wird relativ zur Quelle bewertet, weil
    die Quelle selbst weit außerhalb des Korridors liegen kann."""
    det = m["det_drift"] / src["det_drift"] if src and src["det_drift"] > 1e-6 else m["det_drift"]
    ratio = m.get("det_ratio") if m.get("det_ratio") is not None else 1.0
    sim = m.get("ssim_ref") if m.get("ssim_ref") is not None else 0.55
    g = lambda key, default: m[key] if m.get(key) is not None else default
    if m.get("raum_tiefe") is not None:      # Tiefen-Korrelation: 0.95+ = Raum steht, < 0.6 = verloren
        raum_score = float(np.clip((m["raum_tiefe"] - 0.5) / 0.45, 0, 1))
    elif m.get("raum") is not None:          # Rückfall: Kanten-SSIM (bestraft neue Textur)
        raum_score = float(np.clip((m["raum"] - 0.2) / 0.5, 0, 1))
    else:
        raum_score = 1.0
    parts = {
        # Schärfe soll mindestens der Quelle entsprechen; mehr Textur ist erwünscht, nicht bestraft
        "schaerfe": _corridor(ratio, 0.9, 3.0, 0.3, 6.0),
        "raum": raum_score,
        # Inhalt soll sich wandeln: SSIM zur Quelle weder ~1 (Kopie) noch ~0 (Quelle verloren)
        "wandel": _corridor(sim, 0.4, 0.7, 0.1, 0.95),
        # Leitgröße: Änderung ist Bewegung (gewinn), die Bewegung ist eine klare Kamerafahrt (global),
        # das Flussfeld zappelt nicht (jitter), Bereiche überleben 8 Frames entlang der Bahn (erhalt_8)
        "fluss": 0.40 * float(np.clip(g("fluss_gewinn", 0.0) / REF_FLUSS["fluss_gewinn"], 0, 1))
                 + 0.25 * float(np.clip(g("flow_global", 0.0) / REF_FLUSS["flow_global"], 0, 1))
                 + 0.20 * float(np.clip(1 - (g("flow_jitter", 1.2) - REF_FLUSS["flow_jitter"]) / 0.9, 0, 1))
                 + 0.15 * _corridor(g("erhalt_8", 0.5), 0.45, 0.75, 0.2, 0.98),
        "ruhe": float(np.clip(1 - m["jerk"] / 0.4, 0, 1)),
        "kohaerenz": float(np.clip(1 - m["warp_error"] / 24.0, 0, 1)),
        "mosh": _corridor(m["blockiness"], 1.15, 1.80, 1.00, 3.00),
        "kosten": float(1.0 / (1.0 + m.get("sec_per_frame", 0.0) / 2.0)),
    }
    parts["gesamt"] = float(sum(WEIGHTS[k] * parts[k] for k in WEIGHTS))
    parts["det_rel"] = float(det)
    return parts


# Zielprofil für die Suche: (voll erfüllt von, bis, null bei unten, null bei oben, Gewicht).
# Fluss-/Erhalt-Werte = Deforum-Referenz seq\ref_bond; Raum, Schärfe, Wandel, Licht = Projektziele.
ZIELE = {
    "mitfluss":     (0.60, 1.00, -0.20, 1.01, 0.20),  # Inhalt wandert mit der Szene statt am Bildschirm zu kleben
    "det_drift":    (0.85, 1.25, 0.40, 2.00, 0.08),   # Schärfe hält über die Laufzeit
    "fluss_gewinn": (0.28, 0.50, 0.00, 0.90, 0.22),
    "flow_jitter":  (0.00, 0.35, -0.01, 1.20, 0.12),
    "flow_global":  (0.70, 1.00, 0.20, 1.01, 0.08),
    "erhalt_1":     (0.88, 0.97, 0.60, 1.00, 0.12),   # über 0.97 steht das Bild
    "erhalt_8":     (0.50, 0.75, 0.20, 0.98, 0.08),
    "raum_tiefe":   (0.95, 1.00, 0.60, 1.01, 0.14),
    "det_ratio":    (1.20, 3.50, 0.40, 8.00, 0.10),
    "ssim_ref":     (0.30, 0.60, 0.05, 0.95, 0.06),
    "licht":        (0.75, 1.00, 0.20, 1.01, 0.08),
}


def ziel(m: dict) -> dict:
    """Erfüllungsgrad 0..1 je Zielwert und gewichtete Summe 'ziel' (1.0 = alle Zielkorridore getroffen)."""
    parts, total, wsum = {}, 0.0, 0.0
    for key, (lo, hi, zlo, zhi, w) in ZIELE.items():
        if m.get(key) is None:
            continue
        parts[key] = _corridor(m[key], lo, hi, zlo, zhi)
        total += w * parts[key]
        wsum += w
    parts["ziel"] = total / wsum if wsum else 0.0
    return parts


def measure(out_frames, ref_frames, ref_depth, rdir, sub: int = 1, log=print, mask=None) -> dict:
    """Alle Kennzahlen eines Laufs gegen seine Referenzframes (bei stride: Ausgabe i ↔ Quelle i*stride).
    mask: gültiger Bildbereich für raum_tiefe (Projektionsmapping), sonst aus run.json/params.json."""
    rdir = Path(rdir)
    if mask is None:
        for name in ("params.json", "run.json"):
            p = rdir / name
            if p.exists():
                try:
                    mask = json.loads(p.read_text(encoding="utf-8")).get("mask") or None
                except json.JSONDecodeError:
                    mask = None
                if mask:
                    break
    m = core.analyse(out_frames, ref_frames, sub)
    m.update(core.analyse_fluss(out_frames[:120]))
    m.update(core.punkt_leben(out_frames[:120]))
    m["fluss_gewinn"] = core.fluss_gewinn(m)
    m["mitfluss"] = core.mitfluss(out_frames, Path(rdir) / "soll_fluss.npz")
    m["raum_tiefe"] = None
    if ref_depth:
        try:
            from . import depth as depth_mod
            m["raum_tiefe"] = depth_mod.raum_tiefe(out_frames, ref_depth, rdir / "_depth", sub, log=log, mask=mask)
        except Exception as e:
            log(f"[measure] raum_tiefe fehlgeschlagen ({type(e).__name__}: {e})")
    return m


def _zoom_target(zoom, n: int, sub: int) -> float:
    """Geometrisches Mittel des Schedules, für Sub-Frames auf die Ausgabe-Framerate normiert."""
    zs = core.schedule(zoom, n)[1:]
    return float(np.exp(np.log(zs).mean()) ** (1.0 / max(1, sub))) if len(zs) else 1.0


def run(seq_root, out_dir, count=90, start=0, zoom="0:(1.02)", max_side=768, fps=24,
        runs=None, server="127.0.0.1:8188", pattern=None, log=print, **common):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = runs or DEFAULT_RUNS
    seq = core.load_sequence(seq_root, count, start, pattern)
    common.setdefault("depth_cache", out_dir / "_depth")
    log(f"[bench] Quelle: {seq.root}  ({len(seq)} Frames, Tiefe: {'ja' if seq.depth else 'nein'})")

    src = core.analyse(seq.frames)
    src["sec_per_frame"] = 0.0
    src_depth = None
    try:  # Tiefe der Quelle (vorhanden oder geschätzt) für die Raum-Kennzahl
        from . import depth as depth_mod
        src_depth = depth_mod.ensure(seq, common["depth_cache"], log=log).depth
    except Exception as e:
        log(f"[bench] Keine Quelltiefe, raum_tiefe entfällt ({type(e).__name__}: {e})")
    results = [{"name": "QUELLE", "status": "ok", "metrics": src, "sub": 1,
                "score": score(src, 1.0, None), "zoom_target": None, "dir": str(seq.root)}]
    done: dict[str, dict] = {}

    for spec in runs:
        spec = dict(spec)
        name, key, parent = spec.pop("name"), spec.pop("method"), spec.pop("from", None)
        fn = _method(key)
        rdir = out_dir / name
        res = {"name": name, "dir": str(rdir)}
        results.append(res)
        if fn is None:
            res.update(status="übersprungen", reason=f"Modul dmk.method_{key} fehlt")
            log(f"[bench] {name}: {res['reason']}")
            continue
        if parent and parent not in done:
            res.update(status="übersprungen", reason=f"Vorstufe {parent} fehlt")
            log(f"[bench] {name}: {res['reason']}")
            continue
        kw = dict(common, max_side=max_side, fps=fps, server=server, log=log)
        if parent:  # Kette: volle Länge der Vorstufe, Zoom steckt schon drin
            pre = done[parent]
            kw.update(seq_root=pre["frames_dir"], count=0, start=0, zoom="0:(1.0)", pattern=None)
            kw["fps"] = fps * pre.get("sub", 1)
            sub_in, eff_zoom = pre.get("sub", 1), pre["eff_zoom"]
        else:
            kw.update(seq_root=seq_root, count=count, start=start, zoom=zoom, pattern=pattern)
            sub_in, eff_zoom = 1, zoom
        kw.update(spec)
        # Referenz passend zum Lauf: bei stride > 1 gehört Ausgabeframe i zu Quellframe i*stride
        if parent:
            ref_frames, ref_depth = done[parent]["ref"]
        else:
            eff_zoom = kw["zoom"]
            st = int(kw.get("stride", 1) or 1)
            ref_frames, ref_depth = seq.frames, src_depth
            if st > 1:
                rs = core.load_sequence(seq_root, count, start, pattern, stride=st, wrap=bool(kw.get("wrap")))
                ref_frames, ref_depth = rs.frames, None
                try:
                    ref_depth = depth_mod.ensure(rs, common["depth_cache"], log=log).depth
                except Exception as e:
                    log(f"[bench] Keine Referenztiefe für stride {st} ({type(e).__name__}: {e})")
        log(f"[bench] ── {name} ──")
        try:
            meta = fn(out_dir=rdir, **kw)
        except Exception as e:
            res.update(status="fehlgeschlagen", reason=f"{type(e).__name__}: {str(e).splitlines()[0]}")
            (rdir).mkdir(parents=True, exist_ok=True)
            (rdir / "fehler.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[bench] {name} fehlgeschlagen: {res['reason']}")
            continue
        sub_total = sub_in * meta.get("sub", 1)
        meta.update(sub=sub_total, eff_zoom=eff_zoom, ref=(ref_frames, ref_depth))
        if parent:  # Kosten der Kette = Summe der Stufen
            meta["seconds"] += done[parent]["seconds"]
            meta["sec_per_frame"] = round(meta["seconds"] / max(1, meta["frames"]), 3)
        done[name] = meta
        out_frames = core.list_images(Path(meta["frames_dir"]))
        m = core.analyse(out_frames, ref_frames, sub_total)
        m.update(core.analyse_fluss(out_frames[:120]))
        m["fluss_gewinn"] = core.fluss_gewinn(m)
        m["sec_per_frame"] = meta["sec_per_frame"]
        m["raum_tiefe"] = None
        if ref_depth:
            try:
                m["raum_tiefe"] = depth_mod.raum_tiefe(out_frames, ref_depth, rdir / "_depth", sub_total, log=log)
            except Exception as e:
                log(f"[bench] raum_tiefe für {name} fehlgeschlagen ({type(e).__name__}: {e})")
        zt = _zoom_target(eff_zoom, len(seq), sub_total)
        res.update(status="ok", metrics=m, sub=sub_total, zoom_target=zt, score=score(m, zt, src),
                   seconds=meta["seconds"],
                   meta={k: v for k, v in meta.items() if k not in ("frames_dir", "ref")})

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(out_dir, results, seq, zoom, count)
    log(f"[bench] Bericht: {report}")
    return results


def rescore(seq_root, out_dir, count=90, start=0, zoom="0:(1.02)", pattern=None, depth_cache=None, log=print):
    """Kennzahlen, Wertung und Bericht aus vorhandenen Laufordnern neu berechnen, ohne zu rendern.
    Für Ketten gilt der Sub-Faktor und Zoom aus run.json der Stufe selbst (Näherung)."""
    out_dir = Path(out_dir)
    seq = core.load_sequence(seq_root, count, start, pattern)
    src_depth = None
    try:
        from . import depth as depth_mod
        src_depth = depth_mod.ensure(seq, depth_cache or out_dir / "_depth", log=log).depth
    except Exception as e:
        log(f"[rescore] Keine Quelltiefe ({type(e).__name__}: {e})")
    src = core.analyse(seq.frames)
    src["sec_per_frame"] = 0.0
    results = [{"name": "QUELLE", "status": "ok", "metrics": src, "sub": 1,
                "score": score(src, 1.0, None), "zoom_target": None, "dir": str(seq.root)}]
    for rdir in sorted(p for p in out_dir.iterdir() if p.is_dir() and (p / "run.json").exists()):
        meta = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        frames = core.list_images(rdir / "frames")
        if len(frames) < 2:
            continue
        sub = int(meta.get("sub", 1))
        log(f"[rescore] {rdir.name} ({len(frames)} Frames)")
        st = int(meta.get("stride", 1) or 1)
        ref_frames, ref_depth = seq.frames, src_depth
        if st > 1:  # Ausgabeframe i gehört zu Quellframe i*stride
            rs = core.load_sequence(seq_root, count, start, pattern, stride=st, wrap=bool(meta.get("wrap")))
            ref_frames, ref_depth = rs.frames, None
            try:
                ref_depth = depth_mod.ensure(rs, depth_cache or out_dir / "_depth", log=log).depth
            except Exception:
                pass
        m = core.analyse(frames, ref_frames, sub)
        m.update(core.analyse_fluss(frames[:120]))
        m["fluss_gewinn"] = core.fluss_gewinn(m)
        m["sec_per_frame"] = meta.get("sec_per_frame", 0.0)
        m["raum_tiefe"] = None
        if ref_depth:
            try:
                m["raum_tiefe"] = depth_mod.raum_tiefe(frames, ref_depth, rdir / "_depth", sub, log=log)
            except Exception as e:
                log(f"[rescore] raum_tiefe für {rdir.name} fehlgeschlagen ({type(e).__name__}: {e})")
        run_zoom = meta.get("zoom", zoom)
        zt = _zoom_target(zoom if run_zoom == "0:(1.0)" else run_zoom, len(seq), sub)
        results.append({"name": rdir.name, "dir": str(rdir), "status": "ok", "metrics": m, "sub": sub,
                        "zoom_target": zt, "score": score(m, zt, src), "seconds": meta.get("seconds"),
                        "meta": {k: v for k, v in meta.items() if k != "frames_dir"}})
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(out_dir, results, seq, zoom, count)
    log(f"[rescore] Bericht: {report}")
    return results


def write_report(out_dir: Path, results: list[dict], seq, zoom, count) -> Path:
    ok = [r for r in results if r["status"] == "ok"]
    ranked = sorted((r for r in ok if r["name"] != "QUELLE"), key=lambda r: -r["score"]["gesamt"])
    f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"
    L = [f"# dmk Bench  ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"Quelle: `{seq.root}`, {len(seq)} Frames, Zoom-Schedule `{zoom}`", "",
         "## Kennzahlen", "",
         "| Lauf | Frames | fluss_gewinn | flow_global | flow_jitter | erhalt_1 | erhalt_8 | det_ratio | raum_tiefe | kanten | SSIM_ref | ssim_prev | flow_px | warp_error | jerk | zoom/F | Soll | zoom_std | det_drift | blockiness | smear | s/Frame |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| *Deforum-Referenz* | 248 | {REF_FLUSS['fluss_gewinn']} | {REF_FLUSS['flow_global']} | {REF_FLUSS['flow_jitter']} "
         f"| 0.90 | {REF_FLUSS['erhalt_8']} | – | – | – | – | 0.58 | 2.03 | 10.1 | 0.076 | 1.006 | – | 0.002 | 1.04 | 1.05 | 0.13 | – |"]
    for r in ok:
        m, s = r["metrics"], r["score"]
        L.append(f"| {r['name']} | {m['frames']} | {f(m.get('fluss_gewinn'))} | {f(m.get('flow_global'))} | "
                 f"{f(m.get('flow_jitter'))} | {f(m.get('erhalt_1'))} | {f(m.get('erhalt_8'))} | "
                 f"{f(m.get('det_ratio'), 2)} | {f(m.get('raum_tiefe'))} | "
                 f"{f(m.get('raum'))} | {f(m['ssim_ref'])} | {f(m.get('ssim_prev'))} | {f(m.get('flow_px'), 2)} | "
                 f"{f(m['warp_error'], 2)} | {f(m['jerk'])} | {f(m['zoom'], 4)} | {f(r['zoom_target'], 4)} | "
                 f"{f(m['zoom_std'], 4)} | {f(m['det_drift'])} | {f(m['blockiness'])} | {f(m['smear'])} | "
                 f"{f(m['sec_per_frame'])} |")
    L += ["", "Zielkorridore: " + "; ".join(f"{k} {v}" for k, v in TARGETS.items()) + ".", "",
          "## Wertung", "",
          "Gewichte: " + ", ".join(f"{k} {v}" for k, v in WEIGHTS.items()), "",
          "| Rang | Lauf | gesamt | " + " | ".join(WEIGHTS) + " |", "|---|---|---|" + "---|" * len(WEIGHTS)]
    for i, r in enumerate(ranked, 1):
        s = r["score"]
        L.append(f"| {i} | {r['name']} | **{s['gesamt']:.3f}** | " + " | ".join(f"{s[k]:.2f}" for k in WEIGHTS) + " |")
    L += ["", "> Die Wertung ist ein Wegweiser, kein Urteil: `mosh` belohnt Blockartefakte, `smear` steigt auch "
              "bei bloßem Einfrieren, und `schaerfe` kann durch Rauschen oder Blockkanten steigen, ohne dass echte "
              "Details da sind. Vor jeder Entscheidung Kontaktbögen und MP4s ansehen.", "",
          "## Sichtprüfung", ""]
    for r in ok:
        if r["name"] == "QUELLE":
            continue
        L += [f"### {r['name']}", "", f"![{r['name']}]({r['name']}/kontakt.png)", "",
              f"[video.mp4]({r['name']}/video.mp4)", ""]
    skipped = [r for r in results if r["status"] != "ok"]
    if skipped:
        L += ["## Nicht gelaufen", ""] + [f"- **{r['name']}**: {r['status']}, {r.get('reason', '')}" for r in skipped]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
