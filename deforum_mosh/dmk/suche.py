"""dmk.suche: systematische Suche über die Stellschrauben von Weg D gegen das Zielprofil bench.ZIELE.

Koordinatensuche: von einer Basis aus wird je Achse jeder Wert einzeln getestet; der beste Wert wird
übernommen, wenn er die Zielerfüllung verbessert, dann folgt die nächste Achse. Jeder Lauf wird in
suche.json festgehalten (Schlüssel = Parameter), ein Neustart rechnet nur Fehlendes nach.
Ergebnis: SUCHE.md mit Rangliste, Wirkung je Achse und der besten Einstellung.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import bench, core, method_a

ZEIGE = ("mitfluss", "det_drift", "fluss_gewinn", "flow_jitter", "flow_global", "erhalt_1", "erhalt_8", "raum_tiefe", "det_ratio",
         "ssim_ref", "licht", "textur", "helligkeit", "flow_px")


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def run(out_dir, seq: dict, base: dict, achsen: list[tuple[str, list]], count: int = 72, max_side: int = 768,
        runden: int = 1, server: str = "127.0.0.1:8188", depth_cache=None, min_gewinn: float = 0.004, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "suche.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"runs": {}, "verlauf": []}
    refs: dict = {}

    def ref_for(params):
        k = (int(params.get("stride", 1) or 1), bool(params.get("wrap")))
        if k not in refs:
            rs = core.load_sequence(seq["seq_root"], count, seq.get("start", 0), seq.get("pattern"),
                                    stride=k[0], wrap=k[1])
            try:
                from . import depth as depth_mod
                depth_mod.ensure(rs, depth_cache or out_dir / "_depth", log=log)
            except Exception as e:
                log(f"[suche] keine Referenztiefe ({type(e).__name__}: {e})")
            refs[k] = rs
        return refs[k]

    def evaluate(params: dict, label: str) -> dict:
        key = _key(params)
        if key in state["runs"]:
            return state["runs"][key]
        rdir = out_dir / label
        log(f"[suche] ── {label} ──")
        try:
            meta = method_a.run(out_dir=rdir, count=count, max_side=max_side, server=server,
                                depth_cache=depth_cache or out_dir / "_depth", log=log, **seq, **params)
            qdir = rdir / "quelle"
            if qdir.is_dir() and len(core.list_images(qdir)) >= 2:
                # digitale Kamerafahrt: Referenz ist die tatsächlich benutzte (gezoomte) Quelle des Laufs
                ref_frames = core.list_images(qdir)
                ref_depth = core.list_images(rdir / "quelle_depth") or None
            else:
                rs = ref_for(params)
                ref_frames, ref_depth = rs.frames, rs.depth
            m = bench.measure(core.list_images(rdir / "frames"), ref_frames, ref_depth, rdir, 1, log)
            z = bench.ziel(m)
            res = {"label": label, "params": params, "ziel": z["ziel"], "teile": z,
                   "metrics": {k: m.get(k) for k in ZEIGE}, "sec_per_frame": meta["sec_per_frame"]}
        except Exception as e:
            log(f"[suche] {label} fehlgeschlagen: {type(e).__name__}: {str(e).splitlines()[0]}")
            res = {"label": label, "params": params, "ziel": 0.0, "fehler": f"{type(e).__name__}: {e}"[:400]}
        state["runs"][key] = res
        state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        report(out_dir, state, achsen)
        return res

    best = dict(base)
    best_res = evaluate(best, "basis")
    for runde in range(runden):
        for achse, werte in achsen:
            kandidaten = []
            for w in werte:
                if best.get(achse) == w:
                    continue
                label = re.sub(r"[^A-Za-z0-9_]+", "", f"r{runde + 1}_{achse}_{w}")  # auch ":" und "(" raus (Windows-Pfad)
                kandidaten.append(evaluate(dict(best, **{achse: w}), label))
            top = max(kandidaten, key=lambda r: r["ziel"], default=None)
            if top and top["ziel"] > best_res["ziel"] + min_gewinn:
                state["verlauf"].append({"achse": achse, "von": best.get(achse), "nach": top["params"].get(achse),
                                         "ziel_vorher": best_res["ziel"], "ziel_nachher": top["ziel"]})
                log(f"[suche] {achse}: {best.get(achse)} → {top['params'].get(achse)}  "
                    f"(Ziel {best_res['ziel']:.3f} → {top['ziel']:.3f})")
                best, best_res = dict(top["params"]), top
            else:
                state["verlauf"].append({"achse": achse, "von": best.get(achse), "nach": best.get(achse),
                                         "ziel_vorher": best_res["ziel"], "ziel_nachher": best_res["ziel"]})
            state["best"] = {"params": best, "ziel": best_res["ziel"], "label": best_res["label"]}
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
            report(out_dir, state, achsen)
    return state


def report(out_dir: Path, state: dict, achsen) -> Path:
    f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"
    runs = sorted(state["runs"].values(), key=lambda r: -r["ziel"])
    L = [f"# dmk Suche  ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Zielprofil: " + "; ".join(f"{k} {lo}–{hi}" for k, (lo, hi, *_r) in bench.ZIELE.items()), "",
         "| Rang | Lauf | Ziel | " + " | ".join(ZEIGE) + " | s/Frame |", "|---|---|---|" + "---|" * (len(ZEIGE) + 1)]
    for i, r in enumerate(runs, 1):
        if "metrics" not in r:
            L.append(f"| {i} | {r['label']} | Fehler: {r.get('fehler', '')[:80]} |")
            continue
        m = r["metrics"]
        L.append(f"| {i} | {r['label']} | **{r['ziel']:.3f}** | " + " | ".join(
            f(m.get(k), 0 if k in ("textur", "helligkeit") else 3) for k in ZEIGE) + f" | {f(r['sec_per_frame'], 2)} |")
    if state.get("verlauf"):
        L += ["", "## Wirkung je Achse", "", "| Achse | von | nach | Ziel vorher | Ziel nachher |", "|---|---|---|---|---|"]
        L += [f"| {v['achse']} | {v['von']} | {v['nach']} | {v['ziel_vorher']:.3f} | {v['ziel_nachher']:.3f} |"
              for v in state["verlauf"]]
    if state.get("best"):
        L += ["", f"## Beste Einstellung: `{state['best']['label']}` (Ziel {state['best']['ziel']:.3f})", "", "```json",
              json.dumps(state["best"]["params"], indent=2, ensure_ascii=False), "```",
              "", f"![beste]({state['best']['label']}/kontakt.png)"]
    p = out_dir / "SUCHE.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
