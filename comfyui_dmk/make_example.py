"""Erzeugt den Beispiel-Workflow example_workflows/dmk_beat_grammar.json (ComfyUI-UI-Format) und die API-Fassung für
den Test über /prompt. Ablauf: Load Audio → DMK Audio to Beats → DMK Beat Grammar → Plot + Schedules + Prompts on the Bar."""
from __future__ import annotations

import json
from pathlib import Path

from dmk_beat import DEFAULT_GRAMMAR

HIER = Path(__file__).resolve().parent
AUDIO = "signal_decay_20s.m4a"
PROMPTS = ("brutalist concrete cathedral, raw board-formed concrete, deep coffered ceiling\n"
           "copperplate engraving of a vast basilica interior, cross-hatched arcades\n"
           "cathedral of fungal mycelium, hyphae ribbed vaults, soft bioluminescence")

# id: (type, pos, widgets, inputs [(name, type, (src_id, src_slot))], outputs [(name, type)])
NODES = {
    1: ("LoadAudio", (40, 80), [AUDIO], [], [("AUDIO", "AUDIO")]),
    2: ("DMKAudioBeats", (380, 80), [30.0, 0, 0.0, False, 60.0, 140.0], [("audio", "AUDIO", (1, 0))],
        [("beats", "DMK_BEATS"), ("info", "STRING")]),
    3: ("DMKBeatGrammar", (760, 80), [DEFAULT_GRAMMAR], [("beats", "DMK_BEATS", (2, 0))],
        [("curves", "DMK_CURVES"), ("summary", "STRING")]),
    4: ("DMKPlotCurves", (1180, 80), [], [("beats", "DMK_BEATS", (2, 0)), ("curves", "DMK_CURVES", (3, 0))],
        [("IMAGE", "IMAGE")]),
    5: ("PreviewImage", (1500, 80), [], [("images", "IMAGE", (4, 0))], []),
    6: ("DMKCurveSchedule", (1180, 420), ["zoom", 0, 4], [("curves", "DMK_CURVES", (3, 0))],
        [("schedule", "STRING"), ("value_at_frame", "FLOAT")]),
    7: ("PreviewAny", (1500, 420), [], [("source", "*", (6, 0))], []),
    8: ("DMKBarPrompts", (760, 520), [PROMPTS, 2, 0], [("beats", "DMK_BEATS", (2, 0))],
        [("schedule", "STRING"), ("prompt_at_frame", "STRING"), ("change_frames", "STRING")]),
    9: ("PreviewAny", (1180, 640), [], [("source", "*", (8, 0))], []),
    10: ("PreviewAny", (380, 420), [], [("source", "*", (2, 1))], []),
}


def ui() -> dict:
    links, nodes, lid = [], [], 0
    out_links: dict = {}
    for nid, (typ, pos, wv, ins, outs) in NODES.items():
        for slot, (name, t, (src, sslot)) in enumerate(ins):
            lid += 1
            links.append([lid, src, sslot, nid, slot, t])
            out_links.setdefault((src, sslot), []).append(lid)
    for nid, (typ, pos, wv, ins, outs) in NODES.items():
        nodes.append({
            "id": nid, "type": typ, "pos": list(pos), "size": [320, 200], "flags": {}, "order": nid - 1, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": next(l[0] for l in links if l[3] == nid and l[4] == k)}
                       for k, (n, t, _) in enumerate(ins)],
            "outputs": [{"name": n, "type": t, "links": out_links.get((nid, k), [])} for k, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": typ}, "widgets_values": wv})
    return {"last_node_id": max(NODES), "last_link_id": lid, "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"info": "dmk beat grammar: audio → 4/4 beats on the transient → parameter curves, keyframe "
                              "schedules and world (prompt) changes on the bar. Put signal_decay_20s.m4a into ComfyUI/input."},
            "version": 0.4}


def api() -> dict:
    p = {}
    names = {"LoadAudio": ["audio"], "DMKAudioBeats": ["fps", "frames", "offset_s", "beat_grid", "bpm_min", "bpm_max"],
             "DMKBeatGrammar": ["grammar"], "DMKCurveSchedule": ["parameter", "frame", "decimals"],
             "DMKBarPrompts": ["prompts", "every_bars", "frame"], "PreviewImage": [], "PreviewAny": [], "DMKPlotCurves": []}
    for nid, (typ, pos, wv, ins, outs) in NODES.items():
        inputs = dict(zip(names[typ], wv))
        for name, t, (src, sslot) in ins:
            inputs[name] = [str(src), sslot]
        p[str(nid)] = {"class_type": typ, "inputs": inputs}
    return p


if __name__ == "__main__":
    d = HIER / "example_workflows"
    d.mkdir(exist_ok=True)
    (d / "dmk_beat_grammar.json").write_text(json.dumps(ui(), indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "dmk_beat_grammar_api.json").write_text(json.dumps(api(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", d)
