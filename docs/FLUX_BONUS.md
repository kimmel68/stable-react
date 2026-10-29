# Flux-Bonus („Built with FLUX“) – Stand 26.10.2026

FLUX.1 Kontext [dev] (fp8, Comfy-Org; Lizenz FLUX.1 [dev] Non-Commercial) zeichnet die Beat-Keyframes, die dmk-Taktgrammatik
bestimmt Zeitpunkt und Welt (Wechsel auf der Eins), der optische Fluss des Videos trägt jedes Keyframe bis zum nächsten
Schlag. Nur ComfyUI-Bordmittel (UNETLoader, DualCLIPLoader, ReferenceLatent, FluxGuidance), Aufruf über das Comfy SDK.
Skript: `deforum_mosh/dmk/flux_bonus.py`. RTX 3070 8 GB: ~35 s pro Flux-Bild, 15-s-Clip mit 23 Keyframes in ~14,5 min.

Prompttest (`dmk.flux_prompt_test`, Kantenkorrelation zur Quelle): „Change everything to …, while keeping the same camera
view, framing and outline of every shape“ + Guidance 1,5 hält die Komposition am besten (0,31 gegen 0,08–0,18).

| Clip | Ergebnis |
|---|---|
| f1 Küste (Draufsicht) | Komposition hält in allen Keyframes; Eis → Lava → Marmor → Basalt-Lagune. Stärkster Clip |
| f2 Park & Skyline | Jahreszeiten sehr klar; das erste Frühlingsbild kippte in eine Bodenansicht, danach Luftbild |
| f3 Bergsee (Schrägansicht) | Flux erfindet eigene Postkarten-Seen – Welten schön, Komposition der Quelle verloren |

Befund: Kontext hält Draufsichten zuverlässig, bei Schrägansichten mit Horizont komponiert es neu.
