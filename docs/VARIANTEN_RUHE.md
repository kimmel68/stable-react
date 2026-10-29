# Varianten gegen Zuckeln und Brodeln

Brodeln = Bildänderung pro Frame, die keine Bewegung ist (niedriger = ruhiger). Ruck = Beschleunigung der Bewegung.

Struktur = SSIM zur Quelle bei 64 px (unter 0.6 = eingebrannt, Erhalt und Ruhe sind dann Scheinwerte; starke echte Verwandlung von Farbmaterial liegt bei 0.65-0.70).

| Variante | Struktur | Brodeln | Ruck Ø | Ruck 95 % | Schärfe | erhalt_8 | Beat/Ruhe | Bild-Lag |
|---|---|---|---|---|---|---|---|---|
| Frame to Frame Bild, vorher | 0.89 | 14.92 | 0.428 | 1.094 | 2639 → 2501 → 2600 → 2497 → 2611 | 0.317 | 1.94 | 0 |
| F2F + Seed pro Takt + Glättung (EINGEBRANNT) | 0.558 | 6.27 | 0.019 | 0.081 | 2639 → 1420 → 1533 → 1731 → 1706 | 0.723 | 16.87 | 0 |
| Frame to Frame Bild, Glättung (App-Standard) | 0.902 | 7.46 | 0.223 | 0.567 | 2639 → 2384 → 2237 → 2570 → 2547 | 0.594 | 4.48 | 0 |
| Kamerafahrt Cadence 3 (Referenz) | 0.883 | 4.5 | 0.098 | 0.305 | 2639 → 1410 → 446 → 103 → 206 | 0.667 | 3.5 | 0 |
| Pexels Oberbaum + Seed pro Takt (EINGEBRANNT) | 0.411 | 6.9 | 0.04 | 0.092 | 3791 → 2921 → 2858 → 2965 → 2766 | 0.745 | 14.61 | 0 |
| Pexels Oberbaumbrücke, Glättung | 0.647 | 8.52 | 0.181 | 0.417 | 3791 → 3061 → 3353 → 3548 → 3962 | 0.603 | 4.14 | 0 |
| Pexels Spree, Glättung | 0.7 | 5.91 | 0.206 | 0.415 | 1296 → 1109 → 1169 → 1130 → 1316 | 0.620 | 4.12 | 0 |
