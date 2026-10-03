from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / "fixtures"
rng = random.Random(0)

GLYPHS = [chr(0x12000 + i) for i in range(0, 120, 3)]
READINGS = ["ba", "ka", "lu2", "dug4", "sze", "ma", "an", "ki", "e2", "gal", "ur", "mu", "na", "ni", "ta",
            "da", "la", "ra", "ga", "ab", "u4", "iti", "sila3", "gin2", "udu", "gu4", "kiszib3", "dumu",
            "lugal", "ensi2", "{d}", "{ki}", "sza3", "szu", "ti", "zi", "gi", "me", "nu", "bi"]
SIGN = dict(zip(GLYPHS, READINGS))


def make_line() -> tuple[str, str]:
    words = [[rng.choice(GLYPHS) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 3))]
    glyph_line = "".join("".join(w) for w in words)
    translit_line = " ".join("-".join(SIGN[g] for g in w) for w in words)
    return glyph_line, translit_line


def make_translit(n_tablets: int = 120) -> None:
    out = ROOT / "translit" / "sumtablets_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "glyphs", "transliteration", "period", "genre"])
        for i in range(n_tablets):
            g_surf, t_surf = [], []
            for _ in range(rng.randint(1, 2)):
                lines = [make_line() for _ in range(rng.randint(2, 4))]
                g_surf.append(" ".join(g for g, _ in lines))
                t_surf.append("\n".join(t for _, t in lines))
            glyphs = " ".join(f"<SURFACE> {s}" for s in g_surf)
            translit = " ".join(f"<SURFACE> {s}" for s in t_surf)
            writer.writerow([f"P{100000 + i}", glyphs, translit, "Ur III", "Administrative"])


def make_ocr(n: int = 96) -> None:
    from PIL import ImageFont

    root = ROOT / "ocr"
    (root / "images").mkdir(parents=True, exist_ok=True)
    alphabet = GLYPHS[:8]
    letters = dict(zip(alphabet, "ABCDEFGH"))
    font = ImageFont.load_default()
    rows = []
    for i in range(n):
        seq = [rng.choice(alphabet) for _ in range(rng.randint(3, 6))]
        glyphs = "<SURFACE> " + "".join(seq)
        img = Image.new("RGB", (224, 256), (40, 30, 20))
        draw = ImageDraw.Draw(img)
        for k, g in enumerate(seq):
            x, y = 20 + (k % 3) * 70, 30 + (k // 3) * 90
            draw.rectangle([x - 4, y - 4, x + 50, y + 50], outline=(90, 80, 70))
            for dx in range(0, 36, 12):  # крупнее, чем шрифт по умолчанию
                for dy in range(0, 36, 12):
                    draw.text((x + dx, y + dy), letters[g], fill=(220, 200, 170), font=font)
        name = f"images/T{i:03d}.jpg"
        img.save(root / name)
        split = "test" if i % 8 == 0 else "valid" if i % 8 == 1 else "train"
        rows.append({"id": f"T{i:03d}", "image": name, "glyphs": glyphs, "split": split})
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    make_translit()
    make_ocr()
    print("Фикстуры записаны в", ROOT)
