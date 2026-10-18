"""Renders the installer's wizard images (installer/wizard_large*.bmp, wizard_small*.bmp)
from the icon and app_info: the side panel (164 x 314, and 2x) and the header icon (55 x 55,
and 2x). Run from the repository root after a name or icon change."""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app_info as A  # noqa: E402

ICON = os.path.join(ROOT, "docs", "manual", "img", "brand", "liteclassify_1024.png")
OUT = os.path.join(ROOT, "installer")
FONTS = "C:/Windows/Fonts"


def font(name, size):
    for n in (name, "segoeuib.ttf"):
        p = os.path.join(FONTS, n)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def hexes(d, w, h, r, col):
    dx, dy = 1.5 * r, math.sqrt(3) * r
    c, x = 0, -r
    while x < w + r:
        y = -dy / 2 if c % 2 else 0
        while y < h + dy:
            d.polygon([(x + r * math.cos(math.radians(a)), y + r * math.sin(math.radians(a)))
                       for a in range(0, 360, 60)], outline=col)
            y += dy
        x += dx
        c += 1


def large(icon, s):
    W, H = 164 * s, 314 * s
    im = Image.composite(Image.new("RGB", (W, H), (13, 19, 17)), Image.new("RGB", (W, H), (20, 28, 25)),
                         Image.linear_gradient("L").resize((W, H)))
    d = ImageDraw.Draw(im, "RGBA")
    hexes(d, W, H, 14 * s, (255, 255, 255, 18))
    ic = icon.resize((96 * s, 96 * s), Image.LANCZOS)
    im.paste(ic, ((W - 96 * s) // 2, 60 * s), ic)
    words = A.APP_NAME.split(" ", 1)          # "Lite" / "Classify" on two lines fits 164 px
    tag = A.APP_TAGLINE.split(" ")
    tag_lines = [" ".join(tag[:2]), " ".join(tag[2:])] if len(tag) > 2 else [A.APP_TAGLINE]
    lines = [(w, font("seguibl.ttf", 26 * s), (246, 82, 68)) for w in words]
    lines += [(t, font("segoeuib.ttf", 11 * s), (232, 236, 233)) for t in tag_lines]
    lines += [(f"by {A.APP_AUTHOR}", font("segoeui.ttf", 11 * s), (185, 195, 189))]
    y = 168 * s
    for i, (txt, f, col) in enumerate(lines):
        w = d.textlength(txt, font=f)
        d.text(((W - w) / 2, y), txt, font=f, fill=col)
        y += f.size + (4 * s if i < len(words) else 5 * s)
        if i == len(words) - 1 or i == len(lines) - 2:
            y += 10 * s
    im.save(os.path.join(OUT, f"wizard_large{'' if s == 1 else '_' + str(s * 100)}.bmp"))


def small(icon, s):
    n = 55 * s
    im = Image.new("RGB", (n, n), (255, 255, 255))
    ic = icon.resize((n, n), Image.LANCZOS)
    im.paste(ic, (0, 0), ic)
    im.save(os.path.join(OUT, f"wizard_small{'' if s == 1 else '_' + str(s * 100)}.bmp"))


if __name__ == "__main__":
    icon = Image.open(ICON).convert("RGBA")
    for s in (1, 2):
        large(icon, s)
        small(icon, s)
    print("written to", OUT)
