"""Builds the Lite Classify icon: the Lite Flight drone redrawn over a background of
hexagonal cells (a nod to the segments Lite Classify works in) instead of a street map.

    python docs/manual/tools/make_icon.py <lite-flight-icon.png|webp>

The drone body outline is traced from the Lite Flight icon; hubs, rotor rings and
propellers are redrawn from measurements of it (770 px master). Writes PNGs to
docs/manual/img/brand/ and logo_liteclassify.ico to the repo root."""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi
from skimage.morphology import disk, opening

SRC = sys.argv[1]
OUT = "docs/manual/img/brand"
M = 770            # master size of the source icon
K = 4              # supersampling factor
S = M * K

TOP, BOT = (249, 112, 66), (246, 66, 58)          # background gradient, from the source
HUBS = [(197, 197), (573, 197), (197, 567), (573, 567)]
HUB_R, RING_R, RING_W, RING_SPAN = 30, 131, 20, 166
BLADE_HALF, BLADE_W, BLADE_ALPHA = 128, 50, 178    # ~70 % white propellers
HEX = 62           # hex cell radius (master px)


def body_mask():
    """The drone's four-pointed body, lifted from the source by colour + opening."""
    im = np.array(Image.open(SRC).convert("RGB").resize((M, M), Image.LANCZOS)).astype(int)
    white = np.minimum(im[..., 1], im[..., 2]) > 200
    o = opening(white, disk(6))
    lab, _ = ndi.label(o)
    body = lab == lab[M // 2, M // 2]
    body = ndi.binary_fill_holes(body)
    big = Image.fromarray((body * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
    big = big.filter(ImageFilter.GaussianBlur(K * 2.2))
    return big.point(lambda v: 255 if v > 127 else 0)


def gradient():
    t = np.linspace(0, 1, S)[:, None]
    diag = (t + np.linspace(0, 1, S)[None, :] * 0.35) / 1.35
    rgb = np.zeros((S, S, 3))
    for c in range(3):
        rgb[..., c] = TOP[c] + (BOT[c] - TOP[c]) * diag
    return Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")


def hex_layer():
    """Pointy-top hexagon grid: light outlines, a few cells softly filled, like segments."""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = HEX * K
    w = math.sqrt(3) * r
    rng = np.random.default_rng(7)
    rows = int(S / (1.5 * r)) + 3
    cols = int(S / w) + 3
    for row in range(-1, rows):
        for col in range(-1, cols):
            cx = col * w + (w / 2 if row % 2 else 0)
            cy = row * 1.5 * r
            pts = [(cx + r * math.cos(math.radians(60 * i - 30)), cy + r * math.sin(math.radians(60 * i - 30))) for i in range(6)]
            if rng.random() < 0.22:
                inner = [(cx + 0.78 * (x - cx), cy + 0.78 * (y - cy)) for x, y in pts]
                d.polygon(inner, fill=(255, 255, 255, int(255 * rng.choice([0.10, 0.16, 0.22]))))
            d.line(pts + [pts[0]], fill=(255, 255, 255, 105), width=3 * K, joint="curve")
    return layer


def drone_layer():
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    white = (255, 255, 255, 255)
    for (hx, hy) in HUBS:
        cx, cy = hx * K, hy * K
        out = math.degrees(math.atan2(hy - M / 2, hx - M / 2))        # outward diagonal
        r = RING_R * K
        d.arc([cx - r, cy - r, cx + r, cy + r], out - RING_SPAN / 2, out + RING_SPAN / 2, fill=white, width=RING_W * K)
        for a in (out - RING_SPAN / 2, out + RING_SPAN / 2):           # rounded ring ends
            ex = cx + (r - RING_W * K / 2) * math.cos(math.radians(a))
            ey = cy + (r - RING_W * K / 2) * math.sin(math.radians(a))
            e = RING_W * K / 2
            d.ellipse([ex - e, ey - e, ex + e, ey + e], fill=white)
    blades = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    db = ImageDraw.Draw(blades)
    for (hx, hy) in HUBS:
        cx, cy = hx * K, hy * K
        out = math.atan2(hy - M / 2, hx - M / 2)
        a = out + math.pi / 2                                          # blades across the arm
        dx, dy = math.cos(a) * BLADE_HALF * K, math.sin(a) * BLADE_HALF * K
        db.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(255, 255, 255, 255), width=BLADE_W * K)
        for sx in (-1, 1):
            e = BLADE_W * K / 2
            db.ellipse([cx + sx * dx - e, cy + sx * dy - e, cx + sx * dx + e, cy + sx * dy + e], fill=(255, 255, 255, 255))
    alpha = blades.split()[3].point(lambda v: v * BLADE_ALPHA // 255)
    blades.putalpha(alpha)
    body = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    body.putalpha(body_mask())
    layer = Image.alpha_composite(layer, blades)
    layer = Image.alpha_composite(layer, body)
    d = ImageDraw.Draw(layer)
    for (hx, hy) in HUBS:
        cx, cy, r = hx * K, hy * K, HUB_R * K
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=white)
    return layer


def rounded(img, radius_frac=0.105):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=int(img.width * radius_frac), fill=255)
    img.putalpha(mask)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    icon = Image.alpha_composite(gradient(), hex_layer())
    icon = Image.alpha_composite(icon, drone_layer())
    icon = rounded(icon)
    for size in (1024, 512, 256, 180, 64, 32):
        icon.resize((size, size), Image.LANCZOS).save(os.path.join(OUT, f"liteclassify_{size}.png"), optimize=True)
    icon.resize((256, 256), Image.LANCZOS).save("logo_liteclassify.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("written to", OUT, "and logo_liteclassify.ico")


if __name__ == "__main__":
    main()
