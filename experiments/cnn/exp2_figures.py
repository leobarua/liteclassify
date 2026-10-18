"""Figures for experiment 2: cluster maps over the ortho, and sample patches per cluster."""
import os, pickle
import numpy as np, rasterio
from PIL import Image, ImageDraw
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
d = pickle.load(open(os.path.join(OUT, "forest.pkl"), "rb"))
labels = pickle.load(open(os.path.join(OUT, "exp2_labels.pkl"), "rb"))
ids, cents, seg, px = d["ids"], d["cents"], d["seg"], d["px"]
PAL = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], np.uint8)
S = 5
base = np.moveaxis(d["ortho"][:, ::S, ::S], 0, -1).astype(np.float32)
sub = seg[::S, ::S]

def panel(name, title):
    lut = np.zeros(int(seg.max()) + 1, np.int16) - 1
    lut[ids] = labels[name]
    m = lut[sub]
    col = np.where(m[..., None] >= 0, PAL[np.maximum(m, 0)], 0).astype(np.float32)
    img = np.where(m[..., None] >= 0, 0.45 * base + 0.55 * col, 0).astype(np.uint8)
    im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 8), title, fill=(255, 255, 255))
    return im

names = [("app23+relief", "App 23 features + relief"), ("ctxstats_20m", "Colour/texture stats, 20 m window"),
         ("cnn:dinov2_s_0.1_10m", "DINOv2, 10 m patch, 0.1 m"), ("cnn:dinov2_s_0.1_20m", "DINOv2, 20 m patch, 0.1 m")]
ims = [Image.fromarray(base.astype(np.uint8))] + [panel(n, t) for n, t in names]
w, h = ims[0].size
fig = Image.new("RGB", (w * 5, h), (0, 0, 0))
for i, im in enumerate(ims): fig.paste(im, (i * w, 0))
fig.save(os.path.join(OUT, "fig_cluster_maps.png"))

# sample 20 m patches (0.1 m) per DINOv2-20m cluster
lab = labels["cnn:dinov2_s_0.1_20m"]
rng = np.random.default_rng(1)
xs, ys = rasterio.transform.xy(d["transform"], cents[:, 0], cents[:, 1])
T = 150
with rasterio.open("data/CNN_Sample/CNNSample_Ortho_0.1.tif") as s:
    r, c = rasterio.transform.rowcol(s.transform, xs, ys)
    g = Image.new("RGB", (T * 8 + 40, T * 6), (0, 0, 0))
    for k in range(6):
        sel = rng.choice(np.where(lab == k)[0], 8, replace=False)
        for j, i in enumerate(sel):
            a = s.read((1, 2, 3), window=rasterio.windows.Window(c[i] - 100, r[i] - 100, 200, 200), boundless=True)
            g.paste(Image.fromarray(np.moveaxis(a, 0, -1)).resize((T, T)), (40 + j * T, k * T))
        ImageDraw.Draw(g).rectangle([0, k * T, 34, k * T + T - 1], fill=tuple(int(v) for v in PAL[k]))
        ImageDraw.Draw(g).text((6, k * T + 4), f"{(lab == k).mean():.0%}", fill=(0, 0, 0))
    g.save(os.path.join(OUT, "fig_dinov2_20m_cluster_patches.png"))
print("ok")
