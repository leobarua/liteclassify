"""Renders the splash-screen background, branding/splash_bg.png (2x, 1520 x 780):
raw pixels on the left -- a grid of squares at varying white levels -- turning into
segmented, classified patches on the right, which is the whole idea of OBIA.

Run once when the design changes (needs numpy, scipy, Pillow; the dev environment has
them). liteclassify.py draws the logo and text on top at run time.
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

W, H = 1520, 780          # 2x the 760 x 390 splash
CELL = 30                 # pixel squares, 15 px on screen
BG = np.array([16, 23, 20], float)
# class colours, muted so white text stays readable on top
PALETTE = np.array([[27, 94, 32], [158, 167, 46], [124, 170, 80], [141, 110, 99],
                    [183, 72, 64], [40, 120, 190]], float)


def smooth_field(rng, shape, sigma):
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return (f - f.min()) / (f.max() - f.min())


def main(seed=7):
    rng = np.random.default_rng(seed)
    # one underlying "landscape" both layers describe
    land = 0.65 * smooth_field(rng, (H, W), 70) + 0.35 * smooth_field(rng, (H, W), 22)
    # classes follow a smoother version, so neighbouring segments share a class and
    # the right side reads as a classified map, not a patchwork
    cover = ndimage.gaussian_filter(land, 45)

    # --- left: raw pixels, each square one value
    gy, gx = H // CELL + 1, W // CELL + 1
    cy = np.clip((np.arange(gy) + 0.5) * CELL, 0, H - 1).astype(int)
    cx = np.clip((np.arange(gx) + 0.5) * CELL, 0, W - 1).astype(int)
    val = land[np.ix_(cy, cx)] + 0.18 * rng.standard_normal((gy, gx))
    alpha = np.clip(0.04 + 0.42 * (val - val.min()) / (val.max() - val.min()), 0.03, 0.5)
    a = np.kron(alpha, np.ones((CELL, CELL)))[:H, :W]
    pix = BG + (255 - BG) * a[..., None]
    gap = (np.arange(W) % CELL < 2)[None, :] | (np.arange(H) % CELL < 2)[:, None]   # thin grid lines
    pix[gap] = BG

    # --- right: segments (jittered Voronoi, like superpixels), filled by class
    step = 64
    sx, sy = np.meshgrid(np.arange(step / 2, W, step), np.arange(step / 2, H, step))
    seeds = np.c_[sx.ravel(), sy.ravel()] + rng.uniform(-step * 0.38, step * 0.38, (sx.size, 2))
    yy, xx = np.mgrid[0:H, 0:W]
    _, lab = cKDTree(seeds).query(np.c_[xx.ravel(), yy.ravel()])
    lab = lab.reshape(H, W)
    mean = ndimage.mean(cover, lab, np.arange(len(seeds)))
    cls = np.digitize(mean, np.quantile(mean, [0.24, 0.48, 0.68, 0.82, 0.93]))
    cmap = cls[lab]
    # grayscale: each class a grey tone (the palette's luminance), so the background stays
    # quiet behind the coloured logo and text
    lum = PALETTE @ np.array([0.299, 0.587, 0.114])
    tones = 40 + (lum - lum.min()) / (lum.max() - lum.min()) * 110
    seg = np.repeat(tones[cmap][..., None], 3, axis=2) * 0.62 + BG * 0.38
    seg_edge = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))          # every segment
    seg[seg_edge] = seg[seg_edge] * 0.75 + 255 * 0.25
    cls_edge = (cmap != np.roll(cmap, 1, 0)) | (cmap != np.roll(cmap, 1, 1))      # class boundaries
    cls_edge = ndimage.binary_dilation(cls_edge, iterations=2)
    seg[cls_edge] = seg[cls_edge] * 0.45 + 255 * 0.55

    # --- transition left -> right, with a little per-square jitter so it reads as
    # pixels "becoming" segments rather than a plain cross-fade
    x = np.arange(W) / W
    t = np.clip((x - 0.30) / 0.42, 0, 1)
    t = t * t * (3 - 2 * t)
    jit = np.kron(rng.uniform(-0.12, 0.12, (gy, gx)), np.ones((CELL, CELL)))[:H, :W]
    tt = np.clip(t[None, :] + jit * (t * (1 - t) * 4)[None, :], 0, 1)[..., None]
    img = pix * (1 - tt) + seg * tt

    # --- keep the text areas calm: darken under the name band and the version
    yv = np.arange(H)[:, None] / H
    band = np.clip((yv - 0.50) / 0.18, 0, 1) * 0.62                      # lower band: name, by LeoBar
    ver = np.exp(-(((x[None, :] - 0.80) / 0.17) ** 2 + ((yv - 0.30) / 0.20) ** 2)) * 0.55   # behind "v1.0"
    dark = np.clip(band + ver, 0, 0.8)[..., None]
    img = img * (1 - dark) + BG * dark

    # translucent: the whole layer at OPACITY over the plain dark base
    OPACITY = 0.55
    img = BG + (img - BG) * OPACITY
    img = img.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.85 + img * 0.15   # neutral grey, a hint of the base tint

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splash_bg.png")
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out, optimize=True)
    print(out)


if __name__ == "__main__":
    main()
