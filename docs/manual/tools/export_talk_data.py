"""Exports a small bundle of CAFS worked-example data for the launch-talk animations.

Run from the repository root, with the same CAFS_WS / CAFS_RESULTS folders as
build_figures_cafs.py:

    python docs/manual/tools/export_talk_data.py

Writes talk_export/ at the repository root (a few MB): one crop with its segments and
44-feature rows, the reference points on the preview grid, DINO patches and vectors for
Crop 1 vs Crop 2, a 2-D appearance embedding of training objects, and tile strips for
every class. Each section runs on its own, so one failure does not stop the rest.
"""
import json
import os
import sys
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import find_boundaries

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())
import cafs_common as P

OUT = "talk_export"
CROP_A = (7000, 3950, 9560, 5870)   # the manual's feature-explorer patch (x0, y0, x1, y1), 5 cm grid
PREVIEW = 1314                       # width of the manual's whole-scene previews


def section(fn):
    try:
        fn()
        print("ok  ", fn.__name__)
    except Exception:
        print("FAIL", fn.__name__)
        traceback.print_exc()


S = {}


def load():
    S["s"] = P.session()
    S["cls"] = P.classes()
    S["names"] = list(S["cls"])
    S["train"] = P.training()
    S["seg"], S["tr"], S["crs"] = P.segments()
    rgb, _ = P.ortho_on_grid()
    S["rgb"] = np.asarray(rgb)
    S["df"] = P.features()
    S["feats"] = P.feature_cols(S["df"])
    S["px"] = float(S["s"]["effective_pixel_size"])
    S["seg_class"] = {sid: c for c, ids in S["train"].items() for sid in ids}


def crop_a():
    x0, y0, x1, y1 = CROP_A
    seg = S["seg"][y0:y1, x0:x1]
    rgb = S["rgb"][y0:y1, x0:x1]
    Image.fromarray(rgb).resize(((x1 - x0) // 2, (y1 - y0) // 2), Image.LANCZOS).save(
        os.path.join(OUT, "cropA_rgb.jpg"), quality=90)
    ids = np.unique(seg)
    ids = ids[ids > 0]
    local = np.where(seg > 0, np.searchsorted(ids, seg) + 1, 0).astype(np.uint16)
    lut = {int(s): i + 1 for i, s in enumerate(ids)}
    Image.fromarray(local).save(os.path.join(OUT, "cropA_seg.png"))   # full 5 cm, local ids 1..n
    df = S["df"].loc[[i for i in ids if i in S["df"].index], S["feats"]].copy()
    df.insert(0, "local_id", [lut[int(i)] for i in df.index])
    try:
        names = S["names"]
        n = len(ids) + 1
        for col, full in (("object_class", P.object_map()), ("final_class", P.final_map())):
            a = np.asarray(full[y0:y1, x0:x1]).astype(np.int64)
            cnt = np.bincount((local.astype(np.int64) * 32 + a).ravel(), minlength=n * 32).reshape(n, 32)
            cnt[:, 0] = 0
            best = cnt.argmax(1)
            df[col] = [names[best[lut[int(i)]] - 1] if cnt[lut[int(i)]].max() else "" for i in df.index]
    except Exception:
        traceback.print_exc()
    df["training_class"] = [S["seg_class"].get(int(i), "") for i in df.index]
    df.to_csv(os.path.join(OUT, "cropA_features.csv"), index_label="raster_val")
    # scene-wide distribution of every feature, for histograms
    hist = {}
    for c in S["feats"]:
        v = S["df"][c].to_numpy(float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2, 98])
        cnt, edges = np.histogram(np.clip(v, lo, hi), bins=40, range=(lo, hi))
        hist[c] = {"lo": float(lo), "hi": float(hi), "counts": cnt.tolist(), "edges": edges.tolist()}
    json.dump({"n_segments": int(len(S["df"])), "hist": hist, "classes": S["cls"], "crop": CROP_A,
               "pixel_m": S["px"]}, open(os.path.join(OUT, "scene.json"), "w"))


def points():
    g = P.points()
    k = S["seg"].shape[1] / PREVIEW
    fin = P.final_map()
    names = S["names"]
    rows = []
    for r, c, ref in zip(g["row"].values, g["col"].values, g["reference_class"].values):
        v = int(fin[r, c]) if 0 <= r < fin.shape[0] and 0 <= c < fin.shape[1] else 0
        rows.append([round(c / k, 1), round(r / k, 1), str(ref), names[v - 1] if v else ""])
    json.dump(rows, open(os.path.join(OUT, "points.json"), "w"))


def dino():
    z = np.load(os.path.join(P.WS, "deep_features.npz"))
    ids = z["ids"]
    pos = {int(s): i for i, s in enumerate(ids)}
    f5, f10 = z["f5m"].astype(np.float32), z["f10m"].astype(np.float32)
    info = {"keys": list(z.files)}
    for k_ in z.files:
        if z[k_].ndim == 0:
            info[k_] = str(z[k_])
    # 2-D embedding of training objects by their 10 m appearance
    lab, idx = [], []
    for c, sids in S["train"].items():
        for s in sids:
            if s in pos:
                lab.append(c)
                idx.append(pos[s])
    X = f10[idx]
    X = X - X.mean(0)
    u, sv, vt = np.linalg.svd(X, full_matrices=False)
    E = X @ vt[:2].T
    emb = {}
    for c, (a, b) in zip(lab, E):
        emb.setdefault(c, []).append([round(float(a), 2), round(float(b), 2)])
    info["embedding"] = emb
    # per-class medians the comparison card uses
    df = S["df"]
    med = {}
    for c, sids in S["train"].items():
        d = df.loc[[s for s in sids if s in df.index]]
        med[c] = {f: float(d[f].median()) for f in ["brightness_mean", "g_mean", "exg_mean", "hag_mean",
                                                       "deep_10m_pc1", "deep_5m_pc1", "contrast"] if f in d}
    info["medians"] = med
    # one typical Crop 1 and Crop 2 object: 5 m and 10 m patches and its 384 numbers
    seg, rgb, px = S["seg"], S["rgb"], S["px"]
    info["typical"] = {}
    for c in ["Crop 1", "Crop 2", "Grass", "Bare"]:
        sids = [s for s in S["train"].get(c, []) if s in pos]
        if not sids:
            continue
        F = f5[[pos[s] for s in sids]]
        best = sids[int(np.argmin(((F - F.mean(0)) ** 2).sum(1)))]
        cy, cx = ndimage.center_of_mass(seg == best)
        tag = c.lower().replace(" ", "")
        for m in (5, 10):
            h = int(round(m / 2 / px))
            r0, c0 = max(0, int(cy) - h), max(0, int(cx) - h)
            patch = rgb[r0:r0 + 2 * h, c0:c0 + 2 * h].copy()
            if m == 5:
                patch[find_boundaries(seg[r0:r0 + 2 * h, c0:c0 + 2 * h] == best, mode="inner")] = (0, 255, 255)
            Image.fromarray(patch).resize((448, 448), Image.LANCZOS).save(
                os.path.join(OUT, f"patch_{tag}_{m}m.jpg"), quality=90)
        info["typical"][c] = {"segment": int(best), "f5m": [round(float(v), 3) for v in f5[pos[best]]],
                              "f10m": [round(float(v), 3) for v in f10[pos[best]]]}
    json.dump(info, open(os.path.join(OUT, "dino.json"), "w"))


def chips():
    seg, rgb, px = S["seg"], S["rgb"], S["px"]
    half = int(round(2.5 / px))
    rng = np.random.default_rng(7)
    for c, sids in S["train"].items():
        sids = sorted(sids)
        pick = rng.choice(sids, size=min(8, len(sids)), replace=False)
        cents = ndimage.center_of_mass(np.ones(seg.shape, np.uint8), seg, pick)
        strip = Image.new("RGB", (8 * 120 + 7 * 4, 120), (30, 30, 30))
        for i, (sid, cen) in enumerate(zip(pick, cents)):
            r, cc = (int(round(v)) for v in cen)
            r0, c0 = max(0, r - half), max(0, cc - half)
            chip = rgb[r0:r0 + 2 * half, c0:c0 + 2 * half].copy()
            chip[find_boundaries(seg[r0:r0 + 2 * half, c0:c0 + 2 * half] == sid, mode="inner")] = (0, 255, 255)
            strip.paste(Image.fromarray(chip).resize((120, 120), Image.LANCZOS), (i * 124, 0))
        strip.save(os.path.join(OUT, f"chips_{c.lower().replace(' ', '')}.jpg"), quality=90)


def main():
    os.makedirs(OUT, exist_ok=True)
    load()
    for fn in (crop_a, points, dino, chips):
        section(fn)
    print("written to", os.path.abspath(OUT))
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}  {os.path.getsize(os.path.join(OUT, f)) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
