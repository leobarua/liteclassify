"""Builds the manual's data figures and manual_data.json / manual_data.js from the
Bay, Laguna (Phileco) workspace. Run from the repository root:

    python docs/manual/tools/build_figures_phileco.py

The map in every figure is classified_map.npy, the map the workspace's
accuracy_report.csv scored. The retraining charts are a simulated replay built from the
workspace's final labels (label them as simulated). See tools/README.md.
"""
import json
import os
import sys

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.feature import graycomatrix
from skimage.segmentation import find_boundaries, slic

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())
import phileco_common as P

IMG = "docs/manual/img"
DATA = {}

# Crops on the 10 cm working grid (x0, y0, x1, y1)
CROP_A = (6560, 2720, 7840, 3680)    # 128 x 96 m: road junction, settlement, bare ground, canopy, agroforestry
CROP_F = (7100, 3700, 8380, 4660)    # 128 x 96 m: the disposal facility and its apron
CROP_S = (6300, 2000, 6780, 2360)    # 48 x 36 m: roofs beside canopy (pixels vs objects, SLIC series)
CROP_W = (2500, 2950, 3780, 3910)    # 128 x 96 m: the stream corridor in the north-west
CROP_G = (8300, 3500, 9580, 4460)  # 128 x 96 m: agroforestry rows (east)
CROP_M = (4800, 5500, 6080, 6460)   # 128 x 96 m: grass / canopy mosaic (south)
HERO = (800, 600)


def save(a, name, size=None, quality=86):
    os.makedirs(os.path.dirname(os.path.join(IMG, name)), exist_ok=True)
    im = Image.fromarray(np.asarray(a).astype(np.uint8)) if not isinstance(a, Image.Image) else a
    if size:
        im = im.resize(size, Image.LANCZOS if im.width > size[0] else Image.NEAREST)
    p = os.path.join(IMG, name)
    for attempt in range(5):   # Windows can hold a just-written file briefly (indexer, preview)
        try:
            if name.endswith(".jpg"):
                im.convert("RGB").save(p, quality=quality, optimize=True)
            else:
                im.save(p, optimize=True)
            return
        except OSError:
            if attempt == 4:
                raise
            import time
            time.sleep(0.5)


def crop(a, box):
    x0, y0, x1, y1 = box
    return np.asarray(a[y0:y1, x0:x1])


def font(size):
    for f in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_scalebar(im, metres, px_per_m, label):
    d = ImageDraw.Draw(im)
    L = int(metres * px_per_m)
    x, y = 24, im.height - 30
    d.rectangle([x - 8, y - 26, x + L + 8, y + 12], fill=(255, 255, 255))
    d.rectangle([x, y, x + L, y + 6], fill=(20, 20, 20))
    d.text((x, y - 22), label, fill=(20, 20, 20), font=font(15))
    return im


def colorize_values(v, cmap="viridis", lo=None, hi=None, center=None):
    import matplotlib
    ok = np.isfinite(v)
    if center is not None:
        m = np.nanpercentile(np.abs(v[ok] - center), 98) if ok.any() else 1
        lo, hi = center - m, center + m
    else:
        lo = np.nanpercentile(v[ok], 2) if lo is None else lo
        hi = np.nanpercentile(v[ok], 98) if hi is None else hi
    t = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = (matplotlib.colormaps[cmap](np.where(ok, t, 0))[..., :3] * 255).astype(np.uint8)
    rgb[~ok] = 255
    return rgb, float(lo), float(hi)


def mean_colour(rgb, seg):
    out = np.zeros_like(rgb)
    flat = seg.ravel()
    n = np.bincount(flat)
    for ch in range(3):
        s = np.bincount(flat, weights=rgb[..., ch].ravel().astype(np.float64))
        out[..., ch] = (s / np.maximum(n, 1))[seg].astype(np.uint8)
    return out


def hillshade(dsm, px, az=315, alt=45):
    z = np.where(np.isfinite(dsm), dsm, np.nanmedian(dsm))
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    hs = np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def stratified_estimates(ref, mapped, stratum, strata_px, names):
    """Area proportions p[i, j] (map i, reference j), overall / user's / producer's
    accuracy, for a stratified sample whose strata may differ from the map classes
    (Stehman 2014). ref, mapped, stratum: arrays of class names per point."""
    N = sum(strata_px.values())
    k = len(names)
    p = np.zeros((k, k))
    for h, Nh in strata_px.items():
        sel = stratum == h
        nh = sel.sum()
        if nh == 0:
            continue
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                p[i, j] += (Nh / N) * np.sum((mapped[sel] == a) & (ref[sel] == b)) / nh
    oa = float(np.trace(p))
    ua = {n: float(p[i, i] / p[i].sum()) if p[i].sum() else float("nan") for i, n in enumerate(names)}
    pa = {n: float(p[i, i] / p[:, i].sum()) if p[:, i].sum() else float("nan") for i, n in enumerate(names)}
    return oa, ua, pa


def read_report():
    """The workspace's accuracy_report.csv: count matrix and the area-weighted table."""
    import csv
    rows = list(csv.reader(open(os.path.join(P.WS, "accuracy_report.csv"))))
    i = next(k for k, r in enumerate(rows) if r and r[0] == "Classified")
    cols = rows[i][1:-1]
    mat, rnames = [], []
    for r in rows[i + 1:]:
        if r[0] == "Total":
            break
        rnames.append(r[0]); mat.append([int(x) for x in r[1:-1]])
    out = {"cols": cols, "rows": rnames, "matrix": mat}
    j = next(k for k, r in enumerate(rows) if r and r[0] == "Class" and len(r) > 3)
    est = {}
    for r in rows[j + 1:]:
        if not r or r[0].startswith("Overall"):
            break
        est[r[0]] = {"mapped_ha": float(r[1]), "est_ha": float(r[2]), "est_ci_ha": float(r[3]),
                     "ua": float(r[4]), "ua_ci": float(r[5]), "pa": float(r[6]), "pa_ci": float(r[7])}
    oa_row = next(r for r in rows if r and r[0].startswith("Overall Accuracy (area-weighted)"))
    raw_oa = next(r for r in rows if r and r[0] == "Overall Accuracy")
    out.update({"est": est, "oa_aw": float(oa_row[1]), "oa_aw_ci": float(oa_row[3]), "oa_raw": float(raw_oa[1])})
    return out


def main():
    s = P.session()
    cls = P.classes()
    names = list(cls)
    train = P.training()
    seg, tr, crs = P.segments()
    rgb = P.ortho_on_grid()
    cmap = P.class_map()
    df = P.features()
    feats = P.feature_cols(df)
    px = float(s["effective_pixel_size"])
    H, W = seg.shape
    g = P.points()

    with rasterio.open(P.ORTHO) as src:
        native_px, nW, nH = abs(src.transform.a), src.width, src.height
    with rasterio.open(P.DSM) as src:
        dsm_px = abs(src.transform.a)
    valid = seg > 0
    area_ha = float(valid.sum() * px * px / 1e4)

    DATA["classes"] = [[k, v] for k, v in cls.items()]
    DATA["grid"] = {"width": W, "height": H, "pixel_m": px, "native_pixel_m": native_px,
                    "native_width": nW, "native_height": nH, "dsm_pixel_m": dsm_px,
                    "feature_pixel_m": float(s.get("feature_pixel_size", px)), "area_ha": area_ha}
    DATA["segmentation"] = {"n_segments": s["n_segments"], "compactness": s["compactness"], "sigma": s["sigma"],
                            "segments": int(len(df)), "mean_area_m2": float(df["area"].mean()),
                            "median_area_m2": float(df["area"].median()),
                            "mean_area_px": float(df["area"].mean() / (px * px))}
    DATA["training"] = {"per_class": {k: len(v) for k, v in train.items()},
                        "total": sum(len(v) for v in train.values())}
    DATA["training"]["share_of_segments"] = DATA["training"]["total"] / len(df)
    DATA["features"] = {"base": len(P.BASE), "dsm": len(P.BASE) + len(P.DSM_FEATS),
                        "deep": sum(c.startswith("deep_") for c in feats), "total": len(feats)}

    # ---------- groups (Discover) ----------
    gj = json.load(open(os.path.join(P.WS, "groups.json")))
    groups = gj["groups"]
    seg_group = {int(k): v for k, v in gj["segments"].items()}
    gcount = {}
    for v in seg_group.values():
        gcount[v] = gcount.get(v, 0) + 1
    area = df["area"]
    garea = {}
    for sid, gk in seg_group.items():
        garea[gk] = garea.get(gk, 0.0) + float(area.get(sid, 0.0))
    DATA["discover"] = {"requested": gj["settings"]["n_groups"], "groups": len(groups),
                        "named": sum(v["status"] == "named" for v in groups.values()),
                        "unnamed": sum(v["status"] == "unnamed" for v in groups.values()),
                        "ignored": sum(v["status"] == "ignored" for v in groups.values()),
                        "grouped_segments": len(seg_group),
                        "left_out": int(len(df) - len(seg_group)),
                        "by_name": {}}
    for gk, v in groups.items():
        if v["status"] == "named":
            DATA["discover"]["by_name"].setdefault(v["name"], []).append(gk)

    # ---------- the survey ----------
    k = 8
    ov = np.asarray(rgb[::k, ::k])
    save(draw_scalebar(Image.fromarray(ov.copy()), 100, 1 / (px * k), "100 m"), "sample_ortho.jpg")
    dsm = P.dsm_on_grid()
    dk = np.asarray(dsm[::k, ::k])
    okd = np.isfinite(dk)
    lo, hi = np.percentile(dk[okd], (2, 98))
    col, _, _ = colorize_values(dk, "terrain", lo, hi)
    hs = hillshade(dk, px * k)
    shaded = (col * (0.45 + 0.55 * hs[..., None])).astype(np.uint8)
    shaded[~okd] = 255
    save(draw_scalebar(Image.fromarray(shaded), 100, 1 / (px * k), "100 m"), "sample_dsm.jpg")
    DATA["dsm"] = {"p2": float(lo), "p98": float(hi), "min": float(np.nanmin(dk)), "max": float(np.nanmax(dk))}

    # native-resolution inset: 30 x 30 m around a crop centre in the settlement
    cx, cy = rasterio.transform.xy(tr, (CROP_S[1] + CROP_S[3]) // 2, (CROP_S[0] + CROP_S[2]) // 2)
    with rasterio.open(P.ORTHO) as src:
        r, c = src.index(cx, cy)
        n = int(round(30 / native_px))
        a = src.read((1, 2, 3), window=rasterio.windows.Window(c - n // 2, r - n // 2, n, n))
    im = Image.fromarray(np.transpose(a, (1, 2, 0))).resize((760, 760), Image.LANCZOS)
    save(draw_scalebar(im, 5, 760 / 30, "5 m"), "sample_inset_native.jpg")

    # ---------- the map, crops ----------
    cls_rgb_full = P.colorize(cmap, cls)
    for tag, box in (("", CROP_A), ("_facility", CROP_F), ("_stream", CROP_W), ("_agro", CROP_G), ("_mosaic", CROP_M)):
        cr, cs = crop(rgb, box), crop(seg, box)
        bnd = find_boundaries(cs, mode="inner")
        rb = cr.copy(); rb[bnd] = (0.45 * rb[bnd] + 0.55 * np.array([40, 90, 255])).astype(np.uint8)
        cc = crop(cls_rgb_full, box)
        save(cr, f"view_raw{tag}.jpg", size=HERO)
        save(cc, f"view_classified{tag}.png", size=HERO)
        if tag == "":   # the other views only for the main crop
            save(rb, "view_raw_borders.jpg", size=HERO)
            save(mean_colour(cr, cs), "view_segmented.jpg", size=HERO)
            save((cr * 0.4 + cr * (cc / 255.0) * 0.6).astype(np.uint8), "view_blend.jpg", size=HERO)

    # reference points over the hero crop
    kx = HERO[0] / (CROP_A[2] - CROP_A[0])
    im = Image.fromarray(crop(rgb, CROP_A)).resize(HERO, Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for r_, c_, ref in zip(g["row"], g["col"], g["reference_class"]):
        x, y = (c_ - CROP_A[0]) * kx, (r_ - CROP_A[1]) * kx
        if 0 <= x < HERO[0] and 0 <= y < HERO[1]:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=P.hex2rgb(cls[ref]), outline=(255, 255, 255), width=3)
    save(im, "view_points.jpg")

    # pixels vs objects
    cr, cs = crop(rgb, CROP_S), crop(seg, CROP_S)
    b = find_boundaries(cs, mode="inner")
    x = cr.copy(); x[b] = (255, 236, 64)
    save(cr, "obj_pixels.jpg", size=(720, 540))
    save(x, "obj_boundaries.jpg", size=(720, 540))
    y = mean_colour(cr, cs); y[b] = (255, 255, 255)
    save(y, "obj_means.jpg", size=(720, 540))

    # ---------- SLIC parameter series on CROP_S (CPU engine, this crop alone) ----------
    from scipy.ndimage import gaussian_filter
    img = cr.astype(np.float64) / 255.0
    frac = (CROP_S[2] - CROP_S[0]) * (CROP_S[3] - CROP_S[1]) / valid.sum()
    series = {}
    for tag, (n, c, sg) in {"k20000": (20000, .15, 2.5), "k60000": (60000, .15, 2.5), "k150000": (150000, .15, 2.5),
                            "m005": (60000, .05, 2.5), "m050": (60000, .5, 2.5),
                            "s0": (60000, .15, 0), "s6": (60000, .15, 6.0)}.items():
        im_s = gaussian_filter(img, sigma=[sg, sg, 0]) if sg > 0 else img
        lab = slic(im_s, n_segments=max(2, int(round(n * frac))), compactness=c * 100.0, sigma=0, start_label=1)
        series[tag] = {"requested": n, "crop_segments": int(lab.max()),
                       "mean_area_m2": float(lab.size * px * px / lab.max())}
        out = cr.copy(); out[find_boundaries(lab, mode="inner")] = (255, 236, 64)
        save(out, f"seg_{tag}.jpg", size=(720, 540))
    DATA["seg_series"] = series

    # ---------- feature maps (CROP_A) ----------
    explorer = P.BASE + P.DSM_FEATS + ["deep_5m_pc1", "deep_10m_pc1"]
    cs = crop(seg, CROP_A)
    ids = np.unique(cs); ids = ids[ids > 0]
    sub = df.loc[df.index.intersection(ids)]
    lutn = np.full(int(seg.max()) + 1, np.nan, dtype=np.float32)
    bb = find_boundaries(cs, mode="inner")
    fmeta = {}
    for f in explorer:
        lutn[:] = np.nan
        lutn[sub.index.values] = sub[f].values
        m = lutn[cs]
        center = 0.0 if f.endswith("rel_neighbors") else None
        cm = "RdBu_r" if center is not None else ("cividis" if f.startswith("dsm") or f in ("slope_mean", "hag_mean") else
                                                  ("magma" if f.startswith("deep_") else "viridis"))
        im_f, lo_, hi_ = colorize_values(m, cmap=cm, center=center)
        im_f[bb] = (im_f[bb] * 0.55).astype(np.uint8)
        save(im_f, f"feat/{f}.jpg", size=(640, 480), quality=80)
        fmeta[f] = {"lo": lo_, "hi": hi_, "cmap": cm}
    c3 = crop(rgb, CROP_A).astype(np.float32)
    R_, G_, B_ = c3[..., 0], c3[..., 1], c3[..., 2]
    ss = np.where(R_ + G_ + B_ == 0, 1, R_ + G_ + B_)
    pix = {"exg": 2 * G_ / ss - R_ / ss - B_ / ss,
           "vari": (G_ - R_) / np.where(G_ + R_ - B_ == 0, 1, G_ + R_ - B_),
           "ngrdi": (G_ - R_) / np.where(G_ + R_ == 0, 1, G_ + R_)}
    for kk, v in pix.items():
        im_f, lo_, hi_ = colorize_values(np.clip(v, -1, 1), "viridis")
        save(im_f, f"feat/pixel_{kk}.jpg", size=(640, 480), quality=80)
        fmeta["pixel_" + kk] = {"lo": lo_, "hi": hi_, "cmap": "viridis"}
    DATA["feature_maps"] = fmeta

    # ---------- height layers (CROP_A) ----------
    dc = crop(dsm, CROP_A)
    im_f, lo_, hi_ = colorize_values(dc, "cividis"); save(im_f, "dsm_crop.jpg", size=HERO)
    DATA["dsm_crop"] = {"lo": lo_, "hi": hi_}
    gy, gx = np.gradient(np.where(np.isfinite(dc), dc, np.nanmedian(dc)), px)
    im_f, _, _ = colorize_values(np.degrees(np.arctan(np.hypot(gx, gy))), "magma", 0, 60)
    save(im_f, "slope_crop.jpg", size=HERO)
    hs_c = hillshade(dc, px)
    save((crop(rgb, CROP_A) * 0.4 + (hs_c[..., None] * 255) * 0.6).astype(np.uint8), "hillshade_overlay.jpg", size=HERO)

    # ---------- GLCM worked example: canopy vs a smooth roof, at the feature resolution ----------
    cls_of = {}
    lut_cls = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
    # segment class = the map's class at the segment (the map assigns whole segments)
    ys, xs = np.nonzero(cs)
    lut_cls[cs[ys, xs]] = crop(cmap, CROP_A)[ys, xs]
    for sid in sub.index:
        cls_of[sid] = names[lut_cls[sid] - 1] if lut_cls[sid] else None
    cand = sub[(sub["area"] > 6) & (sub["area"] < 14)]
    compact = cand[(cand["convexity"] > 0.85) & (cand["circularity"] > 0.55)]
    canopy = compact[[cls_of[i] == "Natural canopy" for i in compact.index]]["contrast"].idxmax()
    roof = cand[[cls_of[i] == "Built-up" for i in cand.index]]["contrast"].idxmin()
    glcm_out = {}
    sl = ndimage.find_objects(seg)
    with rasterio.open(P.ORTHO) as src:
        for tag, sid in (("canopy", canopy), ("roof", roof)):
            rs, cs_ = sl[sid - 1]
            pad = 8
            r0, r1, c0, c1 = max(0, rs.start - pad), rs.stop + pad, max(0, cs_.start - pad), cs_.stop + pad
            bounds = rasterio.transform.array_bounds(r1 - r0, c1 - c0, tr * tr.translation(c0, r0))
            win = rasterio.windows.from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            a = np.transpose(src.read((1, 2, 3), window=win), (1, 2, 0)).astype(np.float32)
            labs = np.array(Image.fromarray(seg[r0:r1, c0:c1].astype(np.int32)).resize((a.shape[1], a.shape[0]), Image.NEAREST))
            mask = labs == sid
            grey = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            q = 1 + np.clip((grey / 255.0) * 14, 0, 14).astype(np.uint8)
            q[~mask] = 0
            gl = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True)
            gl[0, :, :, :] = 0; gl[:, 0, :, :] = 0
            gl = gl / gl.sum(axis=(0, 1), keepdims=True)
            Pm = gl.mean(axis=(2, 3))
            view = a.astype(np.uint8).copy()
            view[~mask] = (view[~mask] * 0.35 + 255 * 0.65).astype(np.uint8)
            view[find_boundaries(mask, mode="inner")] = (255, 0, 160)
            save(view, f"glcm_{tag}.png", size=(view.shape[1] * 2, view.shape[0] * 2))
            row = df.loc[sid]
            glcm_out[tag] = {"segment": int(sid), "matrix": np.round(Pm[1:, 1:], 4).tolist(),
                             "contrast": float(row["contrast"]), "homogeneity": float(row["homogeneity"]),
                             "energy": float(row["energy"]), "entropy": float(row["entropy"]),
                             "area": float(row["area"]), "pixels": int(mask.sum())}
    DATA["glcm"] = glcm_out

    # ---------- scene-wide: training samples, groups, reference points, final map ----------
    sk = np.asarray(seg[::k, ::k])
    base = ov.astype(np.float32) * 0.5 + 255 * 0.5
    lut = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
    for i, nm in enumerate(names, start=1):
        lut[list(train[nm])] = i
    tmap = lut[sk]
    out = base.copy()
    for i, nm in enumerate(names, start=1):
        out[tmap == i] = P.hex2rgb(cls[nm])
    save(out.astype(np.uint8), "training_samples.jpg")

    glut = np.zeros(int(seg.max()) + 1, dtype=np.int32)
    gkeys = sorted(groups)
    for sid, gk in seg_group.items():
        glut[sid] = gkeys.index(gk) + 1
    gcol = np.array([(255, 255, 255)] + [P.hex2rgb(groups[gk]["color"]) for gk in gkeys], dtype=np.uint8)
    gm = glut[sk]
    out = gcol[gm].astype(np.float32) * 0.7 + ov * 0.3
    out[gm == 0] = base[gm == 0]
    save(out.astype(np.uint8), "disc_groups_scene.jpg")
    gmc = glut[crop(seg, CROP_A)]
    out = gcol[gmc].astype(np.float32) * 0.55 + crop(rgb, CROP_A) * 0.45
    save(out.astype(np.uint8), "disc_groups_crop.jpg", size=HERO)
    DATA["discover"]["sizes"] = {gk: {"segments": gcount.get(gk, 0), "area_share": garea.get(gk, 0) / float(area.sum()),
                                      "name": groups[gk]["name"], "status": groups[gk]["status"],
                                      "color": groups[gk]["color"]} for gk in gkeys}

    # chips for four groups: 5 m patches around segment centroids, like the app's tiles
    cents = ndimage.center_of_mass(np.ones(seg.shape, np.uint8), seg, np.arange(1, int(seg.max()) + 1))
    half = int(round(2.5 / px))
    rng = np.random.default_rng(1)
    for gk in DATA["discover"]["by_name"].get("Natural canopy", [])[:1] + DATA["discover"]["by_name"].get("Agricultural", [])[:1] \
            + DATA["discover"]["by_name"].get("Solid waste", [])[:1] + DATA["discover"]["by_name"].get("Built-up", [])[:1]:
        members = [sid for sid, v in seg_group.items() if v == gk]
        pick = rng.choice(members, size=min(8, len(members)), replace=False)
        strip = Image.new("RGB", (8 * 120 + 7 * 4, 120), (30, 30, 30))
        for i_, sid in enumerate(pick):
            r_, c_ = (int(round(v)) for v in cents[sid - 1])
            r0, c0 = max(0, r_ - half), max(0, c_ - half)
            chip = np.asarray(rgb[r0:r0 + 2 * half, c0:c0 + 2 * half]).copy()
            m = np.asarray(seg[r0:r0 + 2 * half, c0:c0 + 2 * half]) == sid
            chip[find_boundaries(m, mode="inner")] = (0, 255, 255)
            strip.paste(Image.fromarray(chip).resize((120, 120), Image.LANCZOS), (i_ * 124, 0))
        strip.save(os.path.join(IMG, f"disc_chips_{gk}.jpg"), quality=86)
    DATA["discover"]["chip_groups"] = [gk for gk in gkeys if os.path.exists(os.path.join(IMG, f"disc_chips_{gk}.jpg"))]

    im = Image.fromarray(ov.copy()); d = ImageDraw.Draw(im)
    for r_, c_, ref, pred in zip(g["row"], g["col"], g["reference_class"], g["predicted_class"]):
        x, y = c_ / k, r_ / k
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=P.hex2rgb(cls[ref]), outline=(255, 255, 255), width=1)
        if ref != pred:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 0, 200), width=2)
    save(im, "reference_points.jpg")
    Image.fromarray(cls_rgb_full[::k, ::k]).save(os.path.join(IMG, "final_map.png"), optimize=True)
    DATA["map_share"] = {nm: float((cmap == i).sum() / valid.sum()) for i, nm in enumerate(names, start=1)}
    DATA["map_ha"] = {nm: float((cmap == i).sum() * px * px / 1e4) for i, nm in enumerate(names, start=1)}

    # ---------- accuracy ----------
    rep = read_report()
    ref = g["reference_class"].values.astype(str)
    strat = g["stratum"].values.astype(str)
    mp = g["predicted_class"].values.astype(str)
    DATA["final"] = {"matrix": rep["matrix"], "rows": rep["rows"], "cols": rep["cols"],
                     "n": int(len(g)), "oa": rep["oa_raw"],
                     "ua": {n: rep["matrix"][i][i] / max(1, sum(rep["matrix"][i])) for i, n in enumerate(rep["rows"])},
                     "pa": {n: rep["matrix"][i][i] / max(1, sum(r[i] for r in rep["matrix"])) for i, n in enumerate(rep["cols"])},
                     "oa_aw": rep["oa_aw"], "oa_aw_ci": rep["oa_aw_ci"], "est": rep["est"]}
    DATA["reference_counts"] = {n: int((ref == n).sum()) for n in names}
    strata_px = {h: int(v) for h, v in s["accuracy_strata"]["pixels"].items()}
    DATA["strata"] = {"design": "stratified", "min_per_class": s["accuracy_min_per_class"],
                      "seed": s["accuracy_strata"]["seed"], "pixels": strata_px,
                      "n_strata": len(strata_px), "points_per_stratum": {h: int((strat == h).sum()) for h in strata_px}}
    chk = stratified_estimates(ref, mp, strat, strata_px, names)
    DATA["final"]["oa_aw_check"] = chk[0]
    Image.open(os.path.join(P.WS, "accuracy_matrix.png")).convert("RGB").save(os.path.join(IMG, "accuracy_matrix.jpg"), quality=90)

    # ---------- simulated retraining replay (area-weighted OA on the stratified sample) ----------
    from sklearn.ensemble import RandomForestClassifier
    X_all = df[feats].values
    pos = {sid: i for i, sid in enumerate(df.index)}
    pt_seg = np.asarray(seg)[g["row"].values, g["col"].values]
    pt_idx = np.array([pos.get(int(sid), -1) for sid in pt_seg])
    lab_of = {sid: n for n, v in train.items() for sid in v if sid in pos}
    pool = np.array(sorted(lab_of)); pool_y = np.array([lab_of[i] for i in pool])
    START, ADD, ROUNDS, SEEDS = 5, 8, 8, 10

    def fit_predict(used):
        ids_ = [sid for v in used.values() for sid in v]
        ys_ = [n for n, v in used.items() for _ in v]
        order = np.argsort([pos[i] for i in ids_])     # the app merges in feature-table order
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_all[[pos[ids_[o]] for o in order]], np.array(ys_)[order])
        return clf

    runs = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        used = {n: set(rng.choice(sorted(v & set(pos)), size=min(START, len(v)), replace=False).tolist()) for n, v in train.items()}
        rows = []
        for rnd in range(1, ROUNDS + 1):
            clf = fit_predict(used)
            pred_pts = clf.predict(X_all[pt_idx])
            oa, ua, pa = stratified_estimates(ref, pred_pts, strat, strata_px, names)
            row = {"round": rnd, "samples": sum(len(v) for v in used.values()), "oa": oa, "pa": pa}
            if seed == 0 and rnd in (1, 3, ROUNDS):
                p_all = clf.predict(X_all)
                lut_r = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
                lut_r[df.index.values] = [names.index(q) + 1 for q in p_all]
                row["map"] = lut_r
            rows.append(row)
            pred_pool = clf.predict(X_all[[pos[i] for i in pool]])
            wrong = [(sid, yy) for sid, yy, pp in zip(pool, pool_y, pred_pool) if pp != yy and sid not in used[yy]]
            added = 0
            for n in names:
                cn = [sid for sid, yy in wrong if yy == n]
                rng.shuffle(cn)
                used[n] |= set(int(v) for v in cn[:ADD]); added += min(ADD, len(cn))
            if added == 0:
                break
        runs.append(rows)
        print("replay", seed, [(r["samples"], round(r["oa"], 3)) for r in rows])
    sim = []
    for rnd in range(1, ROUNDS + 1):
        rr = [r[min(rnd, len(r)) - 1] for r in runs]
        oas = [r["oa"] for r in rr]
        sim.append({"round": rnd, "samples": float(np.mean([r["samples"] for r in rr])),
                    "oa_mean": float(np.mean(oas)), "oa_min": float(np.min(oas)), "oa_max": float(np.max(oas)),
                    "pa_mean": {n: float(np.nanmean([r["pa"][n] for r in rr])) for n in names}})
    full = fit_predict({n: v & set(pos) for n, v in train.items()})
    oa_full, _, pa_full = stratified_estimates(ref, full.predict(X_all[pt_idx]), strat, strata_px, names)
    DATA["simulation"] = {"start_per_class": START, "add_per_class": ADD, "seeds": SEEDS, "rounds": sim,
                          "final_full": {"samples": DATA["training"]["total"], "oa": oa_full, "pa": pa_full},
                          "metric": "area-weighted overall accuracy on the 514 stratified points"}
    first = runs[0][0]
    DATA["sparse_example"] = {"oa": first["oa"], "samples": first["samples"]}
    for row in runs[0]:
        if "map" in row:
            m = row["map"][sk]
            save(P.colorize(m, cls), f"sim_round_{row['round']}.png")
    DATA["simulation"]["map_rounds"] = [r["round"] for r in runs[0] if "map" in r]
    m1 = runs[0][0]["map"][crop(seg, CROP_A)]
    save(P.colorize(m1, cls), "sparse_zoom.png", size=HERO)
    save(crop(cls_rgb_full, CROP_A), "full_zoom.png", size=HERO)

    # ---------- sieve ----------
    from rasterio.features import sieve
    sv_rows = []
    pr, pc = g["row"].values, g["col"].values
    for t in (0, 500, 1000, 2000):
        sv = cmap if t == 0 else sieve(cmap.astype(np.int32), t).astype(np.uint8)
        mp_t = np.array([names[v - 1] if v else "" for v in sv[pr, pc]])
        oa, _, _ = stratified_estimates(ref, mp_t, strat, strata_px, names)
        sv_rows.append({"mmu_px": t, "mmu_m2": t * px * px, "oa": oa, "changed_px": float((sv != cmap)[valid].mean())})
        if t == 2000:
            save(crop(P.colorize(sv, cls), CROP_A), "sieve_2000.png", size=HERO)
    save(crop(cls_rgb_full, CROP_A), "sieve_0.png", size=HERO)
    DATA["sieve"] = sv_rows
    DATA["mmu_px"] = 0

    # ---------- merge demo: one roof's pieces, by region growing over Built-up segments ----------
    cs = crop(seg, CROP_S)
    cm_s = crop(cmap, CROP_S)
    # raised Built-up segments only (roofs, not the road they touch)
    roofs = [sid for sid in np.unique(cs) if sid and np.all(cm_s[cs == sid] == names.index("Built-up") + 1)
             and df.loc[sid, "hag_mean"] > 2.5]
    best = None
    sl_s = ndimage.find_objects(cs)
    adj = {}
    for a_, b_ in ((cs[:, :-1], cs[:, 1:]), (cs[:-1, :], cs[1:, :])):
        mm = a_ != b_
        for u, v in zip(a_[mm], b_[mm]):
            adj.setdefault(int(u), set()).add(int(v)); adj.setdefault(int(v), set()).add(int(u))
    rs = set(int(v) for v in roofs)
    seen = set()
    for r0 in rs:
        if r0 in seen:
            continue
        comp, fr = {r0}, [r0]
        while fr:
            u = fr.pop()
            for v in adj.get(u, ()):
                if v in rs and v not in comp:
                    comp.add(v); fr.append(v)
        seen |= comp
        if best is None or len(comp) > len(best):
            best = comp
    before = cr.copy(); before[find_boundaries(cs, mode="inner")] = (255, 236, 64)
    sel = np.isin(cs, list(best))
    before[sel] = (before[sel] * 0.55 + np.array([0, 255, 255]) * 0.45).astype(np.uint8)
    save(before, "merge_before.jpg", size=(720, 540))
    merged = cs.copy(); merged[sel] = min(best)
    after = cr.copy(); after[find_boundaries(merged, mode="inner")] = (255, 236, 64)
    save(after, "merge_after.jpg", size=(720, 540))
    DATA["merge"] = {"pieces": len(best), "area_m2": float(df.loc[list(best), "area"].sum())}

    # ---------- preparation: resolutions, colour correction ----------
    with rasterio.open(P.ORTHO) as src:
        n = int(round(30 / native_px)); m_ = int(round(22 / native_px))
        r, c = src.index(*rasterio.transform.xy(tr, 3000, 7300))
        win = rasterio.windows.Window(c, r, n, m_)
        for tag, f_ in (("native", 1), ("10cm", native_px / 0.10), ("20cm", native_px / 0.20)):
            a = src.read((1, 2, 3), window=win, out_shape=(3, max(1, int(m_ * f_)), max(1, int(n * f_))),
                         resampling=rasterio.enums.Resampling.average)
            save(Image.fromarray(np.transpose(a, (1, 2, 0))).resize((750, 550), Image.NEAREST), f"res_{tag}.jpg")
    p2, p98 = np.percentile(ov[ov.sum(-1) < 765], (2, 98))
    c3 = crop(rgb, CROP_A).astype(np.float32)
    save(crop(rgb, CROP_A), "color_before.jpg", size=HERO)
    save(np.clip((c3 - p2) / (p98 - p2) * 255, 0, 255), "color_auto.jpg", size=HERO)

    # ---------- separability for the current training set (the app's own code) ----------
    # Rendered with the normal Qt platform (offscreen has no fonts). The table + closest
    # pairs go in once; each pair's signature plot is cropped out for the gallery.
    try:
        from itertools import combinations
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from obia_engine import ObiaEngine
        import obia_ui
        e = ObiaEngine()
        e.features_df = df
        res = e.compute_class_separability({n: list(v) for n, v in train.items()}, features=feats)
        pairs = []
        for i_, (a_, b_) in enumerate(combinations(names, 2)):
            qi = obia_ui.render_separability_image(res, cls, focus_pair=(a_, b_), scale=2)
            fp = os.path.join(CACHE_IMG, "sep.png"); qi.save(fp)
            im = Image.open(fp).convert("RGB")
            # split at the widest blank band between the closest pairs and the signature plot
            arr = np.asarray(im.convert("L"), dtype=np.int16)
            blank = (np.abs(arr - arr[:, -1:]).max(axis=1) < 8)
            lo_r, hi_r = int(im.height * 0.45), int(im.height * 0.70)
            best_run, run_start, cut = 0, None, int(im.height * 0.555)
            for rr in range(lo_r, hi_r):
                if blank[rr]:
                    run_start = rr if run_start is None else run_start
                    if rr - run_start > best_run:
                        best_run, cut = rr - run_start, (run_start + rr) // 2
                else:
                    run_start = None
            if i_ == 0:
                cols = np.nonzero((np.abs(arr[:cut] - arr[:cut, -1:]) >= 8).any(axis=0))[0]
                right = min(im.width, int(cols.max()) + 40) if len(cols) else im.width
                top = im.crop((0, 0, right, cut))
                tw = min(1800, top.width)
                top.resize((tw, int(top.height * tw / top.width)), Image.LANCZOS).save(
                    os.path.join(IMG, "separability_table.jpg"), quality=86)
            bot = im.crop((0, cut, im.width, im.height))
            name = f"sep/pair_{i_ + 1:02d}.jpg"
            os.makedirs(os.path.join(IMG, "sep"), exist_ok=True)
            bot.resize((1800, int(bot.height * 1800 / bot.width)), Image.LANCZOS).save(os.path.join(IMG, name), quality=82)
            pairs.append({"a": a_, "b": b_, "img": name})
        DATA["separability"] = {"pairs": pairs}
    except Exception as ex:   # figure only; the manual still builds
        print("separability figure skipped:", ex)

    json.dump(DATA, open("docs/manual/manual_data.json", "w"), indent=1, default=float)
    write_data_js()
    print(json.dumps({k: DATA[k] for k in ("grid", "segmentation", "training", "features", "sieve", "merge",
                                           "sparse_example")}, indent=1, default=float))
    print("simulation", [(round(r["samples"]), round(r["oa_mean"], 3), round(r["oa_min"], 3), round(r["oa_max"], 3))
                         for r in DATA["simulation"]["rounds"]], DATA["simulation"]["final_full"]["oa"])
    print("OA check", DATA["final"]["oa_aw"], DATA["final"]["oa_aw_check"])


CACHE_IMG = os.path.join(P.CACHE, "img")


def write_data_js():
    """The pages read numbers from manual_data.js (a plain script, so it also works
    when the manual is opened straight from disk, where fetch() is blocked)."""
    d = json.load(open("docs/manual/manual_data.json"))
    with open("docs/manual/manual_data.js", "w") as f:
        f.write("window.MANUAL_DATA = " + json.dumps(d) + ";\n")


if __name__ == "__main__":
    os.makedirs(CACHE_IMG, exist_ok=True)
    if "--data-only" in sys.argv:
        write_data_js()
    else:
        main()
