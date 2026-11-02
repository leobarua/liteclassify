"""Builds the manual's data figures and manual_data.json / manual_data.js from the CAFS
worked example. Run from the repository root:

    python docs/manual/tools/build_figures_cafs.py

The map in every figure is the final refined map: the workspace's object map plus the
greenness refinement, with no sieve. The retraining charts are a simulated replay built from
the workspace's final labels (label them as simulated). See tools/README.md.
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
import cafs_common as P

IMG = "docs/manual/img"
DATA = {}

# Crops on the 5 cm working grid (x0, y0, x1, y1)
CROP_A = (7000, 3950, 9560, 5870)    # 128 x 96 m: crop beds, grass strips, bare beds, the road, trees, greenhouses
CROP_S = (8870, 4540, 9830, 5260)    # 48 x 36 m: greenhouses beside plots (pixels vs objects, SLIC series)
GALLERY = {"_ditch": (6450, 6150, 8050, 7350),      # 80 x 60 m: strips beside ploughed fields
           "_dying": (1100, 5450, 2700, 6650),      # dying grass in the south-west
           "_buildings": (4100, 8500, 5700, 9700),  # the buildings and their yard
           "_pond": (1600, 4500, 3200, 5700)}       # open water, floating mats and banks
ROWS = (5058, 2392, 5658, 2992)      # 30 x 30 m: a row field for the greenness figure
RES_AT = (2500, 5100)                # row, col of the resolution gallery's corner (the row field)
HERO = (800, 600)
SEG_N = {"k30000": 30000, "k67536": 67536, "k150000": 150000}
CHIP_GROUPS = {"crop1": "crop 1", "crop2": "crop 2", "building": "building", "mud": "mud"}


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


def main():
    s = P.session()
    cls = P.classes()
    names = list(cls)
    ws_names = list(s["classes"])
    train = P.training()
    seg, tr, crs = P.segments()
    rgb, valid = P.ortho_on_grid()
    rgb = np.asarray(rgb)
    valid = np.asarray(valid)
    obj = P.object_map()
    cmap = P.final_map()
    df = P.features()
    feats = P.feature_cols(df)
    px = float(s["effective_pixel_size"])
    H, W = seg.shape
    g = P.points()
    pr, pc = g["row"].values, g["col"].values

    with rasterio.open(P.ortho_path()) as src:
        native_px, nW, nH = abs(src.transform.a), src.width, src.height
    with rasterio.open(P.dsm_path()) as src:
        dsm_px = abs(src.transform.a)
    area_ha = float((seg > 0).sum() * px * px / 1e4)

    DATA["classes"] = [[k, v] for k, v in cls.items()]
    DATA["grid"] = {"width": W, "height": H, "pixel_m": px, "native_pixel_m": native_px,
                    "native_width": nW, "native_height": nH, "dsm_pixel_m": dsm_px,
                    "feature_pixel_m": px, "area_ha": area_ha}
    DATA["segmentation"] = {"requested": s["n_segments"], "n_segments": s["n_segments"],
                            "compactness": s["compactness"], "sigma": s["sigma"],
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
                        "backbone": gj["settings"].get("backbone"), "by_name": {}}
    for gk, v in groups.items():
        if v["status"] == "named":
            DATA["discover"]["by_name"].setdefault(P.DISPLAY.get(v["name"], v["name"]), []).append(gk)

    # ---------- the survey ----------
    k = 8
    ov = rgb[::k, ::k]
    save(draw_scalebar(Image.fromarray(ov.copy()), 100, 1 / (px * k), "100 m"), "sample_ortho.jpg")
    dsm = P.dsm_on_grid()
    dk = np.asarray(dsm[::k, ::k]).copy()
    dk[~valid[::k, ::k]] = np.nan
    okd = np.isfinite(dk)
    lo, hi = np.percentile(dk[okd], (2, 98))
    col, _, _ = colorize_values(dk, "terrain", lo, hi)
    hs = hillshade(dk, px * k)
    shaded = (col * (0.45 + 0.55 * hs[..., None])).astype(np.uint8)
    shaded[~okd] = 255
    save(draw_scalebar(Image.fromarray(shaded), 100, 1 / (px * k), "100 m"), "sample_dsm.jpg")
    DATA["dsm"] = {"p2": float(lo), "p98": float(hi), "min": float(np.nanmin(dk)), "max": float(np.nanmax(dk))}

    # native-resolution inset: 30 x 30 m over the row field
    cx, cy = rasterio.transform.xy(tr, (ROWS[1] + ROWS[3]) // 2, (ROWS[0] + ROWS[2]) // 2)
    with rasterio.open(P.ortho_path()) as src:
        r, c = src.index(cx, cy)
        n = int(round(30 / native_px))
        a = src.read((1, 2, 3), window=rasterio.windows.Window(c - n // 2, r - n // 2, n, n))
    im = Image.fromarray(np.transpose(a, (1, 2, 0))).resize((760, 760), Image.LANCZOS)
    save(draw_scalebar(im, 5, 760 / 30, "5 m"), "sample_inset_native.jpg")

    # ---------- the map, crops ----------
    cls_rgb_full = P.colorize(cmap, cls)
    cr, cs = crop(rgb, CROP_A), crop(seg, CROP_A)
    bnd = find_boundaries(cs, mode="inner")
    rb = cr.copy(); rb[bnd] = (0.45 * rb[bnd] + 0.55 * np.array([40, 90, 255])).astype(np.uint8)
    save(cr, "view_raw.jpg", size=(1200, 900))
    save(crop(cls_rgb_full, CROP_A), "view_classified.png", size=(1200, 900))
    save(rb, "view_raw_borders.jpg", size=(1600, 1200), quality=80)
    save(mean_colour(cr, cs), "view_segmented.jpg", size=(1200, 900))
    for tag, box in GALLERY.items():
        save(crop(rgb, box), f"view_raw{tag}.jpg", size=(1200, 900))
        save(crop(cls_rgb_full, box), f"view_classified{tag}.png", size=(1200, 900))

    # reference points over the hero crop
    kx = HERO[0] / (CROP_A[2] - CROP_A[0])
    im = Image.fromarray(crop(rgb, CROP_A)).resize(HERO, Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for r_, c_, ref in zip(pr, pc, g["reference_class"]):
        x, y = (c_ - CROP_A[0]) * kx, (r_ - CROP_A[1]) * kx
        if 0 <= x < HERO[0] and 0 <= y < HERO[1]:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=P.hex2rgb(cls[ref]), outline=(255, 255, 255), width=3)
    save(im, "view_points.jpg")

    # pixels vs objects
    cr_s, cs_s = crop(rgb, CROP_S), crop(seg, CROP_S)
    b = find_boundaries(cs_s, mode="inner")
    x = cr_s.copy(); x[b] = (255, 236, 64)
    save(cr_s, "obj_pixels.jpg", size=(720, 540))
    save(x, "obj_boundaries.jpg", size=(720, 540))
    y = mean_colour(cr_s, cs_s); y[b] = (255, 255, 255)
    save(y, "obj_means.jpg", size=(720, 540))

    # ---------- SLIC parameter series on CROP_S (CPU engine, this crop alone) ----------
    from scipy.ndimage import gaussian_filter
    img = cr_s.astype(np.float64) / 255.0
    frac = (CROP_S[2] - CROP_S[0]) * (CROP_S[3] - CROP_S[1]) / (seg > 0).sum()
    series = {}
    used = s["n_segments"]
    runs = {tag: (n, .15, 2.5) for tag, n in SEG_N.items()}
    runs.update({"m005": (used, .05, 2.5), "m050": (used, .5, 2.5), "s0": (used, .15, 0), "s6": (used, .15, 6.0)})
    for tag, (n, c, sg) in runs.items():
        im_s = gaussian_filter(img, sigma=[sg, sg, 0]) if sg > 0 else img
        lab = slic(im_s, n_segments=max(2, int(round(n * frac))), compactness=c * 100.0, sigma=0, start_label=1)
        series[tag] = {"requested": n, "crop_segments": int(lab.max()),
                       "mean_area_m2": float(lab.size * px * px / lab.max())}
        out = cr_s.copy(); out[find_boundaries(lab, mode="inner")] = (255, 236, 64)
        save(out, f"seg_{tag}.jpg", size=(720, 540))
    DATA["seg_series"] = series

    # ---------- feature maps (CROP_A) ----------
    explorer = P.BASE + P.DSM_FEATS + ["deep_5m_pc1", "deep_10m_pc1"]
    ids = np.unique(cs); ids = ids[ids > 0]
    sub = df.loc[df.index.intersection(ids)]
    lutn = np.full(int(seg.max()) + 1, np.nan, dtype=np.float32)
    fmeta = {}
    for f in explorer:
        lutn[:] = np.nan
        lutn[sub.index.values] = sub[f].values
        m = lutn[cs]
        center = 0.0 if f.endswith("rel_neighbors") else None
        cm = "RdBu_r" if center is not None else ("cividis" if f.startswith("dsm") or f in ("slope_mean", "hag_mean") else
                                                  ("magma" if f.startswith("deep_") else "viridis"))
        im_f, lo_, hi_ = colorize_values(m, cmap=cm, center=center)
        save(im_f, f"feat/{f}.jpg", size=(640, 480), quality=80)
        fmeta[f] = {"lo": lo_, "hi": hi_, "cmap": cm}
    c3 = cr.astype(np.float32)
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
    dc = crop(dsm, CROP_A).copy()
    dc[~crop(valid, CROP_A)] = np.nan
    im_f, lo_, hi_ = colorize_values(dc, "cividis"); save(im_f, "dsm_crop.jpg", size=HERO)
    DATA["dsm_crop"] = {"lo": lo_, "hi": hi_}
    gy, gx = np.gradient(np.where(np.isfinite(dc), dc, np.nanmedian(dc)), px)
    slope_ = np.degrees(np.arctan(np.hypot(gx, gy))); slope_[~np.isfinite(dc)] = np.nan
    im_f, _, _ = colorize_values(slope_, "magma", 0, 60)
    save(im_f, "slope_crop.jpg", size=HERO)
    hs_c = hillshade(dc, px)
    save((cr * 0.4 + (hs_c[..., None] * 255) * 0.6).astype(np.uint8), "hillshade_overlay.jpg", size=HERO)

    # ---------- GLCM worked example: a tree crown vs a smooth roof, on the working grid ----------
    lut_cls = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
    ys, xs = np.nonzero(seg)
    lut_cls[seg[ys, xs]] = obj[ys, xs]   # the object map assigns whole segments
    del ys, xs
    cand = df[(df["area"] > 1.5) & (df["area"] < 3.2) & (df["convexity"] > 0.8)]
    tree_c = cand[lut_cls[cand.index.values] == names.index("Tree") + 1]
    roof_c = cand[lut_cls[cand.index.values] == names.index("Building") + 1]
    # a typical crown and a typical smooth roof (nearest the median contrast of each)
    roof_c = roof_c[roof_c["contrast"] > 0.02]
    crown = (tree_c["contrast"] - tree_c["contrast"].median()).abs().idxmin()
    roof = (roof_c["contrast"] - roof_c["contrast"].median()).abs().idxmin()
    glcm_out = {}
    sl = ndimage.find_objects(seg)
    for tag, sid in (("tree", crown), ("roof", roof)):
        rs, cs_ = sl[sid - 1]
        pad = 6
        r0, r1, c0, c1 = max(0, rs.start - pad), rs.stop + pad, max(0, cs_.start - pad), cs_.stop + pad
        a = rgb[r0:r1, c0:c1].astype(np.float32)
        mask = seg[r0:r1, c0:c1] == sid
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
        save(view, f"glcm_{tag}.png", size=(view.shape[1] * 5, view.shape[0] * 5))
        row = df.loc[sid]
        glcm_out[tag] = {"segment": int(sid), "matrix": np.round(Pm[1:, 1:], 4).tolist(),
                         "contrast": float(row["contrast"]), "homogeneity": float(row["homogeneity"]),
                         "energy": float(row["energy"]), "entropy": float(row["entropy"]),
                         "area": float(row["area"]), "pixels": int(mask.sum())}
    DATA["glcm"] = glcm_out

    # ---------- scene-wide: training samples, groups, reference points, final map ----------
    sk = seg[::k, ::k]
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
        if sid < len(glut):
            glut[sid] = gkeys.index(gk) + 1
    gcol = np.array([(255, 255, 255)] + [P.hex2rgb(groups[gk]["color"]) for gk in gkeys], dtype=np.uint8)
    gm = glut[sk]
    out = gcol[gm].astype(np.float32) * 0.7 + ov * 0.3
    out[gm == 0] = base[gm == 0]
    save(out.astype(np.uint8), "disc_groups_scene.jpg")
    gmc = glut[cs]
    out = gcol[gmc].astype(np.float32) * 0.55 + cr * 0.45
    save(out.astype(np.uint8), "disc_groups_crop.jpg", size=HERO)
    DATA["discover"]["sizes"] = {gk: {"segments": gcount.get(gk, 0), "area_share": garea.get(gk, 0) / float(area.sum()),
                                      "name": P.DISPLAY.get(groups[gk]["name"], groups[gk]["name"]),
                                      "status": groups[gk]["status"], "color": groups[gk]["color"]} for gk in gkeys}

    # chips for four named groups: 5 m patches around segment centroids, like the app's tiles
    half = int(round(2.5 / px))
    rng = np.random.default_rng(1)
    picked = {}
    for fname, cname in CHIP_GROUPS.items():
        cands = [gk for gk, v in groups.items() if v["status"] == "named" and v["name"] == cname]
        gk = max(cands, key=lambda q: gcount.get(q, 0))
        picked[fname] = gk
        members = [sid for sid, v in seg_group.items() if v == gk]
        pick = rng.choice(members, size=min(8, len(members)), replace=False)
        cents = ndimage.center_of_mass(np.ones(seg.shape, np.uint8), seg, pick)
        strip = Image.new("RGB", (8 * 120 + 7 * 4, 120), (30, 30, 30))
        for i_, (sid, cen) in enumerate(zip(pick, cents)):
            r_, c_ = (int(round(v)) for v in cen)
            r0, c0 = max(0, r_ - half), max(0, c_ - half)
            chip = rgb[r0:r0 + 2 * half, c0:c0 + 2 * half].copy()
            m = seg[r0:r0 + 2 * half, c0:c0 + 2 * half] == sid
            chip[find_boundaries(m, mode="inner")] = (0, 255, 255)
            strip.paste(Image.fromarray(chip).resize((120, 120), Image.LANCZOS), (i_ * 124, 0))
        save(strip, f"disc_chips_{fname}.jpg")
    DATA["discover"]["chip_groups"] = picked

    mp = np.array([names[v - 1] if v else "" for v in cmap[pr, pc]])
    ref = g["reference_class"].values.astype(str)
    strat = g["stratum"].values.astype(str)
    im = Image.fromarray(ov.copy()); d = ImageDraw.Draw(im)
    for r_, c_, rf, pd_ in zip(pr, pc, ref, mp):
        x, y = c_ / k, r_ / k
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=P.hex2rgb(cls[rf]), outline=(255, 255, 255), width=1)
        if rf != pd_:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 0, 200), width=2)
    save(im, "reference_points.jpg")
    save(cls_rgb_full[::k, ::k], "final_map.png")
    mapped_total = (cmap > 0).sum()
    DATA["map_share"] = {nm: float((cmap == i).sum() / mapped_total) for i, nm in enumerate(names, start=1)}
    DATA["map_ha"] = {nm: float((cmap == i).sum() * px * px / 1e4) for i, nm in enumerate(names, start=1)}

    # ---------- accuracy (the app's own assessment and matrix figure) ----------
    from obia_engine import ObiaEngine
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QColor
    app = QApplication.instance() or QApplication([])
    import obia_ui
    import geopandas as gpd
    e = ObiaEngine()
    e.ortho_meta = {"transform": tr, "crs": crs}
    e.segmented_image = seg
    strata_ws = s["accuracy_strata"]
    strata_px = {P.DISPLAY[h]: int(v) for h, v in strata_ws["pixels"].items()}
    e.accuracy_strata = dict(strata_ws, pixels=strata_px)
    pts = gpd.GeoDataFrame({"point_id": g["point_id"].values, "reference_class": ref, "source": g["source"].values,
                            "predicted_class": [None] * len(g), "stratum": strat}, geometry=g.geometry.values, crs=g.crs)
    e.accuracy_points_gdf = pts
    tmpd = os.path.join(CACHE_IMG, "assess")
    os.makedirs(tmpd, exist_ok=True)
    rep = e.run_accuracy_assessment(cmap, names, tmpd)
    st = rep["stratified"]
    z = 1.96
    mat = rep.get("matrix", rep.get("confusion_matrix"))
    est = {n: {"mapped_ha": float(st["mapped_ha"][n]), "est_ha": float(st["area_prop"][n] * st["total_ha"]),
               "ci_ha": float(z * st["area_prop_se"][n] * st["total_ha"]),
               "est_ci_ha": float(z * st["area_prop_se"][n] * st["total_ha"]),
               "ua": float(st["ua"][n]), "ua_ci": float(z * st["ua_se"][n]),
               "pa": float(st["pa"][n]), "pa_ci": float(z * st["pa_se"][n])} for n in names}
    M = mat.loc[names, names].values.astype(int).tolist()
    DATA["final"] = {"matrix": M, "rows": names, "cols": names, "n": int(rep["n_labeled"]), "oa": float(rep["overall_accuracy"]),
                     "ua": {n: M[i][i] / max(1, sum(M[i])) for i, n in enumerate(names)},
                     "pa": {n: M[i][i] / max(1, sum(r[i] for r in M)) for i, n in enumerate(names)},
                     "oa_aw": float(st["oa"]), "oa_aw_ci": float(z * st["oa_se"]), "est": est}
    DATA["reference_counts"] = {n: int((ref == n).sum()) for n in names}
    DATA["strata"] = {"design": "stratified", "min_per_class": 50, "seed": strata_ws["seed"], "pixels": strata_px,
                      "n_strata": len(strata_px), "points_per_stratum": {h: int((strat == h).sum()) for h in strata_px}}
    DATA["final"]["oa_aw_check"] = stratified_estimates(ref, mp, strat, strata_px, names)[0]
    qcol = {n: QColor(v) for n, v in cls.items()}
    obia_ui.render_confusion_matrix_image(rep, qcol).save(os.path.join(CACHE_IMG, "matrix.png"))
    save(Image.open(os.path.join(CACHE_IMG, "matrix.png")).convert("RGB"), "accuracy_matrix.jpg", quality=90)

    # ---------- greenness refinement ----------
    rf = s["greenness_refinement"]
    CUT, WIN = float(rf["cutoff"]), int(rf["window"])
    i_g, i_b = names.index("Grass") + 1, names.index("Bare") + 1
    e.load_raster(P.ortho_path(), out_shape=seg.shape)   # the ortho as the app holds it, no-data included
    e.segmented_image = seg
    es = np.vstack([e._exg_strip(r0, min(H, r0 + 1024), WIN) for r0 in range(0, H, 1024)])
    green = es > CUT
    n_px = np.bincount(seg.ravel(), minlength=int(seg.max()) + 1)
    n_gr = np.bincount(seg.ravel(), weights=green.ravel().astype(np.float64), minlength=len(n_px))
    share = np.where(n_px > 0, n_gr / np.maximum(n_px, 1), 1.0)
    drop = share < 0.25
    check, info = e.refine_by_greenness(obj, i_g, i_b, CUT, WIN, mode=rf["mode"])
    print("refined map reproduces the final map:", np.array_equal(check, cmap), info)
    e.features_df = df
    sug = e.suggest_greenness_cutoff(train["Grass"], train["Bare"])[0]
    e.ortho_meta = {"transform": tr, "crs": crs}
    mp_obj = np.array([names[v - 1] if v else "" for v in obj[pr, pc]])
    oa_before = stratified_estimates(ref, mp_obj, strat, strata_px, names)
    oa_after = stratified_estimates(ref, mp, strat, strata_px, names)
    DATA["greenness"] = {"cutoff": CUT, "window": WIN, "mode": rf["mode"], "suggested_cutoff": float(sug),
                         "oa_before": oa_before[0], "oa_after": oa_after[0],
                         "to_grass_share": info["to_green"] / info["total"], "to_bare_share": info["to_other"] / info["total"],
                         "grass_found_before": oa_before[2]["Grass"], "grass_found_after": oa_after[2]["Grass"]}
    panels = [("Orthomosaic", crop(rgb, ROWS)), ("Object map", P.colorize(crop(obj, ROWS), cls)),
              ("Refined map", crop(cls_rgb_full, ROWS))]
    sz = ROWS[2] - ROWS[0]
    strip = Image.new("RGB", (3 * sz + 2 * 12, sz + 34), "white")
    dr = ImageDraw.Draw(strip)
    for i_, (t_, a_) in enumerate(panels):
        strip.paste(Image.fromarray(a_), (i_ * (sz + 12), 34))
        dr.text((i_ * (sz + 12) + 4, 7), t_, fill=(20, 20, 20), font=font(20))
    save(strip, "greenness_rows.jpg", quality=88)

    def apply_refine(obj_lut):
        """The hybrid rule on a whole-segment map, at the given points or thumbnails."""
        def at(rows, cols):
            sid = seg[rows, cols]
            base_ = obj_lut[sid]
            m = base_.copy()
            m[(base_ == i_b) & green[rows, cols]] = i_g
            m[(base_ == i_g) & drop[sid]] = i_b
            return m
        return at

    # ---------- simulated retraining replay (area-weighted OA on the stratified sample) ----------
    from sklearn.ensemble import RandomForestClassifier
    X_all = df[feats].fillna(0).values
    pos = {sid: i for i, sid in enumerate(df.index)}
    lab_of = {sid: n for n, v in train.items() for sid in v if sid in pos}
    pool = np.array(sorted(lab_of)); pool_y = np.array([lab_of[i] for i in pool])
    START, ADD, ROUNDS, SEEDS = 5, 8, 8, 10
    seg_lut_size = int(seg.max()) + 1

    def fit_predict(used_):
        ids_ = [sid for v in used_.values() for sid in v]
        ys_ = [n for n, v in used_.items() for _ in v]
        order = np.argsort([pos[i] for i in ids_])     # the app merges in feature-table order
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_all[[pos[ids_[o]] for o in order]], np.array(ys_)[order])
        return clf

    def seg_lut(clf):
        lut_r = np.zeros(seg_lut_size, dtype=np.uint8)
        lut_r[df.index.values] = [names.index(q) + 1 for q in clf.predict(X_all)]
        return lut_r

    def to_names(codes):
        return np.array([names[v - 1] if v else "" for v in codes])

    replays = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        used_ = {n: set(rng.choice(sorted(v & set(pos)), size=min(START, len(v)), replace=False).tolist()) for n, v in train.items()}
        rows = []
        for rnd in range(1, ROUNDS + 1):
            clf = fit_predict(used_)
            lut_r = seg_lut(clf)
            pred_pts = to_names(apply_refine(lut_r)(pr, pc))
            oa, ua, pa = stratified_estimates(ref, pred_pts, strat, strata_px, names)
            row = {"round": rnd, "samples": sum(len(v) for v in used_.values()), "oa": oa, "pa": pa}
            if seed == 0 and rnd in (1, 3, ROUNDS):
                row["map"] = lut_r
            rows.append(row)
            pred_pool = clf.predict(X_all[[pos[i] for i in pool]])
            wrong = [(sid, yy) for sid, yy, pp in zip(pool, pool_y, pred_pool) if pp != yy and sid not in used_[yy]]
            added = 0
            for n in names:
                cn = [sid for sid, yy in wrong if yy == n]
                rng.shuffle(cn)
                used_[n] |= set(int(v) for v in cn[:ADD]); added += min(ADD, len(cn))
            if added == 0:
                break
        replays.append(rows)
        print("replay", seed, [(r["samples"], round(r["oa"], 3)) for r in rows])
    sim = []
    for rnd in range(1, ROUNDS + 1):
        rr = [r[min(rnd, len(r)) - 1] for r in replays]
        oas = [r["oa"] for r in rr]
        sim.append({"round": rnd, "samples": float(np.mean([r["samples"] for r in rr])),
                    "oa_mean": float(np.mean(oas)), "oa_min": float(np.min(oas)), "oa_max": float(np.max(oas)),
                    "pa_mean": {n: float(np.nanmean([r["pa"][n] for r in rr])) for n in names}})
    full = fit_predict({n: v & set(pos) for n, v in train.items()})
    full_lut = seg_lut(full)
    print("refit reproduces the object map on %.2f%% of mapped pixels" % (100 * (full_lut[seg] == obj)[seg > 0].mean()))
    oa_full, _, pa_full = stratified_estimates(ref, to_names(apply_refine(full_lut)(pr, pc)), strat, strata_px, names)
    DATA["simulation"] = {"start_per_class": START, "add_per_class": ADD, "seeds": SEEDS, "rounds": sim,
                          "final_full": {"samples": DATA["training"]["total"], "oa": oa_full, "pa": pa_full},
                          "metric": f"area-weighted overall accuracy on the {len(g)} stratified points"}
    first = replays[0][0]
    DATA["sparse_example"] = {"oa": first["oa"], "samples": first["samples"]}

    def thumb(lut_r, slice_r, slice_c):
        rows_, cols_ = np.mgrid[slice_r, slice_c]
        return apply_refine(lut_r)(rows_, cols_)
    for row in replays[0]:
        if "map" in row:
            save(P.colorize(thumb(row["map"], slice(0, H, k), slice(0, W, k)), cls), f"sim_round_{row['round']}.png")
    DATA["simulation"]["map_rounds"] = [r["round"] for r in replays[0] if "map" in r]
    x0, y0, x1, y1 = CROP_A
    save(P.colorize(thumb(replays[0][0]["map"], slice(y0, y1), slice(x0, x1)), cls), "sparse_zoom.png", size=HERO)
    save(crop(cls_rgb_full, CROP_A), "full_zoom.png", size=HERO)

    # ---------- sieve ----------
    sv_rows = []
    for t in (0, 1000, 2000, 4000):
        sv = cmap if t == 0 else e.apply_sieve_filter(cmap, t)
        oa, _, _ = stratified_estimates(ref, to_names(sv[pr, pc]), strat, strata_px, names)
        sv_rows.append({"mmu_px": t, "mmu_m2": t * px * px, "oa": oa, "changed_px": float((sv != cmap)[cmap > 0].mean())})
        if t == 4000:
            save(crop(P.colorize(sv, cls), CROP_A), "sieve_4000.png", size=HERO)
    save(crop(cls_rgb_full, CROP_A), "sieve_0.png", size=HERO)
    DATA["sieve"] = sv_rows
    DATA["mmu_px"] = 0

    # ---------- merge demo: one roof's pieces, by region growing over Building segments ----------
    cm_s = crop(lut_cls[seg], CROP_S)
    roofs = [sid for sid in np.unique(cs_s) if sid and np.all(cm_s[cs_s == sid] == names.index("Building") + 1)
             and df.loc[sid, "hag_mean"] > 1.0]
    adj = {}
    for a_, b_ in ((cs_s[:, :-1], cs_s[:, 1:]), (cs_s[:-1, :], cs_s[1:, :])):
        mm = a_ != b_
        for u, v in zip(a_[mm], b_[mm]):
            adj.setdefault(int(u), set()).add(int(v)); adj.setdefault(int(v), set()).add(int(u))
    rs = set(int(v) for v in roofs)
    seen, best = set(), None
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
    before = cr_s.copy(); before[find_boundaries(cs_s, mode="inner")] = (255, 236, 64)
    sel = np.isin(cs_s, list(best))
    before[sel] = (before[sel] * 0.55 + np.array([0, 255, 255]) * 0.45).astype(np.uint8)
    save(before, "merge_before.jpg", size=(720, 540))
    merged = cs_s.copy(); merged[sel] = min(best)
    after = cr_s.copy(); after[find_boundaries(merged, mode="inner")] = (255, 236, 64)
    save(after, "merge_after.jpg", size=(720, 540))
    DATA["merge"] = {"pieces": len(best), "area_m2": float(df.loc[list(best), "area"].sum())}

    # ---------- preparation: resolutions, colour correction ----------
    with rasterio.open(P.ortho_path()) as src:
        n = int(round(30 / native_px)); m_ = int(round(22 / native_px))
        r, c = src.index(*rasterio.transform.xy(tr, *RES_AT))
        win = rasterio.windows.Window(c, r, n, m_)
        for tag, f_ in (("native", 1), ("5cm", native_px / 0.05), ("20cm", native_px / 0.20)):
            a = src.read((1, 2, 3), window=win, out_shape=(3, max(1, int(m_ * f_)), max(1, int(n * f_))),
                         resampling=rasterio.enums.Resampling.average)
            save(Image.fromarray(np.transpose(a, (1, 2, 0))).resize((750, 550), Image.NEAREST), f"res_{tag}.jpg")
    p2, p98 = np.percentile(ov[valid[::k, ::k]], (2, 98))
    save(cr, "color_before.jpg", size=HERO)
    save(np.clip((c3 - p2) / (p98 - p2) * 255, 0, 255), "color_auto.jpg", size=HERO)

    # ---------- separability for the current training set (the app's own code) ----------
    # Rendered with the normal Qt platform (offscreen has no fonts). The table + closest
    # pairs go in once; each pair's signature plot is cropped out for the gallery.
    from itertools import combinations
    res = e.compute_class_separability({n: list(v) for n, v in train.items()}, features=feats)
    jm = res["jm"]
    pairs = []
    for i_, (a_, b_) in enumerate(combinations(names, 2)):
        qi = obia_ui.render_separability_image(res, qcol, focus_pair=(a_, b_), scale=2)
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
            save(top.resize((tw, int(top.height * tw / top.width)), Image.LANCZOS), "separability_table.jpg")
        bot = im.crop((0, cut, im.width, im.height))
        name = f"sep/pair_{i_ + 1:02d}.jpg"
        save(bot.resize((1800, int(bot.height * 1800 / bot.width)), Image.LANCZOS), name, quality=82)
        pairs.append({"a": a_, "b": b_, "img": name, "jm": float(jm.loc[a_, b_]),
                      "top": list(res["jm_feature"][(a_, b_)].index[:5])})
    pairs.sort(key=lambda q: q["jm"])
    DATA["separability"] = {"pairs": pairs}

    json.dump(DATA, open("docs/manual/manual_data.json", "w"), indent=1, default=float)
    write_data_js()
    print(json.dumps({k_: DATA[k_] for k_ in ("grid", "segmentation", "training", "features", "sieve", "merge", "greenness",
                                              "sparse_example", "seg_series", "dsm", "glcm_summary" if False else "mmu_px")}, indent=1, default=float))
    print("glcm", {t_: {q_: round(v_[q_], 3) for q_ in ("contrast", "homogeneity", "energy", "entropy", "area")} for t_, v_ in glcm_out.items()})
    print("simulation", [(round(r["samples"]), round(r["oa_mean"], 3), round(r["oa_min"], 3), round(r["oa_max"], 3))
                         for r in DATA["simulation"]["rounds"]], DATA["simulation"]["final_full"]["oa"])
    print("sim PA round1 vs full", {n: (round(sim[0]["pa_mean"][n], 2), round(pa_full[n], 2)) for n in names})
    print("sep closest", [(q["a"], q["b"], round(q["jm"], 3), q["top"][:3]) for q in pairs[:6]])
    print("groups", DATA["discover"]["by_name"], picked)
    print("OA check", DATA["final"]["oa_aw"], DATA["final"]["oa_aw_check"], "full replay", oa_full)


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
