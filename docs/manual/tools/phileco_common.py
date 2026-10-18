"""Shared loading for the manual's figures from the Bay, Laguna (Phileco) workspace.

Reads the saved workspace files directly (no Qt, no engine run): the ortho on the
10 cm working grid, segments.tif, segment_features.csv, classified_map.npy (the map the
accuracy report scored), session.json, groups.json and accuracy_points.gpkg.
Set PHILECO_DIR to the folder holding Phileco_ortho.tif, Phileco_dem.tif and
Phileco_ortho_workspace/. Results are cached in docs/manual/tools/.cache/.
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

ROOT = os.environ.get("PHILECO_DIR", r"D:/GXDX_Paper/02_Data/Aerial/Phileco_Leo")
ORTHO = os.path.join(ROOT, "Phileco_ortho.tif")
DSM = os.path.join(ROOT, "Phileco_dem.tif")
WS = os.path.join(ROOT, "Phileco_ortho_workspace")
CACHE = os.environ.get("MANUAL_CACHE", "docs/manual/tools/.cache")

BASE = ['r_mean', 'r_std', 'g_mean', 'g_std', 'b_mean', 'b_std',
        'area', 'perimeter', 'circularity', 'convexity',
        'contrast', 'homogeneity', 'entropy', 'energy',
        'exg_mean', 'vari_mean', 'ngrdi_mean', 'brightness_mean',
        'brightness_rel_neighbors']
DSM_FEATS = ['dsm_mean', 'dsm_std', 'slope_mean', 'dsm_rel_neighbors', 'hag_mean']


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def session():
    with open(os.path.join(WS, "session.json")) as f:
        return json.load(f)


def classes():
    """{name: '#rrggbb'} in class-list order (raster values 1..n)."""
    return {k: v["color"] for k, v in session()["classes"].items()}


def training():
    return {k: set(int(x) for x in v["training_segments"]) for k, v in session()["classes"].items()}


def segments():
    with rasterio.open(os.path.join(WS, "segments.tif")) as r:
        return r.read(1), r.transform, r.crs


def features():
    return pd.read_csv(os.path.join(WS, "segment_features.csv")).set_index("raster_val")


def feature_cols(df):
    return BASE + DSM_FEATS + [c for c in df.columns if c.startswith("deep_")]


def class_map():
    return np.load(os.path.join(WS, "classified_map.npy"))


def ortho_on_grid():
    """RGB (H, W, 3) uint8 on the segmentation grid, cached as .npy."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "phileco_rgb10.npy")
    seg, tr, _ = segments()
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    with rasterio.open(ORTHO) as src:
        a = src.read((1, 2, 3), out_shape=(3,) + seg.shape, resampling=Resampling.average)
        m = src.read_masks(1, out_shape=seg.shape, resampling=Resampling.nearest)
    rgb = np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
    rgb[m == 0] = 255
    np.save(p, rgb)
    return rgb


def dsm_on_grid():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "phileco_dsm10.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    from rasterio.warp import reproject
    seg, tr, crs = segments()
    out = np.full(seg.shape, np.nan, dtype=np.float32)
    with rasterio.open(DSM) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=tr, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    np.save(p, out)
    return out


def points():
    import geopandas as gpd
    g = gpd.read_file(os.path.join(WS, "accuracy_points.gpkg"))
    _, tr, _ = segments()
    r, c = rasterio.transform.rowcol(tr, g.geometry.x.values, g.geometry.y.values)
    g["row"], g["col"] = np.array(r), np.array(c)
    return g


def colorize(cls_raster, cls):
    out = np.full(cls_raster.shape + (3,), 255, dtype=np.uint8)
    for i, hx in enumerate(cls.values(), start=1):
        out[cls_raster == i] = hex2rgb(hx)
    return out
