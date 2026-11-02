"""Shared loading for the manual's figures from the CAFS worked example.

Reads the saved workspace, its final reference points and the final refined map. Nothing
is written to those folders; results are cached in docs/manual/tools/.cache/.
Set CAFS_WS to the workspace folder and CAFS_RESULTS to the folder holding
final_reference_points.gpkg and final_map_v2.npy.
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

_BASE_DIR = "experiments/manual_v12/stratified_draft"
WS = os.environ.get("CAFS_WS", os.path.join(_BASE_DIR, "stratified_draft_v2", "CAFS_Ortho_workspace"))
RESULTS = os.environ.get("CAFS_RESULTS", os.path.join(_BASE_DIR, "results_v2_final"))
CACHE = os.environ.get("MANUAL_CACHE", "docs/manual/tools/.cache")

# Workspace class name -> the name the manual prints
DISPLAY = {"bare": "Bare", "crop 1": "Crop 1", "crop 2": "Crop 2", "grass": "Grass", "water": "Water",
           "tree": "Tree", "road": "Road", "building": "Building", "mud": "Mud"}

from obia_engine import ObiaEngine  # noqa: E402

BASE = list(ObiaEngine.BASE_FEATURES)     # the app's order: the forest's result depends on it
DSM_FEATS = list(ObiaEngine.DSM_FEATURES)


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def session():
    with open(os.path.join(WS, "session.json")) as f:
        return json.load(f)


def classes():
    """{display name: '#rrggbb'} in class-list order (raster values 1..n)."""
    return {DISPLAY[k]: v["color"] for k, v in session()["classes"].items()}


def training():
    return {DISPLAY[k]: set(int(x) for x in v["training_segments"]) for k, v in session()["classes"].items()}


def segments():
    with rasterio.open(os.path.join(WS, "segments.tif")) as r:
        return r.read(1), r.transform, r.crs


def features():
    return pd.read_csv(os.path.join(WS, "segment_features.csv")).set_index("raster_val")


def feature_cols(df):
    return BASE + DSM_FEATS + [c for c in df.columns if c.startswith("deep_")]


def object_map():
    """The classifier's map before the greenness refinement (what the workspace stores)."""
    return np.load(os.path.join(WS, "classified_map.npy"))


def final_map():
    return np.load(os.path.join(RESULTS, "final_map_v2.npy"))


def ortho_path():
    return session()["ortho_path"]


def dsm_path():
    return session()["dsm_path"]


def ortho_on_grid():
    """RGB (H, W, 3) uint8 on the working grid, as the app loads it, with no-data white.
    Returns (rgb, valid mask); cached as .npy."""
    os.makedirs(CACHE, exist_ok=True)
    p, pm = os.path.join(CACHE, "cafs_rgb5.npy"), os.path.join(CACHE, "cafs_valid5.npy")
    if os.path.exists(p) and os.path.exists(pm):
        return np.load(p, mmap_mode="r"), np.load(pm, mmap_mode="r")
    seg, _, _ = segments()
    e = ObiaEngine()
    e.load_raster(ortho_path(), out_shape=seg.shape)
    rgb = np.ascontiguousarray(np.transpose(e.original_ortho_data, (1, 2, 0)))
    valid = np.asarray(e.valid_mask)
    rgb[~valid] = 255
    np.save(p, rgb)
    np.save(pm, valid)
    return rgb, valid


def dsm_on_grid():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "cafs_dsm5.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    from rasterio.warp import reproject
    seg, tr, crs = segments()
    out = np.full(seg.shape, np.nan, dtype=np.float32)
    with rasterio.open(dsm_path()) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=tr, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    np.save(p, out)
    return out


def points():
    """The final reference points, with class names as the manual prints them."""
    import geopandas as gpd
    g = gpd.read_file(os.path.join(RESULTS, "final_reference_points.gpkg"))
    _, tr, _ = segments()
    r, c = rasterio.transform.rowcol(tr, g.geometry.x.values, g.geometry.y.values)
    g["row"], g["col"] = np.array(r), np.array(c)
    g["reference_class"] = g["reference_class"].map(DISPLAY)
    g["stratum"] = g["stratum"].map(DISPLAY)
    return g


def colorize(cls_raster, cls):
    out = np.full(cls_raster.shape + (3,), 255, dtype=np.uint8)
    for i, hx in enumerate(cls.values(), start=1):
        out[cls_raster == i] = hex2rgb(hx)
    return out
