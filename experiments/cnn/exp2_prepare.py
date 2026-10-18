"""Experiment 2, step 1 - forest sample: segment + features at 0.2 m (engine, as the app does),
local canopy relief from the DSM, and CNN context-patch embeddings cut from the 0.1 m and
0.2 m orthos. Everything is cached in out/forest.pkl. Run from the repo root."""
import os, sys, time, pickle
import numpy as np, rasterio
from scipy import ndimage

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from obia_engine import ObiaEngine
from embed import load_backbone, centroids, embed

D = "data/CNN_Sample"
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
SEG_M2 = 40.0          # target mean segment size (m2)
RELIEF_WIN_M = 40.0    # moving-median window for removing terrain from the DSM

e = ObiaEngine()
e.load_raster(f"{D}/CNNSample_Ortho_0.2.tif")
e.load_dsm(f"{D}/CNNSample_DSM_0.2.tif")
px = e.effective_pixel_size
valid_m2 = e.valid_mask.sum() * px * px
n_seg = int(valid_m2 / SEG_M2)
print(f"valid area {valid_m2 / 1e4:.1f} ha -> n_segments {n_seg}")

t = time.time(); e.segment_image(n_seg, 0.15, 2.5, force_device="gpu"); t_seg = time.time() - t
t = time.time()
e.vectorize_segments()
e.compute_all_features_parallel()
e.build_adjacency_graph()
e.compute_neighbor_relative_features()
t_feat = time.time() - t
seg = e.segmented_image
ids, cents = centroids(seg)
print(f"{len(ids)} segments, seg {t_seg:.0f}s, features {t_feat:.0f}s")

# Local canopy relief: DSM minus a moving median (~40 m) computed on a 2 m grid.
dsm = e.dsm_data.copy()
bad = (dsm < -1000) | ~e.valid_mask
dsm[bad] = np.nan
f = 10
h, w = dsm.shape
H, W = h // f * f, w // f * f
coarse = np.nanmedian(dsm[:H, :W].reshape(H // f, f, W // f, f), axis=(1, 3))
nan_c = np.isnan(coarse)
if nan_c.any():
    idx = ndimage.distance_transform_edt(nan_c, return_distances=False, return_indices=True)
    coarse = coarse[tuple(idx)]
win = int(RELIEF_WIN_M / (px * f)) | 1
trend = ndimage.median_filter(coarse, size=win)
trend = ndimage.zoom(trend, (h / trend.shape[0], w / trend.shape[1]), order=1)[:h, :w]
relief = dsm - trend
lab = seg.copy(); lab[bad] = 0
relief0 = np.nan_to_num(relief)
rel_mean = np.array(ndimage.mean(relief0, lab, ids))
rel_std = np.array(ndimage.standard_deviation(relief0, lab, ids))
rel_max = np.array(ndimage.maximum(relief0, lab, ids))

# CNN embeddings: patches cut around each centroid from the 0.1 m and 0.2 m orthos.
xs, ys = rasterio.transform.xy(e.ortho_meta["transform"], cents[:, 0], cents[:, 1])
orthos = {"0.2": (e.ortho_data, cents)}
with rasterio.open(f"{D}/CNNSample_Ortho_0.1.tif") as s:
    o01 = s.read((1, 2, 3))
    r, c = rasterio.transform.rowcol(s.transform, xs, ys, op=float)
    orthos["0.1"] = (o01, np.c_[r, c])

embs, t_emb = {}, {}
for bb in ("dinov2_s", "resnet50"):
    model = load_backbone(bb)
    for res, (img, cc) in orthos.items():
        p = float(res)
        for m in (5, 10, 20):
            t = time.time()
            embs[f"{bb}_{res}_{m}m"] = embed(img, ids, cc, int(round(m / p)), model)
            t_emb[f"{bb}_{res}_{m}m"] = time.time() - t
            print("embedded", bb, res, m, f"{t_emb[f'{bb}_{res}_{m}m']:.0f}s")
    del model

fd = e.features_df
print(f"features_df rows {len(fd)}, unique segments {fd.index.nunique()}, segments {len(ids)}")
feats = (fd.sort_values("area", ascending=False)
      .pipe(lambda t: t[~t.index.duplicated()]).reindex(ids).copy())
feats["relief_mean"], feats["relief_std"], feats["relief_max"] = rel_mean, rel_std, rel_max
with open(os.path.join(OUT, "forest.pkl"), "wb") as fh:
    pickle.dump(dict(ids=ids, cents=cents, seg=seg, feats=feats, embs=embs, px=px,
                     adjacency=getattr(e, "adjacency_graph", None), relief=relief.astype(np.float32),
                     ortho=e.ortho_data, transform=e.ortho_meta["transform"],
                     timings=dict(seg=t_seg, feat=t_feat, emb=t_emb)), fh)
print("saved")
