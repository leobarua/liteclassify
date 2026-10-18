"""Experiment 4 - Tindalo: train each feature set on all of the user's training segments, score on
the user's 103 expert-visual reference points. Uses the cache from exp3_tindalo.py.
Run from the repo root."""
import os, sys, json, pickle
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.transform import rowcol
import rasterio
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.getcwd())
from obia_engine import ObiaEngine

WS = "data/ForestBrush_Sample/sample2/Tindalo_ortho_workspace"
OUT = os.path.join(os.path.dirname(__file__), "out")
d = pickle.load(open(os.path.join(OUT, "tindalo.pkl"), "rb"))
s = json.load(open(os.path.join(WS, "session.json")))
ids = d["ids"]; pos = {sid: i for i, sid in enumerate(ids)}
classes = list(s["classes"])

with rasterio.open(os.path.join(WS, "segments.tif")) as src:
    seg = src.read(1); tr = src.transform
pts = gpd.read_file(os.path.join(WS, "accuracy_points.gpkg"))
pts = pts[pts["reference_class"].notna()]
r, c = rowcol(tr, pts.geometry.x.values, pts.geometry.y.values)
pt_seg = seg[np.clip(r, 0, seg.shape[0] - 1), np.clip(c, 0, seg.shape[1] - 1)]
ref = np.asarray(pts["reference_class"].astype(str).values, dtype=object)
train_ids = {sid: cl for cl, v in s["classes"].items() for sid in v["training_segments"]}
on_train = np.array([sid in train_ids for sid in pt_seg])
print(f"{len(ref)} points; {on_train.sum()} fall on a training segment"
      + (f" (label agrees: {sum(train_ids[sid] == rc for sid, rc in zip(pt_seg[on_train], ref[on_train]))})" if on_train.any() else ""))
print("points per class:", pd.Series(ref).value_counts().to_dict())
xy = np.c_[pts.geometry.x, pts.geometry.y]
nn = np.array([min(np.hypot(*(xy[i] - xy[j])) for j in range(len(xy)) if j != i) for i in range(len(xy))])
print(f"nearest-neighbour distance between points: median {np.median(nn):.1f} m, min {nn.min():.1f} m")

APP = ObiaEngine.BASE_FEATURES + ObiaEngine.DSM_FEATURES
A = d["feats"][APP].fillna(0).values
B = d["ctx"].values
C = {m: PCA(16, random_state=0).fit_transform(E) for m, E in d["embs"].items()}
ti = np.array([pos[sid] for sid in train_ids]); ty = np.array(list(train_ids.values()), dtype=object)
te = np.array([pos[sid] for sid in pt_seg]); keep = ~on_train

variants = {"A app23": A, "B app23+gap": np.c_[A, B], "C app23+dino10m": np.c_[A, C[10]],
            "C app23+dino20m": np.c_[A, C[20]], "D app23+gap+dino10m": np.c_[A, B, C[10]]}
SEEDS = 10
rows, preds = [], {}
for name, X in variants.items():
    P = []
    for sd in range(SEEDS):
        m = RandomForestClassifier(n_estimators=100, random_state=sd, n_jobs=-1).fit(X[ti], ty)
        P.append(m.predict(X[te]))
    P = np.array(P); preds[name] = P
    oa = (P == ref).mean(1); oa_k = (P[:, keep] == ref[keep]).mean(1)
    row = dict(variant=name, OA=oa.mean(), OA_sd=oa.std(), OA_excl_train_pts=oa_k.mean())
    for cl in classes:
        pa = (P[:, ref == cl] == cl).mean()                       # producer's
        ua = np.mean([(ref[p == cl] == cl).mean() if (p == cl).any() else np.nan for p in P])  # user's
        row[f"PA_{cl}"], row[f"UA_{cl}"] = pa, ua
    rows.append(row)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res[["variant", "OA", "OA_sd", "OA_excl_train_pts"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nProducer's / user's accuracy (mean over seeds):")
for _, rr in res.iterrows():
    print(f"{rr['variant']:<22}", "  ".join(f"{cl} {rr[f'PA_{cl}']:.2f}/{rr[f'UA_{cl}']:.2f}" for cl in classes))
res.to_csv(os.path.join(OUT, "exp4_tindalo_refpoints.csv"), index=False)

# Paired comparison vs A on the same points (seed 0 majority over seeds)
def vote(P):
    return np.array([pd.Series(col).mode()[0] for col in P.T])
va = vote(preds["A app23"])
for name in variants:
    if name == "A app23":
        continue
    vb = vote(preds[name])
    print(f"{name:<22} vs A: fixed {np.sum((va != ref) & (vb == ref))}, broke {np.sum((va == ref) & (vb != ref))}")

print("\nConfusion (rows = map, cols = reference), seed-majority vote:")
for name in ("A app23", "C app23+dino10m", "D app23+gap+dino10m"):
    print(name)
    print(pd.crosstab(pd.Series(vote(preds[name]), name="map"), pd.Series(ref, name="ref")).reindex(index=classes, columns=classes, fill_value=0))
