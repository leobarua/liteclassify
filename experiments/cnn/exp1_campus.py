"""Experiment 1 - plumbing test on the campus workspace.
RF on the 23 app features vs 23 + PCA-reduced CNN embeddings; same 340 training objects,
scored on the same 100 expert-visual reference points (app baseline OA 87 %).
Run from the repo root."""
import os, sys, json
import numpy as np, pandas as pd
from rasterio.transform import rowcol
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "docs/manual/tools")
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_workspace
from classify import FEATURES
from embed import load_backbone, centroids, embed

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

e = load_workspace()
seg = e.segmented_image
px = e.effective_pixel_size
classes = list(e.session["classes"])
train = [(int(s), c) for c in classes for s in e.session["classes"][c]["training_segments"]]
pts = e.accuracy_points_gdf
pts = pts[pts["reference_class"].notna()]
rows, cols = rowcol(e.ortho_meta["transform"], pts.geometry.x.values, pts.geometry.y.values)
pt_seg = seg[np.clip(rows, 0, seg.shape[0] - 1), np.clip(cols, 0, seg.shape[1] - 1)]
ref = pts["reference_class"].values
print(f"{len(train)} training objects, {len(ref)} reference points, px={px:.3f} m")

ids, cents = centroids(seg)
feat = e.features_df.loc[ids, FEATURES].fillna(0).values
pos = {s: i for i, s in enumerate(ids)}
tr_idx = np.array([pos[s] for s, _ in train]); tr_y = np.array([c for _, c in train])
te_idx = np.array([pos[s] for s in pt_seg])

cache = os.path.join(OUT, "campus_emb.npz")
embs = dict(np.load(cache)) if os.path.exists(cache) else {}
for bb in ("resnet50", "dinov2_s"):
    model = None
    for m in (5, 10, 20):
        key = f"{bb}_{m}m"
        if key not in embs:
            model = model or load_backbone(bb)
            embs[key] = embed(e.ortho_data, ids, cents, int(round(m / px)), model)
            print("embedded", key, embs[key].shape)
    del model
np.savez(cache, **embs)


def score(X, seeds=10):
    oas = []
    for sd in range(seeds):
        clf = RandomForestClassifier(n_estimators=100, random_state=sd, n_jobs=-1)
        clf.fit(X[tr_idx], tr_y)
        oas.append((clf.predict(X[te_idx]) == ref).mean())
    return np.mean(oas), np.std(oas)


res = []
m, s = score(feat); res.append(("23 features (baseline)", "-", m, s))
for key, E in embs.items():
    for k in (8, 16):
        Z = PCA(k, random_state=0).fit_transform(E)
        m, s = score(np.hstack([feat, Z])); res.append((f"23 + {key} PCA{k}", key, m, s))
    Z = PCA(16, random_state=0).fit_transform(E)
    m, s = score(Z); res.append((f"{key} PCA16 only", key, m, s))
df = pd.DataFrame(res, columns=["variant", "emb", "OA_mean", "OA_sd"])
df.to_csv(os.path.join(OUT, "exp1_campus.csv"), index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
