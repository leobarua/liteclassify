"""Experiment 5 - cluster-then-label prototype on Tindalo.

Over-clusters all segments (k-means, K clusters) in several feature spaces and scores how well the
clusters line up with the user's labelled segments (training segments + reference points):
  purity       share of labelled segments whose cluster's majority class equals their own class
  transfer     leave-one-out: name each labelled segment by the majority of the *other* labelled
               segments in its cluster ("if you named each cluster, how many would be right")
  nmi          normalised mutual information between clusters and classes
Splitting a class over several clusters is fine (that is the finer detail we want); mixing classes
inside one cluster is what hurts. Then draws chip galleries + a map for the chosen space, and
suggests merges from a Ward hierarchy of the cluster centres. Run from the repo root."""
import os, sys, json, pickle
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.transform import rowcol
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from obia_engine import ObiaEngine
from embed import load_backbone, embed

WS = "data/ForestBrush_Sample/sample2/Tindalo_ortho_workspace"
OUT = os.path.join(os.path.dirname(__file__), "out")
K = 30
s = json.load(open(os.path.join(WS, "session.json")))
d = pickle.load(open(os.path.join(OUT, "tindalo.pkl"), "rb"))
ids, cents, px = d["ids"], d["cents"], d["px"]

with rasterio.open(os.path.join(WS, "segments.tif")) as src:
    seg = src.read(1).astype(np.int32); seg_tr = src.transform
with rasterio.open(s["ortho_path"]) as src:
    ortho = src.read((1, 2, 3), out_shape=(3,) + seg.shape, resampling=rasterio.enums.Resampling.average)

if 5 not in d["embs"]:
    model = load_backbone("dinov2_s")
    d["embs"][5] = embed(ortho, ids, cents, int(round(5 / px)), model)
    pickle.dump(d, open(os.path.join(OUT, "tindalo.pkl"), "wb"))

# Labelled segments: training segments + reference points (point label wins if both)
pos = {sid: i for i, sid in enumerate(ids)}
lab = {}
for c, v in s["classes"].items():
    for sid in v["training_segments"]:
        lab[pos[sid]] = c
pts = gpd.read_file(os.path.join(WS, "accuracy_points.gpkg"))
pts = pts[pts["reference_class"].notna()]
r, cc = rowcol(seg_tr, pts.geometry.x.values, pts.geometry.y.values)
for sid, c in zip(seg[r, cc], pts["reference_class"].astype(str)):
    if sid > 0:
        lab[pos[sid]] = c
li = np.array(list(lab)); ly = np.array(list(lab.values()))
print(f"{len(li)} labelled segments:", pd.Series(ly).value_counts().to_dict())


def block(X, n_pc=None):
    X = StandardScaler().fit_transform(np.nan_to_num(X))
    if n_pc:
        X = PCA(n_pc, random_state=0).fit_transform(X)
    return X / np.sqrt((X.var(0)).sum())          # each block gets unit total variance


APP = ObiaEngine.BASE_FEATURES + ObiaEngine.DSM_FEATURES
HEIGHT = d["feats"][["dsm_std", "dsm_rel_neighbors"]].values
GAP = d["ctx"][["gap_depth_p90_10m", "gap_depth_p90_20m"]].values
b_app = block(d["feats"][APP].values)
b_d5, b_d10 = block(d["embs"][5], 32), block(d["embs"][10], 32)
b_h = block(np.c_[HEIGHT, GAP])
spaces = {
    "app23": b_app,
    "dino5": b_d5,
    "dino10": b_d10,
    "dino5+dino10": np.c_[b_d5, b_d10],
    "dino5+dino10+height": np.c_[b_d5, b_d10, 0.7 * b_h],
    "dino5+dino10+app23": np.c_[b_d5, b_d10, b_app],
}


def score(cl):
    sub = cl[li]
    maj = {}
    for k in np.unique(sub):
        maj[k] = pd.Series(ly[sub == k]).value_counts().idxmax()
    purity = np.mean([maj[k] == y for k, y in zip(sub, ly)])
    ok = []
    for i, (k, y) in enumerate(zip(sub, ly)):
        others = ly[(sub == k) & (np.arange(len(sub)) != i)]
        ok.append(len(others) > 0 and pd.Series(others).value_counts().idxmax() == y)
    return purity, np.mean(ok), normalized_mutual_info_score(ly, sub)


rows, labels = [], {}
for name, X in spaces.items():
    pur, tra, nmi = [], [], []
    for seed in range(5):
        cl = KMeans(K, n_init=4, random_state=seed).fit_predict(X)
        p_, t_, n_ = score(cl); pur.append(p_); tra.append(t_); nmi.append(n_)
        if seed == 0:
            labels[name] = cl
    rows.append(dict(space=name, purity=np.mean(pur), transfer=np.mean(tra), transfer_sd=np.std(tra), nmi=np.mean(nmi)))
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
res.to_csv(os.path.join(OUT, "exp5_clusters.csv"), index=False)
pickle.dump(labels, open(os.path.join(OUT, "exp5_labels.pkl"), "wb"))

# --- Galleries + map + merge suggestions for the chosen space ---
CHOSEN = sys.argv[1] if len(sys.argv) > 1 else res.sort_values("transfer").iloc[-1]["space"]
cl = labels[CHOSEN]
X = spaces[CHOSEN]
print("\ngallery for:", CHOSEN)
centres = np.array([X[cl == k].mean(0) for k in range(K)])
order = np.argsort(-np.bincount(cl, minlength=K))           # largest clusters first

PAL = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in
                ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
                 "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
                 "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000", "#ff7f00", "#1f78b4",
                 "#b2df8a", "#fb9a99", "#cab2d6", "#6a3d9a", "#b15928", "#a6cee3"]], np.uint8)
T, NCH, W = 120, 8, 12.0                                      # chip px, chips per row, window m
half = int(W / px / 2)
padded = np.pad(ortho, ((0, 0), (half, half), (half, half)))
segp = np.pad(seg, ((half, half), (half, half)))
rng = np.random.default_rng(0)
info = []
for page in range(0, K, 10):
    rows_k = order[page:page + 10]
    img = Image.new("RGB", (190 + T * NCH, T * len(rows_k)), (248, 249, 246))
    dr = ImageDraw.Draw(img)
    for rr, k in enumerate(rows_k):
        members = np.where(cl == k)[0]
        dist = np.linalg.norm(X[members] - centres[k], axis=1)
        pick = list(members[np.argsort(dist)[:NCH // 2]])   # most typical...
        rest = np.setdiff1d(members, pick)
        pick += list(rng.choice(rest, min(NCH - len(pick), len(rest)), replace=False))  # ...and random
        for j, i in enumerate(pick):
            r0, c0 = int(cents[i, 0]), int(cents[i, 1])
            chip = padded[:, r0:r0 + 2 * half, c0:c0 + 2 * half]
            m = segp[r0:r0 + 2 * half, c0:c0 + 2 * half] == ids[i]
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            a = np.moveaxis(chip, 0, -1).copy(); a[edge] = (255, 0, 255)
            img.paste(Image.fromarray(a).resize((T, T)), (190 + j * T, rr * T))
        comp = pd.Series(ly[cl[li] == k]).value_counts()
        txt = "\n".join(f"{c} {n}" for c, n in comp.head(4).items()) or "(no labels)"
        dr.rectangle([4, rr * T + 6, 22, rr * T + 24], fill=tuple(int(v) for v in PAL[k]))
        dr.text((28, rr * T + 6), f"C{k:02d}  {len(members) / len(cl):.1%}", fill=(0, 0, 0))
        dr.text((28, rr * T + 26), txt, fill=(90, 90, 90))
        info.append(dict(cluster=k, share=len(members) / len(cl), labels=comp.to_dict()))
    img.save(os.path.join(OUT, f"exp5_gallery_{page // 10 + 1}.png"))

lut = np.zeros(int(seg.max()) + 1, np.int16) - 1
lut[ids] = cl
m = lut[seg[::5, ::5]]
base = np.moveaxis(ortho[:, ::5, ::5], 0, -1).astype(np.float32)
col = np.where(m[..., None] >= 0, PAL[np.maximum(m, 0)], 0).astype(np.float32)
Image.fromarray((0.35 * base + 0.65 * col).astype(np.uint8)).save(os.path.join(OUT, "exp5_map.png"))

Z = linkage(centres, "ward")
print("\nmerge suggestions (closest cluster pairs first):")
groups = {k: [k] for k in range(K)}
for t, (a, b, dist, _) in enumerate(Z[:12]):
    ga, gb = groups.pop(int(a)), groups.pop(int(b))
    groups[K + t] = ga + gb                    # scipy numbers the node made at step t as K + t
    print(f"  {sorted(ga)} + {sorted(gb)}  (distance {dist:.3f})")
pd.DataFrame(info).to_csv(os.path.join(OUT, "exp5_cluster_info.csv"), index=False)
