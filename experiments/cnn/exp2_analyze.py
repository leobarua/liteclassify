"""Experiment 2, step 2 - label-free comparison of feature sets on the forest sample.

For each feature set (app features, CNN embeddings, both):
  * nuisance  - cross-validated R2 for predicting brightness (illumination), elevation and slope
                (terrain) from the set; high = the set mostly encodes nuisances, not canopy.
  * clumping  - k-means (k=6), then the kappa-like excess of same-cluster pairs among segment
                pairs 25-35 m apart (beyond the largest 20 m patch, so overlap can't inflate it).
                > 0 = clusters form spatial clumps at stand scale.
  * maps      - cluster map per set for visual inspection.
Also: agreement (ARI) between 0.1 m and 0.2 m patch clusterings."""
import os, pickle
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from PIL import Image

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
d = pickle.load(open(os.path.join(OUT, "forest.pkl"), "rb"))
ids, cents, seg, F, px = d["ids"], d["cents"], d["seg"], d["feats"], d["px"]
APP = ['r_mean', 'r_std', 'g_mean', 'g_std', 'b_mean', 'b_std', 'area', 'perimeter', 'circularity',
       'convexity', 'contrast', 'homogeneity', 'entropy', 'energy', 'exg_mean', 'vari_mean', 'ngrdi_mean',
       'brightness_mean', 'brightness_rel_neighbors', 'dsm_mean', 'dsm_std', 'slope_mean', 'dsm_rel_neighbors']
REL = ['relief_mean', 'relief_std', 'relief_max']
K = 6
xy = cents * px  # metres

tree = cKDTree(xy)
pairs = tree.query_pairs(35.0, output_type="ndarray")
dist = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
pairs = pairs[dist >= 25.0]
rng = np.random.default_rng(0)
if len(pairs) > 400_000:
    pairs = pairs[rng.choice(len(pairs), 400_000, replace=False)]


def zs(X):
    return StandardScaler().fit_transform(np.nan_to_num(X))


def eta2(lab):
    """Share of each nuisance variable's variance explained by the cluster labels."""
    out = {}
    for t in ("brightness_mean", "dsm_mean", "slope_mean"):
        y = np.nan_to_num(F[t].values)
        means = np.bincount(lab, y, K) / np.maximum(np.bincount(lab, minlength=K), 1)
        out[t] = ((means[lab] - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return out


def clump_kappa(lab):
    same = (lab[pairs[:, 0]] == lab[pairs[:, 1]]).mean()
    p = np.bincount(lab, minlength=K) / len(lab)
    exp = (p ** 2).sum()
    return (same - exp) / (1 - exp)


PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230]], dtype=np.uint8)


def save_map(lab, name):
    lut = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
    lut[ids] = lab + 1
    m = lut[seg[::4, ::4]]
    rgb = np.zeros(m.shape + (3,), np.uint8)
    rgb[m > 0] = PALETTE[m[m > 0] - 1]
    Image.fromarray(rgb).save(os.path.join(OUT, f"map_{name}.png"))


sets = {"app23": zs(F[APP].values), "app23+relief": zs(F[APP + REL].values)}

# Control: plain colour/texture/relief statistics over the same square windows the CNN sees.
import rasterio
from scipy import ndimage
ctx_cache = os.path.join(OUT, "context_stats.npz")
if os.path.exists(ctx_cache):
    ctx = dict(np.load(ctx_cache))
else:
    with rasterio.open("data/CNN_Sample/CNNSample_Ortho_0.1.tif") as s01:
        o = s01.read((1, 2, 3)).astype(np.float32)
        xs, ys = rasterio.transform.xy(d["transform"], cents[:, 0], cents[:, 1])
        r1, c1 = rasterio.transform.rowcol(s01.transform, xs, ys)
    r1, c1 = np.clip(r1, 0, o.shape[1] - 1), np.clip(c1, 0, o.shape[2] - 1)
    bright = o.mean(0)
    exg = (2 * o[1] - o[0] - o[2]) / (o.sum(0) + 1e-6)
    grad = ndimage.gaussian_gradient_magnitude(bright, 1.0)
    layers = dict(r=o[0], g=o[1], b=o[2], bright=bright, exg=exg, grad=grad)
    del o
    rel = np.nan_to_num(d["relief"]); r2_, c2_ = cents[:, 0].astype(int), cents[:, 1].astype(int)
    ctx = {}
    for m in (5, 10, 20):
        w1, w2 = int(m / 0.1), int(m / px)
        cols = []
        for k, L in layers.items():
            mu = ndimage.uniform_filter(L, w1)
            cols.append(mu[r1, c1])
            if k in ("bright", "exg"):
                sd = np.sqrt(np.maximum(ndimage.uniform_filter(L * L, w1) - mu * mu, 0))
                cols.append(sd[r1, c1])
        mu = ndimage.uniform_filter(rel, w2); sd = np.sqrt(np.maximum(ndimage.uniform_filter(rel * rel, w2) - mu * mu, 0))
        cols += [mu[r2_, c2_], sd[r2_, c2_]]
        ctx[f"{m}m"] = np.stack(cols, 1)
        print("context stats", m)
    np.savez(ctx_cache, **ctx)
for m, X in ctx.items():
    sets[f"ctxstats_{m}"] = zs(X)
# no-colour reference: the app set without brightness/colour, to see what's left when colour is removed
for key, E in d["embs"].items():
    Z = PCA(16, random_state=0).fit_transform(E)
    sets[f"cnn:{key}"] = zs(Z)
    if key.startswith("dinov2_s_0.1"):
        sets[f"app23+relief+{key}"] = np.hstack([sets["app23+relief"], zs(Z)])

rows, labels = [], {}
for name, X in sets.items():
    km = KMeans(K, n_init=10, random_state=0).fit(X)
    labels[name] = km.labels_
    nr = eta2(km.labels_)
    seed_ari = adjusted_rand_score(km.labels_, KMeans(K, n_init=10, random_state=1).fit(X).labels_)
    rows.append(dict(set=name, dims=X.shape[1], clump_kappa_25_35m=clump_kappa(km.labels_),
                     eta2_brightness=nr["brightness_mean"], eta2_elevation=nr["dsm_mean"], eta2_slope=nr["slope_mean"],
                     seed_ARI=seed_ari))
    save_map(km.labels_, name.replace(":", "_").replace("+", "_"))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "exp2_labelfree.csv"), index=False)

agree = []
for bb in ("dinov2_s", "resnet50"):
    for m in (5, 10, 20):
        a, b = labels[f"cnn:{bb}_0.1_{m}m"], labels[f"cnn:{bb}_0.2_{m}m"]
        agree.append(dict(backbone=bb, patch_m=m, ARI_0p1_vs_0p2=adjusted_rand_score(a, b)))
for m in (5, 10, 20):
    agree.append(dict(backbone="dinov2 vs resnet (0.1)", patch_m=m,
                      ARI_0p1_vs_0p2=adjusted_rand_score(labels[f"cnn:dinov2_s_0.1_{m}m"], labels[f"cnn:resnet50_0.1_{m}m"])))
ag = pd.DataFrame(agree)
ag.to_csv(os.path.join(OUT, "exp2_agreement.csv"), index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(ag.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
pickle.dump(labels, open(os.path.join(OUT, "exp2_labels.pkl"), "wb"))
