"""Experiment 3 - grass / brush / trees (+ bare, impervious, building) on the Tindalo workspace.

Variants: A app 23 features; B A + gap-context features; C A + DINOv2 context embeddings;
D A + B + C; E two-pass (A + share of the surroundings predicted as trees).
Evaluation: spatially blocked CV on the user's training segments (no reference points exist),
repeated over random block-to-fold assignments; JM for the brush pairs per variant.
Run from the repo root."""
import os, sys, json, pickle
import numpy as np, pandas as pd, rasterio
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from obia_engine import ObiaEngine
from embed import load_backbone, centroids, embed

WS = "data/ForestBrush_Sample/sample2/Tindalo_ortho_workspace"
OUT = os.path.join(os.path.dirname(__file__), "out")
CACHE = os.path.join(OUT, "tindalo.pkl")
BLOCK_M, N_REPEATS, N_FOLDS = 50.0, 30, 5

s = json.load(open(os.path.join(WS, "session.json")))
if os.path.exists(CACHE):
    d = pickle.load(open(CACHE, "rb"))
else:
    with rasterio.open(os.path.join(WS, "segments.tif")) as src:
        seg = src.read(1).astype(np.int32)
    # Read the ortho straight onto the saved segment raster's grid (the script resample path can
    # round one row differently from the app's).
    e = ObiaEngine()
    with rasterio.open(s["ortho_path"]) as src:
        H0, W0 = seg.shape
        e.ortho_data = src.read((1, 2, 3), out_shape=(3, H0, W0), resampling=rasterio.enums.Resampling.average)
        e.original_ortho_data = e.ortho_data
        tr = src.transform * src.transform.scale(src.width / W0, src.height / H0)
        e.ortho_meta = src.meta.copy(); e.ortho_meta.update(width=W0, height=H0, transform=tr)
        e.effective_pixel_size = abs(tr.a)
        e.valid_mask = seg > 0
    e.load_dsm(s["dsm_path"])
    assert seg.shape == e.ortho_data.shape[1:], (seg.shape, e.ortho_data.shape)
    feats = pd.read_csv(os.path.join(WS, "segment_features.csv")).set_index("raster_val")
    ids, cents = centroids(seg)
    px = e.effective_pixel_size

    # Gap-context features on a 0.5 m grid: surrounding canopy top (max DSM) within a radius,
    # and the mean greenness / height of the surroundings.
    f = 5
    dsm = e.dsm_data.copy(); dsm[dsm < -1000] = np.nan
    h, w = dsm.shape; H, W = h // f * f, w // f * f
    dc = np.nanmean(dsm[:H, :W].reshape(H // f, f, W // f, f), axis=(1, 3))
    dc = np.where(np.isnan(dc), np.nanmin(dc), dc)
    o = e.ortho_data[:, :H, :W].astype(np.float32).reshape(3, H // f, f, W // f, f).mean(axis=(2, 4))
    exg = (2 * o[1] - o[0] - o[2]) / (o.sum(0) + 1e-6)
    lab = seg.copy()
    seg_dsm = np.array(ndimage.mean(np.nan_to_num(e.dsm_data), lab, ids))
    rc = np.clip((cents / f).astype(int), 0, [dc.shape[0] - 1, dc.shape[1] - 1])
    ctx = pd.DataFrame(index=ids)
    for r_m in (10, 20):
        win = int(2 * r_m / (px * f)) | 1
        top = ndimage.maximum_filter(dc, size=win)
        p90 = ndimage.percentile_filter(dc, 90, size=win)
        ctx[f"gap_depth_max_{r_m}m"] = top[rc[:, 0], rc[:, 1]] - seg_dsm
        ctx[f"gap_depth_p90_{r_m}m"] = p90[rc[:, 0], rc[:, 1]] - seg_dsm
        ctx[f"surround_exg_{r_m}m"] = ndimage.uniform_filter(exg, win)[rc[:, 0], rc[:, 1]]
        ctx[f"surround_relief_{r_m}m"] = (ndimage.uniform_filter(dc, win) - ndimage.median_filter(dc, win * 3))[rc[:, 0], rc[:, 1]]

    model = load_backbone("dinov2_s")
    embs = {m: embed(e.ortho_data, ids, cents, int(round(m / px)), model) for m in (10, 20, 30)}
    areas = np.bincount(seg.ravel())[ids]
    d = dict(ids=ids, cents=cents, px=px, feats=feats.loc[ids], ctx=ctx, embs=embs, areas=areas,
             transform=e.ortho_meta["transform"], shape=seg.shape)
    pickle.dump(d, open(CACHE, "wb"))

ids, cents, px = d["ids"], d["cents"], d["px"]
APP = ObiaEngine.BASE_FEATURES + ObiaEngine.DSM_FEATURES
A = d["feats"][APP].fillna(0).values
B = d["ctx"].values
Cs = {m: PCA(16, random_state=0).fit_transform(E) for m, E in d["embs"].items()}

pos = {sid: i for i, sid in enumerate(ids)}
train = [(pos[sid], c) for c, v in s["classes"].items() for sid in v["training_segments"] if sid in pos]
ti = np.array([t[0] for t in train]); ty = np.array([t[1] for t in train])
classes = list(s["classes"])
print("training samples:", pd.Series(ty).value_counts().to_dict())

xy = cents * px
blocks = (xy[:, 0] // BLOCK_M).astype(int) * 1000 + (xy[:, 1] // BLOCK_M).astype(int)
tree = cKDTree(xy)
nbrs20 = tree.query_ball_point(xy, 20.0)


def trees_share(pred):
    """Area share of segments within 20 m (excluding self) predicted as trees."""
    is_t = (pred == "trees").astype(float) * d["areas"]
    out = np.empty(len(ids))
    for i, nb in enumerate(nbrs20):
        nb = [j for j in nb if j != i]
        out[i] = is_t[nb].sum() / max(d["areas"][nb].sum(), 1) if nb else 0
    return out


def rf(seed):
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def run(X, two_pass=False):
    preds = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(rep)
        ub = np.unique(blocks[ti])
        fold_of = dict(zip(ub, rng.permutation(len(ub)) % N_FOLDS))
        folds = np.array([fold_of[b] for b in blocks[ti]])
        pred = np.empty(len(ti), dtype=object)
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            if not te.any():
                continue
            m = rf(rep).fit(X[ti[tr]], ty[tr])
            if two_pass:
                share = trees_share(m.predict(X))
                X2 = np.c_[X, share]
                m = rf(rep).fit(X2[ti[tr]], ty[tr])
                pred[te] = m.predict(X2[ti[te]])
            else:
                pred[te] = m.predict(X[ti[te]])
        preds.append(pred)
    return preds


def summarise(name, preds):
    oa = [np.mean(p == ty) for p in preds]
    f1 = np.array([f1_score(ty, p.astype(str), labels=classes, average=None, zero_division=0) for p in preds])
    row = dict(variant=name, OA=np.mean(oa), OA_sd=np.std(oa), macroF1=f1.mean())
    row.update({f"F1_{c}": f1[:, i].mean() for i, c in enumerate(classes)})
    # brush confusion counts, pooled over repeats
    allp = np.concatenate(preds); ally = np.tile(ty, len(preds))
    row["brush_as"] = pd.Series(allp[ally == "brush"]).value_counts(normalize=True).round(2).to_dict()
    row["as_brush_from"] = pd.Series(ally[allp == "brush"]).value_counts(normalize=True).round(2).to_dict()
    return row


variants = {"A app23": (A, False), "B app23+gap": (np.c_[A, B], False)}
for m, C in Cs.items():
    variants[f"C app23+dino{m}m"] = (np.c_[A, C], False)
variants["D app23+gap+dino20m"] = (np.c_[A, B, Cs[20]], False)
variants["E two-pass app23"] = (A, True)
variants["F two-pass app23+gap"] = (np.c_[A, B], True)

rows = [summarise(n, run(X, tp)) for n, (X, tp) in variants.items()]
res = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(res.drop(columns=["brush_as", "as_brush_from"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
for r in rows:
    print(f"{r['variant']:<22} brush predicted as: {r['brush_as']}   predicted brush was: {r['as_brush_from']}")
res.to_csv(os.path.join(OUT, "exp3_tindalo.csv"), index=False)

# JM for the brush pairs per feature set (engine's method on a stand-in features_df)
print("\nJeffries-Matusita (training samples):")
tr_dict = {c: [ids[i] for i, y in zip(ti, ty) if y == c] for c in classes}
for name, X in [("A app23", A), ("B app23+gap", np.c_[A, B]), ("C app23+dino20m", np.c_[A, Cs[20]]),
                ("D app23+gap+dino20m", np.c_[A, B, Cs[20]])]:
    eng = ObiaEngine()
    cols = [f"f{k}" for k in range(X.shape[1])]
    eng.features_df = pd.DataFrame(X, index=ids, columns=cols)
    r = eng.compute_class_separability(tr_dict, features=cols)
    jm = r["jm"]
    print(f"{name:<22}", "  ".join(f"{a}-{b} {jm.loc[a, b]:.3f}" for a, b in
                                    [("brush", "grass"), ("brush", "trees"), ("grass", "trees"), ("grass", "bare")]))
    if name == "B app23+gap":
        names = APP + list(d["ctx"].columns)
        eng.features_df.columns = names
        r = eng.compute_class_separability(tr_dict, features=names)
        for pair in [("grass", "brush"), ("trees", "brush")]:
            key = pair if pair in r["jm_feature"] else pair[::-1]
            print("   best single features", key, r["jm_feature"][key].head(5).round(2).to_dict())
