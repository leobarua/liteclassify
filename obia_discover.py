"""
Discover Classes: deep appearance features per segment, grouping, and the edits a user makes
to the groups (merge, split, ignore, name). No Qt in here -- obia_app.py owns the UI.

Deep features: a frozen DINOv2 ViT-S/14 backbone embeds a square patch of the orthomosaic
around each segment's centroid, at two sizes (5 m: the object itself; 10 m: its
surroundings). Grouping clusters segments on those features and/or the 23 classifier
features, each block scaled to equal weight, with k-means. Groups are deliberately more
numerous than the final classes: the user merges, splits, ignores and names them.
"""
import os
import sys
import time
import warnings

import numpy as np
from scipy import ndimage

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

PATCH_SIZES_M = (5.0, 10.0)
EDGE_VALID_MIN = 0.9          # a segment whose 10 m window is <90 % valid image counts as "edge"
WEIGHTS_NAME = "dinov2_vits14_pretrain.pth"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
GROUP_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ff7f00", "#1f78b4", "#b2df8a", "#fb9a99",
    "#cab2d6", "#6a3d9a", "#b15928", "#a6cee3", "#33a02c", "#fdbf6f",
]


def _base_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _find_weights():
    """ Bundled copy first (works offline), then the torch hub cache, else None. """
    candidates = [os.path.join(_base_dir(), "models", WEIGHTS_NAME)]
    if torch is not None:
        candidates.append(os.path.join(torch.hub.get_dir(), "checkpoints", WEIGHTS_NAME))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_backbone(device):
    """ DINOv2 ViT-S/14 from the vendored code in third_party/ plus the pretrained weights. """
    if torch is None:
        raise RuntimeError("PyTorch is not installed; deep features are unavailable.")
    tp = os.path.join(_base_dir(), "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dinov2.models.vision_transformer import vit_small
    model = vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
    path = _find_weights()
    if path is None:
        print(f" -> DINOv2 weights not found locally; downloading once ({WEIGHTS_URL})...")
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", progress=False)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def seg_fingerprint(segmented_image):
    """ Short hash of a segmentation, so cached features are never applied to a different one. """
    import hashlib
    h = hashlib.md5(np.ascontiguousarray(segmented_image[::5, ::5]).tobytes())
    h.update(str(segmented_image.shape).encode())
    return h.hexdigest()


def segment_centroids(segmented_image):
    ids = np.unique(segmented_image)
    ids = ids[ids > 0]
    cents = np.array(ndimage.center_of_mass(np.ones_like(segmented_image, dtype=np.uint8), segmented_image, ids))
    return ids, cents


def edge_flags(cents, valid_mask, pixel_size):
    """ True for segments whose largest patch window is less than 90 % valid image (image
    border or nodata): their patches are mostly padding, and they group by that. """
    big = max(1, int(round(max(PATCH_SIZES_M) / pixel_size)))
    vm = ndimage.uniform_filter(valid_mask.astype(np.float32), size=big, mode="constant", cval=0.0)
    rc = np.clip(np.round(cents).astype(int), 0, [valid_mask.shape[0] - 1, valid_mask.shape[1] - 1])
    return vm[rc[:, 0], rc[:, 1]] < EDGE_VALID_MIN


def compute_deep_features(rgb_chw, segmented_image, valid_mask, pixel_size, progress=None, cancelled=None):
    """
    cancelled: optional callable checked between batches; when it returns True the run
    stops and None is returned.
    Returns dict(ids, feats {patch_m: (n, 384) float16}, edge (n,) bool, seconds, device).
    rgb_chw: uint8 (3, H, W) orthomosaic on the segmentation grid.
    """
    t0 = time.time()
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    ids, cents = segment_centroids(segmented_image)
    model = load_backbone(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch = 128 if device == "cuda" else 32

    rc = np.clip(np.round(cents).astype(int), 0, [segmented_image.shape[0] - 1, segmented_image.shape[1] - 1])
    edge = edge_flags(cents, valid_mask, pixel_size)

    feats = {}
    n_total = len(ids) * len(PATCH_SIZES_M)
    done = 0
    for patch_m in PATCH_SIZES_M:
        half = max(4, int(round(patch_m / pixel_size / 2)))
        padded = np.pad(rgb_chw, ((0, 0), (half, half), (half, half)), mode="reflect")
        out = np.empty((len(ids), 384), dtype=np.float16)
        with torch.no_grad():
            for i in range(0, len(ids), batch):
                if cancelled is not None and cancelled():
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    return None
                crops = [padded[:, r:r + 2 * half, c:c + 2 * half] for r, c in rc[i:i + batch]]
                x = torch.from_numpy(np.stack(crops)).to(device).float() / 255.0
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
                x = (x - mean) / std
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        f = model(x)
                else:
                    f = model(x)
                out[i:i + len(crops)] = f.float().cpu().numpy().astype(np.float16)
                done += len(crops)
                if progress is not None and (i // batch) % 10 == 0:
                    progress(done, n_total)
        feats[patch_m] = out
        del padded
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(ids=ids, feats=feats, edge=edge, seconds=time.time() - t0, device=device,
                fingerprint=seg_fingerprint(segmented_image))


def subset_deep(deep, present_ids):
    """ Deep features restricted to segments that still exist; None if any are missing
    (a segment that was created, not just removed, needs a recompute). """
    present = np.asarray(sorted(int(s) for s in present_ids))
    if not np.isin(present, deep["ids"]).all():
        return None
    keep = np.isin(deep["ids"], present)
    return dict(ids=deep["ids"][keep], feats={k: v[keep] for k, v in deep["feats"].items()},
                edge=deep["edge"][keep], seconds=deep["seconds"], device=deep["device"],
                fingerprint=deep.get("fingerprint"))


def save_deep_features(path, deep):
    np.savez_compressed(path, ids=deep["ids"], edge=deep["edge"], fingerprint=np.array(deep.get("fingerprint") or ""),
                        **{f"f{int(k)}m": v for k, v in deep["feats"].items()})


def load_deep_features(path):
    z = np.load(path)
    feats = {float(k[1:-1]): z[k] for k in z.files if k.startswith("f") and k.endswith("m")}
    fp = str(z["fingerprint"]) if "fingerprint" in z.files else ""
    return dict(ids=z["ids"], feats=feats, edge=z["edge"], seconds=0.0, device="cache", fingerprint=fp)


# --------------------------------------------------------------------------------------
# Feature space and groups
# --------------------------------------------------------------------------------------

def _block(X, n_pc=None):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X = (X - X.mean(0)) / sd
    if n_pc and X.shape[1] > n_pc:
        from sklearn.decomposition import PCA
        X = PCA(n_pc, random_state=0).fit_transform(X)
    tv = X.var(0).sum()
    return X / np.sqrt(tv) if tv > 0 else X


def build_space(ids, features_df, feature_cols, deep=None, use_deep=True, use_features=True):
    """ Per-segment matrix the groups are built in: PCA-32 of each deep patch size and/or the
    standardised classifier features, every block scaled to unit total variance so none
    dominates by dimension count. """
    blocks = []
    if use_deep and deep is not None:
        pos = {s: i for i, s in enumerate(deep["ids"])}
        idx = np.array([pos[s] for s in ids])
        for k in sorted(deep["feats"]):
            blocks.append(_block(deep["feats"][k][idx].astype(np.float32), 32))
    if use_features:
        blocks.append(_block(features_df.reindex(ids)[feature_cols].values))
    if not blocks:
        raise ValueError("Choose deep features, classifier features, or both.")
    return np.hstack(blocks).astype(np.float32)


def group_key(n):
    """ The name of group n: Group_01, Group_02... (split parts add a, b: Group_21a). """
    return f"Group_{n:02d}"


def upgrade_key(key):
    """ Old-style keys (G01, G21a) to Group_01, Group_21a; anything else unchanged. """
    import re
    m = re.fullmatch(r"G(\d+)([a-z]*)", key or "")
    return f"Group_{m.group(1)}{m.group(2)}" if m else key


class GroupModel:
    """
    Groups of segments and the user's edits to them.
      ids     segment ids (n,)       X       feature space (n, d)
      labels  group key per segment, "" = left out (edge)
      groups  key -> {"name": str, "status": "unnamed"|"named"|"ignored", "color": hex}
    """

    def __init__(self, ids, X, settings):
        self.ids = np.asarray(ids)
        self.X = X
        self.settings = dict(settings)
        self.labels = np.array([""] * len(self.ids), dtype=object)
        self.groups = {}
        self.pos = {int(s): i for i, s in enumerate(self.ids)}

    # --- creation ---------------------------------------------------------------
    @classmethod
    def find(cls, ids, X, n_groups, exclude=None, settings=None):
        from sklearn.cluster import KMeans, MiniBatchKMeans
        m = cls(ids, X, settings or {})
        use = np.ones(len(ids), bool) if exclude is None else ~exclude
        n_groups = int(min(n_groups, max(2, use.sum() // 5)))
        km = (KMeans(n_groups, n_init=4, random_state=0) if use.sum() <= 60000
              else MiniBatchKMeans(n_groups, n_init=3, random_state=0, batch_size=4096))
        lab = km.fit_predict(X[use])
        order = np.argsort(-np.bincount(lab, minlength=n_groups))  # G01 = largest
        rank = np.empty(n_groups, int)
        rank[order] = np.arange(n_groups)
        keys = [group_key(r + 1) for r in range(n_groups)]
        m.labels[np.where(use)[0]] = [keys[rank[l]] for l in lab]
        for i, k in enumerate(keys):
            m.groups[k] = {"name": "", "status": "unnamed", "color": GROUP_PALETTE[i % len(GROUP_PALETTE)]}
        return m

    # --- queries -----------------------------------------------------------------
    def keys(self):
        return list(self.groups)

    def members(self, key):
        return np.where(self.labels == key)[0]

    def size(self, key):
        return int((self.labels == key).sum())

    def centre(self, key):
        mem = self.members(key)
        return self.X[mem].mean(0) if len(mem) else None

    def typical(self, key, n, offset=0):
        """ Member indices, most typical (closest to the centre) first. """
        mem = self.members(key)
        if not len(mem):
            return mem
        d = np.linalg.norm(self.X[mem] - self.centre(key), axis=1)
        return mem[np.argsort(d)][offset:offset + n]

    def closest_groups(self, key, n=3):
        c = self.centre(key)
        out = []
        for k in self.groups:
            if k == key or self.groups[k]["status"] == "ignored" or self.size(k) == 0:
                continue
            out.append((float(np.linalg.norm(self.centre(k) - c)), k))
        return [k for _, k in sorted(out)[:n]]

    def merge_suggestions(self, n=6):
        live = [k for k in self.groups if self.groups[k]["status"] != "ignored" and self.size(k) > 0]
        cents = {k: self.centre(k) for k in live}
        pairs = []
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                pairs.append((float(np.linalg.norm(cents[a] - cents[b])), a, b))
        pairs.sort()
        return [(a, b, d) for d, a, b in pairs[:n]]

    def similar(self, index, n=20, exclude_key=None, query=None):
        """ Indices of the n segments nearest to segment `index` (or to vector `query`). """
        q = self.X[index] if query is None else query
        d = np.linalg.norm(self.X - q, axis=1)
        if index is not None:
            d[index] = np.inf
        if exclude_key is not None:
            d[self.labels == exclude_key] = np.inf
        return np.argsort(d)[:n]

    # --- edits -------------------------------------------------------------------
    def merge(self, keys):
        keys = [k for k in keys if k in self.groups]
        if len(keys) < 2:
            return None
        keep = sorted(keys)[0]
        for k in keys:
            if k != keep:
                self.labels[self.labels == k] = keep
                if not self.groups[keep]["name"] and self.groups[k]["name"]:
                    self.groups[keep]["name"] = self.groups[k]["name"]
                    self.groups[keep]["status"] = self.groups[k]["status"]
                del self.groups[k]
        return keep

    def split(self, key, parts, values=None):
        """ Split a group into `parts`: k-means on the group's own feature rows, or on a
        1-D `values` array (e.g. height or brightness per segment) when given. """
        from sklearn.cluster import KMeans
        mem = self.members(key)
        parts = int(min(parts, len(mem)))
        if parts < 2:
            return []
        if values is None:
            Z = self.X[mem]
        else:
            # Clip to the 5-95 % range so a few extreme values can't claim a part of their own.
            v = np.nan_to_num(np.asarray(values, dtype=np.float64)[mem])
            Z = np.clip(v, *np.percentile(v, [5, 95])).reshape(-1, 1)
        lab = KMeans(parts, n_init=4, random_state=0).fit_predict(Z)
        if values is not None:   # order sub-groups by the value, lowest first
            means = [np.mean(Z[lab == p]) for p in range(parts)]
            remap = {p: r for r, p in enumerate(np.argsort(means))}
            lab = np.array([remap[l] for l in lab])
        base = self.groups[key]
        new_keys = [f"{key}{chr(ord('a') + p)}" for p in range(parts)]
        colors = [base["color"]] + [GROUP_PALETTE[(len(self.groups) + p) % len(GROUP_PALETTE)] for p in range(1, parts)]
        insert_at = list(self.groups).index(key)
        items = list(self.groups.items())
        new_items = [(nk, {"name": base["name"], "status": base["status"], "color": colors[p]}) for p, nk in enumerate(new_keys)]
        self.groups = dict(items[:insert_at] + new_items + items[insert_at + 1:])
        for p, nk in enumerate(new_keys):
            self.labels[mem[lab == p]] = nk
        return new_keys

    def move(self, indices, key):
        self.labels[np.asarray(indices)] = key

    def set_name(self, key, name):
        name = (name or "").strip()
        self.groups[key]["name"] = name
        self.groups[key]["status"] = "named" if name else "unnamed"

    def set_ignored(self, key, ignored):
        g = self.groups[key]
        g["status"] = "ignored" if ignored else ("named" if g["name"] else "unnamed")

    def new_group(self):
        """ An empty group with the next free key (for "move tiles to a new group"). """
        n = len(self.groups) + 1
        while group_key(n) in self.groups:
            n += 1
        key = group_key(n)
        self.groups[key] = {"name": "", "status": "unnamed", "color": GROUP_PALETTE[n % len(GROUP_PALETTE)]}
        return key

    def rename_class(self, old, new):
        """ Follow a class rename/merge (new) or removal (new=None -> groups become unnamed). """
        for g in self.groups.values():
            if g["name"] == old:
                g["name"] = new or ""
                if g["status"] != "ignored":
                    g["status"] = "named" if new else "unnamed"

    def keep_ids(self, present_ids):
        """ Drop segments that no longer exist (e.g. absorbed by a manual merge). """
        keep = np.isin(self.ids, np.asarray(list(present_ids)))
        if keep.all():
            return 0
        self.ids, self.X, self.labels = self.ids[keep], self.X[keep], self.labels[keep]
        self.pos = {int(s): i for i, s in enumerate(self.ids)}
        return int((~keep).sum())

    # --- persistence ----------------------------------------------------------------
    def to_json(self):
        return {"settings": self.settings, "groups": self.groups,
                "segments": {str(int(s)): l for s, l in zip(self.ids, self.labels) if l}}

    @classmethod
    def from_json(cls, data, ids, X):
        m = cls(ids, X, data.get("settings", {}))
        # Workspaces from before v1.0 named groups G01, G21a...: shown as Group_01, Group_21a.
        m.groups = {upgrade_key(k): v for k, v in data["groups"].items()}
        seg = data.get("segments", {})
        m.labels = np.array([upgrade_key(seg.get(str(int(s)), "")) for s in ids], dtype=object)
        return m
