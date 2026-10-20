"""Compare two orthomosaic + DSM products of the same flight (e.g. Metashape vs WebODM).

Both products float on the drone's GPS, so absolute offsets say little about the software.
This script measures relative geometry instead:

1. Both orthos are read onto one common grid (the overlap, at --grid metres).
2. A coarse global shift is found by phase correlation at 1 m, then each tile is matched
   with sub-pixel phase correlation, giving one displacement vector per textured tile.
3. A 2D similarity transform (shift, rotation, scale) is fitted with RANSAC. What is left
   after it (the residuals) is the real geometric difference: random residuals of a pixel
   or two mean the products agree; residuals that grow toward the edges or form a bowl
   mean one of them is distorted.
4. DSMs are sampled at the matched tile centres; after removing the mean vertical offset,
   a plane (tilt) and a radial quadratic term (doming) are fitted to the differences.

Optionally, --points takes manually digitized markers (CSV: name,xa,ya,xb,yb in the
products' CRS) and reports the same similarity fit on them.

    python experiments/webodm/compare.py --ortho-a met_ortho.tif --ortho-b odm_ortho.tif \
        --dsm-a met_dsm.tif --dsm-b odm_dsm.tif --out experiments/webodm/out

Writes a summary (summary.txt), the per-tile vectors (tiles.csv) and, if matplotlib is
installed, two figures (residuals.png, dsm_diff.png) to --out. Not used by the app.
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform


def read_gray(path, transform, shape, crs):
    """ Mean of the RGB bands on the target grid; NaN where the source has no data. """
    with rasterio.open(path) as src:
        nb = min(3, src.count)
        out = np.full((nb,) + shape, np.nan, dtype=np.float32)
        for b in range(nb):
            reproject(rasterio.band(src, b + 1), out[b], dst_transform=transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.average)
        if src.count == 4:                          # RGBA: alpha 0 is outside the mosaic
            a = np.zeros(shape, dtype=np.float32)
            reproject(rasterio.band(src, 4), a, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.min)
            out[:, a < 255] = np.nan
    g = out.mean(axis=0)
    g[(g <= 0) | (g >= 255)] = np.nan             # black/white fill around the mosaic
    return g


def read_dem(path, transform, shape, crs):
    out = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_nd = src.nodata if src.nodata is not None else -9999
        reproject(rasterio.band(src, 1), out, src_nodata=src_nd, dst_transform=transform,
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    out[out < -1000] = np.nan
    return out


def overlap_grid(path_a, path_b, res):
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if a.crs != b.crs:
            print(f"Note: CRS differ ({a.crs} vs {b.crs}); B is reprojected to A's CRS.")
        from rasterio.warp import transform_bounds
        bb = transform_bounds(b.crs, a.crs, *b.bounds)
        left, bottom = max(a.bounds.left, bb[0]), max(a.bounds.bottom, bb[1])
        right, top = min(a.bounds.right, bb[2]), min(a.bounds.top, bb[3])
        if right <= left or top <= bottom:
            raise SystemExit("The two orthomosaics do not overlap.")
        w, h = int((right - left) / res), int((top - bottom) / res)
        return from_origin(left, top, res, res), (h, w), a.crs


def fill(img):
    v = np.isfinite(img)
    out = np.where(v, img, np.nanmean(img) if v.any() else 0).astype(np.float32)
    return out, v


def similarity_fit(src, dst, thresh):
    """ RANSAC similarity src -> dst; returns model, inlier mask, residual vectors. """
    model, inl = ransac((src, dst), SimilarityTransform, min_samples=3,
                        residual_threshold=thresh, max_trials=2000)
    res = dst - model(src)
    return model, inl, res


def describe(model, res, inl, unit="m"):
    r = np.hypot(res[inl, 0], res[inl, 1])
    lines = [
        f"  shift: {model.translation[0]:+.3f}, {model.translation[1]:+.3f} {unit}",
        f"  rotation: {np.degrees(model.rotation) * 3600:+.1f} arcsec",
        f"  scale: {(model.scale - 1) * 1e6:+.0f} ppm",
        f"  inliers: {inl.sum()} of {len(inl)}",
        f"  residual RMSE: {np.sqrt(np.mean(r ** 2)) * 100:.1f} cm, median {np.median(r) * 100:.1f} cm, "
        f"95th pct {np.percentile(r, 95) * 100:.1f} cm, max {r.max() * 100:.1f} cm",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ortho-a", required=True, help="reference product (e.g. Metashape)")
    ap.add_argument("--ortho-b", required=True, help="product to test (e.g. WebODM)")
    ap.add_argument("--dsm-a")
    ap.add_argument("--dsm-b")
    ap.add_argument("--points", help="optional CSV name,xa,ya,xb,yb of manual markers")
    ap.add_argument("--grid", type=float, default=0.10, help="comparison grid (m), default 0.10")
    ap.add_argument("--tile", type=float, default=30.0, help="tile size (m), default 30")
    ap.add_argument("--out", default="experiments/webodm/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = []

    tr, shape, crs = overlap_grid(args.ortho_a, args.ortho_b, args.grid)
    print(f"Common grid: {shape[1]} x {shape[0]} px at {args.grid * 100:.0f} cm ({crs})")
    ga = read_gray(args.ortho_a, tr, shape, crs)
    gb = read_gray(args.ortho_b, tr, shape, crs)

    # Coarse global shift at ~1 m so tiles start roughly aligned (GPS offsets can be metres).
    k = max(1, int(round(1.0 / args.grid)))
    ca = fill(ga[: shape[0] // k * k, : shape[1] // k * k].reshape(shape[0] // k, k, shape[1] // k, k).mean(axis=(1, 3)))[0]
    cb = fill(gb[: shape[0] // k * k, : shape[1] // k * k].reshape(shape[0] // k, k, shape[1] // k, k).mean(axis=(1, 3)))[0]
    coarse, _, _ = phase_cross_correlation(ca, cb, upsample_factor=10)
    coarse = np.asarray(coarse) * k                      # (row, col) shift of B onto A, grid px
    summary.append(f"Coarse shift B->A: {coarse[1] * args.grid:+.2f} m E, {-coarse[0] * args.grid:+.2f} m N")

    # Per-tile sub-pixel matching.
    t = int(args.tile / args.grid)
    rows = []
    for r0 in range(0, shape[0] - t, t):
        for c0 in range(0, shape[1] - t, t):
            a = ga[r0:r0 + t, c0:c0 + t]
            rb, cb0 = r0 - int(round(coarse[0])), c0 - int(round(coarse[1]))
            if rb < 0 or cb0 < 0 or rb + t > shape[0] or cb0 + t > shape[1]:
                continue
            b = gb[rb:rb + t, cb0:cb0 + t]
            if np.isfinite(a).mean() < 0.95 or np.isfinite(b).mean() < 0.95:
                continue
            if np.nanstd(a) < 8:                          # flat, textureless tile
                continue
            fa, _ = fill(a)
            fb, _ = fill(b)
            sh, err, _ = phase_cross_correlation(fa - fa.mean(), fb - fb.mean(), upsample_factor=20)
            # Tile centre in A, and where the same content sits in B (map coordinates).
            ra, cA = r0 + t / 2, c0 + t / 2
            rB, cB = rb + t / 2 - sh[0], cb0 + t / 2 - sh[1]
            xa, ya = tr * (cA, ra)
            xb, yb = tr * (cB, rB)
            rows.append((xa, ya, xb, yb, err))
    if len(rows) < 6:
        raise SystemExit(f"Only {len(rows)} usable tiles; try a larger --tile or check the inputs.")
    T = np.array(rows)
    A, B = T[:, 0:2], T[:, 2:4]
    centre = A.mean(axis=0)
    model, inl, res = similarity_fit(B - centre, A - centre, thresh=max(0.25, 5 * args.grid))
    summary.append(f"\nOrthomosaic, {len(T)} matched {args.tile:.0f} m tiles; similarity B->A:")
    summary += describe(model, res, inl)

    # Does the residual grow toward the edges? (distortion such as bowling or drift)
    d = np.hypot(*(A - centre).T)
    rr = np.hypot(res[:, 0], res[:, 1])
    slope = np.polyfit(d[inl], rr[inl], 1)[0]
    summary.append(f"  residual vs distance from centre: {slope * 100 * 100:+.2f} cm per 100 m")

    np.savetxt(os.path.join(args.out, "tiles.csv"),
               np.column_stack([T, res, inl]), delimiter=",", fmt="%.4f",
               header="xa,ya,xb,yb,pc_error,res_x,res_y,inlier", comments="")

    # DSM comparison at the matched tile centres.
    dz = None
    if args.dsm_a and args.dsm_b:
        za = read_dem(args.dsm_a, tr, shape, crs)
        zb = read_dem(args.dsm_b, tr, shape, crs)

        def sample(z, xy, win=5):
            out = []
            for x, y in xy:
                c, r = ~tr * (x, y)
                r, c = int(r), int(c)
                w = z[max(0, r - win):r + win + 1, max(0, c - win):c + win + 1]
                out.append(np.nanmedian(w) if np.isfinite(w).any() else np.nan)
            return np.array(out)

        dz = sample(zb, B) - sample(za, A)
        ok = np.isfinite(dz) & inl
        off = np.median(dz[ok])
        e = dz - off
        X = A[ok] - centre
        D = np.column_stack([np.ones(ok.sum()), X[:, 0], X[:, 1], (X ** 2).sum(axis=1)])
        coef, *_ = np.linalg.lstsq(D, e[ok], rcond=None)
        rmax = np.hypot(*X.T).max()
        summary.append(f"\nDSM, {ok.sum()} tile centres (median of 11x11 px, mixed cover):")
        summary.append(f"  vertical offset B-A: {off:+.2f} m (GPS altitude, expected)")
        summary.append(f"  spread after offset: std {np.std(e[ok]):.2f} m, MAD {np.median(np.abs(e[ok])):.2f} m")
        summary.append(f"  tilt: {np.hypot(coef[1], coef[2]) * 100 * 100:.1f} cm per 100 m")
        summary.append(f"  doming (centre to {rmax:.0f} m radius): {coef[3] * rmax ** 2 * 100:+.1f} cm")

    if args.points:
        P = np.genfromtxt(args.points, delimiter=",", names=True, dtype=None, encoding=None)
        pa = np.column_stack([P["xa"], P["ya"]]).astype(float)
        pb = np.column_stack([P["xb"], P["yb"]]).astype(float)
        c = pa.mean(axis=0)
        m2 = SimilarityTransform()
        m2.estimate(pb - c, pa - c)
        r2 = (pa - c) - m2(pb - c)
        summary.append(f"\nManual markers ({len(pa)}), similarity B->A (no RANSAC):")
        summary += describe(m2, r2, np.ones(len(pa), bool))
        for n, (rx, ry) in zip(P["name"], r2):
            summary.append(f"    {n}: {np.hypot(rx, ry) * 100:.1f} cm")

    text = "\n".join(summary)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: figures skipped (summary.txt and tiles.csv are written).")
        return
    step = max(1, int(round(1.0 / args.grid)))
    ext = (tr.c, tr.c + shape[1] * args.grid, tr.f - shape[0] * args.grid, tr.f)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(ga[::step, ::step], cmap="gray", extent=ext)
    s = 1000.0                                            # 1 cm residual drawn as 10 m
    ax.quiver(A[inl, 0], A[inl, 1], res[inl, 0] * s, res[inl, 1] * s, angles="xy",
              scale_units="xy", scale=1, color="#f65244", width=0.003)
    ax.plot(A[~inl, 0], A[~inl, 1], "x", color="#888", ms=4, label="outlier tiles")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(f"Ortho residuals after similarity fit (1 cm drawn as {s / 100:.0f} m)")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(args.out, "residuals.png"), dpi=130, bbox_inches="tight")
    if dz is not None:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(ga[::step, ::step], cmap="gray", extent=ext)
        v = np.nanpercentile(np.abs(e[ok]), 95)
        sc = ax.scatter(A[ok, 0], A[ok, 1], c=e[ok], cmap="RdBu_r", vmin=-v, vmax=v, s=40)
        fig.colorbar(sc, ax=ax, label="DSM B - A after offset (m)", shrink=0.8)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_title("DSM differences at matched tiles")
        fig.savefig(os.path.join(args.out, "dsm_diff.png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
