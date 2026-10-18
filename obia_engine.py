import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes, sieve
from shapely.geometry import Point
from skimage.segmentation import slic
from skimage.segmentation._slic import _enforce_label_connectivity_cython, _slic_cython
from skimage.segmentation.slic_superpixels import _get_mask_centroids
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def to_uint8(data):
    """ Imagery as 8-bit, the scale every step after loading assumes (texture levels,
    DINOv2 input, chips, display, colour correction). 16-bit and other integer rasters are
    scaled linearly from their type's full range (uint16: /257, so 65535 -> 255), which keeps
    the colour balance. Floats in 0..1 are scaled to 0..255; other floats are clipped. """
    if data.dtype == np.uint8:
        return data
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if data.dtype == np.uint16:
            return (data // 257).astype(np.uint8)
        scaled = (data.astype(np.float32) - info.min) * (255.0 / (info.max - info.min))
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    finite = data[np.isfinite(data)]
    top = float(finite.max()) if finite.size else 1.0
    scaled = data * 255.0 if top <= 1.0 else data
    return np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)


def cuda_available():
    """ Single source of truth for "is a CUDA GPU usable right now" -- used both by
    segment_image's own auto-detection and by obia_app.py's GPU/CPU toggle (to decide
    whether the GPU option should even be selectable in the UI). """
    return HAS_TORCH and torch is not None and torch.cuda.is_available()


# --- Memory budgets for large images ---
# Peak bytes per pixel of one segmentation run, measured on Tindalo tiles (2-9 Mpx):
# GPU 205 B/px of VRAM; CPU 88 B/px of RAM (110 leaves room for the rest of the app).
CPU_SLIC_BYTES_PER_PX = 110
GPU_SLIC_VRAM_BYTES_PER_PX = 210
# What stays in RAM per working-grid pixel once an image is loaded and segmented: ortho (3),
# valid mask (1), labels (4), adjacency temporaries (~6), classified maps (2), misc (3);
# a DSM adds its two float32 grid copies (8).
WORKING_BYTES_PER_PX = 19
WORKING_DSM_BYTES_PER_PX = 8
# Above this many pixels an orthomosaic is read at reduced resolution on load.
LOAD_MAX_PX = 400e6


def free_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def free_vram_bytes():
    if not cuda_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def working_bytes(height, width, with_dsm=False):
    """ Rough RAM the app holds for an image of this size on the working grid. """
    per_px = WORKING_BYTES_PER_PX + (WORKING_DSM_BYTES_PER_PX if with_dsm else 0)
    return int(height) * int(width) * per_px


def torch_smooth(x, sigma):
    """ scipy.ndimage.gaussian_filter(img, [sigma, sigma, 0]) (mode 'reflect', truncate
    4.0) for an (H, W, 3) float32 CUDA tensor -- same kernel and edge handling, so it
    matches the CPU result to float32 rounding. """
    if sigma <= 0:
        return x
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 / (sigma * sigma) * np.arange(-r, r + 1) ** 2)
    k = torch.tensor(k / k.sum(), dtype=torch.float32, device=x.device)
    t = x.permute(2, 0, 1).unsqueeze(1)                    # (3, 1, H, W)
    t = torch.cat([t[:, :, :r].flip(2), t, t[:, :, -r:].flip(2)], dim=2)
    t = torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1))
    t = torch.cat([t[:, :, :, :r].flip(3), t, t[:, :, :, -r:].flip(3)], dim=3)
    t = torch.nn.functional.conv2d(t, k.view(1, 1, 1, -1))
    return t.squeeze(1).permute(1, 2, 0).contiguous()


def torch_rgb2lab(x):
    """ skimage.color.rgb2lab (D65, 2 deg) for an (H, W, 3) float32 CUDA tensor in [0, 1],
    using skimage's own constants. """
    from skimage.color.colorconv import xyz_from_rgb, xyz_tristimulus_values
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).pow(2.4), x / 12.92)
    xyz = lin @ torch.tensor(xyz_from_rgb.T, dtype=torch.float32, device=x.device)
    del lin
    xyz /= torch.tensor(xyz_tristimulus_values(illuminant="D65", observer="2"),
                        dtype=torch.float32, device=x.device)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=0).pow(1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    del xyz
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-1)


def slic_cpu_scaled(image, n_segments, compactness, mask, imin, imax, max_num_iter=10):
    """ skimage.segmentation.slic (0.26) for an (H, W, 3) float image with a mask, Lab
    conversion and connectivity enforcement -- except that the [0, 1] rescale uses the
    given imin/imax instead of this image's own min/max. Tiles of one scene must share one
    scaling, or segment character changes at the seams. With imin/imax = the image's own
    masked min/max it returns exactly what slic() does (checked on the campus sample). """
    dtype = np.float32
    image = image.astype(dtype, copy=True)
    image -= dtype(imin)
    if imax != imin:
        image /= dtype(imax - imin)
    image = rgb2lab(image[np.newaxis, ...]).astype(dtype, copy=False)
    mask_u8 = np.ascontiguousarray(mask, dtype=bool).view('uint8')[np.newaxis, ...]
    centroids, steps = _get_mask_centroids(mask_u8, n_segments, True)
    n_centroids = centroids.shape[0]
    segments = np.ascontiguousarray(
        np.concatenate([centroids, np.zeros((n_centroids, image.shape[3]))], axis=-1), dtype=dtype)
    spacing = np.ones(3, dtype=dtype)
    step = max(steps)
    image = np.ascontiguousarray(image * (1.0 / compactness), dtype=dtype)
    _slic_cython(image, mask_u8, segments, step, max_num_iter, spacing, False,
                 ignore_color=True, start_label=1)
    labels = _slic_cython(image, mask_u8, segments, step, max_num_iter, spacing, False,
                          ignore_color=False, start_label=1)
    del image
    segment_size = mask_u8.sum() / n_centroids
    labels = _enforce_label_connectivity_cython(
        labels, int(0.5 * segment_size), int(3 * segment_size), start_label=1)
    return labels[0]


# --- Per-segment spectral, texture and shape features ---
def extract_single_segment_features(crop_rgb, crop_mask, seg_id, area, perimeter, conv_area, crop_dsm=None, crop_slope=None):
    """
    Computes spectral, geometric, GLCM, optional DSM/slope, and visible-band vegetation indices on a localized crop.
    Optimized to run in parallel across multiple CPU threads with minimum memory overhead.
    """
    mask_bool = crop_mask == seg_id
    if not np.any(mask_bool):
        return None

    num_pixels = np.sum(mask_bool)
    if num_pixels == 0:
        return None

    r_vals = crop_rgb[0][mask_bool].astype(np.float32)
    g_vals = crop_rgb[1][mask_bool].astype(np.float32)
    b_vals = crop_rgb[2][mask_bool].astype(np.float32)

    r_mean, r_std = float(np.mean(r_vals)), float(np.std(r_vals))
    g_mean, g_std = float(np.mean(g_vals)), float(np.std(g_vals))
    b_mean, b_std = float(np.mean(b_vals)), float(np.std(b_vals))

    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    convexity = area / conv_area if conv_area > 0 else 1.0

    # COMPUTE VEGETATION INDICES OFF SPECTRAL SIGNATURES
    rgb_sum = r_vals + g_vals + b_vals
    rgb_sum_safe = np.where(rgb_sum == 0, 1.0, rgb_sum)
    
    r_norm = r_vals / rgb_sum_safe
    g_norm = g_vals / rgb_sum_safe
    b_norm = b_vals / rgb_sum_safe

    # 1. Excess Green Index (ExG)
    exg_vals = 2.0 * g_norm - r_norm - b_norm
    exg_mean = float(np.mean(exg_vals))

    # 2. Visible Atmospherically Resistant Index (VARI)
    vari_denom = g_vals + r_vals - b_vals
    vari_denom_safe = np.where(vari_denom == 0, 1.0, vari_denom)
    vari_vals = (g_vals - r_vals) / vari_denom_safe
    vari_mean = float(np.mean(vari_vals))

    # 3. Normalized Green-Red Difference Index (NGRDI)
    ngrdi_denom = g_vals + r_vals
    ngrdi_denom_safe = np.where(ngrdi_denom == 0, 1.0, ngrdi_denom)
    ngrdi_vals = (g_vals - r_vals) / ngrdi_denom_safe
    ngrdi_mean = float(np.mean(ngrdi_vals))

    # 4. Brightness (Average Intensity) for canopy shadows
    brightness_vals = rgb_sum / 3.0
    brightness_mean = float(np.mean(brightness_vals))

    if num_pixels >= 16:
        gray_crop = (0.299 * crop_rgb[0] + 0.587 * crop_rgb[1] + 0.114 * crop_rgb[2])
        # Level 0 is reserved exclusively for background/non-member crop pixels so real
        # dark pixels (e.g. shadowed canopy) inside the segment aren't confused with it.
        gray_quant = 1 + np.clip((gray_crop / 255.0) * 14, 0, 14).astype(np.uint8)
        gray_quant[~mask_bool] = 0

        glcm = graycomatrix(gray_quant, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)

        # Drop co-occurrence pairs touching the background bin, then renormalize so
        # texture stats reflect only real intra-segment pixel-pair statistics instead
        # of being dominated by the bounding-box background surrounding the segment.
        glcm[0, :, :, :] = 0.0
        glcm[:, 0, :, :] = 0.0
        pair_sums = glcm.sum(axis=(0, 1), keepdims=True)
        glcm = np.divide(glcm, pair_sums, out=np.zeros_like(glcm), where=pair_sums > 0)

        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy = float(np.mean(graycoprops(glcm, 'ASM')))

        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0.0001]
        entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat))) if glcm_flat.size > 0 else 0.0
    else:
        contrast = 0.0
        homogeneity = 1.0
        energy = 1.0
        entropy = 0.0

    dsm_mean = 0.0
    dsm_std = 0.0
    if crop_dsm is not None:
        dsm_vals = crop_dsm[mask_bool]
        dsm_vals = dsm_vals[np.isfinite(dsm_vals)]   # DSM nodata is NaN
        if len(dsm_vals) > 0:
            dsm_mean = float(np.mean(dsm_vals))
            dsm_std = float(np.std(dsm_vals))

    # Mean surface slope (degrees) distinguishes a coherently tilted plane (e.g. a
    # roof) from a flat surface (e.g. grass) even when their dsm_std is similar --
    # dsm_std alone conflates segment size and general bumpiness with true tilt.
    slope_mean = 0.0
    if crop_slope is not None:
        slope_vals = crop_slope[mask_bool]
        slope_vals = slope_vals[np.isfinite(slope_vals)]
        if len(slope_vals) > 0:
            slope_mean = float(np.mean(slope_vals))

    return {
        'raster_val': int(seg_id),
        'r_mean': r_mean, 'r_std': r_std,
        'g_mean': g_mean, 'g_std': g_std,
        'b_mean': b_mean, 'b_std': b_std,
        'area': float(area), 'perimeter': float(perimeter),
        'circularity': float(circularity), 'convexity': float(convexity),
        'contrast': contrast, 'homogeneity': homogeneity,
        'entropy': entropy, 'energy': energy,
        'exg_mean': exg_mean, 'vari_mean': vari_mean,
        'ngrdi_mean': ngrdi_mean, 'brightness_mean': brightness_mean,
        'dsm_mean': dsm_mean, 'dsm_std': dsm_std,
        'slope_mean': slope_mean
    }


# --- Engine: raster I/O, segmentation (CPU and GPU SLIC), features, accuracy ---
class ObiaEngine:
    def __init__(self):
        self.ortho_path = None
        self.original_ortho_data = None
        self.ortho_data = None
        self.ortho_meta = None
        # Native = the source file's own resolution, read once at load time and never
        # overwritten by a resample (resample_ortho always re-reads self.ortho_path
        # fresh, so these stay the reference point no matter how many times the user
        # resamples back and forth). effective_pixel_size is whatever's *currently*
        # loaded -- equal to native_pixel_size until resample_ortho changes it.
        self.native_pixel_size = None
        self.native_width = None
        self.native_height = None
        self.native_crs_is_geographic = False
        self.source_dtype = None
        self.source_bands = None
        self.effective_pixel_size = None
        self.segmented_image = None
        self.segments_gdf = None
        self.features_df = None
        self.valid_mask = None
        self.dsm_path = None
        self.dsm_data = None
        self.slope_data = None
        self.adjacency_graph = None

        # --- Accuracy assessment (Tab 4) ---
        # One GeoDataFrame holds every reference point regardless of source (CSV import
        # or expert-generated). Columns:
        # point_id (stable int, used by the UI for hit-testing/list rows), geometry
        # (Point, ortho's CRS), reference_class (human/CSV-asserted true class, None
        # until labeled), source ('field_csv' or 'expert_visual'), predicted_class
        # (filled in by run_accuracy_assessment). Stored in real-world CRS coordinates,
        # not pixel/segment indices, so unlike training samples these survive a
        # resample or DSM reload untouched -- only regenerating/reimporting changes them.
        self.accuracy_points_gdf = None
        self.next_accuracy_point_id = 1
        # Stratified sampling: {"pixels": {class: mapped pixel count}, "pixel_area": m2,
        # "seed": int} for the map the strata were drawn from (None = no strata).
        self.accuracy_strata = None

    @staticmethod
    def load_pixel_limit():
        """ Largest image (pixels) read as is on load: LOAD_MAX_PX, or less when free RAM
        could not hold the working-grid arrays for it. """
        ram = free_ram_bytes()
        if ram is None:
            return LOAD_MAX_PX
        return max(25e6, min(LOAD_MAX_PX, 0.5 * ram / WORKING_BYTES_PER_PX))

    @staticmethod
    def _read_ortho(src, out_h, out_w):
        """ RGB + valid mask from an open raster at out_h x out_w (average resampling for
        the colours, nearest for the mask; GDAL uses the file's overviews when it has them).
        Returns (data, meta, valid_mask). """
        same = (out_h == src.height and out_w == src.width)
        if same:
            data = src.read((1, 2, 3))
        else:
            data = src.read((1, 2, 3), out_shape=(3, out_h, out_w),
                            resampling=rasterio.enums.Resampling.average)
        data = to_uint8(data)   # 16-bit and float orthos: everything downstream works in 0..255
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        meta = src.meta.copy()
        meta.update(width=out_w, height=out_h, transform=transform)
        try:
            if same:
                valid_mask = src.read_masks(1) > 0
            else:
                valid_mask = src.read_masks(1, out_shape=(out_h, out_w),
                                            resampling=rasterio.enums.Resampling.nearest) > 0
        except Exception:
            rgb_sum = np.sum(data, axis=0)
            valid_mask = (rgb_sum > 0) & (rgb_sum < 765)
        return data, meta, valid_mask

    def load_raster(self, filepath, max_pixels=None, out_shape=None):
        """ Reads the orthomosaic. Above max_pixels (default: load_pixel_limit()) it is read
        straight at a reduced resolution instead of at native -- a 6 Gpx file would need
        18 GB just to hold. out_shape=(h, w) reads exactly that grid instead (Load
        Workspace). self.load_reduction is the linear factor (1.0 = native). """
        if max_pixels is None:
            max_pixels = self.load_pixel_limit()
        with rasterio.open(filepath) as src:
            native_px = src.width * src.height
            factor = max(1.0, float(np.sqrt(native_px / max_pixels)))
            if factor > 1.0:
                # Snap up to one of the file's overview levels when it has them: GDAL then
                # reads that overview instead of averaging every native pixel (a 1.2x
                # reduction of a 0.6 Gpx file took 35 s; its 2x overview reads in seconds).
                levels = [f for f in (src.overviews(1) or []) if f >= factor]
                if levels:
                    factor = float(min(levels))
            out_w = max(1, int(round(src.width / factor)))
            out_h = max(1, int(round(src.height / factor)))
            if out_shape is not None:
                out_h, out_w = int(out_shape[0]), int(out_shape[1])
                factor = src.width / out_w
            data, meta, valid_mask = self._read_ortho(src, out_h, out_w)
            self.ortho_path = filepath
            self.native_pixel_size = abs(src.transform.a)
            self.native_width = src.width
            self.native_height = src.height
            self.native_crs_is_geographic = bool(src.crs and src.crs.is_geographic)
            self.source_dtype = str(src.dtypes[0])      # before to_uint8 (status line)
            self.source_bands = int(src.count)

        self.original_ortho_data = data
        self.ortho_data = data   # shared until a colour correction replaces it
        self.ortho_meta = meta
        self.valid_mask = valid_mask
        self.effective_pixel_size = abs(meta["transform"].a)
        self.load_reduction = factor
        for name in ("dsm_data", "slope_data", "segmented_image", "segments_gdf",
                     "features_df", "adjacency_graph"):
            setattr(self, name, None)
        return self.ortho_data, self.ortho_meta

    def resample_ortho(self, target_pixel_size=None, scale_percent=None, out_shape=None):
        """
        Re-reads the orthomosaic from disk (self.ortho_path) at a coarser or finer
        resolution than whatever is currently loaded. Always resamples fresh from the
        original file -- never from an already-resampled in-memory array -- so
        resampling back and forth (e.g. 50% then back to 100%) never compounds quality
        loss and always means "relative to the source file's native resolution."

        Exactly one of target_pixel_size (in the raster's own CRS linear units, e.g.
        meters/pixel for a typical projected drone-ortho CRS) or scale_percent (e.g.
        50 = half the native linear resolution, i.e. a quarter of the native pixel
        count) must be given -- or out_shape=(height, width) to hit an exact grid, which
        Load Workspace uses so the ortho always matches the saved segments.tif (rounding
        the size from a pixel size can land one row/column off).

        Uses Resampling.average for the color bands -- the standard choice for
        downsampling continuous raster imagery (avoids the aliasing/moire that
        nearest-neighbor, or even bilinear, can introduce when shrinking an image),
        matching what QGIS/GDAL default to for this operation. The valid-data mask is
        resampled separately with nearest-neighbor, since it's categorical.

        Resampling changes the image's pixel dimensions, which invalidates anything
        computed against the *previous* dimensions: any loaded DSM (regridded to the
        old ortho shape at DSM-load time), the segmentation, its feature table, and its
        adjacency graph are all cleared here as part of this call -- segment IDs from a
        previous segmentation wouldn't even correspond to the same physical footprint
        on the new pixel grid, so there's no safe way to carry them forward. The caller
        (obia_app.py) is responsible for clearing any UI-side state that references
        those (training sample segment IDs, cached preview images, the classifier,
        etc.) and for re-loading the DSM file (if one was active) so it gets regridded
        to the new shape.
        """
        if self.ortho_path is None:
            raise ValueError("No orthomosaic loaded to resample.")
        if out_shape is None and (target_pixel_size is None) == (scale_percent is None):
            raise ValueError("Provide exactly one of target_pixel_size or scale_percent.")

        with rasterio.open(self.ortho_path) as src:
            native_pixel_size = abs(src.transform.a)
            if out_shape is not None:
                factor = src.width / out_shape[1]
            elif scale_percent is not None:
                if scale_percent <= 0:
                    raise ValueError("scale_percent must be > 0.")
                factor = 100.0 / scale_percent
            else:
                if target_pixel_size <= 0:
                    raise ValueError("target_pixel_size must be > 0.")
                factor = target_pixel_size / native_pixel_size
            factor = max(factor, 1e-6)

            out_width = max(1, int(round(src.width / factor)))
            out_height = max(1, int(round(src.height / factor)))
            if out_shape is not None:
                out_height, out_width = int(out_shape[0]), int(out_shape[1])

            # Drop the current image first so the old and new arrays never coexist; if the
            # new size can't be read, the previous grid is read back so the app keeps working.
            prev_shape = self.ortho_data.shape[1:] if self.ortho_data is not None else None
            self.original_ortho_data = self.ortho_data = None
            try:
                data, new_meta, valid_mask = self._read_ortho(src, out_height, out_width)
            except Exception:
                if prev_shape is not None:
                    data, self.ortho_meta, self.valid_mask = self._read_ortho(src, *prev_shape)
                    self.original_ortho_data = self.ortho_data = data
                raise

        self.original_ortho_data = data
        self.ortho_data = data   # shared until a colour correction replaces it
        self.ortho_meta = new_meta
        self.valid_mask = valid_mask
        self.effective_pixel_size = abs(new_meta["transform"].a)

        # Downstream artifacts computed against the previous pixel grid are now stale.
        self.dsm_data = None
        self.slope_data = None
        self.segmented_image = None
        self.segments_gdf = None
        self.features_df = None
        self.adjacency_graph = None

        return self.ortho_data, self.ortho_meta

    # --- DSM: attached on load, matched to the working grid only when first needed ---
    #
    # Every per-segment DSM statistic (dsm_mean/dsm_std/slope_mean/dsm_rel_neighbors)
    # crops dsm_data/slope_data with the same pixel indices as segmented_image, so those
    # two arrays must sit on the orthomosaic's *current* grid. Building them is expensive
    # on a large grid (8 bytes per pixel, plus temporaries), and pointless before the
    # working resolution is settled, so load_dsm() only reads the DSM's header; the grid
    # copies are built the first time dsm_data/slope_data is read (features, merges),
    # which in the app happens inside the segmentation worker thread, not on the UI.
    # has_dsm is the cheap "is a DSM attached" test.

    @property
    def has_dsm(self):
        return getattr(self, "dsm_path", None) is not None

    @property
    def dsm_data(self):
        grid = self.__dict__.get("_dsm_grid", self.__dict__.get("dsm_data"))
        if grid is None and self.has_dsm and self.ortho_meta is not None and self.ortho_data is not None:
            self._match_dsm_to_grid()
            grid = self._dsm_grid
        return grid

    @dsm_data.setter
    def dsm_data(self, value):
        self.__dict__["_dsm_grid"] = value
        if value is None:
            self.__dict__["_slope_grid"] = None

    @property
    def slope_data(self):
        grid = self.__dict__.get("_slope_grid", self.__dict__.get("slope_data"))
        if grid is None and self.has_dsm and self.ortho_meta is not None and self.ortho_data is not None:
            self._match_dsm_to_grid()
            grid = self._slope_grid
        return grid

    @slope_data.setter
    def slope_data(self, value):
        self.__dict__["_slope_grid"] = value

    def load_dsm(self, filepath):
        """
        Attaches a DSM. Reads only its header here (instant, no memory); the DSM is matched
        to the orthomosaic's working grid the first time dsm_data/slope_data is used -- see
        _match_dsm_to_grid. Returns a short description of the DSM for the log.
        """
        with rasterio.open(filepath) as src:
            self.dsm_info = {
                "pixel_size": abs(src.transform.a), "width": src.width, "height": src.height,
                "crs": src.crs, "bounds": src.bounds, "nodata": src.nodata,
            }
        self.dsm_path = filepath
        self.dsm_data = None          # any grid copy from a previous DSM/grid is stale
        info = self.dsm_info
        unit = "units" if self.native_crs_is_geographic else "cm"
        size = info["pixel_size"] if self.native_crs_is_geographic else info["pixel_size"] * 100
        return f"{info['width']}x{info['height']} px at {size:.2f} {unit}/px"

    def dsm_grid_bytes(self):
        """ Memory the grid copies of the DSM and its slope will take (float32 each). """
        if self.ortho_data is None:
            return 0
        h, w = self.ortho_data.shape[1:]
        return int(h) * int(w) * 4 * 2

    def _read_dsm_for(self, dst_transform, dst_shape, dst_crs):
        """ The DSM (and its slope, degrees) on a target grid, placed by coordinates
        (rasterio reproject), never finer than the DSM's own pixels at read time: a DSM
        coarser than the target is read whole and interpolated (bilinear) onto it, a finer
        one is first read already averaged down to about the target's pixel size. Slope is
        computed at that read resolution, where the DSM's real detail lives, then placed
        on the grid the same way. Nodata becomes NaN. float32 throughout. """
        from rasterio.warp import reproject
        dst_px = abs(dst_transform.a)
        with rasterio.open(self.dsm_path) as src:
            factor = max(1.0, dst_px / abs(src.transform.a))
            h = max(1, int(round(src.height / factor)))
            w = max(1, int(round(src.width / factor)))
            dem = src.read(1, out_shape=(h, w), masked=True,
                           resampling=rasterio.enums.Resampling.average if factor > 1.001 else rasterio.enums.Resampling.nearest)
            dem = np.ma.filled(dem.astype(np.float32), np.nan)
            src_tr = src.transform * src.transform.scale(src.width / w, src.height / h)
            src_crs = src.crs or dst_crs
        gy, gx = np.gradient(dem, np.float32(abs(src_tr.e)), np.float32(abs(src_tr.a)))
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
        del gx, gy
        out = []
        for arr in (dem, slope):
            dst = np.full(dst_shape, np.nan, dtype=np.float32)
            reproject(arr, dst, src_transform=src_tr, src_crs=src_crs, src_nodata=np.nan,
                      dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
                      resampling=rasterio.enums.Resampling.bilinear)
            out.append(dst)
        return out[0], out[1]

    def _match_dsm_to_grid(self):
        """ Builds dsm_data/slope_data on the current orthomosaic grid (see _read_dsm_for).
        Refuses with a MemoryError when the grid copies would not fit in free RAM, rather
        than letting the machine swap or the app freeze. """
        need = self.dsm_grid_bytes()
        try:
            import psutil
            free = psutil.virtual_memory().available
        except Exception:
            free = None
        if free is not None and need > 0.6 * free:
            h, w = self.ortho_data.shape[1:]
            raise MemoryError(
                f"Matching the DSM to the current {w}x{h} px grid needs about {need / 1e9:.1f} GB, "
                f"but only {free / 1e9:.1f} GB of memory is free. Resample the orthomosaic to a coarser "
                f"resolution first (Tab 1), then continue.")
        print(f" -> Matching the DSM to the working grid ({need / 1e9:.2f} GB)...")
        dsm, slope = self._read_dsm_for(self.ortho_meta["transform"], self.ortho_data.shape[1:], self.ortho_meta["crs"])
        self.__dict__["_dsm_grid"] = dsm
        self.__dict__["_slope_grid"] = slope

    def compute_hillshade_rgba(self, azimuth=315.0, altitude=45.0, hillshade_weight=0.6, max_side=None):
        """
        Builds a colorized-elevation-over-hillshade preview as a plain uint8 RGBA numpy
        array (h, w, 4) -- no Qt here, by design (obia_engine.py stays Qt-free;
        obia_app.py wraps the result in a QImage). Returns None if no DSM is loaded.

        azimuth/altitude: standard analytical-hillshade light source angles, in degrees
        (default matches the common GIS convention: sun from the northwest at 45 deg).
        hillshade_weight: how strongly the hillshade's light/shadow modulates the flat
        elevation-ramp color underneath, 0 (ramp color only, no shading) to 1 (full
        shading contrast). Nodata pixels (per valid_mask) come back fully transparent.

        max_side: when given (the app uses 2048), the preview is built on a small grid
        covering the orthomosaic (longest side <= max_side) straight from the DSM file,
        and the caller stretches it over the image -- a few MB instead of 4 bytes per
        orthomosaic pixel. None = full working grid (the manual's figure scripts).
        """
        if not self.has_dsm or self.ortho_data is None:
            return None

        H, W = self.ortho_data.shape[1:]
        if max_side is not None and max(H, W) > max_side:
            f = max(H, W) / float(max_side)
            h, w = max(1, int(round(H / f))), max(1, int(round(W / f)))
            tr = self.ortho_meta['transform'] * self.ortho_meta['transform'].scale(W / w, H / h)
            dsm, _ = self._read_dsm_for(tr, (h, w), self.ortho_meta['crs'])
            rows = np.minimum((np.arange(h) * (H / h)).astype(int), H - 1)
            cols = np.minimum((np.arange(w) * (W / w)).astype(int), W - 1)
            ortho_valid = self.valid_mask[np.ix_(rows, cols)] if self.valid_mask is not None else None
            pixel_size_x, pixel_size_y = abs(tr.a), abs(tr.e)
        else:
            dsm = self.dsm_data
            h, w = dsm.shape
            ortho_valid = self.valid_mask
            pixel_size_x = abs(self.ortho_meta['transform'].a)
            pixel_size_y = abs(self.ortho_meta['transform'].e)

        f32 = np.float32
        dzdy, dzdx = np.gradient(dsm, f32(pixel_size_y), f32(pixel_size_x))

        slope_rad = np.arctan(np.hypot(dzdx, dzdy))
        aspect_rad = np.arctan2(dzdy, -dzdx)
        del dzdx, dzdy

        # float32 scalars: numpy scalars are not "weak", so float64 ones would promote
        # every array below to float64 (twice the memory).
        azimuth_rad = f32(np.radians(360.0 - azimuth + 90.0))
        altitude_rad = f32(np.radians(altitude))

        hillshade = (
            np.sin(altitude_rad) * np.cos(slope_rad) +
            np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
        )
        hillshade = np.clip(np.nan_to_num(hillshade, nan=f32(1.0)), f32(0.0), f32(1.0))
        del slope_rad, aspect_rad

        valid = ortho_valid if ortho_valid is not None else np.ones_like(dsm, dtype=bool)
        valid = valid & np.isfinite(dsm)
        if np.any(valid):
            lo, hi = np.percentile(dsm[valid], (2, 98))
        else:
            lo, hi = float(np.nanmin(dsm)), float(np.nanmax(dsm))
        if hi <= lo:
            hi = lo + 1.0
        norm_elev = np.clip(np.nan_to_num((dsm - f32(lo)) / f32(hi - lo), nan=f32(0.0)), f32(0.0), f32(1.0))

        # Simple built-in terrain-style ramp (low -> high): dark green, green, tan,
        # brown, white. Avoids pulling in matplotlib just for a 5-stop color lerp.
        ramp = [
            (0.00, (30, 90, 40)),
            (0.35, (120, 160, 60)),
            (0.65, (170, 130, 70)),
            (0.85, (150, 100, 70)),
            (1.00, (245, 245, 245)),
        ]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for (t0, c0), (t1, c1) in zip(ramp, ramp[1:]):
            seg_mask = (norm_elev >= t0) & (norm_elev <= t1)
            if not np.any(seg_mask):
                continue
            span = (t1 - t0) if (t1 - t0) > 0 else 1.0
            local_t = (norm_elev[seg_mask] - t0) / span
            for ch in range(3):
                rgb[..., ch][seg_mask] = c0[ch] + (c1[ch] - c0[ch]) * local_t

        rgb *= (f32(1.0 - hillshade_weight) + f32(hillshade_weight) * hillshade)[..., None]
        shaded = np.clip(rgb, 0, 255).astype(np.uint8)
        del rgb

        rgba = np.dstack([shaded, np.full((h, w), 255, dtype=np.uint8)])
        rgba[~valid, 3] = 0
        return rgba

    def segment_image(self, n_segments, compactness, sigma, force_device="auto"):
        """
        force_device: "auto" (use GPU if a CUDA device is available, matching the
        original behavior), "gpu" (explicit user request -- falls back to CPU with a
        log message if no CUDA device is actually present, same as an auto GPU
        failure would), or "cpu" (explicit user request, skips GPU regardless of
        CUDA availability).
        """
        cuda_is_available = cuda_available()

        if force_device == "cpu":
            use_gpu = False
            if cuda_is_available:
                print(" -> CUDA device detected but CPU was explicitly selected in the Segmentation Engine toggle.")
        elif force_device == "gpu":
            use_gpu = cuda_is_available
            if not cuda_is_available:
                print(" -> GPU explicitly selected but no CUDA device is available. Falling back to CPU SLIC.")
        else:
            use_gpu = cuda_is_available

        plan = self.plan_segmentation(n_segments, use_gpu)
        if plan["tile"] is not None:
            try:
                self.segmented_image = self._segment_tiled(n_segments, compactness, sigma, use_gpu, plan)
                return self.segmented_image, None
            except Exception as e:
                if not use_gpu:
                    raise
                print(f" -> Tiled GPU segmentation failed ({e}). Retrying the tiles on the CPU...")
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                plan = self.plan_segmentation(n_segments, False)
                if plan["tile"] is not None:
                    self.segmented_image = self._segment_tiled(n_segments, compactness, sigma, False, plan)
                    return self.segmented_image, None
                use_gpu = False

        # Straight to float32 (no intermediate uint8 copy); divide in place.
        image_float = np.transpose(self.ortho_data, (1, 2, 0)).astype(np.float32)
        image_float /= np.float32(255.0)

        # Compensate the target segment count for masked-out nodata area up front so
        # both the GPU and CPU paths request the same effective density over valid pixels.
        total_pixels = image_float.shape[0] * image_float.shape[1]
        valid_pixel_count = np.sum(self.valid_mask)
        valid_ratio = valid_pixel_count / total_pixels if total_pixels > 0 else 1.0
        compensated_n_segments = int(n_segments / valid_ratio) if valid_ratio > 0.05 else n_segments
        print(f"Applying Spatial Density Compensation: Ratio {valid_ratio:.2f}. Inflating target count to: {compensated_n_segments}")

        if use_gpu:
            print("\n================== GPU SLIC INITIALIZATION ==================")
            print(" -> Launching parallelized PyTorch spatial-spectral K-Means on CUDA...")
            try:
                # NOTE: the GPU K-means below is a distinct, custom implementation, not a
                # port of skimage's Cython SLIC kernel. Despite a structurally similar
                # distance formula, empirical testing showed the two are calibrated on
                # different numeric scales for "compactness" -- skimage's own preprocessing
                # is not reproducible outside its private Cython kernel, so compactness is
                # passed through unscaled here rather than forcing an unverified 1:1 match.
                self.segmented_image = self._segment_image_gpu_slic(image_float, compensated_n_segments, compactness, sigma)
                print(" -> GPU Segmentation completed successfully.")
                return self.segmented_image, image_float
            except Exception as e_gpu:
                print(f" -> GPU SLIC failed ({e_gpu}). Falling back to traditional CPU SLIC...")
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()   # release what the failed GPU attempt left on the card
                except Exception:
                    pass
        else:
            print("\n================== CPU SLIC INITIALIZATION ==================")
            if cuda_is_available:
                print(" -> Running on multi-core CPU (CPU explicitly selected)...")
            else:
                print(" -> PyTorch CUDA core is unavailable. Running on multi-core CPU...")

        smoothed_img = image_float
        if sigma > 0:
            smoothed_img = gaussian_filter(image_float, sigma=[sigma, sigma, 0])
            del image_float   # only the smoothed copy is needed from here (saves 12 bytes/px at SLIC's peak)
        else:
            del image_float

        # No nodata compensation here: with mask=, skimage already places its n_segments
        # seeds inside the valid area (the compensated count gave 1/valid-ratio times too
        # many segments, e.g. ~1.6x on the forest sample).
        segmented_raw = slic(
            smoothed_img,
            n_segments=n_segments,
            compactness=compactness * 100.0,
            sigma=0,
            start_label=1,
            mask=self.valid_mask
        )
        del smoothed_img
        self.segmented_image = self._absorb_mask_fragments(
            np.where(self.valid_mask, segmented_raw, 0).astype(np.int32))
        return self.segmented_image, None


    # --- Tiled segmentation (images too large for one run) ---
    #
    # Tiles have a core plus an overlap margin shared with their neighbours. Each tile is
    # segmented including its margin; a segment is kept only if its centroid lies in the
    # tile's core and it does not touch the tile's cut edge, so segments centred in the
    # margin are left to the neighbour that sees them whole and seams follow segment
    # boundaries. Colour scaling is computed once for the scene, and each tile's segment
    # count comes from one target segment size, so segment character does not change at
    # the seams. Pixels claimed by two tiles stay with the first claim; pixels nobody
    # claimed join the neighbour they share the longest border with.

    force_tile_size = None   # tests: tile even when one run would fit

    def plan_segmentation(self, n_segments, use_gpu):
        """ Decides between one run and tiles. Returns {"tile": core size in px or None,
        "margin", "size_px" (target pixels per segment), "n_tiles", "budget_px"}. """
        H, W = self.valid_mask.shape
        total = H * W
        valid = int(np.count_nonzero(self.valid_mask))
        size_px = max(4.0, valid / max(1, n_segments))
        margin = int(3 * np.sqrt(size_px)) + 16   # >= 2 x the largest segment's diameter

        ram = free_ram_bytes()
        budget = (0.5 * ram / CPU_SLIC_BYTES_PER_PX) if ram else 150e6
        if use_gpu:
            vram = free_vram_bytes()
            gpu_budget = (0.7 * vram / GPU_SLIC_VRAM_BYTES_PER_PX) if vram else 40e6
            # the GPU path also holds float/Lab copies of the tile in RAM (~40 B/px)
            budget = min(gpu_budget, (0.5 * ram / 40) if ram else gpu_budget)

        tile = None
        if self.force_tile_size:
            tile = int(self.force_tile_size)
        elif total > budget:
            side = int(np.sqrt(budget))
            tile = int(min(6144, max(1024, side - 2 * margin)) // 256 * 256)
        n_tiles = 1 if tile is None else int(np.ceil(H / tile) * np.ceil(W / tile))
        return {"tile": tile, "margin": margin, "size_px": size_px, "n_tiles": n_tiles,
                "budget_px": budget}

    def _scene_color_stats(self, sigma, with_lab):
        """ The colour scaling SLIC applies, computed once over the whole scene in row
        chunks (each smoothed with enough overlap to match smoothing the whole image): the
        min/max of the smoothed RGB over valid pixels (CPU path, one value for all bands as
        skimage does) and, for the GPU path, the per-channel min/max of its Lab. """
        H, W = self.valid_mask.shape
        pad = int(4.0 * sigma + 0.5) + 1 if sigma > 0 else 0
        rgb_min, rgb_max = np.inf, -np.inf
        lab_min, lab_max = np.full(3, np.inf), np.full(3, -np.inf)
        gpu = cuda_available()
        chunk = max(256, int((16e6 if gpu else 64e6) // max(1, W)))
        for r0 in range(0, H, chunk):
            r1 = min(H, r0 + chunk)
            m = self.valid_mask[r0:r1]
            if not m.any():
                continue
            a, b = max(0, r0 - pad), min(H, r1 + pad)
            img = np.transpose(self.ortho_data[:, a:b, :], (1, 2, 0)).astype(np.float32)
            img /= np.float32(255.0)
            if gpu:
                # same smoothing/Lab as the tiles' GPU path, over row chunks
                with torch.inference_mode():
                    t = torch_smooth(torch.from_numpy(img).cuda(), sigma)[r0 - a:r1 - a]
                    mt = torch.from_numpy(np.ascontiguousarray(m)).cuda()
                    v = t[mt]
                    rgb_min, rgb_max = min(rgb_min, float(v.min())), max(rgb_max, float(v.max()))
                    if with_lab:
                        lab = torch_rgb2lab(t)[mt]
                        lab_min = np.minimum(lab_min, lab.min(dim=0).values.cpu().numpy())
                        lab_max = np.maximum(lab_max, lab.max(dim=0).values.cpu().numpy())
                        del lab
                    del t, mt, v
                del img
                continue
            if sigma > 0:
                img = gaussian_filter(img, sigma=[sigma, sigma, 0])
            img = img[r0 - a:r1 - a]
            v = img[m]
            rgb_min, rgb_max = min(rgb_min, float(v.min())), max(rgb_max, float(v.max()))
            if with_lab:
                lab = rgb2lab(img)[m]
                lab_min = np.minimum(lab_min, lab.min(axis=0))
                lab_max = np.maximum(lab_max, lab.max(axis=0))
                del lab
            del img, v
        if gpu:
            torch.cuda.empty_cache()
        return (rgb_min, rgb_max), (lab_min, lab_max)

    def _segment_tiled(self, n_segments, compactness, sigma, use_gpu, plan):
        from scipy import ndimage
        import gc
        H, W = self.valid_mask.shape
        core, M, size_px = plan["tile"], plan["margin"], plan["size_px"]
        print(f"\n================== TILED SEGMENTATION ({'GPU' if use_gpu else 'CPU'}) ==================")
        print(f" -> {W}x{H} px is too large for one run: {plan['n_tiles']} tiles of {core} px "
              f"+ {M} px overlap, ~{size_px:.0f} px per segment.")
        print(" -> Measuring the scene's colour range once (shared by every tile)...")
        (rgb_min, rgb_max), lab_scale = self._scene_color_stats(sigma, with_lab=use_gpu)

        G = np.zeros((H, W), dtype=np.int32)
        next_id = 1
        touched = set()                # IDs that lost pixels at a seam
        tiles = [(r0, min(r0 + core, H), c0, min(c0 + core, W))
                 for r0 in range(0, H, core) for c0 in range(0, W, core)]
        for i, (r0, r1, c0, c1) in enumerate(tiles, start=1):
            wr0, wr1, wc0, wc1 = max(0, r0 - M), min(H, r1 + M), max(0, c0 - M), min(W, c1 + M)
            m = np.ascontiguousarray(self.valid_mask[wr0:wr1, wc0:wc1])
            n_valid = int(np.count_nonzero(m))
            if n_valid == 0:
                continue
            img = np.transpose(self.ortho_data[:, wr0:wr1, wc0:wc1], (1, 2, 0)).astype(np.float32)
            img /= np.float32(255.0)
            if use_gpu:
                # grid seeds over the whole window, as the one-run GPU path's compensation does
                n_t = max(1, int(round(m.size / size_px)))
                L = self._segment_image_gpu_slic(img, n_t, compactness, sigma, valid_mask=m,
                                                 lab_scale=lab_scale, verbose=False, prep_on_gpu=True)
            else:
                if sigma > 0:
                    img = gaussian_filter(img, sigma=[sigma, sigma, 0])
                n_t = max(1, int(round(n_valid / size_px)))
                L = slic_cpu_scaled(img, n_t, compactness * 100.0, m, rgb_min, rgb_max)
                L = self._absorb_mask_fragments(np.where(m, L, 0).astype(np.int32), verbose=False)
            del img

            h, w = L.shape
            k = int(L.max()) + 1
            flat = L.ravel()
            cnt = np.bincount(flat, minlength=k).astype(np.float64)
            rows = np.broadcast_to(np.arange(wr0, wr1, dtype=np.float32)[:, None], (h, w)).ravel()
            ly = np.bincount(flat, weights=rows, minlength=k) / np.maximum(cnt, 1)
            del rows
            cols = np.broadcast_to(np.arange(wc0, wc1, dtype=np.float32)[None, :], (h, w)).ravel()
            lx = np.bincount(flat, weights=cols, minlength=k) / np.maximum(cnt, 1)
            del cols

            # Segments touching a cut edge (not the image's own edge) may be incomplete.
            cut = np.zeros(k, dtype=bool)
            if wr0 > 0: cut[L[0, :]] = True
            if wr1 < H: cut[L[-1, :]] = True
            if wc0 > 0: cut[L[:, 0]] = True
            if wc1 < W: cut[L[:, -1]] = True
            keep = (cnt > 0) & ~cut & (ly >= r0) & (ly < r1) & (lx >= c0) & (lx < c1)
            keep[0] = False
            n_keep = int(keep.sum())
            new_ids = np.zeros(k, dtype=np.int32)
            new_ids[keep] = np.arange(next_id, next_id + n_keep, dtype=np.int32)
            next_id += n_keep

            Lg = new_ids[L]
            del L
            Gw = G[wr0:wr1, wc0:wc1]
            claim = Lg > 0
            # Pixels already claimed by an earlier tile stay with it, so the later segment's
            # edge there follows the earlier segment's natural boundary. (Giving them to the
            # nearer centroid instead cut the overlap along straight bisector lines.)
            clash = claim & (Gw > 0)
            if clash.any():
                touched.update(np.unique(Lg[clash]).tolist())
                claim &= ~clash
            Gw[claim] = Lg[claim]
            del Lg, claim, clash
            print(f"    -> Tile {i}/{len(tiles)}: kept {n_keep} segments.")
            gc.collect()

        # Seam clean-up: valid pixels no tile claimed.
        holes = self.valid_mask & (G == 0)
        if holes.any():
            comp, n = ndimage.label(holes)
            small, big = [], 0
            for j, sl in enumerate(ndimage.find_objects(comp), start=1):
                rr, cc = np.nonzero(comp[sl] == j)
                rr, cc = rr + sl[0].start, cc + sl[1].start
                if len(rr) >= 0.5 * size_px:
                    G[rr, cc] = next_id     # large enough to stand as a segment of its own
                    next_id += 1
                    big += 1
                else:
                    small.append((rr, cc))
            del comp
            if small:
                self._merge_pieces_into_neighbours(G, small)
            print(f" -> Seam clean-up: {len(small)} unclaimed slivers merged, {big} kept as segments.")
        del holes
        # Segments that lost pixels to a neighbour may have been cut in two.
        if touched:
            self._absorb_mask_fragments(G, ids=touched, verbose=False)
        n_final = int(np.count_nonzero(np.bincount(G.ravel(), minlength=next_id)[1:]))
        print(f" -> Tiled segmentation done: {n_final} segments.")
        return G

    # --- GPU SLIC segmentation ---
    def _segment_image_gpu_slic(self, image_float, n_segments, compactness, sigma,
                                valid_mask=None, lab_scale=None, verbose=True, prep_on_gpu=False):
        """
        Memory-optimized 5D K-Means clustering in PyTorch using CIE Lab color space.
        valid_mask defaults to the whole image's mask; a tile passes its own, plus
        lab_scale=(lab_min, lab_max) computed once for the whole scene (see
        _scene_color_stats) so every tile scales colour the same way. prep_on_gpu does the
        smoothing and Lab conversion on the GPU too (tiled path; ~6 s less per 25 Mpx tile).
        """
        import gc
        log = print if verbose else (lambda *a, **k: None)
        if valid_mask is None:
            valid_mask = self.valid_mask
        log(" -> Clearing CUDA VRAM caches...")
        torch.cuda.empty_cache()
        gc.collect()

        H, W, C = image_float.shape
        device = "cuda"

        img_tensor = None
        if prep_on_gpu:
            with torch.inference_mode():
                t = torch_smooth(torch.from_numpy(np.ascontiguousarray(image_float)).to(device), sigma)
                lab = torch_rgb2lab(t)
                del t
                if lab_scale is None:
                    vals = lab[torch.from_numpy(np.ascontiguousarray(valid_mask)).to(device)]
                    mn, mx = vals.min(dim=0).values, vals.max(dim=0).values
                    del vals
                else:
                    mn = torch.tensor(np.asarray(lab_scale[0]), dtype=torch.float32, device=device)
                    mx = torch.tensor(np.asarray(lab_scale[1]), dtype=torch.float32, device=device)
                rng = torch.where(mx > mn, mx - mn, torch.ones_like(mx))
                img_tensor = ((lab - mn) / rng).contiguous()
                del lab
            image_lab = None
        else:
            if sigma > 0:
                log(f" -> Applying Gaussian Pre-smoothing on CPU (sigma={sigma})...")
                image_float = gaussian_filter(image_float, sigma=[sigma, sigma, 0])

            log(" -> Converting RGB channels to uniform CIE Lab color space...")
            image_lab = rgb2lab(image_float)

        # Rescale each Lab channel independently into [0,1] using its own data range
        # over valid pixels. A single shared min/max across L, a and b (as opposed to
        # per-channel) lets whichever channel has the widest raw numeric range dominate
        # and crushes the others -- for real orthomosaic imagery the L channel's range
        # routinely dwarfs a/b's, which nearly zeroes out color contrast and collapses
        # segmentation into a spatial-only grid that ignores image content.
        if image_lab is not None:
            if lab_scale is None:
                valid_lab_values = image_lab[valid_mask]
                lab_min = valid_lab_values.min(axis=0)
                lab_max = valid_lab_values.max(axis=0)
                del valid_lab_values
            else:
                lab_min, lab_max = lab_scale
            lab_range = np.where(lab_max > lab_min, lab_max - lab_min, 1.0)
            image_lab = (image_lab - lab_min) / lab_range

        log(" -> Partitioning layout boundary coordinates...")
        r = H / W
        Gw = int(np.round(np.sqrt(n_segments / r)))
        Gh = int(np.round(n_segments / Gw))
        Gw = max(1, Gw)
        Gh = max(1, Gh)
        num_clusters = Gh * Gw

        Sy = H / Gh
        Sx = W / Gw
        S = np.sqrt(Sy * Sx)
        
        spatial_scale = compactness / S
        log(f" -> Configured spatial factor: {spatial_scale:.5f}. Targets: {num_clusters} seeds.")

        with torch.inference_mode():
            log(" -> Loading image and coordinate grids onto the GPU...")
            if img_tensor is None:
                img_tensor = torch.from_numpy(image_lab.astype(np.float32)).to(device)
                del image_lab

            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )

            pixel_feats = torch.cat([
                img_tensor,
                (y_coords * spatial_scale).unsqueeze(-1),
                (x_coords * spatial_scale).unsqueeze(-1)
            ], dim=-1)

            log(" -> Initializing grid centers on GPU...")
            gy_indices = torch.arange(Gh, device=device, dtype=torch.float32)
            gx_indices = torch.arange(Gw, device=device, dtype=torch.float32)
            cy = (gy_indices + 0.5) * Sy
            cx = (gx_indices + 0.5) * Sx
            cy_grid, cx_grid = torch.meshgrid(cy, cx, indexing='ij')

            cy_idx = torch.clamp(cy_grid.long(), 0, H - 1)
            cx_idx = torch.clamp(cx_grid.long(), 0, W - 1)
            init_rgb = img_tensor[cy_idx, cx_idx]

            centers = torch.cat([
                init_rgb,
                (cy_grid * spatial_scale).unsqueeze(-1),
                (cx_grid * spatial_scale).unsqueeze(-1)
            ], dim=-1)

            pixel_gy = (y_coords / Sy).long().clamp(0, Gh - 1)
            pixel_gx = (x_coords / Sx).long().clamp(0, Gw - 1)

            # Exclude nodata/invalid pixels from centroid accumulation, matching skimage's
            # mask= behavior on the CPU path (they still get assigned a nearest cluster for
            # the label pass below, but never pull a cluster's color/spatial centroid).
            valid_mask_tensor = torch.from_numpy(valid_mask).to(device)
            valid_flat = valid_mask_tensor.view(-1)

            num_iterations = 6
            log(f" -> Beginning PyTorch CUDA distance minimization loops ({num_iterations} iterations)...")
            for i in range(num_iterations):
                log(f"    -> Running K-Means iteration {i+1}/{num_iterations}...")
                min_distances = torch.full((H, W), 1e9, device=device, dtype=torch.float32)
                assigned_ids = torch.full((H, W), -1, device=device, dtype=torch.long)

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        neigh_gy = pixel_gy + dy
                        neigh_gx = pixel_gx + dx

                        out_of_bounds = (neigh_gy < 0) | (neigh_gy >= Gh) | (neigh_gx < 0) | (neigh_gx >= Gw)

                        clamped_gy = neigh_gy.clamp(0, Gh - 1)
                        clamped_gx = neigh_gx.clamp(0, Gw - 1)

                        sampled_centers = centers[clamped_gy, clamped_gx]
                        dist = torch.sum((pixel_feats - sampled_centers) ** 2, dim=-1)
                        dist = torch.where(out_of_bounds, torch.tensor(1e9, device=device), dist)

                        candidate_id = neigh_gy * Gw + neigh_gx
                        candidate_id = torch.where(out_of_bounds, torch.tensor(-1, device=device, dtype=torch.long), candidate_id)

                        closer_mask = dist < min_distances
                        min_distances = torch.where(closer_mask, dist, min_distances)
                        assigned_ids = torch.where(closer_mask, candidate_id, assigned_ids)

                flat_feats = pixel_feats.view(-1, 5)
                flat_ids = assigned_ids.view(-1)

                valid_mask_ids = (flat_ids >= 0) & valid_flat
                valid_feats = flat_feats[valid_mask_ids]
                valid_ids = flat_ids[valid_mask_ids]

                sum_feats = torch.zeros((num_clusters, 5), device=device, dtype=torch.float32)
                sum_feats.scatter_add_(0, valid_ids.unsqueeze(-1).expand(-1, 5), valid_feats)

                counts = torch.zeros((num_clusters, 1), device=device, dtype=torch.float32)
                counts.scatter_add_(0, valid_ids.unsqueeze(-1), torch.ones_like(valid_ids.unsqueeze(-1).float()))

                new_centers_flat = sum_feats / torch.clamp(counts, min=1.0)
                centers = new_centers_flat.view(Gh, Gw, 5)

            log(" -> Reconstructing label mapping from PyTorch CUDA buffers...")
            raw_labels = assigned_ids.cpu().numpy()

            del pixel_feats, img_tensor, y_coords, x_coords, centers, min_distances, assigned_ids
            torch.cuda.empty_cache()
            gc.collect()

        # skimage's SLIC cleans up fragmented micro-segments and splits oversized blobs
        # with this same connectivity pass; the custom K-means above has no equivalent
        # step of its own, and without it segmentation is either noisy (low compactness)
        # or grid-like (higher compactness) with no usable middle ground between them.
        log(" -> Enforcing spatial connectivity and merging fragmented micro-segments...")
        segment_size = valid_mask.sum() / num_clusters if num_clusters > 0 else 1
        min_size = int(0.5 * segment_size)
        max_size = int(3.0 * segment_size)
        cleaned_labels = _enforce_label_connectivity_cython(
            raw_labels[np.newaxis, ...].astype(np.int64), min_size, max_size, start_label=0
        )[0]

        result_array = np.where(valid_mask, cleaned_labels + 1, 0).astype(np.int32)

        return self._absorb_mask_fragments(result_array, verbose=verbose)

    @staticmethod
    def _absorb_mask_fragments(labels, ids=None, verbose=True):
        """
        Makes every segment a single 4-connected region (the connectivity rasterio's
        shapes() uses when vectorizing). Clustering assigns nodata pixels too, so a
        segment wrapping around a notch in a jagged image edge can be connected only
        through nodata; once the valid mask is applied it falls apart into a main body
        plus a few stray pixels, and vectorize_segments() then yields several polygons
        (and several feature rows) for one segment ID. Each stray piece is merged into
        the neighbouring segment it shares the longest border with; a piece with no
        valid neighbour at all keeps its pixels under a new ID. ids limits the check to
        those segment IDs (the tiled path only re-checks segments that lost pixels at a seam).
        """
        from scipy import ndimage
        fragments = []
        slices = ndimage.find_objects(labels)
        sids = range(1, len(slices) + 1) if ids is None else sorted(int(i) for i in ids)
        for sid in sids:
            sl = slices[sid - 1] if 0 < sid <= len(slices) else None
            if sl is None:
                continue
            comp, n = ndimage.label(labels[sl] == sid)
            if n < 2:
                continue
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            main = sizes.argmax()
            for k in range(1, n + 1):
                if k != main:
                    rr, cc = np.nonzero(comp == k)
                    fragments.append((rr + sl[0].start, cc + sl[1].start))
        if not fragments:
            return labels
        ObiaEngine._merge_pieces_into_neighbours(labels, fragments)
        if verbose:
            print(f" -> Merged {len(fragments)} stray mask-edge fragments into neighbouring segments.")
        return labels

    @staticmethod
    def _merge_pieces_into_neighbours(labels, pieces):
        """ Each piece (rows, cols) joins the neighbouring segment it shares the longest
        border with (pieces next to each other are resolved in rounds); a piece with no
        segment around it at all becomes its own new segment. Modifies labels in place. """
        for rr, cc in pieces:
            labels[rr, cc] = -1
        H, W = labels.shape
        next_id = int(labels.max()) + 1
        pending = pieces
        while pending:
            unresolved = []
            for rr, cc in pending:
                nb = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = rr + dr, cc + dc
                    ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
                    nb.append(labels[r2[ok], c2[ok]])
                nb = np.concatenate(nb)
                nb = nb[nb > 0]
                if len(nb):
                    labels[rr, cc] = np.bincount(nb).argmax()
                else:
                    unresolved.append((rr, cc))
            if len(unresolved) == len(pending):
                # Isolated islands (only nodata or other unresolved pieces around them).
                for rr, cc in unresolved:
                    labels[rr, cc] = next_id
                    next_id += 1
                break
            pending = unresolved
        return labels


    def vectorize_segments(self):
        transform = self.ortho_meta['transform']
        results = (
            {'properties': {'raster_val': int(v)}, 'geometry': s}
            for s, v in shapes(self.segmented_image, mask=(self.segmented_image > 0), transform=transform)
        )
        self.segments_gdf = gpd.GeoDataFrame.from_features(list(results))
        self.segments_gdf.set_crs(self.ortho_meta['crs'], inplace=True)
        # One row per segment: a segment in several pieces (e.g. after merging segments
        # that don't touch) would otherwise get one feature row per piece, each
        # describing only that piece.
        if self.segments_gdf['raster_val'].duplicated().any():
            self.segments_gdf = self.segments_gdf.dissolve(by='raster_val', as_index=False)
        return self.segments_gdf

    def build_adjacency_graph(self):
        """
        Region adjacency graph: which segment IDs share a border. Vectorized shift-compare
        over segmented_image (no per-segment looping). Rebuilt whenever segmentation
        topology changes (fresh segmentation, workspace load, or a merge).
        """
        if self.segmented_image is None:
            self.adjacency_graph = {}
            return self.adjacency_graph

        img = self.segmented_image
        pairs = []

        left, right = img[:, :-1], img[:, 1:]
        m = (left != right) & (left > 0) & (right > 0)
        pairs.append(np.stack([left[m], right[m]], axis=1))

        top, bottom = img[:-1, :], img[1:, :]
        m2 = (top != bottom) & (top > 0) & (bottom > 0)
        pairs.append(np.stack([top[m2], bottom[m2]], axis=1))

        all_pairs = np.concatenate(pairs, axis=0)
        if all_pairs.shape[0] > 0:
            all_pairs = np.unique(np.sort(all_pairs, axis=1), axis=0)

        graph = defaultdict(set)
        for a, b in all_pairs:
            a, b = int(a), int(b)
            graph[a].add(b)
            graph[b].add(a)

        self.adjacency_graph = dict(graph)
        return self.adjacency_graph


    # --- Feature extraction ---
    def compute_all_features_parallel(self, max_workers=None):
        """
        Feature extraction of the 19 / 23 variables (23 when a DSM is attached), one segment
        at a time on its bounding box (views into the arrays, no copies). Runs in one
        thread: measured on 105k and 7k segments, a thread pool was 1.5x SLOWER than a
        plain loop (the per-segment numpy calls are too small to release the GIL usefully).
        max_workers is kept for callers and ignored. With feature_pixel_size set finer
        than the working grid, colour/texture come from the source file instead (see
        _compute_features_from_file).
        """
        if self.segments_gdf is None or self.ortho_data is None:
            return None
        if self.feature_grid() is not None:
            return self._compute_features_from_file()

        from scipy import ndimage
        seg = self.segmented_image
        geom = self._segment_geometry()
        slices = ndimage.find_objects(seg)
        dsm, slope = self.dsm_data, self.slope_data
        rgb = self.original_ortho_data
        total = len(geom)
        print(f"Measuring features for {total} segments...")
        feature_dicts = []
        for i, (sid, (area, perimeter, conv_area)) in enumerate(geom.items(), start=1):
            sl = slices[sid - 1] if 0 < sid <= len(slices) else None
            if sl is None:
                continue
            res = extract_single_segment_features(
                rgb[:, sl[0], sl[1]], seg[sl], sid, area, perimeter, conv_area,
                dsm[sl] if dsm is not None else None, slope[sl] if slope is not None else None)
            if res is not None:
                feature_dicts.append(res)
            if i % 20000 == 0:
                print(f" -> Processed {i}/{total} segment features...")

        self.features_df = pd.DataFrame(feature_dicts).set_index('raster_val')
        return self.features_df

    # --- Colour/texture measured finer than the segmentation grid ---
    #
    # Segments are drawn on the working grid, but their colour and texture can be measured
    # from the source file at a finer feature resolution (feature_pixel_size, CRS units;
    # None = the working grid). Each segment's working-grid labels are scaled up (nearest)
    # onto the feature grid, block by block, so only a block of the fine image is in memory
    # at a time. Area/perimeter (from the polygons) and the DSM statistics are unchanged;
    # GLCM texture compares neighbouring pixels of the feature grid, so it describes finer
    # detail than at the working grid.

    feature_pixel_size = None

    def feature_grid(self):
        """ (height, width) of the feature grid over the whole source file, or None when
        features are measured on the working grid. """
        fp = self.feature_pixel_size
        if not fp or self.native_pixel_size is None or self.native_crs_is_geographic \
                or self.ortho_path is None or self.ortho_data is None:
            return None
        kf = max(1.0, fp / self.native_pixel_size)
        Hf = max(1, int(round(self.native_height / kf)))
        Wf = max(1, int(round(self.native_width / kf)))
        H, W = self.ortho_data.shape[1:]
        if Hf < H * 1.05 and Wf < W * 1.05:
            return None   # not meaningfully finer than the working grid
        return Hf, Wf

    def _file_feature_tasks(self, src, owned, slices, geom, Hf, Wf):
        """ Task tuples for extract_single_segment_features for the segments in owned (all
        inside one block): reads that block of the source at the feature resolution and
        scales the working-grid labels (and DSM/slope) up onto it. """
        seg = self.segmented_image
        H, W = seg.shape
        Hn, Wn = src.height, src.width
        wr0 = min(slices[s - 1][0].start for s in owned)
        wr1 = max(slices[s - 1][0].stop for s in owned)
        wc0 = min(slices[s - 1][1].start for s in owned)
        wc1 = max(slices[s - 1][1].stop for s in owned)
        nr0, nr1 = int(np.floor(wr0 * Hn / H)), min(Hn, int(np.ceil(wr1 * Hn / H)))
        nc0, nc1 = int(np.floor(wc0 * Wn / W)), min(Wn, int(np.ceil(wc1 * Wn / W)))
        fh = max(1, int(round((nr1 - nr0) * Hf / Hn)))
        fw = max(1, int(round((nc1 - nc0) * Wf / Wn)))
        win = rasterio.windows.Window(nc0, nr0, nc1 - nc0, nr1 - nr0)
        same = (fh == nr1 - nr0 and fw == nc1 - nc0)
        rs = rasterio.enums.Resampling.nearest if same else rasterio.enums.Resampling.average
        rgb = to_uint8(src.read((1, 2, 3), window=win, out_shape=(3, fh, fw), resampling=rs))
        try:
            fmask = src.read_masks(1, window=win, out_shape=(fh, fw),
                                   resampling=rasterio.enums.Resampling.nearest) > 0
        except Exception:
            fmask = None
        # Working-grid row/column under the centre of each feature-grid pixel.
        rmap = np.floor((nr0 + (np.arange(fh) + 0.5) * (nr1 - nr0) / fh) * H / Hn).astype(np.int64).clip(0, H - 1)
        cmap = np.floor((nc0 + (np.arange(fw) + 0.5) * (nc1 - nc0) / fw) * W / Wn).astype(np.int64).clip(0, W - 1)
        labf = seg[rmap[:, None], cmap[None, :]]
        if fmask is not None:
            labf = np.where(fmask, labf, 0)
        dsm, slope = self.dsm_data, self.slope_data
        tasks = []
        for sid in owned:
            sl = slices[sid - 1]
            fr0, fr1 = np.searchsorted(rmap, sl[0].start), np.searchsorted(rmap, sl[0].stop)
            fc0, fc1 = np.searchsorted(cmap, sl[1].start), np.searchsorted(cmap, sl[1].stop)
            if fr1 <= fr0 or fc1 <= fc0:
                continue
            rr, cc = rmap[fr0:fr1, None], cmap[None, fc0:fc1]
            area, perimeter, conv_area = geom[sid]
            tasks.append((rgb[:, fr0:fr1, fc0:fc1], labf[fr0:fr1, fc0:fc1], sid, area, perimeter, conv_area,
                          dsm[rr, cc] if dsm is not None else None,
                          slope[rr, cc] if slope is not None else None))
        return tasks

    def _segment_geometry(self, sids=None):
        gdf = self.segments_gdf
        if sids is not None:
            gdf = gdf[gdf['raster_val'].isin(list(sids))]
        hull = gdf.geometry.convex_hull.area.values
        return {int(v): (float(a), float(p), float(h) if h > 0 else float(a))
                for v, a, p, h in zip(gdf['raster_val'].values, gdf.geometry.area.values,
                                      gdf.geometry.length.values, hull)}

    def _compute_features_from_file(self):
        from scipy import ndimage
        Hf, Wf = self.feature_grid()
        seg = self.segmented_image
        H, W = seg.shape
        geom = self._segment_geometry()
        slices = ndimage.find_objects(seg)
        sids = [s for s in geom if 0 < s <= len(slices) and slices[s - 1] is not None]
        fp = self.feature_pixel_size
        print(f"Measuring colour and texture at {fp * 100:.2f} cm/px from the source file "
              f"({Wf}x{Hf} px grid) for {len(sids)} segments...")
        # Blocks of the working grid; a segment belongs to the block holding its bbox's
        # top-left corner, and the block's read window grows to cover its segments whole.
        scale = Hf / H
        B = max(256, int(4096 / max(1.0, scale)))   # ~4096 feature px per block side
        blocks = defaultdict(list)
        for s in sids:
            sl = slices[s - 1]
            blocks[(sl[0].start // B, sl[1].start // B)].append(s)
        feature_dicts = []
        done = 0
        next_report = 20000
        with rasterio.open(self.ortho_path) as src:
            for key in sorted(blocks):
                for t in self._file_feature_tasks(src, blocks[key], slices, geom, Hf, Wf):
                    res = extract_single_segment_features(*t)   # one thread: see compute_all_features_parallel
                    if res is not None:
                        feature_dicts.append(res)
                done += len(blocks[key])
                if done >= next_report:
                    print(f" -> Processed {done}/{len(sids)} segment features...")
                    next_report += 20000
        self.features_df = pd.DataFrame(feature_dicts).set_index('raster_val')
        return self.features_df

    def compute_neighbor_relative_features(self):
        """
        Second pass over the already-computed features_df: for each segment, how its
        brightness (and elevation, if DSM active) compares to the mean of its adjacent
        neighbors. A segment that is both darker AND lower than its surroundings is a
        canopy-gap candidate even when its own absolute values look unremarkable, which
        dsm_mean/brightness_mean alone can't distinguish from e.g. a uniformly dark object.
        Must run after both compute_all_features_parallel() and build_adjacency_graph(),
        since it needs other segments' already-computed values.
        """
        if self.features_df is None or self.adjacency_graph is None:
            return None

        # dsm_mean is always a features_df column (defaults to 0.0 per-segment when no
        # DSM is loaded), so its presence alone doesn't mean DSM is actually active --
        # check dsm_data directly, matching the convention used everywhere else in the app.
        has_dsm = self.has_dsm and 'dsm_mean' in self.features_df.columns
        dsm_dict = self.features_df['dsm_mean'].to_dict() if has_dsm else {}
        bright_dict = self.features_df['brightness_mean'].to_dict()

        dsm_rel, bright_rel = {}, {}
        for seg_id, neighbors in self.adjacency_graph.items():
            if seg_id not in bright_dict:
                continue
            valid_n = [n for n in neighbors if n in bright_dict]
            if not valid_n:
                dsm_rel[seg_id] = 0.0
                bright_rel[seg_id] = 0.0
                continue
            bright_rel[seg_id] = bright_dict[seg_id] - sum(bright_dict[n] for n in valid_n) / len(valid_n)
            dsm_rel[seg_id] = (dsm_dict[seg_id] - sum(dsm_dict[n] for n in valid_n) / len(valid_n)) if has_dsm else 0.0

        # Segments absent from the adjacency graph (isolated islands) default to 0.
        for seg_id in self.features_df.index:
            bright_rel.setdefault(seg_id, 0.0)
            dsm_rel.setdefault(seg_id, 0.0)

        self.features_df['brightness_rel_neighbors'] = pd.Series(bright_rel).reindex(self.features_df.index).fillna(0.0)
        if has_dsm:
            self.features_df['dsm_rel_neighbors'] = pd.Series(dsm_rel).reindex(self.features_df.index).fillna(0.0)

        return self.features_df

    def grow_selection(self, seed_ids, tolerance,
                        feature_cols=('r_mean', 'g_mean', 'b_mean', 'brightness_mean', 'contrast', 'homogeneity')):
        """
        BFS outward from seed_ids across adjacency_graph, pulling in a neighbor if its
        distance to the seed group's mean (Euclidean, in per-column-std-normalized units
        so the tolerance's meaning stays stable regardless of which region is seeded) is
        within tolerance. Always recomputed fresh from seed_ids -- never grows
        incrementally from a prior result -- so lowering the tolerance shrinks the
        selection back rather than only ever ratcheting outward.

        Two things worth knowing if growth looks wrong (leaks into visually different
        terrain while stopping short of a same-material region elsewhere):

        1. This is a uniform-tolerance flood-fill, not a best-first/priority search --
           every neighbor within `tolerance` of the seed mean is pulled in on equal
           footing, regardless of how much *more* similar one candidate is than
           another. There's no preference for growing into the closest match first.
        2. `col_std` (used to normalize each feature so the tolerance means the same
           thing regardless of units/scale) is computed across the *entire scene's*
           segments, not just the local neighborhood. In a spectrally diverse scene,
           that can make the effective tolerance looser than it feels, letting
           genuinely different-looking neighboring material through.

        `feature_cols` originally defaulted to spectral-only (r/g/b/brightness mean).
        `contrast` and `homogeneity` (GLCM texture, already computed per segment) were
        added to the default here because two materials can have near-identical mean
        color yet obviously different texture to the eye (e.g. loose gravel vs.
        compacted dirt/pavement) -- spectral-only distance can't tell them apart, but
        texture usually can. Pass a different feature_cols tuple to tune this further
        (e.g. add 'entropy' or drop 'homogeneity') without changing this method.
        """
        seed_ids = set(int(s) for s in seed_ids)
        if not seed_ids or self.features_df is None or self.adjacency_graph is None:
            return set(seed_ids)

        cols = list(feature_cols)
        feat_lookup = {int(i): row for i, row in zip(self.features_df.index, self.features_df[cols].values)}
        valid_seeds = [s for s in seed_ids if s in feat_lookup]
        if not valid_seeds:
            return set(seed_ids)

        seed_mean = np.mean([feat_lookup[s] for s in valid_seeds], axis=0)
        col_std = np.std(self.features_df[cols].values, axis=0)
        col_std = np.where(col_std < 1e-6, 1.0, col_std)
        max_dist = tolerance * np.sqrt(len(cols))

        selected = set(valid_seeds)
        frontier = list(valid_seeds)
        while frontier:
            next_frontier = []
            for seg_id in frontier:
                for neighbor in self.adjacency_graph.get(seg_id, ()):
                    if neighbor in selected or neighbor not in feat_lookup:
                        continue
                    dist = np.sqrt(np.sum(((feat_lookup[neighbor] - seed_mean) / col_std) ** 2))
                    if dist <= max_dist:
                        selected.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return selected

    def merge_segments(self, segment_ids):
        """
        Merges the given segment IDs into a single object. Survivor = largest-area
        segment among them, chosen before any mutation. Ordering matters: relabel the
        raster first (revectorize needs the already-merged raster), then revectorize
        (targeted feature recompute needs the fresh gdf's bounding box), then recompute
        features for just the survivor (a full compute_all_features_parallel() re-run
        would be correct too but needlessly slow for an interactive click), then rebuild
        the adjacency graph last (needs the final raster). Does NOT recompute
        neighbor-relative features itself -- a merge changes the survivor's row AND
        every one of its neighbors' relative features, so that full (cheap, dict-based)
        re-pass is the caller's responsibility, not scoped to "just the survivor" here.

        Returns (survivor_id, absorbed_ids), or None if the merge couldn't proceed.
        """
        segment_ids = set(int(s) for s in segment_ids)
        valid_ids = [s for s in segment_ids if s in self.features_df.index]
        if len(valid_ids) < 2:
            print("Merge skipped: fewer than two valid segments in selection.")
            return None

        survivor_id = int(self.features_df.loc[valid_ids, 'area'].idxmax())
        absorbed_ids = [s for s in valid_ids if s != survivor_id]

        mask = np.isin(self.segmented_image, valid_ids)
        self.segmented_image[mask] = survivor_id

        self.vectorize_segments()

        survivor_rows = self.segments_gdf[self.segments_gdf['raster_val'] == survivor_id]
        if len(survivor_rows) == 0:
            print(f"Warning: survivor segment {survivor_id} vanished after revectorize (unexpected).")
            return None

        unioned_geom = survivor_rows.geometry.union_all()
        area, perimeter = unioned_geom.area, unioned_geom.length
        conv_area = unioned_geom.convex_hull.area if unioned_geom.convex_hull else area

        minx, miny, maxx, maxy = survivor_rows.total_bounds
        affine = self.ortho_meta['transform']
        col_min, row_max = ~affine * (minx, miny)
        col_max, row_min = ~affine * (maxx, maxy)
        c_min, c_max = max(0, int(np.floor(col_min))), min(self.segmented_image.shape[1], int(np.ceil(col_max)))
        r_min, r_max = max(0, int(np.floor(row_min))), min(self.segmented_image.shape[0], int(np.ceil(row_max)))

        fgrid = self.feature_grid()
        if fgrid is not None:
            # Same feature grid as the rest of the segments (see _compute_features_from_file).
            slices = [None] * survivor_id
            slices[survivor_id - 1] = (slice(r_min, r_max), slice(c_min, c_max))
            with rasterio.open(self.ortho_path) as src:
                tasks = self._file_feature_tasks(src, [survivor_id], slices,
                                                 {survivor_id: (area, perimeter, conv_area)}, *fgrid)
            result = extract_single_segment_features(*tasks[0]) if tasks else None
        else:
            crop_rgb = self.original_ortho_data[:, r_min:r_max, c_min:c_max].copy()
            crop_mask = self.segmented_image[r_min:r_max, c_min:c_max].copy()
            crop_dsm = self.dsm_data[r_min:r_max, c_min:c_max].copy() if self.dsm_data is not None else None
            crop_slope = self.slope_data[r_min:r_max, c_min:c_max].copy() if self.slope_data is not None else None
            result = extract_single_segment_features(crop_rgb, crop_mask, survivor_id, area, perimeter, conv_area, crop_dsm, crop_slope)
        if result is None:
            print(f"Warning: feature extraction failed for merged survivor {survivor_id}.")
            return None

        if 'hag_mean' in self.features_df.columns:
            parts = self.features_df.loc[valid_ids]
            result = dict(result, hag_mean=float(np.average(parts['hag_mean'], weights=parts['area'])))
        deep_cols = [c for c in self.features_df.columns if c.startswith(self.DEEP_PREFIX)]
        if deep_cols:
            parts = self.features_df.loc[valid_ids]
            result = dict(result, **{c: float(np.average(parts[c], weights=parts['area'])) for c in deep_cols})
        self.features_df = self.features_df.drop(index=[i for i in absorbed_ids if i in self.features_df.index])
        for col, val in result.items():
            if col != 'raster_val':
                self.features_df.loc[survivor_id, col] = val

        self.build_adjacency_graph()

        return survivor_id, absorbed_ids

    BASE_FEATURES = [
        'r_mean', 'r_std', 'g_mean', 'g_std', 'b_mean', 'b_std',
        'area', 'perimeter', 'circularity', 'convexity',
        'contrast', 'homogeneity', 'entropy', 'energy',
        'exg_mean', 'vari_mean', 'ngrdi_mean', 'brightness_mean',
        'brightness_rel_neighbors'
    ]
    DSM_FEATURES = ['dsm_mean', 'dsm_std', 'slope_mean', 'dsm_rel_neighbors', 'hag_mean']
    DEEP_PREFIX = 'deep_'
    use_deep_features = False      # classifier also uses the deep_* columns (see add_deep_columns)

    def active_feature_columns(self, include_deep=True):
        """ The features the Random Forest uses: the 19 base features; with a DSM the
        elevation features (23, or 24 with height above ground); with use_deep_features the
        deep_* appearance columns too. Discover passes include_deep=False -- it adds the deep
        features itself. """
        if self.features_df is None:
            return list(self.BASE_FEATURES)
        cols = list(self.BASE_FEATURES)
        if self.has_dsm and 'dsm_mean' in self.features_df.columns:
            cols += [c for c in self.DSM_FEATURES if c in self.features_df.columns]
        if include_deep and self.use_deep_features:
            cols += [c for c in self.features_df.columns if c.startswith(self.DEEP_PREFIX)]
        return cols

    # --- Height above ground (DSM minus a local ground estimate) ---
    GROUND_WINDOW_M = 40.0

    def ground_surface(self, window_m=None):
        """ A ground estimate on the working grid from the DSM alone: the DSM read at ~1 m,
        a morphological opening (minimum then maximum filter) over window_m, which removes
        anything narrower than the window (crowns, buildings), then smoothed. Under closed
        canopy wider than the window it rides on the canopy and height is underestimated --
        the usual limit of DSM-only ground filtering. """
        from scipy import ndimage
        from rasterio.warp import reproject
        window_m = window_m or self.GROUND_WINDOW_M
        dst_tr, dst_shape, dst_crs = self.ortho_meta["transform"], self.ortho_data.shape[1:], self.ortho_meta["crs"]
        with rasterio.open(self.dsm_path) as src:
            # Block minimum to ~1 m pixels (the lowest return in each block is the best
            # ground candidate; rasterio reads can't resample with "min").
            b = max(1, int(round(1.0 / abs(src.transform.a))))
            full = np.ma.filled(src.read(1, masked=True).astype(np.float32), np.nan)
            H0, W0 = (full.shape[0] // b) * b, (full.shape[1] // b) * b
            blocks = full[:H0, :W0].reshape(H0 // b, b, W0 // b, b)
            del full
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dem = np.nanmin(blocks, axis=(1, 3)).astype(np.float32)
            del blocks
            tr = src.transform * src.transform.scale(b, b)
            crs = src.crs or dst_crs
        px = abs(tr.a)
        k = max(3, int(round(window_m / px)) | 1)
        valid = np.isfinite(dem)
        big = np.float32(np.nanmax(dem) + 1000) if valid.any() else np.float32(0)
        g = ndimage.minimum_filter(np.where(valid, dem, big), size=k)      # nodata never lowers ground
        g = ndimage.maximum_filter(g, size=k)
        g = np.where(g >= big, np.nan, g)
        gf = np.where(np.isfinite(g), g, 0).astype(np.float32)
        wts = ndimage.gaussian_filter(np.isfinite(g).astype(np.float32), k / 4)
        g = ndimage.gaussian_filter(gf, k / 4) / np.maximum(wts, 1e-6)
        g = np.where(wts > 0.05, g, np.nan).astype(np.float32)
        out = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(g, out, src_transform=tr, src_crs=crs, src_nodata=np.nan,
                  dst_transform=dst_tr, dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=rasterio.enums.Resampling.bilinear)
        return out

    def compute_height_features(self):
        """ hag_mean: each segment's mean height above the ground estimate (metres), from
        the DSM on the working grid. Adds/overwrites the features_df column. """
        if not self.has_dsm or self.features_df is None or self.segmented_image is None:
            return None
        from scipy import ndimage
        print(f" -> Height above ground: ground estimated from the DSM ({self.GROUND_WINDOW_M:.0f} m opening)...")
        hag = self.dsm_data - self.ground_surface()
        ok = np.isfinite(hag)
        ids = self.features_df.index.values.astype(np.int64)
        lab = self.segmented_image
        s = ndimage.sum(np.where(ok, hag, 0).astype(np.float64), lab, ids)
        n = ndimage.sum(ok, lab, ids)
        del hag, ok
        self.features_df['hag_mean'] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        return self.features_df

    def add_deep_columns(self, deep, n_pc=10):
        """ deep_* columns in features_df from cached DINOv2 features (obia_discover): the
        first n_pc principal components of each patch size, fitted over all segments.
        Segments without deep features get 0. """
        from sklearn.decomposition import PCA
        self.features_df = self.features_df.drop(
            columns=[c for c in self.features_df.columns if c.startswith(self.DEEP_PREFIX)])
        pos = {int(s): i for i, s in enumerate(deep["ids"])}
        rows = np.array([pos.get(int(s), -1) for s in self.features_df.index])
        have = rows >= 0
        for k in sorted(deep["feats"]):
            X = deep["feats"][k].astype(np.float32)
            Z = PCA(n_components=min(n_pc, X.shape[1]), random_state=0).fit_transform(X)
            tag = f"{float(k):g}m" if not isinstance(k, str) else k.replace(".", "_")
            for j in range(Z.shape[1]):
                col = np.zeros(len(rows), dtype=np.float32)
                col[have] = Z[rows[have], j]
                self.features_df[f"{self.DEEP_PREFIX}{tag}_pc{j + 1}"] = col
        return [c for c in self.features_df.columns if c.startswith(self.DEEP_PREFIX)]

    def compute_class_separability(self, training_segments, features=None):
        """
        Class separability of the current training samples, before training.

        training_segments: {class_name: [segment ids]}. Returns a dict with
          'classes'      class names that have >= 2 samples (others are listed in 'skipped')
          'n'            samples per class
          'features'     feature names used
          'mean', 'std'  DataFrames (class x feature) in the features' own units
          'jm'           DataFrame (class x class), Jeffries-Matusita distance, 0..2
          'jm_feature'   {(a, b): Series of single-feature JM, highest first}
          'low_n'        classes with fewer samples than features (estimate less reliable)

        JM = 2 (1 - exp(-B)), B the Bhattacharyya distance between two Gaussians fitted to
        each class. Features are standardised by the pooled training spread so none
        dominates by scale, and each class covariance is a Ledoit-Wolf shrinkage estimate,
        which stays stable when a class has fewer samples than there are features.
        Common reading: >= 1.9 well separated, 1.0-1.9 partly separable, < 1.0 poorly
        separated.
        """
        from sklearn.covariance import LedoitWolf
        features = features or self.active_feature_columns()
        n = {c: len([s for s in ids if s in self.features_df.index]) for c, ids in training_segments.items()}
        classes = [c for c in training_segments if n[c] >= 2]
        skipped = [c for c in training_segments if n[c] < 2]
        data = {c: self.features_df.loc[[s for s in training_segments[c] if s in self.features_df.index],
                                        features].astype(float).fillna(0.0).values for c in classes}
        mean = pd.DataFrame({c: data[c].mean(0) for c in classes}, index=features).T
        std = pd.DataFrame({c: data[c].std(0) for c in classes}, index=features).T
        result = dict(classes=classes, skipped=skipped, n={c: n[c] for c in classes}, features=features,
                      mean=mean, std=std, jm=pd.DataFrame(index=classes, columns=classes, dtype=float),
                      jm_feature={}, low_n=[c for c in classes if n[c] < len(features)])
        if len(classes) < 2:
            return result

        allx = np.vstack([data[c] for c in classes])
        mu0, sd0 = allx.mean(0), allx.std(0)
        sd0[sd0 == 0] = 1.0
        z = {c: (data[c] - mu0) / sd0 for c in classes}
        mu = {c: z[c].mean(0) for c in classes}
        cov = {c: LedoitWolf().fit(z[c]).covariance_ + 1e-6 * np.eye(len(features)) for c in classes}
        var = {c: z[c].var(0) + 1e-6 for c in classes}

        def jm_from_b(b):
            return 2.0 * (1.0 - np.exp(-b))

        for i, a in enumerate(classes):
            result['jm'].loc[a, a] = np.nan
            for b in classes[i + 1:]:
                d = mu[a] - mu[b]
                s = (cov[a] + cov[b]) / 2.0
                _, ld = np.linalg.slogdet(s)
                _, lda = np.linalg.slogdet(cov[a])
                _, ldb = np.linalg.slogdet(cov[b])
                bh = d @ np.linalg.solve(s, d) / 8.0 + 0.5 * (ld - 0.5 * (lda + ldb))
                result['jm'].loc[a, b] = result['jm'].loc[b, a] = jm_from_b(bh)
                # Single-feature JM: which features on their own pull this pair apart.
                vs = (var[a] + var[b]) / 2.0
                b1 = d ** 2 / (8.0 * vs) + 0.5 * np.log(vs / np.sqrt(var[a] * var[b]))
                result['jm_feature'][(a, b)] = pd.Series(jm_from_b(b1), index=features).sort_values(ascending=False)
        return result

    def apply_sieve_filter(self, classified_array, size_threshold):
        if size_threshold <= 0:
            return classified_array
        return sieve(classified_array.astype(np.int32), size_threshold).astype(np.uint8)

    def write_qgis_styling(self, qml_path, classes):
        qgis_root = ET.Element("qgis", version="3.28.0", hasScaleBasedVisibilityFlag="0")
        pipe = ET.SubElement(qgis_root, "pipe")
        provider = ET.SubElement(pipe, "provider")
        ET.SubElement(provider, "resampling", 
                      zoomedOutResamplingMethod="nearestNeighbour", 
                      zoomedInResamplingMethod="nearestNeighbour", 
                      maxOversampling="2", 
                      enabled="false")
        
        renderer = ET.SubElement(pipe, "rasterrenderer", type="paletted", opacity="1", band="1", nodataColor="")
        ET.SubElement(renderer, "rasterTransparency")
        color_palette = ET.SubElement(renderer, "colorPalette")
        ET.SubElement(color_palette, "paletteEntry", color="#000000", alpha="0", value="0", label="Unclassified")
        
        for idx, (name, data) in enumerate(classes.items(), start=1):
            color = data["color"]
            hex_color = color if isinstance(color, str) else color.name()
            ET.SubElement(color_palette, "paletteEntry", color=hex_color, alpha="255", value=str(idx), label=name)
            
        tree = ET.ElementTree(qgis_root)
        ET.indent(tree, space="  ", level=0)
        tree.write(qml_path, encoding="utf-8", xml_declaration=True)

    # --- Accuracy assessment (Tab 4) ---
    def _new_accuracy_points_gdf(self):
        """ Empty GeoDataFrame with the accuracy-points schema, CRS matching the
        currently loaded orthomosaic. Private helper -- callers should go through
        generate_random_accuracy_points/import_accuracy_points_csv, which create this
        lazily the first time it's needed. """
        crs = self.ortho_meta['crs'] if self.ortho_meta else None
        return gpd.GeoDataFrame(
            {
                'point_id': pd.Series(dtype='int64'),
                'reference_class': pd.Series(dtype='object'),
                'source': pd.Series(dtype='object'),
                'predicted_class': pd.Series(dtype='object'),
                'stratum': pd.Series(dtype='object'),
            },
            geometry=gpd.GeoSeries([], dtype='geometry'), crs=crs
        )

    def generate_random_accuracy_points(self, sample_size, exclude_segment_ids=None, source_label='expert_visual'):
        """
        Simple random sampling of point locations across the valid-data area of the
        orthomosaic, deliberately NOT stratified by predicted class -- stratifying by
        the classifier's own output would make the reference sample's *location*
        depend on the model being assessed, the same problem this whole feature exists
        to avoid, just moved one step earlier in the pipeline.

        Always draws sample_size *new* points on top of whatever already exists (of any
        source, CSV-imported or previously generated) -- pressing the button again adds
        another batch rather than topping up toward a fixed total, so it behaves the
        same whether pressed once after a fresh load or again after a workspace reload
        that already restored points.

        exclude_segment_ids: OBIA training-sample segment IDs (owned by obia_app.py's
        self.classes, not by the engine -- passed in like grow_selection/merge_segments
        take IDs as params) to keep out of the eligible area, so the reference sample
        never doubles as training data for the model it's checking. Silently skipped
        (no filtering) if no segmentation has been run yet.

        Returns the number of points actually added (can be less than requested if the
        eligible area doesn't contain enough distinct pixels).
        """
        if self.ortho_data is None or self.valid_mask is None:
            raise ValueError("Load an orthomosaic before generating accuracy points.")
        if self.accuracy_points_gdf is None:
            self.accuracy_points_gdf = self._new_accuracy_points_gdf()

        existing_count = len(self.accuracy_points_gdf)
        n_needed = max(0, int(sample_size))
        if n_needed == 0:
            return 0

        eligible = self.valid_mask.copy()
        if exclude_segment_ids and self.segmented_image is not None:
            eligible &= ~np.isin(self.segmented_image, list(exclude_segment_ids))

        eligible_flat_idx = np.flatnonzero(eligible)
        if eligible_flat_idx.size == 0:
            print(" -> No eligible pixels to sample from (valid area fully excluded by training segments?).")
            return 0

        n_draw = min(n_needed, int(eligible_flat_idx.size))
        if n_draw < n_needed:
            print(f" -> Only {eligible_flat_idx.size} eligible pixel(s) available -- drawing {n_draw} of the requested {n_needed} new point(s).")

        # Rejection sampling on POSITIONS into eligible_flat_idx, not
        # np.random.choice(..., replace=False) -- that call's default algorithm
        # generates a full random permutation of the entire eligible-pixel array
        # before slicing it down, which for a large orthomosaic's valid area can mean
        # shuffling an array of tens to hundreds of millions of entries just to draw a
        # few hundred points. n_draw is always small in practice (a UI-driven sample
        # size), so drawing oversized batches of random positions and deduping via a
        # set converges in one or two rounds and stays O(n_draw), not O(n_eligible).
        n_eligible = int(eligible_flat_idx.size)
        chosen_positions = set()
        while len(chosen_positions) < n_draw:
            remaining = n_draw - len(chosen_positions)
            batch = np.random.randint(0, n_eligible, size=remaining * 2 + 16)
            for pos in batch:
                if len(chosen_positions) >= n_draw:
                    break
                chosen_positions.add(int(pos))
        chosen_flat = eligible_flat_idx[np.fromiter(chosen_positions, dtype=np.int64, count=n_draw)]
        rows, cols = np.unravel_index(chosen_flat, eligible.shape)

        transform = self.ortho_meta['transform']
        geoms = [Point(*(transform * (float(c) + 0.5, float(r) + 0.5))) for r, c in zip(rows, cols)]

        next_id = self.next_accuracy_point_id
        new_gdf = gpd.GeoDataFrame(
            {
                'point_id': list(range(next_id, next_id + n_draw)),
                'reference_class': [None] * n_draw,
                'source': [source_label] * n_draw,
                'predicted_class': [None] * n_draw,
            },
            geometry=geoms, crs=self.ortho_meta['crs']
        )
        self.next_accuracy_point_id = next_id + n_draw

        self.accuracy_points_gdf = gpd.GeoDataFrame(
            pd.concat([self.accuracy_points_gdf, new_gdf], ignore_index=True), crs=self.ortho_meta['crs']
        )
        print(f" -> Generated {n_draw} new random reference point(s) ({existing_count + n_draw} total).")
        return n_draw

    def generate_stratified_accuracy_points(self, classified_raster, class_names_ordered, total,
                                            min_per_class, exclude_segment_ids=None, seed=None,
                                            source_label='expert_visual'):
        """
        Stratified random sampling with the final classified map's classes as strata
        (Olofsson et al. 2014): each class gets max(min_per_class, its area share of
        `total`) points, drawn at random from that class's mapped pixels. Rare classes get
        enough points for a usable user's accuracy, and run_accuracy_assessment weights
        every estimate by the strata's mapped areas, so overall accuracy and class areas
        stay unbiased (Stehman 2014's estimator stays valid even if the map is
        reclassified after sampling).

        Training segments are excluded from where points may fall (never from the strata
        areas). Each point records its stratum; self.accuracy_strata keeps the strata's
        pixel counts, the pixel area and the seed (reproducible: same map + seed = same
        points). Returns {class: points drawn}.
        """
        if self.ortho_meta is None or classified_raster is None:
            raise ValueError("Classify the image before drawing a stratified sample (the strata are the map's classes).")
        if self.accuracy_points_gdf is None:
            self.accuracy_points_gdf = self._new_accuracy_points_gdf()
        gdf = self.accuracy_points_gdf
        if 'stratum' in gdf.columns and gdf['stratum'].notna().any():
            # one sample per map: mixing two stratifications would break the weights
            raise ValueError("A stratified sample already exists. Delete its points before drawing a new one.")
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 31))
        rng = np.random.default_rng(seed)

        classes = list(class_names_ordered)
        counts = np.bincount(classified_raster.ravel(), minlength=len(classes) + 1)
        pixels = {c: int(counts[i]) for i, c in enumerate(classes, start=1)}
        mapped = sum(pixels.values())
        if mapped == 0:
            raise ValueError("The classified map is empty.")
        alloc = {c: (max(int(min_per_class), int(round(total * pixels[c] / mapped))) if pixels[c] > 0 else 0)
                 for c in classes}

        train = None
        if exclude_segment_ids and self.segmented_image is not None:
            train = np.isin(self.segmented_image, list(exclude_segment_ids))
        transform = self.ortho_meta['transform']
        rows_all, cols_all, strata = [], [], []
        drawn = {}
        for i, c in enumerate(classes, start=1):
            if alloc[c] == 0:
                drawn[c] = 0
                continue
            eligible = classified_raster == i
            if train is not None:
                eligible &= ~train
            idx = np.flatnonzero(eligible)
            del eligible
            n = min(alloc[c], idx.size)
            if n < alloc[c]:
                print(f" -> {c}: only {idx.size} eligible pixel(s), drawing {n} of {alloc[c]}.")
            pick = idx[rng.choice(idx.size, size=n, replace=False)] if n > 0 else idx[:0]
            r, cc = np.unravel_index(pick, classified_raster.shape)
            rows_all.append(r); cols_all.append(cc); strata += [c] * n
            drawn[c] = n
        rows = np.concatenate(rows_all) if rows_all else np.zeros(0, int)
        cols = np.concatenate(cols_all) if cols_all else np.zeros(0, int)
        n_draw = len(rows)
        geoms = [Point(*(transform * (float(c) + 0.5, float(r) + 0.5))) for r, c in zip(rows, cols)]
        order = rng.permutation(n_draw)   # labelling order not grouped by map class (less bias)
        next_id = self.next_accuracy_point_id
        new_gdf = gpd.GeoDataFrame(
            {
                'point_id': list(range(next_id, next_id + n_draw)),
                'reference_class': [None] * n_draw,
                'source': [source_label] * n_draw,
                'predicted_class': [None] * n_draw,
                'stratum': [strata[k] for k in order],
            },
            geometry=[geoms[k] for k in order], crs=self.ortho_meta['crs']
        )
        self.next_accuracy_point_id = next_id + n_draw
        self.accuracy_points_gdf = gpd.GeoDataFrame(
            pd.concat([self.accuracy_points_gdf, new_gdf], ignore_index=True), crs=self.ortho_meta['crs'])
        self.accuracy_strata = {"pixels": pixels, "pixel_area": abs(transform.a * transform.e), "seed": seed}
        print(f" -> Stratified sample (seed {seed}): {n_draw} points -- "
              + ", ".join(f"{c} {n}" for c, n in drawn.items()) + ".")
        return drawn

    def _stratified_estimates(self, labeled, class_names):
        """ Area-weighted accuracy and area estimates for a stratified random sample
        (Stehman 2014, eqs. for strata that need not equal the map classes): every
        quantity is a stratified mean (overall accuracy, class area proportions) or a
        ratio of two (user's, producer's accuracy), with standard errors from the
        within-stratum variances (finite-population correction ignored: millions of
        pixels per stratum). Uses only points that carry a stratum. """
        st = self.accuracy_strata
        pts = labeled[labeled['stratum'].notna()]
        Ntot = float(sum(st["pixels"].values()))
        W = {h: n / Ntot for h, n in st["pixels"].items() if n > 0}
        groups = {h: g for h, g in pts.groupby('stratum')}
        missing = [h for h in W if h not in groups]

        def stratum_stats(fy, fx=None):
            out = []
            for h, g in groups.items():
                if h not in W:
                    continue
                y = fy(g).astype(float)
                x = fx(g).astype(float) if fx is not None else None
                out.append((W[h], len(g), y, x))
            return out

        def mean_est(fy):
            Y, V = 0.0, 0.0
            for w, n, y, _ in stratum_stats(fy):
                Y += w * y.mean()
                V += (w * w * y.var(ddof=1) / n) if n > 1 else 0.0
            return Y, np.sqrt(V)

        def ratio_est(fy, fx):
            stats = stratum_stats(fy, fx)
            Y = sum(w * y.mean() for w, n, y, x in stats)
            X = sum(w * x.mean() for w, n, y, x in stats)
            if X <= 0:
                return None, None
            R = Y / X
            V = 0.0
            for w, n, y, x in stats:
                if n > 1:
                    syy, sxx = y.var(ddof=1), x.var(ddof=1)
                    sxy = np.cov(y, x, ddof=1)[0, 1]
                    V += w * w * (syy + R * R * sxx - 2 * R * sxy) / n
            return R, np.sqrt(max(V, 0.0)) / X

        pred, ref = 'predicted_class', 'reference_class'
        oa, oa_se = mean_est(lambda g: (g[pred] == g[ref]).values)
        ua, ua_se, pa, pa_se, ap, ap_se = {}, {}, {}, {}, {}, {}
        for c in class_names:
            diag = (lambda c: lambda g: ((g[pred] == c) & (g[ref] == c)).values)(c)
            ua[c], ua_se[c] = ratio_est(diag, (lambda c: lambda g: (g[pred] == c).values)(c))
            pa[c], pa_se[c] = ratio_est(diag, (lambda c: lambda g: (g[ref] == c).values)(c))
            ap[c], ap_se[c] = mean_est((lambda c: lambda g: (g[ref] == c).values)(c))
        p = pd.DataFrame(0.0, index=list(class_names), columns=list(class_names))
        for h, g in groups.items():
            if h not in W:
                continue
            for (pi, rj), k in g.groupby([pred, ref]).size().items():
                if pi in p.index and rj in p.columns:
                    p.loc[pi, rj] += W[h] * k / len(g)
        ha = Ntot * st["pixel_area"] / 1e4
        return {
            'oa': oa, 'oa_se': oa_se, 'ua': ua, 'ua_se': ua_se, 'pa': pa, 'pa_se': pa_se,
            'area_prop': ap, 'area_prop_se': ap_se, 'p_matrix': p,
            'mapped_ha': {c: st["pixels"].get(c, 0) * st["pixel_area"] / 1e4 for c in class_names},
            'total_ha': ha, 'n_used': len(pts), 'n_without_stratum': len(labeled) - len(pts),
            'strata_without_points': missing, 'seed': st.get("seed"),
        }

    def import_accuracy_points_csv(self, csv_path, valid_class_names, source_label='field_csv'):
        """
        Reads point coordinates + a reference class label from a CSV, in the
        orthomosaic's own CRS (per the plan doc's stated expected format -- this is
        not a lat/lon-anywhere importer). Column names are matched case-insensitively
        against a small set of accepted aliases so a reasonable header (x/lon/easting,
        y/lat/northing, class/label) doesn't force a rename, but a class-name typo is
        never silently guessed at -- any CSV class name that doesn't exactly match one
        of valid_class_names raises, listing every mismatch found, so the user fixes
        the CSV rather than getting a point quietly dropped or mis-labeled.
        """
        if self.ortho_meta is None:
            raise ValueError("Load an orthomosaic before importing ground-truth points.")

        df = pd.read_csv(csv_path)
        col_lookup = {c.lower().strip(): c for c in df.columns}

        def find_col(aliases):
            for alias in aliases:
                if alias in col_lookup:
                    return col_lookup[alias]
            return None

        x_col = find_col(['x', 'lon', 'longitude', 'easting'])
        y_col = find_col(['y', 'lat', 'latitude', 'northing'])
        class_col = find_col(['class', 'reference_class', 'label', 'class_name'])

        missing = [name for name, col in (('x', x_col), ('y', y_col), ('class', class_col)) if col is None]
        if missing:
            raise ValueError(
                f"CSV is missing recognizable column(s) for: {', '.join(missing)}. Expected something "
                f"like 'x'/'y' (or lon/lat, easting/northing) and 'class'. Found columns: {list(df.columns)}"
            )

        csv_classes = set(df[class_col].astype(str).str.strip().unique())
        valid_set = set(valid_class_names)
        mismatched = sorted(csv_classes - valid_set)
        if mismatched:
            raise ValueError(
                f"CSV contains class name(s) that don't exactly match Tab 3's class list: {mismatched}. "
                f"Expected one of: {sorted(valid_set)}. Fix the CSV's class column (or rename the class "
                f"in Tab 3) and try again -- class names are matched exactly, not fuzzy-matched, so a "
                f"mismatch here isn't silently guessed at."
            )

        if self.accuracy_points_gdf is None:
            self.accuracy_points_gdf = self._new_accuracy_points_gdf()

        n = len(df)
        geoms = [Point(x, y) for x, y in zip(df[x_col].astype(float), df[y_col].astype(float))]
        next_id = self.next_accuracy_point_id
        new_gdf = gpd.GeoDataFrame(
            {
                'point_id': list(range(next_id, next_id + n)),
                'reference_class': df[class_col].astype(str).str.strip().tolist(),
                'source': [source_label] * n,
                'predicted_class': [None] * n,
            },
            geometry=geoms, crs=self.ortho_meta['crs']
        )
        self.next_accuracy_point_id = next_id + n

        self.accuracy_points_gdf = gpd.GeoDataFrame(
            pd.concat([self.accuracy_points_gdf, new_gdf], ignore_index=True), crs=self.ortho_meta['crs']
        )
        print(f" -> Imported {n} ground-truth point(s) from CSV ({csv_path}).")
        return n

    def add_manual_accuracy_point(self, x, y, reference_class=None, source_label='expert_visual'):
        """ Places a single point directly at a given real-world (x, y) -- the engine
        side of the map's right-click "Add Point" gesture in obia_app.py, independent
        of the random-sampling path. reference_class may be None (added unassigned,
        labeled later via the usual assign controls) or a class name (assigned
        immediately, e.g. picked straight from the right-click menu). Returns the new
        point's id. """
        if self.ortho_meta is None:
            raise ValueError("Load an orthomosaic before adding accuracy points.")
        if self.accuracy_points_gdf is None:
            self.accuracy_points_gdf = self._new_accuracy_points_gdf()

        point_id = self.next_accuracy_point_id
        new_gdf = gpd.GeoDataFrame(
            {
                'point_id': [point_id],
                'reference_class': [reference_class],
                'source': [source_label],
                'predicted_class': [None],
            },
            geometry=[Point(x, y)], crs=self.ortho_meta['crs']
        )
        self.next_accuracy_point_id = point_id + 1
        self.accuracy_points_gdf = gpd.GeoDataFrame(
            pd.concat([self.accuracy_points_gdf, new_gdf], ignore_index=True), crs=self.ortho_meta['crs']
        )
        return point_id

    def set_accuracy_point_class(self, point_ids, class_name):
        if self.accuracy_points_gdf is None:
            return
        mask = self.accuracy_points_gdf['point_id'].isin(set(int(p) for p in point_ids))
        self.accuracy_points_gdf.loc[mask, 'reference_class'] = class_name

    def delete_accuracy_points(self, point_ids):
        if self.accuracy_points_gdf is None:
            return
        ids = set(int(p) for p in point_ids)
        self.accuracy_points_gdf = self.accuracy_points_gdf[
            ~self.accuracy_points_gdf['point_id'].isin(ids)
        ].reset_index(drop=True)
        gdf = self.accuracy_points_gdf
        if 'stratum' not in gdf.columns or not gdf['stratum'].notna().any():
            self.accuracy_strata = None   # no stratified points left: a new sample may be drawn

    def move_accuracy_point(self, point_id, x, y):
        if self.accuracy_points_gdf is None:
            return
        idx = self.accuracy_points_gdf.index[self.accuracy_points_gdf['point_id'] == int(point_id)]
        if len(idx) == 0:
            return
        self.accuracy_points_gdf.loc[idx, 'geometry'] = Point(x, y)

    def run_accuracy_assessment(self, classified_raster, class_names_ordered, workspace_dir):
        """
        Reads predicted_class for every reference-labeled point off classified_raster
        (the same final raster Export Final Classified TIF writes -- this assesses the
        actual deliverable, not an intermediate result), builds the confusion/error
        matrix (rows = classified/map categories, columns = reference/ground-truth
        categories -- the Congalton & Green convention), and writes two files:
        accuracy_points.gpkg (full spatial data) and accuracy_report.csv (lean matrix +
        User's/Producer's/Overall Accuracy + a transparency footnote on how many points
        came from each source). Kappa is intentionally not computed -- see the plan doc.

        Only points with a non-null reference_class are used; unassigned points are
        silently excluded here (the UI-side caller is responsible for warning the user
        about that count before calling this). A point landing on a never-classified
        (raster value 0) pixel is bucketed as "Unclassified" rather than dropped, so it
        still shows up in the matrix instead of silently vanishing from the sample.

        Returns a dict with the matrix (DataFrame), per-class accuracies, overall
        accuracy, sample size used, and the two output paths.
        """
        if self.accuracy_points_gdf is None or len(self.accuracy_points_gdf) == 0:
            raise ValueError("No accuracy points exist yet -- generate or import some first.")

        labeled = self.accuracy_points_gdf[self.accuracy_points_gdf['reference_class'].notna()].copy()
        if len(labeled) == 0:
            raise ValueError("No accuracy points have a reference class assigned yet.")

        transform = self.ortho_meta['transform']
        h, w = classified_raster.shape
        class_lookup = {idx: name for idx, name in enumerate(class_names_ordered, start=1)}

        predicted = []
        for geom in labeled.geometry:
            col, row = ~transform * (geom.x, geom.y)
            col, row = int(col), int(row)
            if 0 <= row < h and 0 <= col < w:
                raster_val = int(classified_raster[row, col])
                predicted.append(class_lookup.get(raster_val, "Unclassified"))
            else:
                predicted.append("Unclassified")
        labeled['predicted_class'] = predicted
        self.accuracy_points_gdf.loc[labeled.index, 'predicted_class'] = labeled['predicted_class']

        categories = list(class_names_ordered) + (["Unclassified"] if "Unclassified" in set(predicted) else [])
        matrix = pd.crosstab(
            labeled['predicted_class'], labeled['reference_class'],
            rownames=['Classified'], colnames=['Reference']
        ).reindex(index=categories, columns=class_names_ordered, fill_value=0)

        row_totals = matrix.sum(axis=1)
        col_totals = matrix.sum(axis=0)
        grand_total = int(matrix.values.sum())

        users_accuracy, producers_accuracy = {}, {}
        for c in class_names_ordered:
            diag_val = int(matrix.loc[c, c]) if c in matrix.index else 0
            row_total = int(row_totals.get(c, 0))
            col_total = int(col_totals.get(c, 0))
            users_accuracy[c] = (diag_val / row_total) if row_total > 0 else None
            producers_accuracy[c] = (diag_val / col_total) if col_total > 0 else None

        diag_sum = sum(int(matrix.loc[c, c]) for c in class_names_ordered if c in matrix.index)
        overall_accuracy = (diag_sum / grand_total) if grand_total > 0 else 0.0

        strat = None
        if self.accuracy_strata is not None and 'stratum' in labeled and labeled['stratum'].notna().any():
            strat = self._stratified_estimates(labeled, class_names_ordered)
            # Mapped areas from the map being assessed (it may differ from the map the
            # strata were drawn from, e.g. after merging classes and reclassifying).
            cnt = np.bincount(classified_raster.ravel(), minlength=len(class_names_ordered) + 1)
            strat['mapped_ha'] = {c: cnt[i] * self.accuracy_strata["pixel_area"] / 1e4
                                  for i, c in enumerate(class_names_ordered, start=1)}

        os.makedirs(workspace_dir, exist_ok=True)
        gpkg_path = os.path.join(workspace_dir, "accuracy_points.gpkg")
        self.accuracy_points_gdf.to_file(gpkg_path, driver="GPKG")

        csv_path = os.path.join(workspace_dir, "accuracy_report.csv")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        # Totals row/column, added only to the CSV's copy of the matrix -- the returned
        # 'matrix' DataFrame stays the plain crosstab so any future programmatic caller
        # doesn't have to know to strip a synthetic "Total" row/column back out.
        matrix_with_totals = matrix.copy()
        matrix_with_totals['Total'] = row_totals.reindex(matrix.index, fill_value=0)
        matrix_with_totals.loc['Total'] = col_totals.reindex(class_names_ordered, fill_value=0)
        matrix_with_totals.loc['Total', 'Total'] = grand_total
        # The new row's insertion leaves every column upcast to float64 (NaN was
        # present transiently) -- cast back so the CSV shows "5", not "5.0".
        matrix_with_totals = matrix_with_totals.astype(int)

        with open(csv_path, 'w', newline='') as f:
            f.write("Confusion Matrix (rows = Classified/Map, columns = Reference/Ground-Truth)\n")
            matrix_with_totals.to_csv(f)
            f.write("\nClass,User's Accuracy,Producer's Accuracy\n")
            for c in class_names_ordered:
                f.write(f"{c},{fmt(users_accuracy[c])},{fmt(producers_accuracy[c])}\n")
            f.write(f"\nOverall Accuracy,{overall_accuracy:.4f}\n")
            f.write(f"Sample size (labeled points used),{len(labeled)}\n")
            source_counts = self.accuracy_points_gdf.loc[labeled.index, 'source'].value_counts().to_dict()
            footnote_parts = "; ".join(f"{count} point(s) from '{src}'" for src, count in source_counts.items())
            f.write(f"Source breakdown,\"{footnote_parts}\"\n")
            f.write(
                "Note,\"'field_csv' points are field-verified ground truth. 'expert_visual' points are "
                "high-confidence reference data from independent visual interpretation of the aerial "
                "orthomosaic, not field-verified ground truth.\"\n"
            )
            if strat is not None:
                ci = lambda se: "N/A" if se is None else f"{1.96 * se:.4f}"
                f.write("\nSTRATIFIED RANDOM SAMPLE: area-weighted estimates (Olofsson et al. 2014; Stehman 2014)\n")
                f.write(f"Strata = map classes at sampling time; seed {strat['seed']}; points used,{strat['n_used']}\n")
                f.write("Estimated area proportions (rows = map, columns = reference)\n")
                strat['p_matrix'].round(6).to_csv(f)
                f.write("\nClass,Mapped area (ha),Estimated area (ha),95% CI (ha),"
                        "User's Accuracy,95% CI,Producer's Accuracy,95% CI\n")
                for c in class_names_ordered:
                    a, ase = strat['area_prop'][c], strat['area_prop_se'][c]
                    f.write(f"{c},{strat['mapped_ha'][c]:.3f},{a * strat['total_ha']:.3f},"
                            f"{1.96 * ase * strat['total_ha']:.3f},{fmt(strat['ua'][c])},{ci(strat['ua_se'][c])},"
                            f"{fmt(strat['pa'][c])},{ci(strat['pa_se'][c])}\n")
                f.write(f"\nOverall Accuracy (area-weighted),{strat['oa']:.4f},95% CI,{1.96 * strat['oa_se']:.4f}\n")
                if strat['n_without_stratum']:
                    f.write(f"Note,{strat['n_without_stratum']} labeled point(s) without a stratum are in the "
                            f"count matrix above but not in these estimates\n")
                if strat['strata_without_points']:
                    f.write(f"Warning,\"strata with no labeled points: {', '.join(strat['strata_without_points'])}\"\n")

        return {
            'matrix': matrix,
            'users_accuracy': users_accuracy,
            'producers_accuracy': producers_accuracy,
            'overall_accuracy': overall_accuracy,
            'stratified': strat,
            'gpkg_path': gpkg_path,
            'csv_path': csv_path,
            'n_labeled': len(labeled),
            'source_counts': source_counts,
        }