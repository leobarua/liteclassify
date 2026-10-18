"""Builds the labelling layers for the forest sample:
  stands_400m2.gpkg   - coarse SLIC segments (~400 m2) with empty population/confidence fields, to label in QGIS
  segments_40m2.gpkg  - the 40 m2 segments used by the tests (reference only)
One feature per segment (pieces touching only diagonally are dissolved into a MultiPolygon)."""
import os, sys, pickle
import numpy as np, geopandas as gpd
from rasterio.features import shapes
sys.path.insert(0, os.getcwd())
from obia_engine import ObiaEngine

D = "data/CNN_Sample"
OUT = os.path.join(os.path.dirname(__file__), "out")

def to_gdf(lab, transform, crs):
    g = gpd.GeoDataFrame.from_features(
        [{"properties": {"seg_id": int(v)}, "geometry": s}
         for s, v in shapes(lab, mask=lab > 0, transform=transform)], crs=crs)
    return g.dissolve("seg_id").reset_index()

e = ObiaEngine()
e.load_raster(f"{D}/CNNSample_Ortho_0.2.tif")
valid_m2 = e.valid_mask.sum() * e.effective_pixel_size ** 2
e.segment_image(int(valid_m2 / 400), 0.3, 2.5, force_device="gpu")
crs, tr = e.ortho_meta["crs"], e.ortho_meta["transform"]

stands = to_gdf(e.segmented_image, tr, crs)
stands["area_m2"] = stands.area.round(1)
stands["population"] = None
stands["confidence"] = None
stands = stands.astype({"population": "object", "confidence": "Int64"})
stands.to_file(f"{D}/stands_400m2.gpkg", layer="stands", driver="GPKG")

d = pickle.load(open(os.path.join(OUT, "forest.pkl"), "rb"))
fine = to_gdf(d["seg"], d["transform"], crs)
fine["area_m2"] = fine.area.round(1)
fine.to_file(f"{D}/segments_40m2.gpkg", layer="segments", driver="GPKG")
print(f"stands: {len(stands)}, median {stands.area_m2.median():.0f} m2 | fine: {len(fine)}, median {fine.area_m2.median():.0f} m2")
