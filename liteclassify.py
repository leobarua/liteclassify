"""Lite Classify launcher: shows the splash screen at once, then loads the heavy libraries
(GDAL, PyTorch, scikit-learn) and the app behind it, reporting each step.

    python liteclassify.py

This is the entry point build_exe.py packages. `python obia_app.py` still runs the app
directly, without a splash.
"""
import os
import sys
import time

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QIcon, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import QApplication, QSplashScreen

import app_info as A

SPLASH_W, SPLASH_H = 760, 390
MIN_SPLASH_S = 3.0


def resource_path(name):
    """ A bundled file: the PyInstaller bundle root when frozen, else this folder. """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, name)
    if not os.path.exists(path) and os.path.exists(os.path.join(base, "branding", name)):
        path = os.path.join(base, "branding", name)    # source checkout: branding assets
    return path


def _hex_grid(p, w, h, r, color):
    """ Faint hexagon pattern, the same motif as the icon. """
    import math
    p.setPen(QPen(color, 1.2))
    p.setBrush(Qt.NoBrush)
    dx, dy = 1.5 * r, math.sqrt(3) * r
    col = 0
    x = -r
    while x < w + r:
        y = -dy / 2 if col % 2 else 0
        while y < h + dy:
            pts = [QPointF(x + r * math.cos(math.radians(a)), y + r * math.sin(math.radians(a)))
                   for a in range(0, 360, 60)]
            p.drawPolygon(QPolygonF(pts))
            y += dy
        x += dx
        col += 1


def _font(px, weight=QFont.Bold, family="Segoe UI"):
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setWeight(weight)
    return f


def splash_pixmap(dpr=2.0, name=None):
    """ The splash, QGIS style: logo left, version right; the app name below the logo,
    running to the right; "by LeoBar" under it; the status line at the bottom. """
    pm = QPixmap(int(SPLASH_W * dpr), int(SPLASH_H * dpr))
    pm.setDevicePixelRatio(dpr)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)

    name = name or A.APP_NAME
    bg_img = QPixmap(resource_path("splash_bg.png"))
    if not bg_img.isNull():      # pixels turning into classified segments (branding/make_splash_bg.py)
        p.drawPixmap(QRectF(0, 0, SPLASH_W, SPLASH_H), bg_img, QRectF(bg_img.rect()))
    else:
        bg = QLinearGradient(0, 0, SPLASH_W, SPLASH_H)
        bg.setColorAt(0, QColor("#17201c"))
        bg.setColorAt(1, QColor("#0d1311"))
        p.fillRect(QRectF(0, 0, SPLASH_W, SPLASH_H), bg)
        _hex_grid(p, SPLASH_W, SPLASH_H, 26, QColor(255, 255, 255, 14))

    m = 44
    inset = 26          # logo and version sit further in from the edges than the text
    pull = 72           # ... and are drawn towards each other
    # logo, left
    icon = QIcon(resource_path(A.ICON_FILE)).pixmap(256, 256)
    logo = 134          # 128 + 5%
    path = QPainterPath()
    lx, ly = m + inset + pull, m + inset - 6
    path.addRoundedRect(QRectF(lx, ly, logo, logo), 27, 27)
    p.save()
    p.setClipPath(path)
    p.drawPixmap(QRectF(lx, ly, logo, logo), icon, QRectF(icon.rect()))
    p.restore()

    # version, right
    # version, right: a small "v" before large digits, right-aligned, centred on the logo
    big, small = _font(148, QFont.Black), _font(68, QFont.Black)
    fb, fs = QFontMetricsF(big), QFontMetricsF(small)
    right = SPLASH_W - m - inset - pull
    x_num = right - fb.horizontalAdvance(A.APP_VERSION)
    base = ly + logo / 2 + fb.capHeight() / 2
    p.setPen(QColor("#ffffff"))
    p.setFont(big)
    p.drawText(QPointF(x_num, base), A.APP_VERSION)
    p.setFont(small)
    p.drawText(QPointF(x_num - fs.horizontalAdvance("v") - 6, base), "v")

    # app name below the logo, running right
    y_name = ly + logo + 30
    name_font = _font(58, QFont.Black)
    fm = QFontMetricsF(name_font)
    grad = QLinearGradient(m, 0, m + fm.horizontalAdvance(name), 0)
    grad.setColorAt(0, QColor(A.BRAND_ORANGE_LIGHT))
    grad.setColorAt(1, QColor(A.BRAND_ORANGE))
    text = QPainterPath()
    text.addText(m, y_name + fm.ascent(), name_font, name)
    p.fillPath(text, grad)
    x_tag = m + fm.horizontalAdvance(name) + 18
    tag = A.APP_TAGLINE
    size = 23
    while size > 12 and QFontMetricsF(_font(size, QFont.DemiBold)).horizontalAdvance(tag) > SPLASH_W - m - x_tag - 2:
        size -= 1   # shrink to fit the space right of the name
    tag_font = _font(size, QFont.DemiBold)
    p.setFont(tag_font)
    p.setPen(QColor("#e8ece9"))
    p.drawText(QRectF(x_tag, y_name, SPLASH_W - m - x_tag, fm.height() - 8), Qt.AlignLeft | Qt.AlignBottom, tag)
    # rule under the name, to the right edge
    p.setPen(QPen(QColor(A.BRAND_ORANGE), 2))
    p.drawLine(QPointF(m, y_name + fm.height() + 8), QPointF(SPLASH_W - m, y_name + fm.height() + 8))

    # by LeoBar
    p.setFont(_font(19, QFont.DemiBold))
    p.setPen(QColor("#b9c3bd"))
    p.drawText(QRectF(m, y_name + fm.height() + 16, SPLASH_W - 2 * m, 30), Qt.AlignLeft | Qt.AlignVCenter,
               f"by {A.APP_AUTHOR}")
    p.end()
    return pm


class Splash(QSplashScreen):
    def __init__(self):
        super().__init__(splash_pixmap(), Qt.WindowStaysOnTopHint)
        self.setFont(_font(12, QFont.Normal))
        self.shown_at = time.monotonic()

    def status(self, text):
        self.showMessage(text, Qt.AlignLeft | Qt.AlignBottom, QColor("#9aa59f"))
        QApplication.processEvents()


def selftest(ortho, dsm=None, report=None):
    """ Headless check that a built bundle has everything the pipeline needs (GDAL data,
    CPU SLIC, features, DINOv2, Random Forest): `LiteClassify.exe --selftest ortho.tif [dsm.tif]`.
    Writes a JSON report (default %TEMP%/liteclassify_selftest.json) and exits 0 on success. """
    import json, tempfile, traceback
    out = {"ok": False, "steps": {}}
    report = report or os.path.join(tempfile.gettempdir(), "liteclassify_selftest.json")

    def step(name, fn):
        t = time.time()
        r = fn()
        out["steps"][name] = round(time.time() - t, 2)
        return r
    try:
        from obia_engine import ObiaEngine, cuda_available
        import obia_discover as od
        from sklearn.ensemble import RandomForestClassifier
        out["cuda"] = bool(cuda_available())
        e = ObiaEngine()
        step("load", lambda: e.load_raster(ortho))
        step("resample_50cm", lambda: e.resample_ortho(target_pixel_size=0.5))
        if dsm:
            step("dsm", lambda: e.load_dsm(dsm))
        step("segment_cpu", lambda: e.segment_image(400, 0.15, 2.5, force_device="cpu"))
        step("vectorize", e.vectorize_segments)
        step("features", e.compute_all_features_parallel)
        step("adjacency", e.build_adjacency_graph)
        step("neighbours", e.compute_neighbor_relative_features)
        if dsm:
            step("height_above_ground", e.compute_height_features)
        deep = step("dinov2", lambda: od.compute_deep_features(e.original_ortho_data, e.segmented_image,
                                                               e.valid_mask, e.effective_pixel_size))
        cols = e.active_feature_columns()
        X = e.features_df[cols].fillna(0).values
        y = (e.features_df["exg_mean"] > e.features_df["exg_mean"].median()).values
        rf = step("random_forest", lambda: RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y))
        out.update(ok=True, segments=int(len(e.features_df)), features=len(cols),
                   deep_dims=int(next(iter(deep["feats"].values())).shape[1]), rf_train_acc=float(rf.score(X, y)))
    except Exception:
        out["error"] = traceback.format_exc()
    with open(report, "w") as f:
        json.dump(out, f, indent=1)
    return 0 if out["ok"] else 1


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        args = sys.argv[i + 1:]
        sys.exit(selftest(args[0], args[1] if len(args) > 1 else None))
    if sys.platform == "win32":
        try:   # own taskbar button and icon, not python.exe's, when run from source
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(A.APP_ID)
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(A.APP_NAME)
    app.setApplicationVersion(A.APP_VERSION)
    app.setOrganizationName(A.APP_AUTHOR)
    icon_path = resource_path(A.ICON_FILE)
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash = Splash()
    splash.show()
    splash.status("    Starting…")

    # The heavy imports, one at a time so the splash can say what it is doing.
    splash.status("    Loading geospatial libraries (GDAL, rasterio, GeoPandas)…")
    import rasterio, geopandas  # noqa: F401
    splash.status("    Loading PyTorch…")
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    splash.status("    Loading scikit-learn and scikit-image…")
    import sklearn.ensemble, skimage.segmentation  # noqa: F401
    splash.status("    Building the interface…")
    import obia_app
    window = obia_app.ObiaApp()

    left = MIN_SPLASH_S - (time.monotonic() - splash.shown_at)
    splash.status("    Ready.")
    while left > 0:
        QApplication.processEvents()
        time.sleep(0.03)
        left = MIN_SPLASH_S - (time.monotonic() - splash.shown_at)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
